import numpy as np
import torch
from sklearn.metrics import adjusted_mutual_info_score
import pandas as pd
import json
import gseapy as gp

# GSEA
f = open('data/gene_sets/c5.json')
hallmark = json.load(f)
f.close()
gene_sets = {group:hallmark[group]['geneSymbols'] for group in hallmark if group.startswith('GO')}


def preprocess_labels(labels, cfg) -> np.ndarray:
    """
    Preprocesses labels from AnnData object based on configuration.

    Args:
        labels (Dict[str, pd.Categorical]):
    """
    batches = labels["batch"].astype("category")
    celltypes = labels["celltype"].astype("category")
    
    batches_dummies = pd.get_dummies(batches)
    celltypes_dummies = pd.get_dummies(celltypes)
    
    labels['batch'] = {
        label: batches_dummies[label].to_numpy()
        for label in batches.cat.categories
    }
    labels['celltype'] = {
        label: celltypes_dummies[label].to_numpy()
        for label in celltypes.cat.categories
    }
    if cfg.data.name == 'covid' or cfg.data.name == 'covid_evaluation':
        additional_cells = ['T']
        for cell_ in additional_cells:
            labels['celltype'][cell_] = np.array([cell.startswith(cell_) for cell in celltypes])
        keep_cells = ['Monocyte', 'NK', 'T', 'B', 'Plasma', 'HPC', 'Platelets']
        labels['celltype'] = {
            k: v for k, v in labels['celltype'].items()
            if k in keep_cells
        }

    elif cfg.data.name == 'pancreas':
        additional_batches = ['inDrop']
        for batch in additional_batches:
            labels['batch'][batch] = np.array([cell.startswith(batch) for cell in batches])
    elif cfg.data.name == 'lung':
        additional_batches = ['A', 'B']
        for batch in additional_batches:
            labels['batch'][batch] = np.array([cell.startswith(batch) for cell in batches])
        # Add labels for cells starting with numbers
        labels['batch']['starts_with_number'] = np.array([
            cell[0].isdigit() if len(cell) > 0 else False 
            for cell in batches
        ])
        # Add labels for cells starting with letters
        labels['batch']['starts_with_letter'] = np.array([
            cell[0].isalpha() if len(cell) > 0 else False 
            for cell in batches
        ])
    elif cfg.data.name == 'immune': 
        additional_batches = ['Oetjen', 'Sun_sample']
        for batch in additional_batches:
            labels['batch'][batch] = np.array([cell.startswith(batch) for cell in batches])
    else:
        raise ValueError(f"Dataset {cfg.data.name} not recognized.")

    return labels

def normalize_features(features):
    """Normalize features by dividing by their maximum activation across all tokens."""
    features_max = features.view(-1, features.shape[2]).max(dim=0)[0]
    mask = features_max > 0
    features[:, :, mask] = features[:, :, mask] / features_max[mask]
    return features

def label_scoring(feature, labels, cfg):
    """
    Calculate label scoring metrics for a feature
    
    Args:
        feature: Feature activations for all cells
        labels: Dict of label arrays
        cfg: Configuration
        
    Returns:
        Dict with structure {label_name: {metric_name: score}}
    """
    label_scores = {}
    for label in cfg.analysis.label_scoring.label:
        label_scores[label] = {}
        if "AMI" in cfg.analysis.label_scoring.score:
            for sublabel in labels[label]:
                label_scores[label][sublabel] = {}
                AMI = calculate_AMI(feature, labels[label][sublabel])
                label_scores[label][sublabel]["AMI"] = AMI
        if "F1" in cfg.analysis.label_scoring.score:
            for sublabel in labels[label]:
                label_scores[label][sublabel] = {}
                F1 = calculate_F1(feature, labels[label][sublabel])
                label_scores[label][sublabel]["F1"] = F1
    
    return label_scores

def gene_scoring(feature, values, genes, adapter, cfg, background_genes, gene_families_dict):
    """
    Calculate gene scoring metrics for a feature
    
    Args:
        feature: Feature activations
        values: Gene expression values
        genes: Gene identifiers
        adapter: Model adapter with gene mapping
        cfg: Configuration
        background_genes: Background gene list for GSEA
        gene_families_dict: Pre-calculated dict of gene family lists
        
    Returns:
        Dict with gene scoring results
    """
    results = {}
    mask_active = feature > cfg.analysis.gene_scoring.threshold
    genes_active = genes[mask_active]
    genes_active_name = adapter.id2gene(genes_active)
    for family in cfg.analysis.gene_scoring.gene_families:
        results[family] = pd.Series(genes_active_name).isin(gene_families_dict[family]).sum()

    gene_list = pd.Series(genes_active_name).unique().tolist()
    if len(gene_list) >= 10:
        enriched = gp.enrichr(gene_list=gene_list,
                        gene_sets=gene_sets,
                        organism='human',
                        outdir=None, # don't write to disk
                        background=background_genes)
        res = enriched.results.sort_values('Adjusted P-value').head(5)
        results['GO enrichment'] = '; '.join(res['Term'])
        results['Overlap'] = '; '.join(res['Overlap'])
        results['Adjusted P-value'] = '; '.join([str(num) for num in res['Adjusted P-value']])
        results['GSEA_top5'] = res.to_dict(orient='records')
    return results

def expression_scoring(feature, values, cfg):
    """
    Calculate expression statistics for active cells
    
    Args:
        feature: Feature activations
        values: Gene expression values
        cfg: Configuration
        
    Returns:
        Dict with expression statistics
    """
    mask_active = feature > cfg.analysis.expression_scoring.threshold
    results = {}
    
    if "mean" in cfg.analysis.expression_scoring.score:
        results["mean"] = values[mask_active].mean()
    if "std" in cfg.analysis.expression_scoring.score:
        results["std"] = values[mask_active].std()
    
    return results

def density_scoring(features, genes, cfg):
    """
    Calculate density scoring metrics for all features
    
    Args:
        features: All feature activations
        genes: Gene identifiers
        cfg: Configuration
        
    Returns:
        Feature density
    """
    mask_padding = genes == 65535  # [batch, seq_len]
    mask_valid = ~mask_padding  # [batch, seq_len]
    
    # Count how many valid (non-padding) tokens each feature is active on
    is_active = features > cfg.analysis.density_scoring.threshold  # [batch, seq_len, num_features]
    active_counts = (is_active & mask_valid.unsqueeze(-1)).sum(dim=(0, 1))  # [num_features]
    
    # Total number of valid tokens
    total_valid = mask_valid.sum()
    
    # Density for each feature
    density = active_counts / total_valid
    return density

def calculate_AMI(feature, label):
    thresholds = np.arange(0.0, 1.0, 0.2)

    ami = [
        adjusted_mutual_info_score(
            label,
            (feature > t).any(axis=1)
        )
        for t in thresholds
    ]

    return float(np.max(ami))

def calculate_F1(feature, label, threshold_tokens=1):
    P = feature[label]
    N = feature[~label]

    thresholds = np.arange(0.1, 1.0, 0.2)

    tp = np.array([
        ((P >= t).sum(axis=1) >= threshold_tokens).sum()
        for t in thresholds
    ])

    fp = np.array([
        ((N >= t).sum(axis=1) >= threshold_tokens).sum()
        for t in thresholds
    ])

    recall = tp / P.shape[0]
    precision = tp / (tp + fp)

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * precision * recall / (precision + recall)

    return np.nanmax(f1).item()

def gene_families(all_genes):
    """
    Pre-calculate gene family lists for efficient reuse in gene_scoring.
    
    Args:
        all_genes: List of all gene names in the dataset
        
    Returns:
        Dict mapping family names to lists of genes in that family
    """
    families = {}
    families['RP'] = [gene for gene in all_genes if (gene.startswith('RPL') or gene.startswith('RPS') or gene in ['UBA52', 'FAU'])]
    families['MIT'] = [gene for gene in all_genes if gene.startswith('MT-')]
    families['MHC'] = [gene for gene in all_genes if gene.startswith('HLA-')]
    families['TRAV'] = [gene for gene in all_genes if gene.startswith('TR')]
    families['H'] = [gene for gene in all_genes if (gene.startswith('H1') or gene.startswith('H2') or gene.startswith('H3')
                or gene.startswith('H4'))]
    families['MET'] = [gene for gene in all_genes if (gene.startswith('MT1') or gene.startswith('MT2') or gene.startswith('MT3') or gene.startswith('MT4'))]
    families['IG'] = [gene for gene in all_genes if gene.startswith('IG')]
    
    return families