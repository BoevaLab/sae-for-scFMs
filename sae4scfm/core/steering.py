import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import scvi
import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation

from sae4scfm.core.utils import set_seed
from sae4scfm.core.io_utils import load_adata


# ============================================================================
# Feature Selection Strategies
# ============================================================================

class RandomFeatureSelector():
    """Randomly select features"""
    
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.name = 'random'
    
    def select_features(self, n_features: int, dict_size: int, trial: int = 0) -> np.ndarray:
        """Randomly sample features without replacement"""
        trial_rng = np.random.RandomState(self.seed + trial)
        return trial_rng.choice(dict_size, size=n_features, replace=False)


class FileFeatureSelector():
    """Select top-N features from a ranked file"""
    
    def __init__(
        self, 
        feature_file: str
    ):
        self.feature_file = Path(feature_file)
        self.name = 'AMI'
        
        # Load and rank features once
        self.ranked_features = self._load_features()
    
    def _load_features(self) -> np.ndarray:
        """Load features from file and return ranked indices"""
        if not self.feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_file}")
        
        df = pd.read_csv(self.feature_file, header=[0, 1], index_col=0)
        df = df['label_scoring']
        
        # Filter columns containing both "batch" and "AMI"
        filtered_cols = [col for col in df.columns if 'batch' in col.lower() and 'ami' in col.lower()]
        df = df[filtered_cols]
        
        # Sort df
        df = df.max(axis=1).sort_values(ascending=False)
        
        return df.index.values
    
    def select_features(self, n_features: int, dict_size: int, trial: int = 0) -> np.ndarray:
        """Select top-N features from ranked list"""
        if n_features > len(self.ranked_features):
            raise ValueError(
                f"Requested {n_features} features but only {len(self.ranked_features)} available"
            )
        return self.ranked_features[:n_features]


def steer(
        adapter,
        sae,
        data_loader,
        features_to_steer,
        clamp_value: float,
        device
):
    """Steer specified features in the activations to a clamp value
    
    Args:
        adapter: Model adapter 
        sae: Trained sparse autoencoder
        data_loader: DataLoader providing batches
        features_to_steer: Array/list of feature indices to clamp
        clamp_value: Value to clamp active features to
        device: Device to run on
        
    Returns:
        original_embeddings: Embeddings without steering 
        steered_embeddings: Embeddings with steering applied
    """
    
    original_embeddings = []
    steered_embeddings = []

    with torch.no_grad():
        for batch in data_loader:
            mask = adapter.generate_activation_mask(batch).to(device)
            original_embedding, x = adapter.generate_embeddings(batch, device_model=device, return_activations=True)

            # Apply steering: clamp specified features
            feat = sae.encode(x[mask], use_threshold=False)
            sub = feat[:, features_to_steer]
            sub[feat[:, features_to_steer] > 0] = clamp_value
            feat[:, features_to_steer] = sub

            # Decode back and generate steered embedding
            x_hat = sae.decode(feat)
            x[mask] = x_hat 

            steered_embedding = adapter.generate_embeddings(batch, device_model=device, modify_activations=True, x_hat=x)
            
            original_embeddings.append(original_embedding)
            steered_embeddings.append(steered_embedding)

    original_embeddings = torch.vstack(original_embeddings).cpu().numpy()
    steered_embeddings = torch.vstack(steered_embeddings).cpu().numpy()

    return original_embeddings, steered_embeddings


def compute_batch_integration(adata, embeddings):

    emb_keys = list(adata.obsm)

    # Preprocessing
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="cell_ranger", batch_key="batch")
    sc.tl.pca(adata, n_comps=30, use_highly_variable=True)
    adata = adata[:, adata.var.highly_variable].copy()
    adata.obsm["Unintegrated"] = adata.obsm["X_pca"]

    # scVI
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key='batch')
    vae = scvi.model.SCVI(adata, gene_likelihood="nb", n_layers=2, n_latent=30)
    vae.train()
    adata.obsm["scVI"] = vae.get_latent_representation()


    emb_keys += ["Unintegrated", "scVI"]

    # Benchmark
    set_seed(0)
    bm = Benchmarker(
        adata,
        batch_key= 'batch',
        label_key= "celltype",
        bio_conservation_metrics = BioConservation(nmi_ari_cluster_labels_leiden=True, nmi_ari_cluster_labels_kmeans=False),
        embedding_obsm_keys=emb_keys,
        n_jobs=3,
    )
    bm.benchmark()

    return bm