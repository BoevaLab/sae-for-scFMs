"""
Docstring for scripts.feature_analysis
"""
import hydra
from hydra.utils import get_class
from omegaconf import DictConfig

import os
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.utils import set_seed, get_devices
from sae4scfm.core.io_utils import load_sae_checkpoint, load_adata, load_features, save_analysis_results, load_analysis_results
from sae4scfm.core.analysis import preprocess_labels, normalize_features, label_scoring, gene_scoring, expression_scoring, density_scoring, gene_families

@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="analyze", version_base='1.3')
def main(cfg: DictConfig) -> None:

    # Load SAE checkpoint
    device = get_devices()[0]
    checkpoint_path = REPO_ROOT / "experiments/train" / cfg.sae_checkpoint.experiment / cfg.sae_checkpoint.timestamp
    _, original_cfg = load_sae_checkpoint(checkpoint_path, device, load_sae=False)
    set_seed(original_cfg.seed)

    # Overwrite dataset
    data = cfg.get("data", None)
    if data is not None:
        data.preprocess.batch_size = original_cfg.data.preprocess.batch_size
        data.preprocess.subset_hvg = original_cfg.data.preprocess.subset_hvg
        original_cfg.data = data
        print(f'Using {original_cfg.data.name} dataset')

    # Instantiate scfm model adapter
    adapter_class = get_class(original_cfg.scfm._target_)
    adapter = adapter_class(original_cfg)

    # Load adata with test split
    test_fraction = original_cfg.data.preprocess.split if hasattr(original_cfg.data, 'preprocess') else 1.0
    adata = load_adata(original_cfg, sub_sample=test_fraction)

    # Preprocess labels
    labels = {}
    labels['batch'] = adata.obs["batch"].astype("category")
    labels['celltype'] = adata.obs["celltype"].astype("category")
    labels = preprocess_labels(labels, original_cfg)

    # Load features
    features_path = REPO_ROOT / "cached_features" / cfg.sae_checkpoint.experiment / ("features-" + cfg.sae_checkpoint.timestamp + ".h5")
    loaded_data = load_features(features_path, original_cfg)
    features = loaded_data['features']
    #values = loaded_data['values'] # TODO: temporary solution, fix for Geneformer
    genes = loaded_data['genes']
    features = normalize_features(features)
    
    n_features = features.shape[2]
    
    # Try to load existing results
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_dir = Path(hydra_cfg['runtime']['output_dir'])
    existing_df = load_analysis_results(save_dir, cfg.sae_checkpoint.timestamp)
    
    # Initialize results storage with multilevel columns
    results = {}

    # Prepare gene background for GSEA and pre-calculate gene families
    #gene_background = adapter.id2gene(list(pd.Series(genes[genes != 65535]).unique()))
    #gene_families_dict = gene_families(gene_background)
    
    # Run density scoring (operates on all features at once)
    if cfg.analysis.run_density_scoring:
        density_results = density_scoring(features, genes, cfg)
        if density_results:
            for metric, values in density_results.items():
                results[('density', metric)] = values
    
    # Iterate through features for per-feature analysis
    for feature_idx in tqdm(range(n_features)):
        feature = features[:, :, feature_idx]
        
        if cfg.analysis.run_label_scoring:
            label_results = label_scoring(feature, labels, cfg)
            for label_name, sublabels in label_results.items():
                for sublabel_name, metrics in sublabels.items():
                    for metric_name, score in metrics.items():
                        col_key = ('label_scoring', f'{label_name}_{sublabel_name}_{metric_name}')
                        if col_key not in results:
                            results[col_key] = [None] * n_features
                        results[col_key][feature_idx] = score
        
        if cfg.analysis.run_gene_scoring:
            gene_results = gene_scoring(feature, values, genes, adapter, cfg, gene_background, gene_families_dict)
            for metric_name, score in gene_results.items():
                col_key = ('gene_scoring', metric_name)
                if col_key not in results:
                    results[col_key] = [None] * n_features
                results[col_key][feature_idx] = score
        
        if cfg.analysis.run_expression_scoring:
            expr_results = expression_scoring(feature, values, cfg)
            for metric_name, score in expr_results.items():
                col_key = ('expression_scoring', metric_name)
                if col_key not in results:
                    results[col_key] = [None] * n_features
                results[col_key][feature_idx] = score
    
    # Create new dataframe with multilevel columns
    new_df = pd.DataFrame(results)
    new_df.columns = pd.MultiIndex.from_tuples(new_df.columns, names=['analysis_type', 'metric'])
    new_df.index.name = 'feature_id'
    
    # Merge with existing results if present
    if existing_df is not None:
        # Get analysis types from new dataframe
        new_analysis_types = new_df.columns.get_level_values(0).unique()
        
        # Drop existing columns for these analysis types (overwrite)
        mask = ~existing_df.columns.get_level_values(0).isin(new_analysis_types)
        existing_df = existing_df.loc[:, mask]
        if not existing_df.empty:
            # Merge old and new
            final_df = pd.concat([existing_df, new_df], axis=1)
        else:
            # All columns are being replaced
            final_df = new_df
    else:
        final_df = new_df
    
    # Save results
    output_dir = save_analysis_results(final_df, save_dir, cfg.sae_checkpoint.timestamp)
    print(f"Analysis results saved to {output_dir / f'results-{cfg.sae_checkpoint.timestamp}.csv'}")

if __name__ == "__main__":
    main()