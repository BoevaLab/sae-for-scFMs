from pathlib import Path
import scvi
from scanpy.pp import subsample
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_class
import json
import math
import numpy as np
from tqdm import tqdm
import h5py
import hdf5plugin
import torch
import pandas as pd

from sae4scfm.core.utils import set_seed

def load_adata(cfg: DictConfig, sub_sample: float = 1.0):
    """
    Loads an AnnData object based on the dataset name specified in the configuration.

    Args:
        cfg: Configuration object with data loading parameters.
        sub_sample (float): Fraction of data to subsample
    """
    data_path = Path(cfg.data.file_path) / cfg.data.file_name

    if cfg.data.name == 'pbmc':
        adata = scvi.data.pbmc_dataset('data/PBMC_10K/')
        adata.obs["celltype"] = adata.obs["str_labels"].astype("category")
        adata.var = adata.var.set_index("gene_symbols")
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs["batch"] = np.zeros(adata.obs.shape[0])
        adata.obs["batch"] = adata.obs["batch"].astype("category")

    elif cfg.data.name == 'covid':
        adata = scvi.data.read_h5ad(data_path)
        adata.var["feature_name"] = adata.var["feature_name"].astype('str')
        adata.var = adata.var.set_index("feature_name")
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs["batch"] = adata.obs["donor_id"].astype(str)
        adata.obs['celltype'] = adata.obs['annotation_broad']
        #adata.var['ensembl_id'] = adata.var.index

    elif cfg.data.name == 'covid_evaluation':
        adata = scvi.data.read_h5ad(data_path)
        adata.var["feature_name"] = adata.var["feature_name"].astype('str')
        adata.var = adata.var.set_index("feature_name")
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs["batch"] = adata.obs["donor_id"].astype(str)
        adata.obs['celltype'] = adata.obs['annotation_broad']
        #adata.var['ensembl_id'] = adata.var.index

    elif cfg.data.name == 'pancreas':
        adata = scvi.data.read_h5ad(data_path)
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs["batch"] = adata.obs['tech'].astype(str)

    elif cfg.data.name == 'lung':
        adata = scvi.data.read_h5ad(data_path)
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['celltype'] = adata.obs['cell_type']
        adata.obs["batch"] = adata.obs["batch"].astype(str)

    elif cfg.data.name == 'immune':
        adata = scvi.data.read_h5ad(data_path)
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['celltype'] = adata.obs['final_annotation'].astype("category").cat.codes.values
        adata.obs["batch"] = adata.obs["batch"].astype(str)

    elif sum(cfg.data.n_partitions) > 1:
        adata = scvi.data.read_h5ad(data_path)
        adata.var["feature_name"] = adata.var["feature_name"].astype('str')
        adata.var = adata.var.set_index("feature_name")
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs["batch"] = adata.obs["dataset_id"].astype(str)
    else:
        raise ValueError(f"Dataset {cfg.data.name} not recognized.")

    # Subsample data
    if sub_sample < 1:
        subsample(adata, sub_sample, random_state = 42)

    return adata

def save_sae_checkpoint(sae, save_dir: Path) -> None:
    """Save SAE model checkpoint
    
    Args:
        sae: Trained SAE model
        save_dir: Directory to save the model
    """
    torch.save(sae.state_dict(), save_dir / "model.pth")

def load_sae_checkpoint(checkpoint_path: Path, device: str, load_sae=True):
    """Load trained SAE from checkpoint directory
    
    Args:
        checkpoint_path: Path to checkpoint (e.g., experiments/train/exp00/Jan06-14-33)
        device: Device to load model on
        
    Returns:
        sae: Loaded SAE model
        original_cfg: Original training configuration
    """
    # Load original training config
    config_path = checkpoint_path / ".hydra" / "config.yaml"
    model_path = checkpoint_path / "model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    original_cfg = OmegaConf.load(config_path)

    if load_sae:
        sae = get_class(original_cfg.sae.autoencoder._target_).from_pretrained(model_path, device = device)
        sae.eval()
    else:
        sae = None
    
    
    return sae, original_cfg

def save_metrics(metrics_dict, save_dir: Path) -> None:
    """Save evaluation metrics as JSON
    
    Args:
        metrics_dict: Dictionary of evaluation metrics
        save_dir: Directory to save the metrics
    """
    metrics_path = save_dir / "eval_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    return

def generate_and_compress_features(
        adapter, 
        sae, 
        data_loader, 
        device, 
        cfg,
        save_dir: Path,
        date) -> None:
    """Generate and compress features"""
    
    set_seed(cfg.seed)

    # Tensor size and chunking
    n_samples = int(cfg.data.n_cells[0] * cfg.data.preprocess.split)
    n_genes = cfg.scfm.preprocessing.max_len
    dict_size = cfg.sae.dict_size
    chunk_size = 128

    # Get compression config from model adapter config
    compression_datasets = cfg.scfm.compression.datasets
    
    # Create HDF5 file
    with h5py.File(save_dir / ('features-' + date + '.h5'), 'w') as f:
        # Create datasets dynamically based on adapter config
        datasets = {}
        for ds_cfg in compression_datasets:
            name = ds_cfg.name
            dtype_str = ds_cfg.dtype
            
            # Map string dtype to numpy dtype
            dtype = getattr(np, dtype_str)
            
            if name == 'features':
                shape = (n_samples, n_genes, dict_size)
                chunks = (chunk_size, n_genes, 1)
            else:  # values, genes, or other 2D datasets
                shape = (n_samples, n_genes)
                chunks = (chunk_size, n_genes)
            
            datasets[name] = f.create_dataset(
                name, 
                shape=shape, 
                chunks=chunks,
                dtype=dtype, 
                **hdf5plugin.Blosc(cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE)
            )

        write_pos = 0
        
        for data_dict in tqdm(data_loader):
            
            activations = adapter.generate_activations(data_dict, device)
            
            # Reshape activations from [batch_size, seq_len, activation_dim] to [batch_size * seq_len, activation_dim]
            # so that SAE treats each token independently
            batch_size, seq_len, activation_dim = activations.shape
            activations_flat = activations.reshape(batch_size * seq_len, activation_dim)
            
            with torch.no_grad():
                features_flat = sae.encode(activations_flat, use_threshold=False)
            
            # Reshape features back to [batch_size, seq_len, dict_size]
            features = features_flat.reshape(batch_size, seq_len, -1)

            # Adapter returns variable number of arrays based on model
            compressed_data = adapter.prepare_compression(data_dict, features, n_genes, device)

            batch_size = compressed_data[0].shape[0]

            # Write each dataset returned by adapter
            for i, (ds_name, ds) in enumerate(datasets.items()):
                    ds[write_pos:write_pos + batch_size] = compressed_data[i]

            write_pos += batch_size
                      
    
    return

def generate_and_compress_features_blosc(
        adapter, 
        sae, 
        data_loader, 
        device, 
        cfg,
        save_dir: Path,
        date) -> None:
    """Generate and compress features"""
    import blosc2
    set_seed(cfg.seed)
    blosc2.nthreads = 4

    # Tensor size and chunking
    n_samples = int(cfg.data.n_cells[0] * cfg.data.preprocess.split)
    n_genes = cfg.scfm.preprocessing.max_len
    dict_size = cfg.sae.dict_size
    chunk_size = 19200
    compression_datasets = cfg.scfm.compression.datasets
    

    # Compression / decompression params
    cparams_f16 = {'codec':blosc2.Codec.ZSTD, 'clevel':5, 'typesize':np.dtype(np.float16).itemsize}
    cparams_u8  = {'codec':blosc2.Codec.ZSTD, 'clevel':5, 'typesize':np.dtype(np.uint8).itemsize}
    cparams_u16 = {'codec':blosc2.Codec.ZSTD, 'clevel':5, 'typesize':np.dtype(np.uint16).itemsize}

    datasets = {}
    for ds_cfg in compression_datasets:
        name = ds_cfg.name
        dtype_str = ds_cfg.dtype
        
        # Map string dtype to numpy dtype
        dtype = getattr(np, dtype_str)
        
        if name == 'features':
            shape = (n_samples, n_genes, dict_size)
            chunks = (chunk_size, n_genes, 1)
        else:  # values, genes, or other 2D datasets
            shape = (n_samples, n_genes)
            chunks = (chunk_size, n_genes)

        save_path = save_dir / f'{name}-{date}.b2nd'

        if dtype == np.float16:
            params = cparams_f16
        elif dtype == np.uint8:
            params = cparams_u8
        elif dtype == np.uint16:
            params = cparams_u16
        else:
            raise ValueError(f"Type {dtype} not recognized.")

        datasets[name] =  blosc2.zeros(
            shape= shape,
            dtype= dtype,
            chunks = chunks,
            urlpath = str(save_path),
            mode = "w",
            cparams = params
        )
    
    buffer = []

    buffer_pos = 0
    write_pos = 0

    for data_dict in tqdm(data_loader):
            
        activations = adapter.generate_activations(data_dict, device)
        
        # Reshape activations from [batch_size, seq_len, activation_dim] to [batch_size * seq_len, activation_dim]
        # so that SAE treats each token independently
        batch_size, seq_len, activation_dim = activations.shape
        activations_flat = activations.reshape(batch_size * seq_len, activation_dim)
        
        with torch.no_grad():
            features_flat = sae.encode(activations_flat, use_threshold=False)
        
        # Reshape features back to [batch_size, seq_len, dict_size]
        features = features_flat.reshape(batch_size, seq_len, -1)

        # Adapter returns variable number of arrays based on model
        compressed_data = adapter.prepare_compression(data_dict, features, n_genes, device)

        batch_size = compressed_data[0].shape[0]
        buffer.append(compressed_data)
        buffer_pos += batch_size
        # TODO: finish

def generate_pool_and_save_features(
        adapter, 
        sae, 
        data_loader, 
        device, 
        cfg,
        save_dir: Path,
        date) -> None:
    """Generate and compress features"""
    
    set_seed(cfg.seed)

    # Tensor size and chunking
    n_samples = int(cfg.data.n_cells[0] * cfg.data.preprocess.split)
    dict_size = cfg.sae.dict_size
    features = torch.zeros(n_samples, dict_size, dtype=torch.float16, device='cpu')

    samples_filled = 0
    for data_dict in tqdm(data_loader):
        activations = adapter.generate_activations(data_dict, device)
        batch_size, seq_len, activation_dim = activations.shape
        activations_flat = activations.reshape(batch_size * seq_len, activation_dim)

        with torch.no_grad():
            features_flat = sae.encode(activations_flat, use_threshold=False)
        
        # Reshape features back to [batch_size, seq_len, dict_size]
        feats = features_flat.reshape(batch_size, seq_len, -1)

        mask = adapter.generate_activation_mask(data_dict).to(device)
        feats[~mask] = 0
        feats = feats.max(axis=1).values # maxpool genes
        feats = feats.cpu().to(torch.float16)
        
        features[samples_filled:samples_filled + batch_size] = feats
        samples_filled += batch_size

    save_path = save_dir / f'features-{date}.pt'
    torch.save(features, save_path)
                      
    return

def load_features(features_path: Path, cfg: DictConfig):
    """Load features from HDF5 file using compression config
    
    Args:
        features_path: Path to HDF5 features file
        cfg: Configuration object with scfm.compression.datasets
        
    Returns:
        Dictionary mapping dataset names to tensors
    """
    # Map unsupported NumPy dtypes to PyTorch-compatible ones
    DTYPE_MAP = {
        'uint16': 'int32',  # PyTorch doesn't support uint16
        'uint32': 'int64',  # PyTorch doesn't support uint32
        'uint64': 'int64',  # PyTorch doesn't support uint64
    }
    
    compression_datasets = cfg.scfm.compression.datasets
    
    loaded_data = {}
    with h5py.File(features_path, 'r') as f:
        for ds_cfg in compression_datasets:
            name = ds_cfg.name
            data = f[name][:]
            
            # Convert to PyTorch-compatible dtype if needed
            dtype_str = str(data.dtype)
            if dtype_str in DTYPE_MAP:
                data = data.astype(DTYPE_MAP[dtype_str])
            
            loaded_data[name] = torch.from_numpy(data)

    
    return loaded_data

def save_analysis_results(results_df, save_dir: Path, timestamp):
    """Save analysis results to CSV (config saved automatically by Hydra)
    
    Args:
        results_df: DataFrame with multilevel columns (analysis_type, metric)
        save_dir: Directory to save results
        timestamp: Timestamp string (e.g., 'Jan06-14-33')
    """
    
    # Create output directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results with timestamp in filename
    results_df.to_csv(save_dir / f"results-{timestamp}.csv")
    
    return save_dir

def load_analysis_results(save_dir: Path, timestamp):
    """Load existing analysis results
    
    Args:
        save_dir: Directory where results are saved
        timestamp: Timestamp string (e.g., 'Jan06-14-33')
        
    Returns:
        DataFrame with multilevel columns or None if doesn't exist
    """
    
    results_path = save_dir / f"results-{timestamp}.csv"
    
    if results_path.exists():
        return pd.read_csv(results_path, header=[0, 1], index_col=0)
    return None

def load_embeddings(adata, cfg: DictConfig):
    # Load steered embeddings based on experiments in config
    for exp in cfg.experiments:
        number = exp.number
        date = exp.date
        embeddings_path = Path('experiments/steer') / number / date
        for seed in exp.seeds:
            if exp.original:
                embeddings = np.load(embeddings_path / f"original_seed{seed}.npy")
                adata.obsm[f"{date}_original"] = embeddings
            for clamp in exp.clamps:
                for n_feat in exp.n_features:
                    for selector in exp.selector:
                        embeddings = np.load(embeddings_path / f"{selector}_n{n_feat}_seed{seed}_clamp{clamp}.npy")
                        adata.obsm[f"{date}_{selector}_n{n_feat}_seed{seed}_clamp{clamp}"] = embeddings

    return adata