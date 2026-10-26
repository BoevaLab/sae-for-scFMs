from pathlib import Path
from typing import Any
import scvi
from scanpy.pp import subsample
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_class
import json
from tqdm import tqdm
import h5py
import hdf5plugin
import torch
import pandas as pd

from sae4scfm.core.sae_inference import encode_tokens
from sae4scfm.core.utils import set_seed

# Gene id stored for padding positions in feature files (-1 cast to uint16)
INVALID_GENE_ID = 65535

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
        adata.obs["batch"] = adata.obs["sample_id"].astype(str)
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
        #adata.X = adata.layers['counts']
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['celltype'] = adata.obs['cell_type']
        adata.obs["batch"] = adata.obs["batch"].astype(str)

    elif cfg.data.name == 'immune':
        adata = scvi.data.read_h5ad(data_path)
        adata.X = adata.layers['counts']
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['celltype'] = adata.obs['final_annotation'].astype("category").cat.codes.values
        adata.obs["batch"] = adata.obs["batch"].astype(str)

    elif cfg.data.name == 'replogle':
        adata = scvi.data.read_h5ad(data_path)
        #adata = adata[adata.obs['perturbation'].isin(cfg.data.perturbations)]
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['celltype'] = adata.obs['celltype'].cat.codes.values
        adata.obs["batch"] = adata.obs["batch"].astype(str)

    elif cfg.data.name == 'kang':
        adata = scvi.data.read_h5ad(data_path)
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['celltype'] = adata.obs['cell_type'].cat.codes.values
        adata.obs["batch"] = adata.obs["replicate"].astype(str)

    elif cfg.data.name.startswith('tahoe'):
        adata = scvi.data.read_h5ad(data_path)
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs['celltype'] = adata.obs['cell_line'].cat.codes.values
        adata.obs["batch"] = adata.obs["sample"].astype(str)
        n_file = adata.n_obs
        adata = select_tahoe_cells(adata, cfg.data.preprocess, cfg.data.get('control', 'DMSO_TF'))
        if adata.n_obs < n_file:
            print(f"Tahoe cell selection: {n_file} -> {adata.n_obs} cells")

    elif cfg.data.name.startswith('census'):
        adata = scvi.data.read_h5ad(data_path)
        adata.var["feature_name"] = adata.var["feature_name"].astype('str')
        adata.var = adata.var.set_index("feature_name")
        adata.var["gene_name"] = adata.var.index.tolist()
        adata.obs["batch"] = adata.obs["dataset_id"].astype(str)
    else:
        raise ValueError(f"Dataset {cfg.data.name} not recognized.")

    # Subsample data
    if sub_sample < 1:
        subsample(adata, sub_sample, random_state =42)

    return adata

def select_tahoe_cells(adata, preprocess, control: str = 'DMSO_TF'):
    """Cells of a Tahoe file kept by the selection settings in data.preprocess, applied in order:

    dose             keep the controls and the cells treated at this concentration (drugconc, uM)
    skip_cell_lines  drop the treated cells of these lines and keep their controls (SAE training)
    drop_cell_lines  drop every cell of these lines (e.g. drug AMI on the training lines only)
    keep_cell_lines  keep only these lines
    balance          {cell_lines: N, cells: n, seed: s}: the N lines with most cells, then n control
                     and n treated cells drawn per line (the balanced design of the induction sweep;
                     same draw as pandas groupby(['cell_line', group]).sample(n, random_state=s))

    Rows keep their order in the file.
    """
    obs = adata.obs
    is_control = (obs['drug'].astype(str) == control).to_numpy()
    lines = obs['cell_line'].astype(str)
    keep = np.ones(adata.n_obs, dtype=bool)
    if preprocess.get('dose', None) is not None:
        keep &= is_control | np.isclose(obs['drugconc'].astype(float).to_numpy(), float(preprocess.dose))
    if preprocess.get('skip_cell_lines', None):
        keep &= ~(lines.isin(list(preprocess.skip_cell_lines)).to_numpy() & ~is_control)
    if preprocess.get('drop_cell_lines', None):
        keep &= ~lines.isin(list(preprocess.drop_cell_lines)).to_numpy()
    if preprocess.get('keep_cell_lines', None):
        keep &= lines.isin(list(preprocess.keep_cell_lines)).to_numpy()
    balance = preprocess.get('balance', None)
    if balance:
        kept = obs.iloc[np.flatnonzero(keep)][['cell_line']].copy()
        kept.index = np.flatnonzero(keep)  # positions in the file
        kept['group'] = np.where(is_control[kept.index], control, 'other')
        top = kept['cell_line'].value_counts().nlargest(int(balance.cell_lines)).index
        kept = kept[kept['cell_line'].isin(top)]
        groups = kept.groupby(['cell_line', 'group'], observed=True)
        if groups.size().min() < int(balance.cells) or groups.ngroups < 2 * len(top):
            raise ValueError(f"balance: every one of the {len(top)} lines needs {balance.cells} control and "
                             f"{balance.cells} treated cells; smallest group: {groups.size().min()}")
        picked = groups.sample(n=int(balance.cells), random_state=int(balance.get('seed', 0))).index
        keep = np.zeros(adata.n_obs, dtype=bool)
        keep[picked] = True
    return adata if keep.all() else adata[keep]


def save_sae_checkpoint(sae, save_dir: Path) -> None:
    """Save SAE model checkpoint
    
    Args:
        sae: Trained SAE model
        save_dir: Directory to save the model
    """
    torch.save(sae.state_dict(), save_dir / "model.pth")


def _load_training_config(checkpoint_path: Path):
    config_path = checkpoint_path / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return OmegaConf.load(config_path)


def _validate_resume_config(current_cfg: DictConfig, original_cfg: DictConfig) -> None:
    checks = (
        ("sae.autoencoder._target_", current_cfg.sae.autoencoder._target_, original_cfg.sae.autoencoder._target_),
        ("sae.trainer._target_", current_cfg.sae.trainer._target_, original_cfg.sae.trainer._target_),
        ("sae.dict_size", current_cfg.sae.dict_size, original_cfg.sae.dict_size),
        ("scfm.activation_dim", current_cfg.scfm.activation_dim, original_cfg.scfm.activation_dim),
    )
    mismatches = [
        f"{name}: current={current!r}, original={original!r}"
        for name, current, original in checks
        if current != original
    ]
    if mismatches:
        raise ValueError("Resume config mismatch detected:\n" + "\n".join(mismatches))


def save_training_checkpoint(trainer, save_dir: Path, step: int, cfg: DictConfig | None = None) -> Path:
    """Persist training state for resuming a run."""
    checkpoint = trainer.state_dict(step=step)
    checkpoint["metadata"] = {
        "trainer_target": cfg.sae.trainer._target_ if cfg is not None else None,
        "autoencoder_target": cfg.sae.autoencoder._target_ if cfg is not None else None,
    }
    checkpoint_path = save_dir / "training_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_training_checkpoint(
    checkpoint_path: Path,
    device: str,
    current_cfg: DictConfig | None = None,
    strict_config: bool = True,
) -> tuple[dict[str, Any], DictConfig]:
    """Load resumable training state and the original Hydra config."""
    checkpoint_path = Path(checkpoint_path)
    original_cfg = _load_training_config(checkpoint_path)
    if current_cfg is not None and strict_config:
        _validate_resume_config(current_cfg, original_cfg)

    training_checkpoint_path = checkpoint_path / "training_checkpoint.pt"
    if not training_checkpoint_path.exists():
        raise FileNotFoundError(f"Training checkpoint not found: {training_checkpoint_path}")

    training_state = torch.load(training_checkpoint_path, map_location=device)
    return training_state, original_cfg

def load_sae_checkpoint(
    checkpoint_path: Path,
    device: str,
    load_sae: bool = True,
    strict_config: bool = True,
):
    """Load trained SAE from checkpoint directory
    
    Args:
        checkpoint_path: Path to checkpoint (e.g., experiments/train/exp00/Jan06-14-33)
        device: Device to load model on
        load_sae: Whether to instantiate and load the SAE weights
        strict_config: Whether to validate config compatibility when loading training state
        
    Returns:
        sae: Loaded SAE model
        original_cfg: Original training configuration
    """
    # Load original training config
    model_path = checkpoint_path / "model.pth"

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    original_cfg = _load_training_config(checkpoint_path)

    if load_sae:
        sae = get_class(original_cfg.sae.autoencoder._target_).from_pretrained(model_path, device = device)
        sae.eval()
    else:
        sae = None
    
    
    return sae, original_cfg

def save_training_summary(trainer, steps: int, save_dir: Path) -> None:
    """Record how training ended (steps, inference threshold), read back by sae_inference.prepare_sae."""
    summary = {"steps": int(steps)}
    if hasattr(trainer.ae, "threshold"):
        summary["threshold"] = float(trainer.ae.threshold)
        summary["threshold_start_step"] = int(getattr(trainer, "threshold_start_step", 0))
        summary["threshold_beta"] = float(getattr(trainer, "threshold_beta", 0.999))
    with open(Path(save_dir) / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def load_training_summary(checkpoint_path: Path):
    """The summary written by save_training_summary, or None for runs that predate it."""
    path = Path(checkpoint_path) / "training_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


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

def n_rows(data_loader, cfg) -> int:
    """Rows of a feature file: the cells the loader yields when it knows them (single-file data, after
    any cell selection), else data.n_cells x data.preprocess.split (streamed data)."""
    n = data_loader.n_cells() if hasattr(data_loader, 'n_cells') else None
    return n if n is not None else int(cfg.data.n_cells[0] * cfg.data.preprocess.split)


def generate_and_compress_features(
        adapter, 
        sae, 
        data_loader, 
        device, 
        cfg,
        save_dir: Path,
        date,
        inference_cfg=None) -> None:
    """Generate and compress features"""
    
    set_seed(cfg.seed)

    # Tensor size and chunking
    n_samples = n_rows(data_loader, cfg)
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
        # Input position of each row (-1 for rows left empty, e.g. cells the adapter dropped)
        cell_idx_ds = f.create_dataset("cell_idx", data=np.full(n_samples, -1, dtype=np.int64))

        write_pos = 0
        
        for data_dict in tqdm(data_loader):
            
            activations = adapter.generate_activations(data_dict, device)
            
            # Encode real tokens only, so padding takes no share of the batch top-k budget.
            mask = adapter.generate_activation_mask(data_dict).to(device)
            with torch.no_grad():
                features = encode_tokens(sae, activations, mask, inference_cfg)

            # Adapter returns variable number of arrays based on model
            compressed_data = adapter.prepare_compression(data_dict, features, n_genes, device)

            batch_size = compressed_data[0].shape[0]

            # Write each dataset returned by adapter
            for i, (ds_name, ds) in enumerate(datasets.items()):
                ds[write_pos:write_pos + batch_size] = compressed_data[i]
            if "cell_idx" in data_dict:
                cell_idx_ds[write_pos:write_pos + batch_size] = torch.as_tensor(data_dict["cell_idx"]).numpy()

            write_pos += batch_size
                      
    
    return

def generate_pool_and_save_features(
        adapter, 
        sae, 
        data_loader, 
        device, 
        cfg,
        save_dir: Path,
        date,
        inference_cfg=None) -> None:
    """Generate and compress features"""
    
    set_seed(cfg.seed)

    # Tensor size and chunking
    n_samples = n_rows(data_loader, cfg)
    dict_size = cfg.sae.dict_size
    features = torch.zeros(n_samples, dict_size, dtype=torch.float16, device='cpu')
    cell_idx = torch.full((n_samples,), -1, dtype=torch.long)

    samples_filled = 0
    for data_dict in tqdm(data_loader):
        activations = adapter.generate_activations(data_dict, device)
        batch_size = activations.shape[0]

        mask = adapter.generate_activation_mask(data_dict).to(device)
        with torch.no_grad():
            feats = encode_tokens(sae, activations, mask, inference_cfg)  # zero outside the mask
        feats = feats.max(axis=1).values # maxpool genes
        feats = feats.cpu().to(torch.float16)
        
        features[samples_filled:samples_filled + batch_size] = feats
        if "cell_idx" in data_dict:
            cell_idx[samples_filled:samples_filled + batch_size] = torch.as_tensor(data_dict["cell_idx"])
        samples_filled += batch_size

    # Rows in input order; rows of dropped cells (cell_idx -1) go last
    order = torch.argsort(torch.where(cell_idx < 0, cell_idx.max() + 1, cell_idx), stable=True)
    torch.save(features[order], save_dir / f'features-{date}.pt')
    np.save(save_dir / f'features-{date}.cell_idx.npy', cell_idx[order].numpy())
                      
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
        'uint16': np.int32,
        'uint32': np.int64,
        'uint64': np.int64,
    }
    compression_datasets = cfg.scfm.compression.datasets
    loaded_data = {}

    with h5py.File(features_path, 'r') as f:
        for ds_cfg in compression_datasets:
            name = ds_cfg.name
            dataset = f[name]
            dtype_str = str(dataset.dtype)

            if dtype_str in DTYPE_MAP:
                target_dtype = DTYPE_MAP[dtype_str]
                # Pre-allocate the target array directly
                data = np.empty(dataset.shape, dtype=target_dtype)
                dataset.read_direct(data)  # HDF5 converts in-place during read
            else:
                data = dataset[:]

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
    """Add steered embeddings listed in cfg.experiments to adata.obsm.

    Optional per-experiment keys: `suffix` (e.g. "_err" for runs with the reconstruction
    error added back) and `reconstruction: true` to also load recon_seed{seed}.npy.
    """
    cell_idx = None
    for exp in cfg.experiments:
        number = exp.number
        date = exp.date
        suffix = exp.get("suffix", "")
        embeddings_path = Path('experiments/steer') / number / date
        # Steered embeddings cover the cells the adapter kept, in input order (cell_idx.npy)
        if (embeddings_path / "cell_idx.npy").exists():
            idx = np.load(embeddings_path / "cell_idx.npy")
            if cell_idx is None:
                cell_idx = idx
                if not np.array_equal(idx, np.arange(adata.n_obs)):
                    adata._inplace_subset_obs(idx)
            elif not np.array_equal(idx, cell_idx):
                raise ValueError(f"{embeddings_path} covers different cells than the other experiments")
        for seed in exp.seeds:
            if exp.original:
                embeddings = np.load(embeddings_path / f"original_seed{seed}.npy")
                adata.obsm[f"{date}_original"] = embeddings
            if exp.get("reconstruction", False):
                adata.obsm[f"{date}_recon_seed{seed}"] = np.load(embeddings_path / f"recon_seed{seed}.npy")
            for clamp in exp.clamps:
                for n_feat in exp.n_features:
                    for selector in exp.selector:
                        name = f"{selector}_n{n_feat}_seed{seed}_clamp{clamp}{suffix}"
                        adata.obsm[f"{date}_{name}"] = np.load(embeddings_path / f"{name}.npy")

    return adata