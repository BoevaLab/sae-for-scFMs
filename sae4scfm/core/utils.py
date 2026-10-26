"""
Utility functions and classes for SAE training and evaluation.
"""

import contextlib
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from typing import Dict
import logging
log = logging.getLogger(__name__)
from pathlib import Path

class OutputHook:
    """
    Forward hook to save or modify activations
    """
    def __init__(self):
        self.outputs = []
        self.x_hat = None
        self.modify = False

    def __call__(self, module, module_in, module_out):
        if self.modify:
            return self.x_hat
        else:
            self.outputs.append(module_out)
        return module_out

    def clear(self):
        self.outputs = []
    
    def __enter__(self):
        """Context manager entry: clear outputs before use"""
        self.clear()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: always clear outputs after use"""
        self.clear()
        return False  # Don't suppress exceptions

class SeqDataset(Dataset):
    """
    Custom Dataset for sequence data
    """
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    
class GroupedBatchSampler(torch.utils.data.Sampler):
    """Minibatches that each contain cells of a single group (e.g. one sequencing batch).

    Without shuffling, groups are served in sorted order and cells keep their original order
    within a group; with shuffling, cells are shuffled within groups and minibatches across groups.
    """
    def __init__(self, groups, batch_size: int, shuffle: bool = False):
        self.groups = np.asarray(groups)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _batches(self):
        batches = []
        for group in np.unique(self.groups):
            idx = np.flatnonzero(self.groups == group)
            if self.shuffle:
                idx = idx[torch.randperm(len(idx)).numpy()]
            batches += [idx[i:i + self.batch_size].tolist() for i in range(0, len(idx), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches)).tolist()]
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return sum(-(-int((self.groups == g).sum()) // self.batch_size) for g in np.unique(self.groups))


def get_devices() -> tuple:
    """
    Determines available CUDA devices and returns the primary device for training and a list of buffer devices.

    Returns:
        tuple: (device_training, device_buffer) where device_training is a string representing the main device,
               and device_buffer is a list of strings for additional devices (or ['cpu'] if no CUDA devices are available).
    """
    devices =  [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    if len(devices) == 0:
        device_training, device_buffer = 'cpu', ['cpu']
    elif len(devices) == 1:
        device_training, device_buffer = devices[0], [devices[0]]
    else:
        device_training = devices[0]
        device_buffer = devices[1:]
    return (device_training, device_buffer)

@contextlib.contextmanager
def seeded(seed):
    """Run a block with Python, NumPy and torch seeded, then restore the previous RNG states,
    so a seeded forward pass does not reset the random streams of the surrounding code."""
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
        set_seed(seed)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)


def set_seed(seed) -> None:
    """
    Sets the random seed for Python, NumPy, and PyTorch to ensure reproducibility.

    Args:
        seed (int): The seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


HVG_SPANS = (0.3, 0.5, 1.0)  # loess spans tried by highly_variable_genes; scanpy's default first
# Messages of the loess solver (scikit-misc) when the fit is numerically singular
LOESS_SINGULAR = ("reciprocal condition number", "near singularities", "pseudoinverse used")


def loess_singular(error: Exception) -> bool:
    return isinstance(error, ValueError) and any(m in str(error) for m in LOESS_SINGULAR)


def highly_variable_genes(adata, n_top_genes, flavor, subset, batch_key=None, spans=HVG_SPANS):
    """scanpy's highly_variable_genes (on adata.X). When the seurat_v3 loess fit of the
    mean-variance trend is singular (small or very sparse data), it is retried with a wider loess
    span; runs that succeed with the first span are unchanged."""
    import scanpy as sc

    for span in spans:
        try:
            sc.pp.highly_variable_genes(adata, layer=None, n_top_genes=n_top_genes, batch_key=batch_key,
                                        flavor=flavor, subset=subset, span=span)
        except ValueError as e:
            if not loess_singular(e) or span == spans[-1]:
                raise
            log.warning(f"HVG ({flavor}): singular loess fit with span {span}; retrying with a wider span")
            continue
        if span != HVG_SPANS[0]:
            log.warning(f"HVG ({flavor}) selected with loess span {span} (default {HVG_SPANS[0]} failed)")
        return


def subset_to_gene_list(adata, spec, data_cfg):
    """Subset `adata` to a gene list when `spec` (a preprocess.subset_hvg value) is a file name.

    The file, relative to the dataset folder (data_cfg.file_path), lists one gene symbol per line
    (lines starting with # are skipped). Any other `spec` (False or a number of HVGs) leaves
    `adata` unchanged.
    """
    if not isinstance(spec, str):
        return adata
    genes_path = Path(data_cfg.get("file_path", None) or data_cfg.get("path", "")) / spec
    if not genes_path.exists():
        raise FileNotFoundError(f"Gene list file not found: {genes_path}")
    genes = [line.strip() for line in genes_path.read_text().splitlines()
             if line.strip() and not line.strip().startswith("#")]
    if not genes:
        raise ValueError(f"Gene list file is empty: {genes_path}")
    adata = adata[:, adata.var["gene_name"].astype(str).isin(genes).values]
    log.info(f"Gene-list filter applied (subset_hvg='{spec}'). adata shape: {adata.shape}")
    return adata
