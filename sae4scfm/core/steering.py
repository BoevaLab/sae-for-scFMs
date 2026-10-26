from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch

from sae4scfm.core.sae_inference import encode_rows
from sae4scfm.core.utils import set_seed


# ============================================================================
# Feature Selection Strategies
# ============================================================================

class RandomFeatureSelector():
    """Randomly select features"""
    
    def __init__(self, seed: int = 0, feature_file: Optional[str] = None, label: Optional[str] = None):
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
        feature_file: str,
        seed: int = 0,
        label: str = "batch",
    ):
        self.feature_file = Path(feature_file)
        self.label = label
        self.name = 'AMI'
        
        # Load and rank features once
        self.ranked_features = self._load_features()
    
    def _load_features(self) -> np.ndarray:
        """Rank features by their maximum AMI over the sublabels of `self.label`."""
        if not self.feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {self.feature_file}")

        df = pd.read_csv(self.feature_file, header=[0, 1], index_col=0)['label_scoring']
        columns = [c for c in df.columns if c.lower().startswith(self.label.lower()) and c.endswith("_AMI")]
        if not columns:
            raise ValueError(f"{self.feature_file} has no '{self.label}_*_AMI' columns; run the analysis "
                             f"with analysis.label_scoring.label including '{self.label}'.")
        # Features without a defined AMI (dead, or constant at every threshold) are not candidates.
        best = df[columns].max(axis=1).dropna()
        return best.sort_values(ascending=False, kind="stable").index.values
    
    def select_features(self, n_features: int, dict_size: int, trial: int = 0) -> np.ndarray:
        """Select top-N features from ranked list"""
        if n_features > len(self.ranked_features):
            raise ValueError(
                f"Requested {n_features} features but only {len(self.ranked_features)} available"
            )
        return self.ranked_features[:n_features]


# ============================================================================
# Steering
# ============================================================================

@dataclass
class Clamp:
    """Write `value` into SAE `features` of the steered tokens.

    With `fraction=None` the value replaces the features wherever they fire (suppression).
    Otherwise it is written, independently per feature, onto round(fraction * n) randomly
    chosen tokens of each cell with n real tokens (induction).
    """
    features: Sequence[int]
    value: Union[float, Sequence[float]]
    fraction: Optional[float] = None


def random_token_mask(cell_index: torch.Tensor, n_features: int, fraction: float,
                      generator: torch.Generator) -> torch.Tensor:
    """[N, n_features] bool: for every cell and feature, round(fraction * n_cell) random tokens."""
    chosen = torch.zeros(len(cell_index), n_features, dtype=torch.bool)
    cell_index = cell_index.cpu()
    for cell in torch.unique(cell_index):
        rows = torch.nonzero(cell_index == cell).squeeze(1)
        k = int(round(fraction * len(rows)))
        for j in range(n_features):
            chosen[rows[torch.randperm(len(rows), generator=generator)[:k]], j] = True
    return chosen


def apply_clamps(features: torch.Tensor, cell_index: torch.Tensor, clamps: Sequence[Clamp],
                 generator: torch.Generator) -> torch.Tensor:
    """Return a copy of token features [N, F] with the clamps applied (conditions use the input)."""
    steered = features.clone()
    for clamp in clamps:
        idx = torch.as_tensor(list(clamp.features), dtype=torch.long, device=features.device)
        value = torch.as_tensor(clamp.value, dtype=features.dtype, device=features.device).expand(len(idx))
        if clamp.fraction is None:
            where = features[:, idx] > 0
        else:
            where = random_token_mask(cell_index, len(idx), clamp.fraction, generator).to(features.device)
        steered[:, idx] = torch.where(where, value, steered[:, idx])
    return steered


def steer(
        adapter,
        sae,
        data_loader,
        clamp_sets: Dict[str, Sequence[Clamp]],
        device,
        inference_cfg=None,
        add_reconstruction_error: bool = False,
        return_reconstruction: bool = False,
        seed: int = 0,
):
    """Run the scFM with SAE features clamped at the hooked layer.

    Every clamp set is applied to each batch in one pass. Steered tokens are decode(f') by
    default, or x + decode(f') - decode(f) with `add_reconstruction_error`, which leaves tokens
    whose features did not change exactly as they were. Padding and special tokens outside the
    adapter's activation mask are never modified.

    Returns (rows ordered by the cells' position in the input data when batches carry `cell_idx`):
        original: [cells, D] embeddings without steering
        steered: {name: [cells, D]} one entry per clamp set
        reconstruction: [cells, D] embeddings of decode(encode(x)) without clamping, or None
        cell_idx: [cells] input positions of the rows, or None
    """
    generator = torch.Generator().manual_seed(seed)
    original, reconstruction, cell_ids = [], [], []
    steered = {name: [] for name in clamp_sets}

    with torch.no_grad():
        for batch in data_loader:
            mask = adapter.generate_activation_mask(batch).to(device)
            original_embedding, x = adapter.generate_embeddings(batch, device_model=device, return_activations=True)
            tokens = x[mask]
            features = encode_rows(sae, tokens, inference_cfg)
            cell_index = torch.nonzero(mask)[:, 0]
            if add_reconstruction_error or return_reconstruction:
                reconstructed = sae.decode(features)

            for name, clamps in clamp_sets.items():
                decoded = sae.decode(apply_clamps(features, cell_index, clamps, generator))
                x_steered = x.clone()
                x_steered[mask] = tokens + (decoded - reconstructed) if add_reconstruction_error else decoded
                steered[name].append(adapter.generate_embeddings(
                    batch, device_model=device, modify_activations=True, x_hat=x_steered))

            if return_reconstruction:
                x_reconstructed = x.clone()
                x_reconstructed[mask] = reconstructed
                reconstruction.append(adapter.generate_embeddings(
                    batch, device_model=device, modify_activations=True, x_hat=x_reconstructed))
            original.append(original_embedding)
            if isinstance(batch, dict) and "cell_idx" in batch:
                cell_ids.append(torch.as_tensor(batch["cell_idx"]).cpu())

    cell_idx = torch.cat(cell_ids).numpy() if cell_ids else None
    order = np.argsort(cell_idx, kind="stable") if cell_idx is not None else slice(None)
    stack = lambda embeddings: torch.vstack(embeddings).cpu().numpy()[order]
    return (stack(original), {name: stack(e) for name, e in steered.items()},
            stack(reconstruction) if return_reconstruction else None,
            cell_idx[order] if cell_idx is not None else None)


def compute_batch_integration(adata, embeddings):
    from scib_metrics.benchmark import Benchmarker, BioConservation

    emb_keys = list(adata.obsm)

    # Preprocessing
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="seurat_v3")#, batch_key="batch")
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
        n_jobs=1,
    )
    bm.benchmark()

    return bm