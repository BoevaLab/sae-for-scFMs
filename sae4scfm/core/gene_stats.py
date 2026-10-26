"""Gene-resolved activation statistics of SAE features.

`FeatureGeneAccumulator` streams token-level features and keeps, for every (feature, vocab gene):

- ``act_sum``: summed activation;
- ``act_expr_sum`` and ``act_expr_sq_sum``: activation-weighted sums of the token's input
  expression value and of its square;
- ``active_count``: number of tokens where the feature fired;

plus ``gene_occurrence`` (tokens per gene) and, per feature, the sum and sum of squares of its
per-cell max activation. `GeneStats.load` reads them back (sparse), and `feature_table` turns
them into one row of summary metrics per feature.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

TABLES = ("act_sum", "act_expr_sum", "act_expr_sq_sum", "active_count")


class FeatureGeneAccumulator:
    """Streaming [n_features, n_genes] accumulators (float64 sums, int64 counts)."""

    def __init__(self, n_features: int, n_genes: int, device="cpu"):
        self.n_features, self.n_genes = int(n_features), int(n_genes)
        shape = (self.n_features, self.n_genes)
        self.act_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.act_expr_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.act_expr_sq_sum = torch.zeros(shape, dtype=torch.float64, device=device)
        self.active_count = torch.zeros(shape, dtype=torch.int64, device=device)
        self.gene_occurrence = torch.zeros(self.n_genes, dtype=torch.int64, device=device)
        self.cell_sum = torch.zeros(self.n_features, dtype=torch.float64, device=device)
        self.cell_sq_sum = torch.zeros(self.n_features, dtype=torch.float64, device=device)
        self.n_cells = 0
        self.total_valid_tokens = 0
        self.total_active_entries = 0

    @torch.no_grad()
    def update(self, features, gene_ids, expression, gene_mask, token_mask=None) -> None:
        """Add one batch.

        Args:
            features: [B, T, F] SAE features (zero where the feature did not fire)
            gene_ids: [B, T] vocab ids
            expression: [B, T] input expression value of each token
            gene_mask: [B, T] tokens that are genes (no padding or special tokens)
            token_mask: [B, T] tokens the SAE encodes, for the per-cell max (default: gene_mask)
        """
        device = self.act_sum.device
        features, gene_ids = features.to(device), gene_ids.to(device).long()
        expression, gene_mask = expression.to(device), gene_mask.to(device)
        gene_mask = gene_mask & (gene_ids >= 0) & (gene_ids < self.n_genes)
        token_mask = gene_mask if token_mask is None else token_mask.to(device)

        pooled = (features * token_mask[..., None]).amax(dim=1).double()  # features are >= 0
        self.cell_sum += pooled.sum(dim=0)
        self.cell_sq_sum += (pooled * pooled).sum(dim=0)
        self.n_cells += features.shape[0]

        genes = gene_ids[gene_mask]
        self.gene_occurrence += torch.bincount(genes, minlength=self.n_genes)
        self.total_valid_tokens += int(genes.numel())

        b, t, f = (features * gene_mask[..., None]).nonzero(as_tuple=True)
        if b.numel() == 0:
            return
        act = features[b, t, f].double()
        expr = expression[b, t].double()
        flat = f * self.n_genes + gene_ids[b, t]
        self.act_sum.view(-1).index_add_(0, flat, act)
        self.act_expr_sum.view(-1).index_add_(0, flat, act * expr)
        self.act_expr_sq_sum.view(-1).index_add_(0, flat, act * expr * expr)
        self.active_count.view(-1).index_add_(0, flat, torch.ones_like(flat))
        self.total_active_entries += int(flat.numel())

    def save(self, save_dir) -> Path:
        """Write the tables as float32 / int64 CSR ``.npz`` plus ``gene_occurrence.npy``,
        ``cell_moments.npy`` ([2, F]: sum and sum of squares of the per-cell max) and ``meta.json``."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            values = getattr(self, name).cpu().numpy()
            values = values if name == "active_count" else values.astype(np.float32)
            sp.save_npz(save_dir / f"{name}.npz", sp.csr_matrix(values))
        np.save(save_dir / "gene_occurrence.npy", self.gene_occurrence.cpu().numpy())
        np.save(save_dir / "cell_moments.npy", torch.stack([self.cell_sum, self.cell_sq_sum]).cpu().numpy())
        meta = dict(n_features=self.n_features, n_vocab_genes=self.n_genes, sparse=True,
                    save_dtype="float32", n_cells=self.n_cells,
                    total_valid_tokens=self.total_valid_tokens,
                    total_active_entries=self.total_active_entries)
        (save_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return save_dir


def load_table(accum_dir, name: str) -> sp.csr_matrix:
    """One [F, G] table from a saved accumulator (sparse ``.npz`` or, in older runs, dense ``.npy``)."""
    accum_dir = Path(accum_dir)
    if (accum_dir / f"{name}.npz").exists():
        return sp.load_npz(accum_dir / f"{name}.npz").tocsr()
    return sp.csr_matrix(np.load(accum_dir / f"{name}.npy"))


class GeneStats:
    """A saved accumulator, kept sparse."""

    def __init__(self, accum_dir):
        accum_dir = Path(accum_dir)
        self.meta = json.loads((accum_dir / "meta.json").read_text())
        for name in TABLES:
            setattr(self, name, load_table(accum_dir, name))
        self.gene_occurrence = np.load(accum_dir / "gene_occurrence.npy")
        moments = accum_dir / "cell_moments.npy"  # absent in runs from before it was added
        self.cell_sum, self.cell_sq_sum = np.load(moments) if moments.exists() else (None, None)


def profile(act_sum: sp.csr_matrix, gene_occurrence: np.ndarray) -> sp.csr_matrix:
    """Frequency-normalized profile: each feature's mean activation per occurrence of each gene."""
    return sp.csr_matrix(act_sum.multiply(1.0 / np.maximum(gene_occurrence, 1)), dtype=np.float64)


def topk_mass(prof: sp.csr_matrix, ks=(20,)) -> dict:
    """{k: fraction of each row's mass on its k largest entries} (NaN for empty rows)."""
    out = {k: np.full(prof.shape[0], np.nan) for k in ks}
    for i in range(prof.shape[0]):
        row = np.sort(prof.data[prof.indptr[i]:prof.indptr[i + 1]])[::-1]
        if row.sum() > 0:
            for k in ks:
                out[k][i] = row[:k].sum() / row.sum()
    return out


def _ratio(num, den):
    num, den = np.asarray(num, dtype=np.float64).ravel(), np.asarray(den, dtype=np.float64).ravel()
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den > 0)
    return out


def _row_sums(m) -> np.ndarray:
    return np.asarray(m.sum(axis=1), dtype=np.float64).ravel()


def feature_table(stats: GeneStats, min_support: int = 1000, top_k=(5, 20, 50)) -> pd.DataFrame:
    """One row per feature.

    Concentration, on the frequency-normalized profile p over all genes:
        n_eff_genes     (sum p)^2 / sum p^2, the effective number of genes the feature fires on
        topk{k}_mass    fraction of the profile on its top-k genes
    Expression band, over the genes where the feature fired at least `min_support` times:
        expr_focus      activation-weighted mean input expression of the firing tokens
        expr_std        activation-weighted standard deviation; expr_cv = expr_std / expr_focus
        mass_on_supported  fraction of the feature's activation on those genes
    Cells:
        n_eff_cells     (sum a)^2 / sum a^2 of the per-cell max activation a
    """
    prof = profile(stats.act_sum, stats.gene_occurrence)
    total = _row_sums(prof)
    supported = stats.active_count >= min_support
    den = _row_sums(stats.act_sum.multiply(supported))
    focus = _ratio(_row_sums(stats.act_expr_sum.multiply(supported)), den)
    meansq = _ratio(_row_sums(stats.act_expr_sq_sum.multiply(supported)), den)
    std = np.sqrt(np.clip(meansq - focus ** 2, 0, None))
    df = pd.DataFrame({
        "feature_id": np.arange(prof.shape[0]),
        "total_active_count": _row_sums(stats.active_count).astype(np.int64),
        "n_eff_genes": _ratio(total ** 2, _row_sums(prof.multiply(prof))),
        **{f"topk{k}_mass": v for k, v in topk_mass(prof, top_k).items()},
        "expr_focus": focus,
        "expr_std": std,
        "expr_cv": _ratio(std, np.where(focus > 0, focus, 0)),
        "mass_on_supported": _ratio(den, _row_sums(stats.act_sum)),
        "n_eff_cells": (_ratio(stats.cell_sum ** 2, stats.cell_sq_sum) if stats.cell_sum is not None
                        else np.nan),
    })
    df.attrs.update(min_support=int(min_support), profile="freq_normalized")
    return df
