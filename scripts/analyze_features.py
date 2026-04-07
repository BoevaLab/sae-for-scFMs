"""
Score SAE features against cell labels.

Reads the features written by generate_features (token-level ``.h5`` or per-cell max-pooled
``.pt``, with optional ``analysis.tag``), aligns their rows to the cells of the evaluation subset, and scores every feature
against every one-vs-rest sublabel of the configured labels (AMI over activation thresholds,
optionally F1). Writes ``experiments/analysis/<experiment>/results-<timestamp><tag>.csv`` with the
column groups ``label_scoring`` (``<label>_<sublabel>_<score>``), ``label_threshold`` (the
activation threshold that gave the best AMI) and ``label_direction`` (mean scaled activation in the
sublabel's cells minus outside them: positive when the feature marks the sublabel, e.g. treated
rather than control cells).
"""
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter used by the feature files)
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.analysis import label_scoring, preprocess_labels
from sae4scfm.core.io_utils import load_adata, save_analysis_results
from sae4scfm.core.pipeline import register_resolvers, run_config

register_resolvers()


class FeatureFile:
    """Per-cell, max-pooled activations of one feature at a time, from either output format.

    ``cell_idx`` gives each row's position in the evaluation subset (-1 for empty rows); files
    written before it existed are positional.
    """

    def __init__(self, cfg: DictConfig, n_cells: int):
        name = f"features-{cfg.sae_checkpoint.timestamp}{cfg.analysis.get('tag', '')}"
        folder = REPO_ROOT / "cached_features" / cfg.sae_checkpoint.experiment
        h5_path, pt_path = folder / f"{name}.h5", folder / f"{name}.pt"
        fmt = cfg.analysis.get("input", "auto")
        if fmt == "auto":
            fmt = "h5" if h5_path.exists() else "pooled"
        if fmt == "h5":
            self._h5 = h5py.File(h5_path, "r")
            self._features = self._h5["features"]
            row_cells = self._h5["cell_idx"][:] if "cell_idx" in self._h5 else np.arange(self._features.shape[0])
        elif fmt == "pooled":
            self._h5 = None
            self._features = torch.load(pt_path, map_location="cpu").numpy()
            cell_idx_path = folder / f"{name}.cell_idx.npy"
            row_cells = np.load(cell_idx_path) if cell_idx_path.exists() else np.arange(self._features.shape[0])
        else:
            raise ValueError(f"analysis.input must be auto, h5 or pooled, got {fmt!r}")
        self.path = h5_path if fmt == "h5" else pt_path
        self.n_features = self._features.shape[-1]
        self.rows = np.flatnonzero((row_cells >= 0) & (row_cells < n_cells))
        self.cells = row_cells[self.rows]

    def pooled(self, j: int) -> np.ndarray:
        """Feature j per cell (rows in `cells` order), scaled to [0, 1] by its max."""
        x = self._features[:, :, j] if self._h5 is not None else self._features[:, j]
        x = x[self.rows]
        if x.ndim == 2:
            x = x.max(axis=1)
        top = x.max()
        return x / top if top > 0 else x

    def close(self):
        if self._h5 is not None:
            self._h5.close()


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="analyze", version_base='1.3')
def main(cfg: DictConfig) -> None:
    original_cfg = run_config(cfg, REPO_ROOT)
    scoring = cfg.analysis.label_scoring

    # The evaluation subset generate_features encoded
    adata = load_adata(original_cfg, sub_sample=original_cfg.data.preprocess.get("split", 1.0))
    features = FeatureFile(cfg, adata.n_obs)
    print(f"Scoring {features.n_features} features from {features.path} on {len(features.cells)} cells")

    labels = preprocess_labels(adata.obs.iloc[features.cells], scoring.label, original_cfg.data.name,
                               min_cells=scoring.get("min_cells", 10),
                               columns=original_cfg.data.get("label_columns", None))

    results = {}
    for j in tqdm(range(features.n_features)):
        scored = label_scoring(features.pooled(j), labels, scoring.label, scoring.score)
        for label, sublabels in scored.items():
            for sublabel, metrics in sublabels.items():
                for metric, value in metrics.items():
                    if metric == "AMI_threshold":
                        key = ("label_threshold", f"{label}_{sublabel}_AMI")
                    elif metric == "direction":
                        key = ("label_direction", f"{label}_{sublabel}")
                    else:
                        key = ("label_scoring", f"{label}_{sublabel}_{metric}")
                    results.setdefault(key, [np.nan] * features.n_features)[j] = value
    features.close()

    df = pd.DataFrame(results)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["analysis_type", "metric"])
    df.index.name = "feature_id"

    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    name = cfg.sae_checkpoint.timestamp + cfg.analysis.get("tag", "")
    save_analysis_results(df, save_dir, name)
    print(f"Analysis results saved to {save_dir / f'results-{name}.csv'}")


if __name__ == "__main__":
    main()
