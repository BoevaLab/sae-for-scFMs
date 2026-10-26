"""
Score drug induction: how far SAE-steered control cells move toward the treated population.

Reads an induction run of steer_features (steering.mode=induce; original.npy,
steered_f<f>_v<v>.npy, cell_idx.npy) and the metadata of the cells it steered. Embeddings are
standardized on the original cells. For every (f, v) and cell line, the steered control cells are
scored against that line's own control (C) and treated (P) cells (Supplementary Methods S1.6):

- CMS, cluster merging score: fraction of a steered cell's k nearest neighbours in P u C that are
  treated, averaged over cells;
- DAS, directional alignment score: projection of the steered cell onto the C-to-P centroid axis,
  0 at the control centroid and 1 at the treated centroid, averaged over cells.

f and v are selected on the training cell lines by the lowest mean of
w_cms |1 - CMS| + w_das |1 - DAS|; held-out lines are only scored at the selected values.

Outputs (experiments/perturbation/<experiment>/<timestamp>/<drug>/):
``sweep_scores.csv``     CMS and DAS per (f, v) and cell line (split train / heldout), and pooled
                         over all lines (cell_line "all")
``baseline_scores.csv``  the same for unsteered controls and for treated cells
``selection.csv``        selection objective per (f, v) on the training lines; `selected` marks the chosen pair
``dose_das.csv``         if `dose.column` is set: per-cell DAS of steered controls for every f at the
                         selected v and of the cells of each dose, on the axis from the control
                         centroid to the centroid of the highest dose (Fig. 3d). Computed on
                         `dose.run` (an induction run over all doses, e.g. the trajectory run) when
                         given, else on `steer_run`
``dose_activation.csv``  if `dose_activation.features` is set: activation (scaled to its maximum) of
                         those features per cell (`cell`: position in the dataset) and dose, with a
                         Kruskal-Wallis test and Mann-Whitney tests of consecutive doses, BH-corrected
                         per feature (Fig. 3f). The cells' doses come from `dose_activation.cells`
                         (cells.csv of a run on the features' data, e.g. embed_cells) when given,
                         else from the steering run's data
"""
import os
import re
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import kruskal, mannwhitneyu
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.io_utils import load_adata
from sae4scfm.core.pipeline import register_resolvers, run_config

register_resolvers()

STEERED = re.compile(r"^steered_f(?P<f>[\d.]+)_v(?P<v>[\d.]+)(?P<suffix>.*)\.npy$")


# ---------------------------------------------------------------------------- scores

def cms(query: np.ndarray, control: np.ndarray, treated: np.ndarray, k: int = 15) -> np.ndarray:
    """Per query cell, the fraction of its k nearest neighbours in control u treated that are treated."""
    reference = np.vstack([control, treated])
    is_treated = np.r_[np.zeros(len(control)), np.ones(len(treated))]
    _, idx = NearestNeighbors(n_neighbors=k).fit(reference).kneighbors(query)
    return is_treated[idx].mean(axis=1)


def das(query: np.ndarray, control: np.ndarray, treated: np.ndarray) -> np.ndarray:
    """Per query cell, the projection onto the control-to-treated centroid axis (0: control, 1: treated)."""
    c0 = control.mean(axis=0)
    axis = treated.mean(axis=0) - c0
    return (query - c0) @ axis / (axis @ axis)


def score_groups(query, query_lines, original, lines, is_control, is_treated, k):
    """CMS and DAS of the `query` cells (cell lines `query_lines`) against the control and treated
    cells of `original` (cell lines `lines`), per cell line and pooled over all lines. Lines with
    fewer than k reference cells are skipped."""
    rows = []
    groups = [("all", np.ones(len(query_lines), dtype=bool), np.ones(len(lines), dtype=bool))]
    groups += [(line, query_lines == line, lines == line) for line in sorted(np.unique(query_lines))]
    for line, q_in, in_line in groups:
        q, c, t = query[q_in], original[is_control & in_line], original[is_treated & in_line]
        if len(q) == 0 or len(c) == 0 or len(t) == 0 or len(c) + len(t) < k:
            continue
        rows.append(dict(cell_line=line, CMS=cms(q, c, t, k).mean(), DAS=das(q, c, t).mean(),
                         n_query=len(q), n_control=len(c), n_treated=len(t)))
    return rows


# ---------------------------------------------------------------------------- inputs

def steered_files(run_dir: Path, suffix: str) -> dict:
    """{(f, v): path} of the steered embeddings of an induction run."""
    out = {}
    for path in run_dir.glob("steered_f*_v*.npy"):
        m = STEERED.match(path.name)
        if m and m["suffix"] == suffix:
            out[(float(m["f"]), float(m["v"]))] = path
    if not out:
        raise FileNotFoundError(f"No steered_f*_v*{suffix}.npy in {run_dir}")
    return dict(sorted(out.items()))


def run_cells(run_dir: Path) -> pd.DataFrame:
    """Metadata of the cells an induction run steered, in the row order of its embeddings."""
    steer_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    original_cfg = run_config(steer_cfg, REPO_ROOT)
    adata = load_adata(original_cfg, sub_sample=steer_cfg.steering.get("data_split", 1.0))
    cell_idx_path = run_dir / "cell_idx.npy"
    rows = np.load(cell_idx_path) if cell_idx_path.exists() else np.arange(adata.n_obs)
    return adata.obs.iloc[rows].reset_index(drop=True), steer_cfg, original_cfg


def dose_trajectory(original, doses, is_control, is_treated, steered_control, v) -> pd.DataFrame:
    """Per-cell DAS of the controls, of the treated cells of each dose and of the steered controls at
    multiplier `v` (every f), on the axis from the control centroid to the highest dose's centroid.
    `steered_control`: {(f, v): steered embeddings of the control cells}."""
    levels = dose_order(doses[is_treated])
    top = original[is_treated & (doses == levels[-1]).to_numpy()]
    control = original[is_control]
    steered = {fv: x for fv, x in steered_control.items() if fv[1] == v}
    if not steered:
        raise ValueError(f"the dose run has no steered embeddings at the selected v={v}")
    rows = [pd.DataFrame({"group": "control", "dose": 0.0, "das": das(control, control, top)})]
    rows += [pd.DataFrame({"group": "treated", "dose": float(d),
                           "das": das(original[is_treated & (doses == d).to_numpy()], control, top)}) for d in levels]
    rows += [pd.DataFrame({"group": "steered", "f": f, "v": v_, "das": das(x, control, top)})
             for (f, v_), x in steered.items()]
    return pd.concat(rows, ignore_index=True)


def dose_order(values: pd.Series) -> list:
    """Dose levels in numeric order (the notebook sorted them as strings)."""
    return sorted(values.dropna().unique(), key=float)


def dose_activation(pooled: np.ndarray, doses: pd.Series, features, cells=None) -> tuple:
    """Activation of each feature per dose (scaled to its maximum) and tests across doses.
    `cells` (dataset positions of the rows) is carried into the long table."""
    order = dose_order(doses)
    long, tests = [], []
    for j in features:
        x = pooled[:, j] / pooled[:, j].max() if pooled[:, j].max() > 0 else pooled[:, j]
        values = {d: x[(doses == d).to_numpy()] for d in order}
        long.append(pd.DataFrame({"feature": j, "cell": cells if cells is not None else np.arange(len(x)),
                                  "dose": doses.to_numpy(), "activation": x}))
        p_kw = kruskal(*values.values()).pvalue
        pairs = list(zip(order[:-1], order[1:]))
        p_raw = [mannwhitneyu(values[a], values[b], alternative="two-sided").pvalue for a, b in pairs]
        p_adj = multipletests(p_raw, method="fdr_bh")[1]
        tests += [dict(feature=j, dose_low=a, dose_high=b, p_raw=p, p_adj=q, kruskal_p=p_kw,
                       median_low=np.median(values[a]), median_high=np.median(values[b]))
                  for (a, b), p, q in zip(pairs, p_raw, p_adj)]
    return pd.concat(long, ignore_index=True), pd.DataFrame(tests)


# ---------------------------------------------------------------------------- main

@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="score_perturbation", version_base="1.3")
def main(cfg) -> None:
    run_dir = REPO_ROOT / cfg.steer_run
    obs, steer_cfg, original_cfg = run_cells(run_dir)
    drugs = obs[cfg.columns.drug].astype(str)
    drug = cfg.get("drug") or next(iter(sorted(set(drugs) - {cfg.control})))
    is_control, is_treated = (drugs == cfg.control).to_numpy(), (drugs == drug).to_numpy()
    lines = obs[cfg.columns.cell_line].astype(str).to_numpy()
    heldout = set(cfg.heldout_cell_lines)
    split = {line: "heldout" if line in heldout else "train" for line in np.unique(lines)}

    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) / drug.lower()
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_err" if steer_cfg.steering.get("add_reconstruction_error", False) else ""

    original = np.load(run_dir / "original.npy")
    scaler = StandardScaler().fit(original)
    original = scaler.transform(original)
    k = int(cfg.knn)

    def frame(rows, **extra):
        df = pd.DataFrame(rows).assign(drug=drug, **extra)
        df["split"] = df["cell_line"].map(split).fillna("all")
        return df

    baseline = pd.concat([
        frame(score_groups(original[is_control], lines[is_control], original, lines, is_control, is_treated, k),
              query="control"),
        frame(score_groups(original[is_treated], lines[is_treated], original, lines, is_control, is_treated, k),
              query="treated"),
    ], ignore_index=True)
    baseline.to_csv(save_dir / "baseline_scores.csv", index=False)

    sweep, steered_control = [], {}
    for (f, v), path in steered_files(run_dir, suffix).items():
        steered = scaler.transform(np.load(path))
        steered_control[(f, v)] = steered[is_control]
        sweep.append(frame(score_groups(steered[is_control], lines[is_control], original, lines,
                                        is_control, is_treated, k), f=f, v=v))
    sweep = pd.concat(sweep, ignore_index=True)
    sweep.to_csv(save_dir / "sweep_scores.csv", index=False)

    w = cfg.selection.weights
    train = sweep[sweep["split"] == "train"].copy()
    train["objective"] = w.cms * (1 - train["CMS"]).abs() + w.das * (1 - train["DAS"]).abs()
    selection = train.groupby(["f", "v"], as_index=False)["objective"].mean()
    best = selection.loc[selection["objective"].idxmin()]
    selection["selected"] = (selection["f"] == best["f"]) & (selection["v"] == best["v"])
    selection.assign(drug=drug).to_csv(save_dir / "selection.csv", index=False)
    print(f"[score_perturbation] {drug}: selected f={best['f']}, v={best['v']} on {len(set(train['cell_line']))} "
          f"training lines; held-out lines: {sorted(heldout & set(lines))}")

    if cfg.dose.get("column"):
        if cfg.dose.get("run"):  # a run over all doses, standardized on its own original cells
            dose_dir = REPO_ROOT / cfg.dose.run
            d_obs, d_cfg, _ = run_cells(dose_dir)
            d_original = np.load(dose_dir / "original.npy")
            d_scaler = StandardScaler().fit(d_original)
            d_drugs = d_obs[cfg.columns.drug].astype(str)
            d_control, d_treated = (d_drugs == cfg.control).to_numpy(), (d_drugs == drug).to_numpy()
            d_suffix = "_err" if d_cfg.steering.get("add_reconstruction_error", False) else ""
            d_steered = {fv: d_scaler.transform(np.load(p))[d_control] for fv, p in steered_files(dose_dir, d_suffix).items()}
            trajectory = dose_trajectory(d_scaler.transform(d_original), d_obs[cfg.dose.column], d_control, d_treated,
                                         d_steered, best["v"])
        else:
            trajectory = dose_trajectory(original, obs[cfg.dose.column], is_control, is_treated, steered_control, best["v"])
        trajectory.assign(drug=drug).to_csv(save_dir / "dose_das.csv", index=False)

    da = cfg.get("dose_activation", {})
    if da.get("features"):
        name = f"features-{steer_cfg.sae_checkpoint.timestamp}{da.get('tag', '')}"
        folder = REPO_ROOT / "cached_features" / steer_cfg.sae_checkpoint.experiment
        pooled = torch.load(folder / f"{name}.pt", map_location="cpu").float().numpy()
        cell_idx = np.load(folder / f"{name}.cell_idx.npy") if (folder / f"{name}.cell_idx.npy").exists() else np.arange(len(pooled))
        keep = cell_idx >= 0
        if da.get("cells"):  # cells.csv of a run on the features' data (`cell`: position in that data)
            cells = pd.read_csv(REPO_ROOT / da.cells / "cells.csv").set_index("cell").loc[cell_idx[keep]]
        else:
            adata = load_adata(original_cfg, sub_sample=original_cfg.data.preprocess.get("split", 1.0))
            cells = adata.obs.iloc[cell_idx[keep]]
        in_drug = cells[cfg.columns.drug].astype(str).isin([drug, cfg.control]).to_numpy()
        doses = cells[cfg.dose.column].reset_index(drop=True)[in_drug]
        long, tests = dose_activation(pooled[keep][in_drug], doses.reset_index(drop=True), list(da.features),
                                      cells=cell_idx[keep][in_drug])
        long.assign(drug=drug).to_csv(save_dir / "dose_activation.csv", index=False)
        tests.assign(drug=drug).to_csv(save_dir / "dose_activation_tests.csv", index=False)
    print(f"[score_perturbation] outputs -> {save_dir}")


if __name__ == "__main__":
    main()
