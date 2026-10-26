"""
Aggregate scIB benchmark results into the normalized bio / batch / total scores of the paper.

Each summary in the config lists benchmark runs (experiments/benchmark/<comparison_id>/
benchmark_results.csv) per model and dataset. Every individual metric is min-max scaled with a
scaler fit per run (`per_run`), per dataset across models (`per_dataset`) or over everything
(`global`), optionally including the Unintegrated / scVI rows in the fit; bio conservation and
batch correction are the means of the scaled metrics in their groups, and
Total = w_bio * bio + w_batch * batch.

Outputs (experiments/summary/benchmark/):
``<summary>_scores.csv``   one row per embedding: model, dataset, run, kind (baseline / original /
                           recon / steered), baseline (Unintegrated / scVI), selector, n_features, seed, clamp, suffix,
                           the SAE (experiment, date) and its layer (1-based; target_layer is 0-based),
                           raw and normalized scores
``<summary>_summary.csv``  mean and standard deviation over seeds per model, dataset, kind,
                           selector, n_features, clamp, suffix and layer
"""
import os
import re
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.preprocessing import MinMaxScaler

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

BASELINES = ("Unintegrated", "scVI")
STEERED = re.compile(r"^(?P<date>.+?)_(?P<selector>[A-Za-z]+)_n(?P<n_features>\d+)_seed(?P<seed>\d+)"
                     r"_clamp(?P<clamp>-?\d+(?:\.\d+)?)(?P<suffix>.*)$")
GROUP = ["model", "dataset", "kind", "baseline", "selector", "n_features", "clamp", "suffix", "layer"]


def load_results(path: Path) -> pd.DataFrame:
    """A benchmark_results.csv without its "Metric Type" row, as floats."""
    df = pd.read_csv(path, index_col=0)
    return df.loc[df.index != "Metric Type"].astype(float)


def parse_embedding(name: str) -> dict:
    """Kind and steering parameters encoded in a benchmark row name."""
    if name.startswith(BASELINES):  # Unintegrated, scVI or scVI_<seed>
        return dict(kind="baseline", date=None, baseline=next(b for b in BASELINES if name.startswith(b)))
    if name.endswith("_original"):
        return dict(kind="original", date=name[: -len("_original")])
    m = re.match(r"^(?P<date>.+)_recon_seed(?P<seed>\d+)$", name)
    if m:
        return dict(kind="recon", date=m["date"], seed=int(m["seed"]))
    m = STEERED.match(name)
    if m is None:
        raise ValueError(f"Cannot parse benchmark row {name!r}")
    return dict(kind="steered", date=m["date"], selector=m["selector"], n_features=int(m["n_features"]),
                seed=int(m["seed"]), clamp=float(m["clamp"]), suffix=m["suffix"])


def sae_runs(bench_dir: Path) -> dict:
    """{SAE date: (experiment, target_layer)} for the SAEs a benchmark run evaluated (from its Hydra
    config; target_layer is None when the SAE's training config is not available)."""
    cfg_path = bench_dir / ".hydra" / "config.yaml"
    if not cfg_path.exists():
        return {}
    runs = {}
    for exp in OmegaConf.load(cfg_path).get("experiments", []):
        train_cfg = REPO_ROOT / "experiments" / "train" / str(exp.number) / str(exp.date) / ".hydra" / "config.yaml"
        layer = int(OmegaConf.load(train_cfg).sae.target_layer) if train_cfg.exists() else None
        runs[str(exp.date)] = (str(exp.number), layer)
    return runs


def load_summary_runs(spec, models=None) -> pd.DataFrame:
    """All rows of the runs listed in one summary (of `models`; default: all), with their parsed metadata."""
    frames = []
    for model, patterns in spec.runs.items():
        if models is not None and model not in models:
            continue
        for dataset in spec.datasets:
            for seed in spec.get("seeds", [None]):
                for pattern in patterns:
                    run = pattern.format(model=model, dataset=dataset, seed=seed)
                    bench_dir = REPO_ROOT / "experiments" / "benchmark" / run
                    raw = load_results(bench_dir / "benchmark_results.csv")
                    meta = pd.DataFrame([parse_embedding(i) for i in raw.index], index=raw.index)
                    saes = sae_runs(bench_dir)
                    meta["experiment"] = meta["date"].map(lambda d: saes.get(d, (None, None))[0])
                    meta["target_layer"] = meta["date"].map(lambda d: saes.get(d, (None, None))[1])
                    if seed is not None:  # seed given by the run (one run per seed)
                        meta["seed"] = meta.get("seed", pd.Series(index=meta.index, dtype=float)).fillna(seed)
                    frames.append(pd.concat([meta, raw], axis=1).assign(model=model, dataset=dataset, run=run)
                                  .rename_axis("embedding").reset_index())
    df = pd.concat(frames, ignore_index=True)
    for col in ("baseline", "selector", "suffix"):
        df[col] = df[col].fillna("") if col in df else ""
    for col in ("n_features", "seed", "clamp", "target_layer"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    df["layer"] = df["target_layer"] + 1
    return df


def fit_scalers(df: pd.DataFrame, metrics: list, normalization: str, include_baselines: bool) -> dict:
    """{(model, dataset): MinMaxScaler} fit on the rows of that key, its dataset, or all rows."""
    fit_rows = df if include_baselines else df[df["kind"] != "baseline"]
    keys = df[["model", "dataset"]].drop_duplicates().itertuples(index=False, name=None)
    scalers = {}
    for model, dataset in keys:
        if normalization == "per_run":
            rows = fit_rows[(fit_rows["model"] == model) & (fit_rows["dataset"] == dataset)]
        elif normalization == "per_dataset":
            rows = fit_rows[fit_rows["dataset"] == dataset]
        elif normalization == "global":
            rows = fit_rows
        else:
            raise ValueError(f"normalization must be per_run, per_dataset or global, got {normalization!r}")
        scalers[(model, dataset)] = MinMaxScaler().fit(rows[metrics])
    return scalers


def normalized_scores(df: pd.DataFrame, cfg, normalization: str, include_baselines: bool) -> pd.DataFrame:
    """Add the scaled `bio`, `batch` and `total` columns."""
    bio, batch = list(cfg.metrics.bio), list(cfg.metrics.batch)
    scalers = fit_scalers(df, bio + batch, normalization, include_baselines)
    out = []
    for (model, dataset), rows in df.groupby(["model", "dataset"], sort=False):
        scaled = pd.DataFrame(scalers[(model, dataset)].transform(rows[bio + batch]), columns=bio + batch, index=rows.index)
        rows = rows.assign(bio=scaled[bio].mean(axis=1), batch=scaled[batch].mean(axis=1))
        out.append(rows.assign(total=cfg.metrics.weights.bio * rows["bio"] + cfg.metrics.weights.batch * rows["batch"]))
    return pd.concat(out).sort_index()


def summarize(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation (ddof 1) over seeds and repeated rows."""
    g = scores.groupby(GROUP, dropna=False, sort=False)[["bio", "batch", "total"]]
    out = g.agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.assign(n=g.size()).reset_index()


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="summarize_benchmark", version_base="1.3")
def main(cfg) -> None:
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in cfg.summaries.items():
        df = load_summary_runs(spec, cfg.get("models"))
        if spec.get("kinds"):
            df = df[df["kind"].isin(spec.kinds)]
        scores = normalized_scores(df, cfg, spec.normalization, spec.include_baselines)
        scores.to_csv(save_dir / f"{name}_scores.csv", index=False)
        summary = summarize(scores)
        summary.to_csv(save_dir / f"{name}_summary.csv", index=False)
        print(f"[summarize_benchmark] {name}: {len(scores)} rows from {scores['run'].nunique()} runs")


if __name__ == "__main__":
    main()
