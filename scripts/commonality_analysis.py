"""
Cell-level variance partition (commonality analysis) of SAE features against cell metadata.

For each feature's max-pooled cell activation (generate_features output.format=pooled), the
fraction of variance explained by the biological design (cell type x tissue), the technical
design (dataset) and their full interaction is split into unique_bio, unique_tech and shared
components and summarised by B and S (Methods 4.2); features with R2_full >= min_r2_full are
cell-informative. Cells come from datasets with more than two cell types.

S is a commonality index (S < 0: the designs are redundant, S > 0: complementary), not an
interaction test: with biology and study confounded, additive effects of opposite sign also give
S > 0 (suppression). The main-effects model bio + dataset (R2_additive) splits it as
S = S_int + S_add, where S_int is the share of the explained variance that needs the
bio x dataset interaction and S_add the commonality of the main effects alone (< 0 redundancy,
> 0 suppression).

Outputs (experiments/commonality/<experiment>/<timestamp>/):

``variance_3way_full.csv``      per-feature partition, B, S and its split, informative and
                                interaction flags, one-vs-rest AUROC selectivities and a verdict
                                (biological / technical / confounded / unstructured / dead)
``variance_3way_balanced.csv``  the same on a subsample capped per dataset x cell type combination
``permutation_nulls.csv``       per null and seed, the statistics of Supplementary Table S8 over
                                `nulls.n_features` random features
``table_s8.csv``                their mean and standard deviation over seeds
``cell_means.csv``              for `analysis.report_features` only: activation per cell type x
                                tissue x dataset
"""

import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import norm, rankdata
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.pipeline import register_resolvers

register_resolvers()


# ============================================================================
# Variance-partition helpers (memory-safe, vectorised over a chunk of features)
# ============================================================================


def dense_codes(series):
    """Contiguous integer codes over OBSERVED levels only (avoids empty groups)."""
    c, _ = pd.factorize(series.astype(str))
    return c.astype(np.int64), int(c.max() + 1)


def onehot(codes, K):
    n = len(codes)
    return sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), codes)), shape=(n, K))


def rank_int(Y, return_ranks=False):
    """Per-column rank-based inverse-normal transform.

    Y: (n, c) -> (n, c) float32. If ``return_ranks``, also return the average ranks
    (which can be reused directly for AUROC selectivity without a second sort).
    """
    n = Y.shape[0]
    out = np.empty_like(Y, dtype=np.float32)
    R = np.empty_like(Y, dtype=np.float32) if return_ranks else None
    for j in range(Y.shape[1]):
        r = rankdata(Y[:, j], method="average")
        out[:, j] = norm.ppf((r - 0.5) / n).astype(np.float32)
        if return_ranks:
            R[:, j] = r
    return (out, R) if return_ranks else out


def r2_categorical(Y, O, counts):
    """Variance explained (SS_between / SS_total) by a categorical one-hot design.

    Y: (n, c) float32; O: (n, K) csr one-hot; counts: (K,). Returns (c,).
    """
    grand = Y.mean(0)
    gmean = (O.T @ Y) / counts[:, None]
    ss_b = (counts[:, None] * (gmean - grand) ** 2).sum(0)
    ss_t = ((Y - grand) ** 2).sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):  # constant (dead) features: NaN
        return np.asarray(ss_b / ss_t).ravel()


def additive_design(Ob, Od):
    """Main-effects design [Ob | Od] of two one-hots, a generalised inverse of its Gram matrix
    and its rank.

    The one-hots share the intercept, and more columns are dependent when the bio x dataset graph
    of observed combinations is disconnected; fitted values are unique all the same. The
    pseudo-inverse is taken on the count-normalised Gram matrix, whose eigenvalues lie in [0, 2].
    """
    X = sp.hstack([Ob, Od], format="csr")
    gram = (X.T @ X).toarray().astype(np.float64)
    scale = 1.0 / np.sqrt(np.diag(gram))
    w, V = np.linalg.eigh(gram * scale[:, None] * scale[None, :])
    keep = w > 1e-10 * w.max()
    G = scale[:, None] * ((V[:, keep] / w[keep]) @ V[:, keep].T) * scale[None, :]
    return X, G, int(keep.sum())


def _additive_coef(Y, X, G):
    """Centred least-squares fit of Y on X: (X^T Yc, coefficients), float64."""
    XtY = np.asarray(X.T @ (Y - Y.mean(0)), dtype=np.float64)
    return XtY, G @ XtY


def r2_additive(Y, X, G):
    """Variance explained by the main-effects model of ``additive_design``.

    Y: (n, c) float32. Returns (c,).
    """
    XtY, coef = _additive_coef(Y, X, G)
    ss_fit = (coef * XtY).sum(0)
    ss_t = ((Y - Y.mean(0)) ** 2).sum(0, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):  # constant (dead) features: NaN
        return ss_fit / ss_t


def additive_fitted(Y, X, G):
    """Fitted values of the main-effects model of ``additive_design``. Y: (n, c) float32."""
    _, coef = _additive_coef(Y, X, G)
    return np.asarray(X @ coef.astype(np.float32)) + Y.mean(0)


def selectivity(ranks, O, counts):
    """One-vs-rest selectivity per feature = best single-category AUROC (Mann-Whitney).

    ranks: (n, f) average ranks; O: (n, K) csr one-hot; counts: (K,). Returns (f,) in [0,1].
    AUROC_k = P(activation in category k > activation outside k). Size-INVARIANT so it
    captures "this feature cleanly marks ONE category" regardless of category frequency.
    """
    n = ranks.shape[0]
    rank_sum = np.asarray(O.T @ ranks)
    U = rank_sum - counts[:, None] * (counts[:, None] + 1) / 2.0
    auroc = U / (counts[:, None] * (n - counts[:, None]) + 1e-9)
    return auroc.max(0)


def classify(
    R2_full,
    unique_bio,
    unique_tech,
    shared,
    frac_active,
    sel_bio=None,
    sel_tech=None,
    min_explained=0.10,
    min_component=0.10,
    min_active=1e-3,
    min_selectivity=0.90,
):
    """Assign each feature to dead / unstructured / biological / technical / confounded.

    Two-gate logic (both must pass before a feature is attributed):
      (a) VARIANCE: the winning commonality component must exceed ``min_component``.
      (b) SELECTIVITY: the matching factor's best one-vs-rest AUROC must exceed
          ``min_selectivity`` (this is what makes a feature *recognisable*).
    """
    R2_full = np.asarray(R2_full)
    ub = np.asarray(unique_bio)
    ut = np.asarray(unique_tech)
    sh = np.asarray(shared)
    fa = np.asarray(frac_active)
    n = len(R2_full)
    sb = np.full(n, np.inf) if sel_bio is None else np.asarray(sel_bio)
    st = np.full(n, np.inf) if sel_tech is None else np.asarray(sel_tech)
    sel_any = np.maximum(sb, st)
    v = np.full(n, "unstructured", dtype=object)
    structured = R2_full >= min_explained
    is_bio = (ub >= ut) & (ub >= sh) & (ub >= min_component) & (sb >= min_selectivity)
    is_tech = (ut > ub) & (ut >= sh) & (ut >= min_component) & (st >= min_selectivity)
    is_conf = (sh > ub) & (sh > ut) & (sh >= min_component) & (sel_any >= min_selectivity)
    v[structured & is_bio] = "biological"
    v[structured & is_tech] = "technical"
    v[structured & is_conf] = "confounded"
    v[fa < min_active] = "dead"
    return v


# ============================================================================
# 3-way partition
# ============================================================================


def build_designs3(meta):
    """One-hots for cell_type, tissue, dataset and their pairwise / full joins.

    ``bio`` is cell_type x tissue (the full biological side), ``full`` is
    cell_type x tissue x dataset (used as the joint design).
    """
    ct, Kct = dense_codes(meta["cell_type"])
    ds, Kds = dense_codes(meta["dataset"])
    ti, Kti = dense_codes(meta["tissue"])

    def join(*cks):
        key = np.zeros(len(meta), np.int64)
        mult = 1
        for c, K in cks:
            key = key + c * mult
            mult *= K
        j = pd.factorize(key)[0]
        return onehot(j, int(j.max() + 1))

    D = {
        "ct": onehot(ct, Kct),
        "ds": onehot(ds, Kds),
        "ti": onehot(ti, Kti),
        "cd": join((ct, Kct), (ds, Kds)),
        "bio": join((ct, Kct), (ti, Kti)),
        "dt": join((ds, Kds), (ti, Kti)),
        "full": join((ct, Kct), (ds, Kds), (ti, Kti)),
    }
    C = {k: np.asarray(v.sum(0)).ravel() for k, v in D.items()}
    return D, C


def _apply_transform(Y: np.ndarray, kind: str):
    """Apply per-column activation transform, returning (Y_transformed, ranks_or_None).

    ``kind`` in {"int", "log1p", "rank", "none"}. Ranks are returned (for reuse in AUROC
    selectivity) whenever they were computed anyway, i.e. for "int" and "rank".
    """
    kind = str(kind).lower()
    n = Y.shape[0]
    if kind == "int":
        return rank_int(Y, return_ranks=True)
    if kind == "log1p":
        return np.log1p(np.clip(Y, 0.0, None)).astype(np.float32, copy=False), None
    if kind == "rank":
        R = np.empty_like(Y, dtype=np.float32)
        for j in range(Y.shape[1]):
            R[:, j] = rankdata(Y[:, j], method="average")
        return (R / n).astype(np.float32), R
    if kind in ("none", "raw", "false"):
        return Y.astype(np.float32, copy=False), None
    raise ValueError(
        f"Unknown transform {kind!r}; expected one of 'int', 'log1p', 'rank', 'none'."
    )


def _compute_ranks(Y: np.ndarray) -> np.ndarray:
    """Per-column average ranks (for AUROC selectivity when the transform did not need them)."""
    R = np.empty_like(Y, dtype=np.float32)
    for j in range(Y.shape[1]):
        R[:, j] = rankdata(Y[:, j], method="average")
    return R


def partition_3way(acts, meta, chunk=256, transform="int"):
    """3-way commonality partition.

    ``transform`` controls the per-feature activation transform applied before r2:
      * ``"int"``   -- rank-based inverse-normal (default, matches the notebook)
      * ``"log1p"`` -- ``log1p(max(x, 0))`` (magnitude-preserving)
      * ``"rank"``  -- normalised average ranks in [1/n, 1]
      * ``"none"``  -- raw activations
    AUROC selectivity is rank-based and therefore independent of the transform choice.
    """
    D, C = build_designs3(meta)
    X_add, G_add, rank_add = additive_design(D["bio"], D["ds"])
    F = acts.shape[1]
    keys = [
        "R2_celltype", "R2_dataset", "R2_tissue", "R2_bio", "R2_full",
        "bio_celltype", "tissue_within",
        "unique_bio", "unique_tech", "shared", "ambiguous_TD",
        "frac_active", "sel_celltype", "sel_tissue", "sel_dataset",
        "R2_additive", "interaction", "shared_additive",
    ]
    cols = {k: np.empty(F, np.float32) for k in keys}
    for s in tqdm(range(0, F, chunk)):
        e = min(s + chunk, F)
        Y = np.ascontiguousarray(acts[:, s:e].numpy().astype(np.float32))
        cols["frac_active"][s:e] = (Y != 0).mean(0)
        Y_t, R = _apply_transform(Y, transform)
        if R is None:
            # log1p / none: ranks of the raw values suffice (log1p is monotone).
            R = _compute_ranks(Y)
        Y = Y_t
        r_ct = r2_categorical(Y, D["ct"], C["ct"])
        r_ds = r2_categorical(Y, D["ds"], C["ds"])
        r_ti = r2_categorical(Y, D["ti"], C["ti"])
        r_cd = r2_categorical(Y, D["cd"], C["cd"])
        r_bio = r2_categorical(Y, D["bio"], C["bio"])
        r_dt = r2_categorical(Y, D["dt"], C["dt"])
        r_full = r2_categorical(Y, D["full"], C["full"])
        cols["R2_celltype"][s:e] = r_ct
        cols["R2_dataset"][s:e] = r_ds
        cols["R2_tissue"][s:e] = r_ti
        cols["R2_bio"][s:e] = r_bio
        cols["R2_full"][s:e] = r_full
        cols["bio_celltype"][s:e] = r_cd - r_ds
        cols["tissue_within"][s:e] = r_dt - r_ds
        cols["unique_bio"][s:e] = r_full - r_ds
        cols["unique_tech"][s:e] = r_full - r_bio
        cols["shared"][s:e] = r_bio + r_ds - r_full
        cols["ambiguous_TD"][s:e] = r_ti + r_ds - r_dt
        r_add = r2_additive(Y, X_add, G_add)
        cols["R2_additive"][s:e] = r_add
        cols["interaction"][s:e] = r_full - r_add
        cols["shared_additive"][s:e] = r_bio + r_ds - r_add
        Rsel = R if R is not None else Y
        cols["sel_celltype"][s:e] = selectivity(Rsel, D["ct"], C["ct"])
        cols["sel_tissue"][s:e] = selectivity(Rsel, D["ti"], C["ti"])
        cols["sel_dataset"][s:e] = selectivity(Rsel, D["ds"], C["ds"])
        del Y, R
    res = pd.DataFrame(cols)
    res["residual_unexplained"] = 1.0 - res["R2_full"]
    res["sel_bio"] = res[["sel_celltype", "sel_tissue"]].max(1)
    res["sel_tech"] = res["sel_dataset"]
    # Position in the unit l1 ball |B| + |S| <= 1 (Fig. 4a): B > 0 when the biological design
    # explains more than the technical one, S > 0 when the two designs are complementary rather
    # than redundant. S = S_int + S_add: the interaction share of R2_full (>= 0) plus the
    # commonality of the main effects (< 0 redundancy, > 0 suppression).
    r2 = res["R2_full"].where(res["R2_full"] > 0)
    res["B"] = (res["R2_bio"] - res["R2_dataset"]) / r2
    res["S"] = (res["R2_full"] - res["R2_bio"] - res["R2_dataset"]) / r2
    res["S_int"] = res["interaction"] / r2
    res["S_add"] = -res["shared_additive"] / r2
    res.attrs["interaction_df"] = D["full"].shape[1] - rank_add
    return res


def balanced_index(meta, per_combo=300, seed=0):
    """Cap each (dataset x cell_type) combo at ``per_combo`` cells (near-orthogonal design)."""
    rng = np.random.default_rng(seed)
    key = (meta["dataset"].astype(str) + "||" + meta["cell_type"].astype(str)).values
    pos = pd.Series(np.arange(len(meta)))
    keep = []
    for _, idx in pos.groupby(key):
        idx = idx.to_numpy()
        keep.append(idx if len(idx) <= per_combo else rng.choice(idx, per_combo, replace=False))
    return np.sort(np.concatenate(keep))


def cell_means(acts, meta, features):
    """Cell count, mean activation and fraction of active cells per cell type x tissue x dataset
    for each of `features` (long format), to read an interaction off the group means."""
    keys = ["cell_type", "tissue", "dataset"]
    Y = acts[:, list(features)].numpy()
    out = []
    for j, f in enumerate(features):
        g = meta[keys].assign(activation=Y[:, j], active=Y[:, j] != 0).groupby(keys, observed=True)
        out.append(g.agg(n_cells=("activation", "size"), mean=("activation", "mean"),
                         frac_active=("active", "mean")).reset_index().assign(feature=f))
    return pd.concat(out, ignore_index=True)


# ============================================================================
# Permutation nulls (Supplementary Note S3, Table S8)
# ============================================================================

NULLS = {
    # null: (columns shuffled, strata they are shuffled within); "residuals": the residuals of the
    # main-effects fit are shuffled instead of labels
    "global": (None, None),
    "within_study": (["cell_type", "tissue"], ["dataset"]),
    "within_celltype": (["dataset"], ["cell_type", "tissue"]),
    "additive": ("residuals", ["dataset"]),
}
# statistic reported per null: (component, quantile)
NULL_STATISTICS = {
    "global": [("R2_full", 0.99)],
    "within_study": [("unique_bio", 0.99), ("shared", 0.05)],
    "within_celltype": [("unique_tech", 0.99)],
    "additive": [("interaction", 0.99)],
}


def permutation_within(m, strata, rng):
    """Row permutation that moves rows only among the rows sharing their `strata` values."""
    perm = np.arange(len(m))
    for idx in m.groupby(strata, observed=True, sort=True).indices.values():
        perm[idx] = rng.permutation(idx)
    return perm


def null_partition(acts, meta, null, n_features=200, seed=0, transform="int"):
    """Commonality partition of `n_features` random features under a permutation null.

    global: label rows shuffled across all cells (null for the R2_full cutoff);
    within_study: cell type and tissue shuffled among the cells of each dataset (null for
    unique_bio); within_celltype: dataset shuffled among the cells of each cell type x tissue
    (null for unique_tech); additive: labels kept and the residuals of the main-effects fit
    bio + dataset shuffled among the cells of each dataset (Freedman-Lane with studies as
    exchangeability blocks), which keeps both main effects, their confounding (hence any
    suppression) and each study's residual variance, but no interaction (null for interaction).
    """
    rng = np.random.default_rng(seed)
    feat = np.sort(rng.choice(acts.shape[1], min(n_features, acts.shape[1]), replace=False))
    m = meta.reset_index(drop=True).copy()
    Y, _ = _apply_transform(np.ascontiguousarray(acts[:, feat].numpy().astype(np.float32)), transform)
    shuffled, strata = NULLS[null]
    if shuffled == "residuals":
        D, C = build_designs3(m)
        X_add, G_add, _ = additive_design(D["bio"], D["ds"])
        fitted = additive_fitted(Y, X_add, G_add)
        Y = fitted + (Y - fitted)[permutation_within(m, strata, rng)]
    else:
        if shuffled is None:
            m = m.iloc[rng.permutation(len(m))].reset_index(drop=True)
        else:
            m[shuffled] = m.loc[permutation_within(m, strata, rng), shuffled].to_numpy()
        D, C = build_designs3(m)
        X_add, G_add, _ = additive_design(D["bio"], D["ds"])
    r_full = r2_categorical(Y, D["full"], C["full"])
    r_bio = r2_categorical(Y, D["bio"], C["bio"])
    r_ds = r2_categorical(Y, D["ds"], C["ds"])
    r_add = r2_additive(Y, X_add, G_add)
    return dict(R2_full=r_full, unique_bio=r_full - r_ds, unique_tech=r_full - r_bio,
                shared=r_bio + r_ds - r_full, interaction=r_full - r_add)


# ============================================================================
# Inputs
# ============================================================================


def load_meta(labels_path: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_path)
    return pd.DataFrame({
        "cell_type": labels["celltype_name"].astype("category"),
        "dataset": labels["dataset"],
        "technology": labels["original_assay"].map(lambda x: "10x" if "10x" in x else x),
        "tissue": labels["tissue_general"],
    })


def load_cell_features(features_path: Path, meta: pd.DataFrame):
    """Pooled features and the metadata rows of their cells (aligned by the saved cell_idx
    where present; otherwise rows are taken to be in label order)."""
    acts = torch.load(features_path, map_location="cpu")
    cell_idx_path = features_path.with_name(features_path.stem + ".cell_idx.npy")
    if cell_idx_path.exists():
        cell_idx = np.load(cell_idx_path)
        rows = np.flatnonzero(cell_idx >= 0)
        return acts[torch.as_tensor(rows)], meta.iloc[cell_idx[rows]].reset_index(drop=True)
    if len(meta) != acts.shape[0]:
        raise ValueError(f"Row-count mismatch: labels={len(meta)} vs activations={acts.shape[0]}.")
    return acts, meta


def filter_by_dataset_diversity(meta: pd.DataFrame, min_celltypes: int = 3):
    """Keep datasets with at least `min_celltypes` observed cell types."""
    n_types = meta.groupby("dataset")["cell_type"].nunique()
    mask = meta["dataset"].isin(n_types.index[n_types >= min_celltypes]).to_numpy()
    return meta[mask].reset_index(drop=True), mask


# ============================================================================
# Main
# ============================================================================


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="commonality", version_base="1.3")
def main(cfg) -> None:
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    a, v = cfg.analysis, cfg.analysis.verdict

    acts, meta = load_cell_features(REPO_ROOT / cfg.features.path, load_meta(REPO_ROOT / cfg.labels.path))
    meta, mask = filter_by_dataset_diversity(meta, int(a.min_celltypes_per_dataset))
    acts = acts[torch.as_tensor(mask)]
    print(f"[commonality] {len(meta):,} cells, {acts.shape[1]} features", flush=True)

    def partition(acts_, meta_):
        res = partition_3way(acts_, meta_, chunk=int(a.chunk), transform=str(a.transform))
        res["informative"] = res["R2_full"] >= float(a.min_r2_full)
        res["verdict"] = classify(
            res["R2_full"], res["unique_bio"], res["unique_tech"], res["shared"], res["frac_active"],
            sel_bio=res["sel_bio"], sel_tech=res["sel_tech"], min_explained=float(v.min_explained),
            min_component=float(v.min_component), min_active=float(v.min_active),
            min_selectivity=float(v.min_selectivity))
        res["interaction_feature"] = res["interaction"] >= float(a.min_interaction)
        return res

    res_full = partition(acts, meta)
    res_full.to_csv(save_dir / "variance_3way_full.csv", index=False)
    inform = res_full["informative"]
    print(f"[commonality] informative (R2_full >= {a.min_r2_full}): {inform.mean():.3f}; of these, "
          f"interaction >= {a.min_interaction}: {res_full.loc[inform, 'interaction_feature'].mean():.3f} "
          f"(interaction df {res_full.attrs['interaction_df']})", flush=True)
    if a.report_features:
        cell_means(acts, meta, [int(f) for f in a.report_features]).to_csv(save_dir / "cell_means.csv", index=False)

    bal = balanced_index(meta, per_combo=int(a.balanced_per_combo), seed=int(cfg.seed))
    res_bal = partition(acts[torch.as_tensor(bal)], meta.iloc[bal].reset_index(drop=True))
    res_bal.to_csv(save_dir / "variance_3way_balanced.csv", index=False)
    agree = (res_full["verdict"] == res_bal["verdict"]).mean()
    print(f"[commonality] balanced subset: {len(bal):,} cells; verdict agreement {agree:.3f}", flush=True)

    n = cfg.nulls
    rows = []
    for seed in range(int(n.seeds)):
        for null, statistics in NULL_STATISTICS.items():
            d = null_partition(acts, meta, null, n_features=int(n.n_features), seed=seed, transform=str(a.transform))
            for component, q in statistics:
                rows.append(dict(null=null, seed=seed, statistic=f"P{round(q * 100):02d} {component}",
                                 value=float(np.nanquantile(d[component], q))))  # NaN: constant features
    nulls = pd.DataFrame(rows)
    nulls.to_csv(save_dir / "permutation_nulls.csv", index=False)
    table = nulls.groupby(["null", "statistic"], sort=False)["value"].agg(mean="mean", std=lambda x: x.std(ddof=0))
    table.reset_index().to_csv(save_dir / "table_s8.csv", index=False)
    print(table)


if __name__ == "__main__":
    main()
