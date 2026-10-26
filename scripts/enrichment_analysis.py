"""
Pathway enrichment of SAE features (preranked GSEA) and collapsing of redundant terms.

For every feature with activation (or those in ``analysis.features``), genes are ranked by the
log (``analysis.transform=log``) or the raw value (``raw``) of the feature's frequency-normalised
profile (``act_sum / gene_occurrence`` from gene_selectivity) and tested with blitzgsea against
MSigDB Hallmark, GO (BP; optionally CC and MF) and C2 canonical pathways. The gene universe is
the genes observed as input tokens; gene sets are restricted to it and kept when they have
``min_size`` to ``max_size`` genes.

Outputs (experiments/enrichment/<experiment>/<timestamp><run_tag>/; run_tag is "_raw" for
``analysis.transform=raw``, plus "_seed<n>" with ``analysis.seed`` other than 42 and "_selected" with
``analysis.features``):

``pathway_enrichment_long.parquet``
    One row per (feature, term) with ES > 0 and p < ``pval_keep``: feature_id, Term, es, nes,
    pval, sidak, fdr (blitzgsea's BH q within the feature), geneset_size, leading_edge
    (comma-separated genes), leading_edge_size (genes) and global_q (BH q over all tests of all
    features; with ``analysis.features``, the other features' p-values come from the reference
    run in ``inputs.reference``, so global_q matches a run over all features; with another seed,
    the tested features' q-values are ranked among the reference run's other p-values).
``term_clusters.csv``
    DAVID functional-annotation clusters of the tested terms (cluster, Term); a term can belong
    to several clusters, and terms that join no cluster form their own.
``enrichment_collapsed.csv``
    Per feature, one row per cluster with at least one enriched term (global_q < ``q_max``,
    at least ``min_leading_edge`` leading-edge genes), skipping features whose top-20 gene mass
    is ``max_topk20_mass`` or more. The representative is the cluster's most significant enriched
    term; leading_edge_union counts distinct leading-edge genes over the cluster's enriched terms.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import blitzgsea as blitz
import hydra
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.gene_stats import load_table, profile, topk_mass
from sae4scfm.core import vocab
from sae4scfm.core.io_utils import load_sae_checkpoint
from sae4scfm.core.pipeline import checkpoint_dir, register_resolvers

register_resolvers()
# A run on analysis.features writes next to its reference run over all features, not over it
OmegaConf.register_new_resolver("selected_tag", lambda features: "_selected" if features else "", replace=True)
# Runs with another score transform than the default log write to their own folder
OmegaConf.register_new_resolver("transform_tag", lambda transform: "" if transform == "log" else f"_{transform}",
                                replace=True)
# So do runs with another null seed (enrichment_selection checks that q holds across seeds)
OmegaConf.register_new_resolver("seed_tag", lambda seed: "" if int(seed) == 42 else f"_seed{int(seed)}", replace=True)

TRANSFORMS = ("log", "raw")


def _patch_blitzgsea():
    """Make blitzgsea's null estimation reproducible (idempotent; called in every worker).

    blitzgsea 1.3.x draws each calibration anchor's null from np.random but seeds only Python's
    `random` per anchor, so with a process pool the null (and the p-values) depends on which
    worker runs which anchor. Its single-process path omits the `ks_disable` argument. Seed
    NumPy per anchor with the seed blitzgsea already derives, and default the argument.
    """
    original = blitz.estimate_anchor
    if getattr(original, "_seeded", False):
        return

    def seeded_estimate_anchor(signature, abs_signature, signature_map, set_size, permutations,
                               symmetric, seed, ks_disable=False):
        np.random.seed(int(seed) % 2**32)
        return original(signature, abs_signature, signature_map, set_size, permutations,
                        symmetric, seed, ks_disable)

    seeded_estimate_anchor._seeded = True
    blitz.estimate_anchor = seeded_estimate_anchor


# ---------------------------------------------------------------------------- inputs

def load_libraries(hallmark=True, bp=True, cp=True, cc=False, mf=False) -> dict[str, list[str]]:
    """{MSigDB term: [gene symbols]} for the selected collections: Hallmark, GO biological
    process / cellular component / molecular function (C5), and C2 canonical pathways."""
    def read(name):
        with open(REPO_ROOT / "data" / "gene_sets" / f"{name}.json") as f:
            return json.load(f)

    library = {}
    if hallmark:
        library.update({t: s["geneSymbols"] for t, s in read("h").items()})
    go_prefixes = tuple(p for p, on in (("GOBP_", bp), ("GOCC_", cc), ("GOMF_", mf)) if on)
    if go_prefixes:
        library.update({t: s["geneSymbols"] for t, s in read("c5").items() if t.startswith(go_prefixes)})
    if cp:
        library.update({t: s["geneSymbols"] for t, s in read("c2").items() if s["collection"].startswith("C2:CP")})
    return library


def filter_library_to_vocab(library: dict[str, list[str]], vocab_symbols: set[str], min_size: int,
                            max_size: int) -> dict[str, list[str]]:
    """Restrict each gene set to the universe, then apply the size cutoffs."""
    out = {}
    for term, genes in library.items():
        overlap = [g for g in genes if g in vocab_symbols]
        if min_size <= len(overlap) <= max_size:
            out[term] = overlap
    return out


def base_model(cfg) -> str:
    """scFM of the SAE: ``inputs.base_model``, or the training run's (which needs its checkpoint folder)."""
    name = cfg.inputs.get("base_model")
    if name:
        return str(name)
    _, original_cfg = load_sae_checkpoint(checkpoint_dir(cfg, REPO_ROOT), "cpu", load_sae=False)
    return original_cfg.scfm.base_model


def universe_and_library(symbols: list, gene_occ: np.ndarray, libraries, min_size: int, max_size: int):
    """The gene universe (genes with a symbol that occurred as input tokens) and the gene-set library
    restricted to it and size-filtered, as tested by a run."""
    universe = np.array([bool(s) for s in symbols]) & (gene_occ > 0)
    library = load_libraries(**{k: bool(libraries.get(k, False)) for k in ("hallmark", "bp", "cp", "cc", "mf")})
    library = filter_library_to_vocab(library, {s for s, keep in zip(symbols, universe) if keep},
                                      min_size=min_size, max_size=max_size)
    return universe, library


def select_features(active: np.ndarray, features) -> np.ndarray:
    """Sorted ids of the features to test: those in `features` (all when empty) that have activation."""
    if not features:
        return np.flatnonzero(active)
    requested = sorted({int(f) for f in features})
    out_of_range = [f for f in requested if not 0 <= f < len(active)]
    if out_of_range:
        raise ValueError(f"analysis.features outside 0..{len(active) - 1}: {out_of_range}")
    silent = [f for f in requested if not active[f]]
    if silent:
        print(f"[enrich] skipping features without activation: {silent}")
    return np.array([f for f in requested if active[f]], dtype=int)


# ---------------------------------------------------------------------------- GSEA

def make_signature(
    profile_row: np.ndarray,
    gene_symbols: list[str],
    valid_gene_mask: np.ndarray,
    rng: np.random.Generator,
    transform: str = "log",
) -> pd.DataFrame:
    """Build the blitzgsea signature (gene, score) for one feature.

    blitzgsea ranks genes by score and weights each hit by the distance of its score from the
    mean score. `transform` sets the score:

    - ``log``: the log of the frequency-normalised profile score. The raw scores are heavy-tailed,
      so on the raw scale a handful of genes would dominate every test. Zero-score genes sit just
      below the smallest log score.
    - ``raw``: the profile score itself. Zero-score genes get values
      between 0 and the smallest nonzero score.

    Either way the zero tail is in random order (uniform jitter), not a block of ties. Genes
    outside `valid_gene_mask` (no HGNC symbol, or never observed) are dropped.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f"analysis.transform must be one of {TRANSFORMS}, got {transform!r}")
    scores = profile_row[valid_gene_mask].astype(np.float64)
    nonzero = scores > 0
    n_zero = int((~nonzero).sum())
    if transform == "log":
        out = np.empty_like(scores)
        out[nonzero] = np.log(scores[nonzero])
        floor = out[nonzero].min() - 1.0 if nonzero.any() else 0.0
        out[~nonzero] = floor - rng.uniform(0.0, 1e-6, size=n_zero)
    else:
        out = scores.copy()
        smallest = scores[nonzero].min() if nonzero.any() else 1.0
        out[~nonzero] = rng.uniform(0.0, smallest, size=n_zero)
    genes = [g for g, keep in zip(gene_symbols, valid_gene_mask) if keep]
    return pd.DataFrame({0: genes, 1: out})


def _run_one_feature(
    fid: int,
    profile_row: np.ndarray,
    gene_symbols: list,
    valid_gene_mask: np.ndarray,
    library: dict,
    permutations: int,
    pval_keep: float,
    seed: int,
    transform: str = "log",
):
    """Prerank one feature; returns (fid, number of terms tested, rows with ES > 0 and p < pval_keep)."""
    _patch_blitzgsea()
    rng = np.random.default_rng(seed)
    sig = make_signature(profile_row, gene_symbols, valid_gene_mask, rng, transform)
    try:
        res = blitz.gsea(sig, library, permutations=permutations,
                         processes=1,  # joblib parallelizes across features
                         seed=int(rng.integers(0, 2**31 - 1)))
    except Exception as exc:
        print(f"  feature {fid} failed: {exc}", flush=True)
        return int(fid), 0, None
    if "Term" not in res.columns:
        res = res.reset_index().rename(columns={"index": "Term"})
    # blitzgsea assembles leading edges from sets, so their order varies between processes
    res["leading_edge"] = res["leading_edge"].map(lambda le: ",".join(sorted(leading_edge_genes(le))))
    n_test = len(res)
    res = res[(res["es"] > 0) & (res["pval"] < pval_keep)].copy()
    if res.empty:
        return int(fid), n_test, None
    res.insert(0, "feature_id", int(fid))
    return int(fid), n_test, res


def leading_edge_genes(leading_edge) -> list[str]:
    """Genes of a blitzgsea leading edge (comma-separated string; may be empty or missing)."""
    if not isinstance(leading_edge, str):
        return []
    return [g.strip() for g in leading_edge.split(",") if g.strip()]


def global_bh(pvals: np.ndarray, n_tests: int) -> np.ndarray:
    """BH q-values of the given p-values within `n_tests` tests in total.

    The p-values not passed (filtered out upstream) are all larger, so the ranks of the given
    ones in the full list are their ranks here.
    """
    order = np.argsort(pvals, kind="stable")
    raw = n_tests * np.clip(pvals[order], 1e-300, None) / np.arange(1, len(pvals) + 1)
    q = np.empty_like(raw)
    q[order] = np.minimum.accumulate(raw[::-1])[::-1]
    return np.clip(q, 0.0, 1.0)


def global_bh_with_others(pvals: np.ndarray, n_tests: int, other_pvals: np.ndarray,
                          n_other_tests: int) -> np.ndarray:
    """global_bh of some features' p-values as in a run over all features, given the other
    features' kept p-values and their number of tests."""
    return global_bh(np.concatenate([pvals, other_pvals]), n_tests + n_other_tests)[:len(pvals)]


# ---------------------------------------------------------------------------- term clusters

def david_clusters(term2genes: dict[str, list[str]], background_size: int, kappa: float = 0.35,
                   initial_group_size: int = 3, linkage_frac: float = 0.5,
                   merge_overlap: float = 0.5, block: int = 1024) -> list[list[str]]:
    """DAVID functional-annotation clustering of terms by the kappa agreement of their genes.

    Each term with at least `initial_group_size` kappa neighbours (itself included) whose pairs
    mostly agree (`linkage_frac`) seeds a group; groups sharing more than `merge_overlap` of the
    smaller group's terms are merged until nothing changes. Terms in no group become singleton
    clusters. Clusters are returned largest first.
    """
    terms = list(term2genes)
    n = len(terms)
    gene_index = {g: i for i, g in enumerate(sorted({g for genes in term2genes.values() for g in genes}))}
    rows = [i for i, t in enumerate(terms) for _ in term2genes[t]]
    cols = [gene_index[g] for t in terms for g in term2genes[t]]
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, len(gene_index)))
    shared = (X @ X.T).toarray()
    size = np.asarray(X.sum(axis=1)).ravel()
    N = float(background_size)

    agree = np.zeros((n, n), dtype=bool)
    for start in range(0, n, block):
        a = shared[start:start + block]
        si, sj = size[start:start + block, None], size[None, :]
        observed = (N - si - sj + 2 * a) / N  # (a + d) / N
        expected = (si * sj + (N - si) * (N - sj)) / N ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(expected < 1, (observed - expected) / (1 - expected), 0.0)
        agree[start:start + block] = k >= kappa
    np.fill_diagonal(agree, True)

    groups = []
    for i in range(n):
        members = np.flatnonzero(agree[i])
        m = len(members)
        if m < initial_group_size:
            continue
        pairs = (agree[np.ix_(members, members)].sum() - m) / 2
        if pairs / (m * (m - 1) / 2) >= linkage_frac:
            groups.append(set(members.tolist()))

    changed = True
    while changed:
        changed = False
        merged, holders = [], defaultdict(set)  # holders: term -> merged groups containing it
        for group in groups:
            for i in sorted(set().union(*(holders[t] for t in group))):
                if len(group & merged[i]) / min(len(group), len(merged[i])) > merge_overlap:
                    merged[i] |= group
                    for t in group:
                        holders[t].add(i)
                    changed = True
                    break
            else:
                merged.append(set(group))
                for t in group:
                    holders[t].add(len(merged) - 1)
        groups = merged

    clustered = set().union(*groups) if groups else set()
    clusters = [sorted(g) for g in sorted(groups, key=len, reverse=True)]
    clusters += [[i] for i in range(n) if i not in clustered]
    return [[terms[i] for i in c] for c in clusters]


def collapse_enrichment(enriched: pd.DataFrame, clusters: list[list[str]]) -> pd.DataFrame:
    """Per feature, one row per cluster containing at least one of its enriched terms."""
    term_clusters = defaultdict(list)
    for cid, members in enumerate(clusters):
        for term in members:
            term_clusters[term].append(cid)

    rows = []
    for fid, terms in enriched.groupby("feature_id", sort=True):
        terms = terms.sort_values(["global_q", "nes"], ascending=[True, False], kind="stable")
        by_cluster = defaultdict(list)
        for r in terms.to_dict("records"):
            for cid in term_clusters[r["Term"]]:
                by_cluster[cid].append(r)
        feature_rows = []
        for cid, members in by_cluster.items():
            rep = members[0]  # most significant enriched term
            union = sorted({g for r in members for g in leading_edge_genes(r["leading_edge"])})
            feature_rows.append({
                "feature_id": int(fid), "cluster": cid, "representative": rep["Term"],
                "es": rep["es"], "nes": rep["nes"], "pval": rep["pval"], "fdr": rep["fdr"],
                "global_q": rep["global_q"], "geneset_size": rep["geneset_size"],
                "leading_edge_size": rep["leading_edge_size"],
                "n_enriched_terms": len(members), "cluster_size": len(clusters[cid]),
                "best_fdr": min(r["fdr"] for r in members),
                "max_nes": max(r["nes"] for r in members),
                "leading_edge_union_size": len(union),
                "enriched_terms": ";".join(r["Term"] for r in members),
                "leading_edge_union": ";".join(union),
            })
        feature_rows.sort(key=lambda r: (r["global_q"], -r["nes"], -r["n_enriched_terms"]))
        seen = set()  # clusters overlap: report each representative once per feature
        for r in feature_rows:
            if r["representative"] not in seen:
                seen.add(r["representative"])
                rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- main

@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="enrichment", version_base="1.3")
def main(cfg) -> None:
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    accum_dir = Path(str(cfg.inputs.accum_dir))
    accum_dir = accum_dir if accum_dir.is_absolute() else REPO_ROOT / accum_dir
    a, c = cfg.analysis, cfg.collapse

    act_sum = load_table(accum_dir, "act_sum")
    gene_occ = np.load(accum_dir / "gene_occurrence.npy").astype(np.float64)
    denom = np.maximum(gene_occ, 1.0)
    symbols = vocab.gene_symbols(base_model(cfg))
    symbols = (symbols + [""] * act_sum.shape[1])[:act_sum.shape[1]]
    # The universe is the genes the SAE saw as input tokens; vocab genes that never occurred
    # (e.g. outside the HVG input set) would only add a block of tied zeros.
    universe, library = universe_and_library(symbols, gene_occ, cfg.libraries, int(a.min_size), int(a.max_size))
    print(f"[enrich] universe: {int(universe.sum())} observed genes with symbols (vocab {len(symbols)})")
    if not library:
        raise RuntimeError("No gene sets survived the size filter. Check the vocab symbol mapping.")
    active = np.asarray(act_sum.sum(axis=1)).ravel() > 0
    feature_ids = select_features(active, a.get("features"))
    reference = None  # with analysis.features: the kept rows of a run over all features
    if a.get("features"):
        ref_dir = Path(str(cfg.inputs.reference))
        ref_dir = ref_dir if ref_dir.is_absolute() else REPO_ROOT / ref_dir
        if ref_dir.resolve() == save_dir.resolve():
            raise ValueError(f"analysis.features would overwrite the reference run {ref_dir}; set run_tag")
        reference = pd.read_parquet(ref_dir / "pathway_enrichment_long.parquet",
                                    columns=["feature_id", "Term", "pval"])
        ref_cfg = ref_dir / ".hydra" / "config.yaml"
        ref_seed = int(OmegaConf.load(ref_cfg).analysis.get("seed", 42)) if ref_cfg.exists() else int(a.seed)
    transform = str(a.get("transform", "log"))
    if transform not in TRANSFORMS:
        raise ValueError(f"analysis.transform must be one of {TRANSFORMS}, got {transform!r}")
    print(f"[enrich] {len(library)} terms x {len(feature_ids)} features, {transform} scores "
          f"(n_jobs={a.n_jobs})", flush=True)

    # Workers inherit a fixed hash seed: blitzgsea iterates over sets of genes, so with per-process
    # seeds the p-values differ in the last bits from run to run.
    os.environ.setdefault("PYTHONHASHSEED", "0")
    results = Parallel(n_jobs=int(a.get("n_jobs", -1)), verbose=10, backend="loky")(
        delayed(_run_one_feature)(int(fid), act_sum[fid].toarray().ravel() / denom, symbols, universe,
                                  library, int(a.permutations), float(a.pval_keep), int(a.seed) + int(fid),
                                  transform)
        for fid in feature_ids)
    n_tests = sum(n for _, n, _ in results)
    kept = [res for _, _, res in results if res is not None]
    long_df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(
        columns=["feature_id", "Term", "es", "nes", "pval", "sidak", "fdr", "geneset_size", "leading_edge"])
    long_df["leading_edge_size"] = long_df["leading_edge"].map(lambda le: len(leading_edge_genes(le))).astype(int)
    pvals = long_df["pval"].to_numpy(dtype=np.float64)
    if reference is None:
        long_df["global_q"] = global_bh(pvals, n_tests)
    else:
        # Features are seeded by id, so with the same inputs and settings the tested features
        # reproduce their rows in the reference run. With another seed they differ by design.
        mine = reference[reference["feature_id"].isin(feature_ids)]
        both = mine.merge(long_df[["feature_id", "Term", "pval"]], on=["feature_id", "Term"], suffixes=("_ref", ""))
        if ref_seed != int(a.seed):
            print(f"[enrich] seed {int(a.seed)} (reference run: {ref_seed}): the tested features' q-values are "
                  "ranked among the reference run's other features", flush=True)
        elif not (len(both) == len(mine) == len(long_df) and np.allclose(both["pval"], both["pval_ref"], rtol=1e-9)):
            print("[enrich] WARNING: the tested features' results differ from the reference run's "
                  "(other accumulators or settings?); global_q mixes the two runs", flush=True)
        others = reference.loc[~reference["feature_id"].isin(feature_ids), "pval"].to_numpy(dtype=np.float64)
        # Gene sets are restricted to the universe, so every other active feature tested every term
        n_other_tests = len(library) * (int(active.sum()) - len(feature_ids))
        long_df["global_q"] = global_bh_with_others(pvals, n_tests, others, n_other_tests)
    long_df = long_df.sort_values("pval", ignore_index=True, kind="stable")
    long_df.to_parquet(save_dir / "pathway_enrichment_long.parquet")
    print(f"[enrich] kept {len(long_df):,} of {n_tests:,} tests (ES > 0, p < {a.pval_keep})")

    clusters = david_clusters(library, background_size=int(universe.sum()), kappa=c.kappa,
                              initial_group_size=c.initial_group_size, linkage_frac=c.linkage_frac,
                              merge_overlap=c.merge_overlap)
    pd.DataFrame([(cid, t) for cid, members in enumerate(clusters) for t in members],
                 columns=["cluster", "Term"]).to_csv(save_dir / "term_clusters.csv", index=False)

    topk20 = topk_mass(profile(act_sum, gene_occ), ks=(20,))[20]
    enriched = long_df[(long_df["global_q"] < c.q_max)
                       & (long_df["leading_edge_size"] >= c.min_leading_edge)
                       & (topk20[long_df["feature_id"].to_numpy(dtype=int)] < c.max_topk20_mass)]
    collapsed = collapse_enrichment(enriched, clusters)
    collapsed.to_csv(save_dir / "enrichment_collapsed.csv", index=False)
    print(f"[enrich] {len(clusters)} term clusters; {enriched['feature_id'].nunique()} enriched features, "
          f"{len(collapsed)} feature-cluster rows -> {save_dir}")


if __name__ == "__main__":
    main()
