"""
Representative enriched features per gene program (Fig. 4e).

Reads an enrichment_analysis run over all features (the paper: ``analysis.transform=raw`` with GO
biological process, cellular component and molecular function), optional runs of the same SAE with
other null seeds (``analysis.seed=<n>``, over all features or a subset), and the accumulators the
run used. The programs and their columns come from ``config/programs/`` (``programs=fig4e``).

For each feature and panel term:

- GSEA statistics of the run (global q, leading edge), the effective size of the leading edge,
  (sum w)^2 / sum w^2 with w = |profile score - mean profile score| (blitzgsea's hit weights), and,
  for programs with a ``leading_edge_set``, the share of the leading edge in that set;
- the over-representation of the term among the feature's ``top_genes`` highest-scoring annotated
  genes (genes in a tested gene set; hypergeometric, BH over all features and panel terms);
- the median global q over the runs and whether the term is significant in every run (robust).

A feature qualifies for a program when one of the program's anchor terms has, in the run, global
q < ``q_max``, at least ``min_leading_edge`` leading-edge genes and as many effective ones, a
top-gene over-representation q < ``ora_q_max`` with at least ``ora_min_overlap`` genes and, when the
program has a ``leading_edge_set``, at least ``min_share`` of the leading edge in it; and the feature
is not concentrated (top-20 mass < ``max_topk20_mass``, at least ``min_effective_genes`` effective
genes). Specificity counts the panel columns where the feature has a dot (global q < ``q_max`` and
at least ``min_leading_edge`` leading-edge genes): tier 1 has at most ``single_program_max_off``
columns outside the program; tier 2 has the program among the feature's two strongest (by best q)
and at most ``two_program_max_off`` columns outside those two. Candidates rank by tier, then by
score = min(-log10 q, -log10 q_ORA) of their best qualifying term. A program's pick is its first
candidate with a robust anchor term: candidates that fail in a seed run are skipped, and one that
not every seed run tested stops the walk (status "needs seed runs", see seed_features.txt).

Outputs (experiments/enrichment_selection/<experiment>/<timestamp>/):

``panel_terms.parquet``
    One row per feature and panel term: program, column, anchor, es, nes, pval, global_q,
    leading_edge_size, leading_edge (when q < q_max), le_eff, leading_edge_share, ora_overlap,
    ora_q, q_<run> and le_<run> per run, n_runs (runs that tested the feature), q_median, robust
    (null when a seed run did not test the feature).
``candidates.csv``
    Qualifying features per program in rank order: tier, score, term, column, q, ora_q,
    programs (the feature's two strongest), off (columns with a dot outside the program, or outside
    the two programs in tier 2), robust, top leading-edge genes, topk20_mass, n_eff_genes.
``selection.csv``
    Per program: the pick, its status, the numbers of candidates (all, tier 1).
``seed_features.txt``
    Candidates that the seed runs did not test (the first ``seed_top_k`` per program), with the
    enrichment_analysis command for each seed.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from omegaconf import OmegaConf
from scipy.stats import hypergeom

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core import vocab
from sae4scfm.core.gene_stats import load_table, profile, topk_mass
from scripts.enrichment_analysis import base_model, global_bh, leading_edge_genes, universe_and_library

SEED_RUNS = (1042, 2042)  # suggested seeds for the stability runs
CANDIDATE_COLUMNS = ["program", "rank", "feature_id", "tier", "score", "term", "column", "q", "ora_q", "programs",
                     "off", "robust", "top_genes", "topk20_mass", "n_eff_genes"]


def resolve(path) -> Path:
    path = Path(str(path))
    return path if path.is_absolute() else REPO_ROOT / path


# ---------------------------------------------------------------------------- panel and statistics

def panel_terms(programs) -> pd.DataFrame:
    """One row per panel term: Term, program (code), column, anchor (whether it qualifies a feature)."""
    rows = []
    for p in programs:
        labels = [lab for lab, _ in p["columns"]]
        anchor = set(p.get("anchor") or labels)
        unknown = anchor - set(labels)
        if unknown:
            raise ValueError(f"program {p['code']}: anchor columns not among its columns: {sorted(unknown)}")
        for lab, terms in p["columns"]:
            rows += [dict(Term=t, program=p["code"], column=lab, anchor=lab in anchor) for t in terms]
    out = pd.DataFrame(rows)
    dup = out.Term[out.Term.duplicated()].unique()
    if len(dup):
        raise ValueError(f"terms in more than one column: {list(dup)}")
    return out


def effective_size(weights) -> float:
    """(sum w)^2 / sum w^2 of |w|: the number of equally weighted genes with the same spread."""
    w = np.abs(np.asarray(weights, dtype=np.float64))
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.sum() > 0 else 0.0


def top_gene_mask(prof: sp.csr_matrix, eligible: np.ndarray, n: int) -> sp.csr_matrix:
    """[F, G] boolean: each feature's n highest-scoring eligible genes with a positive score (all of
    them when it has fewer)."""
    rows, cols = [], []
    for f in range(prof.shape[0]):
        lo, hi = prof.indptr[f], prof.indptr[f + 1]
        idx, v = prof.indices[lo:hi], prof.data[lo:hi]
        keep = eligible[idx] & (v > 0)
        idx, v = idx[keep], v[keep]
        if len(v) > n:
            idx = idx[np.argpartition(-v, n - 1)[:n]]
        rows.append(np.full(len(idx), f)); cols.append(idx)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=prof.shape)


def over_representation(top: sp.csr_matrix, members: sp.csr_matrix, background: np.ndarray):
    """Overlap and hypergeometric p-value ([F, T]) of each term (rows of `members`, [T, G] boolean)
    among each feature's top genes (rows of `top`, [F, G] boolean), within the background genes."""
    bg = sp.diags(background.astype(np.float64))
    top = (top.astype(np.float64) @ bg).tocsr()
    members = (members.astype(np.float64) @ bg).tocsr()
    overlap = np.asarray((top @ members.T).todense()).astype(np.int64)
    n = np.asarray(top.sum(axis=1)).ravel().astype(np.int64)
    k = np.asarray(members.sum(axis=1)).ravel().astype(np.int64)
    p = hypergeom.sf(overlap - 1, int(background.sum()), k[None, :], n[:, None])
    p[n == 0] = 1.0
    return overlap, p


def tiers(dots: pd.DataFrame, program: str, single_max_off: int, two_max_off: int) -> pd.DataFrame:
    """Tier of each feature with a dot, for `program`, from its dots (feature_id, program, column, q).

    Tier 1: at most `single_max_off` columns with a dot outside the program. Tier 2: the program is
    one of the feature's two strongest (by best q) and at most `two_max_off` columns lie outside
    those two. Returns feature_id, tier (NaN: neither), off, programs (the two strongest, "+"-joined).
    """
    cols = dots.drop_duplicates(["feature_id", "column"])
    best = dots.groupby(["feature_id", "program"], as_index=False).q.min()
    best = best.sort_values(["feature_id", "q", "program"], kind="stable")
    best["rank"] = best.groupby("feature_id").cumcount()
    top2 = best[best["rank"] < 2]
    two = top2.groupby("feature_id").program.agg(lambda s: "+".join(sorted(s)))
    in_two = cols.merge(top2[["feature_id", "program"]].assign(_in=True), on=["feature_id", "program"], how="left")
    off1 = cols[cols.program != program].groupby("feature_id").size()
    off2 = in_two[in_two._in.isna()].groupby("feature_id").size()
    out = pd.DataFrame({"feature_id": sorted(cols.feature_id.unique())}).set_index("feature_id")
    out["off1"] = off1.reindex(out.index).fillna(0).astype(int)
    out["off2"] = off2.reindex(out.index).fillna(0).astype(int)
    out["programs"] = two.reindex(out.index)
    has = out.programs.fillna("").str.split("+").map(lambda ps: program in ps)
    out["tier"] = np.where(out.off1 <= single_max_off, 1, np.where(has & (out.off2 <= two_max_off), 2, np.nan))
    out["off"] = np.where(out.tier == 1, out.off1, out.off2)
    out.loc[out.tier == 1, "programs"] = program
    return out.reset_index()[["feature_id", "tier", "off", "programs"]]


def pick(candidates: pd.DataFrame):
    """(feature_id or None, status): the first robust candidate in rank order; a candidate with an
    unknown robustness (not tested by every seed run) stops the walk."""
    for r in candidates.itertuples():
        if pd.isna(r.robust):
            return None, f"needs seed runs (feature {r.feature_id})"
        if bool(r.robust):
            return int(r.feature_id), "robust"
    return None, "no robust candidate"


# ---------------------------------------------------------------------------- runs

def read_terms(run_dir: Path, terms, columns) -> pd.DataFrame:
    """Rows of a run's long table for the given terms (single-threaded read)."""
    return pq.read_table(run_dir / "pathway_enrichment_long.parquet", columns=columns,
                         filters=[("Term", "in", list(terms))], use_threads=False).to_pandas()


def tested_features(run_cfg, active: np.ndarray) -> set:
    features = run_cfg.analysis.get("features") or []
    return set(int(f) for f in features) & set(np.flatnonzero(active).tolist()) if features else \
        set(np.flatnonzero(active).tolist())


# ---------------------------------------------------------------------------- main

@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="enrichment_selection", version_base="1.3")
def main(cfg) -> None:
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    s = cfg.selection
    run_dir = resolve(cfg.inputs.enrichment)
    run_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    if run_cfg.analysis.get("features"):
        raise ValueError(f"{run_dir} tested a subset of features; inputs.enrichment must be a run over all features")
    seed_dirs = ([resolve(p) for p in cfg.inputs.seed_runs] if cfg.inputs.seed_runs is not None
                 else sorted(p for p in run_dir.parent.glob(run_dir.name + "_seed*") if p.is_dir()))
    runs = [("seed" + str(run_cfg.analysis.get("seed", 42)), run_dir, run_cfg)]
    for d in seed_dirs:
        c = OmegaConf.load(d / ".hydra" / "config.yaml")
        for key in ("transform", "min_size", "max_size", "permutations"):
            if c.analysis.get(key) != run_cfg.analysis.get(key):
                raise ValueError(f"seed run {d} differs from {run_dir} in analysis.{key}")
        if OmegaConf.to_container(c.libraries) != OmegaConf.to_container(run_cfg.libraries):
            raise ValueError(f"seed run {d} tested other gene-set libraries than {run_dir}")
        runs.append((f"seed{c.analysis.seed}", d, c))
    if len({label for label, *_ in runs}) < len(runs):
        raise ValueError(f"runs with the same seed: {[str(d) for _, d, _ in runs]}")
    print(f"[select] runs: {', '.join(f'{label} ({d.name})' for label, d, _ in runs)}", flush=True)

    # Accumulators, universe and library exactly as the run used them
    accum_dir = resolve(run_cfg.inputs.accum_dir)
    act_sum = load_table(accum_dir, "act_sum")
    occ = np.load(accum_dir / "gene_occurrence.npy").astype(np.float64)
    symbols = vocab.gene_symbols(base_model(cfg))
    symbols = (symbols + [""] * act_sum.shape[1])[:act_sum.shape[1]]
    universe, library = universe_and_library(symbols, occ, run_cfg.libraries,
                                             int(run_cfg.analysis.min_size), int(run_cfg.analysis.max_size))
    terms = panel_terms(cfg.programs)
    missing = terms[~terms.Term.isin(library)]
    if len(missing):
        print(f"[select] not tested by the run (library or size filter), left out: {missing.Term.tolist()}")
    terms = terms[terms.Term.isin(library)].reset_index(drop=True)
    if terms.empty:
        raise RuntimeError("no panel term was tested by the run")
    F, G = act_sum.shape
    active = np.asarray(act_sum.sum(axis=1)).ravel() > 0
    pos = {symbols[i]: i for i in np.flatnonzero(universe)}

    prof = profile(act_sum, occ)
    topk20 = topk_mass(prof, ks=(20,))[20]
    total = np.asarray(prof.sum(axis=1)).ravel()
    sq = np.asarray(prof.multiply(prof).sum(axis=1)).ravel()
    n_eff = np.divide(total ** 2, sq, out=np.zeros(F), where=sq > 0)
    mean_u = np.asarray(prof[:, np.flatnonzero(universe)].sum(axis=1)).ravel() / max(int(universe.sum()), 1)

    # Over-representation among each feature's top annotated genes
    annotated = np.zeros(G, dtype=bool)
    for genes in library.values():
        annotated[[pos[g] for g in genes]] = True
    top = top_gene_mask(prof, annotated, int(s.top_genes))
    member_rows = [[pos[g] for g in library[t]] for t in terms.Term]
    members = sp.csr_matrix((np.ones(sum(map(len, member_rows)), dtype=bool),
                             (np.repeat(np.arange(len(terms)), [len(r) for r in member_rows]), np.concatenate(member_rows))),
                            shape=(len(terms), G))
    overlap, p_ora = over_representation(top, members, annotated)
    q_ora = global_bh(p_ora.ravel(), n_tests=p_ora.size).reshape(p_ora.shape)
    table = pd.DataFrame({"feature_id": np.repeat(np.arange(F), len(terms)), "Term": np.tile(terms.Term.to_numpy(), F),
                          "ora_overlap": overlap.ravel(), "ora_q": q_ora.ravel()})
    table = table.merge(terms, on="Term")

    # The run's statistics, leading-edge weights and reference-set shares
    main_rows = read_terms(run_dir, terms.Term, ["feature_id", "Term", "es", "nes", "pval", "global_q",
                                                 "leading_edge_size", "leading_edge"])
    table = table.merge(main_rows, on=["feature_id", "Term"], how="left")
    table["global_q"] = table.global_q.fillna(1.0)
    table["leading_edge_size"] = table.leading_edge_size.fillna(0).astype(int)
    ref_sets = {}
    for p in cfg.programs:
        ref = p.get("leading_edge_set")
        if ref:
            with open(REPO_ROOT / "data" / "gene_sets" / f"{ref.collection}.json") as f:
                ref_sets[p["code"]] = (set(json.load(f)[ref.term]["geneSymbols"]), float(ref.min_share))
    le_eff = np.zeros(len(table)); share = np.full(len(table), np.nan)
    has_le = table.leading_edge.notna().to_numpy()
    for fid, rows in table[has_le].groupby("feature_id").groups.items():
        row = np.asarray(prof.getrow(fid).todense()).ravel()
        for i in rows:
            genes = [g for g in leading_edge_genes(table.at[i, "leading_edge"]) if g in pos]
            le_eff[i] = effective_size(row[[pos[g] for g in genes]] - mean_u[fid]) if genes else 0.0
            ref = ref_sets.get(table.at[i, "program"])
            if ref and genes:
                share[i] = len(set(genes) & ref[0]) / len(genes)
    table["le_eff"], table["leading_edge_share"] = le_eff, share
    table.loc[table.global_q >= s.q_max, "leading_edge"] = None

    # q-values in every run, and robustness
    tested = np.zeros((len(runs), F), dtype=bool)
    for k, (label, d, c) in enumerate(runs):
        rows = main_rows if k == 0 else read_terms(d, terms.Term, ["feature_id", "Term", "global_q", "leading_edge_size"])
        rows = rows[["feature_id", "Term", "global_q", "leading_edge_size"]].rename(
            columns={"global_q": f"q_{label}", "leading_edge_size": f"le_{label}"})
        table = table.merge(rows, on=["feature_id", "Term"], how="left")
        tested[k, sorted(tested_features(c, active))] = True
        t = tested[k, table.feature_id.to_numpy()]
        table[f"q_{label}"] = np.where(t, table[f"q_{label}"].fillna(1.0), np.nan)
        table[f"le_{label}"] = np.where(t, table[f"le_{label}"].fillna(0), np.nan)
    qs = table[[f"q_{label}" for label, *_ in runs]].to_numpy()
    les = table[[f"le_{label}" for label, *_ in runs]].to_numpy()
    table["n_runs"] = np.isfinite(qs).sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # features without activation: no run tested them
        table["q_median"] = np.nanmedian(qs, axis=1)
    sig = (qs < s.q_max) & (les >= s.min_leading_edge)
    table["robust"] = pd.array(np.where(table.n_runs == len(runs), sig.all(axis=1), pd.NA), dtype="boolean")
    feat = pd.DataFrame({"feature_id": np.arange(F), "topk20_mass": topk20, "n_eff_genes": n_eff})

    # Candidates per program
    dots = table[(table.global_q < s.q_max) & (table.leading_edge_size >= s.min_leading_edge)] \
        .rename(columns={"global_q": "q"})[["feature_id", "program", "column", "q"]]
    concentrated = set(feat.feature_id[~((feat.topk20_mass < s.max_topk20_mass) & (feat.n_eff_genes >= s.min_effective_genes))])
    q = table[table.anchor & (table.global_q < s.q_max) & (table.leading_edge_size >= s.min_leading_edge)
              & (table.le_eff >= s.min_leading_edge) & (table.ora_q < s.ora_q_max) & (table.ora_overlap >= s.ora_min_overlap)
              & ~table.feature_id.isin(concentrated)].copy()
    for code, (_, min_share) in ref_sets.items():
        q = q[(q.program != code) | (q.leading_edge_share >= min_share)]
    q["score"] = np.minimum(-np.log10(q.global_q.clip(lower=1e-300)), -np.log10(q.ora_q.clip(lower=1e-300)))
    best = q.sort_values(["score", "global_q"], ascending=[False, True], kind="stable").drop_duplicates(["feature_id", "program"])
    robust_prog = table[table.anchor].groupby(["feature_id", "program"]).robust.agg(
        lambda r: pd.NA if r.isna().all() else bool(r.fillna(False).any()))
    cands, selection, seed_ids = [], [], set()
    for p in cfg.programs:
        code = p["code"]
        b = best[best.program == code]
        if b.empty:
            selection.append(dict(program=code, pick=None, status="no candidate", n_candidates=0, n_tier1=0))
            continue
        c = b.merge(tiers(dots, code, int(s.single_program_max_off), int(s.two_program_max_off)), on="feature_id")
        c = c[c.tier.notna()].sort_values(["tier", "score", "global_q"], ascending=[True, False, True], kind="stable")
        c["robust"] = [robust_prog.get((f, code), pd.NA) for f in c.feature_id]
        c["rank"] = np.arange(1, len(c) + 1)
        c = c.merge(feat, on="feature_id", how="left")
        c["top_genes"] = [",".join(sorted(leading_edge_genes(le), key=lambda g, f=f: -prof[f, pos[g]] if g in pos else 0)[:8])
                          for f, le in zip(c.feature_id, c.leading_edge)]
        fid, status = pick(c)
        if len(runs) == 1 and fid is not None:
            status = "single run (no seed runs)"
        selection.append(dict(program=code, pick=fid, status=status, n_candidates=len(c), n_tier1=int((c.tier == 1).sum())))
        # untested candidates among the top seed_top_k (all of them without seed runs), and the one that stops the walk
        head = c.head(int(s.seed_top_k))
        seed_ids |= set(head.feature_id[head.robust.isna() | (len(runs) == 1)].astype(int))
        if fid is None and status.startswith("needs"):
            seed_ids.add(int(status.split("feature ")[1].rstrip(")")))
        cands.append(c.rename(columns={"global_q": "q", "Term": "term"})[CANDIDATE_COLUMNS])

    table.to_parquet(save_dir / "panel_terms.parquet")
    (pd.concat(cands, ignore_index=True) if cands else pd.DataFrame(columns=CANDIDATE_COLUMNS)).to_csv(
        save_dir / "candidates.csv", index=False)
    selection = pd.DataFrame(selection).astype({"pick": "Int64"})
    selection.to_csv(save_dir / "selection.csv", index=False)
    lines = []
    if seed_ids:
        lib = " ".join(f"libraries.{k}={str(bool(v)).lower()}" for k, v in run_cfg.libraries.items())
        for seed in SEED_RUNS:
            lines.append(f"python -m scripts.enrichment_analysis sae_checkpoint.experiment={cfg.sae_checkpoint.experiment} "
                         f"sae_checkpoint.timestamp={cfg.sae_checkpoint.timestamp} "
                         f"analysis.transform={run_cfg.analysis.get('transform', 'log')} {lib} analysis.seed={seed} "
                         f"'analysis.features=[{','.join(map(str, sorted(seed_ids)))}]'")
    (save_dir / "seed_features.txt").write_text(
        (f"# {len(seed_ids)} candidates not tested by every seed run\n" + "\n".join(lines) + "\n") if seed_ids
        else "# every candidate up to the picks was tested by every seed run\n")
    print(selection.to_string(index=False))
    print(f"[select] -> {save_dir}" + (f"; {len(seed_ids)} candidates need seed runs (seed_features.txt)" if seed_ids else ""))


if __name__ == "__main__":
    main()
