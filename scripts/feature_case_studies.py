"""
Individual-feature case studies (Fig. 5b-d, Supplementary Methods S1.8). Each analysis runs when
its `features` list is set.

encoding_strategy (S1.8, Fig. 5b)
    For each feature, the fraction of its genes that are ribosomal-protein genes, up- or down-regulated in `group`, or up-regulated in another cell
    type but not in `group` (up_in_other). Differential expression: each cell type against all other
    cells, on log-normalized counts; significant at BH-adjusted p < `max_padj` and |log2 fold change|
    >= `min_logfc`. A feature's genes are the fewest of its top genes (by frequency-normalized
    profile) that carry `mass` of its profile, or its top `top_n` genes when `mass` is null.
group_activation (Fig. 5c)
    Mean per-cell (max over tokens) activation of each feature per combination of `groupby`
    columns, after optionally renaming values (`rename`; unmapped values are dropped).
gene_properties (Fig. 5d)
    GC content and length of each feature's top `top_n` genes against all observed genes,
    compared with two-sample Kolmogorov-Smirnov tests. `metadata` is a CSV with columns
    hgnc_symbol, gc (GC fraction) and len (bp), e.g. exported from Ensembl BioMart.

Outputs (experiments/case_studies/<experiment>/<timestamp>/): encoding_strategy.csv (+ the
genes of each feature in encoding_strategy_genes.csv), group_activation.csv, gene_properties.csv
(tests) and gene_properties_genes.csv (GC and length of each feature's genes and of the background).
"""
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import ks_2samp

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.analysis import is_ribosomal_protein_gene
from sae4scfm.core.gene_stats import load_table, profile
from sae4scfm.core.io_utils import load_adata
from sae4scfm.core.pipeline import register_resolvers, run_config
from sae4scfm.core.vocab import gene_symbols

register_resolvers()


def top_genes(prof_row: np.ndarray, symbols: list, n: int) -> list:
    """Symbols of the n genes with the highest profile (genes the feature fired on, with a symbol)."""
    order = np.argsort(-prof_row, kind="stable")
    genes = [symbols[i] for i in order if prof_row[i] > 0 and i < len(symbols) and symbols[i]]
    return genes[:n]


def mass_genes(prof_row: np.ndarray, symbols: list, mass: float) -> list:
    """Symbols of the fewest genes with the highest profile that carry a fraction `mass` of the
    feature's profile (over the genes it fired on that have a symbol)."""
    order = np.argsort(-prof_row, kind="stable")
    named = [i for i in order if prof_row[i] > 0 and i < len(symbols) and symbols[i]]
    if not named:
        return []
    cumulative = np.cumsum(prof_row[named]) / prof_row[named].sum()
    return [symbols[i] for i in named[: int(np.searchsorted(cumulative, mass)) + 1]]


def de_genes(adata, groupby: str, group: str, method: str, max_padj: float, min_logfc: float = 0.0):
    """Genes up- and down-regulated in `group`, and genes up-regulated in any other cell type but not
    in `group`, each type tested against all other cells (log-normalized counts)."""
    counts = (adata.raw.X if adata.raw is not None else adata.X).copy()  # normalized below, in place
    ad = sc.AnnData(counts, obs=adata.obs[[groupby]].copy(),
                    var=pd.DataFrame(index=(adata.raw.var_names if adata.raw is not None else adata.var_names)))
    if adata.raw is not None and "name" in adata.raw.var:
        ad.var_names = adata.raw.var["name"].astype(str).values
    ad.var_names_make_unique()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    sizes = ad.obs[groupby].astype(str).value_counts()
    groups = [g for g, n in sizes.items() if n >= 2]  # a type with one cell has no variance
    sc.tl.rank_genes_groups(ad, groupby, groups=groups, reference="rest", method=method)

    def significant(g, sign):
        de = sc.get.rank_genes_groups_df(ad, group=g)
        lfc = sign * de["logfoldchanges"]
        return set(de.loc[(de["pvals_adj"] < max_padj) & (lfc > 0) & (lfc >= min_logfc), "names"])

    up, down = significant(group, 1), significant(group, -1)
    up_other = set().union(*(significant(g, 1) for g in groups if g != group)) - up
    return up, down, up_other


def per_cell_max(h5_path: Path, features, n_cells: int):
    """[cells, features] max activation over tokens, and the cells (positions) of the rows."""
    with h5py.File(h5_path, "r") as f:
        rows = f["cell_idx"][:] if "cell_idx" in f else np.arange(f["features"].shape[0])
        keep = np.flatnonzero((rows >= 0) & (rows < n_cells))
        values = np.stack([f["features"][:, :, j][keep].max(axis=1) for j in features], axis=1).astype(np.float64)
    return values, rows[keep]


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="feature_case_studies", version_base="1.3")
def main(cfg) -> None:
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    original_cfg = run_config(cfg, REPO_ROOT)
    symbols = gene_symbols(original_cfg.scfm.base_model)
    act_sum = load_table(REPO_ROOT / cfg.inputs.selectivity_dir, "act_sum")
    gene_occ = np.load(REPO_ROOT / cfg.inputs.selectivity_dir / "gene_occurrence.npy")
    prof = profile(act_sum, gene_occ)
    adata = None

    def data():
        nonlocal adata
        if adata is None:
            adata = load_adata(original_cfg, sub_sample=original_cfg.data.preprocess.get("split", 1.0))
        return adata

    es = cfg.encoding_strategy
    if es.features:
        up, down, up_other = de_genes(data(), es.groupby, es.group, es.de_method, es.max_padj, es.get("min_logfc", 0.0))
        sets = {"ribosomal": {s for s in symbols if is_ribosomal_protein_gene(s)},
                f"up_in_{es.group}": up, f"down_in_{es.group}": down, "up_in_other": up_other}
        rows, gene_rows = [], []
        for j in es.features:
            row = prof[j].toarray().ravel()
            genes = mass_genes(row, symbols, es.mass) if es.get("mass") else top_genes(row, symbols, es.top_n)
            rows.append(dict(feature_id=j, n_genes=len(genes), **{k: np.mean([g in v for g in genes]) for k, v in sets.items()}))
            gene_rows += [dict(feature_id=j, rank=r + 1, gene=g, **{k: g in v for k, v in sets.items()}) for r, g in enumerate(genes)]
        pd.DataFrame(rows).to_csv(save_dir / "encoding_strategy.csv", index=False)
        pd.DataFrame(gene_rows).to_csv(save_dir / "encoding_strategy_genes.csv", index=False)
        print(pd.DataFrame(rows).round(3).to_string(index=False))

    ga = cfg.group_activation
    if ga.features:
        values, cells = per_cell_max(REPO_ROOT / cfg.inputs.features_h5, list(ga.features), data().n_obs)
        obs = data().obs.iloc[cells][list(ga.groupby)].astype(str).reset_index(drop=True)
        for col, mapping in (ga.get("rename") or {}).items():
            if col in obs:
                obs[col] = obs[col].map(dict(mapping))
        df = pd.concat([obs, pd.DataFrame(values, columns=[f"feature_{j}" for j in ga.features])], axis=1).dropna()
        out = (df.melt(id_vars=list(ga.groupby), var_name="feature", value_name="activation")
               .groupby(["feature", *ga.groupby])["activation"].agg(["mean", "size"]).reset_index())
        out.to_csv(save_dir / "group_activation.csv", index=False)

    gp = cfg.gene_properties
    if gp.features:
        meta = pd.read_csv(REPO_ROOT / gp.metadata).drop_duplicates("hgnc_symbol").set_index("hgnc_symbol")
        background = sorted({symbols[i] for i in np.flatnonzero(gene_occ > 0) if i < len(symbols) and symbols[i]} & set(meta.index))
        tests, gene_rows = [], []
        for j in gp.features:
            genes = [g for g in top_genes(prof[j].toarray().ravel(), symbols, gp.top_n) if g in meta.index]
            for prop in ("gc", "len"):
                a, b = meta.loc[genes, prop].dropna(), meta.loc[background, prop].dropna()
                stat, p = ks_2samp(a, b)
                tests.append(dict(feature_id=j, property=prop, n_feature_genes=len(a), n_background=len(b),
                                  median_feature=a.median(), median_background=b.median(), ks_statistic=stat, p=p))
            gene_rows += [dict(group="feature", feature_id=j, gene=g, gc=meta.at[g, "gc"], len=meta.at[g, "len"]) for g in genes]
        gene_rows += [dict(group="background", feature_id=-1, gene=g, gc=meta.at[g, "gc"], len=meta.at[g, "len"])
                      for g in background]
        pd.DataFrame(tests).to_csv(save_dir / "gene_properties.csv", index=False)
        pd.DataFrame(gene_rows).to_csv(save_dir / "gene_properties_genes.csv", index=False)
        print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()
