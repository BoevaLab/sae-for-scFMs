"""
Cell-type composition of cell-level SAE features (Supplementary Methods S1.7; Fig. 5a, S19, S20).

A feature is assigned to a target cell type when its one-vs-rest AMI for that type (analyze_features)
exceeds `min_ami` and is the feature's highest AMI over all scored sublabels. Each assigned feature
gets the first category that applies:

1. low_expression: the activation-weighted mean input value of its firing tokens is below the
   `low_expression_quantile` of the input values of all gene tokens (above the 1 - q quantile for
   rank-encoded models such as Geneformer, whose input is the token's position);
2. ribosomal: its share of activation on ribosomal-protein genes exceeds `ribosomal_enrichment`
   times their share of gene-token occurrences;
3. concentrated: its frequency-normalized profile has more than `concentrated_above` of its mass on
   its top-20 genes;
4. dispersed.

Inputs: results-<timestamp>.csv (analyze_features), the gene_selectivity accumulators, and the
input values of the generate_features h5, all from the same SAE and dataset.

Outputs (experiments/composition/<experiment>/<timestamp>/):
``celltype_features.csv``     assigned features with their AMI, metrics and category
``celltype_composition.csv``  number of features per cell type and category
"""
import os
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import hydra
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.analysis import is_ribosomal_protein_gene
from sae4scfm.core.gene_stats import GeneStats, profile, topk_mass
from sae4scfm.core.io_utils import INVALID_GENE_ID, load_analysis_results
from sae4scfm.core.pipeline import register_resolvers, run_config
from sae4scfm.core.vocab import gene_symbols, gene_token_mask

register_resolvers()

CATEGORIES = ["low_expression", "ribosomal", "concentrated", "dispersed"]
RANK_ENCODED = {"geneformer"}  # input value = position in the expression-ranked sequence


def assign_cell_types(results: pd.DataFrame, cell_types, label: str, min_ami: float) -> pd.DataFrame:
    """Features whose AMI for a target cell type exceeds `min_ami` and is their highest AMI."""
    ami = results["label_scoring"]
    ami = ami[[c for c in ami.columns if c.endswith("_AMI")]]
    best = ami.max(axis=1)
    rows = []
    for cell_type in cell_types:
        col = f"{label}_{cell_type}_AMI"
        if col not in ami:
            print(f"[celltype_composition] no {col} in the analysis results; skipped")
            continue
        hit = (ami[col] > min_ami) & (ami[col] >= best)
        rows += [dict(feature_id=int(f), cell_type=cell_type, AMI=float(ami.at[f, col])) for f in ami.index[hit]]
    return pd.DataFrame(rows, columns=["feature_id", "cell_type", "AMI"])


def input_value_quantile(h5_path: Path, symbols: list, q: float, rank_encoded: bool) -> float:
    """Quantile of the input values of all gene tokens (positions for rank-encoded models)."""
    with h5py.File(h5_path, "r") as f:
        genes = f["genes"][:].astype(np.int64)
        values = f["values"][:] if "values" in f else np.broadcast_to(np.arange(genes.shape[1]), genes.shape)
    return float(np.quantile(values[gene_token_mask(genes, symbols, INVALID_GENE_ID)], q))


def categorize(stats: GeneStats, features, symbols, threshold: float, rank_encoded: bool, cfg) -> pd.DataFrame:
    rows = np.asarray(features, dtype=int)
    act = stats.act_sum[rows].toarray().astype(np.float64)
    act_expr = stats.act_expr_sum[rows].toarray().astype(np.float64)
    occ = stats.gene_occurrence.astype(np.float64)
    ribo = np.array([is_ribosomal_protein_gene(s) for s in symbols[:act.shape[1]]] + [False] * max(0, act.shape[1] - len(symbols)))
    total = act.sum(axis=1)
    mean_value = np.divide(act_expr.sum(axis=1), total, out=np.full(len(rows), np.nan), where=total > 0)
    observed = np.divide(act[:, ribo].sum(axis=1), total, out=np.zeros(len(rows)), where=total > 0)
    ribo_enrichment = observed / (occ[ribo].sum() / occ.sum())
    topk20 = topk_mass(profile(stats.act_sum[rows], stats.gene_occurrence), ks=(20,))[20]

    low = mean_value > threshold if rank_encoded else mean_value < threshold
    category = np.select([low, ribo_enrichment > cfg.ribosomal_enrichment, topk20 > cfg.concentrated_above],
                         CATEGORIES[:3], default=CATEGORIES[3])
    return pd.DataFrame({"feature_id": rows, "mean_input_value": mean_value, "ribosomal_enrichment": ribo_enrichment,
                         "topk20_mass": topk20, "category": category})


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="celltype_composition", version_base="1.3")
def main(cfg) -> None:
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    original_cfg = run_config(cfg, REPO_ROOT)
    model = original_cfg.scfm.base_model
    symbols = gene_symbols(model)
    rank_encoded = model in RANK_ENCODED

    results = load_analysis_results(REPO_ROOT / cfg.inputs.analysis_dir, cfg.sae_checkpoint.timestamp + cfg.inputs.get("tag", ""))
    assigned = assign_cell_types(results, cfg.cell_types, cfg.label, cfg.min_ami)
    q = cfg.low_expression_quantile
    threshold = input_value_quantile(REPO_ROOT / cfg.inputs.features_h5, symbols, 1 - q if rank_encoded else q, rank_encoded)
    print(f"[celltype_composition] {len(assigned)} assignments; low-expression threshold {threshold:.3g} "
          f"({'above' if rank_encoded else 'below'})")

    stats = GeneStats(REPO_ROOT / cfg.inputs.selectivity_dir)
    metrics = categorize(stats, sorted(assigned["feature_id"].unique()), symbols, threshold, rank_encoded, cfg)
    features = assigned.merge(metrics, on="feature_id", how="left")
    features.to_csv(save_dir / "celltype_features.csv", index=False)
    composition = (features.groupby(["cell_type", "category"]).size()
                   .reindex(pd.MultiIndex.from_product([list(cfg.cell_types), CATEGORIES], names=["cell_type", "category"]),
                            fill_value=0).rename("count").reset_index())
    composition.to_csv(save_dir / "celltype_composition.csv", index=False)
    print(composition.pivot(index="cell_type", columns="category", values="count").to_string())


if __name__ == "__main__":
    main()
