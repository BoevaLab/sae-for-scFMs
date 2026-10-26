"""
Feature typology across the cell and gene axes (Fig. 4a-d, Supplementary Fig. S18).

Joins, per feature, the cell-level variance partition (commonality_analysis), the gene-level
summary (gene_selectivity's features.parquet) and the enrichment calls (enrichment_analysis), and
classifies features by top-20 gene mass: diffuse (< `diffuse_below`), concentrated
(> `concentrated_above`) or intermediate; features that never fired have no class.

Outputs (experiments/typology/<experiment>/<timestamp>/):
``feature_typology.csv``  one row per feature: R2_full, unique_bio, unique_tech, shared, B, S,
                          R2_additive, interaction, S_int, S_add (when the commonality run has
                          them), informative (R2_full >= min_r2_full), topk{5,20,50}_mass, n_eff_genes,
                          expr_focus, expr_std, class, enriched (has a collapsed enrichment cluster)
``typology_counts.csv``   per class (and all features): number of features and how many of them
                          are informative and enriched (the counts behind Fig. 4b-d and the Venn
                          diagram of enriched vs cell-informative features)
"""
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

PARTITION = ["R2_full", "R2_bio", "R2_dataset", "unique_bio", "unique_tech", "shared", "B", "S",
             "R2_additive", "interaction", "S_int", "S_add"]


def classify(topk20: pd.Series, diffuse_below: float, concentrated_above: float) -> pd.Series:
    out = pd.Series("intermediate", index=topk20.index, dtype=object)
    out[topk20 < diffuse_below] = "diffuse"
    out[topk20 > concentrated_above] = "concentrated"
    out[topk20.isna()] = None
    return out


def typology(genes: pd.DataFrame, cells: pd.DataFrame | None, enriched_ids, cfg) -> pd.DataFrame:
    df = genes.set_index("feature_id").copy()
    df["class"] = classify(df["topk20_mass"], cfg.diffuse_below, cfg.concentrated_above)
    if cells is not None:
        df = df.join(cells[[c for c in PARTITION if c in cells]], how="left")
        df["informative"] = df["R2_full"] >= cfg.min_r2_full
    else:
        df["informative"] = np.nan
    df["enriched"] = df.index.isin(set(enriched_ids)) if enriched_ids is not None else np.nan
    return df.reset_index()


def counts(df: pd.DataFrame) -> pd.DataFrame:
    """Features, informative and enriched features per class, plus all features and the non-diffuse
    group (intermediate + concentrated) used for the mosaic of Fig. 4d."""
    groups = {"all": df, "diffuse": df[df["class"] == "diffuse"],
              "intermediate": df[df["class"] == "intermediate"],
              "concentrated": df[df["class"] == "concentrated"],
              "non-diffuse": df[df["class"].isin(["intermediate", "concentrated"])]}
    rows = []
    for name, g in groups.items():
        informative = g["informative"].astype("boolean")
        enriched = g["enriched"].astype("boolean")
        rows.append(dict(group=name, n=len(g), n_informative=int(informative.sum()),
                         n_enriched=int(enriched.sum()), n_informative_enriched=int((informative & enriched).sum()),
                         frac_informative=float(informative.mean()) if len(g) else np.nan))
    return pd.DataFrame(rows)


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="feature_typology", version_base="1.3")
def main(cfg) -> None:
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    genes = pd.read_parquet(REPO_ROOT / cfg.inputs.selectivity)
    cells_path, enrich_path = REPO_ROOT / cfg.inputs.commonality, REPO_ROOT / cfg.inputs.enrichment
    cells = pd.read_csv(cells_path).rename_axis("feature_id") if cells_path.exists() else None
    enriched = (pd.read_csv(enrich_path)["feature_id"].unique()
                if enrich_path.exists() and enrich_path.stat().st_size > 1 else
                ([] if enrich_path.exists() else None))
    for name, found in (("commonality", cells is not None), ("enrichment", enriched is not None)):
        if not found:
            print(f"[feature_typology] no {name} output; its columns are left empty")
    df = typology(genes, cells, enriched, cfg)
    df.to_csv(save_dir / "feature_typology.csv", index=False)
    summary = counts(df)
    summary.to_csv(save_dir / "typology_counts.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
