"""Cell-label scoring of SAE features and gene family helpers."""
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score


def preprocess_labels(obs: pd.DataFrame, names: Iterable[str], dataset: str, min_cells: int = 10,
                      columns: Mapping[str, str] | None = None) -> dict:
    """Turn categorical cell labels into one-vs-rest indicator sublabels.

    Args:
        obs: per-cell metadata (adata.obs), one row per scored cell
        names: labels to build, e.g. ["batch", "celltype"]; label `name` is read from the obs
            column `columns.get(name, name)`
        dataset: dataset name; some datasets add composite sublabels (see `_add_composites`)
        min_cells: sublabels with fewer positive cells (or no negative cells) are dropped;
            their AMI would be undefined or dominated by a handful of cells
        columns: label name -> obs column, where they differ

    Returns:
        {label: {sublabel: boolean array over cells}}
    """
    columns = dict(columns or {})

    def column(name):
        return obs[columns.get(name, name)].astype("category").cat.remove_unused_categories()

    labels = {}
    for name in names:
        values = column(name)
        labels[name] = {category: (values == category).to_numpy() for category in values.cat.categories}
    _add_composites(labels, column, dataset)

    return {name: {k: v for k, v in sublabels.items() if min_cells <= int(np.sum(v)) < len(v)}
            for name, sublabels in labels.items()}


def _add_composites(labels: dict, column, dataset: str) -> None:
    """Dataset-specific sublabels that group or combine categories (only for requested labels)."""
    def prefix_groups(name, prefixes):
        values = column(name).astype(str)
        for prefix in prefixes:
            labels[name][prefix] = values.str.startswith(prefix).to_numpy()

    if dataset in ("covid", "covid_evaluation") and "celltype" in labels:
        prefix_groups("celltype", ["T"])
        labels["celltype"]["Phagocytes"] = column("celltype").astype(str).isin(["Monocyte", "DC"]).to_numpy()
        keep = ["Monocyte", "NK", "T", "B", "Plasma", "HPC", "Platelets", "Phagocytes"]
        labels["celltype"] = {k: v for k, v in labels["celltype"].items() if k in keep}
    elif dataset == "pancreas" and "batch" in labels:
        prefix_groups("batch", ["inDrop"])
    elif dataset == "lung" and "batch" in labels:
        prefix_groups("batch", ["A", "B"])
        batches = column("batch").astype(str)
        labels["batch"]["starts_with_number"] = batches.str.match(r"\d").to_numpy()
        labels["batch"]["starts_with_letter"] = batches.str.match(r"[^\W\d_]").to_numpy()
    elif dataset == "immune" and "batch" in labels:
        prefix_groups("batch", ["Oetjen", "Sun_sample"])
    elif dataset == "kang" and "perturbation" in labels:
        # Cells of one cell type that were stimulated
        stim = labels["perturbation"]["stim"]
        celltypes = column("celltype")
        for celltype in celltypes.cat.categories:
            labels["perturbation"][f"{celltype}_stim"] = (celltypes == celltype).to_numpy() & stim
    elif dataset.startswith("tahoe") and "drug" in labels:
        # Treated cells of one cell line, for every drug other than the DMSO control
        cell_lines = column("cell_line")
        for drug, treated in list(labels["drug"].items()):
            if drug == "DMSO_TF":
                continue
            for cell_line in cell_lines.cat.categories:
                labels["drug"][f"{cell_line}_{drug.lower()}"] = (cell_lines == cell_line).to_numpy() & treated


def pool_cells(feature) -> np.ndarray:
    """[cells] or [cells, tokens] -> [cells]: a cell's activation is its max over tokens."""
    feature = np.asarray(feature)
    return feature.max(axis=1) if feature.ndim == 2 else feature


def label_scoring(feature, labels: dict, names: Iterable[str], scores: Iterable[str] = ("AMI",)) -> dict:
    """Score one feature against every sublabel of the given labels.

    Args:
        feature: per-cell activations [cells] or token activations [cells, tokens], scaled to [0, 1]
        labels: output of `preprocess_labels`
        names: labels to score
        scores: any of "AMI" (also returns the best threshold as "AMI_threshold") and "F1"

    Returns:
        {label: {sublabel: {score: value}}}, always with "direction": mean activation in the
        sublabel's cells minus outside them (> 0: the feature marks the sublabel, < 0: its absence)
    """
    pooled = pool_cells(feature).astype(np.float64)
    results = {}
    for name in names:
        results[name] = {}
        for sublabel, label in labels[name].items():
            out = results[name][sublabel] = {}
            label = np.asarray(label).astype(bool)
            out["direction"] = float(pooled[label].mean() - pooled[~label].mean())
            if "AMI" in scores:
                out["AMI"], out["AMI_threshold"] = calculate_AMI(pool_cells(feature), label, return_threshold=True)
            if "F1" in scores:
                out["F1"] = calculate_F1(pooled, label)
    return results


def calculate_AMI(feature, label, return_threshold: bool = False):
    """Max over activation thresholds of the AMI between `feature > t` (per cell) and a 0/1 label.

    `feature` is [cells] (pooled) or [cells, tokens] (a cell is active if any token exceeds t),
    normalized to [0, 1]. Thresholds at which every cell is (in)active carry no information and
    are skipped; when no threshold splits the cells, or the label is constant, AMI is NaN
    (sklearn would report 1.0 when both labelings are constant).
    """
    label = np.asarray(label)
    feature = pool_cells(feature)
    best, best_t = float("nan"), float("nan")
    if label.min() == label.max():
        return (best, best_t) if return_threshold else best
    for t in np.arange(0.0, 1.0, 0.2):
        active = feature > t
        if active.all() or not active.any():
            continue
        ami = adjusted_mutual_info_score(label, active)
        if not ami <= best:  # also true while best is NaN
            best, best_t = float(ami), float(t)
    return (best, best_t) if return_threshold else best


def calculate_F1(feature, label) -> float:
    """Max over activation thresholds of the F1 of `feature >= t` (per cell) against a 0/1 label."""
    label = np.asarray(label).astype(bool)
    feature = pool_cells(feature)
    best = float("nan")
    for t in np.arange(0.1, 1.0, 0.2):
        active = feature >= t
        tp = int((active & label).sum())
        errors = int((active ^ label).sum())
        if tp + errors:
            f1 = 2 * tp / (2 * tp + errors)
            best = f1 if np.isnan(best) else max(best, f1)
    return best


RIBOSOMAL_PROTEIN = re.compile(r"^(RP[LS]\d+(A|AL|L|X|Y\d?)?|RPLP[0-2]|RPSA|UBA52|FAU|RPS6K[ABCL]\d)$")


def is_ribosomal_protein_gene(gene: str) -> bool:
    """Cytosolic ribosomal protein genes (RPL*/RPS* incl. paralogs such as RPL3L, RPS4Y1; RPLP0-2,
    RPSA, UBA52, FAU) and the ribosomal protein S6 kinases (RPS6KA1-6, RPS6KB1-2, RPS6KC1, RPS6KL1).
    Excludes pseudogenes (RPL21P*), readthroughs (RPL36A-HNRNPH2) and mitochondrial ribosomal
    proteins (MRPL*, MRPS*)."""
    return bool(RIBOSOMAL_PROTEIN.match(str(gene)))


def gene_families(all_genes) -> dict:
    """Genes of a few families (by symbol): ribosomal proteins, mitochondrial, MHC, T-cell receptor,
    histones, metallothioneins and immunoglobulins."""
    families = {}
    families['RP'] = [gene for gene in all_genes if is_ribosomal_protein_gene(gene)]
    families['MIT'] = [gene for gene in all_genes if gene.startswith('MT-')]
    families['MHC'] = [gene for gene in all_genes if gene.startswith('HLA-')]
    families['TRAV'] = [gene for gene in all_genes if gene.startswith('TR')]
    families['H'] = [gene for gene in all_genes if gene.startswith(('H1', 'H2', 'H3', 'H4'))]
    families['MET'] = [gene for gene in all_genes if gene.startswith(('MT1', 'MT2', 'MT3', 'MT4'))]
    families['IG'] = [gene for gene in all_genes if gene.startswith('IG')]
    return families
