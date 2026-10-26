"""
Tokenize a dataset for Geneformer finetuning (rank value encoding, as Geneformer's
TranscriptomeTokenizer does).

Genes are mapped to Ensembl ids in the Geneformer vocabulary, each cell's expression is
normalized by the genes' corpus medians and ranked, and the ranked token ids are written as a
Hugging Face dataset with the configured cell attributes. Counts are scaled to 10,000 per cell
first (data.preprocess.counts: true); otherwise the matrix is taken to be log1p-normalized and is
exponentiated back. Cells without any vocabulary gene are dropped.

Usage: python -m scripts.geneformer_tokenize data=lung
"""
import logging
import os
import pickle
import sys
from pathlib import Path

import hydra
import numpy as np
import scipy.sparse as sp
from geneformer import ENSEMBL_DICTIONARY_FILE, GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE, TranscriptomeTokenizer
from geneformer.tokenizer import rank_genes

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.io_utils import load_adata

log = logging.getLogger(__name__)


def partition_files(cfg):
    """h5ad file names of the dataset: one for single-file datasets, else one per partition."""
    n_partitions = cfg.data.get("n_partitions", None)
    if not n_partitions or sum(n_partitions) <= 1:
        yield cfg.data.file_name
        return
    offset = cfg.data.get("offset_partition", 0)
    for query_id, n_parts in zip(cfg.data.query_list, n_partitions):
        for i in range(int(n_parts)):
            yield f"{query_id}/partition_{i + offset}.h5ad"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="tokenize_geneformer", version_base="1.3")
def main(cfg) -> None:
    vocab = load_pickle(TOKEN_DICTIONARY_FILE)
    gene_median = load_pickle(GENE_MEDIAN_FILE)
    ensembl_of = load_pickle(ENSEMBL_DICTIONARY_FILE)
    counts = cfg.data.preprocess.get("counts", False)

    tokenized, metadata = [], {k: [] for k in cfg.cell_attributes}
    for file_name in partition_files(cfg):
        cfg.data.file_name = file_name
        log.info(f"Tokenizing {file_name}")
        adata = load_adata(cfg)
        if "ensembl_id" not in adata.var.columns:
            adata = adata[:, adata.var["gene_name"].isin(ensembl_of.keys())].copy()
            adata.var["ensembl_id"] = adata.var["gene_name"].map(ensembl_of)
        adata = adata[:, adata.var["ensembl_id"].isin(vocab.keys())]
        medians = np.array([gene_median[g] for g in adata.var["ensembl_id"]])
        tokens = np.array([vocab[g] for g in adata.var["ensembl_id"]])

        for start in range(0, adata.n_obs, cfg.chunk_size):
            chunk = adata[start:start + cfg.chunk_size]
            X = sp.csr_matrix(chunk.X, dtype=np.float64)
            if counts:
                X = sp.diags(1e4 / np.maximum(np.asarray(X.sum(axis=1)).ravel(), 1)) @ X
            else:
                X = X.expm1()
            X = sp.csr_matrix(X @ sp.diags(1.0 / medians))
            for i in range(X.shape[0]):
                row = X.getrow(i)
                tokenized.append(rank_genes(row.data, tokens[row.indices]))
            for k in metadata:
                metadata[k] += chunk.obs[k].tolist()

    keep = [i for i, cell in enumerate(tokenized) if cell.size > 0]
    if len(keep) < len(tokenized):
        log.warning(f"Dropping {len(tokenized) - len(keep)} cells without any gene in the Geneformer vocabulary")
    tokenized = [tokenized[i] for i in keep]
    metadata = {k: [v[i] for i in keep] for k, v in metadata.items()}

    tokenizer = TranscriptomeTokenizer({k: k for k in cfg.cell_attributes}, nproc=1, model_version=cfg.model_version)
    dataset = tokenizer.create_dataset(tokenized, metadata, [])
    out = Path(cfg.output_dir)
    dataset.save_to_disk(str(out / f"{cfg.data.name}.dataset"))
    with open(out / f"{cfg.data.name}_lengths.pkl", "wb") as f:
        pickle.dump(list(dataset["length"]), f)
    log.info(f"Wrote {len(dataset)} cells to {out / f'{cfg.data.name}.dataset'}")


if __name__ == "__main__":
    main()
