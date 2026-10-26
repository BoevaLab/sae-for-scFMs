"""Gene symbols of the scFM vocabularies, read from the model files (no scFM package needed)."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


def gene_symbols(model: str, data_root: Path = DATA_ROOT) -> list:
    """Symbol of every vocabulary id of `model` (scgpt, geneformer, scfoundation); "" for special
    tokens (<pad>, <cls>, ...) and for ids without a symbol."""
    path = Path(data_root) / model
    if model == "geneformer":
        with open(path / "token_dictionary_gc104M.pkl", "rb") as f:
            vocab = pickle.load(f)  # Ensembl id -> token id
        with open(path / "gene_name_id_dict_gc104M.pkl", "rb") as f:
            symbol_of = {ensembl: symbol for symbol, ensembl in pickle.load(f).items()}
        id2symbol = {i: symbol_of.get(ensembl, "") for ensembl, i in vocab.items()}
    elif model == "scgpt":
        with open(path / "vocab.json") as f:
            id2symbol = {i: g for g, i in json.load(f).items()}
    elif model == "scfoundation":
        id2symbol = dict(enumerate(pd.read_csv(path / "OS_scRNA_gene_index.19264.tsv", sep="\t")["gene_name"]))
    else:
        raise ValueError(f"Unknown model {model!r}")
    symbols = [""] * (max(id2symbol) + 1)
    for i, symbol in id2symbol.items():
        symbols[i] = "" if str(symbol).startswith("<") else str(symbol)
    return symbols


def gene_token_mask(genes: np.ndarray, symbols: list, invalid_id: int = 65535) -> np.ndarray:
    """True for tokens that are genes of the vocabulary (not padding, special or appended tokens)."""
    genes = np.asarray(genes)
    has_symbol = np.array([bool(s) for s in symbols] + [False])
    return (genes != invalid_id) & has_symbol[np.clip(genes, 0, len(symbols))]
