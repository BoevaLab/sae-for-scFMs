from pathlib import Path
from typing import Literal
import torch
import pickle
from omegaconf import DictConfig
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import geneformer
from geneformer import perturber_utils as pu
from geneformer.tokenizer import rank_genes

from ..base import ModelAdapter
from sae4scfm.core.utils import OutputHook, SeqDataset, highly_variable_genes, seeded


def geneformer_file(name: str) -> str:
    """A Geneformer resource: a file constant of the geneformer package (e.g. TOKEN_DICTIONARY_FILE) or a path."""
    return getattr(geneformer, name, name)


def geneformer_collate_fn(batch):
    genes = [item["genes"] for item in batch]
    lengths = [item["length"] for item in batch]

    return {
        "genes": genes,
        "length": lengths,
        "cell_idx": torch.tensor([int(item["cell_idx"]) for item in batch], dtype=torch.long),
    }


class GeneformerAdapter(ModelAdapter):
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg: Hydra config for this adapter
        """
        self.cfg = cfg
        self.model_dir = Path(cfg.scfm.model_dir)
        self.model = None
        self.vocab = None
        self.handle = None
        self.output_hook = OutputHook()
        self.train_data_loader = None
        self.test_data_loader = None
    
    def load_model(self):
        """Load vocab and model"""
        self.vocab = self._load_vocab()
        self.model = pu.load_model('Pretrained', None, self.model_dir, 'eval', quantize=False)
        self.model.eval()
        return self.model

    def _load_vocab(self):
        """Load Geneformer vocabulary from file"""
        with open(geneformer_file(self.cfg.scfm.vocab_file), "rb") as f:
            vocab = pickle.load(f)
        return vocab
    
    def _load_additional_files(self):
        with open(geneformer_file(self.cfg.scfm.gene_median_file), "rb") as f:
            self.gene_median_dict = pickle.load(f)

        with open(geneformer_file(self.cfg.scfm.ensembl_dictionary_file), "rb") as f:
            self.ensembl_dict = pickle.load(f)

    def preprocess_data(self, adata, shuffle=True):
        """Preprocess AnnData and create DataLoaders"""
        self._load_additional_files()

        # if ensembl_id not in adata.var, map gene names to ensembl ids
        if 'ensembl_id' not in adata.var.columns:
            # filter by existing match
            adata.var["id_in_vocab"] = [1 if gene in self.ensembl_dict.keys() else -1 for gene in adata.var["gene_name"]]
            adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()
            adata.var['ensembl_id'] = adata.var["gene_name"].map(self.ensembl_dict)

        # filter genes by those in vocab
        adata.var["id_in_vocab"] = [1 if gene in self.vocab.keys() else -1 for gene in adata.var["ensembl_id"]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0].copy()

        gene_col = "ensembl_id"

        if self.cfg.data.preprocess.subset_hvg:
            print('HVG', flush=True)
            highly_variable_genes(adata, self.cfg.data.preprocess.subset_hvg,
                                  "seurat_v3" if self.cfg.data.preprocess.counts else "cell_ranger", subset=True)

        norm_factor_vector = np.array([self.gene_median_dict[i] for i in adata.var[gene_col]])

        # check if counts matrix or normalized
        if self.cfg.data.preprocess.counts:
            adata.obs['n_counts'] = adata.X.sum(axis=1)

            coding_miRNA_ids = adata.var[gene_col]
            coding_miRNA_tokens = np.array([self.vocab[i] for i in coding_miRNA_ids])
            target_sum = 10000

            tokenized_cells = []
            for i in range(0, adata.shape[0], 512):
                idx = slice(i, min(i + 512, adata.shape[0]))
                n_counts = adata[idx].obs["n_counts"].values[:, None]
                X_view = adata[idx, :].X
                X_norm_unscaled = X_view / n_counts * target_sum
                X_norm = X_norm_unscaled / norm_factor_vector
                X_norm = sp.csr_matrix(X_norm)
                tokenized_cells += [
                    rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
                    for i in range(X_norm.shape[0])
                ]
        else: # undo log transform
            adata.X = np.expm1(adata.X)
            coding_miRNA_ids = adata.var[gene_col]
            coding_miRNA_tokens = np.array([self.vocab[i] for i in coding_miRNA_ids])

            tokenized_cells = []
            for i in range(0, adata.shape[0], 512):
                idx = slice(i, min(i + 512, adata.shape[0]))
                X_view = adata[idx, :].X
                X_norm = X_view / norm_factor_vector
                X_norm = sp.csr_matrix(X_norm)
                tokenized_cells += [
                    rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
                    for i in range(X_norm.shape[0])
                ]

        # Cells without any vocab gene cannot be tokenized; drop them but keep the positions of the
        # remaining cells (cell_idx) so outputs stay aligned with the input AnnData.
        cell_idx = [i for i, cell in enumerate(tokenized_cells) if cell.size > 0]
        if len(cell_idx) < len(tokenized_cells):
            print(f"Warning: {len(tokenized_cells) - len(cell_idx)} cells have no genes in the token "
                  "dictionary (empty droplets or invalid cells?) and are excluded.")
            tokenized_cells = [tokenized_cells[i] for i in cell_idx]


        if self.cfg.scfm.version == 'v2':  # v2 uses CLS + EOS tokens
            cls_id = self.vocab.get("<cls>")
            eos_id = self.vocab.get("<eos>")
            if cls_id is None or eos_id is None:
                raise ValueError("Geneformer v2 requires <cls> and <eos> tokens in vocab.")

            genes_list = []
            lengths_list = []
            for item in tokenized_cells:
                # Truncate to leave space for CLS/EOS so final length <= max_len
                core = torch.from_numpy(item).int()[: self.cfg.scfm.preprocessing.max_len - 2]
                genes = torch.cat(
                    [
                        torch.tensor([cls_id], dtype=torch.int32),
                        core,
                        torch.tensor([eos_id], dtype=torch.int32),
                    ]
                )
                genes_list.append(genes)
                lengths_list.append(int(genes.shape[0]))

            tokenized_data = {"genes": genes_list, "length": lengths_list}
        else:
            genes_list = [torch.from_numpy(item).int()[: self.cfg.scfm.preprocessing.max_len] for item in tokenized_cells]
            tokenized_data = {
                "genes": genes_list,
                "length": [int(g.shape[0]) for g in genes_list],
            }
        tokenized_data["cell_idx"] = cell_idx

        
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed)

        data_loader = DataLoader(
                    dataset= SeqDataset(tokenized_data),
                    batch_size=self.cfg.data.preprocess.batch_size,
                    num_workers=self.cfg.buffer.num_workers,
                    pin_memory=True,
                    shuffle=shuffle,
                    collate_fn=geneformer_collate_fn,
                    generator=generator
                )

        return data_loader

    def setup_hook(self, modify=False):
        """Register forward hooks for specified layer"""
        submodule = eval(f"self.model.{self.cfg.scfm.hook_layers}")
        self.handle = submodule.register_forward_hook(self.output_hook)
        if modify:
            self.output_hook.modify = True

    def forward(self, inputs, device_model, return_embeddings=False):
        max_len_batch = int(max(inputs["length"]))
        model_input_size = self.cfg.scfm.preprocessing.max_len
        input_data_minibatch = pu.pad_tensor_list(
        inputs['genes'], max_len_batch, self.cfg.scfm.preprocessing.pad_id, model_input_size)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_data_minibatch.to(device_model),
                attention_mask=pu.gen_attention_mask(inputs),
            )

        if return_embeddings:
            original_lens = torch.tensor(inputs["length"], device=device_model)
            if self.cfg.scfm.version == 'v2':
                # Last hidden layer (Geneformer's EmbExtractor default, emb_layer=-1, is the second-to-last).
                embs_i = outputs.hidden_states[-1]
                non_cls_embs = embs_i[:, 1:, :]
                mean_embs = pu.mean_nonpadding_embs(non_cls_embs, original_lens - 2) # remove cls and eos
            else:
                embs_i = outputs.hidden_states[-1]
                mean_embs = pu.mean_nonpadding_embs(embs_i, original_lens)
            return mean_embs.cpu()
        else:
            return

    def generate_activations(self, inputs, device_model):
        """Get activations for given inputs"""
        with self.output_hook:
            self.forward(inputs, device_model)
            return self.output_hook.outputs[0]
        
    def generate_embeddings(self, inputs, device_model, return_activations=False, modify_activations=False, x_hat=None):
        """Generate embeddings, optionally modifying activations"""
        try:
            if modify_activations:
                self.output_hook.modify = True
                self.output_hook.x_hat = x_hat

            with self.output_hook, seeded(self.cfg.seed):
                embeddings = self.forward(inputs, device_model, return_embeddings=True)
                
                if return_activations:
                    activations = self.output_hook.outputs[0]
                    return embeddings, activations
                else:
                    return embeddings
        finally:
            # Always reset modification state
            self.output_hook.modify = False
            self.output_hook.x_hat = None
    
    def generate_activation_mask(self, batch):
        """Generate mask for activations based on padding tokens"""
        max_len_batch = int(max(batch["length"]))
        original_lengths = batch["length"]
        positions = torch.arange(max_len_batch).unsqueeze(0)
        lengths = torch.tensor(original_lengths).unsqueeze(1)
        mask = positions < lengths

        # For Geneformer v2, exclude CLS (pos 0) and EOS (pos length-1) from the mask.
        # This aligns token sampling + SAE reconstruction with how cell embeddings are computed
        # (mean pooling over non-special tokens).
        if self.cfg.scfm.version == "v2":
            # positions > 0 removes CLS
            mask = mask & (positions > 0)
            # positions < (lengths - 1) removes EOS for each sequence
            mask = mask & (positions < (lengths - 1))
        
        return mask

    def prepare_compression(self, batch, features, n_genes, device):
        """Prepare data to be saved to disk compressed
        
        Returns tuple in order specified by cfg.scfm.compression.datasets:
        (features, genes) for Geneformer
        """
        mask = self.generate_activation_mask(batch)
        genes = batch["genes"]

        features[~mask] = 0

        # Pad to max length
        features = torch.nn.functional.pad(features, (0, 0, 0, n_genes - features.shape[1]), value=0)
        genes = pad_sequence(genes, batch_first=True, padding_value=-1)
        genes = torch.nn.functional.pad(genes, (0, n_genes - genes.shape[1]), value=-1)

        # Convert to CPU numpy arrays with dtypes from config
        features_cpu = features.cpu().numpy().astype(self.cfg.scfm.compression.datasets[0].dtype)
        genes_cpu = genes.cpu().numpy().astype(self.cfg.scfm.compression.datasets[1].dtype)

        return features_cpu, genes_cpu

    def id2gene(self, ids):
        """Convert list of gene ids to gene names"""
        if not hasattr(self, 'ensembl_dict') or self.vocab is None:
            self._load_additional_files()
            self.vocab = self._load_vocab()
        inv_ensembl_dict = {v: k for k, v in self.ensembl_dict.items()}
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return [inv_ensembl_dict.get(inv_vocab[i], inv_vocab[i]) for i in ids]  # special tokens keep their name

    def special_token_ids(self) -> set:
        """Token ids that are not genes (<pad>, <mask>, <cls>, <eos>)."""
        if self.vocab is None:
            self.vocab = self._load_vocab()
        return {i for token, i in self.vocab.items() if str(token).startswith("<")}

    def clear_hooks(self, type = Literal['save','modify']):
        """Remove registered hook"""
        if hasattr(self, 'handle') and self.handle is not None:
            self.handle.remove()
    
    def __del__(self):
        self.clear_hooks()