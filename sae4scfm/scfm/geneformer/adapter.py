from pathlib import Path
from typing import Literal
import torch
import pickle
from omegaconf import DictConfig, OmegaConf
import numpy as np
import scipy.sparse as sp
import scanpy as sc
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from geneformer import TOKEN_DICTIONARY_FILE_30M, TOKEN_DICTIONARY_FILE, GENE_MEDIAN_FILE_30M, GENE_MEDIAN_FILE, ENSEMBL_DICTIONARY_FILE_30M, ENSEMBL_DICTIONARY_FILE, ENSEMBL_MAPPING_FILE_30M, ENSEMBL_MAPPING_FILE
from geneformer import perturber_utils as pu
from geneformer.tokenizer import rank_genes

from ..base import ModelAdapter
from sae4scfm.core.utils import OutputHook, SeqDataset, set_seed



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
        with open(eval(self.cfg.scfm.vocab_file), "rb") as f:
            vocab = pickle.load(f)
        return vocab
    
    def _load_additional_files(self):
        with open(eval(self.cfg.scfm.gene_median_file), "rb") as f:
            self.gene_median_dict = pickle.load(f)

        with open(eval(self.cfg.scfm.ensembl_dictionary_file), "rb") as f:
            self.ensembl_dict = pickle.load(f)

    def preprocess_data(self, adata, shuffle=True):
        """Preprocess AnnData and create DataLoaders"""
        self._load_additional_files()

        # if ensembl_id not in adata.var, map gene names to ensembl ids
        if 'ensembl_id' not in adata.var.columns:
            # filter by existing match
            adata.var["id_in_vocab"] = [1 if gene in self.ensembl_dict.keys() else -1 for gene in adata.var["gene_name"]]
            adata = adata[:, adata.var["id_in_vocab"] >= 0]
            adata.var['ensembl_id'] = adata.var["gene_name"].map(self.ensembl_dict)

        # filter genes by those in vocab
        adata.var["id_in_vocab"] = [1 if gene in self.vocab.keys() else -1 for gene in adata.var["ensembl_id"]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        gene_col = "ensembl_id"

        if self.cfg.data.preprocess.subset_hvg:
            print('HVG', flush=True)
            sc.pp.highly_variable_genes(
                        adata,
                        layer=None,
                        n_top_genes=self.cfg.data.preprocess.subset_hvg,
                        batch_key=None,
                        flavor="seurat_v3" if self.cfg.data.preprocess.counts else "cell_ranger",
                        subset=True,
                    )

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

        # ensure no tokenized_cells are empty
        empty_cell_indices = [i for i, cell in enumerate(tokenized_cells) if cell.size == 0]
        if len(empty_cell_indices) > 0:
            print(
                "Warning: cells without any genes in token dictionary detected. This is unusual and may indicate empty droplets or otherwise invalid cells within the input data. Consider further QC prior to tokenization. Proceeding with excluding empty cells."
            )
            empty_cell_indices.sort(reverse=True) # for safe deletion
            for index in empty_cell_indices:
                del tokenized_cells[index]


        if self.cfg.scfm.version == 'v2': # v2 uses a CLS token
            tokenized_data = {"genes": [
                torch.cat([torch.tensor([self.vocab.get("<cls>")], dtype=torch.int32),
                        torch.from_numpy(item).int()[:self.cfg.scfm.preprocessing.max_len - 2],
                        torch.tensor([self.vocab.get("<eos>")], dtype=torch.int32)])
                for item in tokenized_cells],
                                "length": [min(len(x), self.cfg.scfm.preprocessing.max_len) for x in tokenized_cells]
                            }
        else:
            tokenized_data = {"genes": [torch.from_numpy(item).int()[:self.cfg.scfm.preprocessing.max_len] for item in tokenized_cells],
                                "length": [min(len(x), self.cfg.scfm.preprocessing.max_len) for x in tokenized_cells]
                            }

        def collate_fn(batch):
            genes = [item["genes"] for item in batch]
            lengths = [item["length"] for item in batch]

            return {
                "genes": genes,
                "length": lengths
            }

        data_loader = DataLoader(
                    dataset= SeqDataset(tokenized_data),
                    batch_size=self.cfg.data.preprocess.batch_size,
                    num_workers= 0,
                    pin_memory=True,
                    shuffle=shuffle,
                    collate_fn=collate_fn
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

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_data_minibatch.to(device_model),
                attention_mask=pu.gen_attention_mask(inputs),
            )

        if return_embeddings:
            original_lens = torch.tensor(inputs["length"], device=device_model)
            if self.cfg.scfm.version == 'v2':
                embs_i = outputs.hidden_states[12]
                non_cls_embs = embs_i[:, 1:, :]
                mean_embs = pu.mean_nonpadding_embs(non_cls_embs, original_lens - 2) # remove cls and eos
            else:
                embs_i = outputs.hidden_states[6]
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

            with self.output_hook:
                set_seed(self.cfg.seed)
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
        return [inv_ensembl_dict[inv_vocab[i]] for i in ids]

    def clear_hooks(self, type = Literal['save','modify']):
        """Remove registered hook"""
        if hasattr(self, 'handle') and self.handle is not None:
            self.handle.remove()
    
    def __del__(self):
        self.clear_hooks()