"""
scFoundation model adapter
"""
import pandas as pd
import torch
from pathlib import Path
import copy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
import os
from typing import Literal
import scanpy as sc
from torch.utils.data import DataLoader

from .utils import gatherData, load_model_frommmf
from ..base import ModelAdapter
from sae4scfm.core.utils import OutputHook, SeqDataset, set_seed


class scFoundationAdapter(ModelAdapter):
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

        ckpt_path = self.model_dir / 'models.ckpt'
        key = 'cell'
        self.model, pretrainconfig = load_model_frommmf(ckpt_path, key)
        self.model.eval()

        return self.model

    def _load_vocab(self) -> list:
        vocab_file = self.model_dir / self.cfg.scfm.vocab_file
        gene_list_df = pd.read_csv(vocab_file, header=0, delimiter='\t')
        gene_list = list(gene_list_df['gene_name'])
        #gene_list.append('T')
        #gene_list.append('S')
        return gene_list

    def preprocess_data(self, adata, shuffle=True):
        """Preprocess AnnData and create DataLoader"""
        # TODO: vocab ??
        if self.cfg.data.preprocess.subset_hvg:
            sc.pp.highly_variable_genes(
                        adata,
                        layer=None,
                        n_top_genes=self.cfg.data.preprocess.subset_hvg,
                        batch_key=None,
                        flavor="seurat_v3" if self.cfg.data.preprocess.counts else "cell_ranger",
                        subset=True,
                    )
        
        # Memory-efficient approach: work with sparse matrix as long as possible
        col = adata.var["gene_name"].tolist()
        
        # Reindex to vocab order, filling missing genes with zeros (stays sparse)
        vocab_set = set(self.vocab)
        
        # Find indices of genes that exist in both vocab and data
        gene_to_vocab_idx = {gene: i for i, gene in enumerate(self.vocab)}
        col_to_data_idx = {gene: i for i, gene in enumerate(col)}
        
        # Create mapping for reordering
        data_indices = []
        vocab_indices = []
        for gene in col:
            if gene in vocab_set:
                data_indices.append(col_to_data_idx[gene])
                vocab_indices.append(gene_to_vocab_idx[gene])
        
        # Work with sparse matrix directly - only convert needed columns
        X_reordered = np.zeros((adata.n_obs, len(self.vocab)), dtype=np.float32)
        if issparse(adata.X):
            X_subset = adata.X[:, data_indices].toarray()
        else:
            X_subset = adata.X[:, data_indices]
        X_reordered[:, vocab_indices] = X_subset
        
        del adata, X_subset  # Free memory immediately
        
        # Normalize using vocab gene sums (same as original logic)
        row_sums = X_reordered.sum(axis=1, keepdims=True)
        row_sums_safe = row_sums.copy()
        row_sums_safe[row_sums_safe == 0] = 1  # Avoid division by zero
        X_reordered = np.log1p((X_reordered / row_sums_safe) * 1e4)
        
        # Add T and S columns (S uses same row_sums as normalization)
        total_counts = np.full((X_reordered.shape[0], 1), 4, dtype=np.float32)
        log_sums = np.log10(row_sums.flatten() + 1e-10).astype(np.float32).reshape(-1, 1)  # Small epsilon to avoid log(0)
        X_final = np.concatenate([X_reordered, total_counts, log_sums], axis=1)
        
        del X_reordered, row_sums, row_sums_safe  # Free memory
        
        # Gene IDs must match final matrix dimensions (vocab + T + S)
        data_gene_ids = torch.arange(X_final.shape[1], device='cpu')
        
        # Tokenize the data
        max_len = (X_final > 0).sum(axis=1).max()
        tokenized_data = gatherData(X_final, np.array(data_gene_ids.ravel()), max_len, pad_id=103, pad_value=103)
        
        data_loader = DataLoader(
                    dataset= SeqDataset(tokenized_data),
                    batch_size=self.cfg.data.preprocess.batch_size,
                    num_workers=0,
                    pin_memory=True,
                    shuffle=shuffle
                )

        return data_loader

    def setup_hook(self, modify=False):
        """Register forward hooks for specified layer"""
        submodule = eval(f"self.model.{self.cfg.scfm.hook_layers}")
        self.handle = submodule.register_forward_hook(self.output_hook)
        if modify:
            self.output_hook.modify = True

    def forward(self, inputs, device_model, return_embeddings=False):
        """Run model forward pass"""
        with torch.no_grad():
            input_gene_ids = inputs["genes"].to(device_model) 
            input_values = inputs["values"].to(device_model)
            cutoff = (input_values != self.cfg.scfm.preprocessing.pad_value).sum(axis=1).max()
            input_gene_ids = input_gene_ids[:, :cutoff]
            input_values = input_values[:, :cutoff]
            src_key_padding_mask = input_values.eq(self.cfg.scfm.preprocessing.pad_value)
            x = self.model.token_emb(torch.unsqueeze(input_values, 2).float(), output_weight = 0)
            position_emb = self.model.pos_emb(input_gene_ids)
            x += position_emb
            if return_embeddings:
                geneemb = self.model.encoder(x, src_key_padding_mask)
                seq_lens = (~src_key_padding_mask).sum(dim=1)  # number of real tokens per sample
                geneemb1 = geneemb[torch.arange(geneemb.size(0)), seq_lens - 1, :]  # last real
                geneemb2 = geneemb[torch.arange(geneemb.size(0)), seq_lens - 2, :]  # second last real
                
                mask = src_key_padding_mask.unsqueeze(-1)
                nonpad = (~mask).float()  # 1 for real, 0 for pad
                masked_geneemb = geneemb.masked_fill(mask, float('-inf'))
                geneemb3, _ = masked_geneemb.max(dim=1)

                sum_emb = (geneemb * nonpad).sum(dim=1)
                count = nonpad.sum(dim=1)  # (batch, 1)
                geneemb4 = sum_emb / count.clamp(min=1)  # avoid div/0

                geneembmerge = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
                return geneembmerge
            else:
                self.model.encoder(x, src_key_padding_mask)
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
        cutoff = (batch["values"] != self.cfg.scfm.preprocessing.pad_value).sum(axis=1).max()
        mask = batch['values'] != self.cfg.scfm.preprocessing.pad_value
        return mask[:, :cutoff]

    def prepare_compression(self, batch, features, n_genes, device):
        """Prepare data to be saved to disk compressed
        
        Returns tuple in order specified by cfg.scfm.compression.datasets:
        (features, values, genes) for scFoundation
        """
        mask = self.generate_activation_mask(batch).to(device)
        values = batch["values"].to(device)
        genes = batch["genes"].to(device)  

        # TODO: ???
        cutoff = mask.sum(axis=1).max()
        values = values[:, :cutoff]
        genes = genes[:, :cutoff]

        features[~mask] = 0
        values[~mask] = 0
        genes[~mask] = -1

        # Pad to max length
        features = torch.nn.functional.pad(features, (0, 0, 0, n_genes - features.shape[1]), value=0)
        values = torch.nn.functional.pad(values, (0, n_genes - values.shape[1]), value=0)
        genes = torch.nn.functional.pad(genes, (0, n_genes - genes.shape[1]), value=-1)

        # Convert to CPU numpy arrays with dtypes from config
        features_cpu = features.cpu().numpy().astype(self.cfg.scfm.compression.datasets[0].dtype)
        values_cpu = values.cpu().numpy().astype(self.cfg.scfm.compression.datasets[1].dtype)
        genes_cpu = genes.cpu().numpy().astype(self.cfg.scfm.compression.datasets[2].dtype)

        return features_cpu, values_cpu, genes_cpu
        
    def clear_hooks(self, type = Literal['save','modify']):
        """Remove registered hook"""
        if hasattr(self, 'handle') and self.handle is not None:
            self.handle.remove()
    
    def __del__(self):
        self.clear_hooks()