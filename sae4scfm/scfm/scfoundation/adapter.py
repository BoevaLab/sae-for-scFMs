"""
scFoundation model adapter
"""
import pandas as pd
import torch
from pathlib import Path
from omegaconf import DictConfig
import numpy as np
from scipy.sparse import issparse
from typing import List, Literal
import scanpy as sc
from torch.utils.data import DataLoader

from .utils import gatherData, load_model_frommmf
from ..base import ModelAdapter
from sae4scfm.core.utils import OutputHook, SeqDataset, highly_variable_genes, seeded


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

        ckpt_path = self.model_dir  / self.cfg.scfm.model_file
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
        """Preprocess AnnData into scFoundation inputs and create a DataLoader.

        Expression is placed in vocab order, library sizes (the S token and the 1e4 scaling)
        are computed over all vocab genes, and only then is the input restricted to highly
        variable genes when subset_hvg is set. T and S are appended as the last two tokens.
        """
        adata.obs["_cell_idx"] = np.arange(adata.n_obs)
        hvg = None
        if self.cfg.data.preprocess.subset_hvg:
            highly_variable_genes(adata, self.cfg.data.preprocess.subset_hvg,
                                  "seurat_v3" if self.cfg.data.preprocess.counts else "cell_ranger", subset=False)
            hvg = set(adata.var["gene_name"][adata.var["highly_variable"]])

        # Reindex to vocab order, filling missing genes with zeros
        gene_to_vocab_idx = {gene: i for i, gene in enumerate(self.vocab)}
        data_indices, vocab_indices = [], []
        for i, gene in enumerate(adata.var["gene_name"].tolist()):
            if gene in gene_to_vocab_idx:
                data_indices.append(i)
                vocab_indices.append(gene_to_vocab_idx[gene])

        X_reordered = np.zeros((adata.n_obs, len(self.vocab)), dtype=np.float32)
        X_subset = adata.X[:, data_indices]
        X_reordered[:, vocab_indices] = X_subset.toarray() if issparse(X_subset) else X_subset
        cell_idx = adata.obs["_cell_idx"].to_numpy()
        del adata, X_subset

        # If data is already normalized + log1p-transformed, reverse log1p and
        # rescale each cell so its smallest nonzero value maps to 1, then round to
        # pseudo-integer counts. This gives row_sums that vary across cells
        # (capturing relative expression complexity) rather than all converging to
        # ~target_sum, which would make the S token nearly constant everywhere.
        # Per-cell normalization means true library sizes are unrecoverable, but
        # this is an alternative proxy.
        if not self.cfg.data.preprocess.counts:
            X_reordered = np.expm1(X_reordered)
            X_tmp = X_reordered.copy()
            X_tmp[X_tmp == 0] = np.inf
            min_nz = X_tmp.min(axis=1, keepdims=True)  # per-cell minimum nonzero
            del X_tmp
            safe_min = np.where(np.isinf(min_nz), 1.0, min_nz)
            X_reordered = np.round(X_reordered / safe_min).astype(np.float32)

        # Library size over all vocab genes, as in scFoundation
        row_sums = X_reordered.sum(axis=1, keepdims=True)
        row_sums_safe = row_sums.copy()
        row_sums_safe[row_sums_safe == 0] = 1  # Avoid division by zero
        if hvg is not None:
            X_reordered[:, [g not in hvg for g in self.vocab]] = 0
        X_reordered = np.log1p((X_reordered / row_sums_safe) * 1e4)

        # Add T and S columns (S uses same row_sums as normalization)
        total_counts = np.full((X_reordered.shape[0], 1), 4, dtype=np.float32)
        log_sums = np.log10(row_sums.flatten() + 1e-10).astype(np.float32).reshape(-1, 1)  # Small epsilon to avoid log(0)
        X_final = np.concatenate([X_reordered, total_counts, log_sums], axis=1)

        del X_reordered, row_sums, row_sums_safe  # Free memory

        # Gene IDs must match final matrix dimensions (vocab + T + S)
        data_gene_ids = np.arange(X_final.shape[1])
        n_vocab = len(self.vocab)

        # Tokenize the data
        max_len = (X_final > 0).sum(axis=1).max()
        max_len = min(int(max_len), int(self.cfg.scfm.preprocessing.max_len))
        pad_token = int(self.cfg.scfm.preprocessing.pad_value)
        tokenized_data = gatherData(X_final, data_gene_ids, max_len, pad_id=pad_token, pad_value=pad_token,
                                    always_keep=(n_vocab, n_vocab + 1))
        tokenized_data["cell_idx"] = torch.as_tensor(cell_idx, dtype=torch.long)

        data_loader = DataLoader(
                    dataset= SeqDataset(tokenized_data),
                    batch_size=self.cfg.data.preprocess.batch_size,
                    num_workers=self.cfg.buffer.num_workers,
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
        use_amp = str(device_model).startswith("cuda")
        with torch.inference_mode():
            input_gene_ids = inputs["genes"].to(device_model)
            input_values = inputs["values"].to(device_model)
            cutoff = (input_values != self.cfg.scfm.preprocessing.pad_value).sum(axis=1).max()
            input_gene_ids = input_gene_ids[:, :cutoff]
            input_values = input_values[:, :cutoff]
            src_key_padding_mask = input_values.eq(self.cfg.scfm.preprocessing.pad_value)

            # Use bf16 autocast on CUDA (A100-friendly). Avoid explicit fp32 upcasts inside.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                x = self.model.token_emb(torch.unsqueeze(input_values, 2), output_weight=0)
                position_emb = self.model.pos_emb(input_gene_ids)
                x += position_emb
                if return_embeddings:
                    geneemb = self.model.encoder(x, src_key_padding_mask)
                else:
                    self.model.encoder(x, src_key_padding_mask)
                    return

            # Cell embedding as in scFoundation: [S output, T output, max and mean over genes].
            # T and S are the last two real tokens; the max and mean exclude them.
            seq_lens = (~src_key_padding_mask).sum(dim=1)  # number of real tokens per sample
            geneemb1 = geneemb[torch.arange(geneemb.size(0)), seq_lens - 1, :]  # S
            geneemb2 = geneemb[torch.arange(geneemb.size(0)), seq_lens - 2, :]  # T

            positions = torch.arange(geneemb.size(1), device=geneemb.device)
            is_gene = (positions[None, :] < (seq_lens - 2)[:, None]).unsqueeze(-1)
            geneemb3, _ = geneemb.masked_fill(~is_gene, float('-inf')).max(dim=1)
            geneemb4 = (geneemb * is_gene).sum(dim=1) / is_gene.sum(dim=1).clamp(min=1)

            geneembmerge = torch.cat([geneemb1, geneemb2, geneemb3, geneemb4], dim=1)
            return geneembmerge
            
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
        
    def id2gene(self, genes: List[int]) -> List[str]:
        """Gene names for token ids; the T and S tokens (ids len(vocab), len(vocab) + 1) map to "T" and "S"."""
        if self.vocab is None:
            self.vocab = self._load_vocab()
        names = list(self.vocab) + ["T", "S"]
        return [names[int(g)] for g in genes]

    def special_token_ids(self) -> set:
        """Token ids that are not genes: the appended T and S tokens."""
        if self.vocab is None:
            self.vocab = self._load_vocab()
        return {len(self.vocab), len(self.vocab) + 1}
    
    def clear_hooks(self, type = Literal['save','modify']):
        """Remove registered hook"""
        if hasattr(self, 'handle') and self.handle is not None:
            self.handle.remove()
    
    def __del__(self):
        self.clear_hooks()