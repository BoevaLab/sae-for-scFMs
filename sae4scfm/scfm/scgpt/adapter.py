"""
scGPT model adapter
"""
import torch
import json
from pathlib import Path
import copy
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse
from torch.utils.data import DataLoader
import os
from typing import List, Literal

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.utils import load_pretrained
from scgpt.model import TransformerModel

from ..base import ModelAdapter
from sae4scfm.core.utils import OutputHook, SeqDataset, set_seed

class scGPTAdapter(ModelAdapter):
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
        
        # Override with Hydra config values
        model_config = OmegaConf.to_container(self.cfg.scfm.model, resolve=True)

        # Add vocab-dependent params
        model_config['ntoken'] = len(self.vocab)
        model_config['vocab'] = self.vocab
        
        # Instantiate model
        self.model = TransformerModel(**model_config)
        
        # Load pretrained weights
        self._load_pretrained_weights()
        
        self.model.eval()
        return self.model
    
    def _load_vocab(self) -> GeneVocab:
        """Load and setup vocabulary"""
        vocab_file = self.model_dir / self.cfg.scfm.vocab_file
        vocab = GeneVocab.from_file(vocab_file)
        
        # Add special tokens
        for token in self.cfg.scfm.preprocessing.special_tokens:
            if token not in vocab:
                vocab.append_token(token)
        
        vocab.set_default_index(vocab[self.cfg.scfm.preprocessing.pad_token])
        return vocab
    
    def _load_model_config(self) -> dict:
        """Load model architecture config from JSON"""
        config_file = self.model_dir / self.cfg.scfm.config_file
        with open(config_file, "r") as f:
            return json.load(f)
    
    def _load_pretrained_weights(self):
        """Load pretrained weights, optionally randomizing some layers"""
        model_file = self.model_dir / self.cfg.scfm.model_file
        state_dict = torch.load(model_file, map_location='cpu')
        
        # Optional: randomize transformer encoder
        if self.cfg.get('random_init', False):
            state_dict_random = copy.deepcopy(self.model.state_dict())
            for key in state_dict.keys():
                if key.startswith('transformer_encoder'):
                    state_dict[key] = state_dict_random[key]
        
        load_pretrained(self.model, state_dict, verbose=False)
    
    def preprocess_data(self, adata, shuffle=True):
        """Preprocess AnnData and create DataLoaders"""

        gene_col = "gene_name"
        batch_col = "batch"

        # Keep only genes in vocab
        adata.var["id_in_vocab"] = [1 if gene in self.vocab else -1 for gene in adata.var["gene_name"]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # Preprocess
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=self.cfg.data.preprocess.filter_gene_by_counts,  # step 1
            filter_cell_by_counts=self.cfg.data.preprocess.filter_cell_by_counts,  # step 2
            normalize_total=self.cfg.data.preprocess.normalize_total,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=self.cfg.data.preprocess.log1p,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=self.cfg.data.preprocess.subset_hvg,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if self.cfg.data.preprocess.counts else "cell_ranger",
            binning=self.cfg.data.preprocess.binning,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        preprocessor(adata, batch_key=None)

        # Genes, expressions
        input_layer_key = 'X_binned'
        count_matrix = np.float32(
            adata.layers[input_layer_key].A
            if issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )
        genes = adata.var[gene_col].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)

        # Tokenize the data
        tokenized_data = tokenize_and_pad_batch(
            count_matrix,
            gene_ids,
            max_len = self.cfg.scfm.preprocessing.max_len,
            vocab=self.vocab,
            pad_token = self.cfg.scfm.preprocessing.pad_token,
            pad_value = self.cfg.scfm.preprocessing.pad_value,
            append_cls = False,
            include_zero_gene = False,
        )

        tokenized_data['batch_labels'] = torch.LongTensor(adata.obs[batch_col].astype("category").cat.codes.values)

        data_loader = DataLoader(
                    dataset= SeqDataset(tokenized_data),
                    batch_size=self.cfg.data.preprocess.batch_size,
                    num_workers=0,
                    pin_memory=True,
                    shuffle= shuffle
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
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            input_gene_ids = inputs["genes"].to(device_model)
            input_values = inputs["values"].to(device_model)
            src_key_padding_mask = input_values.eq(self.cfg.scfm.preprocessing.pad_value)
            if self.cfg.scfm.model.use_batch_labels:
                batch_labels = inputs["batch_labels"].to(device_model)
            else:
                batch_labels = None
            if return_embeddings:
                output_dict = self.model(input_gene_ids,
                                input_values,
                                src_key_padding_mask=src_key_padding_mask,
                                batch_labels = batch_labels)
                return output_dict["cell_emb"].cpu()
            else:
                self.model._encode(input_gene_ids,
                                   input_values,
                                   src_key_padding_mask=src_key_padding_mask,
                                   batch_labels = batch_labels)
    
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
        mask = batch['values'] != self.cfg.scfm.preprocessing.pad_value
        return mask

    def prepare_compression(self, batch, features, n_genes, device):
        """Prepare data to be saved to disk compressed
        
        Returns tuple in order specified by cfg.scfm.compression.datasets:
        (features, values, genes) for scGPT
        """
        mask = self.generate_activation_mask(batch).to(device)
        values = batch["values"].to(device)
        genes = batch["genes"].to(device)  

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
        if self.vocab is None:
            self.vocab = self._load_vocab()
        gene_names = self.vocab.lookup_tokens(genes)
        return gene_names

    def clear_hooks(self, type = Literal['save','modify']):
        """Remove registered hook"""
        if hasattr(self, 'handle') and self.handle is not None:
            self.handle.remove()
    
    def __del__(self):
        self.clear_hooks()