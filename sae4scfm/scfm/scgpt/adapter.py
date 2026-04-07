"""
scGPT model adapter
"""
import torch
import json
from pathlib import Path
import copy
from omegaconf import DictConfig, OmegaConf
import numpy as np
from scipy.sparse import issparse
from torch.utils.data import DataLoader
from typing import List, Literal
from scanpy.get import _get_obs_rep

from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch
from scgpt.utils import load_pretrained
from scgpt.model import TransformerModel

from ..base import ModelAdapter
from sae4scfm.core.utils import (HVG_SPANS, GroupedBatchSampler, OutputHook, SeqDataset, highly_variable_genes,
                                 loess_singular, seeded, subset_to_gene_list)

BATCH_MAPPING_FILE = "batch_mapping.json"


def load_vocab(vocab_file, special_tokens=("<pad>", "<cls>", "<eoc>"), pad_token="<pad>") -> GeneVocab:
    """scGPT gene vocabulary with the special tokens added; unknown genes map to the pad token."""
    vocab = GeneVocab.from_file(vocab_file)
    for token in special_tokens:
        if token not in vocab:
            vocab.append_token(token)
    vocab.set_default_index(vocab[pad_token])
    return vocab


def use_masked_avg_pool(model: TransformerModel) -> TransformerModel:
    """Make scGPT's `avg-pool` cell embedding average over real tokens only.

    Upstream `_get_cell_emb_from_layer` takes `torch.mean(layer_output, dim=1)`, which also
    averages the padding positions (cells are padded to the longest cell of the dataset).
    Used for inference and in scripts/scgpt_finetune.py.
    """
    forward = model.forward

    def forward_with_mask(*args, **kwargs):
        model._padding_mask = kwargs["src_key_padding_mask"] if "src_key_padding_mask" in kwargs else args[2]
        return forward(*args, **kwargs)

    def masked_mean(layer_output, weights=None):
        keep = (~model._padding_mask).unsqueeze(-1).to(layer_output.dtype)
        return (layer_output * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)

    model.forward = forward_with_mask
    model._get_cell_emb_from_layer = masked_mean
    return model

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
        state_dict = torch.load(self.model_dir / self.cfg.scfm.model_file, map_location='cpu')

        model_config = self._build_model_config(state_dict)
        self.model = TransformerModel(**model_config)
        self._load_pretrained_weights(state_dict)
        if model_config.get("cell_emb_style") == "avg-pool":
            use_masked_avg_pool(self.model)

        self.model.eval()
        return self.model

    def _build_model_config(self, state_dict: dict) -> dict:
        """Assemble the scGPT model kwargs from Hydra config and checkpoint."""
        model_config = OmegaConf.to_container(self.cfg.scfm.model, resolve=True)

        # Add vocab-dependent params
        model_config['ntoken'] = len(self.vocab)
        model_config['vocab'] = self.vocab

        # num_batch_labels must match the pretrained weights whenever any
        # batch-aware component is enabled, so infer it from the checkpoint.
        if (model_config.get('use_batch_labels')
                or model_config.get('domain_spec_batchnorm')
                or model_config.get('do_dab')):
            num_batch_labels = self._infer_num_batch_labels(state_dict)
            if num_batch_labels is None:
                raise ValueError(
                    "Batch-aware components are enabled (use_batch_labels/"
                    "domain_spec_batchnorm/do_dab) but num_batch_labels could "
                    "not be inferred from the checkpoint."
                )
            model_config['num_batch_labels'] = num_batch_labels

        return model_config

    @staticmethod
    def _infer_num_batch_labels(state_dict: dict):
        """Infer number of batch labels from a scGPT checkpoint.

        Prefers the batch embedding table shape, falling back to the count of
        domain-specific batchnorm sub-layers.
        """
        emb = state_dict.get('batch_encoder.embedding.weight')
        if emb is not None:
            return int(emb.shape[0])

        dsbn_indices = {
            int(key.split('.')[2])
            for key in state_dict
            if key.startswith('dsbn.bns.')
        }
        if dsbn_indices:
            return max(dsbn_indices) + 1

        return None
    
    def _load_vocab(self) -> GeneVocab:
        return load_vocab(self.model_dir / self.cfg.scfm.vocab_file, self.cfg.scfm.preprocessing.special_tokens,
                          self.cfg.scfm.preprocessing.pad_token)
    
    def _load_model_config(self) -> dict:
        """Load model architecture config from JSON"""
        config_file = self.model_dir / self.cfg.scfm.config_file
        with open(config_file, "r") as f:
            return json.load(f)
    
    def _load_pretrained_weights(self, state_dict: dict):
        """Load pretrained weights, optionally randomizing some layers"""
        # Optional: randomize transformer encoder
        if self.cfg.scfm.get('random_init', False):
            state_dict_random = copy.deepcopy(self.model.state_dict())
            for key in state_dict.keys():
                if key.startswith('transformer_encoder'):
                    state_dict[key] = state_dict_random[key]
        
        load_pretrained(self.model, state_dict, verbose=False)
    
    def preprocess_data(self, adata, shuffle=True):
        """Preprocess AnnData and create DataLoaders"""

        gene_col = "gene_name"
        batch_col = "batch"
        adata.obs["_cell_idx"] = np.arange(adata.n_obs)  # position in the input, kept through filtering

        # Filter genes from list
        adata = subset_to_gene_list(adata, self.cfg.data.preprocess.get("subset_hvg", False), self.cfg.data)

        # Keep only genes in vocab
        adata.var["id_in_vocab"] = [1 if gene in self.vocab else -1 for gene in adata.var["gene_name"]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        subset_hvg = self.cfg.data.preprocess.subset_hvg
        if isinstance(subset_hvg, str):
            subset_hvg = False

        # Preprocess
        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=self.cfg.data.preprocess.filter_gene_by_counts,  # step 1
            filter_cell_by_counts=self.cfg.data.preprocess.filter_cell_by_counts,  # step 2
            normalize_total=self.cfg.data.preprocess.normalize_total,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=self.cfg.data.preprocess.log1p,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=subset_hvg,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if self.cfg.data.preprocess.counts else "cell_ranger",
            binning=False,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
        try:
            preprocessor(adata)#, batch_key=batch_col)
        except ValueError as e:  # seurat_v3's loess fit was singular: HVGs with a wider span, then the rest
            if not subset_hvg or not loess_singular(e):
                raise
            highly_variable_genes(adata, subset_hvg, preprocessor.hvg_flavor, subset=True, spans=HVG_SPANS[1:])
            preprocessor.subset_hvg = False
            preprocessor(adata)
        if self.cfg.data.preprocess.binning:
            self._binning(adata, self.cfg.data.preprocess.binning)

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
            include_zero_gene = bool(self.cfg.data.preprocess.get("include_zero_gene", False))
        )

        tokenized_data['cell_idx'] = torch.as_tensor(adata.obs["_cell_idx"].to_numpy(), dtype=torch.long)
        batch_aware = self.cfg.scfm.model.get("use_batch_labels") or self.cfg.scfm.model.get("domain_spec_batchnorm")
        if batch_aware:
            tokenized_data['batch_labels'] = torch.as_tensor(self._batch_ids(adata.obs[batch_col]), dtype=torch.long)
        else:
            tokenized_data['batch_labels'] = torch.LongTensor(adata.obs[batch_col].astype("category").cat.codes.values)

        loader_kwargs = dict(num_workers=self.cfg.buffer.get("num_workers", 0), pin_memory=True)
        if self.cfg.scfm.model.get("domain_spec_batchnorm"):
            # scGPT's DSBN normalizes a whole minibatch with the batchnorm of its first cell's
            # batch, so every minibatch must come from a single batch (as in finetuning).
            sampler = GroupedBatchSampler(tokenized_data['batch_labels'].numpy(),
                                          self.cfg.data.preprocess.batch_size, shuffle=shuffle)
            data_loader = DataLoader(dataset=SeqDataset(tokenized_data), batch_sampler=sampler, **loader_kwargs)
        else:
            data_loader = DataLoader(dataset=SeqDataset(tokenized_data), batch_size=self.cfg.data.preprocess.batch_size,
                                     shuffle=shuffle, **loader_kwargs)

        return data_loader

    def _batch_ids(self, batches) -> np.ndarray:
        """Map batch names to the ids the model was finetuned with (batch_mapping.json)."""
        path = self.model_dir / BATCH_MAPPING_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. This scGPT model uses batch labels, and their ids must match "
                f"finetuning: finetune with scripts/scgpt_finetune.py, which saves {BATCH_MAPPING_FILE}.")
        mapping = json.loads(path.read_text())
        unknown = sorted(set(map(str, batches)) - set(mapping))
        if unknown:
            raise ValueError(f"Batches {unknown[:5]} were not seen during finetuning ({path}).")
        return np.array([mapping[str(b)] for b in batches])

    def _binning(self, adata, n_bins):
        key_to_process = 'X' # needs to be changed if normalizing or log1p
        result_binned_key="X_binned"
        adata.layers['X'] = adata.X
        layer_data = _get_obs_rep(adata, layer=key_to_process)
        n_cells, n_genes = layer_data.shape

        binned_rows = np.zeros((n_cells, n_genes), dtype=np.uint8)
        bin_edges = np.zeros((n_cells, n_bins))

        for i in range(n_cells):
            row = layer_data.getrow(i).toarray().ravel() if issparse(layer_data) else layer_data[i]
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            if len(non_zero_row) == 0:
                continue
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            digits = self._digitize(non_zero_row, bins)
            binned_rows[i, non_zero_ids] = digits
            bin_edges[i] = np.concatenate([[0], bins])
        adata.layers[result_binned_key] = binned_rows
        adata.obsm["bin_edges"] = bin_edges

    def _digitize(self, x: np.ndarray, bins: np.ndarray, side="both") -> np.ndarray:
        assert x.ndim == 1 and bins.ndim == 1
        left_digits = np.digitize(x, bins)
        if side == "one":
            return left_digits
        right_difits = np.digitize(x, bins, right=True)
        rands = np.random.rand(len(x))  # uniform random numbers
        digits = rands * (right_difits - left_digits) + left_digits
        digits = np.ceil(digits).astype(np.int64)
        return digits

    def setup_hook(self, modify=False):
        """Register forward hooks for specified layer"""
        submodule = eval(f"self.model.{self.cfg.scfm.hook_layers}")
        self.handle = submodule.register_forward_hook(self.output_hook)
        if modify:
            self.output_hook.modify = True
    
    def forward(self, inputs, device_model, return_embeddings=False):
        """Run model forward pass"""
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
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

    def special_token_ids(self) -> set:
        """Token ids that are not genes (<pad>, <cls>, <eoc>)."""
        if self.vocab is None:
            self.vocab = self._load_vocab()
        return {self.vocab[t] for t in self.cfg.scfm.preprocessing.special_tokens if t in self.vocab}

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