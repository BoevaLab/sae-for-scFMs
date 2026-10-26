"""
Script for finetuning scGPT
"""

import hydra
from omegaconf import OmegaConf

from pathlib import Path
import json
import logging
import torch
import numpy as np
import scipy
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import time
import os
import sys
import warnings
import wandb
import pandas as pd
import copy
import shutil
from functools import partial
import yaml

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from scgpt.loss import masked_mse_loss, criterion_neg_log_bernoulli
from scgpt.model import TransformerModel
from scgpt.utils import load_pretrained
from scgpt.preprocess import Preprocessor
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt import SubsetsBatchSampler

from sae4scfm.core.data_loader import create_data_loader, partitioned
from sae4scfm.core.io_utils import load_adata
from sae4scfm.core.utils import SeqDataset, get_devices, set_seed, subset_to_gene_list
from sae4scfm.scfm.scgpt.adapter import BATCH_MAPPING_FILE, load_vocab, use_masked_avg_pool

log = logging.getLogger(__name__)


def batch_mapping(batches) -> dict:
    """Deterministic batch name -> id mapping (sorted names); saved with the model for inference."""
    return {b: i for i, b in enumerate(sorted(set(map(str, batches))))}


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
def LoRA(model, device, rank, alpha):
    assign_lora = partial(LinearWithLoRA, rank=rank, alpha=alpha)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.cls_decoder.parameters():
        param.requires_grad = True
    for layer in model.transformer_encoder.layers:
        layer.linear1 = assign_lora(layer.linear1)
        layer.linear2 = assign_lora(layer.linear2)
        layer.self_attn.Wqkv = assign_lora(layer.self_attn.Wqkv)
        layer.self_attn.out_proj = assign_lora(layer.self_attn.out_proj)
    model.mvc_decoder.gene2query = assign_lora(model.mvc_decoder.gene2query)
    model.mvc_decoder.W = assign_lora(model.mvc_decoder.W)
    model.decoder.fc[0] = assign_lora(model.decoder.fc[0])
    model.decoder.fc[2] = assign_lora(model.decoder.fc[2])
    model.decoder.fc[4] = assign_lora(model.decoder.fc[4])
    model.to(device)
    return model

def define_wandb_metrics():
    wandb.define_metric("valid/gep", summary="min", step_metric="epoch")
    wandb.define_metric("valid/gepc", summary="min", step_metric="epoch")
    wandb.define_metric("valid/cls", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")

def load_model(cfg, vocab, device):
    model_dir = Path('data') / cfg.model.name
    model_file = model_dir / "best_model.pt"

    model = TransformerModel(
        ntoken = len(vocab),
        d_model = cfg.model.d_model,
        nhead = cfg.model.nhead,
        d_hid = cfg.model.d_hid,
        nlayers = cfg.model.nlayers,
        nlayers_cls = cfg.model.nlayers_cls,
        n_cls = cfg.model.n_cls,
        vocab = vocab,
        dropout = cfg.model.dropout,
        pad_token = cfg.preprocess.pad_token,
        pad_value = cfg.preprocess.pad_value,
        do_mvc = cfg.model.do_mvc,
        do_dab = cfg.model.do_dab,
        use_batch_labels = cfg.model.use_batch_labels,
        num_batch_labels = cfg.model.num_batch_labels,
        domain_spec_batchnorm = cfg.model.domain_spec_batchnorm,
        explicit_zero_prob = cfg.model.explicit_zero_prob,
        use_fast_transformer = cfg.model.use_fast_transformer,
        fast_transformer_backend = cfg.model.fast_transformer_backend,
        pre_norm = cfg.model.pre_norm,
        cell_emb_style = cfg.model.cell_emb_style,
    )
    load_pretrained(model, torch.load(model_file, map_location="cpu"), verbose=False)
    if cfg.model.cell_emb_style == "avg-pool":
        use_masked_avg_pool(model)  # average real tokens only (upstream also averages padding)
    model.to(device)
    return model


class ScGPTStreamingAdapter:
    """
    Adapter used by `sae4scfm.core.data_loader` to preprocess one `.h5ad` at a time
    and return a PyTorch DataLoader yielding scGPT finetuning batches.
    """

    def __init__(self, cfg, vocab, split: str = "train"):
        if cfg.model.use_batch_labels or cfg.model.do_cls:
            raise NotImplementedError(
                "Streaming finetuning supports the masked-expression objectives only: batch and cell-type "
                "labels would need one id mapping across all partitions.")
        self.cfg = cfg
        self.vocab = vocab
        if split not in {"train", "valid"}:
            raise ValueError(f"split must be 'train' or 'valid', got {split!r}")
        self.split = split

    def preprocess_data(self, adata, shuffle: bool = True):
        adata = subset_to_gene_list(adata, self.cfg.preprocess.get("subset_hvg", False), self.cfg.data)

        # keep only genes in vocab
        if "gene_name" not in adata.var:
            raise ValueError("Expected `adata.var['gene_name']` to exist for scGPT preprocessing.")
        adata.var["id_in_vocab"] = [1 if gene in self.vocab else -1 for gene in adata.var["gene_name"]]
        adata = adata[:, adata.var["id_in_vocab"] >= 0]
        log.info(f"After filtering genes. adata shape: {adata.shape}")

        # Preprocess and bin
        subset_hvg = self.cfg.preprocess.subset_hvg
        if isinstance(subset_hvg, str):
            subset_hvg = False

        preprocessor = Preprocessor(
            use_key="X",
            filter_gene_by_counts=self.cfg.preprocess.filter_gene_by_counts,
            filter_cell_by_counts=self.cfg.preprocess.filter_cell_by_counts,
            normalize_total=self.cfg.preprocess.normalize_total,
            result_normed_key="X_normed",
            log1p=self.cfg.preprocess.log1p,
            result_log1p_key="X_log1p",
            subset_hvg=subset_hvg,
            hvg_flavor="cell_ranger",
            binning=self.cfg.preprocess.binning,
            result_binned_key="X_binned",
        )
        preprocessor(adata, batch_key="batch" if self.cfg.model.use_batch_labels else None)
        log.info(f"After preprocessor (HVG/binning). adata shape: {adata.shape}")

        # Dense matrix for tokenization (partition-sized, not full dataset)
        input_layer_key = "X_binned"
        count_matrix = np.float32(
            adata.layers[input_layer_key].A
            if scipy.sparse.issparse(adata.layers[input_layer_key])
            else adata.layers[input_layer_key]
        )

        genes = adata.var["gene_name"].tolist()
        gene_ids = np.array(self.vocab(genes), dtype=int)

        celltypes_labels = adata.obs["celltype"].astype("category").cat.codes.to_numpy()
        if self.cfg.model.use_batch_labels:
            batch_ids = np.array(adata.obs["batch"].tolist())
            split_out = train_test_split(
                count_matrix,
                celltypes_labels,
                batch_ids,
                test_size=0.1,
                shuffle=True,
                random_state=self.cfg.seed,
            )
            train_data, valid_data, train_cell, valid_cell, train_batch, valid_batch = split_out
        else:
            split_out = train_test_split(
                count_matrix,
                celltypes_labels,
                test_size=0.1,
                shuffle=True,
                random_state=self.cfg.seed,
            )
            train_data, valid_data, train_cell, valid_cell = split_out
            train_batch = valid_batch = None

        if self.split == "train":
            data = train_data
            cell = train_cell
            batch = train_batch
        else:
            data = valid_data
            cell = valid_cell
            batch = valid_batch

        tokenized = tokenize_and_pad_batch(
            data,
            gene_ids,
            max_len=self.cfg.preprocess.max_seq_len,
            vocab=self.vocab,
            pad_token="<pad>",
            pad_value=self.cfg.preprocess.pad_value,
            append_cls=False,
            include_zero_gene=bool(self.cfg.preprocess.get("include_zero_gene", False)),
        )

        masked_values = random_mask_value(
            tokenized["values"],
            mask_ratio=self.cfg.preprocess.mask_ratio,
            mask_value=self.cfg.preprocess.mask_value,
            pad_value=self.cfg.preprocess.pad_value,
        )

        data_pt = {
            "gene_ids": tokenized["genes"],
            "values": masked_values,
            "target_values": tokenized["values"],
            "celltype_labels": torch.from_numpy(cell).long(),
        }
        if self.cfg.model.use_batch_labels and batch is not None:
            data_pt["batch_labels"] = torch.from_numpy(batch).long()

        return DataLoader(
            dataset=SeqDataset(data_pt),
            batch_size=self.cfg.preprocess.batch_size,
            shuffle=shuffle,
            num_workers=min(len(os.sched_getaffinity(0)), self.cfg.training.batch_size),
            pin_memory=True,
            drop_last=False,
        )

def preprocess_data(adata, cfg, vocab, test_size = 0.1):
    adata = subset_to_gene_list(adata, cfg.preprocess.get("subset_hvg", False), cfg.data)

    # Preprocess
    subset_hvg = cfg.preprocess.subset_hvg
    if isinstance(subset_hvg, str):
        subset_hvg = False

    preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=cfg.preprocess.filter_gene_by_counts,  # step 1
        filter_cell_by_counts=cfg.preprocess.filter_cell_by_counts,  # step 2
        normalize_total=cfg.preprocess.normalize_total,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=cfg.preprocess.log1p,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=subset_hvg,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3",#"seurat_v3" if cfg.preprocess.log1p else "cell_ranger",
        binning=cfg.preprocess.binning,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )
    preprocessor(adata, batch_key="batch" if cfg.model.use_batch_labels else None)
    log.info(f"After preprocessor (HVG/binning). adata shape: {adata.shape}")
    

    # Genes, expressions
    input_layer_key = 'X_binned'
    count_matrix = np.float32(
        adata.layers[input_layer_key].A
        if scipy.sparse.issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var['gene_name'].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    # cell labels
    celltypes_labels = np.array(adata.obs["celltype"].cat.codes.tolist())  # codes count from 0

    # batch labels
    if cfg.model.use_batch_labels:
        batch_ids = adata.obs["batch"].tolist()
        batch_to_int = batch_mapping(batch_ids)
        batch_ids = np.array([batch_to_int[str(b)] for b in batch_ids])
        #batch_ids = np.array(batch_ids)
        # Split and generate datasets
        (train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,) = train_test_split(
            count_matrix, celltypes_labels, batch_ids, test_size=test_size, shuffle=True, random_state=cfg.seed)

    else:
        # Split and generate datasets
        (train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels) = train_test_split(
            count_matrix, celltypes_labels,  test_size=test_size, shuffle=True, random_state=cfg.seed)
        train_batch_labels, valid_batch_labels = None, None
        #filter_test = adata.obs['donor_id'] == 'AP6'
        #train_data = count_matrix[~filter_test]
        #valid_data = count_matrix[filter_test]
        #train_celltype_labels = celltypes_labels[~filter_test]
        #valid_celltype_labels = celltypes_labels[filter_test]


    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len = cfg.preprocess.max_seq_len,
        vocab=vocab,
        pad_token = "<pad>",
        pad_value = cfg.preprocess.pad_value,
        append_cls = False,
        include_zero_gene=bool(cfg.preprocess.get("include_zero_gene", False)),
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len = cfg.preprocess.max_seq_len,
        vocab = vocab,
        pad_token = "<pad>",
        pad_value = cfg.preprocess.pad_value,
        append_cls = False,
        include_zero_gene=bool(cfg.preprocess.get("include_zero_gene", False)),
    )
    return tokenized_train, tokenized_valid, train_celltype_labels, valid_celltype_labels, train_batch_labels, valid_batch_labels
    
def mask_data(
    tokenized_train,
    tokenized_valid,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
    cfg):

    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=cfg.preprocess.mask_ratio,
        mask_value=cfg.preprocess.mask_value,
        pad_value=cfg.preprocess.pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=cfg.preprocess.mask_ratio,
        mask_value=cfg.preprocess.mask_value,
        pad_value=cfg.preprocess.pad_value,
    )

    if cfg.model.use_batch_labels:
        train_sort_ids = np.argsort(train_batch_labels)
        valid_sort_ids = np.argsort(valid_batch_labels)
        
        train_data_pt = {
            "gene_ids": tokenized_train["genes"][train_sort_ids],
            "values": masked_values_train[train_sort_ids],
            "target_values": tokenized_train["values"][train_sort_ids],
            "celltype_labels": torch.from_numpy(train_celltype_labels).long()[train_sort_ids],
            "batch_labels": torch.from_numpy(train_batch_labels).long()[train_sort_ids]
        }
        valid_data_pt = {
            "gene_ids": tokenized_valid["genes"][valid_sort_ids],
            "values": masked_values_valid[valid_sort_ids],
            "target_values": tokenized_valid["values"][valid_sort_ids],
            "celltype_labels": torch.from_numpy(valid_celltype_labels).long()[valid_sort_ids],
            "batch_labels": torch.from_numpy(valid_batch_labels).long()[valid_sort_ids]
        }
    
    else:
        train_data_pt = {
            "gene_ids": tokenized_train["genes"],
            "values": masked_values_train,
            "target_values": tokenized_train["values"],
            "celltype_labels": torch.from_numpy(train_celltype_labels).long()
        }
        valid_data_pt = {
            "gene_ids": tokenized_valid["genes"],
            "values": masked_values_valid,
            "target_values": tokenized_valid["values"],
            "celltype_labels": torch.from_numpy(valid_celltype_labels).long()
        }

    return train_data_pt, valid_data_pt

def prepare_dataloader(data_pt, cfg, intra_domain_shuffle=False):

    if cfg.model.use_batch_labels:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=SeqDataset(data_pt),
            batch_sampler=SubsetsBatchSampler(
                subsets,
                cfg.training.batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=False,
                drop_last=False,
            ),
            num_workers=min(len(os.sched_getaffinity(0)), cfg.training.batch_size),
            pin_memory=True,
        )
        return data_loader

    else:
        data_loader = DataLoader(
                dataset= SeqDataset(data_pt),
                batch_size=cfg.training.batch_size,
                num_workers=min(len(os.sched_getaffinity(0)), cfg.training.batch_size),
                pin_memory=True
            )
        return data_loader

def train(model, data_loader, criterion, criterion_cls_dab, scaler, optimizer, scheduler, cfg, device, log, epoch):
    model.train()
    total_loss, total_cls, total_gepc, total_gep, total_dab = 0.0, 0.0, 0.0, 0.0, 0.0
    log_interval = cfg.log_interval
    start_time = time.time()

    for batch, batch_data in enumerate(data_loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        if cfg.model.use_batch_labels:
            batch_labels = batch_data["batch_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(cfg.preprocess.pad_idx)
        with torch.cuda.amp.autocast(enabled=cfg.training.amp):
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask = src_key_padding_mask,
                batch_labels=batch_labels if cfg.model.use_batch_labels else None,
                MVC = cfg.model.do_mvc,
                CLS=cfg.model.do_cls
            )

            masked_positions = input_values.eq(cfg.preprocess.mask_value)  # the postions to predict
            loss = 0.0
            metrics_to_log = {}

            if cfg.preprocess.mask_ratio > 0:
                loss_gep = criterion(
                    output_dict["mlm_output"], target_values, masked_positions
                )
                loss = loss + loss_gep
                metrics_to_log.update({"train/gep": loss_gep.item()})

            if cfg.preprocess.mask_ratio > 0 and cfg.model.explicit_zero_prob:
                loss_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mlm_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_zero_log_prob
                metrics_to_log.update({"train/gep_nzlp": loss_zero_log_prob.item()})

            if cfg.model.do_mvc:
                loss_gepc = criterion(
                    output_dict["mvc_output"], target_values, masked_positions
                )
                loss = loss + loss_gepc
                metrics_to_log.update({"train/gepc": loss_gepc.item()})
            
            if cfg.model.do_mvc and cfg.model.explicit_zero_prob:
                loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
                    output_dict["mvc_zero_probs"], target_values, masked_positions
                )
                loss = loss + loss_gepc_zero_log_prob
                metrics_to_log.update({"train/gepc_nzlp": loss_gepc_zero_log_prob.item()})
    
            if cfg.model.do_cls:
                loss_cls = criterion_cls_dab(output_dict["cls_output"], celltype_labels)
                loss = loss + loss_cls
                metrics_to_log.update({"train/cls": loss_cls.item()})

            if cfg.model.do_dab:
                loss_dab = criterion_cls_dab(output_dict["dab_output"], batch_labels)
                loss = loss + cfg.model.dab_weight * loss_dab
                metrics_to_log.update({"train/dab": loss_dab.item()})

            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            with warnings.catch_warnings(record=True) as w:
                warnings.filterwarnings("always")
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    error_if_nonfinite=False if scaler.is_enabled() else True,
                )
                if len(w) > 0:
                    log.warning(
                        f"Found infinite gradient. This may be caused by the gradient "
                        f"scaler. The current scale is {scaler.get_scale()}. This warning "
                        "can be ignored if no longer occurs after autoscaling of the scaler."
                    )
            scaler.step(optimizer)
            scaler.update()

            wandb.log(metrics_to_log)

            total_loss += loss.item()
            total_gep += loss_gep.item() if cfg.preprocess.mask_ratio > 0 else 0.0
            total_cls += loss_cls.item() if cfg.model.do_cls else 0.0
            total_gepc += loss_gepc.item() if cfg.model.do_mvc else 0.0
            total_dab += loss_dab.item() if cfg.model.do_dab else 0.0

            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                cur_gep = total_gep / log_interval if cfg.preprocess.mask_ratio > 0 else 0.0
                cur_cls = total_cls / log_interval if cfg.model.do_cls else 0.0
                cur_gepc = total_gepc / log_interval if cfg.model.do_mvc else 0.0
                cur_dab = total_dab / log_interval if cfg.model.do_dab else 0.0

                log.info(
                    f"| epoch {epoch:3d} | {batch:3d} batches | "
                    f"lr {lr:05.5f} | ms/batch {ms_per_batch:5.2f} | "
                    f"loss {cur_loss:5.2f} | "
                    + (f"gep {cur_gep:5.2f} |" if cfg.preprocess.mask_ratio > 0 else "")
                    + (f"cls {cur_cls:5.2f} | " if cfg.model.do_cls else "")
                    + (f"gepc {cur_gepc:5.2f} |" if cfg.model.do_mvc else "")
                    + (f"dab {cur_dab:5.2f} |" if cfg.model.do_dab else "")
                )
                total_loss = 0
                total_gep = 0
                total_cls = 0
                total_gepc = 0
                total_dab = 0
                start_time = time.time()

def evaluate(model, data_loader, criterion, criterion_cls_dab, cfg, device, epoch):
    model.eval()
    total_gep = 0.0
    total_gepc = 0.0
    total_cls = 0.0
    total_dab = 0.0
    total_num = 0
    with torch.no_grad():
        for batch_data in data_loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            if cfg.model.use_batch_labels:
                batch_labels = batch_data["batch_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(cfg.preprocess.pad_idx)
            with torch.cuda.amp.autocast(enabled=cfg.training.amp):
                output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask = src_key_padding_mask,
                    batch_labels=batch_labels if cfg.model.use_batch_labels else None,
                    MVC = cfg.model.do_mvc,
                    CLS=cfg.model.do_cls
                )

                masked_positions = input_values.eq(cfg.preprocess.mask_value)  # the postions to predict

                if cfg.preprocess.mask_ratio > 0:
                    loss_gep = criterion(
                        output_dict["mlm_output"], target_values, masked_positions
                    )

                if cfg.model.do_mvc:
                    loss_gepc = criterion(
                        output_dict["mvc_output"], target_values, masked_positions
                    )

                if cfg.model.do_cls:
                    loss_cls = criterion_cls_dab(output_dict["cls_output"], celltype_labels)

                if cfg.model.do_dab:
                    loss_dab = criterion_cls_dab(output_dict["dab_output"], batch_labels)

            total_gep += loss_gep.item() * len(input_gene_ids) if cfg.preprocess.mask_ratio > 0 else 0.0
            total_gepc += loss_gepc.item() * len(input_gene_ids) if cfg.model.do_mvc else 0.0
            total_cls += loss_cls.item() * len(input_gene_ids) if cfg.model.do_cls else 0.0
            total_dab += loss_dab.item() * len(input_gene_ids) if cfg.model.do_dab else 0.0
            total_num += len(input_gene_ids)
            
    wandb.log(
        {
            "valid/gep": total_gep / total_num,
            "valid/gepc": total_gepc / total_num,
            "valid/cls": total_cls / total_num,
            "valid/dab": total_dab / total_num,
            "epoch": epoch,
        },
    )

    return total_gep / total_num, total_gepc / total_num, total_cls / total_num, total_dab / total_num


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="finetune_scgpt", version_base='1.3')
def finetune_scGPT(cfg):

    os.environ["WANDB_MODE"] = "offline"
    run = wandb.init(
    project="scGPT finetuning")
    
    set_seed(cfg.seed)
    device_model = get_devices()[0]
    
    # load vocab
    vocab = load_vocab(Path('data') / cfg.model.name / "vocab.json")
    cfg.preprocess.pad_idx = vocab[cfg.preprocess.pad_token]
    log.info("Vocab loaded")

    use_streaming = partitioned(cfg)

    # If using the shared streaming loader, ensure expected config fields exist.
    if use_streaming:
        # `sae4scfm.core.io_utils.load_adata` expects `cfg.data.file_path`.
        if not cfg.data.get("file_path", None):
            cfg.data.file_path = cfg.data.get("path", "")

        # Infer n_cls / num_batch_labels from the first partition if needed.
        if cfg.model.do_cls or cfg.model.use_batch_labels:
            if len(cfg.data.get("query_list", [])) == 0:
                raise ValueError("Streaming mode requires `cfg.data.query_list` to be set.")
            first_query = cfg.data.query_list[0]
            cfg.data.file_name = f"{first_query}/partition_0.h5ad"
            adata0 = load_adata(cfg)
            if cfg.model.do_cls:
                cfg.model.n_cls = int(pd.Series(adata0.obs["celltype"]).nunique())
            if cfg.model.use_batch_labels:
                cfg.model.num_batch_labels = int(pd.Series(adata0.obs["batch"]).nunique())
        log.info("Streaming data loader enabled")
    else:
        # Fallback: single in-memory dataset (legacy behavior)
        adata = load_adata(cfg)
        log.info("Data loaded")

        (tokenized_train, tokenized_valid,
         train_celltype_labels, valid_celltype_labels,
         train_batch_labels, valid_batch_labels) = preprocess_data(adata, cfg, vocab)
        log.info("Data processed")

        if cfg.model.do_cls:
            cfg.model.n_cls = pd.Series(train_celltype_labels).nunique()
        if cfg.model.use_batch_labels:
            cfg.model.num_batch_labels = pd.Series(train_batch_labels).nunique() if train_batch_labels is not None else 0

    # load model
    model = load_model(cfg, vocab, device_model)
    log.info("Model loaded")

    # LoRA finetuning
    if cfg.training.lora.do:
        model = LoRA(model, device_model, cfg.training.lora.rank, cfg.training.lora.alpha)
        log.info("LoRA initialized")

    # initialize
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)
    criterion_cls_dab = torch.nn.CrossEntropyLoss()
    criterion = masked_mse_loss

    # train
    best_val_loss = float("inf")
    best_model = model
    define_wandb_metrics()
    for epoch in range(1, cfg.training.epochs + 1):
        epoch_start_time = time.time()
        if use_streaming:
            # Stream partitions/files lazily using shared loader.
            train_adapter = ScGPTStreamingAdapter(cfg, vocab, split="train")
            valid_adapter = ScGPTStreamingAdapter(cfg, vocab, split="valid")

            train_stream = create_data_loader(cfg=cfg, adapter=train_adapter, test_mode=False)
            valid_stream = create_data_loader(cfg=cfg, adapter=valid_adapter, test_mode=True)

            train(model, train_stream, criterion, criterion_cls_dab, scaler, optimizer, scheduler, cfg, device_model, log, epoch)
            val_gep, val_gepc, val_cls, val_dab = evaluate(model, valid_stream, criterion, criterion_cls_dab, cfg, device_model, epoch)
        else:
            train_data_pt, valid_data_pt = mask_data(
                tokenized_train,
                tokenized_valid,
                train_celltype_labels,
                valid_celltype_labels,
                train_batch_labels,
                valid_batch_labels,
                cfg,
            )

            train_loader = prepare_dataloader(train_data_pt, cfg, intra_domain_shuffle=True)
            valid_loader = prepare_dataloader(valid_data_pt, cfg)

            train(model, train_loader, criterion, criterion_cls_dab, scaler, optimizer, scheduler, cfg, device_model, log, epoch)
            val_gep, val_gepc, val_cls, val_dab = evaluate(model, valid_loader, criterion, criterion_cls_dab, cfg, device_model, epoch)

        elapsed = time.time() - epoch_start_time
        log.info("-" * 89)
        log.info(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid gep {val_gep:5.4f} | gepc {val_gepc:5.4f} | cls {val_cls:5.4f} | dab {val_dab:5.4f}"
        )
        log.info("-" * 89)
        
        val_loss = val_gep + val_gepc + val_cls + val_dab
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            log.info(f"Best model with score {best_val_loss:5.4f}")

        scheduler.step()

    # SAVE
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), save_dir / "best_model.pt")
    for name in ("vocab.json", "args.json"):  # the folder then loads as a scfm model_dir
        shutil.copy(Path('data') / cfg.model.name / name, save_dir / name)
    if cfg.model.use_batch_labels and not use_streaming:
        # Batch ids the model was trained with; the scGPT adapter reads them at inference.
        with open(save_dir / BATCH_MAPPING_FILE, "w") as f:
            json.dump(batch_mapping(adata.obs["batch"]), f, indent=2)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.save_dir) / "CONFIG.yaml", 'w') as f:
        yaml.dump(cfg_dict, f)

    run.finish()
    wandb.finish()


if __name__ == "__main__":
    finetune_scGPT()