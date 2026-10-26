"""
Full encoder-decoder finetuning of scFoundation (xTrimoGene / MaeAutobin).

Implements the Read-Depth-Aware (RDA) pretraining objective described in the
Methods of Hao et al., Nature Methods (2024):

  1. Hierarchical Bayesian downsampling of raw counts to produce a
     low-total-count input vector x' with gamma = T_input / T_raw.
     With probability p_downsample, gamma ~ Uniform[gamma_min, gamma_max];
     otherwise gamma = 1 (input == raw). Per gene g, x'_g ~ Binomial(x_g, gamma).
  2. log1p(CPM) normalization of both raw and input vectors.
  3. Append count indicators T = log10(sum(raw)) and S = log10(sum(input))
     to obtain length-19,266 sequences.
  4. Random masking of 30% of the gene positions (zero and nonzero).
  5. Encoder receives only nonzero AND non-masked positions (plus T, S).
     Decoder receives all 19,266 positions (encoder output scattered into
     non-masked nonzero slots; mask embedding at masked slots; zero embedding
     at non-masked zero slots).
  6. MLP projects decoder outputs to per-gene scalars; MSE is computed only
     on masked positions, against the log1p(CPM) of the *raw* counts.
     Optionally restrict the loss to genes in the dataset that map to the vocab
     (see preprocess.restrict_loss_to_dataset_genes).
"""

import collections
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import hydra
import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from omegaconf import DictConfig, OmegaConf
from scipy.sparse import csr_matrix, issparse
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Make repository imports work regardless of cwd / launch mode
REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

# scFoundation model code (NO adapter)
from sae4scfm.scfm.scfoundation.scfoundation_model_files import select_model

# Repo data + util helpers
from sae4scfm.core.io_utils import load_adata
from sae4scfm.core.data_loader import create_data_loader, partitioned, create_multiprocess_data_loader
from sae4scfm.core.utils import set_seed

log = logging.getLogger(__name__)


# =============================================================================
# Distributed setup helpers
# =============================================================================

def setup_distributed() -> Tuple[bool, int, int, torch.device]:
    """Initialize torch.distributed if launched via torchrun, else single device."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, device
    torch.distributed.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    world_size = int(os.environ["WORLD_SIZE"])
    return True, local_rank, world_size, torch.device(f"cuda:{local_rank}")


def is_main(rank: int) -> bool:
    return rank == 0


def cleanup_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


# =============================================================================
# Pretrained checkpoint loading (device-agnostic; mirrors utils.load_model_frommmf
# but without the hard-coded .cuda() call so this also runs on CPU/Mac)
# =============================================================================

def _convert_pretrain_config(ckpt: dict) -> dict:
    new = {"config": {}}
    model_type = ckpt["config"]["model"]
    for k, v in ckpt["config"]["model_config"][model_type].items():
        new["config"][k] = v
    for k, v in ckpt["config"]["dataset_config"]["rnaseq"].items():
        new["config"][k] = v
    if model_type == "performergau_resolution":
        model_type = "performer_gau"
    sd = collections.OrderedDict()
    for k, v in ckpt["state_dict"].items():
        sd[str(k).split("model.")[1]] = v
    new["config"]["model_type"] = model_type
    new["model_state_dict"] = sd
    new["config"]["pos_embed"] = False
    new["config"]["device"] = "cuda"
    return new


def load_scfoundation_pretrained(
    ckpt_path: Path, key: str, device: torch.device
) -> Tuple[nn.Module, dict]:
    raw = torch.load(ckpt_path, map_location="cpu")
    converted = _convert_pretrain_config(raw[key])
    config = converted["config"]
    if "qv_dim" not in config and config.get("model") != "mae_autobin":
        config["qv_dim"] = config.get("dim_head", 64)
    if "ppi_edge" not in config:
        config["ppi_edge"] = None
    model = select_model(config)
    model.load_state_dict(converted["model_state_dict"])
    model.to(device)
    return model, config


# =============================================================================
# Vectorized gather (matches the official `gatherData` from scFoundation/load.py;
# returns positions in ascending row-major order so that the scatter inside
# MaeAutobin.forward `decoder_data[batch_idx, gen_idx] = x[~padding_label]` aligns)
# =============================================================================

def gather_data(
    data: torch.Tensor, labels: torch.Tensor, pad_token_id: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather entries of `data` where `labels==True` per row, padded to the
    per-batch max count using `pad_token_id`. Vectorized."""
    B, N = data.shape
    counts = labels.sum(dim=1)
    max_num = int(counts.max().item())
    if max_num == 0:
        out = torch.full((B, 1), float(pad_token_id), device=data.device, dtype=data.dtype)
        pad = torch.ones((B, 1), dtype=torch.bool, device=data.device)
        return out, pad

    fake_data = torch.full((B, max_num), float(pad_token_id), device=data.device, dtype=data.dtype)
    data_ext = torch.hstack([data, fake_data])

    fake_label = torch.full((B, max_num), 1.0, device=labels.device)
    none_labels = ~labels
    lab = labels.float().clone()
    lab[none_labels] = -float("inf")
    tmp = torch.tensor(
        [(i + 1) * 20000 for i in range(N, 0, -1)],
        device=labels.device,
        dtype=torch.float32,
    )
    lab = lab + tmp
    lab = torch.hstack([lab, fake_label])

    idx = lab.topk(max_num).indices
    new_data = torch.gather(data_ext, 1, idx)
    padding_mask = new_data == float(pad_token_id)
    return new_data, padding_mask


# =============================================================================
# Data preparation: vocab reindexing + raw count materialization
# =============================================================================

def load_vocab(model_dir: Path, vocab_file: str) -> List[str]:
    df = pd.read_csv(model_dir / vocab_file, header=0, delimiter="\t")
    return list(df["gene_name"])


def reindex_to_vocab(adata, vocab: list[str]) -> csr_matrix:
    """Reindex adata.X columns to the 19,264-gene vocabulary order, zero-padding
    missing genes. Returns a CSR matrix [n_cells, 19264]."""
    src_genes = list(adata.var["gene_name"])
    vocab_idx = {g: i for i, g in enumerate(vocab)}

    src_keep, dst_keep = [], []
    for j, g in enumerate(src_genes):
        vi = vocab_idx.get(g)
        if vi is not None:
            src_keep.append(j)
            dst_keep.append(vi)
    src_keep = np.asarray(src_keep, dtype=np.int64)
    dst_keep = np.asarray(dst_keep, dtype=np.int64)

    X = adata.X if issparse(adata.X) else csr_matrix(adata.X)
    P = csr_matrix(
        (np.ones(len(src_keep), dtype=np.float32), (src_keep, dst_keep)),
        shape=(X.shape[1], len(vocab)),
    )
    out = (X @ P).tocsr()
    log.info(
        f"Reindexed adata: {len(src_keep)}/{len(src_genes)} input genes mapped, "
        f"{len(vocab) - len(set(dst_keep.tolist()))} vocab genes zero-padded"
    )
    return out


def build_dataset_loss_gene_mask(adata, vocab: List[str]) -> torch.Tensor:
    """Bool mask [n_genes]: True for adata genes that map to the scFoundation vocab."""
    vocab_idx = {g: i for i, g in enumerate(vocab)}
    mask = torch.zeros(len(vocab), dtype=torch.bool)
    for g in adata.var["gene_name"].astype(str):
        vi = vocab_idx.get(g)
        if vi is not None:
            mask[vi] = True
    n = int(mask.sum())
    log.info(
        f"Loss restricted to dataset genes in vocab: {n}/{len(vocab)} positions "
        f"({len(adata.var)} genes in adata)"
    )
    if n == 0:
        raise ValueError("No adata genes mapped to the scFoundation vocab")
    return mask


def ensure_raw_counts(X: csr_matrix, is_counts: bool) -> csr_matrix:
    """scFoundation expects raw integer counts.

    If `is_counts` is True, return X unchanged. Otherwise we assume `X` was
    log1p(CPM)-style normalized (the standard scanpy preprocessing for the
    pancreas preset) and back-convert per the Methods recipe:

      1. Reverse the log1p transform via expm1 to recover the CPM-like values.
      2. Treat the smallest nonzero value as a raw count of 1; divide all
         nonzero values by this minimum and take the integer part.

    `is_counts` should come from `cfg.data.preprocess.counts` (treated as True
    by default; explicit `False` triggers the back-conversion).
    """
    if is_counts:
        return X
    if X.data.size == 0:
        return X
    log.warning(
        "cfg.data.preprocess.counts is False; assuming log1p(CPM)-normalized "
        "data and back-converting to raw counts (expm1 -> smallest-nonzero=1)"
    )
    # 1. reverse log1p
    unlogged = np.expm1(X.data).astype(np.float32)
    # Numerical guard: expm1 of tiny noise can yield very small positives.
    unlogged[unlogged < 0] = 0.0
    nz = unlogged[unlogged > 0]
    if nz.size == 0:
        return csr_matrix((unlogged, X.indices, X.indptr), shape=X.shape)
    # 2. smallest nonzero -> count of 1; floor the rest
    min_nz = float(nz.min())
    new_data = np.floor(unlogged / min_nz).astype(np.float32)
    return csr_matrix((new_data, X.indices, X.indptr), shape=X.shape)


class CountsDataset(Dataset):
    """Returns one cell's raw count vector [19264] as float32 per __getitem__."""

    def __init__(self, X: csr_matrix):
        self.X = X

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.X[idx].toarray().ravel().astype(np.float32)
        return torch.from_numpy(row)

class ScFoundationCountsAdapter:
    """
    Adapter exposing `preprocess_data(adata, shuffle=...)` so we can plug
    scFoundation finetuning into the repo's `create_data_loader` factory.

    It converts an AnnData into a torch DataLoader that yields raw-count
    tensors of shape [B, n_genes_vocab] in vocab order.
    """

    def __init__(self, cfg: DictConfig, vocab: List[str], device: torch.device):
        self.cfg = cfg
        self.vocab = vocab
        self.device = device

    def preprocess_data(self, adata, shuffle: bool = True) -> torch.utils.data.DataLoader:
        X = reindex_to_vocab(adata, self.vocab)
        is_counts = self.cfg.data.preprocess.get("counts", True) is not False
        X = ensure_raw_counts(X, is_counts=is_counts)

        ds = CountsDataset(X)
        # Prefer training.batch_size; fall back to data.preprocess.batch_size.
        bs = int(getattr(self.cfg.training, "batch_size", self.cfg.data.preprocess.batch_size))
        num_workers = int(getattr(self.cfg.training, "num_workers", 0))
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=bool(shuffle),
            num_workers=num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=bool(shuffle),
        )


# =============================================================================
# RDA preprocessing: hierarchical Bayesian downsampling + masking + tokenization
# =============================================================================

def hierarchical_bayesian_downsample(
    raw_counts: torch.Tensor,
    p_downsample: float,
    gamma_min: float,
    gamma_max: float,
) -> torch.Tensor:
    """Hierarchical Bayesian downsampling.

    Per cell:
        do_ds ~ Bernoulli(p_downsample)
        gamma = Uniform(gamma_min, gamma_max) if do_ds else 1.0
        x'_g ~ Binomial(x_g, gamma)   (per gene)
    """
    B = raw_counts.shape[0]
    device = raw_counts.device
    do_ds = torch.rand(B, device=device) < p_downsample
    gamma = torch.where(
        do_ds,
        torch.empty(B, device=device).uniform_(gamma_min, gamma_max),
        torch.ones(B, device=device),
    )
    gamma_b = gamma.unsqueeze(1).expand_as(raw_counts)
    return torch.binomial(raw_counts, gamma_b)


def build_rda_batch(
    raw_counts: torch.Tensor,
    pad_token_id: float,
    mask_token_id: float,
    seq_pad_pos: int,
    mask_ratio: float,
    p_downsample: float,
    gamma_min: float,
    gamma_max: float,
    cpm_total: float,
    fixed_T: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Build all tensors needed for one RDA training step from a batch of raw
    integer counts [B, n_genes].

    Returns dict with keys consumed by `MaeAutobin.forward` plus `target` /
    `mask_positions` for the loss."""
    B, N = raw_counts.shape  # N = 19264
    device = raw_counts.device

    # 1. hierarchical Bayesian downsampling
    input_counts = hierarchical_bayesian_downsample(
        raw_counts, p_downsample, gamma_min, gamma_max
    )

    # Totals for T (raw) and S (input). Clamp to >=1 to avoid log10(0) NaNs on
    # pathological cells (the paper QC requires >200 expressed genes so this
    # is just defensive).
    raw_total = raw_counts.sum(dim=1).clamp_min(1.0)
    input_total = input_counts.sum(dim=1).clamp_min(1.0)

    # 2. log1p(CPM) normalization
    raw_norm = torch.log1p(raw_counts / raw_total.unsqueeze(1) * cpm_total)
    input_norm = torch.log1p(input_counts / input_total.unsqueeze(1) * cpm_total)

    # 3. count indicators (T = targeted high resolution, S = source/input)
    # If `fixed_T` is provided, match authors' inference convention (e.g. t4).
    if fixed_T is None:
        T = torch.log10(raw_total)
    else:
        T = torch.full((B,), float(fixed_T), device=device, dtype=raw_total.dtype)
    S = torch.log10(input_total)

    # 4. random masking of 30% of gene positions (zero AND nonzero)
    mask_positions = torch.rand((B, N), device=device) < mask_ratio  # [B, N], bool
    masked_input = input_norm.clone()
    masked_input[mask_positions] = float(mask_token_id)

    # 5. assemble length-(N+2) decoder input. T at index N, S at index N+1.
    decoder_data = torch.cat([masked_input, T.unsqueeze(1), S.unsqueeze(1)], dim=1)
    seq_dim = N + 2

    # 6. encoder_labels: positions sent through encoder.
    #    For 0..N-1: nonzero(input) AND not masked. T,S always included.
    enc_genes = (input_counts > 0) & (~mask_positions)
    enc_TS = torch.ones((B, 2), dtype=torch.bool, device=device)
    encoder_labels = torch.cat([enc_genes, enc_TS], dim=1)

    # 7. gather encoder values + position ids in ascending-position order
    encoder_data, encoder_padding = gather_data(decoder_data, encoder_labels, pad_token_id)
    gene_ids_full = (
        torch.arange(seq_dim, device=device, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
    )
    enc_pos_ids_f, _ = gather_data(gene_ids_full, encoder_labels, pad_token_id)
    encoder_position_gene_ids = enc_pos_ids_f.long()
    # Padding positions point to the dedicated pad-position id (= seq_len).
    encoder_position_gene_ids[encoder_padding] = seq_pad_pos

    decoder_position_gene_ids = gene_ids_full.long()
    decoder_data_padding_labels = torch.zeros((B, seq_dim), dtype=torch.bool, device=device)

    return {
        "x": encoder_data,
        "padding_label": encoder_padding,
        "encoder_position_gene_ids": encoder_position_gene_ids,
        "encoder_labels": encoder_labels,
        "decoder_data": decoder_data,
        "decoder_position_gene_ids": decoder_position_gene_ids,
        "decoder_data_padding_labels": decoder_data_padding_labels,
        "target": raw_norm,                # [B, N]
        "mask_positions": mask_positions,  # [B, N]
    }


# =============================================================================
# Training / evaluation
# =============================================================================

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE over masked positions only."""
    diff = (pred.float() - target.float()) ** 2
    m = mask.float()
    denom = m.sum().clamp_min(1.0)
    return (diff * m).sum() / denom


def _model_forward(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        x=batch["x"],
        padding_label=batch["padding_label"],
        encoder_position_gene_ids=batch["encoder_position_gene_ids"],
        encoder_labels=batch["encoder_labels"],
        decoder_data=batch["decoder_data"],
        mask_gene_name=False,
        mask_labels=None,
        decoder_position_gene_ids=batch["decoder_position_gene_ids"],
        decoder_data_padding_labels=batch["decoder_data_padding_labels"],
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    cfg,
    device,
    epoch,
    rank,
    loss_gene_mask: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    use_amp = device.type == "cuda" and cfg.training.amp
    pad_id = cfg.preprocess.pad_token_id
    mask_id = cfg.preprocess.mask_token_id
    seq_pad_pos = cfg.preprocess.seq_len
    max_steps = cfg.training.get("max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)

    total_loss = 0.0
    n_steps = 0
    t0 = time.time()

    for step, raw_counts in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        raw_counts = raw_counts.to(device, non_blocking=True)
        batch = build_rda_batch(
            raw_counts,
            pad_token_id=pad_id,
            mask_token_id=mask_id,
            seq_pad_pos=seq_pad_pos,
            mask_ratio=cfg.preprocess.mask_ratio,
            p_downsample=cfg.preprocess.p_downsample,
            gamma_min=cfg.preprocess.gamma_min,
            gamma_max=cfg.preprocess.gamma_max,
            cpm_total=cfg.preprocess.cpm_norm_total,
            fixed_T=cfg.preprocess.get("fixed_T", None),
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred_full = _model_forward(model, batch)            # [B, N+2]
            pred = pred_full[:, : batch["target"].shape[1]]      # drop T, S
            loss_mask = batch["mask_positions"]
            if loss_gene_mask is not None:
                loss_mask = loss_mask & loss_gene_mask.unsqueeze(0)
            loss = masked_mse(pred, batch["target"], loss_mask)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_clip
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.grad_clip
            )
            optimizer.step()

        total_loss += float(loss.item())
        n_steps += 1

        if is_main(rank) and step % cfg.log_interval == 0:
            ms = (time.time() - t0) * 1000 / max(1, step + 1)
            msg = f"epoch {epoch} step {step} loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e} ms/step={ms:.1f}"
            log_payload = {
                "train/loss_step": float(loss.item()),
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }
            if cfg.training.get("log_grad_norm", False):
                gn = float(grad_norm)
                msg += f" grad_norm={gn:.4f}"
                log_payload["train/grad_norm"] = gn
            log.info(msg)
            wandb.log(log_payload)

    return total_loss / max(1, n_steps)


@torch.no_grad()
def evaluate(
    model,
    loader,
    cfg,
    device,
    epoch,
    rank,
    world_size: int,
    loss_gene_mask: Optional[torch.Tensor] = None,
) -> float:
    model.eval()
    use_amp = device.type == "cuda" and cfg.training.amp
    pad_id = cfg.preprocess.pad_token_id
    mask_id = cfg.preprocess.mask_token_id
    seq_pad_pos = cfg.preprocess.seq_len

    total_sse = torch.zeros(1, device=device)
    total_n = torch.zeros(1, device=device)

    for raw_counts in loader:
        raw_counts = raw_counts.to(device, non_blocking=True)
        batch = build_rda_batch(
            raw_counts,
            pad_token_id=pad_id,
            mask_token_id=mask_id,
            seq_pad_pos=seq_pad_pos,
            mask_ratio=cfg.preprocess.mask_ratio,
            p_downsample=cfg.preprocess.p_downsample,
            gamma_min=cfg.preprocess.gamma_min,
            gamma_max=cfg.preprocess.gamma_max,
            cpm_total=cfg.preprocess.cpm_norm_total,
            fixed_T=cfg.preprocess.get("fixed_T", None),
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred_full = _model_forward(model, batch)
            pred = pred_full[:, : batch["target"].shape[1]]
            diff = (pred.float() - batch["target"].float()) ** 2
            m = batch["mask_positions"]
            if loss_gene_mask is not None:
                m = m & loss_gene_mask.unsqueeze(0)
            m = m.float()
            total_sse += (diff * m).sum()
            total_n += m.sum()

    if world_size > 1:
        torch.distributed.all_reduce(total_sse, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_n, op=torch.distributed.ReduceOp.SUM)

    val = float(total_sse.item() / max(1.0, total_n.item()))
    if is_main(rank):
        wandb.log({"val/loss": val, "epoch": epoch})
    return val


# =============================================================================
# Checkpoint save
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    pretrainconfig: dict,
    save_dir: Path,
    name: str = "best_model.pt",
):
    save_dir.mkdir(parents=True, exist_ok=True)
    raw = model.module if hasattr(model, "module") else model
    torch.save(raw.state_dict(), save_dir / name)
    with open(save_dir / "pretrainconfig.pkl", "wb") as f:
        pickle.dump(pretrainconfig, f)


# =============================================================================
# Main
# =============================================================================

@hydra.main(
    config_path=str(REPO_ROOT / "config"),
    config_name="finetune_scfoundation",
    version_base="1.3",
)
def finetune(cfg: DictConfig):
    ddp, local_rank, world_size, device = setup_distributed()
    set_seed(cfg.seed + local_rank)

    if is_main(local_rank):
        os.environ["WANDB_MODE"] = cfg.wandb.mode
        wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
        log.info(f"DDP={ddp} world_size={world_size} device={device}")

    # ---- 1. Load pretrained scFoundation ----
    model_dir = Path(cfg.model.model_dir)
    ckpt_path = model_dir / cfg.model.ckpt_file
    model, pretrainconfig = load_scfoundation_pretrained(
        ckpt_path, key=cfg.model.key, device=device
    )
    if is_main(local_rank):
        log.info(f"Loaded pretrained scFoundation from {ckpt_path}")

    # Canonical token-ids and seq-len come from the loaded config
    cfg.preprocess.pad_token_id = float(pretrainconfig["pad_token_id"])
    cfg.preprocess.mask_token_id = float(pretrainconfig["mask_token_id"])
    cfg.preprocess.seq_len = int(pretrainconfig["seq_len"])
    if is_main(local_rank):
        log.info(
            f"pretrainconfig: pad={cfg.preprocess.pad_token_id} "
            f"mask={cfg.preprocess.mask_token_id} seq_len={cfg.preprocess.seq_len}"
        )

    # ---- 2. Vocab + adata + raw counts ----
    vocab = load_vocab(model_dir, cfg.model.vocab_file)
    assert len(vocab) == cfg.model.n_genes, (
        f"vocab size {len(vocab)} != cfg.model.n_genes {cfg.model.n_genes}"
    )
    # ---- 3. Train / val loaders ----
    adapter = ScFoundationCountsAdapter(cfg=cfg, vocab=vocab, device=device)

    # Partitioned (census / multi-query) data streams through the repo loaders; the
    # loss gene mask would vary per partition, so restriction is disabled there.
    is_query_mode = partitioned(cfg)
    restrict_loss = cfg.preprocess.get("restrict_loss_to_dataset_genes", True)
    loss_gene_mask = None

    if is_query_mode:
        if is_main(local_rank):
            if restrict_loss:
                log.warning("restrict_loss_to_dataset_genes=True ignored for partitioned loading.")
            log.warning("Partitioned loading has no held-out split: validation is a subsample of "
                        "the training partitions (cfg.data.preprocess.split).")
        if ddp:
            train_loader = create_multiprocess_data_loader(
                cfg=cfg, worker_id=local_rank, n_workers=world_size, adapter=adapter, test_mode=False)
            val_loader = create_multiprocess_data_loader(
                cfg=cfg, worker_id=local_rank, n_workers=world_size, adapter=adapter, test_mode=True)
        else:
            train_loader = create_data_loader(cfg=cfg, adapter=adapter, test_mode=False)
            val_loader = create_data_loader(cfg=cfg, adapter=adapter, test_mode=True)
    else:
        # Single file: hold out a disjoint set of cells for validation. The split uses
        # cfg.seed (not the per-rank seed) so every rank sees the same partition.
        adata = load_adata(cfg)
        if is_main(local_rank):
            log.info(f"adata loaded: shape={adata.shape}")
        if restrict_loss:
            loss_gene_mask = build_dataset_loss_gene_mask(adata, vocab).to(device)

        perm = np.random.default_rng(cfg.seed).permutation(adata.n_obs)
        n_val = max(1, int(cfg.training.val_split * adata.n_obs))
        is_counts = cfg.data.preprocess.get("counts", True) is not False
        X = ensure_raw_counts(reindex_to_vocab(adata, vocab), is_counts=is_counts)
        del adata
        train_ds = CountsDataset(X[np.sort(perm[n_val:])])
        val_ds = CountsDataset(X[np.sort(perm[:n_val])])
        if is_main(local_rank):
            log.info(f"train cells: {len(train_ds)}, val cells: {len(val_ds)}")

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True) if ddp else None
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False) if ddp else None
        loader_kwargs = dict(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
                             pin_memory=(device.type == "cuda"))
        train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=(train_sampler is None),
                                  drop_last=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, sampler=val_sampler, shuffle=False, drop_last=False, **loader_kwargs)

    # ---- 4. DDP wrap ----
    if ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False
        )

    # ---- 5. Optimizer / scheduler / scaler ----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.training.lr_gamma)
    # bf16 autocast does not need a GradScaler; only enable scaler for fp16.
    scaler = None

    # ---- 6. Train ----
    save_dir = Path(cfg.save_dir)
    if is_main(local_rank):
        save_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, cfg.training.epochs + 1):
        t0 = time.time()
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            cfg,
            device,
            epoch,
            local_rank,
            loss_gene_mask=loss_gene_mask,
        )
        val_loss = evaluate(
            model,
            val_loader,
            cfg,
            device,
            epoch,
            local_rank,
            world_size,
            loss_gene_mask=loss_gene_mask,
        )
        scheduler.step()

        if is_main(local_rank):
            log.info(
                f"epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| {(time.time() - t0):.1f}s"
            )
            wandb.log({"train/loss_epoch": train_loss, "val/loss_epoch": val_loss, "epoch": epoch})

            if cfg.training.save_every_epoch:
                save_checkpoint(model, pretrainconfig, save_dir, name=f"epoch{epoch}.pt")

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(model, pretrainconfig, save_dir, name="best_model.pt")
                log.info(f"new best (val={best_val:.4f})")

    # ---- 7. Final dump of resolved config ----
    if is_main(local_rank):
        with open(save_dir / "CONFIG.yaml", "w") as f:
            yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, sort_keys=False)
        wandb.finish()

    cleanup_distributed()


if __name__ == "__main__":
    finetune()
