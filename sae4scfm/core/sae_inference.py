"""
How a trained SAE is applied at inference: batch top-k or the learned threshold.

BatchTopK SAEs are trained with a top-k over the whole batch, so at inference the set of
active features for a token depends on the other tokens in the batch. The learned threshold
(an EMA of the smallest kept activation during training) gives a per-token alternative.
Config (group `sae_inference`):
    mode: batchtopk | threshold
    threshold: null  # threshold mode only; null keeps the checkpoint's value
"""
import inspect
import logging

import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

MODES = ("batchtopk", "threshold")


def inference_mode(inference_cfg) -> str:
    mode = "batchtopk" if inference_cfg is None else inference_cfg.get("mode", "batchtopk")
    if mode not in MODES:
        raise ValueError(f"sae_inference.mode must be one of {MODES}, got {mode!r}")
    return mode


def prepare_sae(sae, inference_cfg, train_cfg: DictConfig, training_summary=None) -> None:
    """Apply a threshold override and warn when threshold mode uses an uncalibrated threshold.

    `training_summary` (from training_summary.json) gives the real number of training steps;
    older runs fall back to the step estimate in the training config.
    """
    if inference_mode(inference_cfg) != "threshold":
        return
    if not hasattr(sae, "threshold"):
        log.warning("%s has no learned threshold; sae_inference.mode=threshold uses its plain encode.",
                    type(sae).__name__)
        return

    override = inference_cfg.get("threshold", None)
    if override is not None:
        sae.threshold.fill_(float(override))
        log.info("Using threshold override %.4g", float(override))
        return

    threshold = float(sae.threshold)
    if threshold < 0:
        log.warning("SAE threshold is %.3g: it was never calibrated during training, so threshold "
                    "mode keeps every positive latent (a dense ReLU). Set sae_inference.threshold "
                    "or use batchtopk.", threshold)
        return
    hyper = train_cfg.sae.get("hyperparams", {})
    start = hyper.get("threshold_start_step", 1000)
    beta = hyper.get("threshold_beta", 0.999)
    if training_summary is not None:
        steps = training_summary["steps"]
        start = training_summary.get("threshold_start_step", start)
        beta = training_summary.get("threshold_beta", beta)
    else:
        try:
            steps = int(OmegaConf.select(train_cfg, "sae.steps"))
        except Exception:  # unresolvable interpolation in an old config
            steps = None
    if steps is not None and steps - start < 1 / (1 - beta):
        log.warning("SAE threshold %.4g comes from only ~%d EMA updates (beta=%s); it mostly reflects "
                    "the first batches after step %d.", threshold, max(steps - start, 0), beta, start)


def encode_rows(sae, rows: torch.Tensor, inference_cfg) -> torch.Tensor:
    """Encode activations of shape [N, D]; for batch top-k, all N rows share the k * N budget."""
    params = inspect.signature(sae.encode).parameters
    if "use_threshold" not in params:
        return sae.encode(rows)
    return sae.encode(rows, use_threshold=inference_mode(inference_cfg) == "threshold")


def encode_tokens(sae, x: torch.Tensor, mask, inference_cfg) -> torch.Tensor:
    """Encode token activations [B, L, D] into features [B, L, F].

    With a mask ([B, L] bool) only the real tokens are encoded and the rest of the output is
    zero. Without a mask every position is encoded, padding included.
    """
    batch_size, seq_len, dim = x.shape
    if mask is None:
        return encode_rows(sae, x.reshape(batch_size * seq_len, dim), inference_cfg).reshape(batch_size, seq_len, -1)
    rows = encode_rows(sae, x[mask], inference_cfg)
    features = rows.new_zeros(batch_size, seq_len, rows.shape[-1])
    features[mask] = rows
    return features
