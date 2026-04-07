"""
Evaluation utilities for SAE training
"""

import inspect
import logging
from typing import Dict

import torch

from sae4scfm.core.sae_inference import encode_rows

log = logging.getLogger(__name__)

# Computed here from the reconstruction; trainer-reported values with these names are not added again.
_OWN_METRICS = ("l0", "mse_loss", "frac_variance_explained")

def evaluate_reconstruction(
    sae_trainer,
    buffer,
    device: str = 'cuda',
    step: int = 100
) -> Dict[str, float]:
    """
    Evaluate SAE reconstruction quality
    
    Args:
        sae_trainer: SAE trainer
        buffer: ActivationBuffer 
        device: Device for evaluation
        step: Training step for logging
    
    Returns:
        Dict of metrics: 
    """
    original_batches = buffer.get_eval_data()
    metrics = {el:0.0 for el in buffer.cfg.sae.metrics}
    # Evaluation must not advance training state (inference threshold, dead-feature counters).
    loss_kwargs = {"update_state": False} if "update_state" in inspect.signature(sae_trainer.loss).parameters else {}

    if buffer.use_multiprocessing:
        buffer._ensure_model_and_hook(device)

    try:
        with torch.inference_mode():
            for batch in original_batches:
                activations = buffer.adapter.generate_activations(batch, device)
                selected_activations = buffer._sample_tokens(activations, batch)
                
                loss_out = sae_trainer.loss(selected_activations, step=step, logging=True, **loss_kwargs)
                # Most trainers return a namedtuple("LossLog", ["x", <x_hat/recon>, "f", "losses"]).
                # Older code may unpack it into (x, x_hat, f, losses_dict); handle both.
                if hasattr(loss_out, "losses"):
                    act = getattr(loss_out, "x", loss_out[0])
                    act_hat = (
                        getattr(loss_out, "x_hat", None)
                        if hasattr(loss_out, "x_hat")
                        else getattr(loss_out, "recon", None)
                    )
                    if act_hat is None:
                        act_hat = loss_out[1]
                    f = getattr(loss_out, "f", loss_out[2])
                    losslog = loss_out
                else:
                    # Assume it's already a 4-tuple: (x, x_hat, f, losses_dict)
                    act, act_hat, f, losses_dict = loss_out
                    losslog = losses_dict


                # Calculate losses
                l0 = (f != 0).float().sum(dim=-1).mean().item()
                total_variance = torch.var(act, dim=0).sum()
                residual_variance = torch.var(act - act_hat, dim=0).sum()
                frac_variance_explained = 1 - residual_variance / total_variance
                metrics["l0"] += l0
                metrics["mse_loss"] += (act - act_hat).pow(2).sum(dim=-1).mean().cpu().item()
                metrics["frac_variance_explained"] += frac_variance_explained.item()
                
                # Accumulate losses from losslog
                if hasattr(losslog, "losses") and isinstance(losslog.losses, dict):
                    losses_dict = losslog.losses
                elif isinstance(losslog, dict):
                    losses_dict = losslog
                else:
                    losses_dict = {}

                for k, v in losses_dict.items():
                    if k in _OWN_METRICS:
                        continue
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().item()
                    metrics[k] = metrics.get(k, 0.0) + float(v)
                
                # Accumulate trainer logging parameters
                trainer_log = sae_trainer.get_logging_parameters()
                for name, value in trainer_log.items():
                    if isinstance(value, torch.Tensor):
                        value = value.cpu().item()
                    metrics[name] = metrics.get(name, 0.0) + float(value)
                
        # Average
        for key in metrics:
            metrics[key] /= len(original_batches)

        embedding_recovery = embedding_recovery_score(
            sae=sae_trainer.ae,
            buffer=buffer,
            adapter=buffer.adapter,
            device=device
        )

        metrics['embedding_recovery_score'] = embedding_recovery
        
        log.info(
            f"Eval ({buffer.eval_strategy}, {len(original_batches)} batches): "
            f"MSE={metrics['mse_loss']:.4f} Embedding Recovery={embedding_recovery:.4f}"
        )
        
        return metrics
    finally:
        if buffer.use_multiprocessing:
            buffer.release_eval_model()

def embedding_recovery_score(
    sae,
    buffer,
    adapter,
    device: str = 'cuda'
) -> float:
    """
    Compute the embedding recovery score 
    Args:
        sae: SAE 
        buffer: ActivationBuffer
        adapter: Model adapter
        device: Device for evaluation
    
    Returns:
        embedding_recovery: float
    """
    original_batches = buffer.get_eval_data()

    # initialize (keep as tensors for device-safe accumulation)
    total_mz = None
    total_mr = None

    with torch.no_grad():
        for batch in original_batches:
            mask = adapter.generate_activation_mask(batch).to(device)
            embedding_model, x = adapter.generate_embeddings(
                batch, device_model=device, return_activations=True
            )

            # Reconstruct only non-padding positions, without mutating `x` in-place; encode with
            # batch top-k, as in training (the threshold may not be calibrated yet).
            x_recon = x.clone()
            x_recon[mask] = sae.decode(encode_rows(sae, x[mask], None))

            # Zero baseline: zero only positions we intend to reconstruct
            x_zero = x.clone()
            x_zero[mask] = 0

            embedding_reconstructed = adapter.generate_embeddings(
                batch,
                device_model=device,
                modify_activations=True,
                x_hat=x_recon,
            )
            embedding_zero_batch = adapter.generate_embeddings(
                batch,
                device_model=device,
                modify_activations=True,
                x_hat=x_zero,
            )

            mz = torch.nn.functional.mse_loss(
                embedding_zero_batch, embedding_model, reduction='sum'
            )
            mr = torch.nn.functional.mse_loss(
                embedding_model, embedding_reconstructed, reduction='sum'
            )

            total_mz = mz if total_mz is None else (total_mz + mz)
            total_mr = mr if total_mr is None else (total_mr + mr)

    # Avoid divide-by-zero in degenerate cases (e.g. empty mask everywhere)
    if total_mz is None or total_mz.item() == 0:
        return 0.0

    embedding_recovery = (total_mz - total_mr) / total_mz
    return embedding_recovery.item()
