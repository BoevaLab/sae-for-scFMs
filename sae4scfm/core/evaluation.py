"""
Evaluation utilities for SAE training
"""

import torch
import logging
from typing import Dict

log = logging.getLogger(__name__)

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
    
    with torch.no_grad():
        for batch in original_batches:
            activations = buffer.adapter.generate_activations(batch, device)
            selected_activations = buffer._sample_tokens(activations, batch)
            
            act, act_hat, f, losslog = sae_trainer.loss(selected_activations, step=step, logging=True)


            # Calculate losses
            l0 = (f != 0).float().sum(dim=-1).mean().item()
            total_variance = torch.var(act, dim=0).sum()
            residual_variance = torch.var(act - act_hat, dim=0).sum()
            frac_variance_explained = 1 - residual_variance / total_variance
            metrics["l0"] += l0
            metrics["mse_loss"] += (act - act_hat).pow(2).sum(dim=-1).mean().cpu().item()
            metrics["frac_variance_explained"] += frac_variance_explained.item()
            
            # Accumulate losses from losslog
            if hasattr(losslog, 'losses') and isinstance(losslog.losses, dict):
                for k, v in losslog.losses.items():
                    metrics[k] = metrics.get(k, 0.0) + (v.cpu().item() if isinstance(v, torch.Tensor) else v)
            
            # Accumulate trainer logging parameters
            trainer_log = sae_trainer.get_logging_parameters()
            for name, value in trainer_log.items():
                if isinstance(value, torch.Tensor):
                    value = value.cpu().item()
                metrics[f"{name}"] += value
            
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
    
    # initialize
    total_mz = 0.0
    total_mr = 0.0
    
    for batch in original_batches:
        mask = adapter.generate_activation_mask(batch).to(device)
        embedding_model, x = adapter.generate_embeddings(batch, device_model=device, return_activations=True)
        
        sae.eval()
        x_hat = sae(x[mask])
        x[mask] = x_hat
        sae.train()

        embedding_reconstructed = adapter.generate_embeddings(batch, device_model=device, modify_activations=True, x_hat=x)
        embedding_zero_batch = adapter.generate_embeddings(batch, device_model=device, modify_activations=True, x_hat=torch.zeros_like(x))

        # Accumulate totals
        total_mz += torch.nn.functional.mse_loss(embedding_zero_batch, embedding_model, reduction='sum')
        total_mr += torch.nn.functional.mse_loss(embedding_model, embedding_reconstructed, reduction='sum')
    
    embedding_recovery = (total_mz - total_mr) / total_mz
    
    return embedding_recovery.item()
