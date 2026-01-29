import os
# Set W&B to offline mode BEFORE importing wandb
os.environ["WANDB_MODE"] = "offline"

import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

import torch.multiprocessing as mp
import sys
import wandb
import logging
from pathlib import Path
from tqdm import tqdm
from contextlib import nullcontext
import torch

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.utils import set_seed, get_devices
from sae4scfm.core.io_utils import save_sae_checkpoint, save_metrics
from sae4scfm.core.buffer import create_buffer
from sae4scfm.core.evaluation import evaluate_reconstruction

mp.set_start_method('spawn', force=True)
log = logging.getLogger(__name__)


# Register custom OmegaConf resolvers for computed values
OmegaConf.register_new_resolver(
    "compute_dict_size",
    lambda multiplier, activation_dim: round(multiplier * activation_dim)
)

OmegaConf.register_new_resolver(
    "compute_steps",
    lambda n_cells, ctx_len, batch_size, n_eval_batches, refresh_batch_size: (
        int((sum(n_cells) * ctx_len / batch_size) - (n_eval_batches * refresh_batch_size * ctx_len / batch_size))
        )
)


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="train", version_base='1.3')
def main(cfg: DictConfig) -> None:
    
    # Setup logging
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    logging.basicConfig(filename=Path(hydra_cfg['runtime']['output_dir']) / "train.log", level=logging.INFO)
    log.info("Starting training with configuration:")
    log.info("\n" + OmegaConf.to_yaml(cfg))

    # Initialize W&B
    wandb.init(project="SAE-scFM", name=cfg.experiment.number, mode='disabled')

    # Set random seed and get devices
    set_seed(cfg.seed)
    devices = get_devices()
    
    # Update device in config for resolvers
    OmegaConf.update(cfg, "device", devices[0], merge=False)

    # Instantiate scfm model adapter
    adapter_class = get_class(cfg.scfm._target_)
    adapter = adapter_class(cfg)
    adapter.load_model()

    # Instantiate the buffer
    buffer, buffer_processes = create_buffer(cfg, adapter, devices[0], devices[1])

    # Instantiate trainer directly
    trainer_cfg = OmegaConf.to_container(cfg.sae.hyperparams, resolve=True)
    trainer_cfg['dict_class'] = get_class(cfg.sae.autoencoder._target_)
    trainer = instantiate(cfg.sae.trainer, **trainer_cfg)

    # Train the sparse autoencoder
    autocast_dtype = torch.float32
    autocast_context = nullcontext() if devices[0] == "cpu" else torch.autocast(device_type='cuda', dtype=autocast_dtype)
    
    # Initialize evaluation metrics storage
    all_eval_metrics = {metric: [] for metric in cfg.sae.metrics}
    
    for step, act in enumerate(tqdm(buffer, total=cfg.sae.steps)):
        act = act.to(dtype=autocast_dtype)

        # Logging and evaluation
        if step > 0 and step % cfg.experiment.eval_steps == 0:
            eval_metrics = evaluate_reconstruction(
                sae_trainer=trainer,
                buffer=buffer,
                device=devices[0],
                step=step
            )
            wandb.log(eval_metrics, step=step)
            
            # Append metric values to lists
            for metric_name, metric_value in eval_metrics.items():
                all_eval_metrics[metric_name].append(metric_value)

        # update step
        with autocast_context:
            trainer.update(step, act)


    # Join buffer processes
    for refill_process in buffer_processes:
        refill_process.join()

    # Save the trained SAE model and metrics
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_dir = Path(hydra_cfg['runtime']['output_dir'])
    save_sae_checkpoint(trainer.ae, save_dir)
    save_metrics(all_eval_metrics, save_dir)
    

if __name__ == "__main__":
    main()