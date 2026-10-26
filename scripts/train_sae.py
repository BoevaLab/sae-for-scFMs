import os
import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

import torch.multiprocessing as mp
import sys
import logging
from pathlib import Path
from tqdm import tqdm
import torch

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.utils import set_seed, get_devices
from sae4scfm.core.io_utils import (
    load_training_checkpoint,
    save_sae_checkpoint,
    save_metrics,
    save_training_checkpoint,
    save_training_summary,
)
from sae4scfm.core.buffer import create_buffer
from sae4scfm.core.evaluation import evaluate_reconstruction
from sae4scfm.core.pipeline import register_resolvers

mp.set_start_method('spawn', force=True)
log = logging.getLogger(__name__)


register_resolvers()


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="train", version_base='1.3')
def main(cfg: DictConfig) -> None:
    
    # Setup logging
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    logging.basicConfig(filename=Path(hydra_cfg['runtime']['output_dir']) / "train.log", level=logging.INFO)
    log.info("Starting training with configuration:")
    log.info("\n" + OmegaConf.to_yaml(cfg))

    # Set random seed and get devices
    set_seed(cfg.seed)
    devices = get_devices()
    
    # Update device in config for resolvers
    OmegaConf.update(cfg, "device", devices[0], merge=False)

    # Instantiate scfm model adapter
    adapter_class = get_class(cfg.scfm._target_)
    adapter = adapter_class(cfg)
    if not cfg.buffer.use_multiprocessing:
        adapter.load_model()
    else: 
        log.info("Using multiprocessing for buffer. There are %d gpus available and %d cpu cores available.", torch.cuda.device_count(), torch.get_num_threads() )

    # Instantiate the buffer
    buffer, buffer_processes = create_buffer(cfg, adapter, devices[0], devices[1])

    # Instantiate trainer directly
    trainer_cfg = OmegaConf.to_container(cfg.sae.hyperparams, resolve=True)
    trainer_cfg['dict_class'] = get_class(cfg.sae.autoencoder._target_)
    trainer = instantiate(cfg.sae.trainer, **trainer_cfg)
    start_step = 0

    if cfg.resume.enabled:
        if not cfg.resume.checkpoint_dir:
            raise ValueError("`resume.checkpoint_dir` must be set when `resume.enabled=true`.")

        training_state, original_cfg = load_training_checkpoint(
            checkpoint_path=Path(cfg.resume.checkpoint_dir),
            device=devices[0],
            current_cfg=cfg,
            strict_config=cfg.resume.strict_config,
        )
        start_step = trainer.load_state_dict(training_state)
        log.info(
            "Resumed training from %s at step %d (source experiment: %s).",
            cfg.resume.checkpoint_dir,
            start_step,
            original_cfg.experiment.number,
        )

    # Train the sparse autoencoder (fp32; eval metrics keyed as reported, starting with cfg.sae.metrics)
    all_eval_metrics = {metric: [] for metric in cfg.sae.metrics}
    remaining_steps = max(cfg.sae.steps - start_step, 0)
    last_completed_step = start_step

    if remaining_steps > 0:
        for step, act in enumerate(tqdm(buffer, total=remaining_steps), start=start_step):
            act = act.float()

            # Logging and evaluation
            if step > 0 and step % cfg.experiment.eval_steps == 0:
                eval_metrics = evaluate_reconstruction(
                    sae_trainer=trainer,
                    buffer=buffer,
                    device=devices[0],
                    step=step
                )
                for metric_name, metric_value in eval_metrics.items():
                    all_eval_metrics.setdefault(metric_name, []).append(metric_value)

            trainer.update(step, act)
            last_completed_step = step + 1


    # Join buffer processes
    for refill_process in buffer_processes:
        refill_process.join()

    # Save the trained SAE model and metrics
    save_dir = Path(hydra_cfg['runtime']['output_dir'])
    if cfg.resume.save_training_checkpoint:
        save_training_checkpoint(trainer, save_dir, last_completed_step, cfg)
    save_sae_checkpoint(trainer.ae, save_dir)
    save_metrics(all_eval_metrics, save_dir)
    save_training_summary(trainer, last_completed_step, save_dir)
    if hasattr(trainer.ae, "threshold") and float(trainer.ae.threshold) < 0:
        log.warning("The SAE threshold was never calibrated (training ended before "
                    "threshold_start_step); sae_inference=threshold would act as a dense ReLU.")
    

if __name__ == "__main__":
    main()
