"""
Docstring for scripts.feature_steering
"""
import hydra
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
import sys
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.data_loader import create_data_loader
from sae4scfm.core.utils import set_seed, get_devices
from sae4scfm.core.io_utils import load_sae_checkpoint
from sae4scfm.core.steering import steer

log = logging.getLogger(__name__)

# Register custom OmegaConf resolvers for computed values
OmegaConf.register_new_resolver(
    "compute_dict_size",
    lambda multiplier, activation_dim: round(multiplier * activation_dim)
)

@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="steer", version_base='1.3')
def main(cfg: DictConfig) -> None:
    
    # Setup logging
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    logging.basicConfig(filename=Path(hydra_cfg['runtime']['output_dir']) / "steering.log", level=logging.INFO)
    log.info("Starting training with configuration:")
    log.info("\n" + OmegaConf.to_yaml(cfg))
    
    # Load SAE checkpoint
    device = get_devices()[0]
    checkpoint_path = REPO_ROOT / "experiments/train" / cfg.sae_checkpoint.experiment / cfg.sae_checkpoint.timestamp
    sae, original_cfg = load_sae_checkpoint(checkpoint_path, device)
    set_seed(original_cfg.seed)
    sae = sae.to(device)
    
    # Instantiate scfm model adapter
    adapter_class = get_class(original_cfg.scfm._target_)
    adapter = adapter_class(original_cfg)

    # Load scfm and setup hooks
    adapter.load_model()
    adapter.setup_hook()
    adapter.model.to(device)

    # Load data
    data_loader = create_data_loader(original_cfg, adapter, test_mode=True)
    
    # Initialize feature selector
    feature_selector = instantiate(cfg.steering.feature_selection)

    # Output directory
    output_dir = Path(hydra_cfg['runtime']['output_dir'])
    
    # Run steering experiments
    for seed in tqdm(cfg.steering.seeds):
        for n_features in tqdm(cfg.steering.n_features_list):
            for clamp_value in tqdm(cfg.steering.clamp_values):
                log.info(f"Selected {n_features} features for trial {seed} with clamp {clamp_value}")
                set_seed(seed)

                # Select features
                selected_features = feature_selector.select_features(
                    n_features=n_features,
                    dict_size=original_cfg.sae.dict_size,
                    trial=seed
                )
                    
                # Perform steering
                original_emb, steered_emb = steer(
                    adapter=adapter,
                    sae=sae,
                    data_loader=data_loader,
                    features_to_steer=selected_features,
                    clamp_value=clamp_value,
                    device=device
                )
                
                # Save embeddings
                output_dir.mkdir(parents=True, exist_ok=True)
                #np.save(output_dir / "original.npy", original_emb)
                np.save(output_dir / f"{feature_selector.name}_n{n_features}_seed{seed}_clamp{clamp_value}.npy", steered_emb)
    

        np.save(output_dir / f"original_seed{seed}.npy", original_emb)


if __name__ == "__main__":
    main()