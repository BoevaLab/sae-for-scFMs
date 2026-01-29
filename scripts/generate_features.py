"""
Docstring for scripts.generate_features
"""

import os
import sys
from pathlib import Path
import hydra
from hydra.utils import get_class
from omegaconf import OmegaConf

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.utils import set_seed, get_devices
from sae4scfm.core.data_loader import create_data_loader
from sae4scfm.core.io_utils import load_sae_checkpoint, generate_and_compress_features, generate_pool_and_save_features

# Register custom OmegaConf resolvers for computed values
OmegaConf.register_new_resolver(
    "compute_dict_size",
    lambda multiplier, activation_dim: round(multiplier * activation_dim)
)

@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="generate", version_base='1.3')
def main(cfg) -> None:

    # Load SAE checkpoint
    device = get_devices()[0]
    checkpoint_path = REPO_ROOT / "experiments/train" /cfg.sae_checkpoint.experiment / cfg.sae_checkpoint.timestamp
    sae, original_cfg = load_sae_checkpoint(checkpoint_path, device)
    set_seed(original_cfg.seed)

    # Overwrite dataset
    data = cfg.get("data", None)
    if data is not None:
        data.preprocess.batch_size = original_cfg.data.preprocess.batch_size
        data.preprocess.subset_hvg = original_cfg.data.preprocess.subset_hvg
        original_cfg.data = data
        print(f'Using {original_cfg.data.name} dataset')

    # Instantiate scfm model adapter
    adapter_class = get_class(original_cfg.scfm._target_)
    adapter = adapter_class(original_cfg)

    # Load scfm and setup hooks
    adapter.load_model()
    adapter.setup_hook()
    adapter.model.to(device)

    # Load data
    data_loader = create_data_loader(original_cfg, adapter, test_mode=True)

    # Generate and compress features
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_dir = Path(hydra_cfg['runtime']['output_dir'])

    generate_and_compress_features(
        adapter = adapter,
        sae = sae,
        data_loader = data_loader,
        device = device,
        cfg = original_cfg,
        save_dir=save_dir,
        date = cfg.sae_checkpoint.timestamp)
    


if __name__ == "__main__":
    main()