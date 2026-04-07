"""
Encode a dataset with a trained SAE and write its features to cached_features/<experiment>/.

output.format=h5 writes token-level features (with gene ids and input values) to
features-<timestamp><tag>.h5; output.format=pooled writes each cell's max over tokens to
features-<timestamp><tag>.pt, with the cells' input positions in features-<timestamp><tag>.cell_idx.npy.
"""

import os
import sys
from pathlib import Path
import hydra

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.data_loader import create_data_loader
from sae4scfm.core.io_utils import generate_and_compress_features, generate_pool_and_save_features
from sae4scfm.core.pipeline import load_run, register_resolvers

register_resolvers()


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="generate", version_base='1.3')
def main(cfg) -> None:
    sae, original_cfg, adapter, device = load_run(cfg, REPO_ROOT)
    data_loader = create_data_loader(original_cfg, adapter, test_mode=True)

    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_dir = Path(hydra_cfg['runtime']['output_dir'])

    output = cfg.get("output", {})
    writers = {"h5": generate_and_compress_features, "pooled": generate_pool_and_save_features}
    fmt = output.get("format", "h5")
    if fmt not in writers:
        raise ValueError(f"output.format must be h5 or pooled, got {fmt!r}")
    writers[fmt](
        adapter=adapter,
        sae=sae,
        data_loader=data_loader,
        device=device,
        cfg=original_cfg,
        save_dir=save_dir,
        date=cfg.sae_checkpoint.timestamp + output.get("tag", ""),
        inference_cfg=cfg.get("sae_inference", None))


if __name__ == "__main__":
    main()
