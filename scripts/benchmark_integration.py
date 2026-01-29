import hydra
from hydra.utils import instantiate, get_class
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf

import os
import sys
from pathlib import Path
import logging


# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.utils import set_seed
from sae4scfm.core.steering import compute_batch_integration
from sae4scfm.core.io_utils import load_embeddings, load_adata

log = logging.getLogger(__name__)

@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="benchmark", version_base='1.3')
def main(cfg) -> None:
    # Load adata with test split
    test_fraction = cfg.data.preprocess.split if hasattr(cfg.data, 'preprocess') else 1.0
    adata = load_adata(cfg, sub_sample=test_fraction)

    load_embeddings(adata, cfg)
    log.info(f"Loaded embeddings")

    results = compute_batch_integration(adata, cfg)

    # Save results
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = Path(hydra_cfg['runtime']['output_dir'])
    results.get_results(min_max_scale=False).to_csv(output_dir / "benchmark_results.csv")
    log.info(f"Saved benchmark results to {output_dir / 'benchmark_results.csv'}")

if __name__ == "__main__":
    main()