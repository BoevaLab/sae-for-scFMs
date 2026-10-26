"""
Steer SAE features during the scFM forward pass and save the steered cell embeddings.

steering.mode=suppress clamps selected features wherever they fire (one run per seed,
number of features and clamp value); steering.mode=induce writes features onto a random
fraction of each cell's tokens for every (fraction, multiplier) pair. cells.csv holds the
metadata (steering.cell_columns present in the data) of the rows of every saved embedding.
"""
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import os
import sys
import logging
import numpy as np
import torch
from tqdm import tqdm

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.data_loader import create_data_loader
from sae4scfm.core.io_utils import load_adata
from sae4scfm.core.pipeline import load_run, register_resolvers
from sae4scfm.core.steering import Clamp, steer

log = logging.getLogger(__name__)

register_resolvers()


def load_max_activation(path, features) -> np.ndarray:
    """Maximum activation of each requested feature over all cells (and tokens).

    Accepts pooled [cells, features] activations (.pt/.npy) or a token-level feature file
    written by generate_features.py (.h5), whose max over cells and tokens is the same.
    """
    path = Path(path)
    if path.suffix == ".h5":
        import h5py
        import hdf5plugin  # noqa: F401  (Blosc filter)
        with h5py.File(path, "r") as f:
            return np.array([f["features"][:, :, j].max() for j in features], dtype=np.float32)
    pooled = torch.load(path, map_location="cpu").float().numpy() if path.suffix == ".pt" else np.load(path)
    return pooled[:, features].max(axis=0).astype(np.float32)


def save_cells(output_dir, original_cfg, columns, cell_idx) -> None:
    """Metadata of the embedding rows: `cell` (position in the steered dataset) and the requested
    obs columns that the dataset has."""
    obs = load_adata(original_cfg, sub_sample=original_cfg.data.preprocess.split).obs
    rows = np.arange(len(obs)) if cell_idx is None else np.asarray(cell_idx)
    cells = obs.iloc[rows][[c for c in columns if c in obs]].reset_index(drop=True)
    cells.insert(0, "cell", rows)
    cells.to_csv(output_dir / "cells.csv", index=False)


def run_suppression(cfg, sae, original_cfg, adapter, device, output_dir, suffix):
    # One loader for all runs: preprocessing draws from the global RNG (e.g. scGPT binning),
    # so the cells must be tokenized once and reused.
    data_loader = create_data_loader(original_cfg, adapter, test_mode=True)
    feature_selector = instantiate(cfg.steering.feature_selection)
    for seed in tqdm(cfg.steering.seeds):
        for n_features in tqdm(cfg.steering.n_features_list):
            for clamp_value in tqdm(cfg.steering.clamp_values):
                log.info(f"Selected {n_features} features for trial {seed} with clamp {clamp_value}")
                selected_features = feature_selector.select_features(
                    n_features=n_features,
                    dict_size=original_cfg.sae.dict_size,
                    trial=seed
                )
                original_emb, steered, reconstruction, cell_idx = steer(
                    adapter=adapter,
                    sae=sae,
                    data_loader=data_loader,
                    clamp_sets={"steered": [Clamp(selected_features, clamp_value)]},
                    device=device,
                    inference_cfg=cfg.get("sae_inference", None),
                    add_reconstruction_error=cfg.steering.get("add_reconstruction_error", False),
                    return_reconstruction=cfg.steering.get("save_reconstruction_baseline", False),
                    seed=seed,
                )
                np.save(output_dir / f"{feature_selector.name}_n{n_features}_seed{seed}_clamp{clamp_value}{suffix}.npy",
                        steered["steered"])

        np.save(output_dir / f"original_seed{seed}.npy", original_emb)
        if reconstruction is not None:
            np.save(output_dir / f"recon_seed{seed}.npy", reconstruction)
    if cell_idx is not None:
        np.save(output_dir / "cell_idx.npy", cell_idx)  # rows of every saved embedding, as input positions
    return cell_idx


def run_induction(cfg, sae, original_cfg, adapter, device, output_dir, suffix):
    induce = cfg.steering.induce
    features = list(induce.features)
    if not features:
        raise ValueError("steering.induce.features is empty (see config/induction/ for examples)")
    if induce.get("max_activation_file") is None:
        raise ValueError("steering.induce.max_activation_file is required: activations of the reference "
                         "(treated training) cells, pooled .pt/.npy or a generate_features .h5")
    max_activation = load_max_activation(induce.max_activation_file, features)
    log.info(f"Inducing features {features} (max activations {np.round(max_activation, 3).tolist()})")

    negative = [Clamp(list(induce.negative_features), induce.negative_value)] if induce.negative_features else []
    clamp_sets = {
        f"steered_f{f}_v{v}": [Clamp(features, v * max_activation, fraction=f)] + negative
        for f in induce.fractions for v in induce.multipliers
    }
    data_loader = create_data_loader(original_cfg, adapter, test_mode=True)
    original_emb, steered, reconstruction, cell_idx = steer(
        adapter=adapter,
        sae=sae,
        data_loader=data_loader,
        clamp_sets=clamp_sets,
        device=device,
        inference_cfg=cfg.get("sae_inference", None),
        add_reconstruction_error=cfg.steering.get("add_reconstruction_error", False),
        return_reconstruction=cfg.steering.get("save_reconstruction_baseline", False),
        seed=induce.get("seed", 0),
    )
    np.save(output_dir / "original.npy", original_emb)
    for name, embedding in steered.items():
        np.save(output_dir / f"{name}{suffix}.npy", embedding)
    if reconstruction is not None:
        np.save(output_dir / "recon.npy", reconstruction)
    if cell_idx is not None:
        np.save(output_dir / "cell_idx.npy", cell_idx)
    return cell_idx


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="steer", version_base='1.3')
def main(cfg: DictConfig) -> None:

    # Setup logging
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    output_dir = Path(hydra_cfg['runtime']['output_dir'])
    logging.basicConfig(filename=output_dir / "steering.log", level=logging.INFO)
    log.info("Starting steering with configuration:")
    log.info("\n" + OmegaConf.to_yaml(cfg))

    sae, original_cfg, adapter, device = load_run(cfg, REPO_ROOT)
    original_cfg.data.preprocess.split = cfg.steering.get("data_split", 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_err" if cfg.steering.get("add_reconstruction_error", False) else ""
    mode = cfg.steering.get("mode", "suppress")
    if mode == "suppress":
        cell_idx = run_suppression(cfg, sae, original_cfg, adapter, device, output_dir, suffix)
    elif mode == "induce":
        cell_idx = run_induction(cfg, sae, original_cfg, adapter, device, output_dir, suffix)
    else:
        raise ValueError(f"steering.mode must be 'suppress' or 'induce', got {mode!r}")
    save_cells(output_dir, original_cfg, cfg.steering.get("cell_columns", []), cell_idx)


if __name__ == "__main__":
    main()
