"""
Shared setup for the pipeline scripts: config resolvers and loading a trained SAE run.
"""
from pathlib import Path

from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from sae4scfm.core.io_utils import load_sae_checkpoint, load_training_summary
from sae4scfm.core.sae_inference import prepare_sae
from sae4scfm.core.utils import get_devices, set_seed


def register_resolvers() -> None:
    """Register the OmegaConf resolvers used in the training config (safe to call repeatedly)."""
    OmegaConf.register_new_resolver(
        "compute_dict_size",
        lambda multiplier, activation_dim: round(multiplier * activation_dim),
        replace=True,
    )
    OmegaConf.register_new_resolver(
        "compute_steps",
        lambda n_cells, ctx_len, batch_size, n_eval_batches, refresh_batch_size: (
            int((sum(n_cells) * ctx_len / batch_size) - (n_eval_batches * refresh_batch_size * ctx_len / batch_size))
        ),
        replace=True,
    )


def checkpoint_dir(cfg: DictConfig, repo_root: Path) -> Path:
    """Directory of the trained run named by cfg.sae_checkpoint."""
    return Path(repo_root) / "experiments/train" / cfg.sae_checkpoint.experiment / cfg.sae_checkpoint.timestamp


def apply_data_override(original_cfg: DictConfig, cfg: DictConfig) -> DictConfig:
    """Swap in `cfg.data` when the script config provides one, keeping the run's batch size and
    HVG setting."""
    data = cfg.get("data", None)
    if data is not None:
        if "name" not in data:
            raise ValueError("data.* overrides need a data config to apply to: pass +data=<name> before them "
                             "(e.g. +data=tahoe_panobinostat_all ++data.preprocess.dose=0.5)")
        data.preprocess.batch_size = original_cfg.data.preprocess.batch_size
        data.preprocess.subset_hvg = original_cfg.data.preprocess.subset_hvg
        original_cfg.data = data
        print(f"Using {original_cfg.data.name} dataset")
    return original_cfg


def run_config(cfg: DictConfig, repo_root: Path) -> DictConfig:
    """Training config of the run named by cfg.sae_checkpoint, with the data override applied.
    Needs no scFM packages (unlike `load_run`)."""
    _, original_cfg = load_sae_checkpoint(checkpoint_dir(cfg, repo_root), "cpu", load_sae=False)
    return apply_data_override(original_cfg, cfg)


def load_run(cfg: DictConfig, repo_root: Path, load_sae: bool = True, load_model: bool = True):
    """Load a trained SAE run and the scFM adapter it was trained with.

    Loads the checkpoint and its training config, applies `cfg.sae_inference` to the SAE,
    seeds with the training seed, and swaps in `cfg.data` when the script config provides
    one (keeping the run's batch size and HVG setting). With `load_model`, the scFM is
    loaded, the activation hook registered and the model moved to the device.

    Returns:
        (sae, original_cfg, adapter, device); sae is None when `load_sae` is False.
    """
    device = get_devices()[0]
    run_dir = checkpoint_dir(cfg, repo_root)
    sae, original_cfg = load_sae_checkpoint(run_dir, device, load_sae=load_sae)
    if sae is not None:
        prepare_sae(sae, cfg.get("sae_inference", None), original_cfg, load_training_summary(run_dir))
    set_seed(original_cfg.seed)
    apply_data_override(original_cfg, cfg)

    adapter = get_class(original_cfg.scfm._target_)(original_cfg)
    if load_model:
        adapter.load_model()
        adapter.setup_hook()
        adapter.model.to(device)
    return sae, original_cfg, adapter, device
