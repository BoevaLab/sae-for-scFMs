"""
Cell embeddings of an scFM without an SAE, e.g. the DAR baseline of the batch-correction
benchmark (scGPT finetuned with DAR), which needs no SAE.

Writes them where and as steer_features writes its unsteered embeddings, so benchmark_integration
reads them with `original: true`: experiments/steer/<experiment>/s<seed>/ with
original_seed<seed>.npy, cell_idx.npy (input position of each row) and cells.csv. Preprocessing and
seeding follow SAE training (config/train.yaml), so the embeddings equal the unsteered ones of an
SAE run with the same scFM, data and seed.
"""
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_class

REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.data_loader import create_data_loader  # noqa: E402
from sae4scfm.core.pipeline import register_resolvers  # noqa: E402
from sae4scfm.core.utils import get_devices, set_seed  # noqa: E402
from scripts.steer_features import save_cells  # noqa: E402

register_resolvers()


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="embed_cells", version_base="1.3")
def main(cfg) -> None:
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_devices()[0]
    set_seed(cfg.seed)  # as load_run does before building the adapter (binning draws from it)
    adapter = get_class(cfg.scfm._target_)(cfg)
    adapter.load_model()
    adapter.model.to(device)

    embeddings, cell_ids = [], []
    with torch.no_grad():
        for batch in create_data_loader(cfg, adapter, test_mode=True):
            embeddings.append(adapter.generate_embeddings(batch, device_model=device))
            if isinstance(batch, dict) and "cell_idx" in batch:
                cell_ids.append(torch.as_tensor(batch["cell_idx"]).cpu())
    cell_idx = torch.cat(cell_ids).numpy() if cell_ids else None
    order = np.argsort(cell_idx, kind="stable") if cell_idx is not None else slice(None)

    original = torch.vstack(embeddings).cpu().numpy()[order]
    np.save(output_dir / f"original_seed{cfg.seed}.npy", original)
    if cell_idx is not None:
        cell_idx = cell_idx[order]
        np.save(output_dir / "cell_idx.npy", cell_idx)
    save_cells(output_dir, cfg, cfg.cell_columns, cell_idx)
    print(f"[embed_cells] {len(original)} cells -> {output_dir}")


if __name__ == "__main__":
    main()
