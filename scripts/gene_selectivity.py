"""
Gene-level statistics of SAE features.

Encodes the evaluation subset, accumulates per (feature, gene) activation sums, counts and
activation-weighted expression moments (sae4scfm/core/gene_stats.py), and writes to
experiments/selectivity/<experiment>/<timestamp>/:

- the raw accumulators (act_sum.npz, act_expr_sum.npz, act_expr_sq_sum.npz, active_count.npz,
  gene_occurrence.npy, cell_moments.npy, meta.json), which enrichment_analysis also reads;
- features.parquet, one row per feature (topk{k}_mass, n_eff_genes, expr_focus, expr_std,
  expr_cv, mass_on_supported, n_eff_cells, total_active_count; see `feature_table`).

The expression value of a token is its model input: the binned value for scGPT, the
log-normalized value for scFoundation, and the token's position in the rank-ordered sequence
for Geneformer (so lower expr_focus means more highly expressed genes there).
"""
import os
import sys
from pathlib import Path

import hydra
import torch
from tqdm import tqdm

# Ensure working directory is the repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from sae4scfm.core.data_loader import create_data_loader
from sae4scfm.core.gene_stats import FeatureGeneAccumulator, GeneStats, feature_table
from sae4scfm.core.pipeline import load_run, register_resolvers
from sae4scfm.core.sae_inference import encode_tokens

register_resolvers()


def token_inputs(batch, token_mask):
    """Gene ids and expression values aligned with the encoded tokens, both [B, T]."""
    T = token_mask.shape[1]
    genes = batch["genes"]
    if isinstance(genes, list):  # Geneformer: variable-length sequences
        genes = torch.nn.utils.rnn.pad_sequence(genes, batch_first=True, padding_value=-1)
    genes = genes[:, :T].long()  # scFoundation crops padding columns before the model
    if "values" in batch:
        values = batch["values"][:, :T].float()
    else:  # rank-based input: the position is the expression axis
        values = torch.arange(T, dtype=torch.float32).expand(genes.shape[0], T)
    return genes, values


@hydra.main(config_path=str(REPO_ROOT / "config"), config_name="selectivity", version_base="1.3")
def main(cfg) -> None:
    sae, original_cfg, adapter, device = load_run(cfg, REPO_ROOT)
    data_loader = create_data_loader(original_cfg, adapter, test_mode=True)
    save_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    accumulator = FeatureGeneAccumulator(original_cfg.sae.dict_size, len(adapter.vocab), device=cfg.get("accum_device", "cpu"))
    special = torch.tensor(sorted(adapter.special_token_ids()))
    for batch in tqdm(data_loader):
        activations = adapter.generate_activations(batch, device)
        token_mask = adapter.generate_activation_mask(batch).to(device)
        with torch.no_grad():
            features = encode_tokens(sae, activations, token_mask, cfg.get("sae_inference", None))
        genes, values = token_inputs(batch, token_mask)
        token_mask = token_mask.cpu()
        gene_mask = token_mask & ~torch.isin(genes, special)
        accumulator.update(features, genes, values, gene_mask, token_mask)

    accumulator.save(save_dir)
    table = feature_table(GeneStats(save_dir), min_support=cfg.table.min_support, top_k=list(cfg.table.top_k))
    table.to_parquet(save_dir / "features.parquet")
    print(f"[gene_selectivity] accumulators and features.parquet -> {save_dir}")


if __name__ == "__main__":
    main()
