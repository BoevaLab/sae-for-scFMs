# Sparse autoencoders for single-cell foundation models

Code for *Interpreting and steering single-cell foundation models with sparse autoencoders*.

Sparse autoencoders (SAEs) are trained on the residual stream of **scGPT**, **scFoundation** and
**Geneformer V2**, their features are characterized at the cell and gene level, and the features are
used to steer the models:

- **Suppress** batch-associated features and batch effects are corrected (Fig. 2)
- **Induce** drug-associated features and control cells move toward treated states (Fig. 3)
- **Read** features as gene programs and cell-level concepts (Fig. 4, 5)

## What is in here

| path | contents |
|---|---|
| `sae4scfm/` | the library: SAE architectures (`sae/`), scFM adapters (`scfm/`), data loading, activation buffer, inference, steering and analysis (`core/`) |
| `scripts/` | one script per pipeline step, each reading the Hydra config of the same name |
| `config/` | one config per script, plus the groups `scfm`, `data`, `sae`, `buffer`, `sae_inference`, `induction`, `programs` |
| `jobs/` | SLURM jobs (`job_<step>`) and the drivers of the paper's experiments (`batch_jobs/`) |
| `figures/` | one notebook per figure, reading script outputs via `figures/paths.yaml` |
| `tests/` | unit tests and a golden-output regression harness |

## Setup

The code runs in three containers, one per scFM, because their dependencies conflict: `scgpt`,
`scfoundation` and `geneformer`. Scripts that do not load an scFM (analysis, enrichment, summaries)
run in any of them; enrichment also needs `blitzgsea` (`pip install blitzgsea`, or the image in
`tests/regression/docker/enrich.Dockerfile`).

Models and datasets go in `data/` (not tracked):

```
data/
  scgpt/                    args.json, best_model.pt, vocab.json
  scfoundation/             models.ckpt, OS_scRNA_gene_index.19264.tsv
  geneformer/               model.safetensors, config.json, token_dictionary_gc104M.pkl, gene_name_id_dict_gc104M.pkl
  <scfm>_finetuned_<data>/  finetuned models (config/scfm/*_finetuned_*.yaml)
  <dataset>/                the .h5ad files named in config/data/<dataset>.yaml
  gene_sets/                MSigDB h.json, c2.json, c5.json
```

## Quick start

Run everything from the repository root. Arguments are Hydra overrides.

```bash
# 1. train an SAE on a model's activations
python -m scripts.train_sae scfm=scgpt data=pancreas sae=batchtopk sae.target_layer=5

# 2. encode a dataset with it
python -m scripts.generate_features sae_checkpoint.experiment=X sae_checkpoint.timestamp=Y

# 3. score the features against cell labels
python -m scripts.analyze_features sae_checkpoint.experiment=X sae_checkpoint.timestamp=Y

# 4. steer the model with them
python -m scripts.steer_features sae_checkpoint.experiment=X sae_checkpoint.timestamp=Y
```

Every step after training names its SAE with `sae_checkpoint.experiment` and
`sae_checkpoint.timestamp` (the run folder `experiments/train/<experiment>/<timestamp>`) and uses the
dataset the SAE was trained on; `+data=<dataset>` switches to another one. Training with
`hydra.run.dir=experiments/train/<experiment>/<name>` gives a run a fixed name instead of a timestamp.

On a SLURM cluster the same overrides go to `jobs/job_<step>`, which runs the step in
`$CONTAINERS/<image>.sif` (default `containers/`):

```bash
IMAGE=geneformer sbatch jobs/job_train scfm=geneformer-v2 scfm.model_file=model.safetensors data=pancreas
sbatch jobs/job_analyze sae_checkpoint.experiment=X sae_checkpoint.timestamp=Y +data=covid_evaluation
```

## Pipeline steps

| step | script | output |
|---|---|---|
| finetune an scFM (optional) | `scgpt_finetune`, `scfoundation_finetune`, `geneformer_tokenize` + `geneformer_finetune` | a finetuned model in `data/` |
| train an SAE | `train_sae` | `experiments/train/` (model.pth, eval_metrics.json) |
| encode a dataset | `generate_features` | `cached_features/` (token level `.h5`, or per-cell max with `output.format=pooled`) |
| score features against labels | `analyze_features` | `experiments/analysis/` (AMI per sublabel, threshold, direction) |
| gene-level statistics | `gene_selectivity` | `experiments/selectivity/` (accumulators, features.parquet) |
| pathway enrichment | `enrichment_analysis`, `enrichment_selection` | `experiments/enrichment/`, `experiments/enrichment_selection/` |
| variance partition, typology, composition | `commonality_analysis`, `feature_typology`, `celltype_composition` | `experiments/commonality/`, `typology/`, `composition/` |
| case studies | `feature_case_studies` | `experiments/case_studies/` |
| steering | `steer_features` | `experiments/steer/` (original and steered embeddings, cells.csv) |
| embeddings without an SAE (DAR baseline) | `embed_cells` | `experiments/steer/<experiment>/s<seed>/` |
| induction scores | `score_perturbation` | `<steer run>/perturbation_scores/<drug>/` |
| batch-integration benchmark | `benchmark_integration`, `summarize_benchmark` | `experiments/benchmark/`, `experiments/summary/` |

## Configuration notes

- **Layers are 0-based** (`sae.target_layer`); the paper counts from 1, so paper layer L is
  `sae.target_layer=L-1`. The hook reads the residual stream after that layer's final normalization.
- **SAE architecture** (`sae=`): `batchtopk` (default, used in the paper), `matryoshka`, `standard`,
  `gated`. The last two use an L1 penalty instead of k: pass `~sae.hyperparams.k`.
- **SAE inference** (`sae_inference=`): `batchtopk` (default) keeps the k * n largest activations
  over the n real tokens of a batch; `threshold` keeps activations above the threshold learned during
  training. Padding never takes part.
- **Steering** (`config/steer.yaml`): `steering.mode=suppress` clamps the selected features wherever
  they fire; `steering.mode=induce` writes them onto a random fraction f of each cell's gene tokens at
  v times their maximum activation.
  `steering.add_reconstruction_error=true` keeps the SAE's reconstruction error, so only the clamped
  directions change.
- **Labels** (`analyze_features`): `analysis.label_scoring.label` lists obs columns;
  `data.label_columns` maps label names to columns where they differ.
- **Tahoe cell selection** (`data.preprocess`): `dose`, `skip_cell_lines`, `drop_cell_lines` /
  `keep_cell_lines` and `balance`, passed after `+data=<config>`.
