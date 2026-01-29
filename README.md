# SAE for Single-Cell Foundation Models (scFMs)

## Overview

This project provides tools to train sparse autoencoders on activations from various single-cell foundation models (scFMs), enabling mechanistic interpretability of these models. By decomposing neural network activations into sparse, interpretable features, we can better understand what biological patterns and cell states these models have learned.

## Features

- **Multiple SAE Architectures**: Support for various sparse autoencoder variants including:
  - Standard SAE with L1 regularization
  - Top-K SAE
  - Batch Top-K SAE
  - Matryoshka Batch Top-K SAE
  - Gated SAE
  - JumpReLU SAE
  - P-Anneal and Gated-Anneal SAE

- **Multiple scFM Adapters**: Compatible with leading single-cell foundation models:
  - scGPT
  - scFoundation
  - Geneformer

- **Comprehensive Analysis Tools**:
  - Label scoring (cell type, batch associations)
  - Gene scoring and enrichment analysis
  - Expression pattern analysis
  - Feature density scoring
  - Feature steering and manipulation

- **Efficient Training Pipeline**:
  - Parallel data loading with activation buffering
  - Multi-GPU support
  - Hydra-based configuration management
  - Weights & Biases integration for experiment tracking

## Quick Start

### Training an SAE

Train a sparse autoencoder on a foundation model's activations:

```bash
python scripts/train_sae.py
```

The training script uses Hydra for configuration. You can override parameters:

```bash
python scripts/train_sae.py \
  scfm=scfoundation \
  data=pbmc \
  sae=batchtopk \
  sae.target_layer=5 \
  sae.dictionary_multiplier=0.66666
```

### Generating Features

Extract SAE features for downstream analysis:

```bash
python scripts/generate_features.py \
  sae_checkpoint.experiment=layer_sweeps \
  sae_checkpoint.timestamp=Jan29-10-00
```

### Analyzing Features

Run interpretability analysis on extracted features:

```bash
python scripts/analyze_features.py \
  sae_checkpoint.experiment=layer_sweeps \
  sae_checkpoint.timestamp=Jan29-10-00 \
  analysis.run_label_scoring=true \
  analysis.run_gene_scoring=true
```

### Feature Steering

Manipulate model behavior using learned SAE features:

```bash
python scripts/steer_features.py \
  sae_checkpoint.experiment=layer_sweeps \
  sae_checkpoint.timestamp=Jan29-10-00
```

## Configuration

The project uses Hydra for hierarchical configuration management. Configurations are organized in [config/](config/):

- [config/train.yaml](config/train.yaml) - Main training configuration
- [config/scfm/](config/scfm/) - Foundation model configurations
- [config/sae/](config/sae/) - SAE architecture configurations
- [config/data/](config/data/) - Dataset configurations (PBMC, COVID, Census)
- [config/buffer/](config/buffer/) - Activation buffer configurations

### Key Configuration Parameters

**Training:**
- `sae.target_layer`: Which model layer to extract activations from
- `sae.dictionary_multiplier`: SAE hidden dimension as multiple of input dimension
- `sae.hyperparams.k`: Sparsity parameter (for Top-K variants)
- `sae.hyperparams.lr`: Learning rate
- `seed`: Random seed for reproducibility

**Data:**
- `data.name`: Dataset name (pbmc, covid, census)
- `data.n_cells`: Number of cells to use
- `data.preprocess.split`: Train/test split fraction
- `data.preprocess.subset_hvg`: Whether to subset to highly variable genes

## Project Structure

```
sae-for-scFMs/
├── config/              # Hydra configuration files
│   ├── data/           # Dataset configs
│   ├── sae/            # SAE architecture configs
│   ├── scfm/           # Foundation model configs
│   └── buffer/         # Data buffer configs
├── scripts/            # Entry point scripts
│   ├── train_sae.py           # Train sparse autoencoders
│   ├── generate_features.py  # Extract SAE features
│   ├── analyze_features.py   # Analyze feature interpretability
│   ├── steer_features.py     # Feature steering experiments
│   └── benchmark_integration.py
├── sae4scfm/           # Main package
│   ├── core/          # Core utilities
│   │   ├── buffer.py         # Activation buffering
│   │   ├── data_loader.py    # Data loading
│   │   ├── evaluation.py     # Evaluation metrics
│   │   ├── analysis.py       # Feature analysis
│   │   ├── steering.py       # Feature steering
│   │   └── io_utils.py       # I/O utilities
│   ├── sae/           # SAE implementations
│   │   ├── standard.py       # Standard SAE
│   │   ├── top_k.py         # Top-K SAE
│   │   ├── batch_top_k.py   # Batch Top-K SAE
│   │   ├── matryoshka_batch_top_k.py
│   │   ├── gdm.py           # Gated SAE
│   │   ├── jumprelu.py      # JumpReLU SAE
│   │   └── trainer.py       # Base trainer
│   └── scfm/          # Foundation model adapters
│       ├── base.py          # Abstract adapter interface
│       ├── scgpt/          # scGPT adapter
│       ├── scfoundation/   # scFoundation adapter
│       └── geneformer/     # Geneformer adapter
└── jobs/               # Job submission scripts
```

## Model Adapters

Each foundation model requires a specific adapter that implements the `ModelAdapter` interface defined in [sae4scfm/scfm/base.py](sae4scfm/scfm/base.py). Adapters handle:

- Model loading and initialization
- Data preprocessing for the specific model format
- Forward hook registration for activation extraction
- Model-specific embedding generation

Currently supported models:
- **scGPT**: Generative pre-trained transformer for single-cell RNA-seq
- **scFoundation**: Foundation model with performer architecture
- **Geneformer**: Transformer model trained on rank-value gene encodings

## SAE Architectures

The framework supports multiple SAE architectures optimized for different use cases:

- **Standard SAE**: Classic autoencoder with L1 sparsity penalty
- **Top-K SAE**: Fixed sparsity using top-k activation selection
- **Batch Top-K SAE**: Batch-level top-k for improved feature diversity
- **Matryoshka SAE**: Nested feature learning at multiple scales
- **Gated SAE**: Gating mechanism for improved reconstruction
- **JumpReLU SAE**: Jump ReLU activation for sharper features

See [sae4scfm/sae/](sae4scfm/sae/) for implementations.

## Analysis Capabilities

The framework provides comprehensive feature analysis tools:

- **Label Scoring**: Statistical association with cell types, batches, and other metadata
- **Gene Scoring**: Gene-level feature attribution and enrichment
- **Expression Scoring**: Relationship to gene expression patterns
- **Density Scoring**: Feature activation density across cells
- **Gene Family Analysis**: GSEA and gene set enrichment

Results are saved as structured CSV files with multi-level columns for easy downstream analysis.

## Experiment Tracking

Training runs are automatically tracked using:
- **Weights & Biases** (configurable, defaults to offline mode)
- **Hydra output directories** with timestamped runs
- **Checkpoint saving** for trained SAE models
- **Metric logging** for reconstruction quality and sparsity

## Citation

If you use this code in your research, please cite:

```bibtex
@software{sae4scfm2025,
  title={SAE for Single-Cell Foundation Models},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/sae-for-scFMs}
}
```