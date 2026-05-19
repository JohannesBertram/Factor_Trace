# CLAUDE.md

## Project Overview

Research into neural network interpretability via **weight-activation products** and
**NMF factorization** of joint layer-activation patterns.  The core idea: decompose
the matrix of neuron activations across layers into interpretable factors ("network
scaffolding") that reveal which groups of neurons drive classification of each class.

## Python Environment

| venv | Python | Use |
|------|--------|-----|
| `.venv` | 3.13 | All notebooks |

```bash
source .venv/bin/activate
```

When running Python from the CLI use `.venv/bin/python`.

## Repository Layout

```
Weight_Interpretability/
├── src/
│   ├── __init__.py          # re-exports all public symbols
│   ├── models.py            # SimpleMLP (configurable, returns intermediate activations)
│   ├── training.py          # train_epoch, evaluate, correct, label transforms
│   ├── data_utils.py        # get_mnist_loaders, collect_activations
│   ├── factorization.py     # run_nmf, normalize_factors, full_nmf_pipeline
│   ├── neuron_analysis.py   # synaptic arbors, PCA decoding, per-class plots
│   └── plot_utils.py        # plot_nmf_component, plot_factor_graph
├── notebooks/
│   └── 01_mnist_odd_vs_even.ipynb   # interactive analysis (imports from src/)
├── data/                    # MNIST downloaded automatically on first run
├── requirements.txt
└── CLAUDE.md
```

## Code Architecture

All shared logic lives in `src/`.  Notebooks import via:

```python
import sys; sys.path.insert(0, '..')
from src import *   # or selective imports
```

### `src/` modules

| Module | Key symbols |
|--------|-------------|
| `models.py` | `SimpleMLP` |
| `training.py` | `train_epoch`, `evaluate`, `correct`, `label_transform_even_odd` |
| `data_utils.py` | `get_mnist_loaders`, `collect_activations` |
| `factorization.py` | `full_nmf_pipeline`, `run_nmf`, `normalize_factors`, `sort_by_lambda` |
| `neuron_analysis.py` | `compute_synaptic_arbors`, `pca_decoding`, `plot_mean_arbors`, `single_neuron_report` |
| `plot_utils.py` | `plot_nmf_component`, `plot_factor_graph` |

### Adding a new model

1. Add the class to `src/models.py` (must implement `forward(x, inference=False) -> (output, feature_maps)`).
2. The rest of the pipeline (`collect_activations`, NMF, neuron analysis) is model-agnostic.

### Adding a new dataset

1. Write a loader in `src/data_utils.py` returning a `(DataLoader, DataLoader)` pair.
2. Pass the dataset to `collect_activations` with appropriate `layer_indices` and `label_transform`.

## Analysis Pipeline

1. **Train** — `train_epoch` / `evaluate` in `src/training.py`
2. **Collect activations** — `collect_activations` captures post-activation outputs at specified layer indices for correctly classified samples
3. **NMF** — `full_nmf_pipeline` returns `(img_factors, neural_factors, lambdas)`
4. **Visualize factors** — `plot_nmf_component` and `plot_factor_graph`
5. **Single-neuron analysis** — `single_neuron_report` gives PCA decoding + mean arbors + histogram

## Original notebook

`mnist-mlp-2-hidden-layers-new-idea-odd-vs-even.ipynb` at the repo root is the
original self-contained prototype.  It is preserved for reference; use
`notebooks/01_mnist_odd_vs_even.ipynb` for ongoing work.
