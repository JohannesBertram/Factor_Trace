# CLAUDE.md

## Project Overview

Research into neural network interpretability via **Backward Factor Trace (BFT)** —
decomposing a trained network's computation by tracing weight-activation products
backward from the output to the input. At each layer, a joint arbor matrix is
factorised with NMF to identify interpretable computational pathways ("scaffolds")
that reveal which neuron groups drive each class prediction.

Supports MLP, CNN, and Transformer architectures. Primary datasets: MNIST (odd/even),
CIFAR-10.

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
│   ├── models.py            # SimpleMLP, SmallCNN
│   ├── training.py          # train_epoch, evaluate, correct, label transforms
│   ├── data_utils.py        # get_mnist_loaders, get_cifar10_loaders, collect_activations,
│   │                        #   collect_layer_inputs, collect_layer_inputs_generic,
│   │                        #   collect_layer_data
│   ├── bft.py               # BFT core: run_nmf, full_nmf_pipeline, auto_nmf_pipeline,
│   │                        #   compute_joint_arbors_normalized, compute_conv/attn_joint_arbors,
│   │                        #   trace_single_layer, bft (main entry point), nodes_at_layer
│   ├── r1d.py               # Rank-1 decomposition: run_r1d, run_r1d_sparse, r1d, r1d_sparse
│   ├── scaffold_utils.py    # bft_node_vals, build_scaffold_edges, scaffold_loading_from_edges
│   ├── fingerprint_utils.py # Factor fingerprints and NNLS projection:
│   │                        #   extract_factor_fingerprint, extract_fingerprint_matrix,
│   │                        #   compute_stimulus_similarity, compute_fingerprints,
│   │                        #   project_stimuli_onto_tree, project_onto_bft
│   ├── ablation_utils.py    # assign_factors_to_classes, ablate_model, run_ablation_sweep,
│   │                        #   taylor_scores, magnitude_scores, per_class_accuracy
│   ├── checkpoint.py        # save_experiment, load_experiment, MODEL/TRANSFORM/DATASET_REGISTRY
│   ├── neuron_analysis.py   # synaptic arbors, PCA decoding, per-class plots
│   └── plot_utils.py        # plot_nmf_component, plot_factor_graph, plot_scaffold_graph
├── notebooks/
│   ├── 01_mnist_odd_vs_even.ipynb   # MNIST even/odd baseline
│   ├── 02_factor_trace.ipynb        # BFT factor trace (MLP)
│   ├── 03_ablation.ipynb            # Ablation / causal intervention study
│   ├── 04_cifar10_trace.ipynb       # BFT on CIFAR-10 CNN
│   ├── 05_transformer_trace.ipynb   # BFT on ViT (tiny MNIST transformer)
│   ├── 06_stimulus_factor_analysis.ipynb  # Stimulus fingerprints & OOD projection
│   ├── 08_cifar10_stimulus_analysis.ipynb
│   ├── 09_nmf_comparison.ipynb
│   ├── 10_architecture_search.ipynb
│   └── 11_nmf_benchmark.ipynb
├── scripts/
│   └── train.py             # CLI training script
├── experiments/             # Saved model checkpoints (weights.pt + config.json)
│   ├── mnist_even_odd_mlp_*/
│   └── mnist_even_odd_vit_tiny/
├── figs/                    # Output figures organised by notebook number
├── data/                    # Datasets downloaded automatically on first run
├── requirements.txt
└── CLAUDE.md
```

## Code Architecture

All shared logic lives in `src/`. Notebooks import via:

```python
import sys; sys.path.insert(0, '..')
from src import *   # or selective imports
```

### `src/` modules

| Module | Key symbols |
|--------|-------------|
| `models.py` | `SimpleMLP`, `SmallCNN` |
| `training.py` | `train_epoch`, `evaluate`, `correct`, `label_transform_even_odd` |
| `data_utils.py` | `get_mnist_loaders`, `get_cifar10_loaders`, `collect_activations`, `collect_layer_inputs`, `collect_layer_inputs_generic`, `collect_layer_data` |
| `bft.py` | `bft`, `full_nmf_pipeline`, `auto_nmf_pipeline`, `run_nmf`, `run_nmf_minibatch`, `normalize_factors`, `sort_by_lambda`, `compute_joint_arbors_normalized`, `compute_conv_joint_arbors`, `compute_attn_joint_arbors`, `trace_single_layer`, `nodes_at_layer` |
| `r1d.py` | `run_r1d`, `run_r1d_sparse`, `run_r1d_sparse2`, `r1d`, `r1d_sparse`, `rec_err_curve` |
| `scaffold_utils.py` | `bft_node_vals`, `build_scaffold_edges`, `scaffold_loading_from_edges`, `scaffold_layer_sizes_from_edges` |
| `fingerprint_utils.py` | `extract_factor_fingerprint`, `extract_fingerprint_matrix`, `compute_stimulus_similarity`, `compute_fingerprints`, `project_stimuli_onto_tree`, `project_onto_bft` |
| `ablation_utils.py` | `assign_factors_to_classes`, `ablate_model`, `run_ablation_sweep`, `taylor_scores`, `magnitude_scores`, `act_magnitude_scores`, `per_class_accuracy`, `path_from_child`, `extract_importance_scores`, `all_weight_keys` |
| `checkpoint.py` | `save_experiment`, `load_experiment`, `get_transform`, `get_loaders_from_config`, `MODEL_REGISTRY`, `TRANSFORM_REGISTRY`, `DATASET_REGISTRY` |
| `neuron_analysis.py` | `compute_synaptic_arbors`, `pca_decoding`, `plot_mean_arbors`, `single_neuron_report` |
| `plot_utils.py` | `plot_nmf_component`, `plot_factor_graph`, `plot_scaffold_graph`, `plot_nmf_scree`, `extract_tree_nodes`, `compute_node_activations`, `plot_factor_tree`, `extract_factor_tree_nodes`, `compute_factor_activations`, `top_stimuli_factor_activations` |

### BFT calling conventions

`bft()` supports two modes:

**Model-protocol mode** (SimpleMLP-style):
```python
bft(model, layer_inputs_list, ...)
# model must expose linear_layer_indices() and model.layers[i].weight
```

**Layer-dict mode** (CNN, Transformer, any architecture):
```python
bft(layer_dicts, ...)
# layer_dicts: list of dicts, one per layer, forward order:
# {'type': 'fc'|'conv'|'attn', 'weight': ndarray, 'input_fmap': ndarray,
#  'attn_weights': ndarray}  # attn_weights only for 'attn' layers
```

### Experiment checkpoints

Experiments are saved as directories under `experiments/` with:
- `weights.pt` — `model.state_dict()`
- `config.json` — arch, dataset, analysis params

Use `save_experiment` / `load_experiment` from `checkpoint.py`. Registries
(`MODEL_REGISTRY`, `DATASET_REGISTRY`) control which architectures/datasets
are supported.

### Adding a new model

1. Add the class to `src/models.py`.
2. Register it in `checkpoint.py`'s `MODEL_REGISTRY`.
3. For BFT: implement `linear_layer_indices()` + `model.layers[i].weight`, or use layer-dict mode.

### Adding a new dataset

1. Write a loader in `src/data_utils.py` returning `(train_loader, test_loader)`.
2. Register it in `checkpoint.py`'s `DATASET_REGISTRY`.

## Analysis Pipeline

1. **Train** — `scripts/train.py` or `train_epoch` / `evaluate` in `src/training.py`
2. **Save checkpoint** — `save_experiment(model, config, path)` → `experiments/<name>/`
3. **Collect layer data** — `collect_layer_data` / `collect_layer_inputs_generic` (hook-based, works with any model)
4. **Run BFT** — `bft(layer_dicts, n_components=K, n_branches=B)` → tree of nodes
5. **Visualise scaffold** — `plot_scaffold_graph`, `plot_factor_graph`, `bft_node_vals`
6. **Stimulus analysis** — `extract_factor_fingerprint`, `project_stimuli_onto_tree` for OOD/fingerprint comparisons
7. **Ablation** — `run_ablation_sweep` with `taylor_scores` / `magnitude_scores` to validate causal claims
