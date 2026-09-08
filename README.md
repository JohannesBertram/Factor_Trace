# Backward Factor Traces

A library for mechanistic interpretability of neural networks via **Backward Factor Trace (BFT)** — a technique that decomposes a network's computation into interpretable factors by tracing weight-activation products backward through layers and factorizing each layer's joint arbor matrix with Non-negative Matrix Factorization (NMF).

The four main analyses built on BFT are:

| Analysis | What it gives you |
|---|---|
| **BFT** | A tree of NMF factors tracing how each layer transforms its input |
| **Fingerprints** | Per-stimulus representations from the tree's top-2 slice |
| **Scaffold** | Network graph visualization weighted by BFT-derived activity |
| **Pruning** | Weight ablation guided by BFT importance scores |

An interactive circuit explorer for all five models lives in `docs/` (a static GitHub-Pages site).

---

## Repository layout

```
notebooks/   the five model notebooks (01-05) — the entire analysis pipeline
src/         the library the notebooks use
scripts/     train.py, run_nb.py, render_figures.py, compute_checklist_stats.py,
             build_validation_bundles.py, build_pruning_bundle.py
figures/     rendered paper figures + figdata/ (committed plot-data bundles)
docs/        interactive circuit-explorer website (build_data.py regenerates it)
figstyle.py  publication figure styling (with check_figure.py and .figstyle/)
```

**The figdata contract**: notebooks run where the data and models are, and export every array a figure could need into a small committed bundle (`figures/figdata/*.npz` + `.json`, via `src/figdata.py` / `src/figexport.py`). Figures are pure numpy+matplotlib functions over those bundles (`src/paper_figures.py`), so every paper figure rebuilds on any checkout — no GPU, no datasets, no models:

```bash
python scripts/render_figures.py            # redraw every paper figure
python scripts/render_figures.py --list     # what exists
python scripts/compute_checklist_stats.py   # every headline number + CIs, from bundles only
```

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10, PyTorch ≥ 2.1. Virtual environment lives at `.venv` in the repo root.

---

## Reproducing the analyses

Each notebook is self-contained and covers one model end to end, with the same section layout: §1 config, §2 BFT trace, §3 circuit-figure export, §4–§5 fingerprints + export, §6 hyperparameter check (held-out rank re-derivation), §7 validation suite, §8 causal pruning, §9 fingerprint separability.

| Notebook | Architecture | Dataset |
|---|---|---|
| `01_MLP_8_4_0134.ipynb` | SimpleMLP 784→8→4→2 | MNIST even/odd (digits 0, 1, 3, 4) |
| `02_MLP_40_20_digits.ipynb` | SimpleMLP 784→40→20→10 | MNIST |
| `03_CNN_CIFAR10.ipynb` | SmallCNN | CIFAR-10 |
| `04_ViT.ipynb` | TinyViT | MNIST even/odd |
| `05_imagenet_cnn.ipynb` | Pretrained SqueezeNet 1.1 | ImageNet (8 categories) |

```bash
source .venv/bin/activate
python scripts/train.py --help                       # train model checkpoints -> data/models/
python scripts/run_nb.py notebooks/01_MLP_8_4_0134.ipynb   # execute headlessly
```

MNIST/CIFAR-10 download automatically into `data/`; ImageNet needs the validation split at `data/val` (ImageFolder layout). Expensive intermediates (the fitted trees, §6–§9 results) cache to `data/cache/`, keyed by an explicit tag plus a hash of the hyperparameters — change an HP and the affected steps recompute; set `NB_NOCACHE=1` to bypass caching entirely. One notebook run writes all of its figure bundles: `nb0N_circuits`, `nb0N_fingerprints`, `nb09_<exp>_validation`, and the pruning bundle (guarded by a per-experiment observation floor so smoke runs never overwrite committed results). `scripts/build_validation_bundles.py` / `build_pruning_bundle.py` rebuild bundles from stored results JSONs (`logs/results/`, `data/results/`) without re-running anything.

The website is rebuilt from the circuit bundles with `python docs/build_data.py`.

---

## Quick Start (library)

```python
import sys; sys.path.insert(0, '.')
from src import *

# 1. Train or load a model
model = SimpleMLP(input_dim=784, hidden_dims=[20, 10], output_dim=2)
train_loader, test_loader = get_mnist_loaders(batch_size=64)
# ... train ...

# 2. Run BFT — returns a tree of NMF factors rooted at the output layer
result = bft(model, test_loader, k_max=5, n_branches=2, verbose=1)

# 3. Fingerprints — the tree's top-2 slice, one row per stimulus
fp_tree = truncate_tree(result, depth=2)
F = extract_fingerprint_matrix(fp_tree, range(result.root.img_factors.shape[0]))
S = compute_stimulus_similarity(F)

# 4. Scaffold — weighted network graph per BFT path
from src import build_scaffold_edges, scaffold_loading_from_edges, plot_scaffold_graph

# 5. Pruning experiment
ab = ablation_sweep(model, result, test_loader, target_class=0)
```

---

## Extending BFT: Custom Factorizations and Normalizations

`bft()` accepts two extension points that let you swap the matrix decomposition or the joint-arbor preprocessing without touching library code.

### Factorization interface

Any callable that matches the signature below can be passed as `factorization=`:

```python
def my_factorization(
    X: np.ndarray,       # (n_stimuli, n_features) non-negative joint arbor matrix
    n_components: int,   # exact number of factors to extract
    **params,            # contents of factorization_params dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    W       : (n_stimuli, n_components)   per-stimulus factor loadings
    H       : (n_features, n_components)  per-feature factor loadings
    lambdas : (n_components,)             factor importance, descending
    """
    ...
```

Pass keyword arguments for the factorizer via `factorization_params` (a plain dict). The legacy NMF params `random_state`, `max_iter`, `init`, `l1_ratio` are still accepted as top-level `bft()` kwargs for backward compatibility; `factorization_params` overrides them when both are given.

Built-in strings: `'nmf'` (default, MiniBatch NMF via sklearn).

### Normalization interface

Any callable that matches the signature below can be passed as `normalization=`. It is applied to the raw joint arbor matrix after construction but before the positive/negative split and before factorization:

```python
def my_normalization(
    joint: np.ndarray,   # (n_stimuli, n_features) raw joint arbor matrix
) -> np.ndarray:         # same shape, normalized
    ...
```

Built-in strings: `'none'` (default, no-op), `'l2_per_stimulus'`, `'l1_per_stimulus'`, `'frobenius'`.

### Worked example: L1-regularized NMF

```python
result = bft(
    model, test_loader,
    factorization='nmf',
    factorization_params={'l1_ratio': 0.5, 'max_iter': 300},
    normalization='l2_per_stimulus',
)
```

### Registering a built-in

Add your function to `_FACTORIZATIONS` or `_NORMALIZATIONS` in `src/bft.py` to make it available as a string key.

---

## Core Concepts

### BFT — Backward Factor Trace

**Module:** `src/bft.py` — primary entry point `bft(model, loader, ...)`.

BFT works by:
1. Running a forward pass through the model to collect layer inputs via hooks.
2. For each layer (from output to input), computing the *joint arbor matrix* — the outer product of each output neuron's weight row with its (normalized) input activations.
3. Factorizing that matrix with MiniBatchNMF to discover shared patterns across neurons.
4. Propagating the top factor's per-stimulus importance weights backward into the next layer, recursively.

The result is a tree: the root is the output layer, children point toward the input. Each `BFTNode` holds the NMF factors for one layer.

```python
# Primary (recommended) — hooks model automatically
result = bft(model, loader, k_max=5, n_branches=2)

# Layer-dict mode — pass pre-collected activations
layer_dicts = collect_layer_dicts(model, loader)['layer_data']
result = bft(layer_dicts)
```

Key parameters: `k_max` (NMF rank bound per layer, int or list), `n_branches` (top factors followed per layer), `only_correct` (keep only correctly classified samples, default True), `weighting` (`'img_selectivity'` default), `k_fixed`, `recon_threshold`, `verbose`.

```python
result.root          # BFTNode at the output layer
result.images        # (N, C, H, W) — input stimuli
result.targets       # (N,) — class labels

node = result.root
node.img_factors          # (N, K) — per-stimulus NMF loadings
node.connection_factors   # (n_out * n_in, K) — NMF basis (arbor space)
node.lambdas              # (K,) — component magnitudes, descending
node.children             # list[BFTNode] toward the input
```

Both excitatory (positive) and inhibitory (negative) arbors are factorized separately; inhibitory factors live in `node.neg_*` fields (`None` when the arbor has no negative region).

Architecture support: **FC** (`nn.Linear`), **Conv** (`nn.Conv2d`, im2col + spatial pooling), and **attention** (CLS-row scores collapse tokens into a weighted effective input). For non-sequential architectures (e.g. SqueezeNet), restrict capture with `layer_filter=lambda name, mod: ...`.

### Fingerprints

**Module:** `src/fingerprint_utils.py`

A *fingerprint* for a stimulus is the concatenation of its `img_factors` loadings across the nodes of the circuit tree's **top two levels** (`truncate_tree(result, depth=2)`) — the output-layer factors plus their immediate sub-circuits. Two stimuli with similar fingerprints activate the same pattern of factors.

```python
from src import truncate_tree, extract_fingerprint_matrix, compute_stimulus_similarity
from src import project_onto_bft

fp_tree = truncate_tree(result, depth=2)
F = extract_fingerprint_matrix(fp_tree, indices)     # (n, D)
S = compute_stimulus_similarity(F)                   # (n, n) cosine similarity

# Project new (held-out / OOD) stimuli onto the FIXED factors via NNLS —
# no NMF refitting; works with any DataLoader
fp_new = project_onto_bft(result, model, ood_loader)
```

### Scaffold

**Modules:** `src/scaffold_utils.py` + `src/plot_utils.py`

```python
from src import (build_scaffold_edges, scaffold_layer_sizes_from_edges,
                 scaffold_loading_from_edges, plot_scaffold_graph)

# nodes: list[BFTNode] in forward order (input-side first)
edges, neg_edges = build_scaffold_edges(nodes, fi='path', top_pct=0.05)
fig = plot_scaffold_graph(scaffold_loading_from_edges(edges), edges,
                          scaffold_layer_sizes_from_edges(edges),
                          neg_edge_matrices=neg_edges)
```

### Pruning / Ablation

**Modules:** `src/pruning.py` (driver) + `src/ablation_utils.py` (mechanics) + `src/bundles.py` (stats)

BFT assigns an importance score to every weight via the connection factors of the class-selective circuit. `run_pruning` sweeps (seed, target-class) observations; `bundles.pruning_bundle` computes all aggregate curves and paired tests and writes the figure bundle.

```python
from src import run_pruning, pruning_results_dict, pruning_bundle

reps = [{'seed': 0, 'model': model, 'tree': result,
         'layer_names': layer_names, 'targets': targets}]
res = run_pruning(reps, test_loader, target_classes=range(10),
                  fractions=(0.02, 0.05, 0.1, 0.2))
pruning_bundle('mlp_digit', pruning_results_dict('mlp_digit', res))
```

Methods: `bft_top` (most-important weights first — should damage the target class), `bft_bottom` (least-important first — should be benign), `random` (matched-count baseline, averaged over repeats). Lower-level blocks: `select_class_circuit`, `ablate_model`, `per_class_accuracy` (with `label_transform`/`pred_transform` hooks for mapped label spaces, e.g. ImageNet → super-categories).

---

## Citation & License

MIT licensed (see `LICENSE`). Citation information will follow publication.
