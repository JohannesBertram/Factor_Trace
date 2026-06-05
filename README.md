# Weight Interpretability

A library for mechanistic interpretability of neural networks via **Backward Factor Trace (BFT)** — a technique that decomposes a network's computation into interpretable factors by tracing weight-activation products backward through layers using Non-negative Matrix Factorization (NMF).

The four main analyses built on BFT are:

| Analysis | What it gives you |
|---|---|
| **BFT** | A tree of NMF factors tracing how each layer transforms its input |
| **Fingerprints** | Per-stimulus representations derived from BFT factor loadings |
| **Scaffold** | Network graph visualization weighted by BFT-derived activity |
| **Pruning** | Weight ablation guided by BFT importance scores |

---

## Installation

```bash
pip install -r requirements.txt
```

Python ≥ 3.10, PyTorch ≥ 2.1. Virtual environment lives at `.venv` in the repo root.

---

## Quick Start

```python
import sys; sys.path.insert(0, '.')
from src import *

# 1. Train or load a model
model = SimpleMLP(input_dim=784, hidden_dims=[20, 10], output_dim=2)
train_loader, test_loader = get_mnist_loaders(batch_size=64)
# ... train ...

# 2. Run BFT — returns a tree of NMF factors rooted at the output layer
result = bft(model, test_loader, k_max=5, n_branches=2, verbose=1)

# 3. Extract fingerprints — per-stimulus representation across the factor tree
fp = compute_fingerprints(result)          # fp.matrix (N, D), fp.similarity (N, N)

# 4. Build scaffold — weighted network graph per BFT path
from src.plot_utils import plot_scaffold
figs = plot_scaffold(result, model, test_loader)

# 5. Run pruning experiment
ab = ablation_sweep(model, result, test_loader, target_class=0)
```

---

## Notebooks

Five analysis notebooks in `notebooks/` illustrate the full pipeline on different architectures:

| Notebook | Architecture | Dataset |
|---|---|---|
| `01_MLP_8_4_0134.ipynb` | SimpleMLP | MNIST (even/odd) |
| `02_MLP_40_20_digit.ipynb` | SimpleMLP | MNIST |
| `03_CNN_CIFAR10.ipynb` | SmallCNN | CIFAR-10 |
| `04_ViT.ipynb` | TinyViT | MNIST |
| `05_imagenet_cnn.ipynb` | SmallCNN | ImageNet |

Each notebook covers: model training → BFT → fingerprints → scaffold → pruning. Run them in order within the `.venv` environment (`source .venv/bin/activate`).

---

## Core Concepts

### BFT — Backward Factor Trace

**Module:** `src/bft.py`  
**Primary entry point:** `bft(model, loader, ...)`

BFT works by:
1. Running a forward pass through the model to collect layer inputs via hooks.
2. For each layer (from output to input), computing the *joint arbor matrix* — the outer product of each output neuron's weight row with its (normalised) input activations.
3. Factorizing that matrix with MiniBatchNMF to discover shared patterns across neurons.
4. Propagating the top factor's per-stimulus importance weights backward into the next layer, recursively.

The result is a tree: the root is the output layer, children point toward the input. Each `BFTNode` holds the NMF factors for one layer.

#### Calling conventions

```python
# Primary (recommended) — hooks model automatically
result = bft(model, loader, k_max=5, n_branches=2)

# Layer-dict mode — pass pre-collected activations
layer_dicts = collect_layer_dicts(model, loader)['layer_data']
result = bft(layer_dicts)

# Legacy (SimpleMLP only)
result = bft(model, layer_inputs_list)
```

#### Key parameters

| Parameter | Default | Effect |
|---|---|---|
| `k_max` | `5` | Upper bound on NMF rank per layer (can be a list) |
| `n_branches` | `2` | How many top factors to follow at each layer |
| `only_correct` | `True` | Keep only correctly classified samples |
| `weighting` | `'img_selectivity'` | How stimulus weights are propagated backward |
| `k_fixed` | `None` | Fix NMF rank rather than auto-select |
| `recon_threshold` | `None` | Max acceptable relative Frobenius error for auto-rank |
| `verbose` | `0` | `1` = per-layer summary, `2` = timing detail |

#### BFTResult and BFTNode

```python
result.root          # BFTNode at the output layer
result.images        # (N, C, H, W) — input stimuli
result.targets       # (N,) — class labels
result.confidences   # (N,) — model confidence

node = result.root
node.img_factors          # (N, K) — per-stimulus NMF loadings
node.connection_factors   # (n_out * n_in, K) — NMF basis (arbor space)
node.lambdas              # (K,) — component magnitudes, descending
node.weight               # layer weight matrix
node.layer_type           # 'fc', 'conv', or 'attn'
node.children             # list[BFTNode] toward the input
```

Both excitatory (positive) and inhibitory (negative) arbors are factorized separately. Inhibitory factors live in `node.neg_img_factors`, `node.neg_connection_factors`, `node.neg_lambdas`.

#### Getting nodes at a specific layer

```python
from src.bft import nodes_at_layer
nodes = nodes_at_layer(result, target_layer_idx=0)   # 0 = input-side leaves
```

#### Architecture support

- **FC layers** (`nn.Linear`): standard joint arbor
- **Conv layers** (`nn.Conv2d`): im2col + spatial pooling (`avg`, `max`, or `center`)
- **Attention layers**: CLS-row attention scores collapse the token sequence into a weighted effective input before forming the joint arbor; pass `attn_weights` in the layer dict

For non-sequential architectures (e.g. SqueezeNet), use `layer_filter` to restrict capture:

```python
result = bft(model, loader,
             layer_filter=lambda name, mod: 'squeeze' in name)
```

---

### Fingerprints

**Module:** `src/fingerprint_utils.py`

A *fingerprint* for a stimulus is the concatenation of its `img_factors` loadings across every node in the BFT tree (BFS order). Two stimuli with similar fingerprints activate the same pattern of factors at every layer.

```python
from src.fingerprint_utils import compute_fingerprints, project_onto_bft

# Fingerprints for training stimuli
fp = compute_fingerprints(result)
# fp.matrix     (N, D)  — D = sum of K_i over all nodes
# fp.similarity (N, N)  — pairwise cosine similarity
# fp.indices    (N,)    — which sample indices were used

# Fingerprint for a subset
fp_sub = compute_fingerprints(result, indices=[0, 5, 12])

# Project new (held-out / OOD) stimuli onto fixed BFT factors via NNLS
# — no NMF refitting; works with any DataLoader
fp_new = project_onto_bft(result, model, ood_loader)
```

`project_onto_bft` mirrors the full BFT backward pass using NNLS, propagating per-factor importance weights the same way. Use it to test generalization without running a new BFT.

#### Lower-level helpers

```python
from src.fingerprint_utils import extract_fingerprint_matrix, compute_stimulus_similarity

mat = extract_fingerprint_matrix(result.root, indices)   # (n, D)
sim = compute_stimulus_similarity(mat)                   # (n, n) cosine similarity
```

---

### Scaffold

**Module:** `src/scaffold_utils.py` + `src/plot_utils.py`

The scaffold is a weighted network graph where:
- **Nodes** are neurons, sized and coloured by their factor-selective activity
- **Edges** between adjacent layers reflect BFT connection factors (excitatory in one colormap, inhibitory in another)

#### From a BFTResult (recommended)

```python
from src.plot_utils import plot_scaffold

figs = plot_scaffold(result, model, dataloader)
# Returns dict {path_tuple: Figure} — one figure per BFT leaf path

# Select specific paths
figs = plot_scaffold(result, model, dataloader, paths=[(0,), (1,)])

# Styling options
figs = plot_scaffold(result, model, dataloader,
                     normalize='layer',           # 'global' or 'layer'
                     magnitude_transform='sqrt',  # None | 'sqrt' | 'cbrt' | 'log'
                     max_nodes_per_layer=8,        # hide low-importance neurons
                     show_neg=True)               # draw inhibitory edges
```

#### From pre-computed edge matrices

`build_scaffold_edges` extracts the raw edge matrices from a list of BFT nodes:

```python
from src.scaffold_utils import (build_scaffold_edges, scaffold_layer_sizes_from_edges,
                                 scaffold_loading_from_edges)

# results: list[BFTNode] in forward order (input-side first)
edge_matrices, neg_edge_matrices = build_scaffold_edges(results, fi='path', top_pct=0.05)
layer_sizes = scaffold_layer_sizes_from_edges(edge_matrices)
loading     = scaffold_loading_from_edges(edge_matrices)

from src.plot_utils import plot_scaffold_graph
fig = plot_scaffold_graph(loading, edge_matrices, layer_sizes,
                           neg_edge_matrices=neg_edge_matrices)
```

`build_scaffold_edges` parameters:

| Parameter | Default | Effect |
|---|---|---|
| `fi` | `'path'` | Factor index per layer: int or `'path'` (follow BFT path) |
| `top_pct` | `0.05` | Fraction of top-loading stimuli for mean reconstruction |
| `use_reconstruction` | `True` | Mean reconstruction over top stimuli vs raw factor |
| `edge_threshold` | `0.0` | Zero edges below `threshold × layer_max` |
| `aggregate_conv` | `True` | Sum conv kernel spatial dims → (C_out, C_in) |

---

### Pruning / Ablation

**Module:** `src/ablation_utils.py`

BFT assigns an importance score to every weight via the connection factors of the class-selective circuit. These scores drive a pruning experiment: ablate the top-ranked weights and measure how class accuracy degrades compared to random ablation.

#### High-level entry point

```python
from src.ablation_utils import ablation_sweep, ablation_layer_sweep

# Single experiment: ablate globally
ab = ablation_sweep(
    model, result, test_loader,
    target_class=0,
    fractions=(0.05, 0.10, 0.20, 0.30, 0.50),
    methods=('bft_top', 'bft_bottom', 'magnitude', 'random'),
)
# ab.results   {method: {fraction: {class_id: accuracy}}}
# ab.baseline  {class_id: accuracy}

# Layer sweep: ablate last 1, 2, …, L layers
sweep = ablation_layer_sweep(model, result, test_loader, target_class=0)
# sweep  {depth: AblationResult}
```

`methods`:
- `bft_top` — ablate most-important weights first (should damage target class)
- `bft_bottom` — ablate least-important weights first (should be benign)
- `magnitude` — ablate largest-magnitude weights first (baseline)
- `random` — random ablation (averaged over `n_random_repeats`)

#### Visualize pruning results

```python
from src.plot_utils import plot_pruning_results, plot_pruning_by_layer_depth

# Restructure results for the plotting helper
pruning_data = {0: {**{f: ab.results[f] for f in ab.results}}}
# add fraction=0 baseline:
for method in pruning_data[0]:
    pruning_data[0][method][0.0] = ab.baseline

figs = plot_pruning_results(pruning_data, class_names=['even', 'odd'],
                             methods=['bft_top', 'bft_bottom', 'magnitude', 'random'],
                             fractions=[0.0, 0.05, 0.10, 0.20, 0.30, 0.50])

fig_depth = plot_pruning_by_layer_depth(sweep, fractions=[0.10, 0.20, 0.30],
                                         target_class=0, class_names=['even', 'odd'],
                                         methods=['bft_top', 'magnitude', 'random'])
```

#### Lower-level building blocks

```python
from src.ablation_utils import select_class_circuit, ablate_model, per_class_accuracy

# Extract importance scores for one class
scores, info = select_class_circuit(result, result.targets, class_d=0)
# scores: {(layer_idx, i, j): float}
# info:   {'k_star', 'selectivity', 'all_selectivities', 'is_selective', 'warning'}

# Zero the top 20% of weights
n_ablate = int(0.20 * len(scores))
ranked   = sorted(scores, key=scores.get, reverse=True)
ablated  = ablate_model(model, ranked[:n_ablate])
acc      = per_class_accuracy(ablated, test_loader, label_transform=None, device='cpu')
```

---

## Module Reference

### `src/bft.py`

| Symbol | Description |
|---|---|
| `bft(model, data, ...)` | **Primary entry point.** Run BFT on model + DataLoader. |
| `collect_layer_dicts(model, loader, ...)` | Hook-based layer activation collection; returns data usable as `bft()` layer-dict input. |
| `nodes_at_layer(root, layer_idx)` | Return all BFTNodes at a given layer depth. |
| `full_nmf_pipeline(X, k, ...)` | Fit NMF at fixed rank, normalise, sort by importance. |
| `auto_nmf_pipeline(X, k_max, ...)` | Fit NMF at `k_max`, auto-select effective rank K\*. |
| `run_nmf_minibatch(X, k, ...)` | Raw MiniBatchNMF in float32. |
| `trace_single_layer(W, act, sw, ...)` | One BFT step for a single layer. |
| `compute_joint_arbors_normalized(W, act, ...)` | Joint arbor matrix for FC layers. |
| `compute_conv_joint_arbors(W, fmap, ...)` | Joint arbor matrix for Conv2d layers. |
| `compute_attn_joint_arbors(W_V, x_tokens, attn_w, ...)` | Joint arbor matrix for attention layers. |

### `src/fingerprint_utils.py`

| Symbol | Description |
|---|---|
| `compute_fingerprints(result, indices, normalize)` | High-level: fingerprints + cosine similarity for a BFTResult. |
| `extract_fingerprint_matrix(root, indices)` | Low-level: (n, D) matrix of concatenated img_factors. |
| `compute_stimulus_similarity(matrix)` | Pairwise cosine similarity from a fingerprint matrix. |
| `project_onto_bft(result, model, loader, ...)` | Project new stimuli via NNLS onto fixed BFT factors. |
| `project_stimuli_onto_tree(root, new_inputs)` | Low-level: NNLS projection without a DataLoader. |

### `src/scaffold_utils.py`

| Symbol | Description |
|---|---|
| `build_scaffold_edges(results, ...)` | Extract (pos, neg) edge matrices from a list of BFT nodes. |
| `scaffold_layer_sizes_from_edges(edges)` | Derive `layer_sizes` list from edge matrices. |
| `scaffold_loading_from_edges(edges)` | Derive per-neuron node loading vector from edge matrices. |
| `bft_node_vals(node, fi, top_pct, ...)` | Extract flat node-value vector from one BFT node. |

### `src/ablation_utils.py`

| Symbol | Description |
|---|---|
| `ablation_sweep(model, result, loader, ...)` | **Primary entry point.** Pruning experiment for one class. |
| `ablation_layer_sweep(model, result, loader, ...)` | Sweep over pruning depth (last 1, 2, … L layers). |
| `select_class_circuit(root, targets, class_d)` | BFT importance scores for the circuit selective to `class_d`. |
| `ablate_model(model, keys)` | Deep-copy model with specified weights zeroed. |
| `run_ablation_sweep(model, scores, fractions, ...)` | Low-level ablation loop for one method. |
| `magnitude_scores(model)` | `{(l,i,j): |W[i,j]|}` baseline importance. |
| `taylor_scores(model, loader, ...)` | `|grad(L) * W|` Taylor-criterion importance. |
| `normalize_scores_per_layer(scores)` | Min-max normalise within each layer. |
| `per_class_accuracy(model, loader, ...)` | Per-class accuracy dict. |

### `src/models.py`

| Class | Description |
|---|---|
| `SimpleMLP` | Configurable MLP; exposes `linear_layer_indices()`. Trained on MNIST. |
| `SmallCNN` | N-block CNN (default: 4 × [32,64,128,256] + GlobalAvgPool). Trained on CIFAR-10. |
| `TinyViT` | Tiny ViT for 28×28 images; single transformer block + CLS token. |

### `src/checkpoint.py`

```python
save_experiment(model, config, exp_dir)   # saves weights.pt + config.json
model, config = load_experiment(exp_dir)  # restores model in eval mode
```

Config schema keys: `arch`, `arch_kwargs`, `dataset`, `dataset_kwargs`, `label_transform`, `analysis_layer_indices`, `n_per_class`, `description`.

### `src/robustness_utils.py`

| Symbol | Description |
|---|---|
| `compute_nmf_stability(X, k, n_seeds)` | NMF initialisation robustness: pairwise cosine similarity across random seeds. |
| `align_factors(reference, others)` | Hungarian-align a list of factor matrices to a reference. |
| `compute_k_sensitivity(X, k_star, ...)` | Compare factors at K\*-1, K\*, K\*+1. |

### `src/data_utils.py`

| Symbol | Description |
|---|---|
| `get_mnist_loaders(batch_size, digit_filter)` | MNIST train/test DataLoaders. |
| `get_cifar10_loaders(batch_size, augment)` | CIFAR-10 loaders; augment: `'baseline'|'cutout'|'strong'|'none'`. |
| `get_imagenet_loaders(batch_size, root, ...)` | ImageNet loaders (ImageFolder layout). |
| `collect_layer_inputs_generic(model, dataset, ...)` | Hook-based layer input collection for any `nn.Module`. |
| `collect_activations(model, dataset, ...)` | Activation collection for `SimpleMLP` (uses `inference=True` mode). |

### `src/plot_utils.py`

Visualization helpers for BFT outputs. Key functions:

| Symbol | Description |
|---|---|
| `plot_scaffold(result, model, loader, ...)` | **Scaffold graph** driven by real forward-pass activity. One figure per BFT path. |
| `plot_scaffold_graph(loading, edges, sizes, ...)` | Low-level scaffold graph from precomputed edge matrices. |
| `plot_factor_overview_panel(node, images, ...)` | Per-factor summary: λ bar, connection heatmap, stimulus weights, top images. |
| `plot_input_layer_factors(node, images, arch)` | Input-layer factor visualization (FC pixel RFs / conv kernels / attention maps). |
| `plot_factor_gallery(node, images, targets, ..., k)` | Top/bottom-N image gallery for factor k. |
| `plot_pruning_results(data, class_names, ...)` | One figure per class: target vs bystander accuracy across pruning fractions. |
| `plot_pruning_by_layer_depth(sweep, ...)` | Layer-depth pruning plot from `ablation_layer_sweep()`. |
| `plot_embedding_comparison(fp, acts, labels, ...)` | PCA + MDS comparison of BFT fingerprints vs network activations. |
| `plot_factor_tree(tree_nodes, activations, ...)` | BFT factor tree coloured by stimulus activation. |
| `extract_tree_nodes(root)` | BFS walk of BFT tree into a flat list of metadata dicts. |

### `src/neuron_analysis.py`

Single-neuron analysis tools (synaptic arbors, PCA decoding, mean arbor plots). Main entry point: `single_neuron_report(model_layers, layer_idx, neuron_idx, ...)`.

---

## Data Layout

```
data/       (not tracked — generated by notebooks)
└── experiments/   experiment directories (weights.pt + config.json)

src/        core library
notebooks/  analysis notebooks
scripts/
├── train.py       training entry point
└── run_nb.py      notebook runner
```
