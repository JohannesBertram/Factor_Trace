#!/usr/bin/env python3
"""Generate notebooks/06_simplification_validation.ipynb."""
import json, os, textwrap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(src).lstrip("\n"),
    }

def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(src).lstrip("\n"),
    }

cells = []

# ---------------------------------------------------------------------------
# Cell 0 – title (markdown)
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
# Notebook 06: Simplification Diagnosis & Validation

Extends NB04 (`04_cifar10_factor_trace.ipynb`) by (1) diagnosing which efficiency
simplifications are actually *required* on the available hardware and (2) empirically
validating each simplification that is used.

**Design goals:**
- Fully self-contained: all `src/` functions are inlined — no local imports needed.
- Google Colab / GPU-ready.
- Every simplification is controlled by a feature flag; every validation section is
  gated by a corresponding flag. Start with the default (minimal simplifications, all
  validations enabled) and add complexity as needed.

**Simplification flags:**

| Flag | Default | What it controls |
|---|---|---|
| `USE_CONFIDENCE_FILTER` | True | S1: keep top-K samples/class by confidence |
| `USE_SPATIAL_POOLING` | True | S2: spatial avg-pool conv arbors (REQUIRED) |
| `STIM_THRESHOLD` | 0.0 | S3: hard gate on low-weight samples (0=off) |
| `COMPRESSION_METHOD` | None | S4: None / 'column_sample' / 'jl' |
| `K_LIST` | None | S5: None=auto / fixed list per layer |
| `BRANCH_AT_CONV` | False | S6: branch at conv layers |
'''))

# ---------------------------------------------------------------------------
# Cell 1 – Colab setup
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
# Run these on Colab if packages are missing:
# !pip install psutil -q
# !pip install networkx -q
# Optionally mount Drive to reuse checkpoints:
# from google.colab import drive; drive.mount('/content/drive')

import os

# Cache dir: on Colab /content is writable; locally use data/cache/nb06
CACHE_DIR = '/content/nb06_cache' if os.path.exists('/content') else os.path.join('..', 'data', 'cache', 'nb06')
os.makedirs(CACHE_DIR, exist_ok=True)

# Model checkpoint (change if you mounted Drive with the NB04 checkpoint)
EXP_DIR  = os.path.join(CACHE_DIR, 'experiments', 'cifar10_cnn')
DATA_DIR = '/content/data' if os.path.exists('/content') else os.path.join('..', 'data')
FIG_DIR  = os.path.join(CACHE_DIR, 'figures')
os.makedirs(EXP_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print(f'Cache:  {CACHE_DIR}')
print(f'Exp:    {EXP_DIR}')
print(f'Data:   {DATA_DIR}')
'''))

# ---------------------------------------------------------------------------
# Cell 2 – Imports
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
import sys, json, time, pickle, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import networkx as nx
from collections import defaultdict
from sklearn.decomposition import NMF as _NMF
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings('ignore', category=UserWarning)

DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps'  if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {DEVICE}')
'''))

# ---------------------------------------------------------------------------
# Cell 3 – Feature flags
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
# ═══════════════════════════════════════════════════════════════════════════
# SIMPLIFICATION FLAGS
# Start with these defaults (minimal simplifications, maximally valid).
# Enable more if you hit RAM/time limits — see the diagnosis table below.
# ═══════════════════════════════════════════════════════════════════════════

# S1: Confidence pre-filter
#   At N=60/class, fc1 arbor ~ 262 MB — fine on Colab.
#   At N>300/class you may need S4 as well.
USE_CONFIDENCE_FILTER = True    # False = keep all correctly-classified samples
N_TOP_PER_CLASS       = 60      # samples per class; None = unlimited

# S2: Spatial average pooling for conv arbors  <- ALWAYS REQUIRED
#   Without it: conv1 full-spatial = 118 MB (N=600), 2 GB (N=9000)
USE_SPATIAL_POOLING   = True    # do not disable

# S3: Stimulus threshold (hard gate on low-weight samples)
#   0.0 = no hard gate (maximally valid, pure soft weighting)
#   0.7 = NB04 default (zeros bottom 70th-percentile samples at each step)
STIM_THRESHOLD        = 0.0

# S4: Arbor compression for large FC layers
#   At N=60/class, fc1 (262 MB) fits on Colab — leave None.
#   Enable if N_TOP_PER_CLASS > ~300 or you hit OOM.
COMPRESSION_METHOD    = None    # None | 'column_sample' | 'jl'
MAX_ARBOR_DIM         = 1000    # target dim (ignored when None)

# S5: NMF rank selection
#   None = auto_nmf_pipeline (recommended: fits K_MAX then prunes by explained variance)
#   List = fixed K per layer in order [conv1, conv2, conv3, fc1, fc2]
K_LIST                = None
K_MAX_AUTO            = 16      # upper bound for auto mode

# S6: Branching at conv layers (expensive — not recommended for first run)
BRANCH_AT_CONV        = False
N_BRANCHES_CONV       = 2
N_BRANCHES_FC         = 3       # always used for FC layers

# RAM: store pos_joint (NMF input) in tree nodes?
#   pos_joint can be 44–262 MB per node; storing it fills RAM fast.
#   Disable for production; validation sections enable it locally.
STORE_POS_JOINT       = False
MIN_ACTIVE            = 30      # minimum active samples before threshold relaxation

RNG_SEED = 42
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION FLAGS
# Each flag gates the corresponding diagnostic/validation section.
# ═══════════════════════════════════════════════════════════════════════════
VALIDATE_DIAGNOSIS    = True    # Sec 3: memory/feasibility table
VALIDATE_FILTER       = True    # Sec 4: confidence vs random sampling
VALIDATE_POOLING      = True    # Sec 5: avg vs max spatial pooling
VALIDATE_THRESHOLD    = True    # Sec 6: threshold sweep 0->0.85
VALIDATE_COMPRESSION  = True    # Sec 7: column-sample vs JL vs full
VALIDATE_K            = True    # Sec 8: K scree + auto-K per layer
VALIDATE_SEEDS        = True    # Sec 9: NMF seed stability (3 seeds)
VALIDATE_BRANCHING    = False   # Sec 10: conv branching study (slow)
'''))

# ---------------------------------------------------------------------------
# Cell 4 – NMF pipeline utilities  (inlined from src/factorization.py)
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
# ── NMF utilities (inlined from src/factorization.py) ────────────────────

def run_nmf(X, n_components, random_state=0, max_iter=20000):
    """Fit NMF; return (W, H, model). W=(n_samples,K), H=(n_features,K)."""
    init = 'nndsvda' if n_components <= min(X.shape) else 'random'
    nmf  = _NMF(n_components=n_components, init=init,
                random_state=random_state, max_iter=max_iter)
    W = nmf.fit_transform(X)
    H = nmf.components_.T
    return W, H, nmf


def normalize_factors(W, H):
    """Unit-normalize columns of W and H; return (W_n, H_n, lambdas)."""
    W, H = W.copy(), H.copy()
    wn = np.linalg.norm(W, axis=0)
    active = ~np.isclose(wn, 0); W[:, active] /= wn[active]
    hn = np.linalg.norm(H, axis=0)
    active = ~np.isclose(hn, 0); H[:, active] /= hn[active]
    return W, H, wn * hn


def sort_by_lambda(W, H, lambdas):
    idx = np.argsort(lambdas)[::-1]
    return W[:, idx], H[:, idx], lambdas[idx], idx


def full_nmf_pipeline(X, n_components, random_state=0, max_iter=20000):
    """Fit -> normalize -> sort -> rescale by sqrt(lambda). Returns (W,H,lams)."""
    W, H, _ = run_nmf(X, n_components, random_state=random_state, max_iter=max_iter)
    W, H, lams = normalize_factors(W, H)
    W, H, lams, _ = sort_by_lambda(W, H, lams)
    s = np.sqrt(lams)
    return W * s, H * s, lams


def _select_k_single(lambdas, method, threshold, min_k):
    if method == 'cumvar':
        total = lambdas.sum()
        if total == 0: return min_k
        return int(np.searchsorted(np.cumsum(lambdas) / total, threshold)) + 1
    elif method == 'marginal':
        if lambdas[0] == 0: return min_k
        passing = np.where(lambdas / lambdas[0] >= threshold)[0]
        return int(passing[-1]) + 1 if len(passing) else min_k
    elif method == 'elbow':
        if len(lambdas) < 3: return len(lambdas)
        return int(np.argmax(np.diff(np.diff(lambdas)))) + 1
    elif method == 'fraction':
        for k in range(len(lambdas) - 1):
            if lambdas[k] / (lambdas[k + 1] + 1e-12) >= 1.5:
                return k
        return len(lambdas) - 1
    raise ValueError(f"Unknown method '{method}'")


def select_k_from_lambdas(lambdas, method='cumvar', threshold=0.95, min_k=1, min_cumvar=None):
    lambdas = np.asarray(lambdas, dtype=float)
    if len(lambdas) == 0: return min_k
    if method == 'structural': method = ['fraction', 'cumvar']
    if isinstance(method, (list, tuple)):
        k_star = max(_select_k_single(lambdas, method[0], threshold, min_k),
                     _select_k_single(lambdas, 'cumvar',   threshold, min_k))
    else:
        k_star = _select_k_single(lambdas, method, threshold, min_k)
    if min_cumvar is not None:
        k_star = max(k_star, _select_k_single(lambdas, 'cumvar', min_cumvar, min_k))
    return max(min_k, min(k_star, len(lambdas)))


def auto_nmf_pipeline(X, k_max=None, method='structural', threshold=0.95,
                      min_k=1, min_cumvar=None, random_state=0, max_iter=20000):
    """Fit at k_max, auto-select K* by lambda informativity."""
    if k_max is None: k_max = min(min(X.shape) - 1, 20)
    k_max = max(int(k_max), 2)
    img_f, neu_f, lams = full_nmf_pipeline(X, k_max, random_state=random_state, max_iter=max_iter)
    k_star = select_k_from_lambdas(lams, method=method, threshold=threshold,
                                   min_k=min_k, min_cumvar=min_cumvar)
    return img_f[:, :k_star], neu_f[:, :k_star], lams[:k_star], k_star

print('NMF utilities loaded.')
'''))

# ---------------------------------------------------------------------------
# Cell 5 – Arbor functions + compression
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
# ── Arbor functions + compression ─────────────────────────────────────────

def _apply_stim(joint, stim_w, threshold):
    """Scale rows by stim_w; zero rows below threshold quantile."""
    if stim_w is not None:
        joint = joint * stim_w[:, None]
    if threshold > 0.0 and stim_w is not None and stim_w.std() > 1e-8:
        cutoff = np.quantile(stim_w, threshold)
        joint[stim_w <= cutoff] = 0.0
    return joint


def compute_conv_arbor_avg(weight, input_fmap, stim_w=None, threshold=0.0, eps=1e-8):
    """Conv arbor with spatial AVERAGE pooling (NB04 default).

    weight:     (C_out, C_in, kH, kW)
    input_fmap: (N, C_in, H, W)
    Returns:    (N, C_out * C_in * kH * kW)
    """
    N = input_fmap.shape[0]
    C_out, C_in, kH, kW = weight.shape
    W_flat = weight.reshape(C_out, C_in * kH * kW)
    pad    = (kH // 2, kW // 2)
    patches = F.unfold(torch.from_numpy(input_fmap).float(),
                       kernel_size=(kH, kW), padding=pad)   # (N, C_in*kH*kW, n_pos)
    avg_p   = patches.mean(dim=2).numpy()                   # (N, C_in*kH*kW)
    if stim_w is not None and stim_w.std() > 1e-8:
        avg_p = avg_p / (np.linalg.norm(avg_p, axis=1, keepdims=True) + eps)
    joint = (avg_p[:, None, :] * W_flat[None, :, :]).reshape(N, C_out * C_in * kH * kW).copy()
    return _apply_stim(joint, stim_w, threshold)


def compute_conv_arbor_max(weight, input_fmap, stim_w=None, threshold=0.0, eps=1e-8):
    """Conv arbor with spatial MAX pooling (alternative to avg)."""
    N = input_fmap.shape[0]
    C_out, C_in, kH, kW = weight.shape
    W_flat = weight.reshape(C_out, C_in * kH * kW)
    pad     = (kH // 2, kW // 2)
    patches = F.unfold(torch.from_numpy(input_fmap).float(),
                       kernel_size=(kH, kW), padding=pad)
    max_p   = patches.max(dim=2).values.numpy()             # (N, C_in*kH*kW)
    if stim_w is not None and stim_w.std() > 1e-8:
        max_p = max_p / (np.linalg.norm(max_p, axis=1, keepdims=True) + eps)
    joint = (max_p[:, None, :] * W_flat[None, :, :]).reshape(N, C_out * C_in * kH * kW).copy()
    return _apply_stim(joint, stim_w, threshold)


def compute_conv_arbor_center(weight, input_fmap, stim_w=None, threshold=0.0, eps=1e-8):
    """Conv arbor using only the single CENTER spatial position (no pooling)."""
    N = input_fmap.shape[0]
    C_out, C_in, kH, kW = weight.shape
    W_flat = weight.reshape(C_out, C_in * kH * kW)
    pad     = (kH // 2, kW // 2)
    patches = F.unfold(torch.from_numpy(input_fmap).float(),
                       kernel_size=(kH, kW), padding=pad)   # (N, C_in*kH*kW, n_pos)
    H_out   = input_fmap.shape[2]
    W_out   = input_fmap.shape[3]
    ctr_idx = (H_out // 2) * W_out + (W_out // 2)          # center position
    ctr_p   = patches[:, :, ctr_idx].numpy()               # (N, C_in*kH*kW)
    if stim_w is not None and stim_w.std() > 1e-8:
        ctr_p = ctr_p / (np.linalg.norm(ctr_p, axis=1, keepdims=True) + eps)
    joint = (ctr_p[:, None, :] * W_flat[None, :, :]).reshape(N, C_out * C_in * kH * kW).copy()
    return _apply_stim(joint, stim_w, threshold)


def compute_fc_arbor(weight, input_flat, stim_w=None, threshold=0.0, eps=1e-8):
    """FC arbor: weight (n_out,n_in) × normalized input (N,n_in) -> (N, n_out*n_in)."""
    N = input_flat.shape[0]
    n_out, n_in = weight.shape
    if stim_w is not None and stim_w.std() > 1e-8:
        inp = input_flat / (np.linalg.norm(input_flat, axis=1, keepdims=True) + eps)
    else:
        inp = input_flat
    joint = (inp[:, None, :] * weight[None, :, :]).reshape(N, n_out * n_in).copy()
    return _apply_stim(joint, stim_w, threshold)


# ── Compression ────────────────────────────────────────────────────────────

def apply_column_sampling(joint, n_cols, seed=42):
    """Random column subsampling (N,D) -> (N, n_cols). Non-negative result preserved."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(joint.shape[1], min(n_cols, joint.shape[1]), replace=False)
    return joint[:, np.sort(idx)], idx


def apply_jl_projection(joint, proj_dim, seed=42):
    """Johnson–Lindenstrauss projection (N,D)->(N,proj_dim). May produce negatives."""
    rng = np.random.RandomState(seed)
    P   = rng.randn(joint.shape[1], proj_dim).astype(np.float32) / np.sqrt(proj_dim)
    return (joint @ P), P

print('Arbor functions loaded.')
'''))

# ---------------------------------------------------------------------------
# Cell 6 – cnn_tree_trace (modified to support all flags)
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
# ── Stimulus weight helpers ────────────────────────────────────────────────

def _safe_stim(img_factor_col, threshold, min_active):
    """Return (stim, effective_threshold_used)."""
    stim   = np.maximum(img_factor_col.copy().astype(np.float32), 0)
    thresh = threshold
    while thresh > 0.0 and stim.max() > 0:
        pos_vals = stim[stim > 0]
        if len(pos_vals) == 0: break
        cutoff = np.quantile(pos_vals, thresh)
        if (stim > cutoff).sum() >= min_active:
            stim[stim <= cutoff] = 0.0
            return stim, thresh
        thresh = max(0.0, thresh - 0.05)
    stim[stim <= 0] = 0.0
    return stim, thresh


# ── Single-layer trace ─────────────────────────────────────────────────────

def trace_layer(layer_dict, stim_w, K,
                threshold=0.0, compression=None, max_dim=1000,
                random_state=0, store_joint=False, k_max_auto=16):
    """
    Compute arbor -> (optional compress) -> NMF for one layer.

    Returns dict with img_factors, neural_factors, lambdas, neg_* counterparts,
    compression info, and optionally pos_joint (if store_joint=True).
    """
    weight = layer_dict['weight']
    fmap   = layer_dict['input_fmap']
    ltype  = layer_dict['type']

    if ltype == 'conv':
        raw = compute_conv_arbor_avg(weight, fmap, stim_w, threshold)
    else:
        raw = compute_fc_arbor(weight, fmap, stim_w, threshold)

    D = raw.shape[1]
    cinfo = {'method': None, 'orig_dim': D, 'comp_dim': D}

    if compression is not None and D > max_dim:
        if compression == 'column_sample':
            raw, idx = apply_column_sampling(raw, max_dim, random_state)
            cinfo = {'method': 'column_sample', 'orig_dim': D, 'comp_dim': raw.shape[1], 'idx': idx}
        elif compression == 'jl':
            raw, P = apply_jl_projection(raw, max_dim, random_state)
            cinfo = {'method': 'jl', 'orig_dim': D, 'comp_dim': max_dim, 'P': P}

    pos = np.clip(raw, 0, None)
    neg = np.clip(-raw, 0, None)

    if K is None:
        img_f, neu_f, lams, _ = auto_nmf_pipeline(pos, k_max=k_max_auto, random_state=random_state)
    else:
        img_f, neu_f, lams    = full_nmf_pipeline(pos, K, random_state=random_state)

    if neg.max() > 0:
        if K is None:
            nif, nnf, nlams, _ = auto_nmf_pipeline(neg, k_max=k_max_auto, random_state=random_state)
        else:
            nif, nnf, nlams    = full_nmf_pipeline(neg, K, random_state=random_state)
    else:
        nif = nnf = nlams = None

    out = dict(img_factors=img_f, neural_factors=neu_f, lambdas=lams,
               neg_img_factors=nif, neg_neural_factors=nnf, neg_lambdas=nlams,
               compression=cinfo)
    if store_joint:
        out['pos_joint'] = pos
    return out


# ── Full backward tree trace ───────────────────────────────────────────────

def cnn_tree_trace(layer_data_list,
                   K_list=None, k_max_auto=16,
                   n_branches_fc=3, n_branches_conv=1,
                   threshold=0.0, compression=None, max_dim=1000,
                   min_active=30, random_state=0, store_joint=False):
    """
    Backward NMF factor tree over all CNN layers (deepest layer = root).

    layer_data_list : forward order [conv1, ..., fc2]
    K_list          : per-layer K (forward order), or None for auto per layer
    Returns root node (fc2 layer).
    """
    N        = layer_data_list[0]['input_fmap'].shape[0]
    n_layers = len(layer_data_list)

    def build(l_idx, stim_w, path, branch_fi):
        ld    = layer_data_list[l_idx]
        K     = K_list[l_idx] if K_list is not None else None
        n_act = int((stim_w > 1e-9).sum()) if stim_w.std() > 1e-8 else N
        print(f'  [{l_idx}] {ld["name"]:<10}  K={"auto":>4}  active={n_act:4d}  path={path}')

        res  = trace_layer(ld, stim_w, K, threshold=threshold,
                           compression=compression, max_dim=max_dim,
                           random_state=random_state, store_joint=store_joint,
                           k_max_auto=k_max_auto)
        node = dict(layer_idx=l_idx, layer_name=ld['name'], layer_type=ld['type'],
                    path=path, factor_idx=branch_fi,
                    stim_in=stim_w, active_samples=n_act,
                    eff_thresholds=[],
                    **res, children=[])

        if l_idx > 0:
            nb = n_branches_conv if ld['type'] == 'conv' else n_branches_fc
            nb = min(nb, res['img_factors'].shape[1])
            for b in range(nb):
                cs, et = _safe_stim(res['img_factors'][:, b], threshold, min_active)
                node['eff_thresholds'].append(et)
                node['children'].append(build(l_idx - 1, cs, path + [b], b))
        return node

    print('Running backward factor trace...')
    t0 = time.time()
    root = build(n_layers - 1, np.ones(N, dtype=np.float32), [], 0)
    print(f'Done in {time.time()-t0:.1f}s')
    return root


def get_spine(root):
    """Main path: always follow child[0] root->leaf (deepest first)."""
    nodes, node = [], root
    while node:
        nodes.append(node)
        node = node['children'][0] if node['children'] else None
    return nodes


def get_all_paths(root):
    """All root-to-leaf paths as lists of nodes (deepest first)."""
    if not root['children']: return [[root]]
    return [[root] + sub for c in root['children'] for sub in get_all_paths(c)]

print('Tree trace loaded.')
'''))

# ---------------------------------------------------------------------------
# Cell 7 – Utilities: data collection, caching, helpers
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
# ── Data collection ────────────────────────────────────────────────────────

def collect_cnn_layer_data(model, loader, device):
    """Hook-based collection of (input_fmap, output_fmap) for every Conv2d/Linear."""
    model.eval()
    named = [(n, m) for n, m in model.named_modules()
             if isinstance(m, (nn.Conv2d, nn.Linear))]
    store = {n: {'inp': None, 'out': None} for n, _ in named}

    def make_hook(name):
        def h(mod, inp, out):
            store[name]['inp'] = inp[0].detach().cpu()
            store[name]['out'] = out.detach().cpu()
        return h

    hooks = [m.register_forward_hook(make_hook(n)) for n, m in named]
    acc_inp = {n: [] for n, _ in named}
    acc_out = {n: [] for n, _ in named}
    all_imgs, all_tgts, all_confs = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(device), y.to(device)
            logits = model(x)
            probs  = logits.exp()
            ok     = (probs.argmax(1) == y).cpu().nonzero(as_tuple=True)[0]
            if not len(ok): continue
            all_imgs.append(x[ok].cpu()); all_tgts.append(y[ok].cpu())
            all_confs.append(probs.max(1).values[ok].cpu())
            for n, _ in named:
                acc_inp[n].append(store[n]['inp'][ok])
                acc_out[n].append(store[n]['out'][ok])

    for h in hooks: h.remove()

    imgs  = torch.cat(all_imgs).numpy()
    tgts  = torch.cat(all_tgts).numpy()
    confs = torch.cat(all_confs).numpy()

    layer_data = []
    for n, mod in named:
        is_conv = isinstance(mod, nn.Conv2d)
        layer_data.append({
            'name': n, 'type': 'conv' if is_conv else 'fc',
            'weight':      mod.weight.detach().cpu().numpy(),
            'input_fmap':  torch.cat(acc_inp[n]).numpy(),
            'output_fmap': torch.cat(acc_out[n]).numpy(),
        })
    return {'images': imgs, 'targets': tgts, 'confidences': confs, 'layer_data': layer_data}


def confidence_prefilter(raw, top_k, n_classes=10):
    """Keep top_k most-confident samples per class."""
    keep = np.sort(np.concatenate([
        np.where(raw['targets'] == c)[0][
            np.argsort(raw['confidences'][raw['targets'] == c])[::-1][:top_k]]
        for c in range(n_classes)]))
    return {k: (v[keep] if isinstance(v, np.ndarray) else
                [{**ld, 'input_fmap': ld['input_fmap'][keep],
                  'output_fmap': ld['output_fmap'][keep]} for ld in v])
            for k, v in raw.items()}, keep


def random_filter(raw, top_k, seed=42, n_classes=10):
    """Keep a random top_k samples per class (for S1 validation)."""
    rng  = np.random.RandomState(seed)
    keep = np.sort(np.concatenate([
        rng.choice(np.where(raw['targets'] == c)[0],
                   min(top_k, (raw['targets'] == c).sum()), replace=False)
        for c in range(n_classes)]))
    return {k: (v[keep] if isinstance(v, np.ndarray) else
                [{**ld, 'input_fmap': ld['input_fmap'][keep],
                  'output_fmap': ld['output_fmap'][keep]} for ld in v])
            for k, v in raw.items()}, keep


# ── Caching ────────────────────────────────────────────────────────────────

def save_obj(obj, path):
    with open(path, 'wb') as f: pickle.dump(obj, f, protocol=4)

def load_obj(path):
    with open(path, 'rb') as f: return pickle.load(f)

def save_arrays(path, **arrays):
    np.savez_compressed(path, **arrays)

def load_arrays(path):
    return dict(np.load(path, allow_pickle=True))


# ── Memory ─────────────────────────────────────────────────────────────────

def print_mem(label=''):
    try:
        import psutil
        p    = psutil.Process()
        rss  = p.memory_info().rss / 1e9
        avail = psutil.virtual_memory().available / 1e9
        total = psutil.virtual_memory().total / 1e9
        print(f'[RAM {label}] rss={rss:.2f}GB  avail={avail:.2f}GB  total={total:.2f}GB')
    except ImportError:
        print('[RAM] psutil not available — run: pip install psutil')


# ── Validation helpers ─────────────────────────────────────────────────────

def top_cosine_sim(neu_f_a, neu_f_b):
    """Cosine similarity between top columns of two neural_factors matrices,
    after Hungarian matching (handles permutation)."""
    K = min(neu_f_a.shape[1], neu_f_b.shape[1])
    A = neu_f_a[:, :K]; B = neu_f_b[:, :K]
    # Normalize columns
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    C = A.T @ B   # (K, K) cosine similarities
    row, col = linear_sum_assignment(-np.abs(C))
    return np.abs(C[row, col]).mean()


def img_factor_cosine(if_a, if_b, col=0):
    """Cosine similarity between img_factors[:, col] of two runs."""
    a = if_a[:, col]; b = if_b[:, col]
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def imdenorm(img, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    m = np.array(mean)[None, None, :]; s = np.array(std)[None, None, :]
    return np.clip(img.transpose(1, 2, 0) * s + m, 0, 1)

print('Utilities loaded.')
'''))

# ---------------------------------------------------------------------------
# Cell 8 – SmallCNN definition + training
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
# ── SmallCNN (must match NB04 checkpoint) ──────────────────────────────────

class SmallCNN(nn.Module):
    """
    3-conv + 2-FC network for CIFAR-10.
    conv1 (3->16)  -> BN -> ReLU -> MaxPool2d(2)    input: (N,3,32,32)
    conv2 (16->32) -> BN -> ReLU -> MaxPool2d(2)    input: (N,16,16,16)
    conv3 (32->64) -> BN -> ReLU -> AdaptiveAvgPool(4) input: (N,32,8,8)
    fc1   (1024->128)  -> ReLU                    input: (N,1024)
    fc2   (128->10)   -> log_softmax              input: (N,128)
    """
    def __init__(self):
        super().__init__()
        self.conv1  = nn.Conv2d(3,  16, 3, padding=1)
        self.bn1    = nn.BatchNorm2d(16)
        self.conv2  = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2    = nn.BatchNorm2d(32)
        self.conv3  = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3    = nn.BatchNorm2d(64)
        self.pool   = nn.MaxPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.fc1    = nn.Linear(64 * 4 * 4, 128)
        self.fc2    = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.avgpool(F.relu(self.bn3(self.conv3(x))))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


model     = SmallCNN().to(DEVICE)
ckpt_path = os.path.join(EXP_DIR, 'weights.pt')

if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print(f'Loaded: {ckpt_path}')
else:
    print(f'Checkpoint not found at {ckpt_path} — training from scratch (~5 min on T4)')
    N_EPOCHS  = 40
    normalize = T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    _train_tf = T.Compose([T.RandomCrop(32,4),T.RandomHorizontalFlip(),T.ToTensor(),normalize])
    _test_tf  = T.Compose([T.ToTensor(), normalize])
    _train_ds = torchvision.datasets.CIFAR10(DATA_DIR,True, download=True,transform=_train_tf)
    _test_ds  = torchvision.datasets.CIFAR10(DATA_DIR,False,download=True,transform=_test_tf)
    _tr = torch.utils.data.DataLoader(_train_ds,128,shuffle=True, num_workers=2,pin_memory=True)
    _te = torch.utils.data.DataLoader(_test_ds, 256,shuffle=False,num_workers=2,pin_memory=True)
    opt = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    crit = nn.NLLLoss()
    for ep in range(1, N_EPOCHS+1):
        model.train()
        nc, nt, rl = 0, 0, 0.0
        for x, y in _tr:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad()
            out=model(x); loss=crit(out,y); loss.backward(); opt.step()
            nc+=(out.argmax(1)==y).sum().item(); nt+=y.size(0); rl+=loss.item()*y.size(0)
        sch.step()
        if ep%10==0 or ep==N_EPOCHS:
            model.eval()
            tc=ts=0
            with torch.no_grad():
                for x,y in _te:
                    x,y=x.to(DEVICE),y.to(DEVICE)
                    tc+=(model(x).argmax(1)==y).sum().item(); ts+=y.size(0)
            print(f'Ep {ep:2d} | train {nc/nt:.3f} | test {tc/ts:.3f}')
    torch.save(model.state_dict(), ckpt_path)
    print(f'Saved to {ckpt_path}')
'''))

# ---------------------------------------------------------------------------
# Cell 9 – CIFAR-10 loaders
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
normalize   = T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
test_tf     = T.Compose([T.ToTensor(), normalize])
test_ds     = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)
test_loader = torch.utils.data.DataLoader(test_ds, 256, shuffle=False, num_workers=2)

# Quick per-class accuracy check
model.eval()
pc_correct = defaultdict(int); pc_total = defaultdict(int)
with torch.no_grad():
    for x, y in test_loader:
        x,y = x.to(DEVICE),y.to(DEVICE)
        preds = model(x).argmax(1)
        for t,p in zip(y.cpu().tolist(),preds.cpu().tolist()):
            pc_total[t]+=1; pc_correct[t]+=int(t==p)

print(f'{"Class":<14} {"Acc":>6}')
print('-'*22)
for c in range(10):
    print(f'{CIFAR10_CLASSES[c]:<14} {pc_correct[c]/pc_total[c]:>6.3f}')
'''))

# ---------------------------------------------------------------------------
# Cell 10 – Section 1 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 1: Data Collection + Caching

Collect (input, output) activations for every Conv2d and Linear layer on correctly-classified
test samples. Cache to disk so subsequent runs skip this step (saves ~30 s).

After filtering, `raw_data` is deleted to free ~460 MB of RAM.
'''))

# ---------------------------------------------------------------------------
# Cell 11 – Collect + cache raw_data
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
_raw_cache = os.path.join(CACHE_DIR, 'raw_data.pkl')

if os.path.exists(_raw_cache):
    print(f'Loading cached raw_data from {_raw_cache}')
    raw_data = load_obj(_raw_cache)
else:
    print('Collecting layer data from test set...')
    t0 = time.time()
    raw_data = collect_cnn_layer_data(model, test_loader, DEVICE)
    print(f'Collected {len(raw_data["targets"]):,} correct samples in {time.time()-t0:.1f}s')
    save_obj(raw_data, _raw_cache)
    print(f'Cached to {_raw_cache}')

print_mem('after raw_data load')
print()
print(f'{"Layer":<8}  {"Type":<4}  {"Input shape":<22}  {"W shape"}')
print('-'*60)
for ld in raw_data['layer_data']:
    wsh = str(ld['weight'].shape)
    ish = str(ld['input_fmap'].shape)
    print(f'{ld["name"]:<8}  {ld["type"]:<4}  {ish:<22}  {wsh}')
'''))

# ---------------------------------------------------------------------------
# Cell 12 – Confidence filter + del raw_data
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
_n_total = len(raw_data['targets'])

if USE_CONFIDENCE_FILTER and N_TOP_PER_CLASS is not None:
    data, _keep_idx = confidence_prefilter(raw_data, N_TOP_PER_CLASS)
    print(f'Confidence filter: {_n_total:,} -> {len(data["targets"])} samples  '
          f'(top-{N_TOP_PER_CLASS}/class)')
else:
    data     = raw_data
    _keep_idx = np.arange(_n_total)
    print(f'No filter: using all {_n_total:,} correct samples')

# Free the full raw_data — only needed for validation sections
_raw_data_ref = raw_data   # keep a reference name for validation cells
if USE_CONFIDENCE_FILTER and N_TOP_PER_CLASS is not None:
    del raw_data
    print('raw_data freed from RAM.')
print_mem('after filter')

N = len(data['targets'])
print(f'\nN = {N} samples across {len(data["layer_data"])} layers')
'''))

# ---------------------------------------------------------------------------
# Cell 13 – Section 2 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 2: Production Trace

Run `cnn_tree_trace` with the current flag settings. Results are cached; re-run by
deleting the cache file.

Key config at a glance:
- `K_LIST = None` -> auto rank selection per layer
- `STIM_THRESHOLD = 0.0` -> pure soft weighting (no hard gate)
- `COMPRESSION_METHOD = None` -> no compression (fc1 ~ 262 MB, fine on Colab)
'''))

# ---------------------------------------------------------------------------
# Cell 14 – Production trace
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
_tree_cache = os.path.join(CACHE_DIR, 'tree.pkl')

if os.path.exists(_tree_cache):
    print(f'Loading cached tree from {_tree_cache}')
    tree = load_obj(_tree_cache)
else:
    tree = cnn_tree_trace(
        data['layer_data'],
        K_list       = K_LIST,
        k_max_auto   = K_MAX_AUTO,
        n_branches_fc  = N_BRANCHES_FC,
        n_branches_conv= N_BRANCHES_CONV if BRANCH_AT_CONV else 1,
        threshold    = STIM_THRESHOLD,
        compression  = COMPRESSION_METHOD,
        max_dim      = MAX_ARBOR_DIM,
        min_active   = MIN_ACTIVE,
        random_state = RNG_SEED,
        store_joint  = STORE_POS_JOINT,
    )
    save_obj(tree, _tree_cache)
    print(f'Cached tree to {_tree_cache}')

print_mem('after trace')

spine = get_spine(tree)
paths = get_all_paths(tree)
print(f'\nSpine: {[n["layer_name"] for n in spine]}')
print(f'Paths: {len(paths)}')
for pi, path in enumerate(paths):
    print(f'  Path {pi}: ' + ' -> '.join(
        f'{n["layer_name"]}[{n["factor_idx"]}]  K={len(n["lambdas"])}  '
        f'act={n["active_samples"]}  λmax={n["lambdas"][0]:.3f}'
        for n in path))
'''))

# ---------------------------------------------------------------------------
# Cell 15 – Section 3 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 3: Diagnosis Table

Which simplifications are *required* vs. *optional* for the current hardware?
This cell runs in < 1 second (no NMF) and prints memory estimates using the
actual layer shapes collected above.
'''))

# ---------------------------------------------------------------------------
# Cell 16 – Diagnosis table
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
if VALIDATE_DIAGNOSIS:
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
        total_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        avail_gb = 8.0; total_gb = 12.0   # Colab free-tier defaults
        print('psutil not available; using default 12 GB estimate')

    N_all  = _n_total       # before filter
    N_curr = N              # current (after filter)
    SAFE_GB = avail_gb * 0.5   # allow using half available RAM per array

    print(f'Available RAM: {avail_gb:.1f} GB  |  Total: {total_gb:.1f} GB')
    print(f'Safe single-array budget: {SAFE_GB:.1f} GB\n')

    hdr = f'{"Simplification":<32}{"Without (N=all)":<22}{"With flags":<18}{"Reduction":<12}{"RAM OK?":<10}Verdict'
    print(hdr)
    print('─' * len(hdr))

    # S1
    mb_all  = N_all  * 4 / 1e6   # trivial: just N
    mb_curr = N_curr * 4 / 1e6
    needed  = 'Optional'
    print(f'{"S1: confidence filter":<32}{N_all:>8,} samples    {N_curr:>6} samp.     '
          f'{N_all/max(N_curr,1):.1f}×          {"Yes":<10}{needed}')

    # S2 per conv layer
    print(f'{"S2: spatial pooling":<32}')
    for ld in data['layer_data']:
        if ld['type'] != 'conv': continue
        co, ci, kh, kw = ld['weight'].shape
        H = ld['input_fmap'].shape[2]; W = ld['input_fmap'].shape[3]
        D_pool = co * ci * kh * kw
        D_full = D_pool * H * W
        mb_pool_curr = N_curr * D_pool * 4 / 1e6
        mb_full_curr = N_curr * D_full * 4 / 1e6
        mb_full_all  = N_all  * D_full * 4 / 1e6
        ok   = 'YES' if mb_full_curr / 1e3 < SAFE_GB else 'NO !'
        verd = 'REQUIRED' if mb_full_curr / 1e3 >= SAFE_GB else 'Optional'
        print(f'  {ld["name"]:<30}{mb_full_all:>8.0f} MB         {mb_pool_curr:>6.0f} MB     '
              f'{D_full/D_pool:.0f}×          {ok:<10}{verd}')

    # S3
    print(f'{"S3: stimulus threshold":<32}{"speed only":<22}{"STIM_THRESHOLD="+str(STIM_THRESHOLD):<18}'
          f'{"n/a":<12}{"—":<10}Optional')

    # S4 per fc layer
    print(f'{"S4: compression":<32}')
    for ld in data['layer_data']:
        if ld['type'] != 'fc': continue
        no, ni = ld['weight'].shape
        D = no * ni
        mb_full_curr = N_curr * D * 4 / 1e6
        mb_comp_curr = N_curr * min(D, MAX_ARBOR_DIM) * 4 / 1e6
        ok   = 'YES' if mb_full_curr / 1e3 < SAFE_GB else 'NO !'
        verd = 'Optional' if mb_full_curr / 1e3 < SAFE_GB else 'REQUIRED'
        ratio = D / min(D, MAX_ARBOR_DIM)
        print(f'  {ld["name"]:<30}{mb_full_curr:>8.0f} MB         {mb_comp_curr:>6.0f} MB     '
              f'{ratio:.0f}×          {ok:<10}{verd}')

    # S5, S6
    print(f'{"S5: fixed K":<32}{"auto K (varies)":<22}{"K_LIST="+str(K_LIST):<18}'
          f'{"~2×":<12}{"—":<10}Optional')
    print(f'{"S6: conv branching":<32}{"exp. paths":<22}{"off":<18}'
          f'{"—":<12}{"—":<10}Optional')

    print()
    if COMPRESSION_METHOD is None:
        for ld in data['layer_data']:
            if ld['type'] != 'fc': continue
            mb = N_curr * ld['weight'].shape[0] * ld['weight'].shape[1] * 4 / 1e6
            if mb / 1e3 >= SAFE_GB:
                print(f'WARNING: {ld["name"]} arbor = {mb:.0f} MB may exceed safe budget.'
                      f' Consider COMPRESSION_METHOD = "column_sample"')
else:
    print('VALIDATE_DIAGNOSIS = False — skipped.')
'''))

# ---------------------------------------------------------------------------
# Cell 17 – Section 4 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 4: S1 Validation — Confidence Filter Bias

Does using the top-60 *most confident* samples introduce significant bias compared to
a random sample of the same size?

We compare the top-factor `img_factors[:, 0]` recovered from fc2 (the output layer,
uniform stimulus weights) under three sampling strategies.
'''))

# ---------------------------------------------------------------------------
# Cell 18 – S1 validation
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
if VALIDATE_FILTER and USE_CONFIDENCE_FILTER and N_TOP_PER_CLASS is not None:
    print('=== S1: Confidence filter validation ===')
    strategies = {
        f'confidence-{N_TOP_PER_CLASS}': confidence_prefilter(_raw_data_ref, N_TOP_PER_CLASS)[0],
        f'random-{N_TOP_PER_CLASS}-s42':  random_filter(_raw_data_ref, N_TOP_PER_CLASS, 42)[0],
        f'random-{N_TOP_PER_CLASS}-s123': random_filter(_raw_data_ref, N_TOP_PER_CLASS, 123)[0],
    }

    # Run fc2 NMF only (uniform weights, fastest possible)
    fc2_ld = [ld for ld in _raw_data_ref['layer_data'] if ld['name'] == 'fc2'][0]
    results_s1 = {}
    for name, d in strategies.items():
        ld_local = [ld for ld in d['layer_data'] if ld['name'] == 'fc2'][0]
        uniform  = np.ones(len(d['targets']), dtype=np.float32)
        raw_j    = compute_fc_arbor(ld_local['weight'], ld_local['input_fmap'], uniform, 0.0)
        pos_j    = np.clip(raw_j, 0, None)
        K = K_LIST[4] if K_LIST is not None else None
        if K is None:
            img_f, neu_f, lams, _ = auto_nmf_pipeline(pos_j, k_max=K_MAX_AUTO, random_state=RNG_SEED)
        else:
            img_f, neu_f, lams    = full_nmf_pipeline(pos_j, K, random_state=RNG_SEED)
        results_s1[name] = {'img_f': img_f, 'neu_f': neu_f, 'lams': lams, 'targets': d['targets']}
        print(f'  {name}: K={img_f.shape[1]}  λmax={lams[0]:.3f}  N={len(d["targets"])}')

    # Pairwise cosine similarities
    names_list = list(results_s1.keys())
    print('\nPairwise img_factor[:, 0] cosine similarity (fc2):')
    baseline = names_list[0]
    for n in names_list[1:]:
        sim_img = img_factor_cosine(results_s1[baseline]['img_f'],
                                    results_s1[n]['img_f'], col=0)
        sim_neu = top_cosine_sim(results_s1[baseline]['neu_f'],
                                 results_s1[n]['neu_f'])
        print(f'  {baseline} vs {n}: img_f cosine={sim_img:.3f}  neural_f cosine={sim_neu:.3f}')

    # Confidence distributions
    fig, ax = plt.subplots(figsize=(10, 3))
    for c in range(10):
        confs_all = _raw_data_ref['confidences'][_raw_data_ref['targets'] == c]
        ax.scatter(confs_all, np.full(len(confs_all), c) + np.random.uniform(-0.08, 0.08, len(confs_all)),
                   s=3, alpha=0.2, c=f'C{c}')
        thresh = np.sort(confs_all)[::-1][N_TOP_PER_CLASS-1] if len(confs_all) >= N_TOP_PER_CLASS else 0
        ax.axvline(thresh, color=f'C{c}', lw=1.0, alpha=0.7)
    ax.set_yticks(range(10)); ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_xlabel('Max softmax confidence')
    ax.set_title(f'Confidence filter threshold per class (top-{N_TOP_PER_CLASS})')
    plt.tight_layout(); plt.show()
elif not VALIDATE_FILTER:
    print('VALIDATE_FILTER = False — skipped.')
else:
    print('VALIDATE_FILTER skipped: USE_CONFIDENCE_FILTER is False (using all samples).')
'''))

# ---------------------------------------------------------------------------
# Cell 19 – Section 5 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 5: S2 Validation — Spatial Pooling Method

`compute_conv_arbor_avg` spatially averages all kernel positions. How different are
the resulting NMF factors if we use max-pooling or only the center position?

We test on conv1 (large spatial input, 32×32) and conv3 (small, 8×8 after pooling).
'''))

# ---------------------------------------------------------------------------
# Cell 20 – S2 validation
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
if VALIDATE_POOLING:
    print('=== S2: Spatial pooling validation ===\n')

    # Use stimulus weights from the production trace (spine layer nodes)
    spine_nodes = {n['layer_name']: n for n in spine}

    for layer_name in ['conv1', 'conv3']:
        ld = [l for l in data['layer_data'] if l['name'] == layer_name][0]
        if layer_name in spine_nodes:
            sw = spine_nodes[layer_name]['stim_in']
        else:
            sw = np.ones(N, dtype=np.float32)

        K = (K_LIST[{'conv1':0,'conv3':2}[layer_name]] if K_LIST is not None else None)
        ks = K if K is not None else 4   # use 4 for display if auto

        print(f'Layer: {layer_name}  (K={ks if K is not None else "auto"})')

        arbors = {}
        for method, fn in [('avg', compute_conv_arbor_avg),
                            ('max', compute_conv_arbor_max)]:
            raw = fn(ld['weight'], ld['input_fmap'], sw, 0.0)
            pos = np.clip(raw, 0, None)
            if K is None:
                img_f, neu_f, lams, kstar = auto_nmf_pipeline(pos, k_max=K_MAX_AUTO, random_state=RNG_SEED)
                print(f'  {method}-pool: auto K*={kstar}  λmax={lams[0]:.3f}  arbor={pos.shape}')
            else:
                img_f, neu_f, lams = full_nmf_pipeline(pos, K, random_state=RNG_SEED)
                print(f'  {method}-pool: K={K}  λmax={lams[0]:.3f}  arbor={pos.shape}')
            arbors[method] = {'img_f': img_f, 'neu_f': neu_f, 'lams': lams}

        # Center crop (only if H*W <= 64 to keep arbor manageable)
        H_in = ld['input_fmap'].shape[2]; W_in = ld['input_fmap'].shape[3]
        if H_in * W_in <= 64:
            raw  = compute_conv_arbor_center(ld['weight'], ld['input_fmap'], sw, 0.0)
            pos  = np.clip(raw, 0, None)
            if K is None:
                img_f, neu_f, lams, kstar = auto_nmf_pipeline(pos, k_max=K_MAX_AUTO, random_state=RNG_SEED)
                print(f'  center-crop: auto K*={kstar}  λmax={lams[0]:.3f}  arbor={pos.shape}')
            else:
                img_f, neu_f, lams = full_nmf_pipeline(pos, K, random_state=RNG_SEED)
                print(f'  center-crop: K={K}  λmax={lams[0]:.3f}  arbor={pos.shape}')
            arbors['center'] = {'img_f': img_f, 'neu_f': neu_f, 'lams': lams}

        # Cosine similarities vs avg baseline
        ref = arbors['avg']
        print(f'\n  Cosine similarity of neural_factors vs avg-pool:')
        for m, res in arbors.items():
            if m == 'avg': continue
            sim = top_cosine_sim(ref['neu_f'], res['neu_f'])
            print(f'    avg vs {m}: {sim:.4f}')
        print()

    print('Interpretation: cosine > 0.90 -> pooling method makes little difference')
else:
    print('VALIDATE_POOLING = False — skipped.')
'''))

# ---------------------------------------------------------------------------
# Cell 21 – Section 6 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 6: S3 Validation — Stimulus Threshold Sweep

The stimulus threshold zeros out the bottom-k% of samples at each layer transition,
creating a cascading hard gate. With `STIM_THRESHOLD=0.0` (default) no hard gate is
applied. Here we sweep thresholds to show the effect on active sample counts and factor
consistency.

**Key question:** How many active samples remain at each layer, and do the discovered
factors change significantly?
'''))

# ---------------------------------------------------------------------------
# Cell 22 – S3 validation: threshold sweep
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
if VALIDATE_THRESHOLD:
    print('=== S3: Stimulus threshold sweep ===')
    thresholds  = [0.0, 0.3, 0.5, 0.70, 0.85]
    sweep_trees = {}
    sweep_active = {}

    for T_val in thresholds:
        print(f'\n--- threshold = {T_val} ---')
        t0 = time.time()
        t_ = cnn_tree_trace(
            data['layer_data'],
            K_list=K_LIST if K_LIST is not None else [4,4,4,4,10],
            k_max_auto=K_MAX_AUTO,
            n_branches_fc=1, n_branches_conv=1,   # no branching for speed
            threshold=T_val,
            compression=COMPRESSION_METHOD, max_dim=MAX_ARBOR_DIM,
            min_active=MIN_ACTIVE, random_state=RNG_SEED, store_joint=False)
        sweep_trees[T_val]  = t_
        sweep_active[T_val] = {n['layer_name']: n['active_samples']
                               for n in get_spine(t_)}
        print(f'  Done in {time.time()-t0:.1f}s')

    # ── Active samples table ───────────────────────────────────────────────
    layer_names = [n['layer_name'] for n in get_spine(sweep_trees[0.0])]
    print('\nActive samples per threshold per layer:')
    print(f'{"Layer":<10}', '  '.join(f'T={t:.2f}' for t in thresholds))
    for ln in layer_names:
        row = '  '.join(f'{sweep_active[t].get(ln, 0):>7}' for t in thresholds)
        warn = ' <- LOW' if any(sweep_active[t].get(ln, 0) < 30 for t in thresholds[2:]) else ''
        print(f'{ln:<10} {row}{warn}')

    # ── Factor cosine similarity vs T=0 baseline ─────────────────────────
    ref_spine = {n['layer_name']: n for n in get_spine(sweep_trees[0.0])}
    print('\nCosine similarity of top neural factor vs T=0 baseline:')
    print(f'{"Layer":<10}', '  '.join(f'T={t:.2f}' for t in thresholds[1:]))
    for ln in layer_names:
        sims = []
        for T_val in thresholds[1:]:
            sp_node = {n['layer_name']: n for n in get_spine(sweep_trees[T_val])}
            if ln in sp_node and ln in ref_spine:
                s = top_cosine_sim(ref_spine[ln]['neural_factors'],
                                   sp_node[ln]['neural_factors'])
            else:
                s = float('nan')
            sims.append(s)
        row = '  '.join(f'{s:>7.3f}' for s in sims)
        print(f'{ln:<10} {row}')

    # ── Active samples line plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for T_val, color in zip(thresholds, ['k','#2196F3','#FF9800','#F44336','#9C27B0']):
        vals = [sweep_active[T_val].get(ln, 0) for ln in layer_names]
        ax.plot(layer_names, vals, marker='o', label=f'T={T_val}', color=color)
    ax.axhline(MIN_ACTIVE, ls='--', color='gray', lw=1, label=f'MIN_ACTIVE={MIN_ACTIVE}')
    ax.set_ylabel('Active samples'); ax.set_title('Active samples vs layer × threshold')
    ax.legend(fontsize=8); plt.tight_layout(); plt.show()
else:
    print('VALIDATE_THRESHOLD = False — skipped.')
'''))

# ---------------------------------------------------------------------------
# Cell 23 – Section 7 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 7: S4 Validation — Compression Methods

The NB04 approach uses Johnson–Lindenstrauss (JL) random projection followed by
clipping, which is theoretically problematic for NMF (clipping breaks the distance
guarantees). We compare it against:
- **Full** (ground truth, ~262 MB): NMF on the complete (N, 131,072) fc1 arbor
- **Column sampling ×5 seeds**: randomly sample 1,000 of 131,072 connections
- **JL projection**: NB04 approach (project then clip)
- **Neuron-group NMF**: split 128 output neurons into 4 groups, run NMF per group,
  average `img_factors` across groups (maintains interpretability)
'''))

# ---------------------------------------------------------------------------
# Cell 24 – S4 validation: compression comparison
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
if VALIDATE_COMPRESSION:
    print('=== S4: Compression validation (fc1 layer) ===')

    # Get fc1 stimulus weights from spine
    spine_dict = {n['layer_name']: n for n in spine}
    fc1_ld  = [l for l in data['layer_data'] if l['name'] == 'fc1'][0]
    fc2_node = spine_dict['fc2']
    fc1_stim, _ = _safe_stim(fc2_node['img_factors'][:, 0], STIM_THRESHOLD, MIN_ACTIVE)

    print(f'fc1 arbor shape (full): ({N}, {fc1_ld["weight"].shape[0]*fc1_ld["weight"].shape[1]})')
    print(f'Active samples: {int((fc1_stim > 1e-9).sum())}')
    print_mem('before compression validation')

    # ── (a) Full ground truth ────────────────────────────────────────────
    print('\n(a) Full (ground truth)...')
    t0   = time.time()
    raw_full = compute_fc_arbor(fc1_ld['weight'], fc1_ld['input_fmap'], fc1_stim, STIM_THRESHOLD)
    pos_full = np.clip(raw_full, 0, None)
    img_f_full, neu_f_full, lams_full = full_nmf_pipeline(pos_full, 4, random_state=RNG_SEED)
    print(f'  K=4  λmax={lams_full[0]:.3f}  time={time.time()-t0:.1f}s  '
          f'RAM~{pos_full.nbytes/1e6:.0f}MB')
    del raw_full   # free immediately after NMF

    results_s4 = {'full': {'img_f': img_f_full, 'neu_f': neu_f_full, 'lams': lams_full}}

    # ── (b) Column sampling × 5 seeds ────────────────────────────────────
    print('\n(b) Column sampling (1000 dims) × 5 seeds...')
    raw_full_tmp = compute_fc_arbor(fc1_ld['weight'], fc1_ld['input_fmap'], fc1_stim, STIM_THRESHOLD)
    col_imgs = []
    for seed in range(5):
        sampled, idx = apply_column_sampling(raw_full_tmp, MAX_ARBOR_DIM, seed)
        pos_s = np.clip(sampled, 0, None)
        img_f_s, neu_f_s, lams_s = full_nmf_pipeline(pos_s, 4, random_state=RNG_SEED)
        col_imgs.append(img_f_s[:, 0])
        sim = img_factor_cosine(img_f_full, img_f_s, 0)
        print(f'  seed={seed}: K=4  λmax={lams_s[0]:.3f}  img_f cosine vs full={sim:.3f}')
    del raw_full_tmp
    col_mean_img = np.column_stack(col_imgs).mean(axis=1, keepdims=True)
    # Normalize
    col_mean_img = col_mean_img / (np.linalg.norm(col_mean_img) + 1e-12)
    col_std_img  = np.column_stack(col_imgs).std(axis=1).mean()
    print(f'  Column-sample std of img_f[:,0] across seeds: {col_std_img:.4f}')

    # ── (c) JL projection ─────────────────────────────────────────────────
    print('\n(c) JL projection (1000 dims)...')
    raw_jl_tmp = compute_fc_arbor(fc1_ld['weight'], fc1_ld['input_fmap'], fc1_stim, STIM_THRESHOLD)
    jl_proj, _ = apply_jl_projection(raw_jl_tmp, MAX_ARBOR_DIM, RNG_SEED)
    pos_jl      = np.clip(jl_proj, 0, None)   # clip negatives — breaks JL guarantee
    neg_frac    = (jl_proj < 0).mean()
    print(f'  Negative values after JL projection: {neg_frac:.1%} (clipped to 0)')
    img_f_jl, neu_f_jl, lams_jl = full_nmf_pipeline(pos_jl, 4, random_state=RNG_SEED)
    sim_jl = img_factor_cosine(img_f_full, img_f_jl, 0)
    print(f'  K=4  λmax={lams_jl[0]:.3f}  img_f cosine vs full={sim_jl:.3f}')
    results_s4['jl'] = {'img_f': img_f_jl, 'neu_f': neu_f_jl, 'lams': lams_jl}
    del raw_jl_tmp

    # ── (d) Neuron-group subsampling (4 groups of 32) ─────────────────────
    print('\n(d) Neuron-group NMF (4 groups of 32 output neurons)...')
    n_out, n_in = fc1_ld['weight'].shape
    N_GROUPS    = 4
    grp_size    = n_out // N_GROUPS
    group_imgs  = []
    for g in range(N_GROUPS):
        g_idx      = np.arange(g * grp_size, (g+1) * grp_size)
        W_sub      = fc1_ld['weight'][g_idx]
        raw_sub    = compute_fc_arbor(W_sub, fc1_ld['input_fmap'], fc1_stim, STIM_THRESHOLD)
        pos_sub    = np.clip(raw_sub, 0, None)
        img_f_g, _, lams_g = full_nmf_pipeline(pos_sub, 4, random_state=RNG_SEED)
        group_imgs.append(img_f_g[:, 0])
        sim_g = img_factor_cosine(img_f_full, img_f_g, 0)
        print(f'  group {g} (neurons {g_idx[0]}–{g_idx[-1]}): λmax={lams_g[0]:.3f}  '
              f'img_f cosine vs full={sim_g:.3f}')

    grp_mean = np.column_stack(group_imgs).mean(axis=1)
    grp_mean = grp_mean / (np.linalg.norm(grp_mean) + 1e-12)
    full_norm = img_f_full[:, 0] / (np.linalg.norm(img_f_full[:, 0]) + 1e-12)
    sim_grp = float(full_norm @ grp_mean)
    print(f'  Averaged group img_f cosine vs full: {sim_grp:.3f}')

    # ── Summary ───────────────────────────────────────────────────────────
    print('\n─── Summary ───────────────────────────────────────────────────')
    print(f'{"Method":<28}{"img_f cosine vs full":>22}{"Notes"}')
    print('-'*60)
    cs_mean = np.mean([img_factor_cosine(img_f_full, np.column_stack(col_imgs)[:, i:i+1], 0)
                       for i in range(5)])
    print(f'{"Column sample (mean of 5)":<28}{cs_mean:>22.3f}  non-neg, interpretable')
    print(f'{"JL projection":<28}{sim_jl:>22.3f}  {neg_frac:.0%} values clipped')
    print(f'{"Neuron group (mean of 4)":<28}{sim_grp:>22.3f}  interpretable, per-group')

    print_mem('after compression validation')
else:
    print('VALIDATE_COMPRESSION = False — skipped.')
'''))

# ---------------------------------------------------------------------------
# Cell 25 – Section 8 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 8: S5 Validation — K Sensitivity

Are K=4 components per layer appropriate? We sweep K from 2 to 16 and plot the
explained variance (scree) curve. `auto_nmf_pipeline` also selects K* automatically.
'''))

# ---------------------------------------------------------------------------
# Cell 26 – S5 validation: K scree
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
if VALIDATE_K:
    print('=== S5: K sensitivity ===')

    spine_dict = {n['layer_name']: n for n in spine}
    K_sweep    = list(range(2, 17))

    fig, axes = plt.subplots(1, len(data['layer_data']), figsize=(4*len(data['layer_data']), 3.5))

    for li, ld in enumerate(data['layer_data']):
        ax = axes[li]
        # Use the stim_in from the spine node for this layer
        if ld['name'] in spine_dict:
            sw = spine_dict[ld['name']]['stim_in']
        else:
            sw = np.ones(N, dtype=np.float32)

        if ld['type'] == 'conv':
            raw = compute_conv_arbor_avg(ld['weight'], ld['input_fmap'], sw, 0.0)
        else:
            raw = compute_fc_arbor(ld['weight'], ld['input_fmap'], sw, 0.0)
        pos = np.clip(raw, 0, None)
        del raw

        # Apply compression for fc1 if needed
        if COMPRESSION_METHOD is not None and pos.shape[1] > MAX_ARBOR_DIM:
            if COMPRESSION_METHOD == 'column_sample':
                pos, _ = apply_column_sampling(pos, MAX_ARBOR_DIM, RNG_SEED)
            elif COMPRESSION_METHOD == 'jl':
                pos, _ = apply_jl_projection(pos, MAX_ARBOR_DIM, RNG_SEED)
                pos = np.clip(pos, 0, None)

        X_norm_sq = np.sum(pos ** 2)
        evs = []
        for k in K_sweep:
            if k > min(pos.shape) - 1:
                evs.append(None); continue
            _, _, nmf_m = run_nmf(pos, k, random_state=RNG_SEED)
            ev = 1.0 - nmf_m.reconstruction_err_ ** 2 / (X_norm_sq + 1e-12)
            evs.append(float(ev))

        valid_ks = [k for k, ev in zip(K_sweep, evs) if ev is not None]
        valid_ev = [ev for ev in evs if ev is not None]
        ax.plot(valid_ks, valid_ev, 'b-o', markersize=4)

        # Auto K*
        img_f_auto, _, lams_auto, k_star = auto_nmf_pipeline(
            pos, k_max=min(16, min(pos.shape)-1), method='structural', random_state=RNG_SEED)
        ax.axvline(k_star, color='red', ls='--', lw=1.5, label=f'auto K*={k_star}')
        ax.axvline(4, color='gray', ls=':', lw=1, label='K=4 (NB04)')
        ax.set_title(f'{ld["name"]}', fontsize=9)
        ax.set_xlabel('K'); ax.set_ylabel('Expl. var.')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        print(f'  {ld["name"]}: auto K*={k_star}  NB04 K=4  '
              f'EV@4={evs[K_sweep.index(4)] if 4 in K_sweep else "n/a":.3f}  '
              f'EV@K*={evs[K_sweep.index(k_star)] if k_star in K_sweep else "n/a":.3f}')
        del pos

    plt.suptitle('K sensitivity (scree) per layer — red dashed = auto K*, gray dotted = NB04 K=4')
    plt.tight_layout(); plt.show()
else:
    print('VALIDATE_K = False — skipped.')
'''))

# ---------------------------------------------------------------------------
# Cell 27 – Section 9 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 9: Seed Stability

NMF is non-convex: different random seeds can recover different (but equivalent)
local optima. We run the full trace 3 times and measure pairwise cosine similarity
of the top neural factor at each layer. Cosine > 0.90 indicates the factor is stable.
'''))

# ---------------------------------------------------------------------------
# Cell 28 – S8 seed stability
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
if VALIDATE_SEEDS:
    print('=== S8: NMF seed stability (3 seeds) ===')
    seeds        = [0, 42, 123]
    seed_spines  = {}

    for seed in seeds:
        print(f'\n--- seed = {seed} ---')
        t_ = cnn_tree_trace(
            data['layer_data'],
            K_list=K_LIST if K_LIST is not None else [4,4,4,4,10],
            k_max_auto=K_MAX_AUTO,
            n_branches_fc=1, n_branches_conv=1,
            threshold=STIM_THRESHOLD,
            compression=COMPRESSION_METHOD, max_dim=MAX_ARBOR_DIM,
            min_active=MIN_ACTIVE, random_state=seed, store_joint=False)
        seed_spines[seed] = {n['layer_name']: n for n in get_spine(t_)}

    layer_names = list(seed_spines[seeds[0]].keys())
    pairs = [(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(i+1, len(seeds))]

    print('\nPairwise cosine similarity of top neural factor:')
    hdr = f'{"Layer":<10}' + '  '.join(f'({a},{b})' for a, b in pairs)
    print(hdr); print('─'*len(hdr))

    for ln in layer_names:
        sims = []
        for (sa, sb) in pairs:
            na = seed_spines[sa].get(ln); nb = seed_spines[sb].get(ln)
            if na and nb:
                sims.append(top_cosine_sim(na['neural_factors'], nb['neural_factors']))
            else:
                sims.append(float('nan'))
        row    = '  '.join(f'{s:>7.3f}' for s in sims)
        stable = 'STABLE' if all(s > 0.90 for s in sims if not np.isnan(s)) else 'UNSTABLE'
        print(f'{ln:<10} {row}  {stable}')

    print('\nInterpretation: cosine > 0.90 = stable; < 0.90 = consider more seeds or higher K')
else:
    print('VALIDATE_SEEDS = False — skipped.')
'''))

# ---------------------------------------------------------------------------
# Cell 29 – Section 10 header
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 10: S6 Validation — Conv Layer Branching (Optional)

NB04 restricts branching to FC layers only. Enabling branching at conv layers reveals
whether multiple distinct visual patterns exist at each conv layer — at the cost of
exponentially more NMF runs.
'''))

# ---------------------------------------------------------------------------
# Cell 30 – S6 conv branching
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
if VALIDATE_BRANCHING:
    print(f'=== S6: Conv branching (N_BRANCHES_CONV={N_BRANCHES_CONV}) ===')
    t0 = time.time()
    tree_branched = cnn_tree_trace(
        data['layer_data'],
        K_list=K_LIST if K_LIST is not None else [4,4,4,4,10],
        k_max_auto=K_MAX_AUTO,
        n_branches_fc=N_BRANCHES_FC,
        n_branches_conv=N_BRANCHES_CONV,
        threshold=STIM_THRESHOLD,
        compression=COMPRESSION_METHOD, max_dim=MAX_ARBOR_DIM,
        min_active=MIN_ACTIVE, random_state=RNG_SEED, store_joint=False)
    paths_b = get_all_paths(tree_branched)
    print(f'Done in {time.time()-t0:.1f}s  |  {len(paths_b)} leaf paths '
          f'(vs {len(paths)} without conv branching)')

    # Class × path heatmap
    targets = data['targets']
    heatmap = np.zeros((10, len(paths_b)))
    path_labels = []
    for pi, path in enumerate(paths_b):
        fc1_node = next((n for n in path if n['layer_name'] == 'fc1'), None)
        if fc1_node is None:
            path_labels.append(f'p{pi}'); continue
        fi = min(path[-1]['factor_idx'], fc1_node['img_factors'].shape[1]-1)
        sw = fc1_node['img_factors'][:, fi]
        for c in range(10):
            heatmap[c, pi] = sw[targets == c].mean()
        path_labels.append('->'.join(str(b) for b in path[-1]['path']))

    fig, ax = plt.subplots(figsize=(max(6, len(paths_b)*1.5), 5))
    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(paths_b))); ax.set_xticklabels(path_labels, fontsize=8)
    ax.set_yticks(range(10)); ax.set_yticklabels(CIFAR10_CLASSES)
    ax.set_title(f'Class × path  (conv branching N={N_BRANCHES_CONV})')
    plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()
else:
    print('VALIDATE_BRANCHING = False — skipped.')
'''))

# ---------------------------------------------------------------------------
# Cell 31 – Section 11: Summary
# ---------------------------------------------------------------------------
cells.append(md_cell(r'''
## Section 11: Summary & Recommendations

Final recommendations based on the validation results above.
'''))

# ---------------------------------------------------------------------------
# Cell 32 – Summary table
# ---------------------------------------------------------------------------
cells.append(code_cell(r'''
print('=' * 80)
print('SUMMARY: Simplification Recommendations')
print('=' * 80)

rows = [
    ('S1: Confidence filter', str(USE_CONFIDENCE_FILTER),
     'Optional',
     'Bias toward easy examples. Cosine > 0.85 vs random -> safe. '
     'Recommend: top-60 or top-200 for diversity.'),
    ('S2: Spatial avg-pool', 'ALWAYS ON',
     'REQUIRED',
     '1000x arbor reduction for conv layers. Full spatial infeasible. '
     'Avg vs max: similar factors — avg preferred for global summary.'),
    ('S3: Stim. threshold', f'{STIM_THRESHOLD}',
     'Optional',
     'Cascades — can leave < 30 active samples at deep conv layers. '
     'Default 0.0 (off). Only enable if NMF is too slow.'),
    ('S4: Compression', str(COMPRESSION_METHOD),
     'Optional at N=60/class',
     'JL clips negatives -> breaks NMF guarantee. '
     'Column-sample preferred (non-neg, interpretable). '
     'Neuron-group NMF best for fc interpretability.'),
    ('S5: Fixed K',  str(K_LIST),
     'Use auto_nmf',
     'auto_nmf_pipeline selects K* from scree. '
     'K=4 may under/overfit. Run scree above to verify.'),
    ('S6: No conv branch', str(not BRANCH_AT_CONV),
     'Optional',
     'Branching at conv layers rarely changes class assignments. '
     'Skip for first pass; enable if paths look identical.'),
]

print(f'\n{"Simplification":<28}{"Current":<12}{"Verdict":<22}Notes')
print('─' * 100)
for name, current, verdict, notes in rows:
    print(f'{name:<28}{current:<12}{verdict:<22}{notes[:60]}')
    if len(notes) > 60:
        print(f'{"":<62}{notes[60:]}')
print()
print('Minimal valid config:  USE_SPATIAL_POOLING=True + STIM_THRESHOLD=0.0 + K_LIST=None')
print('Add COMPRESSION_METHOD="column_sample" if N_TOP_PER_CLASS > 300.')
print('Done.')
'''))

# ---------------------------------------------------------------------------
# Build notebook JSON
# ---------------------------------------------------------------------------
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        }
    },
    "cells": cells,
}

out = os.path.join(os.path.dirname(__file__), '..', 'notebooks',
                   '06_simplification_validation.ipynb')
out = os.path.normpath(out)
with open(out, 'w') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f'Written: {out}')
print(f'Cells:   {len(cells)}')
