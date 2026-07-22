"""Turn a BFT trace into a *superset* of plot-ready arrays.

The notebooks run where the data is; the figures are drawn elsewhere from a
``figdata`` bundle. Whatever is not exported here cannot be plotted later without
re-running the trace, so these helpers deliberately export more than any current
figure uses: every factor (not just the ones a panel shows), both signs, the
per-stimulus loadings, the top stimuli, and the class profiles.

Everything returned is plain numpy / python — no BFTNode ever reaches a bundle.

    from src import figdata, figexport
    D = figdata.save('nb03_circuits', dict(
        meta=figexport.trace_meta(tree, k_max=K_MAX, n_branches=N_BRANCHES),
        nodes=figexport.export_tree(tree.root, labels=all_targets0,
                                    classes=range(10), images=all_images0),
        stimuli=figexport.example_stimuli(all_images0, all_targets0, range(10))))

Size guards matter for the big models: connection matrices above ``max_matrix``
elements are stored as marginals instead of in full, and images are downsampled
(``img_max_side``) before the per-factor averages and examples are stored.
"""

from __future__ import annotations

import numpy as np

__all__ = ['factor_matrix', 'class_profile', 'node_summary', 'export_tree',
           'scaffold_summary', 'example_stimuli', 'trace_meta', 'subsample_by_class']


# ── small helpers ─────────────────────────────────────────────────────────────

def _downsample(imgs, max_side):
    """(N, C, H, W) -> stride-subsampled copy with H, W <= max_side."""
    if imgs.ndim != 4 or max(imgs.shape[-2:]) <= max_side:
        return imgs
    step = int(np.ceil(max(imgs.shape[-2:]) / max_side))
    return imgs[..., ::step, ::step]


def subsample_by_class(labels, classes, per_class, seed=0):
    """Indices of `per_class` stimuli per class — for anything too big to export whole."""
    rng = np.random.default_rng(seed)
    out = []
    for c in classes:
        idx = np.where(np.asarray(labels) == c)[0]
        if len(idx):
            out.append(rng.choice(idx, min(per_class, len(idx)), replace=False))
    return np.concatenate(out) if out else np.array([], dtype=int)


# ── one factor's connection map, whatever the layer type ─────────────────────

def factor_matrix(node, k, neg=False, aggregate_conv=True):
    """Factor k's connection column, shaped (out unit, in unit).

    fc / attn : (n_out, n_in) straight from the weight shape.
    conv      : (C_out, C_in) with the kernel summed out (aggregate_conv), else
                (C_out, C_in * kH * kW).
    """
    F = node.neg_connection_factors if neg else node.connection_factors
    if F is None:
        return None
    W = node.weight
    if node.layer_type == 'conv':
        C_out, C_in, kH, kW = W.shape
        m = F[:, k].reshape(C_out, C_in, kH, kW)
        return m.sum((-2, -1)) if aggregate_conv else m.reshape(C_out, C_in * kH * kW)
    n_out, n_in = W.shape[0], int(np.prod(W.shape[1:]))
    return F[:, k].reshape(n_out, n_in)


def _conn_export(node, neg, max_matrix):
    """All factors' connection maps, or their marginals when that is too big."""
    F = node.neg_connection_factors if neg else node.connection_factors
    if F is None:
        return None
    mats = [factor_matrix(node, k, neg=neg) for k in range(F.shape[1])]
    M = np.stack(mats)                                   # (K, out, in)
    if M.size <= max_matrix:
        return dict(full=M, out_mass=M.sum(2), in_mass=M.sum(1))
    return dict(out_mass=M.sum(2), in_mass=M.sum(1))     # marginals only


def class_profile(node, labels, classes, neg=False):
    """(K, n_classes): share of each factor's mean loading contributed by each class."""
    H = node.neg_img_factors if neg else node.img_factors
    if H is None:
        return None
    labels = np.asarray(labels)
    out = np.zeros((H.shape[1], len(list(classes))))
    for j, c in enumerate(classes):
        m = labels == c
        if m.any():
            out[:, j] = H[m].mean(0)
    return out / (out.sum(1, keepdims=True) + 1e-12)


# ── one node ──────────────────────────────────────────────────────────────────

def node_summary(node, *, labels=None, classes=None, images=None, n_top=8,
                 stim_idx=None, max_matrix=400_000, img_max_side=64,
                 example_max_side=112):
    """Everything about one trace node that a figure could plausibly need.

    `stim_idx` limits the *per-stimulus* arrays (which scale with N) to a subset;
    class profiles and top stimuli are still computed on all stimuli.
    """
    H, W = node.img_factors, node.weight
    K = H.shape[1]
    lam = np.asarray(node.lambdas, float)
    keep = slice(None) if stim_idx is None else np.asarray(stim_idx)
    d = dict(
        path=np.asarray(node.path, dtype=int) if len(node.path) else np.zeros(0, int),
        layer_idx=int(node.layer_idx), layer_name=str(node.layer_name),
        layer_type=str(node.layer_type), n_factors=K,
        weight_shape=np.asarray(W.shape, dtype=int),
        lam=lam, lam_share=lam / (lam.sum() + 1e-12),
        img_factors=H[keep].astype(np.float32),           # (n_kept, K) loadings
        stimulus_weights=np.asarray(node.stimulus_weights)[keep].astype(np.float32),
        top_idx=np.stack([np.argsort(H[:, k])[::-1][:n_top] for k in range(K)]),
    )
    if stim_idx is not None:
        d['stim_idx'] = np.asarray(stim_idx)
    if node.neg_lambdas is not None:
        nl = np.asarray(node.neg_lambdas, float)
        d['neg_lam'] = nl
        d['neg_lam_share'] = nl / (nl.sum() + 1e-12)
    if node.neg_img_factors is not None:
        d['neg_img_factors'] = node.neg_img_factors[keep].astype(np.float32)

    conn = _conn_export(node, False, max_matrix)
    if conn:
        d['conn'] = conn
    neg_conn = _conn_export(node, True, max_matrix)
    if neg_conn:
        d['neg_conn'] = neg_conn

    # input-side conv nodes: the kernels themselves are what one plots (RGB)
    if node.layer_type == 'conv' and W.ndim == 4 and W.shape[1] in (1, 3):
        C_out, C_in, kH, kW = W.shape
        d['input_kernels'] = np.stack([
            node.connection_factors[:, k].reshape(C_out, C_in, kH, kW)
            for k in range(K)]).astype(np.float32)
    if node.layer_type == 'attn' and node.attn_weights is not None:
        A = np.asarray(node.attn_weights)                 # (N, T) CLS-row scores
        d['attn_mean'] = np.stack([
            (H[:, k, None] * A).sum(0) / (H[:, k].sum() + 1e-12) for k in range(K)])

    if labels is not None and classes is not None:
        d['class_profile'] = class_profile(node, labels, classes)
        if node.neg_img_factors is not None:
            d['neg_class_profile'] = class_profile(node, labels, classes, neg=True)

    if images is not None:
        imgs = _downsample(np.asarray(images), img_max_side)
        flat = imgs.reshape(len(imgs), -1)
        d['wavg'] = np.stack([(H[:, k, None] * flat).sum(0) / (H[:, k].sum() + 1e-12)
                              for k in range(K)]).reshape((K,) + imgs.shape[1:])
        ex = _downsample(np.asarray(images), example_max_side)
        d['top_images'] = np.stack([ex[d['top_idx'][k]] for k in range(K)])
    return d


def export_tree(root, **kw):
    """Depth-first list of node summaries — the whole trace, ready for a bundle."""
    out, stack = [], [root]
    while stack:
        n = stack.pop(0)
        out.append(node_summary(n, **kw))
        stack.extend(n.children)
    return out


# ── circuits and stimuli ──────────────────────────────────────────────────────

def scaffold_summary(edges, neg_edges, loading, layer_sizes):
    """One traced circuit's graph, as plain arrays."""
    return dict(edges=[np.asarray(e, np.float32) for e in edges],
                neg_edges=[np.asarray(e, np.float32) for e in neg_edges],
                loading=np.asarray(loading, np.float32),
                layer_sizes=np.asarray(layer_sizes, int))


def example_stimuli(images, labels, classes, per_class=8, max_side=112, seed=0):
    """A few real stimuli per class, small enough to commit."""
    idx = subsample_by_class(labels, classes, per_class, seed=seed)
    imgs = _downsample(np.asarray(images)[idx], max_side)
    return dict(images=imgs.astype(np.float32), labels=np.asarray(labels)[idx],
                index=idx)


def trace_meta(result_or_root, **config):
    """Model- and trace-level scalars worth keeping next to the arrays."""
    root = getattr(result_or_root, 'root', result_or_root)
    meta = {k: (list(v) if isinstance(v, (list, tuple)) else v)
            for k, v in config.items() if v is not None}
    meta.update(root_layer=int(root.layer_idx), root_factors=int(root.img_factors.shape[1]),
                n_stimuli=int(root.img_factors.shape[0]))
    return meta
