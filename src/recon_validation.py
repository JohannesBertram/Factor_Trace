"""Causal reconstruction validation for BFT factorizations.

The NMF reconstruction error reported during a BFT trace is purely algebraic: it
measures how well the factors reproduce the *joint arbor matrix*. It says nothing
about whether the factorized approximation, when used *in place of* the real
activity, still drives the rest of the network correctly.

This module provides a *causal* check. For a single layer's factorization it:

  1. reconstructs the layer's pre-activation implied by the factors
     (signed: positive minus inhibitory),
  2. injects that reconstruction in place of the layer's real output via a forward
     hook, runs the rest of the model normally, and
  3. measures the cross-entropy loss ratio
        loss_ratio = CE(recon_forward, targets) / CE(real_forward, targets)
     A value near 1.0 means the factorized activity is as usable as the real
     activity; larger values mean the factorization loses task-relevant structure.

Reconstruction math
-------------------
BFT factorizes the *normalized, stimulus-weighted, connection-weighted, threshold-
zeroed* joint arbor (see ``compute_joint_arbors_normalized`` in ``bft.py``). For an
FC/attn layer the arbor row for output neuron ``i`` on sample ``s`` is

    arbor[s, i, :] = stim_w[s] * cw[i] * (act_norm[s] * W[i])

with ``act_norm = act / ||act||`` (no normalization when stimulus weights are
uniform). Summing the *signed* reconstructed arbor over the input axis gives the
pre-activation in this weighted space; dividing out ``stim_w``, ``cw[i]`` and
multiplying back the per-sample norm recovers an estimate of the true pre-activation
``z = W @ act``. Output neurons with ``cw[i] == 0`` are not modeled and fall back to
the real pre-activation.

Only ``fc`` layers are validated. ``conv`` layers pool away spatial structure so the
full feature map is not recoverable; ``attn`` layers (TinyViT) are collected in
layer-dict mode where no model is available to run the forward pass.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── reconstruction ────────────────────────────────────────────────────────────

def reconstruct_preactivation(W, act_input, img_f, con_f, neg_img_f, neg_con_f,
                              stimulus_weights, connection_weights,
                              stimulus_threshold, eps=1e-8, factor_subset=None):
    """Reconstruct an FC layer's true-scale pre-activation from its NMF factors.

    Mirrors and inverts the weighting applied in ``compute_joint_arbors_normalized``.

    Parameters
    ----------
    W                : (n_out, n_in) layer weight matrix
    act_input        : (N, n_in) input activations to the layer
    img_f, con_f     : (N, K) / (n_out*n_in, K) positive NMF factors
    neg_img_f, neg_con_f : inhibitory factors, or None
    stimulus_weights : (N,) per-sample importance used during tracing
    connection_weights : (n_out,) per-neuron importance, or None (root layer)
    stimulus_threshold : fraction in [0, 1) zeroed during tracing
    factor_subset    : sequence of positive-factor indices to use, or None (all)

    Returns
    -------
    z_recon     : (N, n_out) reconstructed pre-activation (pre-bias)
    active_mask : (N,) bool — samples that were not threshold-zeroed during tracing
    """
    N = act_input.shape[0]
    n_out = W.shape[0]
    n_in = W.shape[1]

    S = list(range(con_f.shape[1])) if factor_subset is None else list(factor_subset)
    X = img_f[:, S] @ con_f[:, S].T                         # (N, n_out*n_in)
    if neg_img_f is not None and neg_con_f is not None:
        X = X - (neg_img_f @ neg_con_f.T)
    z_norm = X.reshape(N, n_out, n_in).sum(axis=2)          # (N, n_out)

    uniform = stimulus_weights is None or np.allclose(stimulus_weights, 1.0)
    if uniform:
        # No normalization / stimulus weighting was applied during tracing.
        z = z_norm
    else:
        norms = np.linalg.norm(act_input, axis=1, keepdims=True) + eps  # (N, 1)
        sw = np.where(stimulus_weights > 0, stimulus_weights, 1.0)[:, None]
        z = z_norm * norms / sw

    if connection_weights is not None:
        cw = np.asarray(connection_weights, dtype=z.dtype)
        modeled = cw > 0
        z[:, modeled] = z[:, modeled] / cw[modeled][None, :]
        # Unmodeled neurons (cw == 0) fall back to the real pre-activation.
        if (~modeled).any():
            real_z = act_input @ W.T                        # (N, n_out)
            z[:, ~modeled] = real_z[:, ~modeled]

    # Samples zeroed by the stimulus threshold carry no arbor signal.
    if stimulus_threshold > 0.0 and not uniform:
        cutoff = np.quantile(stimulus_weights, stimulus_threshold)
        active_mask = stimulus_weights > cutoff
    else:
        active_mask = np.ones(N, dtype=bool)

    return z.astype(np.float32), active_mask


# ── forward passes ─────────────────────────────────────────────────────────────

def _logprobs_from_output(out):
    """Return log-probabilities from a model output of unknown convention.

    Handles raw logits (SmallCNN), log_softmax output (TinyViT), and softmax
    probabilities (SimpleMLP).
    """
    lse = out.logsumexp(dim=1)
    if torch.allclose(lse, torch.zeros_like(lse), atol=1e-3):
        return out                                          # already log_softmax
    row_sums = out.sum(dim=1)
    if out.min() >= 0 and torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        return torch.log(out.clamp_min(1e-12))              # softmax probs
    return F.log_softmax(out, dim=1)                        # raw logits


def _model_logits(model, x):
    out = model(x)
    if isinstance(out, tuple):
        out = out[0]
    return out


def _forward_ce(model, images, targets, device, hook_module=None, z_recon=None,
                batch_size=512):
    """Mean cross-entropy of model(images) vs targets, in row order.

    When ``hook_module`` and ``z_recon`` are given, the module's output is replaced
    by ``z_recon + module.bias`` for the corresponding rows (BFT sample order).
    """
    model.eval()
    n = len(images)
    bias = None
    if hook_module is not None and getattr(hook_module, 'bias', None) is not None:
        bias = hook_module.bias.detach()

    offset = {'i': 0}

    def hook(mod, inp, out):
        b = out.shape[0]
        z = torch.from_numpy(z_recon[offset['i']:offset['i'] + b]).to(out.device, out.dtype)
        if bias is not None:
            z = z + bias.to(out.device, out.dtype)
        offset['i'] += b
        return z

    handle = hook_module.register_forward_hook(hook) if hook_module is not None else None
    total = 0.0
    try:
        with torch.no_grad():
            for s in range(0, n, batch_size):
                x = torch.as_tensor(images[s:s + batch_size]).to(device)
                y = torch.as_tensor(targets[s:s + batch_size]).to(device).long()
                logp = _logprobs_from_output(_model_logits(model, x))
                ce = F.nll_loss(logp, y, reduction='sum')
                total += float(ce)
    finally:
        if handle is not None:
            handle.remove()
    return total / max(n, 1)


# ── node-level validation ──────────────────────────────────────────────────────

def validate_node_reconstruction(model, images, targets, node, device,
                                 top_m=20, real_ce_cache=None):
    """Causal reconstruction validation for one BFT node.

    Returns None when validation is not applicable (non-fc layer, no model, or the
    layer module cannot be resolved in the model).

    The full reconstruction (all factors) is evaluated on the images that the
    currently traced path routes into this node: at the root / last layer the
    incoming trace weights are uniform, so all active samples are used; at every
    deeper layer only the ``top_m`` images with the largest incoming trace weight
    (``stimulus_weights_in``) are used.

    Parameters
    ----------
    model    : nn.Module — the traced model (primary mode only)
    images   : (N, ...) input stimuli, in BFT sample order
    targets  : (N,) labels, in BFT sample order
    node     : internal node dict from ``_trace_node``
    device   : torch device
    top_m    : number of top images (by incoming trace weight) used to evaluate
               fidelity at non-root layers; ignored at the root (all images used)
    real_ce_cache : dict mapping a frozenset of row indices -> real CE, reused
                    across nodes to avoid recomputing the real forward pass

    Returns
    -------
    dict with keys loss_ratio_all, real_ce, recon_ce, n_eval
    """
    if model is None or node.get('layer_type') != 'fc':
        return None
    name = node.get('layer_name')
    module = dict(model.named_modules()).get(name)
    if module is None:
        return None
    if real_ce_cache is None:
        real_ce_cache = {}

    W = node['W']
    act_input = node['act_input']
    img_f = node['img_factors']
    con_f = node['connection_factors']
    sw = node.get('stimulus_weights_in')
    cw = node.get('connection_weights')

    z_recon, active_mask = reconstruct_preactivation(
        W, act_input, img_f, con_f,
        node.get('neg_img_factors'), node.get('neg_connection_factors'),
        sw, cw, node.get('stimulus_threshold', 0.0),
    )

    def ce_on(idx):
        idx = np.asarray(idx)
        key = frozenset(idx.tolist())
        if key not in real_ce_cache:
            real_ce_cache[key] = _forward_ce(model, images[idx], targets[idx], device)
        real = real_ce_cache[key]
        recon = _forward_ce(model, images[idx], targets[idx], device,
                            hook_module=module, z_recon=z_recon[idx])
        return real, recon

    # Select the images the current path routes into this node. At the root the
    # incoming trace weights are uniform, so all active samples are used; deeper in
    # the tree only the top-M images by incoming weight (stimulus_weights_in) count.
    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        return None
    uniform = sw is None or np.allclose(sw, 1.0)
    if uniform:
        eval_idx = active_idx
    else:
        order = np.argsort(sw)[::-1]
        order = order[active_mask[order]]                  # keep only active samples
        eval_idx = order[:top_m]
    if len(eval_idx) == 0:
        return None

    real_all, recon_all = ce_on(eval_idx)
    loss_ratio_all = recon_all / real_all if real_all > 0 else float('inf')

    return {
        'loss_ratio_all': loss_ratio_all,
        'real_ce': real_all,
        'recon_ce': recon_all,
        'n_eval': int(len(eval_idx)),
    }


# ── aggregation ────────────────────────────────────────────────────────────────

def _stats(values):
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return {'n': 0, 'mean': float('nan'), 'median': float('nan'),
                'min': float('nan'), 'max': float('nan'), 'std': float('nan')}
    return {'n': int(len(a)), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'min': float(a.min()), 'max': float(a.max()), 'std': float(a.std())}


def summarize_validation(nodes):
    """Aggregate per-node recon_validation into per-layer and overall statistics.

    Parameters
    ----------
    nodes : iterable of BFTNode (e.g. result.nodes())

    Returns
    -------
    dict with keys 'overall', 'per_layer', and 'individual', or None if no node
    carries validation results.
    """
    individual = []
    for nd in nodes:
        rv = getattr(nd, 'recon_validation', None)
        if rv is None:
            continue
        individual.append({
            'layer_idx': nd.layer_idx,
            'layer_name': nd.layer_name,
            'path': nd.path,
            'loss_ratio_all': rv['loss_ratio_all'],
        })
    if not individual:
        return None

    per_layer = {}
    for li in sorted({d['layer_idx'] for d in individual}):
        vals = [d['loss_ratio_all'] for d in individual if d['layer_idx'] == li]
        per_layer[li] = _stats(vals)

    overall = _stats([d['loss_ratio_all'] for d in individual])
    return {'overall': overall, 'per_layer': per_layer, 'individual': individual}
