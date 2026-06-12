"""
Backward Factor Trace (BFT)
===========================
BFT decomposes the computation of a trained neural network by tracing
weight-activation products backward from the output layer to the input.

At each layer it builds a *joint arbor matrix* — the outer product of every
output neuron's weight row with the (normalised) input activations — then
factorises that matrix with NMF. The top NMF factor's per-stimulus loadings
are passed as importance weights to the preceding layer, recursively
attributing the network's output to increasingly fine-grained input patterns.

Public entry point
------------------
    bft(model, loader, ...)
        Primary interface. Pass any nn.Module and a DataLoader; layer data is
        collected automatically via forward hooks on all Conv2d and Linear
        sub-modules, then BFT is run. The returned BFTResult includes
        images, targets, and confidences metadata.

    bft(layer_dicts, ...)
        Advanced / backward-compat mode. Pass a pre-collected list of dicts,
        one per layer in forward order:
            {'type': 'fc' | 'conv' | 'attn',
             'weight': ndarray,          # fc: (n_out,n_in); conv: (C_out,C_in,kH,kW);
                                         # attn: (d_v,d_model) value-projection W_V
             'input_fmap': ndarray,      # fc: (N,n_in); conv: (N,C_in,H,W);
                                         # attn: (N,T,d_model) all token activations
             'attn_weights': ndarray}    # attn only: (N,T) CLS-row scores (head-avg'd)

    bft(model, layer_inputs_list, ...)
        Legacy model-protocol mode (SimpleMLP). model must expose
        linear_layer_indices() and model.layers[i].weight.

collect_layer_dicts(model, loader, device, only_correct=True)
    Public helper: run forward hooks and return layer data suitable for the
    layer-dict mode of bft(). Useful when you need to inspect or reuse the
    collected activations separately from running BFT.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import MiniBatchNMF
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from .types import BFTNode, BFTResult


# ── NMF building blocks ───────────────────────────────────────────────────────

def _safe_init(n_components, n_samples, n_features):
    """Use nndsvda unless it violates n_components <= min(n_samples, n_features)."""
    if n_components > min(n_samples, n_features):
        return 'random'
    return 'nndsvda'


def run_nmf_minibatch(X, n_components, random_state=0, max_iter=500,
                      batch_size=None, init=None, l1_ratio=0,
                      n_jobs=None, **kwargs):
    """MiniBatchNMF (online MU) in float32.

    Parameters
    ----------
    X          : (n_samples, n_features) non-negative matrix
    batch_size : rows per mini-batch; defaults to max(64, min(1024, n_samples//4)),
                 which keeps ~4 batches per epoch across all typical input sizes.
    l1_ratio   : float in [0, 1] — regularisation mix (0 = L2, 1 = L1).
                 Passed directly to MiniBatchNMF.
    n_jobs     : BLAS thread count (via threadpoolctl); None = OS default.
                 Mini-batches are processed sequentially — n_jobs does NOT
                 parallelise batches; it only widens the BLAS thread pool
                 for matrix multiplications inside each step.
    """
    X32 = X.astype(np.float32, copy=False)
    n_s, n_f = X32.shape
    if batch_size is None:
        batch_size = max(64, min(1024, n_s // 4))
    resolved_init = init if init is not None else _safe_init(n_components, n_s, n_f)
    nmf = MiniBatchNMF(n_components=n_components, init=resolved_init,
                       random_state=random_state, max_iter=max_iter,
                       batch_size=batch_size, l1_ratio=l1_ratio,
                       tol=1e-3, max_no_improvement=5, **kwargs)
    if n_jobs is not None:
        with threadpool_limits(limits=n_jobs):
            W = nmf.fit_transform(X32)
    else:
        W = nmf.fit_transform(X32)
    H = nmf.components_.T
    return W, H, nmf


def normalize_factors(W, H):
    """Rescale W and H to unit column-norm; return absorbed norms as lambdas.

    Each column of W and H is divided by its L2 norm.  The product of the
    two norms (lambda_k = ||W[:,k]|| * ||H[:,k]||) captures the overall
    importance of component k.

    Returns (W_norm, H_norm, lambdas).
    """
    W, H = W.copy(), H.copy()
    w_norms = np.linalg.norm(W, axis=0)
    active = ~np.isclose(w_norms, 0)
    W[:, active] /= w_norms[active]

    h_norms = np.linalg.norm(H, axis=0)
    active = ~np.isclose(h_norms, 0)
    H[:, active] /= h_norms[active]

    lambdas = w_norms * h_norms
    return W, H, lambdas


def sort_by_lambda(W, H, lambdas):
    """Sort components by descending lambda. Returns (W, H, lambdas, sort_idx)."""
    idx = np.argsort(lambdas)[::-1]
    return W[:, idx], H[:, idx], lambdas[idx], idx


def _partial_recon_errors(X, W, H):
    """Relative Frobenius reconstruction error for K=1..kmax using a k_max NMF fit.

    W : (N, kmax)        — sqrt(lambda)-scaled img_factors, sorted descending
    H : (features, kmax) — sqrt(lambda)-scaled connection_factors, sorted descending
    Returns recon_errors : (kmax,) where recon_errors[k] = ||X - X_{k+1}||_F / ||X||_F
    """
    X_norm_sq = float(np.dot(X.ravel(), X.ravel()))
    if X_norm_sq == 0:
        return np.zeros(W.shape[1])
    kmax = W.shape[1]
    # a[k] = W[:,k]^T @ X @ H[:,k] — cross term for component k
    WtX = W.T @ X                           # (kmax, features)
    a   = np.einsum('kf,fk->k', WtX, H)    # (kmax,)
    # Gram matrices for ||X_K||^2 = sum_{j,l<K} GW[j,l] * GH[j,l]
    G   = (W.T @ W) * (H.T @ H)            # element-wise Gram product, (kmax, kmax)
    recon_errors = np.empty(kmax)
    for k in range(kmax):
        K = k + 1
        err_sq = X_norm_sq - 2.0 * a[:K].sum() + G[:K, :K].sum()
        recon_errors[k] = np.sqrt(max(0.0, err_sq) / X_norm_sq)
    return recon_errors


def _select_k_from_recon(recon_errors, threshold, min_k=1):
    """Smallest K where relative reconstruction error <= threshold.

    recon_errors : (kmax,) array from _partial_recon_errors (index k → rank K=k+1)
    threshold    : maximum acceptable relative Frobenius error (e.g. 0.2 = 20%)
    """
    passing = np.where(recon_errors <= threshold)[0]
    if len(passing) == 0:
        return len(recon_errors)    # even kmax doesn't meet threshold — use all
    return max(min_k, int(passing[0]) + 1)


def _select_k_single(lambdas, min_k=1):
    """K at the first consecutive-ratio drop larger than 1.5x (fraction method).

    +1e-12 avoids division by zero when a lambda is exactly 0.
    """
    for k in range(len(lambdas) - 1):
        if lambdas[k] / (lambdas[k + 1] + 1e-12) >= 1.5:
            return max(k, min_k)
    return max(len(lambdas) - 1, min_k)


# ── NMF pipeline ──────────────────────────────────────────────────────────────

def full_nmf_pipeline(X, n_components, random_state=0, max_iter=500, init=None,
                      l1_ratio=0, **factorizer_kwargs):
    """Fit MiniBatchNMF (f32), normalise, sort by importance, and rescale by sqrt(lambda).

    The sqrt(lambda) rescaling distributes importance equally between the
    image factors (W) and connection factors (H) so that their dot product
    reconstructs X with unit-norm columns carrying equal weight.

    Returns (img_factors, connection_factors, lambdas).
      img_factors         : (n_samples, n_components)
      connection_factors  : (n_connections, n_components)
      lambdas             : (n_components,) descending
    """
    W, H, _ = run_nmf_minibatch(X, n_components, random_state=random_state,
                                 max_iter=max_iter, init=init,
                                 l1_ratio=l1_ratio, **factorizer_kwargs)
    W, H, lambdas = normalize_factors(W, H)
    W, H, lambdas, _ = sort_by_lambda(W, H, lambdas)
    scale = np.sqrt(lambdas)
    return W * scale, H * scale, lambdas


def auto_nmf_pipeline(X, k_max=None, random_state=0, max_iter=500, init=None,
                      l1_ratio=0, recon_threshold=None, **factorizer_kwargs):
    """Fit NMF at rank k_max then automatically select effective rank K*.

    A single NMF fit is performed at k_max; components are pruned to K* so
    re-fitting at every candidate rank is avoided.

    K* is selected via the structural_recon heuristic: take the max of the
    consecutive-ratio lambda drop (>= 1.5×) and the reconstruction-error floor.

    Parameters
    ----------
    X               : (n_samples, n_features) non-negative matrix
    k_max           : int or None — upper bound on rank; None → min(min(X.shape)-1, 20)
    recon_threshold : float or None — maximum acceptable relative Frobenius reconstruction
                      error. Default None uses 0.4 (40% error).

    Returns
    -------
    img_factors        : (n_samples, k_star)
    connection_factors : (n_features, k_star)
    lambdas            : (k_star,) descending
    k_star             : int — automatically selected rank
    """
    if k_max is None:
        k_max = min(min(X.shape) - 1, 20)
    k_max = max(int(k_max), 2)

    img_f, con_f, lams = full_nmf_pipeline(X, k_max, random_state=random_state,
                                            max_iter=max_iter, init=init,
                                            l1_ratio=l1_ratio, **factorizer_kwargs)

    rt = recon_threshold if recon_threshold is not None else 0.4
    recon_errs = _partial_recon_errors(X, img_f, con_f)
    k_frac  = _select_k_single(lams, min_k=1)
    k_recon = _select_k_from_recon(recon_errs, rt, min_k=1)
    k_star  = max(1, min(max(k_frac, k_recon), len(lams)))

    return img_f[:, :k_star], con_f[:, :k_star], lams[:k_star], k_star


# ── BFT core ──────────────────────────────────────────────────────────────────

def compute_joint_arbors_normalized(W, act_input, stimulus_weights=None, eps=1e-8,
                                    stimulus_threshold=0.0, connection_weights=None):
    """Compute the normalised, stimulus-weighted joint arbor matrix for an FC layer.

    A connection's *arbor* for a given stimulus is the element-wise product of its
    weight row and the (normalised) input activation: W[i] * act_norm[s].
    This quantity represents the synaptic contribution of each input dimension
    to neuron i's pre-activation on stimulus s.

    The *joint* arbor concatenates all neurons' arbors horizontally so that
    a single NMF can discover patterns shared across neurons simultaneously.

    Parameters
    ----------
    W                   : (n_out, n_in) layer weight matrix
    act_input           : (n_samples, n_in) input activations to this layer
    stimulus_weights    : (n_samples,) per-sample importance; uniform if None
    eps                 : stabiliser for L2 norm division (not exposed externally)
    stimulus_threshold  : fraction in [0, 1); samples in the lowest
                          stimulus_threshold quantile of weights are zeroed.
                          E.g. 0.1 zeros the bottom 10% of weights.
    connection_weights  : (n_out,) per-output-neuron importance from the layer
                          above; each neuron i's arbor is scaled by this scalar.
                          None = all neurons equally weighted.

    Returns
    -------
    joint_arbor : (n_samples, n_out * n_in), signed — caller clips as needed
    """
    # L2-normalize to remove per-stimulus energy differences so NMF recovers pattern structure.
    if stimulus_weights is not None and np.allclose(stimulus_weights, 1.0):
        act_norm = act_input
    else:
        norms = np.linalg.norm(act_input, axis=1, keepdims=True)
        act_norm = act_input / (norms + eps)

    # Scale by connection_weights when available (importance from layer above).
    if connection_weights is not None:
        arbors = [act_norm * W[i] * connection_weights[i] for i in range(W.shape[0])]
    else:
        arbors = [act_norm * W[i] for i in range(W.shape[0])]

    # Joint matrix enables NMF to find cross-neuron patterns.
    joint = np.concatenate(arbors, axis=1)

    if stimulus_weights is not None:
        joint = joint * stimulus_weights[:, np.newaxis]

    # Zero out lowest-weight stimuli to avoid diluting factor structure.
    if stimulus_threshold > 0.0 and (stimulus_weights is not None) and not np.allclose(stimulus_weights, 1.0):
        cutoff = np.quantile(stimulus_weights, stimulus_threshold)
        joint[stimulus_weights <= cutoff] = 0.0

    return joint


def compute_conv_joint_arbors(weight, input_fmap, stimulus_weights=None, eps=1e-8,
                               stimulus_threshold=0.0, connection_weights=None,
                               pool_method='avg'):
    """Compute the normalised, stimulus-weighted joint arbor matrix for a Conv2d layer.

    Analogous to compute_joint_arbors_normalized but handles spatial feature maps.
    The spatial dimension is collapsed via pooling before forming the arbor, so that
    a single (N, C_out * C_in * kH * kW) matrix represents the layer's computation
    in the same format as an FC joint arbor.

    Parameters
    ----------
    weight           : (C_out, C_in, kH, kW) conv weight tensor as numpy array
    input_fmap       : (N, C_in, H, W) input feature map as numpy array
    stimulus_weights : (N,) per-sample importance; uniform if None
    eps              : stabiliser for L2 norm division (internal only)
    stimulus_threshold : fraction in [0, 1) — zeros the lowest-weight stimuli
    connection_weights : (C_out,) per-output-channel importance from the layer above;
                         each output channel c's arbor block is scaled by
                         connection_weights[c]. None = all channels equally weighted.
    pool_method      : str — how to reduce the spatial dimension of extracted patches:
                       'avg'    mean over all spatial positions (default)
                       'max'    max over all spatial positions
                       'center' single center spatial position only

    Returns
    -------
    joint_arbor : (N, C_out * C_in * kH * kW), signed — caller clips as needed
    """
    N, C_in, H, W_in = input_fmap.shape
    C_out, _, kH, kW = weight.shape
    W_flat = weight.reshape(C_out, C_in * kH * kW)   # (C_out, C_in*kH*kW)

    # Extract patches via im2col with padding to preserve spatial size.
    pad = (kH // 2, kW // 2)
    fmap_t = torch.from_numpy(input_fmap).float()
    patches = F.unfold(fmap_t, kernel_size=(kH, kW), padding=pad).numpy()

    # Reduce spatial dimension to (N, C_in*kH*kW).
    if pool_method == 'avg':
        pooled = patches.mean(axis=2)                          # (N, C_in*kH*kW)
    elif pool_method == 'max':
        pooled = patches.max(axis=2)                           # (N, C_in*kH*kW)
    elif pool_method == 'center':
        ctr_idx = (H // 2) * W_in + (W_in // 2)               # flat index of center position
        pooled = patches[:, :, ctr_idx]                        # (N, C_in*kH*kW)
    else:
        raise ValueError(f"Unknown pool_method {pool_method!r}. Use 'avg', 'max', or 'center'.")

    # L2-normalize to remove per-stimulus energy differences.
    if stimulus_weights is not None and np.allclose(stimulus_weights, 1.0):
        act_norm = pooled
    else:
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        act_norm = pooled / (norms + eps)

    if connection_weights is not None:
        arbors = [act_norm * W_flat[c] * connection_weights[c] for c in range(C_out)]
    else:
        arbors = [act_norm * W_flat[c] for c in range(C_out)]

    joint = np.concatenate(arbors, axis=1)

    if stimulus_weights is not None:
        joint = joint * stimulus_weights[:, np.newaxis]

    if stimulus_threshold > 0.0 and (stimulus_weights is not None) and not np.allclose(stimulus_weights, 1.0):
        cutoff = np.quantile(stimulus_weights, stimulus_threshold)
        joint[stimulus_weights <= cutoff] = 0.0

    return joint


def compute_attn_joint_arbors(W_V, x_tokens, attn_weights_cls, stimulus_weights=None,
                               eps=1e-8, stimulus_threshold=0.0, connection_weights=None):
    """Compute the normalised, stimulus-weighted joint arbor matrix for an attention layer.

    Attention mixes token representations using data-dependent weights (the softmax scores),
    making it impossible to form a single fixed weight matrix as in FC or conv layers.  The
    solution used here is to collapse the token sequence into a single *attention-weighted
    effective input* per sample:

        x_eff[n] = sum_j  attn_weights_cls[n, j] * x_tokens[n, j]   # (N, d_model)

    Because attn_weights_cls are non-negative (softmax output), x_eff is a convex combination
    of the token activations.  We then form the joint arbor exactly as for an FC layer with
    W_V as the weight matrix and x_eff as the input.

    Parameters
    ----------
    W_V               : (d_v, d_model) — value-projection weight matrix.
    x_tokens          : (N, T, d_model) — all T token activations entering the attention
                        sublayer (after layer-norm), for each of the N samples.
    attn_weights_cls  : (N, T) — attention scores from the CLS token to every token,
                        already softmax-normalised. For multi-head, pass head-averaged CLS row.
    stimulus_weights  : (N,) per-sample importance propagated from the layer above.
    eps               : stabiliser for L2 norm division (internal only)
    stimulus_threshold : fraction in [0, 1) — zeros the lowest-weight stimuli.
    connection_weights : (d_v,) per-output-dimension importance from the layer above.

    Returns
    -------
    joint_arbor : (N, d_v * d_model), signed — caller clips as needed.
    """
    # Collapse token sequence into attention-weighted effective input.
    x_eff = np.einsum('nt,ntd->nd', attn_weights_cls, x_tokens)  # (N, d_model)

    return compute_joint_arbors_normalized(W_V, x_eff, stimulus_weights,
                                           eps=eps,
                                           stimulus_threshold=stimulus_threshold,
                                           connection_weights=connection_weights)


def trace_single_layer(W, act_input, stimulus_weights, k_max=None,
                        stimulus_threshold=0.0, connection_weights=None,
                        random_state=0, max_iter=500, init=None, l1_ratio=0,
                        layer_type='fc', conv_pool_method='avg', k_fixed=None,
                        attn_weights=None, recon_threshold=None,
                        verbose=0, _layer_tag='', **factorizer_kwargs):
    """One BFT step: build joint arbors for a layer and factorise with NMF.

    Parameters
    ----------
    W                   : (n_out, n_in) for 'fc'; (C_out, C_in, kH, kW) for 'conv';
                          (d_v, d_model) for 'attn' (the value-projection W_V)
    act_input           : (n_samples, n_in) for 'fc'; (N, C_in, H, W) for 'conv';
                          (N, T, d_model) for 'attn' (all token activations, post-LN)
    stimulus_weights    : (n_samples,) per-sample importance from the layer above
    k_max               : upper bound on NMF rank; None uses auto default.
                          Ignored when k_fixed is set.
    stimulus_threshold  : passed through to the arbor-computation function
    connection_weights  : per-neuron (FC) or per-channel (conv) or per-dim (attn) importance
    random_state, max_iter, init, l1_ratio : passed through to run_nmf_minibatch
    layer_type          : 'fc' (default), 'conv', or 'attn' — selects the arbor function
    conv_pool_method    : spatial pooling for conv arbors: 'avg', 'max', 'center'
    k_fixed             : int or None — when set, use exactly this many components
                          (calls full_nmf_pipeline directly at k_fixed, bypasses auto)
    attn_weights        : (N, T) or None — required when layer_type=='attn'.

    Returns
    -------
    img_factors            : (n_samples, K*)
    connection_factors     : (n_out * n_in, K*) — for conv: (C_out * C_in * kH * kW, K*);
                             for attn: (d_v * d_model, K*)
    lambdas                : (K*,) descending
    pos_joint              : positive joint matrix (pre-NMF)
    neg_img_factors        : (n_samples, K*) or None
    neg_connection_factors : same shape as connection_factors, or None
    neg_lambdas            : (K*,) or None
    """
    if layer_type == 'conv':
        raw_joint = compute_conv_joint_arbors(W, act_input, stimulus_weights,
                                              stimulus_threshold=stimulus_threshold,
                                              connection_weights=connection_weights,
                                              pool_method=conv_pool_method)
    elif layer_type == 'attn':
        if attn_weights is None:
            raise ValueError(
                "layer_type='attn' requires attn_weights (N, T) CLS-row scores. "
                "Add 'attn_weights' to the layer dict."
            )
        raw_joint = compute_attn_joint_arbors(W, act_input, attn_weights,
                                              stimulus_weights=stimulus_weights,
                                              stimulus_threshold=stimulus_threshold,
                                              connection_weights=connection_weights)
    else:
        raw_joint = compute_joint_arbors_normalized(W, act_input, stimulus_weights,
                                                    stimulus_threshold=stimulus_threshold,
                                                    connection_weights=connection_weights)

    # Split into positive/negative parts for separate NMF factorization.
    pos_joint = np.clip(raw_joint, 0, None)
    neg_joint = np.clip(-raw_joint, 0, None)

    if verbose >= 2:
        tag = f'[BFT{_layer_tag}]'
        print(f'{tag}   joint arbor: {raw_joint.shape}  '
              f'pos {pos_joint.shape}  neg {neg_joint.shape}')

    _t0 = time.perf_counter() if verbose >= 2 else None
    if k_fixed is not None:
        k = max(int(k_fixed), 1)
        k = min(k, min(pos_joint.shape) - 1)
        img_f, con_f, lams = full_nmf_pipeline(
            pos_joint, k, random_state=random_state, max_iter=max_iter,
            init=init, l1_ratio=l1_ratio, **factorizer_kwargs,
        )
    else:
        img_f, con_f, lams, _ = auto_nmf_pipeline(
            pos_joint, k_max=k_max,
            random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
            recon_threshold=recon_threshold,
            **factorizer_kwargs,
        )
    if verbose >= 2:
        print(f'{tag}   NMF pos  K={len(lams)}  {time.perf_counter() - _t0:.2f} s')

    if neg_joint.max() > 0:
        _t0_neg = time.perf_counter() if verbose >= 2 else None
        k_neg = k_fixed if k_fixed is not None else None
        if k_neg is not None:
            k_neg = max(1, min(int(k_neg), min(neg_joint.shape) - 1))
            neg_img_f, neg_con_f, neg_lams = full_nmf_pipeline(
                neg_joint, k_neg, random_state=random_state, max_iter=max_iter,
                init=init, l1_ratio=l1_ratio, **factorizer_kwargs,
            )
        else:
            neg_img_f, neg_con_f, neg_lams, _ = auto_nmf_pipeline(
                neg_joint, k_max=k_max,
                random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
                recon_threshold=recon_threshold,
                **factorizer_kwargs,
            )
        if verbose >= 2:
            print(f'{tag}   NMF neg  K={len(neg_lams)}  {time.perf_counter() - _t0_neg:.2f} s')
    else:
        neg_img_f = neg_con_f = neg_lams = None

    return img_f, con_f, lams, pos_joint, neg_img_f, neg_con_f, neg_lams


_EPS = 1e-12

_WEIGHTING_OPTIONS = ('img_selectivity', 'img_factor')


def _compute_trace_transition(weighting, img_f, lams, fi):
    """Compute (stimulus_weights, connection_weights) to pass into the next lower layer.

    Parameters
    ----------
    weighting : str — 'img_selectivity' (default) or 'img_factor'
    img_f     : (n_samples, K) NMF img_factors from layer L
    lams      : (K,) lambda values (descending) from layer L
    fi        : int — factor index to trace

    Returns
    -------
    sw : (n_samples,) stimulus weights for layer L-1's NMF
    cw : None (neither mode produces connection weights)
    """
    if weighting == 'img_selectivity':
        weighted = img_f * lams[np.newaxis, :]        # (n_samples, K)
        total    = weighted.sum(axis=1)               # (n_samples,)
        sw       = weighted[:, fi] / (total + _EPS)   # (n_samples,) ∈ [0, 1]
        return sw, None

    elif weighting == 'img_factor':
        return img_f[:, fi], None

    else:
        raise ValueError(
            f"Unknown weighting {weighting!r}. Choose one of: {_WEIGHTING_OPTIONS}"
        )


# ── Data collection ───────────────────────────────────────────────────────────

def _collect_layer_dicts(model, loader, device=None, only_correct=True,
                         layer_filter=None, label_transform=None):
    """Hook Conv2d/Linear sub-modules, run the loader, return layer data.

    Parameters
    ----------
    model        : nn.Module — any architecture
    loader       : DataLoader — yields (images, labels) batches
    device       : torch device; defaults to model's first parameter device
    only_correct : bool — when True, keeps only samples where argmax(output)==label
    layer_filter : callable(name: str, mod: nn.Module) -> bool, or None.
                   When provided, only modules for which the callable returns True
                   are captured. Useful for architectures with parallel branches
                   (e.g. SqueezeNet Fire modules) where capturing all Conv2d layers
                   would violate BFT's sequential-layer assumption.
                   When None, all Conv2d and Linear layers are captured (default).
    label_transform : callable(tensor) -> tensor, or None.
                   When provided, the loader's raw labels are mapped through it to
                   obtain the targets used for the correctness check and returned as
                   'targets'; the original raw labels are returned as 'digits'. Use
                   this when the model is trained on transformed labels (e.g. even/odd)
                   while the dataset yields raw labels, so the returned sample order
                   matches what BFT primary mode produces on the transformed loader.

    Returns
    -------
    dict with keys:
        'images'      : (N, C, H, W) float32 numpy array
        'targets'     : (N,) int numpy array of labels used for the correctness check
                        (transformed when label_transform is given, else raw)
        'digits'      : (N,) int numpy array of original raw labels
        'confidences' : (N,) float32 numpy array — max output probability per sample
        'layer_data'  : list of dicts, one per captured layer in forward order:
            {'name': str, 'type': 'conv'|'fc',
             'weight':      ndarray,
             'input_fmap':  ndarray,
             'output_fmap': ndarray}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    named = [(n, m) for n, m in model.named_modules()
             if isinstance(m, (nn.Conv2d, nn.Linear))
             and (layer_filter is None or layer_filter(n, m))]
    store = {n: {'inp': None, 'out': None} for n, _ in named}

    def make_hook(name):
        def h(mod, inp, out):
            store[name]['inp'] = inp[0].detach().cpu()
            store[name]['out'] = out.detach().cpu()
        return h

    hooks = [m.register_forward_hook(make_hook(n)) for n, m in named]
    acc_inp = {n: [] for n, _ in named}
    acc_out = {n: [] for n, _ in named}
    all_imgs, all_tgts, all_digits, all_confs = [], [], [], []

    with torch.no_grad():
        for x, y_raw in loader:
            x, y_raw = x.to(device), y_raw.to(device)
            y = label_transform(y_raw) if label_transform is not None else y_raw
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            # Support log-softmax output (negative values).
            probs = out.exp() if out.min() < 0 else out
            confs = probs.max(1).values
            preds = probs.argmax(1)
            ok = (preds == y).cpu().nonzero(as_tuple=True)[0] if only_correct \
                 else torch.arange(len(y))
            if not len(ok):
                continue
            all_imgs.append(x[ok].cpu())
            all_tgts.append(y[ok].cpu())
            all_digits.append(y_raw[ok].cpu())
            all_confs.append(confs[ok].cpu())
            for n, _ in named:
                acc_inp[n].append(store[n]['inp'][ok])
                acc_out[n].append(store[n]['out'][ok])

    for h in hooks:
        h.remove()

    imgs   = torch.cat(all_imgs).numpy()
    tgts   = torch.cat(all_tgts).numpy()
    digits = torch.cat(all_digits).numpy()
    confs  = torch.cat(all_confs).numpy()

    layer_data = []
    for n, mod in named:
        is_conv = isinstance(mod, nn.Conv2d)
        layer_data.append({
            'name':        n,
            'type':        'conv' if is_conv else 'fc',
            'weight':      mod.weight.detach().cpu().numpy(),
            'input_fmap':  torch.cat(acc_inp[n]).numpy(),
            'output_fmap': torch.cat(acc_out[n]).numpy(),
        })
    return {'images': imgs, 'targets': tgts, 'digits': digits,
            'confidences': confs, 'layer_data': layer_data}


def collect_layer_dicts(model, loader, device=None, only_correct=True,
                        layer_filter=None, label_transform=None):
    """Collect layer-dict data for Conv2d/Linear layers in the model.

    Thin public wrapper around the internal hook-based collection. Use this
    when you need to inspect or reuse the collected activations independently
    from running BFT. The returned 'layer_data' list is directly usable as
    the layer_dicts argument to bft().

    Parameters
    ----------
    model        : nn.Module
    loader       : DataLoader yielding (images, labels) batches
    device       : torch device; defaults to model's first parameter device
    only_correct : bool — keep only correctly classified samples (default True)
    layer_filter : callable(name: str, mod: nn.Module) -> bool, or None.
                   When provided, restricts capture to layers for which the
                   callable returns True. Useful for non-sequential architectures
                   (e.g. pass only the squeeze-conv spine of SqueezeNet to keep
                   BFT's sequential-layer assumption valid). Default None captures
                   all Conv2d and Linear layers.
    label_transform : callable(tensor) -> tensor, or None. When provided, labels are
                   mapped through it for the correctness check (returned as 'targets')
                   and the raw labels are returned as 'digits'. Pass the same transform
                   here that the model was trained on so the returned sample order
                   matches BFT primary mode run on label_transformed_loader(loader, ...).

    Returns
    -------
    dict: {'images', 'targets', 'digits', 'confidences', 'layer_data'}
    """
    return _collect_layer_dicts(model, loader, device=device,
                                only_correct=only_correct,
                                layer_filter=layer_filter,
                                label_transform=label_transform)


# ── BFT public entry point ────────────────────────────────────────────────────

def _dict_to_bft_node(d: dict) -> BFTNode:
    """Convert an internal node dict to a BFTNode dataclass."""
    return BFTNode(
        layer_idx=d['layer_idx'],
        layer_name=d.get('layer_name', ''),
        layer_type=d.get('layer_type', 'fc'),
        path=tuple(d.get('path', ())),
        weight=d['W'],
        img_factors=d['img_factors'],
        connection_factors=d['connection_factors'],
        lambdas=d['lambdas'],
        stimulus_weights=d.get('stimulus_weights_in', np.ones(d['img_factors'].shape[0])),
        neg_img_factors=d.get('neg_img_factors'),
        neg_connection_factors=d.get('neg_connection_factors'),
        neg_lambdas=d.get('neg_lambdas'),
        weighting=d.get('weighting', 'img_selectivity'),
        stimulus_threshold=d.get('stimulus_threshold', 0.0),
        attn_weights=d.get('attn_weights'),
        recon_validation=d.get('recon_validation'),
        children=[_dict_to_bft_node(c) for c in d.get('children', [])],
    )


def bft(model, data=None, *, k_max=5, n_branches=2, only_correct=True,
        device=None, stimulus_threshold=0.0, weighting='img_selectivity',
        random_state=0, max_iter=500, init=None, l1_ratio=0, k_fixed=None,
        conv_pool_method='avg', recon_threshold=None,
        validate=False, validate_top_m=20, validate_device=None,
        verbose=0, **factorizer_kwargs):
    """Backward Factor Trace (BFT).

    Traces the network's computation from the output layer to the input by
    factorising weight-activation products at each layer with NMF, then
    propagating the top factor's importance signal backward.

    Calling conventions
    -------------------
    Primary (recommended):
        result = bft(model, loader, ...)
        Automatically collects layer data via forward hooks, then runs the trace.
        result.images and result.targets expose the collected sample metadata.

    Layer-dict mode (advanced / pre-collected):
        result = bft(layer_dicts, ...)
        layer_dicts is a list of dicts, one per layer in forward order.
        See module docstring for the dict schema.

    Legacy model-protocol mode (SimpleMLP):
        result = bft(model, layer_inputs_list, ...)
        model must expose linear_layer_indices() and model.layers[i].weight.

    Parameters
    ----------
    model               : nn.Module (primary/legacy modes) or list of layer dicts
    data                : DataLoader (primary), list[ndarray] (legacy), or None (layer-dict)
    only_correct        : bool — keep only correctly classified samples (primary mode only)
    device              : torch device for data collection (primary mode only)
    k_max               : int, list[int or None], or None — upper bound on NMF rank per layer
    n_branches          : int or list[int] — how many top factors to follow at each layer
    stimulus_threshold  : fraction in [0, 1) — zeros the lowest-weight stimuli
    weighting           : 'img_selectivity' (default) or 'img_factor'
    random_state        : int — NMF random seed
    max_iter            : int — NMF iteration cap (default 500 for MiniBatchNMF)
    init                : str or None — NMF initialisation
    l1_ratio            : float — NMF L1/L2 regularisation mix (0 = L2, 1 = L1)
    k_fixed             : list[int or None] or None — per-layer fixed NMF rank
    conv_pool_method    : 'avg', 'max', or 'center' — spatial pooling for conv layers
    recon_threshold     : float or None — max acceptable relative Frobenius error
    validate            : bool — run causal reconstruction validation per factorization.
                          Each fc factorization's reconstructed activity replaces the
                          layer's real output; the rest of the model runs normally and
                          the cross-entropy loss ratio (recon / real) is recorded on the
                          node. Only available in primary mode (bft(model, loader)).
    validate_top_m      : int — images per factor for the per-factor fidelity check
    validate_device     : torch device for validation forward passes (default: model's)
    verbose             : 0 (silent), 1 (per-layer summary), 2 (detailed timing)

    Returns
    -------
    BFTResult — contains root BFTNode tree plus images, targets, confidences metadata.
                When validate=True, each fc node carries a recon_validation dict and
                BFTResult.validation_summary() aggregates them.
    """
    images_meta = targets_meta = confidences_meta = None

    if isinstance(model, list) and model and isinstance(model[0], dict):
        # Layer-dict mode (pre-collected dicts).
        layer_dicts   = model
        weights       = [d['weight']     for d in layer_dicts]
        inputs        = [d['input_fmap'] for d in layer_dicts]
        types         = [d.get('type', 'fc') for d in layer_dicts]
        names         = [d.get('name', str(i)) for i, d in enumerate(layer_dicts)]
        linear_idxs   = [None] * len(layer_dicts)
        attn_weights_list = [d.get('attn_weights') for d in layer_dicts]

    elif isinstance(data, DataLoader):
        # Primary mode: collect layer data, then trace.
        raw = _collect_layer_dicts(model, data, device=device, only_correct=only_correct)
        images_meta      = raw['images']
        targets_meta     = raw['targets']
        confidences_meta = raw['confidences']
        layer_dicts      = raw['layer_data']
        weights          = [d['weight']    for d in layer_dicts]
        inputs           = [d['input_fmap'] for d in layer_dicts]
        types            = [d['type']       for d in layer_dicts]
        names            = [d['name']       for d in layer_dicts]
        linear_idxs      = [None] * len(layer_dicts)
        attn_weights_list = [d.get('attn_weights') for d in layer_dicts]

    else:
        # Legacy mode: SimpleMLP with linear_layer_indices().
        layer_inputs_list = data
        linear_idxs       = model.linear_layer_indices()
        weights           = [model.layers[li].weight.detach().cpu().numpy() for li in linear_idxs]
        inputs            = layer_inputs_list
        types             = ['fc'] * len(linear_idxs)
        names             = [str(i) for i in range(len(linear_idxs))]
        attn_weights_list = [None] * len(linear_idxs)

    n_layers = len(weights)
    k_list  = list(k_max)      if isinstance(k_max,      (list, tuple)) else [k_max]      * n_layers
    nb_list = list(n_branches) if isinstance(n_branches, (list, tuple)) else [n_branches] * n_layers
    kf_list = list(k_fixed)    if isinstance(k_fixed,    (list, tuple)) else [k_fixed]    * n_layers
    assert len(k_list)  == n_layers, f"k_max list length {len(k_list)} != n_layers {n_layers}"
    assert len(nb_list) == n_layers, f"n_branches list length {len(nb_list)} != n_layers {n_layers}"
    assert len(kf_list) == n_layers, f"k_fixed list length {len(kf_list)} != n_layers {n_layers}"

    n_samples = inputs[0].shape[0]

    # Causal reconstruction validation is only possible in primary mode, where the
    # original model and input images are available to run forward passes.
    val_model = model if (validate and isinstance(data, DataLoader)) else None
    if val_model is not None:
        from .recon_validation import validate_node_reconstruction
        val_device = validate_device or next(val_model.parameters()).device
        _real_ce_cache = {}

    def _trace_node(l_idx, stimulus_weights, connection_weights, path):
        W         = weights[l_idx]
        act_input = inputs[l_idx]
        ltype     = types[l_idx]
        attn_w    = attn_weights_list[l_idx]

        layer_tag = f' L{l_idx + 1} {names[l_idx]!r} ({ltype})'
        if verbose >= 1:
            path_str = str(path) if path else '[]'
            print(f'[BFT] Layer {l_idx + 1}/{n_layers} {names[l_idx]!r} ({ltype})  '
                  f'path={path_str}')

        _t_layer = time.perf_counter()
        img_f, con_f, lams, joint, neg_img_f, neg_con_f, neg_lams = trace_single_layer(
            W, act_input, stimulus_weights,
            k_max=k_list[l_idx], k_fixed=kf_list[l_idx],
            stimulus_threshold=stimulus_threshold, connection_weights=connection_weights,
            random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
            layer_type=ltype, conv_pool_method=conv_pool_method,
            attn_weights=attn_w, recon_threshold=recon_threshold,
            verbose=verbose, _layer_tag=layer_tag,
            **factorizer_kwargs,
        )
        _t_nmf = time.perf_counter() - _t_layer
        if verbose >= 1:
            print(f'[BFT{layer_tag}]   K={len(lams)}  t={_t_nmf:.3f}s')
        if verbose >= 2:
            print(f'[BFT{layer_tag}]   total: {_t_nmf:.2f} s')

        if stimulus_weights.std() > 1e-8:
            n_active = int((stimulus_weights > 1e-9).sum())
        else:
            n_active = n_samples
        node = {
            'layer_idx': l_idx, 'layer_type': ltype, 'layer_name': names[l_idx],
            'factor_idx': path[-1] if path else 0,
            'active_samples': n_active,
            'W': W,
            'img_factors': img_f,
            'connection_factors': con_f,
            'lambdas': lams,
            'neg_img_factors': neg_img_f,
            'neg_connection_factors': neg_con_f,
            'neg_lambdas': neg_lams,
            'stimulus_weights_in': stimulus_weights.copy(),
            'weighting': weighting,
            'stimulus_threshold': stimulus_threshold,
            'path': path, 'children': [],
        }
        if ltype == 'attn' and attn_w is not None:
            node['attn_weights'] = attn_w

        if val_model is not None:
            node['act_input'] = act_input
            node['connection_weights'] = connection_weights
            rv = validate_node_reconstruction(
                val_model, images_meta, targets_meta, node, val_device,
                top_m=validate_top_m, real_ce_cache=_real_ce_cache,
            )
            node['recon_validation'] = rv
            # Drop the large activation reference so it is not retained on the node.
            del node['act_input'], node['connection_weights']
            if verbose >= 1 and rv is not None:
                print(f'[BFT{layer_tag}]   recon loss-ratio (all factors) = '
                      f'{rv["loss_ratio_all"]:.4f}  (n_active={rv["n_active"]})')

        if l_idx > 0:
            for fi in range(min(nb_list[l_idx], len(lams))):
                sw_fi, cw_fi = _compute_trace_transition(weighting, img_f, lams, fi)
                node['children'].append(_trace_node(l_idx - 1, sw_fi, cw_fi, path + [fi]))

        return node

    root_dict = _trace_node(n_layers - 1, np.ones(n_samples), None, [])

    root_node = _dict_to_bft_node(root_dict)

    if images_meta is None:
        # Use empty arrays for non-primary modes.
        n = n_samples
        images_meta      = np.zeros((n, 1), dtype=np.float32)
        targets_meta     = np.zeros(n, dtype=np.int64)
        confidences_meta = np.zeros(n, dtype=np.float32)

    return BFTResult(
        root=root_node,
        images=images_meta,
        targets=targets_meta,
        confidences=confidences_meta,
    )


def nodes_at_layer(root_node, target_layer_idx):
    """Return all BFT nodes whose layer_idx equals target_layer_idx.

    Parameters
    ----------
    root_node        : BFTNode or BFTResult
    target_layer_idx : int — 0 = input-side leaves, max = output root

    Returns
    -------
    list[BFTNode]  in BFS order
    """
    from collections import deque
    if isinstance(root_node, BFTResult):
        root_node = root_node.root
    result, queue = [], deque([root_node])
    while queue:
        n = queue.popleft()
        if n.layer_idx == target_layer_idx:
            result.append(n)
        queue.extend(n.children)
    return result
