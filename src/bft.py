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
    bft(model_or_layers, ...)
        Runs BFT from the last layer to the first.  With n_branches=1 at
        every layer this yields a single linear trace; with n_branches>1 it
        produces a tree of parallel pathways.

        Two calling conventions are supported:

        1. Model-protocol mode (SimpleMLP-style):
               bft(model, layer_inputs_list, ...)
           model must expose linear_layer_indices() and model.layers[i].weight.

        2. Layer-dict mode (model-agnostic, works with CNNs, transformers, etc.):
               bft(layer_dicts, ...)
           layer_dicts is a list of dicts, one per layer in forward order:
               {'type': 'fc' | 'conv' | 'attn',
                'weight': ndarray,          # fc: (n_out,n_in); conv: (C_out,C_in,kH,kW);
                                            # attn: (d_v,d_model) value-projection W_V
                'input_fmap': ndarray,      # fc: (N,n_in); conv: (N,C_in,H,W);
                                            # attn: (N,T,d_model) all token activations
                'attn_weights': ndarray}    # attn only: (N,T) CLS-row scores (head-avg'd)

Model protocol (mode 1)
-----------------------
Any model object is accepted as long as it exposes:
    model.linear_layer_indices() -> list[int]
        Indices (into model.layers) of every nn.Linear sub-module, ordered
        first (nearest input) to last (nearest output).
    model.layers[i].weight
        The weight tensor of the i-th sub-module (PyTorch nn.Linear
        convention: shape (n_out, n_in), .detach().numpy() must work).
    All layers are assumed to be fully-connected (type 'fc').
"""

import os
import pickle
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import NMF
from .r1d import run_r1d


# ── NMF building blocks ───────────────────────────────────────────────────────

def run_nmf(X, n_components, random_state=0, max_iter=20000, init=None,
            l1_ratio=0, **kwargs):
    """Fit NMF and return (img_factors W, neural_factors H, fitted model).

    Parameters
    ----------
    X            : (n_samples, n_features) non-negative matrix
    n_components : int — number of components to fit
    random_state : int — for reproducibility
    max_iter     : int — sklearn NMF iteration cap
    init         : str or None — initialisation strategy; None auto-selects
                   'nndsvda' when n_components <= min(X.shape), else 'random'
    l1_ratio     : float in [0, 1] — L1 vs L2 regularisation mix for sklearn
                   NMF; 0 = pure L2 (Frobenius), 1 = pure L1
    **kwargs     : forwarded to sklearn.decomposition.NMF
    """
    # nndsvda initialisation requires n_components <= min(n_samples, n_features)
    if init is None:
        init = 'nndsvda' if n_components <= min(X.shape) else 'random'
    nmf = NMF(n_components=n_components, init=init,
              random_state=random_state, max_iter=max_iter,
              l1_ratio=l1_ratio, **kwargs)
    W = nmf.fit_transform(X)        # (n_samples, n_components)
    H = nmf.components_.T           # (n_neurons, n_components)
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
    H : (features, kmax) — sqrt(lambda)-scaled neural_factors, sorted descending
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

def full_nmf_pipeline(X, n_components, random_state=0, max_iter=20000,
                      init=None, l1_ratio=0, factorizer='sklearn',
                      **factorizer_kwargs):
    """Fit NMF, normalise, sort by importance, and rescale by sqrt(lambda).

    The sqrt(lambda) rescaling distributes importance equally between the
    image factors (W) and neural factors (H) so that their dot product
    reconstructs X with unit-norm columns carrying equal weight.

    Parameters
    ----------
    factorizer : str
        'sklearn' (default) — use sklearn NMF.
        'r1d'               — use greedy rank-1 deflation (see src/r1d.py).
    **factorizer_kwargs
        Passed to run_r1d when factorizer='r1d' (e.g. gamma, maxiters,
        penalize_lownorm, monotonic). Ignored for sklearn.

    Returns (img_factors, neural_factors, lambdas).
      img_factors    : (n_samples, n_components)
      neural_factors : (n_neurons, n_components)
      lambdas        : (n_components,) descending
    """
    if factorizer == 'r1d':
        W, H, _ = run_r1d(X, n_components, **factorizer_kwargs)
    else:
        W, H, _ = run_nmf(X, n_components, random_state=random_state,
                          max_iter=max_iter, init=init, l1_ratio=l1_ratio)
    W, H, lambdas = normalize_factors(W, H)
    W, H, lambdas, _ = sort_by_lambda(W, H, lambdas)
    scale = np.sqrt(lambdas)
    return W * scale, H * scale, lambdas


def auto_nmf_pipeline(X, k_max=None, random_state=0, max_iter=20000,
                      init=None, l1_ratio=0, recon_threshold=None,
                      factorizer='sklearn', **factorizer_kwargs):
    """Fit NMF at rank k_max then automatically select effective rank K*.

    Uses the structural_recon method: fraction drop as primary signal and actual
    Frobenius reconstruction error as a floor.  A single NMF fit is performed at
    k_max; components are pruned to K* so re-fitting at every candidate rank is
    avoided (trade-off acceptable for relative informativity selection).

    Parameters
    ----------
    X               : (n_samples, n_features) non-negative matrix
    k_max           : int or None — upper bound on rank; None → min(min(X.shape)-1, 20)
    recon_threshold : float or None — maximum acceptable relative Frobenius reconstruction
                      error. K* is at least the smallest K where
                      ||X - X_K||_F / ||X||_F <= recon_threshold.
                      Default None uses 0.2 (20% error).
    random_state, max_iter, init, l1_ratio : passed to run_nmf (sklearn path only)
    factorizer : str — 'sklearn' (default) or 'r1d'; see full_nmf_pipeline.
    **factorizer_kwargs : passed to run_r1d when factorizer='r1d'.

    Returns
    -------
    img_factors    : (n_samples, k_star)
    neural_factors : (n_features, k_star)
    lambdas        : (k_star,) descending
    k_star         : int — automatically selected rank
    """
    if k_max is None:
        k_max = min(min(X.shape) - 1, 20)
    k_max = max(int(k_max), 2)

    img_f, neu_f, lams = full_nmf_pipeline(X, k_max, random_state=random_state,
                                            max_iter=max_iter, init=init,
                                            l1_ratio=l1_ratio,
                                            factorizer=factorizer,
                                            **factorizer_kwargs)

    rt = recon_threshold if recon_threshold is not None else 0.2
    recon_errs = _partial_recon_errors(X, img_f, neu_f)
    k_frac  = _select_k_single(lams, min_k=1)
    k_recon = _select_k_from_recon(recon_errs, rt, min_k=1)
    k_star  = max(1, min(max(k_frac, k_recon), len(lams)))
    return img_f[:, :k_star], neu_f[:, :k_star], lams[:k_star], k_star


# ── BFT core ──────────────────────────────────────────────────────────────────

def compute_joint_arbors_normalized(W, act_input, stimulus_weights=None, eps=1e-8,
                                    stimulus_threshold=0.0, neuron_weights=None):
    """Compute the normalised, stimulus-weighted joint arbor matrix for an FC layer.

    A neuron's *arbor* for a given stimulus is the element-wise product of its
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
    neuron_weights      : (n_out,) per-output-neuron importance from the layer
                          above; each neuron i's arbor is scaled by this scalar.
                          None = all neurons equally weighted.

    Returns
    -------
    joint_arbor : (n_samples, n_out * n_in), signed — caller clips as needed
    """
    # L2-normalise each stimulus's input vector: removes per-stimulus energy
    # differences so NMF recovers pattern structure rather than activation magnitude.
    # Skip normalisation when stimulus_weights are uniform (no importance gradient).
    if stimulus_weights is not None and np.allclose(stimulus_weights, 1.0):
        act_norm = act_input
    else:
        norms = np.linalg.norm(act_input, axis=1, keepdims=True)
        act_norm = act_input / (norms + eps)

    # Build each neuron's arbor: W[i] * act_norm gives the synaptic contribution
    # of every input dimension to neuron i's pre-activation, scaled by neuron_weights
    # when an importance signal from the layer above is available.
    if neuron_weights is not None:
        arbors = [act_norm * W[i] * neuron_weights[i] for i in range(W.shape[0])]
    else:
        arbors = [act_norm * W[i] for i in range(W.shape[0])]

    # Stack all neurons' arbors side by side: produces (n_samples, n_out * n_in).
    # NMF on this joint matrix finds patterns that co-activate multiple neurons,
    # enabling cross-neuron decomposition rather than per-neuron analysis.
    joint = np.concatenate(arbors, axis=1)

    # Scale each stimulus row by its importance weight propagated from above,
    # so NMF concentrates on stimuli that are relevant to the traced factor.
    if stimulus_weights is not None:
        joint = joint * stimulus_weights[:, np.newaxis]

    # Zero out the lowest-weight stimuli entirely: removes low-relevance samples
    # that would otherwise dilute the factor structure.
    if stimulus_threshold > 0.0 and (stimulus_weights is not None) and not np.allclose(stimulus_weights, 1.0):
        cutoff = np.quantile(stimulus_weights, stimulus_threshold)
        joint[stimulus_weights <= cutoff] = 0.0

    return joint


def compute_conv_joint_arbors(weight, input_fmap, stimulus_weights=None, eps=1e-8,
                               stimulus_threshold=0.0, neuron_weights=None,
                               pool_method='avg'):
    """Compute the normalised, stimulus-weighted joint arbor matrix for a Conv2d layer.

    Analogous to compute_joint_arbors_normalized but handles spatial feature maps.
    The spatial dimension is collapsed via pooling before forming the arbor, so that
    a single (N, C_out * C_in * kH * kW) matrix represents the layer's computation
    in the same format as an FC joint arbor.

    Parameters
    ----------
    weight          : (C_out, C_in, kH, kW) conv weight tensor as numpy array
    input_fmap      : (N, C_in, H, W) input feature map as numpy array
    stimulus_weights : (N,) per-sample importance; uniform if None
    eps             : stabiliser for L2 norm division (internal only)
    stimulus_threshold : fraction in [0, 1) — zeros the lowest-weight stimuli
    neuron_weights  : (C_out,) per-output-channel importance from the layer above;
                      each output channel c's arbor block is scaled by neuron_weights[c].
                      None = all channels equally weighted.
    pool_method     : str — how to reduce the spatial dimension of extracted patches:
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

    # Extract local patches at every spatial position via im2col.
    # Padding=(kH//2, kW//2) keeps the output spatial size equal to the input.
    pad = (kH // 2, kW // 2)
    fmap_t = torch.from_numpy(input_fmap).float()
    # patches shape: (N, C_in*kH*kW, H*W_in)
    patches = F.unfold(fmap_t, kernel_size=(kH, kW), padding=pad).numpy()

    # Reduce the spatial dimension so each sample is a single vector (N, C_in*kH*kW).
    # The chosen pooling captures the most representative local patch per stimulus.
    if pool_method == 'avg':
        pooled = patches.mean(axis=2)                          # (N, C_in*kH*kW)
    elif pool_method == 'max':
        pooled = patches.max(axis=2)                           # (N, C_in*kH*kW)
    elif pool_method == 'center':
        ctr_idx = (H // 2) * W_in + (W_in // 2)               # flat index of center position
        pooled = patches[:, :, ctr_idx]                        # (N, C_in*kH*kW)
    else:
        raise ValueError(f"Unknown pool_method {pool_method!r}. Use 'avg', 'max', or 'center'.")

    # L2-normalise each stimulus's pooled patch vector: removes per-stimulus energy
    # differences so NMF recovers spatial pattern structure, not activation magnitude.
    # Skip when stimulus_weights are uniform (no importance gradient present).
    if stimulus_weights is not None and np.allclose(stimulus_weights, 1.0):
        act_norm = pooled
    else:
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        act_norm = pooled / (norms + eps)

    # Build each output channel's arbor: W_flat[c] * act_norm gives the contribution
    # of each input patch element to channel c's pre-activation; scale by neuron_weights[c]
    # when an importance signal from the layer above is available.
    if neuron_weights is not None:
        arbors = [act_norm * W_flat[c] * neuron_weights[c] for c in range(C_out)]
    else:
        arbors = [act_norm * W_flat[c] for c in range(C_out)]

    # Stack all channels' arbors to form (N, C_out * C_in * kH * kW):
    # NMF on this joint matrix discovers cross-channel patterns in the conv layer.
    joint = np.concatenate(arbors, axis=1)

    # Apply stimulus weighting and threshold (identical logic to FC joint arbors).
    if stimulus_weights is not None:
        joint = joint * stimulus_weights[:, np.newaxis]

    if stimulus_threshold > 0.0 and (stimulus_weights is not None) and not np.allclose(stimulus_weights, 1.0):
        cutoff = np.quantile(stimulus_weights, stimulus_threshold)
        joint[stimulus_weights <= cutoff] = 0.0

    return joint


def compute_attn_joint_arbors(W_V, x_tokens, attn_weights_cls, stimulus_weights=None,
                               eps=1e-8, stimulus_threshold=0.0, neuron_weights=None):
    """Compute the normalised, stimulus-weighted joint arbor matrix for an attention layer.

    Attention mixes token representations using data-dependent weights (the softmax scores),
    making it impossible to form a single fixed weight matrix as in FC or conv layers.  The
    solution used here is to collapse the token sequence into a single *attention-weighted
    effective input* per sample:

        x_eff[n] = sum_j  attn_weights_cls[n, j] * x_tokens[n, j]   # (N, d_model)

    Because attn_weights_cls are non-negative (softmax output), x_eff is a convex combination
    of the token activations.  We then form the joint arbor exactly as for an FC layer with
    W_V as the weight matrix and x_eff as the input.  This means each column of the joint
    arbor represents the attention-reweighted synaptic contribution of each input dimension
    to one output dimension of the value projection.

    The approximation discards per-token spatial resolution (all tokens are blended into one
    effective vector).  Use the ``attn_weights_cls`` stored in the returned BFT node to
    recover per-token attribution in downstream analysis.

    Parameters
    ----------
    W_V               : (d_v, d_model) — value-projection weight matrix.
                        For multi-head attention, pass the full concatenated W_V
                        (H * d_head_v, d_model).
    x_tokens          : (N, T, d_model) — all T token activations entering the attention
                        sublayer (after layer-norm), for each of the N samples.
    attn_weights_cls  : (N, T) — attention scores from the CLS token to every token,
                        already softmax-normalised (non-negative, sum≈1 per row).
                        For multi-head, pass the head-averaged CLS row.
    stimulus_weights  : (N,) per-sample importance propagated from the layer above; None
                        means all samples are equally weighted.
    eps               : stabiliser for L2 norm division (internal only)
    stimulus_threshold : fraction in [0, 1) — zeros the lowest-weight stimuli after
                        applying stimulus_weights (matches behaviour of FC/conv variants).
    neuron_weights    : (d_v,) per-output-dimension importance from the layer above
                        (e.g., from W_O tracing); None = all dimensions equally weighted.

    Returns
    -------
    joint_arbor : (N, d_v * d_model), signed — caller clips as needed.
                  Same layout as compute_joint_arbors_normalized: each block of d_model
                  columns corresponds to one output dimension of W_V.
    """
    # Collapse the token sequence into a single attention-weighted effective input.
    # attn_weights_cls has shape (N, T); x_tokens has shape (N, T, d_model).
    # The einsum computes, for each sample n: sum_j attn[n,j] * x_tokens[n,j,:]
    x_eff = np.einsum('nt,ntd->nd', attn_weights_cls, x_tokens)  # (N, d_model)

    # Delegate to the FC arbor function: x_eff acts as a 2-D activation matrix.
    # All stimulus weighting, normalisation, and threshold logic is handled there.
    return compute_joint_arbors_normalized(W_V, x_eff, stimulus_weights,
                                           eps=eps,
                                           stimulus_threshold=stimulus_threshold,
                                           neuron_weights=neuron_weights)


def trace_single_layer(W, act_input, stimulus_weights, k_max=None,
                        stimulus_threshold=0.0, neuron_weights=None,
                        random_state=0, max_iter=20000, init=None, l1_ratio=0,
                        layer_type='fc', conv_pool_method='avg', k_fixed=None,
                        attn_weights=None, recon_threshold=None,
                        factorizer='sklearn', verbose=0, _layer_tag='',
                        **factorizer_kwargs):
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
    neuron_weights      : per-neuron (FC) or per-channel (conv) or per-dim (attn) importance
    random_state, max_iter, init, l1_ratio : passed through to run_nmf
    layer_type          : 'fc' (default), 'conv', or 'attn' — selects the arbor function
    conv_pool_method    : spatial pooling for conv arbors: 'avg', 'max', 'center'
    k_fixed             : int or None — when set, use exactly this many components
                          (calls full_nmf_pipeline directly at k_fixed, bypasses auto)
    attn_weights        : (N, T) or None — required when layer_type=='attn'.
                          CLS-row attention scores (head-averaged, softmax-normalised).

    Returns
    -------
    img_factors        : (n_samples, K*)
    neural_factors     : (n_out * n_in, K*) — for conv: (C_out * C_in * kH * kW, K*);
                         for attn: (d_v * d_model, K*)
    lambdas            : (K*,) descending
    joint_arbor        : positive joint matrix (pre-NMF)
    neg_img_factors    : (n_samples, K*) or None
    neg_neural_factors : same shape as neural_factors, or None
    neg_lambdas        : (K*,) or None
    """
    # Compute the joint arbor matrix using the layer-type-appropriate function.
    if layer_type == 'conv':
        raw_joint = compute_conv_joint_arbors(W, act_input, stimulus_weights,
                                              stimulus_threshold=stimulus_threshold,
                                              neuron_weights=neuron_weights,
                                              pool_method=conv_pool_method)
    elif layer_type == 'attn':
        # Attention layers require the full token tensor and the CLS attention scores.
        # attn_weights must be provided as a (N, T) array when layer_type=='attn'.
        if attn_weights is None:
            raise ValueError(
                "layer_type='attn' requires attn_weights (N, T) CLS-row scores. "
                "Add 'attn_weights' to the layer dict."
            )
        raw_joint = compute_attn_joint_arbors(W, act_input, attn_weights,
                                              stimulus_weights=stimulus_weights,
                                              stimulus_threshold=stimulus_threshold,
                                              neuron_weights=neuron_weights)
    else:
        raw_joint = compute_joint_arbors_normalized(W, act_input, stimulus_weights,
                                                    stimulus_threshold=stimulus_threshold,
                                                    neuron_weights=neuron_weights)

    # Split signed arbors into excitatory and inhibitory parts: NMF requires
    # non-negative inputs, so the two polarities are factorised separately to
    # preserve both directions without sign ambiguity.
    pos_joint = np.clip(raw_joint, 0, None)
    neg_joint = np.clip(-raw_joint, 0, None)

    if verbose >= 2:
        tag = f'[BFT{_layer_tag}]'
        print(f'{tag}   joint arbor: {raw_joint.shape}  '
              f'pos {pos_joint.shape}  neg {neg_joint.shape}')

    # Factorise the excitatory joint arbors. When k_fixed is set, fit at exactly
    # that rank instead of using auto-selection (useful for strict reproducibility
    # or controlled ablation experiments).
    _t0 = time.perf_counter() if verbose >= 2 else None
    if k_fixed is not None:
        k = max(int(k_fixed), 1)
        k = min(k, min(pos_joint.shape) - 1)
        img_f, neu_f, lams = full_nmf_pipeline(
            pos_joint, k, random_state=random_state, max_iter=max_iter,
            init=init, l1_ratio=l1_ratio,
            factorizer=factorizer, **factorizer_kwargs,
        )
    else:
        # Factorise excitatory joint arbors: img_factors are per-stimulus loadings,
        # neural_factors are per-synapse pattern vectors, lambdas rank importance.
        img_f, neu_f, lams, _ = auto_nmf_pipeline(
            pos_joint, k_max=k_max,
            random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
            recon_threshold=recon_threshold,
            factorizer=factorizer, **factorizer_kwargs,
        )
    if verbose >= 2:
        print(f'{tag}   NMF pos  K={len(lams)}  {time.perf_counter() - _t0:.2f} s')

    # Factorise inhibitory joint arbors only when inhibitory content exists;
    # returns None triplet when the layer has no inhibitory weight-activation products.
    if neg_joint.max() > 0:
        _t0_neg = time.perf_counter() if verbose >= 2 else None
        k_neg = k_fixed if k_fixed is not None else None
        if k_neg is not None:
            k_neg = max(1, min(int(k_neg), min(neg_joint.shape) - 1))
            neg_img_f, neg_neu_f, neg_lams = full_nmf_pipeline(
                neg_joint, k_neg, random_state=random_state, max_iter=max_iter,
                init=init, l1_ratio=l1_ratio,
                factorizer=factorizer, **factorizer_kwargs,
            )
        else:
            neg_img_f, neg_neu_f, neg_lams, _ = auto_nmf_pipeline(
                neg_joint, k_max=k_max,
                random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
                recon_threshold=recon_threshold,
                factorizer=factorizer, **factorizer_kwargs,
            )
        if verbose >= 2:
            print(f'{tag}   NMF neg  K={len(neg_lams)}  {time.perf_counter() - _t0_neg:.2f} s')
    else:
        neg_img_f = neg_neu_f = neg_lams = None

    return img_f, neu_f, lams, pos_joint, neg_img_f, neg_neu_f, neg_lams


_EPS = 1e-12

_WEIGHTING_OPTIONS = ('img_selectivity', 'img_factor')


def _compute_trace_transition(weighting, img_f, lams, fi):
    """Compute (stimulus_weights, neuron_weights) to pass into the next lower layer.

    Called at the layer-L → layer-(L-1) transition.  Returns importance signals
    that guide the preceding layer's trace_single_layer call.

    Parameters
    ----------
    weighting : str — 'img_selectivity' (default) or 'img_factor'
    img_f     : (n_samples, K) NMF img_factors from layer L
    lams      : (K,) lambda values (descending) from layer L
    fi        : int — factor index to trace

    Returns
    -------
    sw : (n_samples,) stimulus weights for layer L-1's NMF
    nw : None (neither mode produces neuron weights)
    """
    if weighting == 'img_selectivity':
        # Fraction of each stimulus's lambda-weighted total NMF activation that
        # belongs to factor fi.  Favours stimuli selective for the traced factor,
        # not just stimuli that activate fi in absolute terms.
        weighted = img_f * lams[np.newaxis, :]        # (n_samples, K)
        total    = weighted.sum(axis=1)               # (n_samples,)
        sw       = weighted[:, fi] / (total + _EPS)   # (n_samples,) ∈ [0, 1]
        return sw, None

    elif weighting == 'img_factor':
        # Factor's per-stimulus loading directly as stimulus weights.
        return img_f[:, fi], None

    else:
        raise ValueError(
            f"Unknown weighting {weighting!r}. Choose one of: {_WEIGHTING_OPTIONS}"
        )


# ── BFT public entry point ────────────────────────────────────────────────────

def bft(model, layer_inputs_list=None, k_max=5, n_branches=2,
        stimulus_threshold=0.0, weighting='img_selectivity', random_state=0,
        min_k=1, max_iter=20000, init=None, l1_ratio=0, k_fixed=None,
        cache_dir=None, conv_pool_method='avg', recon_threshold=None,
        factorizer='sklearn', verbose=0, **factorizer_kwargs):
    """Backward Factor Trace (BFT).

    Traces the network's computation from the output layer to the input by
    factorising weight-activation products at each layer with NMF, then
    propagating the top factor's importance signal backward.

    With n_branches=1 at every layer this produces a single linear trace
    (one pathway from output to input).  With n_branches>1 it produces a
    tree, branching over the top n_branches NMF factors at each layer to
    discover parallel computational pathways.

    Two calling conventions are supported — see module docstring for details.

    Parameters
    ----------
    model               : model object (model-protocol mode) or list of layer
                          dicts (layer-dict mode — model-agnostic).
                          See module docstring for the layer-dict schema.
    layer_inputs_list   : list[ndarray] per linear layer, first-to-last.
                          Required in model-protocol mode; ignored in layer-dict mode.
    k_max               : int, list[int or None], or None — upper bound on NMF
                          rank per layer (L1 first). A single int/None applies
                          to all layers. Ignored for layers where k_fixed is set.
    n_branches          : int or list[int] — how many top factors to follow at
                          each layer (first-to-last). A single int applies to all.
    stimulus_threshold  : fraction in [0, 1) — zeros the lowest-weight stimuli
    weighting           : str — how to propagate importance between layers; one of:
                          'img_selectivity' (default) — fraction of each stimulus's
                                            lambda-weighted NMF activity belonging to
                                            factor fi; favours stimuli selective for the
                                            traced factor
                          'img_factor'      img_f[:,fi] directly as stimulus weights
    random_state        : int — NMF random seed
    min_k               : int — minimum NMF components per layer
    max_iter            : int — NMF iteration cap
    init                : str or None — NMF initialisation
    l1_ratio            : float — NMF L1/L2 mix (0 = Frobenius, 1 = pure L1)
    k_fixed             : list[int or None] or None — per-layer fixed NMF rank.
                          When an entry is not None, that layer uses exactly that K
                          (calls full_nmf_pipeline directly, bypasses auto_nmf_pipeline).
                          Length must equal n_layers when provided as a list; None
                          means auto for all layers.
    cache_dir           : str or None — directory for per-path on-disk caching.
                          Each completed leaf path is pickled as
                          `<cache_dir>/bft_path_<seq>.pkl` (e.g. 'bft_path_0-1-2.pkl').
                          On re-run, existing cached paths are loaded from disk to
                          avoid redundant NMF computation (useful for large models).
                          None disables caching.
    conv_pool_method    : str — spatial pooling for conv layers: 'avg', 'max', 'center'
    recon_threshold     : float or None — maximum acceptable relative Frobenius
                          reconstruction error floor (structural_recon method).
                          Default None uses 0.2 (20% error).
    verbose             : int — verbosity level (default 0 = silent).
                          1 — print one line per layer/branch showing layer index,
                              name, type, and current branch path.
                          2 — additionally print joint arbor shapes, per-NMF timing,
                              and total trace_single_layer time for each step.

    Returns
    -------
    root : dict — last-layer node, with keys:
        layer_idx           : 0-based index (0 = first/input-side layer)
        layer_type          : 'fc' or 'conv'
        W                   : weight matrix (numpy)
        joint_arbor         : positive joint matrix used for NMF
        img_factors         : (n_samples, K*)
        neural_factors      : (n_out*n_in, K*) or (C_out*C_in*kH*kW, K*)
        lambdas             : (K*,) descending
        neg_img_factors     : (n_samples, K*) or None
        neg_neural_factors  : same shape or None
        neg_lambdas         : (K*,) or None
        stimulus_weights_in : (n_samples,) weights passed into this layer
        neuron_weights_in   : per-neuron/channel importance passed in, or None
        path                : list[int] — factor indices from root to this node
        children            : list[dict] — preceding-layer nodes (empty at L1)
        [model-protocol only]
        linear_idx          : index in model.layers Sequential
    """
    # ── Resolve inputs from the two calling conventions ───────────────────────
    if isinstance(model, list):
        # Layer-dict mode: extract weights, activations, and types from the dicts.
        layer_dicts   = model
        weights       = [d['weight']     for d in layer_dicts]
        inputs        = [d['input_fmap'] for d in layer_dicts]
        types         = [d.get('type', 'fc') for d in layer_dicts]
        names         = [d.get('name', str(i)) for i, d in enumerate(layer_dicts)]
        linear_idxs   = [None] * len(layer_dicts)
        # attn_weights_list holds the (N, T) CLS attention scores for each layer that
        # has type=='attn'; None for all other layer types.  Kept as a parallel list so
        # _trace_node can look up the scores by layer index without touching layer_dicts.
        attn_weights_list = [d.get('attn_weights') for d in layer_dicts]
    else:
        # Model-protocol mode: all layers assumed FC; no attention layers.
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

    def _cache_path(path_seq):
        seq_str = '-'.join(str(b) for b in path_seq) if path_seq else 'root'
        return os.path.join(cache_dir, f'bft_path_{seq_str}.pkl')

    def _trace_node(l_idx, stimulus_weights, neuron_weights, path):
        W         = weights[l_idx]
        act_input = inputs[l_idx]
        ltype     = types[l_idx]
        li        = linear_idxs[l_idx]
        # Retrieve the CLS-row attention scores for this layer (None for non-attn layers).
        attn_w    = attn_weights_list[l_idx]

        # Load from cache if this leaf path was previously computed.
        # Cache is keyed by the full factor-index path from root to this node.
        if cache_dir is not None and l_idx == 0:
            cp = _cache_path(path)
            if os.path.exists(cp):
                with open(cp, 'rb') as fh:
                    return pickle.load(fh)

        layer_tag = f' L{l_idx + 1} {names[l_idx]!r} ({ltype})'
        if verbose >= 1:
            path_str = str(path) if path else '[]'
            print(f'[BFT] Layer {l_idx + 1}/{n_layers} {names[l_idx]!r} ({ltype})  '
                  f'path={path_str}  factorizer={factorizer}')

        # Run one BFT step: build the joint arbor matrix for this layer and
        # decompose it with NMF using the importance weights from the layer above.
        # For 'attn' layers, attn_w is forwarded so compute_attn_joint_arbors is used.
        _t_layer = time.perf_counter()
        img_f, neu_f, lams, joint, neg_img_f, neg_neu_f, neg_lams = trace_single_layer(
            W, act_input, stimulus_weights,
            k_max=k_list[l_idx], k_fixed=kf_list[l_idx],
            stimulus_threshold=stimulus_threshold, neuron_weights=neuron_weights,
            random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
            layer_type=ltype, conv_pool_method=conv_pool_method,
            attn_weights=attn_w, recon_threshold=recon_threshold,
            factorizer=factorizer, verbose=verbose, _layer_tag=layer_tag,
            **factorizer_kwargs,
        )
        _t_nmf = time.perf_counter() - _t_layer
        if verbose >= 1:
            print(f'[BFT{layer_tag}]   K={len(lams)}  t={_t_nmf:.3f}s')
        if verbose >= 2:
            print(f'[BFT{layer_tag}]   total: {_t_nmf:.2f} s')
        # active_samples: how many stimuli have non-negligible weight.
        # When weights are uniform (std ≈ 0) every sample is active.
        if stimulus_weights.std() > 1e-8:
            n_active = int((stimulus_weights > 1e-9).sum())
        else:
            n_active = n_samples
        node = {
            'layer_idx': l_idx, 'layer_type': ltype, 'layer_name': names[l_idx],
            'factor_idx': path[-1] if path else 0,
            'active_samples': n_active,
            'W': W,
            'joint_arbor': joint, 'img_factors': img_f,
            'neural_factors': neu_f, 'lambdas': lams,
            'neg_img_factors': neg_img_f,
            'neg_neural_factors': neg_neu_f,
            'neg_lambdas': neg_lams,
            'stimulus_weights_in': stimulus_weights.copy(),
            'neuron_weights_in': neuron_weights.copy() if neuron_weights is not None else None,
            'weighting': weighting,
            'stimulus_threshold': stimulus_threshold,
            'path': path, 'children': [],
        }
        # For attention layers, store the CLS attention scores in the node so downstream
        # analysis can recover per-token attribution (e.g., spatial grounding on patches).
        if ltype == 'attn' and attn_w is not None:
            node['attn_weights'] = attn_w
        if li is not None:
            node['linear_idx'] = li

        if l_idx > 0:
            # Branch over the top n_branches factors: each factor seeds an
            # independent trace into the preceding layer, producing parallel
            # pathways. With nb_list[l_idx]=1 this reduces to a single chain.
            for fi in range(min(nb_list[l_idx], len(lams))):
                # Compute the importance signal to pass to the preceding layer
                # based on factor fi's loadings and the chosen weighting mode.
                # For 'attn' layers, attn_w is passed so the effective input can be
                # reconstructed when computing projections in the transition.
                sw_fi, nw_fi = _compute_trace_transition(weighting, img_f, lams, fi)
                node['children'].append(_trace_node(l_idx - 1, sw_fi, nw_fi, path + [fi]))

        # Persist completed leaf-path nodes to disk so reruns can skip NMF.
        if cache_dir is not None and l_idx == 0:
            os.makedirs(cache_dir, exist_ok=True)
            with open(_cache_path(path), 'wb') as fh:
                pickle.dump(node, fh)

        return node

    # Start from the last (output) layer with uniform stimulus weights: every
    # sample is equally relevant before any factorisation; structure emerges
    # from the decomposition itself, not from a prior importance signal.
    return _trace_node(n_layers - 1, np.ones(n_samples), None, [])
