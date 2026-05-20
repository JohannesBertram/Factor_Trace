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


def _select_k_single(lambdas, method, threshold, min_k):
    """Single-method K selection. lambdas must be a numpy array, sorted descending."""
    if method == 'cumvar':
        # Smallest K where the cumulative sum of lambdas covers >= threshold of the total
        total = lambdas.sum()
        if total == 0:
            return min_k
        cumvar = np.cumsum(lambdas) / total
        return int(np.searchsorted(cumvar, threshold)) + 1
    elif method == 'marginal':
        # Largest K where each lambda is at least `threshold` fraction of the first
        if lambdas[0] == 0:
            return min_k
        passing = np.where(lambdas / lambdas[0] >= threshold)[0]
        return int(passing[-1]) + 1 if len(passing) else min_k
    elif method == 'elbow':
        # K at the inflection point (maximum second-difference) of the lambda curve
        if len(lambdas) < 3:
            return len(lambdas)
        d2 = np.diff(np.diff(lambdas))
        return int(np.argmax(d2)) + 1
    elif method == 'fraction':
        # K at the first consecutive-ratio drop larger than 1.5x; +1e-12 avoids
        # division by zero when a lambda is exactly 0.
        for k in range(len(lambdas) - 1):
            if lambdas[k] / (lambdas[k + 1] + 1e-12) >= 1.5:
                return k
        return len(lambdas) - 1
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'cumvar', 'marginal', 'elbow', or 'fraction'."
        )


def select_k_from_lambdas(lambdas, method='cumvar', threshold=0.95, min_k=1,
                          min_cumvar=None):
    """Return the optimal number of NMF components based on factor informativity.

    Parameters
    ----------
    lambdas    : (K,) array — factor importances, assumed sorted descending
    method     : str or list[str]
                   'cumvar'     smallest K where Σλ[:K]/Σλ >= threshold
                   'marginal'   largest K where λ[k]/λ[0] >= threshold (drop-ratio)
                   'elbow'      K at the maximum second-difference of the λ curve
                   'fraction'   K at the first consecutive-ratio drop >= 1.5
                   'structural' alias for ['fraction', 'cumvar'] — fraction as primary
                                signal, cumvar floor at threshold (recommended default)
                   list         ['<structural>', 'cumvar'] — K* = max(k_structural,
                                k_cumvar), where both use the same threshold
    threshold  : float — cumvar fraction or drop-ratio threshold; meaning is
                 method-specific (see above)
    min_k      : int — hard lower bound on returned K
    min_cumvar : float or None — if set, K* is at least the K needed to reach
                 this cumvar fraction, regardless of method. Useful as a safety
                 floor when using a single structural method.

    Returns
    -------
    k_star : int in [min_k, len(lambdas)]
    """
    lambdas = np.asarray(lambdas, dtype=float)
    if len(lambdas) == 0:
        return min_k

    # Resolve alias
    if method == 'structural':
        method = ['fraction', 'cumvar']

    if isinstance(method, (list, tuple)):
        # List form: K* = max(k from structural method, k from cumvar floor)
        k_struct = _select_k_single(lambdas, method[0], threshold, min_k)
        k_floor  = _select_k_single(lambdas, 'cumvar',   threshold, min_k)
        k_star   = max(k_struct, k_floor)
    else:
        k_star = _select_k_single(lambdas, method, threshold, min_k)

    # Optional explicit cumvar floor: guarantee at least this much variance is covered
    if min_cumvar is not None:
        k_floor = _select_k_single(lambdas, 'cumvar', min_cumvar, min_k)
        k_star  = max(k_star, k_floor)

    return max(min_k, min(k_star, len(lambdas)))


# ── NMF pipeline ──────────────────────────────────────────────────────────────

def full_nmf_pipeline(X, n_components, random_state=0, max_iter=20000,
                      init=None, l1_ratio=0):
    """Fit NMF, normalise, sort by importance, and rescale by sqrt(lambda).

    The sqrt(lambda) rescaling distributes importance equally between the
    image factors (W) and neural factors (H) so that their dot product
    reconstructs X with unit-norm columns carrying equal weight.

    Returns (img_factors, neural_factors, lambdas).
      img_factors    : (n_samples, n_components)
      neural_factors : (n_neurons, n_components)
      lambdas        : (n_components,) descending
    """
    W, H, _ = run_nmf(X, n_components, random_state=random_state,
                      max_iter=max_iter, init=init, l1_ratio=l1_ratio)
    W, H, lambdas = normalize_factors(W, H)
    W, H, lambdas, _ = sort_by_lambda(W, H, lambdas)
    scale = np.sqrt(lambdas)
    return W * scale, H * scale, lambdas


def auto_nmf_pipeline(X, k_max=None, method='cumvar', threshold=0.95,
                      min_k=1, min_cumvar=None, random_state=0, max_iter=20000,
                      init=None, l1_ratio=0):
    """Fit NMF at rank k_max then automatically select effective rank K*.

    A single NMF fit is performed at k_max; components are pruned to K*
    based on sorted lambda values via select_k_from_lambdas. This avoids
    re-fitting for every candidate rank (NMF is not hierarchical like PCA,
    so K*-component results will differ from a fresh fit at K*, but the
    trade-off is acceptable for relative informativity selection).

    Parameters
    ----------
    X          : (n_samples, n_features) non-negative matrix
    k_max      : int or None — upper bound on rank; None → min(min(X.shape)-1, 20)
    method     : str or list — passed to select_k_from_lambdas; 'structural'
                 recommended (fraction as primary, cumvar floor at threshold)
    threshold  : float — passed to select_k_from_lambdas
    min_k      : int — minimum number of components to return
    min_cumvar : float or None — explicit cumvar floor; passed to select_k_from_lambdas
    random_state, max_iter, init, l1_ratio : passed to run_nmf

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
                                            l1_ratio=l1_ratio)
    k_star = select_k_from_lambdas(lams, method=method, threshold=threshold,
                                   min_k=min_k, min_cumvar=min_cumvar)
    return img_f[:, :k_star], neu_f[:, :k_star], lams[:k_star], k_star


def nmf_component_sweep(X, n_components_range, max_iter=20000):
    """Run NMF for each value of k in n_components_range and report fit quality.

    Explained variance is defined as 1 - ||X - WH||_F^2 / ||X||_F^2,
    analogous to R² for PCA.  Useful for choosing k_max before running BFT.

    Returns dict {k: explained_variance} for each k in n_components_range.
    """
    X_norm_sq = np.sum(X ** 2)
    results = {}
    for k in n_components_range:
        _, _, nmf_model = run_nmf(X, k, max_iter=max_iter)
        ev = 1.0 - nmf_model.reconstruction_err_ ** 2 / X_norm_sq
        results[k] = float(ev)
    return results


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
                        method='cumvar', threshold=0.95, min_k=1, min_cumvar=None,
                        stimulus_threshold=0.0, neuron_weights=None,
                        random_state=0, max_iter=20000, init=None, l1_ratio=0,
                        layer_type='fc', conv_pool_method='avg', k_fixed=None,
                        attn_weights=None, verbose=0, _layer_tag=''):
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
    method, threshold, min_k, min_cumvar : rank-selection criteria for auto_nmf_pipeline
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
        )
    else:
        # Factorise excitatory joint arbors: img_factors are per-stimulus loadings,
        # neural_factors are per-synapse pattern vectors, lambdas rank importance.
        img_f, neu_f, lams, _ = auto_nmf_pipeline(
            pos_joint, k_max=k_max, method=method, threshold=threshold,
            min_k=min_k, min_cumvar=min_cumvar,
            random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
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
            )
        else:
            neg_img_f, neg_neu_f, neg_lams, _ = auto_nmf_pipeline(
                neg_joint, k_max=k_max, method=method, threshold=threshold,
                min_k=min_k, min_cumvar=min_cumvar,
                random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
            )
        if verbose >= 2:
            print(f'{tag}   NMF neg  K={len(neg_lams)}  {time.perf_counter() - _t0_neg:.2f} s')
    else:
        neg_img_f = neg_neu_f = neg_lams = None

    return img_f, neu_f, lams, pos_joint, neg_img_f, neg_neu_f, neg_lams


_EPS = 1e-12

_WEIGHTING_OPTIONS = ('img_factor', 'factor_project_raw', 'factor_project_corrected',
                      'img_factor_neuron', 'cosine_activation', 'cosine_img_mix',
                      'img_selectivity')


def _compute_trace_transition(weighting, img_f, neu_f, W, act_into_current, fi,
                               layer_type='fc', attn_weights=None, factor_quantile=0.0,
                               cosine_mix=0.5, lams=None):
    """Compute (stimulus_weights, neuron_weights) to pass into the next lower layer.

    Called at the layer-L → layer-(L-1) transition.  Returns importance signals
    that guide the preceding layer's trace_single_layer call.

    For conv layers (layer_type='conv'), the 4-D input feature map is spatially
    average-pooled to a 2-D matrix before any projection so that all weighting
    modes produce a valid (N,) stimulus weight vector.  'img_factor' is the
    recommended default for conv-heavy models as it needs no projection.

    For attention layers (layer_type='attn'), the token tensor is collapsed to a
    single attention-weighted effective input before projections, mirroring the
    arbor-computation step.  All weighting modes then operate on this effective
    2-D matrix exactly as for FC layers.

    Parameters
    ----------
    weighting        : str — one of the _WEIGHTING_OPTIONS
    img_f            : (n_samples, K) NMF img_factors from layer L
    neu_f            : (n_out*n_in, K) NMF neural_factors from layer L
                       for conv: (C_out * C_in * kH * kW, K)
                       for attn: (d_v * d_model, K)
    W                : weight matrix of layer L
                       FC: (n_out, n_in); conv: (C_out, C_in, kH, kW);
                       attn: (d_v, d_model) value projection W_V
    act_into_current : activations flowing from layer L-1 into layer L
                       FC:   (n_samples, n_in)
                       conv: (n_samples, C_in, H, W)
                       attn: (n_samples, T, d_model) — all token activations
    fi               : int — factor index to trace
    layer_type       : 'fc' (default), 'conv', or 'attn'
    attn_weights     : (N, T) or None — required when layer_type=='attn'.
                       CLS-row attention scores (same array used in arbor construction).

    Returns
    -------
    sw : (n_samples,) stimulus weights for layer L-1's NMF
    nw : neuron weights for layer L-1's NMF, or None
         FC: shape (n_in,); conv: shape (C_in,) representing per-channel importance;
         attn: shape (d_model,) representing per-input-dimension importance
    """
    # Reduce any 3-D or 4-D input to a 2-D (N, features) matrix so that all
    # weighting modes below can apply uniform dot-product logic.
    if layer_type == 'conv':
        C_out, C_in, kH, kW = W.shape
        n_out_flat = C_out
        n_in_flat  = C_in * kH * kW
        # Global average pool: (N, C_in, H, W) → (N, C_in)
        act_2d = act_into_current.mean(axis=(2, 3))
    elif layer_type == 'attn':
        # Collapse the token sequence to the attention-weighted effective input.
        # This mirrors compute_attn_joint_arbors exactly: x_eff[n] = sum_j A[n,j]*x[n,j]
        # The result is (N, d_model), matching the FC input layout for W_V.
        if attn_weights is None:
            raise ValueError(
                "_compute_trace_transition: attn_weights required for layer_type='attn'."
            )
        # attn_weights: (N, T), act_into_current: (N, T, d_model) → x_eff: (N, d_model)
        act_2d = np.einsum('nt,ntd->nd', attn_weights, act_into_current)
        n_out_flat, n_in_flat = W.shape   # (d_v, d_model)
    else:
        n_out_flat, n_in_flat = W.shape
        act_2d = act_into_current

    if weighting == 'img_factor':
        # Use the factor's per-stimulus loading directly as stimulus weights.
        # Selects stimuli by how strongly they activate this factor; no neuron weights.
        # Works identically for FC and conv layers.
        return img_f[:, fi], None

    elif weighting == 'factor_project_raw':
        # Project input activations onto the arbor-space direction of the neural factor.
        # The dot product measures how well each stimulus aligns with the synaptic pattern;
        # returns the direction itself as neuron weights to guide the preceding layer.
        nw_full = neu_f[:, fi].reshape(n_out_flat, n_in_flat).sum(axis=0)  # (n_in_flat,)
        if layer_type == 'conv':
            # Reduce from patch space (C_in*kH*kW,) to channel space (C_in,)
            # by averaging over kernel positions so nw aligns with the pooled act_2d.
            nw = nw_full.reshape(C_in, kH * kW).mean(axis=1)               # (C_in,)
        else:
            nw = nw_full
        target = nw / (np.linalg.norm(nw) + _EPS)
        sw = np.clip(act_2d @ target, 0, None)
        return sw, nw

    elif weighting == 'factor_project_corrected':
        # Same projection as factor_project_raw, but first divide the neural factor
        # by the absolute weight values to convert from arbor space to activation space.
        # This corrects the bias that large weights would otherwise introduce.
        H_mat = neu_f[:, fi].reshape(n_out_flat, n_in_flat)                 # (n_out, n_in)
        if layer_type == 'conv':
            abs_W = np.abs(W).reshape(C_out, C_in * kH * kW)
            eps_w = max(1e-6 * float(abs_W.mean()), 1e-12)
            corrected_full = (H_mat / (abs_W + eps_w)).sum(axis=0)          # (C_in*kH*kW,)
            corrected = corrected_full.reshape(C_in, kH * kW).mean(axis=1)  # (C_in,)
        else:
            abs_W = np.abs(W)
            eps_w = max(1e-6 * float(abs_W.mean()), 1e-12)
            corrected = (H_mat / (abs_W + eps_w)).sum(axis=0)               # (n_in,)
        target = corrected / (np.linalg.norm(corrected) + _EPS)
        sw = np.clip(act_2d @ target, 0, None)
        return sw, corrected

    elif weighting == 'img_factor_neuron':
        # Use the factor's loading as stimulus weights; compute neuron weights as the
        # loading-weighted mean activation so both stimulus and neuron selection are
        # grounded in where the factor is actually active.
        sw = img_f[:, fi]                                                    # (n_samples,)
        # act_2d is (N, n_in) or (N, C_in): compute loading-weighted mean activation.
        nw = np.clip((sw @ act_2d) / (sw.sum() + _EPS), 0, None)
        return sw, nw

    elif weighting == 'cosine_activation':
        # Recover the prototypical L2-normalised activation in layer L's input space
        # by dividing each active factor entry H_mat[i,j] by the corresponding weight
        # W[i,j], then averaging over output neurons i for each input neuron j.
        #
        # "Active" entries: H_mat[i,j] > thresh, where thresh is the factor_quantile-th
        # percentile of all strictly-positive factor values (0.0 default → all nonzero).
        # Negative-weight positions are already zero in the positive-arbor NMF and are
        # therefore excluded automatically without any explicit sign filter.
        H_mat = neu_f[:, fi].reshape(n_out_flat, n_in_flat)                   # (n_out, n_in)

        pos_vals = H_mat[H_mat > 0]
        if len(pos_vals) == 0:
            return np.ones(act_2d.shape[0]), None

        thresh = 0.0 if factor_quantile == 0.0 else float(np.quantile(pos_vals, factor_quantile))
        active = H_mat > thresh                                                # (n_out, n_in) bool

        if layer_type == 'conv':
            W_2d = W.reshape(C_out, C_in * kH * kW)                           # (C_out, C_in*kH*kW)
        else:
            W_2d = W                                                           # (n_out, n_in)

        # Stabilised signed denominator: adds eps to |W| while preserving sign.
        # Exactly-zero weights produce a small positive denominator; their H_mat
        # entries are zero by construction so they don't influence proto.
        eps_w = max(1e-6 * float(np.abs(W_2d).mean()), 1e-12)
        W_denom = np.where(W_2d > 0, W_2d + eps_w,
                  np.where(W_2d < 0, W_2d - eps_w, eps_w))                   # (n_out, n_in)

        # For each input neuron j, average (H/W) over active output neurons.
        # Non-active positions are zero before summing; active_count prevents /0.
        H_active = np.where(active, H_mat, 0.0)                              # (n_out, n_in)
        recovered = H_active / W_denom                                        # (n_out, n_in)
        active_count = active.sum(axis=0).clip(min=1).astype(float)          # (n_in,)
        proto = recovered.sum(axis=0) / active_count                          # (n_in,)

        if layer_type == 'conv':
            # Collapse patch space (C_in*kH*kW) → channel space (C_in) to align
            # with the globally-pooled act_2d used by the conv weighting modes.
            proto = proto.reshape(C_in, kH * kW).mean(axis=1)               # (C_in,)

        # Guard: if the prototype is all-zero (dead factor), fall back to uniform weights.
        proto_norm_val = np.linalg.norm(proto)
        if proto_norm_val < _EPS:
            return np.ones(act_2d.shape[0]), None
        proto_unit = proto / proto_norm_val

        # Cosine similarity: L2-normalise each stimulus's actual activation, then dot
        # with the prototype unit vector. Values in [-1, 1]; clip to non-negative.
        act_row_norms = np.linalg.norm(act_2d, axis=1, keepdims=True)        # (N, 1)
        act_normed = act_2d / (act_row_norms + _EPS)                          # (N, n_in or C_in)
        cos_sims = act_normed @ proto_unit                                     # (N,) ∈ [-1, 1]
        return np.clip(cos_sims, 0, None), None

    elif weighting == 'img_selectivity':
        # For each stimulus, compute the fraction of its total lambda-weighted NMF
        # activation that belongs to factor fi.  Selects stimuli that strongly
        # activate fi but not other factors (high selectivity), not just stimuli
        # that activate fi in absolute terms.
        weighted = img_f * lams[np.newaxis, :]        # (n_samples, K)
        total    = weighted.sum(axis=1)               # (n_samples,)
        sw       = weighted[:, fi] / (total + _EPS)   # (n_samples,) ∈ [0, 1]
        return sw, None

    elif weighting == 'cosine_img_mix':
        # Weighted average of 'cosine_activation' and 'img_factor' stimulus weights.
        # Both signals are normalised to unit sum before blending so that cosine_mix
        # is a true interpolation coefficient: 0 → pure img_factor, 1 → pure cosine.
        # Uses the same factor_quantile threshold as 'cosine_activation'.

        # -- cosine_activation component --
        H_mat = neu_f[:, fi].reshape(n_out_flat, n_in_flat)
        pos_vals = H_mat[H_mat > 0]
        if len(pos_vals) > 0:
            thresh = 0.0 if factor_quantile == 0.0 else float(np.quantile(pos_vals, factor_quantile))
            active = H_mat > thresh
            if layer_type == 'conv':
                W_2d = W.reshape(C_out, C_in * kH * kW)
            else:
                W_2d = W
            eps_w = max(1e-6 * float(np.abs(W_2d).mean()), 1e-12)
            W_denom = np.where(W_2d > 0, W_2d + eps_w,
                      np.where(W_2d < 0, W_2d - eps_w, eps_w))
            H_active = np.where(active, H_mat, 0.0)
            recovered = H_active / W_denom
            active_count = active.sum(axis=0).clip(min=1).astype(float)
            proto = recovered.sum(axis=0) / active_count
            if layer_type == 'conv':
                proto = proto.reshape(C_in, kH * kW).mean(axis=1)
            proto_norm_val = np.linalg.norm(proto)
            if proto_norm_val >= _EPS:
                proto_unit = proto / proto_norm_val
                act_row_norms = np.linalg.norm(act_2d, axis=1, keepdims=True)
                act_normed = act_2d / (act_row_norms + _EPS)
                sw_cos = np.clip(act_normed @ proto_unit, 0, None)
            else:
                sw_cos = np.ones(act_2d.shape[0])
        else:
            sw_cos = np.ones(act_2d.shape[0])

        # -- img_factor component --
        sw_img = img_f[:, fi]

        # Normalise both to unit sum so cosine_mix interpolates on the same scale.
        sw_cos = sw_cos / (sw_cos.sum() + _EPS)
        sw_img = sw_img / (sw_img.sum() + _EPS)

        sw = cosine_mix * sw_cos + (1.0 - cosine_mix) * sw_img
        return sw, None

    else:
        raise ValueError(
            f"Unknown weighting {weighting!r}. Choose one of: {_WEIGHTING_OPTIONS}"
        )


# ── BFT public entry point ────────────────────────────────────────────────────

def bft(model, layer_inputs_list=None, k_max=5, n_branches=2, method='cumvar',
        threshold=0.95, min_cumvar=None, stimulus_threshold=0.0,
        weighting='img_factor', random_state=0, min_k=1, max_iter=20000,
        init=None, l1_ratio=0, k_fixed=None, cache_dir=None,
        conv_pool_method='avg', factor_quantile=0.0, cosine_mix=0.5,
        verbose=0):
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
    method              : str or list — rank-selection criterion for auto_nmf_pipeline
    threshold           : float — passed to auto_nmf_pipeline
    min_cumvar          : float or None — explicit cumvar floor
    stimulus_threshold  : fraction in [0, 1) — zeros the lowest-weight stimuli
    weighting           : str — how to compute stimulus/neuron weights at each
                          layer transition; one of:
                          'img_factor'             img_f[:,fi] as stimulus weights (default)
                          'factor_project_raw'     project onto arbor-space factor direction
                          'factor_project_corrected' same, corrected for weight magnitude
                          'img_factor_neuron'      img_f + loading-weighted mean activation
                          'cosine_activation'      cosine sim between L2-normed stimuli
                                                   and the weight-corrected prototype;
                                                   stimulus-only weighting (nw=None)
                          'img_selectivity'        fraction of each stimulus's lambda-
                                                   weighted total NMF activity that
                                                   belongs to factor fi; favours stimuli
                                                   selective for the traced factor
                          For conv layers 'img_factor' is always safe; the projection modes
                          use global average pooling of the input feature map.
    factor_quantile     : float in [0, 1) — only used when weighting='cosine_activation'.
                          Quantile threshold applied to strictly-positive neural factor
                          entries; only values above this percentile contribute to the
                          prototype activation estimate.
                          0.0 (default) uses all nonzero entries;
                          0.1 uses the top 90%; 0.9 uses the top 10%.
    cosine_mix          : float in [0, 1] — only used when weighting='cosine_img_mix'.
                          Blend coefficient: 0.0 → pure img_factor, 1.0 → pure
                          cosine_activation. Both signals are unit-sum normalised
                          before blending. Default 0.5 (equal weight).
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
            print(f'[BFT] Layer {l_idx + 1}/{n_layers} {names[l_idx]!r} ({ltype})  path={path_str}')

        # Run one BFT step: build the joint arbor matrix for this layer and
        # decompose it with NMF using the importance weights from the layer above.
        # For 'attn' layers, attn_w is forwarded so compute_attn_joint_arbors is used.
        _t_layer = time.perf_counter() if verbose >= 2 else None
        img_f, neu_f, lams, joint, neg_img_f, neg_neu_f, neg_lams = trace_single_layer(
            W, act_input, stimulus_weights,
            k_max=k_list[l_idx], k_fixed=kf_list[l_idx],
            method=method, threshold=threshold,
            min_k=min_k, min_cumvar=min_cumvar,
            stimulus_threshold=stimulus_threshold, neuron_weights=neuron_weights,
            random_state=random_state, max_iter=max_iter, init=init, l1_ratio=l1_ratio,
            layer_type=ltype, conv_pool_method=conv_pool_method,
            attn_weights=attn_w,
            verbose=verbose, _layer_tag=layer_tag,
        )
        if verbose >= 2:
            print(f'[BFT{layer_tag}]   total: {time.perf_counter() - _t_layer:.2f} s')
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
                sw_fi, nw_fi = _compute_trace_transition(
                    weighting, img_f, neu_f, W, act_input, fi=fi,
                    layer_type=ltype, attn_weights=attn_w,
                    factor_quantile=factor_quantile, cosine_mix=cosine_mix,
                    lams=lams)
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
