import numpy as np
from sklearn.decomposition import NMF


def run_nmf(X, n_components, random_state=0, max_iter=20000, init=None, **kwargs):
    """Fit NMF and return (img_factors W, neural_factors H, fitted model)."""
    # nndsvda requires n_components <= min(n_samples, n_features); fall back to random otherwise

    if init is None:
        init = 'nndsvda' if n_components <= min(X.shape) else 'random'
    nmf = NMF(n_components=n_components, init=init,
              random_state=random_state, max_iter=max_iter, **kwargs)#, l1_ratio=)
    W = nmf.fit_transform(X)        # (n_samples, n_components)
    H = nmf.components_.T           # (n_neurons, n_components)
    return W, H, nmf


def normalize_factors(W, H):
    """
    Rescale W and H to unit column-norm, returning the absorbed norms as lambdas.

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
        total = lambdas.sum()
        if total == 0:
            return min_k
        cumvar = np.cumsum(lambdas) / total
        return int(np.searchsorted(cumvar, threshold)) + 1
    elif method == 'marginal':
        if lambdas[0] == 0:
            return min_k
        passing = np.where(lambdas / lambdas[0] >= threshold)[0]
        return int(passing[-1]) + 1 if len(passing) else min_k
    elif method == 'elbow':
        if len(lambdas) < 3:
            return len(lambdas)
        d2 = np.diff(np.diff(lambdas))
        return int(np.argmax(d2)) + 1
    elif method == 'fraction':
        for k in range(len(lambdas) - 1):
            if lambdas[k] / lambdas[k + 1] >= 1.5:
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
        # List form: [structural_method, 'cumvar'] → K* = max(k_struct, k_floor)
        k_struct = _select_k_single(lambdas, method[0], threshold, min_k)
        k_floor  = _select_k_single(lambdas, 'cumvar',   threshold, min_k)
        k_star   = max(k_struct, k_floor)
    else:
        k_star = _select_k_single(lambdas, method, threshold, min_k)

    # Optional explicit cumvar floor
    if min_cumvar is not None:
        k_floor = _select_k_single(lambdas, 'cumvar', min_cumvar, min_k)
        k_star  = max(k_star, k_floor)

    return max(min_k, min(k_star, len(lambdas)))


def reconstruct_from_components(img_factors, neural_factors, lambdas, keep_k):
    """Reconstruct X from the top keep_k NMF components.

    Since full_nmf_pipeline scales factors by sqrt(lambda), the reconstruction
    is simply img_factors[:, :keep_k] @ neural_factors[:, :keep_k].T.

    Returns (n_samples, n_neurons) array.
    """
    return img_factors[:, :keep_k] @ neural_factors[:, :keep_k].T


def nmf_component_sweep(X, n_components_range, max_iter=20000):
    """
    Run NMF for each value of k in n_components_range.

    Explained variance is defined as 1 - ||X - WH||_F^2 / ||X||_F^2,
    analogous to R² for PCA.

    Returns dict {k: ev} for each k in n_components_range.
    """
    X_norm_sq = np.sum(X ** 2)
    results = {}
    for k in n_components_range:
        _, _, nmf_model = run_nmf(X, k, max_iter=max_iter)
        ev = 1.0 - nmf_model.reconstruction_err_ ** 2 / X_norm_sq
        results[k] = float(ev)
    return results


def compute_stimulus_loading(mask, img_f_by_lii, lams_by_lii,
                             all_acts, act_offsets, layer1_end):
    """
    Compute stimulus-conditioned neuron loadings across all layers.

    For layers without per-neuron NMF (layer 1): loading = mean activation
    over selected stimuli.  For layers with per-neuron NMF: loading =
    Σ_k λ_k · mean_{s∈S}(img_f[s, k]).

    Parameters
    ----------
    mask         : (n_samples,) bool array selecting stimuli
    img_f_by_lii : dict {linear_layer_idx: list of (n_samples, K) arrays}
    lams_by_lii  : dict {linear_layer_idx: list of (K,) arrays}
    all_acts     : (n_samples, n_neurons_total) concatenated activations
    act_offsets  : list of int — cumulative neuron counts, e.g. [0, 20, 30, 32]
    layer1_end   : int — act_offsets[1], where layer-1 activations end

    Returns
    -------
    loading : (n_neurons_total,) non-negative loading per neuron
    """
    loading = [all_acts[mask, :layer1_end].mean(axis=0)]
    for lii in sorted(img_f_by_lii.keys()):
        per_neuron = [
            (lams * img_f[mask].mean(axis=0)).sum()
            for img_f, lams in zip(img_f_by_lii[lii], lams_by_lii[lii])
        ]
        loading.append(np.array(per_neuron))
    return np.concatenate(loading)


def compute_effective_arbors(img_f_list, arb_f_list, lams_list, mask):
    """
    Stimulus-conditioned effective arbor matrix for one layer.

    E[n, i] = Σ_k lams[k] · mean_{s∈S}(img_f[s, k]) · arb_f[i, k]

    All terms are non-negative (NMF outputs), so E is non-negative.

    Parameters
    ----------
    img_f_list : list of (n_samples, K) arrays — per-neuron img_factors
    arb_f_list : list of (n_inputs, K) arrays — per-neuron arbor_factors
    lams_list  : list of (K,) arrays — per-neuron lambdas
    mask       : (n_samples,) bool — selected stimuli

    Returns
    -------
    E : (n_neurons, n_inputs) effective arbor matrix
    """
    n_inputs = arb_f_list[0].shape[0]
    E = np.zeros((len(img_f_list), n_inputs))
    for n, (img_f, arb_f, lams) in enumerate(zip(img_f_list, arb_f_list, lams_list)):
        coefs = lams * img_f[mask].mean(axis=0)   # (K,) lambda-weighted mean coefficients
        E[n] = (arb_f * coefs).sum(axis=1)        # (n_inputs,)
    return E


def full_nmf_pipeline(X, n_components, random_state=0, max_iter=20000, init=None):
    """
    Fit NMF, normalize, sort by importance, and rescale by sqrt(lambda).

    Returns (img_factors, neural_factors, lambdas).
      img_factors    : (n_samples, n_components)
      neural_factors : (n_neurons, n_components)
      lambdas        : (n_components,) descending
    """
    W, H, _ = run_nmf(X, n_components, random_state=random_state, max_iter=max_iter, init=init)
    W, H, lambdas = normalize_factors(W, H)
    W, H, lambdas, _ = sort_by_lambda(W, H, lambdas)
    scale = np.sqrt(lambdas)
    return W * scale, H * scale, lambdas


def auto_nmf_pipeline(X, k_max=None, method='cumvar', threshold=0.95,
                      min_k=1, min_cumvar=None, random_state=0, max_iter=20000):
    """Fit NMF at rank k_max then automatically select effective rank K* by informativity.

    A single NMF fit is performed at k_max; components are then pruned to K*
    based on sorted lambda values via select_k_from_lambdas. This is more
    efficient than re-fitting for every candidate rank.

    Note: NMF is not hierarchical (unlike PCA), so the K*-component result
    differs from fitting NMF directly at K*. For rank selection by relative
    informativity this trade-off is acceptable.

    Parameters
    ----------
    X          : (n_samples, n_features) non-negative matrix
    k_max      : int or None — upper bound on rank; None → min(min(X.shape)-1, 20)
    method     : str or list — passed to select_k_from_lambdas; 'structural'
                 recommended (fraction as primary, cumvar floor at threshold)
    threshold  : float — passed to select_k_from_lambdas
    min_k      : int — minimum number of components to return
    min_cumvar : float or None — explicit cumvar floor; passed to select_k_from_lambdas
    random_state, max_iter : passed to run_nmf

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
                                            max_iter=max_iter)
    k_star = select_k_from_lambdas(lams, method=method, threshold=threshold,
                                   min_k=min_k, min_cumvar=min_cumvar)
    return img_f[:, :k_star], neu_f[:, :k_star], lams[:k_star], k_star


# ── Backward factor trace ─────────────────────────────────────────────────────

def compute_joint_arbors_normalized(W, act_input, stimulus_weights=None, eps=1e-8,
                                    stimulus_threshold=0.0, neuron_weights=None):
    """
    Compute the normalized, rescaled joint arbor matrix for all neurons in a layer.

    For each sample s the activation row is L2-normalised (skipped when
    stimulus_weights is uniform), then for each neuron i:
    arbor_i[s, j] = W[i, j] * act_norm[s, j] * neuron_weights[i].
    All neurons' arbors are concatenated horizontally, rescaled by
    stimulus_weights, and rows where stimulus_weights[s] <= stimulus_threshold
    are zeroed out.

    Parameters
    ----------
    W                   : (n_out, n_in) layer weight matrix
    act_input           : (n_samples, n_in) input activations to this layer
    stimulus_weights    : (n_samples,) per-sample importance; uniform if None
    eps                 : stabiliser for L2 norm division
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
    if stimulus_weights is not None and np.allclose(stimulus_weights, 1.0):
        act_norm = act_input
    else:
        norms = np.linalg.norm(act_input, axis=1, keepdims=True)
        act_norm = act_input / (norms + eps)
    if neuron_weights is not None:
        arbors = [act_norm * W[i] * neuron_weights[i] for i in range(W.shape[0])]
    else:
        arbors = [act_norm * W[i] for i in range(W.shape[0])]
    joint = np.concatenate(arbors, axis=1)
    if stimulus_weights is not None:
        joint = joint * stimulus_weights[:, np.newaxis]
    if stimulus_threshold > 0.0 and (stimulus_weights is not None) and not np.allclose(stimulus_weights, 1.0):
        cutoff = np.quantile(stimulus_weights, stimulus_threshold)
        joint[stimulus_weights <= cutoff] = 0.0
    return joint


def trace_single_layer(W, act_input, stimulus_weights, k_max=None,
                       method='cumvar', threshold=0.95, min_cumvar=None,
                       stimulus_threshold=0.0, neuron_weights=None):
    """
    One backward factor-trace step: compute joint arbors and factorise.

    Parameters
    ----------
    W                   : (n_out, n_in)
    act_input           : (n_samples, n_in)
    stimulus_weights    : (n_samples,) per-sample importance from the layer above
    k_max               : upper bound on NMF rank passed to auto_nmf_pipeline;
                          None uses the default (min(min(X.shape)-1, 20))
    method, threshold   : rank-selection criterion for auto_nmf_pipeline
    min_cumvar          : optional cumvar floor passed to auto_nmf_pipeline
    stimulus_threshold  : passed through to compute_joint_arbors_normalized
    neuron_weights      : (n_out,) per-neuron importance; passed through

    Returns
    -------
    img_factors        : (n_samples, K*)
    neural_factors     : (n_out * n_in, K*)
    lambdas            : (K*,) descending — K* selected automatically
    joint_arbor        : (n_samples, n_out * n_in) positive joint (pre-NMF)
    neg_img_factors    : (n_samples, K*) or None
    neg_neural_factors : (n_out * n_in, K*) or None
    neg_lambdas        : (K*,) or None
    """
    raw_joint = compute_joint_arbors_normalized(W, act_input, stimulus_weights,
                                                stimulus_threshold=stimulus_threshold,
                                                neuron_weights=neuron_weights)
    pos_joint = np.clip(raw_joint, 0, None)
    neg_joint = np.clip(-raw_joint, 0, None)

    img_f, neu_f, lams, _ = auto_nmf_pipeline(pos_joint, k_max=k_max,
                                               method=method, threshold=threshold,
                                               min_cumvar=min_cumvar)

    if neg_joint.max() > 0:
        neg_img_f, neg_neu_f, neg_lams, _ = auto_nmf_pipeline(neg_joint, k_max=k_max,
                                                               method=method, threshold=threshold,
                                                               min_cumvar=min_cumvar)
    else:
        neg_img_f = neg_neu_f = neg_lams = None

    return img_f, neu_f, lams, pos_joint, neg_img_f, neg_neu_f, neg_lams


_EPS = 1e-12

_WEIGHTING_OPTIONS = ('img_factor', 'factor_project_raw', 'factor_project_corrected',
                      'img_factor_neuron')


def _compute_trace_transition(weighting, img_f, neu_f, W, act_into_current, fi):
    """
    Compute (stimulus_weights, neuron_weights) for the next lower layer.

    Called at the layer-L → layer-(L-1) transition.  Returns weights to pass
    into layer L-1's trace_single_layer call.

    Parameters
    ----------
    weighting        : str — one of 'img_factor', 'factor_project_raw',
                       'factor_project_corrected', 'img_factor_neuron'
    img_f            : (n_samples, K) NMF img_factors from layer L
    neu_f            : (n_out*n_in, K) NMF neural_factors from layer L
    W                : (n_out, n_in) weight matrix of layer L;
                       n_in equals the number of output neurons of layer L-1
    act_into_current : (n_samples, n_in) activations flowing FROM layer L-1
                       INTO layer L (= layer_inputs_list[l_idx]);
                       column i is the activation of output neuron i of layer L-1
    fi               : int — factor index to trace

    Returns
    -------
    sw : (n_samples,) stimulus weights for layer L-1's NMF
    nw : (n_in,) neuron weights for layer L-1's NMF, or None.
         nw[i] scales column block i (= output neuron i of layer L-1) inside
         compute_joint_arbors_normalized.
    """
    n_out, n_in = W.shape

    if weighting == 'img_factor':
        return img_f[:, fi], None

    elif weighting == 'factor_project_raw':
        nw = neu_f[:, fi].reshape(n_out, n_in).sum(axis=0)     # (n_in,) arbor-space direction
        target = nw / (np.linalg.norm(nw) + _EPS)
        sw = np.clip(act_into_current @ target, 0, None)
        return sw, nw

    elif weighting == 'factor_project_corrected':
        H_mat = neu_f[:, fi].reshape(n_out, n_in)              # (n_out, n_in)
        abs_W = np.abs(W)
        eps_w = max(1e-6 * float(abs_W.mean()), 1e-12)
        corrected = (H_mat / (abs_W + eps_w)).sum(axis=0)      # (n_in,) activation-space
        target = corrected / (np.linalg.norm(corrected) + _EPS)
        sw = np.clip(act_into_current @ target, 0, None)
        return sw, corrected

    elif weighting == 'img_factor_neuron':
        sw = img_f[:, fi]                                       # (n_samples,)
        nw = np.clip((sw @ act_into_current) / (sw.sum() + _EPS), 0, None)  # (n_in,)
        return sw, nw

    else:
        raise ValueError(
            f"Unknown weighting {weighting!r}. Choose one of: {_WEIGHTING_OPTIONS}"
        )


def dfs_trace(model, layer_inputs_list, k_max=5, method='cumvar', threshold=0.95,
              min_cumvar=None, stimulus_threshold=0.0, weighting='img_factor'):
    """
    Backward factor trace: single-chain depth-first from last layer to first.

    Starts with uniform stimulus weights at the last layer; each layer's top
    NMF factor provides stimulus_weights for the preceding layer.

    Parameters
    ----------
    model               : SimpleMLP — provides weight matrices via linear_layer_indices()
    layer_inputs_list   : list[ndarray] — (n_samples, n_in) per linear layer,
                          indexed first-to-last (pixel inputs first)
    k_max               : int, list[int or None], or None — upper bound on NMF rank
                          per layer (L1 first). A single int/None applies to all layers.
                          None entries use the auto_nmf_pipeline default.
    method, threshold   : rank-selection criterion passed to auto_nmf_pipeline
    min_cumvar          : optional cumvar floor passed through to auto_nmf_pipeline
    stimulus_threshold  : fraction in [0, 1); samples in the lowest quantile of
                          stimulus weights are zeroed in the joint arbor.
                          E.g. 0.1 zeros the bottom 10% by weight.
    weighting           : str — how to compute stimulus_weights and neuron_weights
                          for each layer transition; one of:
                          'img_factor'             img_f[:,0] as stimulus weights, no neuron weights
                          'factor_project_raw'     project activations onto arbor-space factor (original)
                          'factor_project_corrected' same but divide H by |W| first (activation-space)
                          'img_factor_neuron'      img_f[:,0] as stimulus weights + img-weighted
                                                   mean activation as neuron weights

    Returns
    -------
    list of dicts ordered last-layer-first, each containing:
        layer_idx           : 0-based (0 = L1, last = output layer)
        linear_idx          : index in model.layers Sequential
        W                   : (n_out, n_in) weight matrix
        joint_arbor         : (n_samples, n_out * n_in)
        img_factors         : (n_samples, K*)
        neural_factors      : (n_out * n_in, K*)
        lambdas             : (K*,) descending — K* selected automatically
        stimulus_weights_in : (n_samples,) weights passed into this layer
        neuron_weights_in   : (n_in,) per-neuron importance passed in, or None for last layer
    """
    linear_indices = model.linear_layer_indices()
    n_layers = len(linear_indices)
    k_list = list(k_max) if isinstance(k_max, (list, tuple)) else [k_max] * n_layers
    assert len(k_list) == n_layers, f"k_max list length {len(k_list)} != n_layers {n_layers}"

    n_samples = layer_inputs_list[0].shape[0]
    stimulus_weights = np.ones(n_samples)
    neuron_weights = None
    results = []
    for l_idx in reversed(range(n_layers)):
        li = linear_indices[l_idx]
        W = model.layers[li].weight.detach().numpy()
        img_f, neu_f, lams, joint, neg_img_f, neg_neu_f, neg_lams = trace_single_layer(
            W, layer_inputs_list[l_idx], stimulus_weights,
            k_max=k_list[l_idx], method=method, threshold=threshold,
            min_cumvar=min_cumvar, stimulus_threshold=stimulus_threshold,
            neuron_weights=neuron_weights,
        )
        results.append({
            'layer_idx': l_idx,
            'linear_idx': li,
            'W': W,
            'joint_arbor': joint,
            'img_factors': img_f,
            'neural_factors': neu_f,
            'lambdas': lams,
            'neg_img_factors': neg_img_f,
            'neg_neural_factors': neg_neu_f,
            'neg_lambdas': neg_lams,
            'stimulus_weights_in': stimulus_weights.copy(),
            'neuron_weights_in': neuron_weights.copy() if neuron_weights is not None else None,
        })
        stimulus_weights, neuron_weights = _compute_trace_transition(
            weighting, img_f, neu_f, W, layer_inputs_list[l_idx], fi=0)
    return results


def tree_trace(model, layer_inputs_list, k_max=5, n_branches=2, method='cumvar',
               threshold=0.95, min_cumvar=None, stimulus_threshold=0.0,
               weighting='img_factor'):
    """
    Backward factor trace with branching.

    At each layer the top n_branches NMF factors are each used to seed an
    independent NMF run on the preceding layer, producing a tree of results.

    Parameters
    ----------
    model               : SimpleMLP
    layer_inputs_list   : list[ndarray] — (n_samples, n_in) per linear layer,
                          indexed first-to-last
    k_max               : int, list[int or None], or None — upper bound on NMF rank
                          per layer (L1 first). A single int/None applies to all layers.
                          None entries use the auto_nmf_pipeline default.
    n_branches          : int or list[int] — how many top factors to follow at
                          each layer (indexed first-to-last, same as k_max).
                          A single int uses the same number at every layer.
    method, threshold   : rank-selection criterion passed to auto_nmf_pipeline
    min_cumvar          : optional cumvar floor passed through to auto_nmf_pipeline
    stimulus_threshold  : fraction in [0, 1) — zeros lowest-weight stimuli
    weighting           : str — one of 'img_factor', 'factor_project_raw',
                          'factor_project_corrected', 'img_factor_neuron';
                          see _compute_trace_transition for details

    Returns
    -------
    root : dict — last-layer node, same keys as dfs_trace nodes plus:
        'path'     : list[int] — factor indices chosen from root to this node
        'children' : list[dict] — preceding-layer nodes (empty at L1)
    """
    linear_indices = model.linear_layer_indices()
    n_layers = len(linear_indices)
    k_list = list(k_max) if isinstance(k_max, (list, tuple)) else [k_max] * n_layers
    assert len(k_list) == n_layers, f"k_max list length {len(k_list)} != n_layers {n_layers}"
    nb_list = (list(n_branches) if isinstance(n_branches, (list, tuple))
               else [n_branches] * n_layers)
    assert len(nb_list) == n_layers, f"n_branches list length {len(nb_list)} != n_layers {n_layers}"

    n_samples = layer_inputs_list[0].shape[0]

    def _trace_node(l_idx, stimulus_weights, neuron_weights, path):
        li = linear_indices[l_idx]
        W = model.layers[li].weight.detach().numpy()
        img_f, neu_f, lams, joint, neg_img_f, neg_neu_f, neg_lams = trace_single_layer(
            W, layer_inputs_list[l_idx], stimulus_weights,
            k_max=k_list[l_idx], method=method, threshold=threshold,
            min_cumvar=min_cumvar, stimulus_threshold=stimulus_threshold,
            neuron_weights=neuron_weights,
        )
        node = {
            'layer_idx': l_idx, 'linear_idx': li, 'W': W,
            'joint_arbor': joint, 'img_factors': img_f,
            'neural_factors': neu_f, 'lambdas': lams,
            'neg_img_factors': neg_img_f,
            'neg_neural_factors': neg_neu_f,
            'neg_lambdas': neg_lams,
            'stimulus_weights_in': stimulus_weights.copy(),
            'neuron_weights_in': neuron_weights.copy() if neuron_weights is not None else None,
            'path': path, 'children': [],
        }
        if l_idx > 0:
            for fi in range(min(nb_list[l_idx], len(lams))):
                sw_fi, nw_fi = _compute_trace_transition(
                    weighting, img_f, neu_f, W, layer_inputs_list[l_idx], fi=fi)
                node['children'].append(_trace_node(l_idx - 1, sw_fi, nw_fi, path + [fi]))
        return node

    return _trace_node(len(linear_indices) - 1, np.ones(n_samples), None, [])
