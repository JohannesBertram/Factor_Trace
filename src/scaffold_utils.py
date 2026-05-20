import numpy as np


def bft_node_vals(result, fi=0, top_pct=0.05, use_reconstruction=True, use_neg=False):
    """
    Extract a flat node-value vector from one BFT result dict.

    Parameters
    ----------
    result : dict
        BFT result node with keys img_factors, neural_factors, W, layer_type.
    fi : int
        Factor column index to use.
    top_pct : float
        Fraction of top-loading stimuli (only used when use_reconstruction=True).
    use_reconstruction : bool
        True  → mean reconstruction over top-pct stimuli (realistic activity).
        False → neural_factors[:, fi] directly (single-factor, old behaviour).
    use_neg : bool
        Use neg_img_factors / neg_neural_factors instead. Returns None if absent.

    Returns
    -------
    ndarray of shape (n_features,), or None when use_neg=True and factors absent.
    """
    if use_neg:
        img_f = result.get('neg_img_factors')
        neu_f = result.get('neg_neural_factors')
        if img_f is None or neu_f is None:
            return None
    else:
        img_f = result['img_factors']
        neu_f = result['neural_factors']

    if not use_reconstruction:
        return neu_f[:, fi]

    n_top = max(1, int(np.ceil(len(img_f) * top_pct)))
    top_idx = np.argsort(img_f[:, fi])[-n_top:]
    return (img_f[top_idx] @ neu_f.T).mean(axis=0)


def _fi_for(results, i, fi, fi_seed):
    """Resolve factor index for results[i]'s edge matrix."""
    if isinstance(fi, int):
        return fi
    # fi == 'path': use path[-1] of the preceding (shallower) result.
    # That element records which factor of results[i] was selected during tracing.
    if i == 0:
        if fi_seed is not None:
            path = fi_seed.get('path', [])
            return path[-1] if path else 0
        return 0
    path = results[i - 1].get('path', [])
    return path[-1] if path else 0


def _reshape_vals(vals, result, aggregate_conv):
    """Reshape flat node-value vector to (n_out, n_in); conv: optionally aggregate spatial."""
    ltype = result.get('layer_type', 'fc')
    W = result['W']
    if ltype == 'conv':
        C_out, C_in, kH, kW = W.shape
        mat = vals.reshape(C_out, C_in, kH, kW)
        return mat.sum((-2, -1)) if aggregate_conv else mat.reshape(C_out, C_in * kH * kW)
    n_out, n_in = W.shape
    return vals.reshape(n_out, n_in)


def build_scaffold_edges(
    results,
    fi='path',
    fi_seed=None,
    top_pct=0.05,
    use_reconstruction=True,
    edge_threshold=0.0,
    aggregate_conv=True,
):
    """
    Build edge and neg-edge matrices from a list of BFT result dicts.

    Parameters
    ----------
    results : list of dict
        BFT result nodes in FORWARD order (shallowest / input-side first).
        One edge matrix is produced per result.
    fi : int or 'path'
        int   → same factor index for every layer.
        'path' → for results[i], derives fi from results[i-1]['path'][-1].
                  This correctly identifies which column of the deeper node was
                  selected during BFT tracing. Fixes the latent NB04 bug where
                  intermediate branching nodes used fi=0 unconditionally.
    fi_seed : dict or None
        When fi='path', provides path context for results[0] (no i-1 element).
        Pass the result that logically precedes results[0] in the full layer list.
        Required for NB02 to handle secondary branches correctly:
            build_scaffold_edges(layer_results[1:], fi_seed=layer_results[0], ...)
    top_pct : float
        Fraction of top-loading stimuli per layer (use_reconstruction=True only).
    use_reconstruction : bool
        True  → realistic mean over top-pct stimuli (new default).
        False → raw neural_factors[:, fi] (old behaviour; backward-compatible).
    edge_threshold : float
        After positive clipping, zero out values below (edge_threshold × layer_max).
        0.0 disables thresholding. Use e.g. 0.85 for NB04's sparsification.
    aggregate_conv : bool
        For conv layers, sum over spatial kernel dims kH×kW → (C_out, C_in).

    Returns
    -------
    edge_matrices     : list of ndarray, shape (n_out, n_in) or (C_out, C_in), forward order
    neg_edge_matrices : list of ndarray, same shape; all-zeros where no neg factors
    """
    edge_matrices = []
    neg_edge_matrices = []

    for i, result in enumerate(results):
        fi_i = _fi_for(results, i, fi, fi_seed)

        vals = bft_node_vals(result, fi_i, top_pct, use_reconstruction)
        E = np.clip(_reshape_vals(vals, result, aggregate_conv), 0, None)

        neg_vals = bft_node_vals(result, fi_i, top_pct, use_reconstruction, use_neg=True)
        if neg_vals is not None:
            neg_E = np.clip(_reshape_vals(neg_vals, result, aggregate_conv), 0, None)
        else:
            neg_E = np.zeros_like(E)

        if edge_threshold > 0:
            if E.max() > 0:
                E[E < E.max() * edge_threshold] = 0
            if neg_E.max() > 0:
                neg_E[neg_E < neg_E.max() * edge_threshold] = 0

        edge_matrices.append(E)
        neg_edge_matrices.append(neg_E)

    return edge_matrices, neg_edge_matrices


def scaffold_layer_sizes_from_edges(edge_matrices):
    """
    Derive layer_sizes from edge matrices: [n_in_of_first] + [n_out for each edge].
    Works for any mix of fc and conv (post-aggregate) edges.
    """
    sizes = [edge_matrices[0].shape[1]]
    for E in edge_matrices:
        sizes.append(E.shape[0])
    return sizes


def scaffold_loading_from_edges(edge_matrices):
    """
    Derive per-neuron node loading from edge matrices.

    For N edge matrices (N layer boundaries) produces N+1 node groups:
      groups 0..N-1 : edge_matrices[i].sum(axis=0)  — per-input-neuron outgoing weight
      group N       : edge_matrices[-1].sum(axis=1) — per-output-neuron incoming weight

    This matches NB02's hand-rolled loading computation exactly.
    For N=2, shapes (4,8) and (2,4): output is (8,)+(4,)+(2,) = 14 values.
    """
    parts = [E.sum(axis=0) for E in edge_matrices]
    parts.append(edge_matrices[-1].sum(axis=1))
    return np.concatenate(parts)
