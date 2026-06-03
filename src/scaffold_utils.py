import numpy as np


def bft_node_vals(result, fi=0, top_pct=0.05, use_reconstruction=True, use_neg=False):
    """Extract a flat node-value vector from one BFT result node.

    Parameters
    ----------
    result : BFTNode
        BFT node with img_factors, connection_factors, weight, layer_type.
    fi : int
        Factor column index to use.
    top_pct : float
        Fraction of top-loading stimuli (only used when use_reconstruction=True).
    use_reconstruction : bool
        True  → mean reconstruction over top-pct stimuli (realistic activity).
        False → connection_factors[:, fi] directly (single-factor).
    use_neg : bool
        Use neg_img_factors / neg_connection_factors instead. Returns None if absent.

    Returns
    -------
    ndarray of shape (n_features,), or None when use_neg=True and factors absent.
    """
    if use_neg:
        img_f = result.neg_img_factors
        con_f = result.neg_connection_factors
        if img_f is None or con_f is None:
            return None
    else:
        img_f = result.img_factors
        con_f = result.connection_factors

    if not use_reconstruction:
        return con_f[:, fi]

    n_top = max(1, int(np.ceil(len(img_f) * top_pct)))
    top_idx = np.argsort(img_f[:, fi])[-n_top:]
    return (img_f[top_idx] @ con_f.T).mean(axis=0)


def _fi_for(results, i, fi, fi_seed):
    """Resolve factor index for results[i]'s edge matrix."""
    if isinstance(fi, int):
        return fi
    # fi == 'path': use path[-1] of the preceding (shallower) result.
    # That element records which factor of results[i] was selected during tracing.
    if i == 0:
        if fi_seed is not None:
            path = fi_seed.path
            return path[-1] if path else 0
        return 0
    path = results[i - 1].path
    return path[-1] if path else 0


def _reshape_vals(vals, result, aggregate_conv):
    """Reshape flat node-value vector to (n_out, n_in); conv: optionally aggregate spatial."""
    ltype = result.layer_type
    W = result.weight
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
    """Build edge and neg-edge matrices from a list of BFT nodes.

    Parameters
    ----------
    results : list of BFTNode
        BFT nodes in FORWARD order (shallowest / input-side first).
        One edge matrix is produced per result.
    fi : int or 'path'
        int   → same factor index for every layer.
        'path' → for results[i], derives fi from results[i-1].path[-1].
    fi_seed : BFTNode or None
        When fi='path', provides path context for results[0] (no i-1 element).
        Pass the node that logically precedes results[0] in the full layer list.
    top_pct : float
        Fraction of top-loading stimuli per layer (use_reconstruction=True only).
    use_reconstruction : bool
        True  → realistic mean over top-pct stimuli (default).
        False → raw connection_factors[:, fi].
    edge_threshold : float
        After positive clipping, zero out values below (edge_threshold × layer_max).
        0.0 disables thresholding. Use e.g. 0.85 for sparsification.
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

    # At conv→FC boundaries the FC weight's n_in is spatially flattened
    # (prev_channels × H × W). Aggregate over the spatial positions so every
    # edge matrix has shape (layer_out, layer_in) using channel counts only.
    for i in range(1, len(edge_matrices)):
        prev_out = edge_matrices[i - 1].shape[0]
        curr_in  = edge_matrices[i].shape[1]
        if curr_in != prev_out and curr_in % prev_out == 0:
            spatial = curr_in // prev_out
            edge_matrices[i] = edge_matrices[i].reshape(
                edge_matrices[i].shape[0], prev_out, spatial).sum(axis=-1)
            neg_edge_matrices[i] = neg_edge_matrices[i].reshape(
                neg_edge_matrices[i].shape[0], prev_out, spatial).sum(axis=-1)

    return edge_matrices, neg_edge_matrices


def scaffold_layer_sizes_from_edges(edge_matrices):
    """Derive layer_sizes from edge matrices.

    Returns [n_in of E0, n_out of E0, n_out of E1, ..., n_out of E_{N-1}].
    Chaining via output dims ensures consistency after conv→FC spatial aggregation.
    """
    sizes = [edge_matrices[0].shape[1]]
    for E in edge_matrices:
        sizes.append(E.shape[0])
    return sizes


def scaffold_loading_from_edges(edge_matrices):
    """Derive per-neuron node loading from edge matrices.

    For N edge matrices (N layer boundaries) produces N+1 node groups:
      groups 0..N-1 : edge_matrices[i].sum(axis=0)  — per-input-neuron outgoing weight
      group N       : edge_matrices[-1].sum(axis=1) — per-output-neuron incoming weight

    For N=2, shapes (4,8) and (2,4): output is (8,)+(4,)+(2,) = 14 values.
    """
    parts = [E.sum(axis=0) for E in edge_matrices]
    parts.append(edge_matrices[-1].sum(axis=1))
    return np.concatenate(parts)
