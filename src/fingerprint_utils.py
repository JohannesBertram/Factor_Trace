"""fingerprint_utils.py — Factor fingerprints and NNLS-based stimulus projection.

Two main analyses:
1. Factor fingerprints  — represent stimuli as concatenated img_factors vectors
                          across all BFT tree nodes and compute pairwise cosine
                          similarity.
2. NNLS projection      — project new stimuli onto fixed BFT factors without
                          refitting NMF, producing img_factors for OOD / held-out
                          stimuli.
"""

import copy
import numpy as np
from scipy.optimize import nnls as scipy_nnls

from .bft import (
    compute_joint_arbors_normalized,
    compute_conv_joint_arbors,
    compute_attn_joint_arbors,
    _compute_trace_transition,
    _collect_layer_dicts,
)
from .types import BFTNode, BFTResult, FingerprintResult


# ── 1. Factor fingerprints ─────────────────────────────────────────────────────

def extract_fingerprint_matrix(root_node, stimulus_indices):
    """Build a (n_stimuli, fingerprint_dim) fingerprint matrix.

    Single BFS pass over the tree; all stimuli are indexed at once.

    Parameters
    ----------
    root_node        : BFTNode or BFTResult
    stimulus_indices : array-like of int

    Returns
    -------
    np.ndarray  shape (n_stimuli, fingerprint_dim)
    """
    if isinstance(root_node, BFTResult):
        root_node = root_node.root
    indices = np.asarray(stimulus_indices)
    parts = []
    queue = [root_node]
    while queue:
        node = queue.pop(0)
        parts.append(node.img_factors[indices, :])  # (n_stimuli, K_i)
        queue.extend(node.children)
    return np.concatenate(parts, axis=1)            # (n_stimuli, fingerprint_dim)


def extract_factor_fingerprint(root_node, stimulus_index):
    """Return the fingerprint vector for a single stimulus.

    Equivalent to extract_fingerprint_matrix(root_node, [stimulus_index])[0].

    Parameters
    ----------
    root_node      : BFTNode or BFTResult
    stimulus_index : int

    Returns
    -------
    np.ndarray  shape (fingerprint_dim,)
    """
    return extract_fingerprint_matrix(root_node, [stimulus_index])[0]


def compute_stimulus_similarity(fingerprint_matrix):
    """Pairwise cosine similarity from fingerprint vectors.

    Parameters
    ----------
    fingerprint_matrix : (n_stimuli, fingerprint_dim) array

    Returns
    -------
    np.ndarray  (n_stimuli, n_stimuli) symmetric cosine similarity matrix, range [-1, 1]
    """
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(fingerprint_matrix)


def compute_fingerprints(bft_result, indices=None, normalize=True):
    """Compute factor fingerprints for a set of stimuli from a BFT result.

    Paper: Sec. 2.4 (Factor Fingerprints).

    Parameters
    ----------
    bft_result : BFTResult or BFTNode
    indices    : array-like of int or None — None uses all N samples
    normalize  : bool — L2-normalize rows before computing cosine similarity

    Returns
    -------
    FingerprintResult with:
        .matrix     (N, D) — per-stimulus concatenated factor loadings
        .similarity (N, N) — pairwise cosine similarity
        .indices    (N,)   — which sample indices were used
    """
    from sklearn.metrics.pairwise import cosine_similarity

    if isinstance(bft_result, BFTResult):
        root = bft_result.root
        n_total = bft_result.n_samples
    else:
        root = bft_result
        n_total = root.img_factors.shape[0]

    if indices is None:
        indices = np.arange(n_total)
    indices = np.asarray(indices)

    matrix = extract_fingerprint_matrix(root, indices)

    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norm = matrix / (norms + 1e-12)
    else:
        matrix_norm = matrix

    similarity = cosine_similarity(matrix_norm)
    return FingerprintResult(matrix=matrix, similarity=similarity, indices=indices)


# ── 2. NNLS projection ─────────────────────────────────────────────────────────

def _nnls_project(connection_factors, pos_joint):
    """Solve NNLS per row of pos_joint against the fixed connection_factors basis.

    Solves:  min_{x >= 0}  ||connection_factors @ x − pos_joint[s]||²  for each row s.

    Parameters
    ----------
    connection_factors : (n_features, K)  — fixed NMF basis from bft()
    pos_joint          : (n_new, n_features)

    Returns
    -------
    np.ndarray  (n_new, K)
    """
    n_new = pos_joint.shape[0]
    K     = connection_factors.shape[1]
    img_f = np.zeros((n_new, K))
    for s in range(n_new):
        img_f[s], _ = scipy_nnls(connection_factors, pos_joint[s], maxiter=300)
    return img_f


def _joint_for_node_weighted(node, new_input, stimulus_weights):
    """Compute the raw signed joint arbor for new_input at a BFT node.

    Parameters
    ----------
    node             : BFTNode
    new_input        : ndarray (fc/conv) or dict with 'x_tokens'/'attn_weights' (attn)
    stimulus_weights : (n_new,) per-sample importance weights

    Returns
    -------
    np.ndarray  (n_new, n_out * n_in)
    """
    ltype = node.layer_type
    W     = node.weight
    st    = node.stimulus_threshold

    if ltype == 'conv':
        return compute_conv_joint_arbors(W, new_input,
                                          stimulus_weights=stimulus_weights,
                                          stimulus_threshold=st)
    elif ltype == 'attn':
        if not isinstance(new_input, dict):
            raise ValueError(
                "For 'attn' nodes, new_layer_inputs[layer_idx] must be a dict "
                "with keys 'x_tokens' (N,T,d_model) and 'attn_weights' (N,T)."
            )
        return compute_attn_joint_arbors(
            W,
            new_input['x_tokens'],
            new_input['attn_weights'],
            stimulus_weights=stimulus_weights,
            stimulus_threshold=st,
        )
    else:
        return compute_joint_arbors_normalized(W, new_input,
                                               stimulus_weights=stimulus_weights,
                                               stimulus_threshold=st)


def project_stimuli_onto_tree(root_node, new_layer_inputs):
    """Project new stimuli onto fixed BFT factors via backward-weighted NNLS.

    Mirrors the BFT backward pass: starts at the root with uniform stimulus
    weights and works toward the input, propagating per-factor weights using the
    same strategy as the original BFT.

    Parameters
    ----------
    root_node        : BFTNode or BFTResult
    new_layer_inputs : list — one entry per layer in forward order (0 = input side).
                       FC/conv: ndarray (n_new, ...).
                       Attn: dict with 'x_tokens' (n_new, T, d_model) and
                             'attn_weights' (n_new, T).

    Returns
    -------
    BFTNode — deep copy of the tree with img_factors replaced by NNLS projections.
    """
    if isinstance(root_node, BFTResult):
        root_node = root_node.root

    new_root = copy.deepcopy(root_node)

    root_input = new_layer_inputs[root_node.layer_idx]
    n_new = (root_input['x_tokens'].shape[0]
             if isinstance(root_input, dict) else root_input.shape[0])

    def _project(node, stimulus_weights):
        l_idx     = node.layer_idx
        new_input = new_layer_inputs[l_idx]

        raw_joint = _joint_for_node_weighted(node, new_input, stimulus_weights)
        pos_joint = np.clip(raw_joint,  0, None)
        neg_joint = np.clip(-raw_joint, 0, None)

        node.img_factors = _nnls_project(node.connection_factors, pos_joint)

        if node.neg_connection_factors is not None:
            node.neg_img_factors = _nnls_project(node.neg_connection_factors, neg_joint)

        node.stimulus_weights = stimulus_weights.copy()

        for child in node.children:
            fi = child.factor_idx
            sw_fi, _ = _compute_trace_transition(node.weighting, node.img_factors,
                                                  node.lambdas, fi)
            _project(child, sw_fi)

    _project(new_root, np.ones(n_new))
    return new_root


def project_onto_bft(bft_result, model, data, *,
                     only_correct=False, device=None, layer_filter=None):
    """Project new stimuli onto fixed BFT factors and return a new BFTResult.

    Paper: App. "Projecting New Stimuli" (NNLS); Sec. 2.4.

    Mirrors the primary BFT interface: collects layer activations from the model
    and dataloader, then projects via NNLS onto the fixed factors in bft_result.

    Parameters
    ----------
    bft_result   : BFTResult — fixed tree from a prior bft() call
    model        : nn.Module
    data         : DataLoader — yields (images, labels) batches
    only_correct : bool — keep only correctly classified samples (default False)
    device       : torch device or None
    layer_filter : callable(name, mod) -> bool or None — passed to _collect_layer_dicts;
                   use this for architectures with parallel branches (e.g. SqueezeNet)
                   to select only the sequential spine layers

    Returns
    -------
    BFTResult — same tree structure (same connection_factors) but new img_factors.
    """
    raw = _collect_layer_dicts(model, data, device=device, only_correct=only_correct,
                               layer_filter=layer_filter)
    new_layer_inputs = [d['input_fmap'] for d in raw['layer_data']]

    projected_root = project_stimuli_onto_tree(bft_result, new_layer_inputs)

    return BFTResult(
        root=projected_root,
        images=raw['images'],
        targets=raw['targets'],
        confidences=raw['confidences'],
    )
