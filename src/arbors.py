"""Per-node arbor and activation-matrix construction.

Small, context-free helpers lifted out of the analysis notebooks (nb09/15/16) so
the HP-selection, separability and validation code can all build a node's joint
arbor the same way. A *node* here is a ``BFTNode``; a *layer input* is the
per-stimulus input feature map for that node's layer (``layer_inputs[node.layer_idx]``
in the notebooks, i.e. the ``input_fmap`` collected for that layer).

Nothing here re-fits NMF; it only rebuilds the matrices a fitted tree was fit on,
which is what stability / rank-sweep / weight-term-control all need.
"""
import numpy as np

from .bft import (compute_joint_arbors_normalized, compute_conv_joint_arbors,
                  compute_attn_joint_arbors)


def node_joint_arbor(node, layer_input, pool_method='avg'):
    """Rebuild a node's *signed* joint arbor, dispatching on layer type.

    Parameters
    ----------
    node         : BFTNode  (carries weight, layer_type, stimulus_weights, threshold,
                   and for attn nodes attn_weights)
    layer_input  : ndarray  — fc:(N,n_in); conv:(N,C,H,W); attn:(N,T,d_model)
    pool_method  : conv spatial pooling ('avg'|'max'|'center')

    Returns
    -------
    ndarray (N, n_out*n_in) signed — caller clips as needed.
    """
    lt = node.layer_type
    sw = node.stimulus_weights
    st = getattr(node, 'stimulus_threshold', 0.0)
    if lt == 'conv':
        return compute_conv_joint_arbors(node.weight, layer_input, stimulus_weights=sw,
                                         stimulus_threshold=st, pool_method=pool_method)
    if lt == 'attn':
        return compute_attn_joint_arbors(node.weight, layer_input, node.attn_weights,
                                         stimulus_weights=sw, stimulus_threshold=st)
    return compute_joint_arbors_normalized(node.weight, layer_input,
                                           stimulus_weights=sw, stimulus_threshold=st)


def node_pos_arbor(node, layer_input, pool_method='avg'):
    """Positive half of the joint arbor — the matrix BFT's main NMF is fit to."""
    return np.clip(node_joint_arbor(node, layer_input, pool_method), 0, None)


def activation_matrix(node, layer_input):
    """The activation-only matrix (N, features) matching a node's arbor input.

    Used by the weight-term control (arbor vs activations at matched rank) and as
    the per-layer activation baseline. Conv maps are average-pooled over space and
    attention inputs are collapsed by the same CLS weighting the arbor uses, so the
    activation baseline sees the same effective input as the arbor, minus the weight.
    """
    li = layer_input
    if node.layer_type == 'attn' and getattr(node, 'attn_weights', None) is not None:
        return np.einsum('nt,ntd->nd', node.attn_weights, li)
    if li.ndim == 4:
        return li.mean(axis=(2, 3))
    return li.reshape(len(li), -1)


def nodes_by_layer(tree):
    """{layer_idx: representative BFTNode} — first node seen per layer, BFS order."""
    from .types import BFTResult
    root = tree.root if isinstance(tree, BFTResult) else tree
    out, queue = {}, [root]
    while queue:
        nd = queue.pop(0)
        out.setdefault(nd.layer_idx, nd)
        queue.extend(nd.children)
    return out


def nodes_per_layer(tree, max_nodes=None):
    """{layer_idx: [BFTNodes]} — all nodes per layer (optionally capped), BFS order."""
    from .types import BFTResult
    root = tree.root if isinstance(tree, BFTResult) else tree
    out, queue = {}, [root]
    while queue:
        nd = queue.pop(0)
        out.setdefault(nd.layer_idx, []).append(nd)
        queue.extend(nd.children)
    if max_nodes is not None:
        out = {li: nds[:max_nodes] for li, nds in out.items()}
    return out
