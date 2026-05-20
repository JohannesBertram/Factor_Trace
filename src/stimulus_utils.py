"""stimulus_utils.py — Stimulus-conditioned analysis of BFT factor trees.

Three main analyses:
1. Factor tree visualisation  — colour a BFT tree by per-stimulus factor activations.
2. Factor fingerprints        — represent stimuli as concatenated img_factors vectors
                               and compute pairwise cosine similarity.
3. OOD / adversarial projection — project new stimuli onto fixed BFT factors via NNLS,
                               producing img_factors without refitting NMF.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
from scipy.optimize import nnls as scipy_nnls

from .bft import (
    compute_joint_arbors_normalized,
    compute_conv_joint_arbors,
    compute_attn_joint_arbors,
)


# ── 1. Tree extraction ────────────────────────────────────────────────────────

def extract_tree_nodes(root_node):
    """Walk the full BFT tree (BFS) and return a flat list of node metadata dicts.

    Each returned dict contains:
        node_id             : tuple  — unique path-based ID; root = ()
        parent_id           : tuple or None
        layer_idx           : int    — 0 = input-side layer (leaves), max = output (root)
        layer_name          : str
        layer_type          : 'fc' | 'conv' | 'attn'
        factor_idx          : int    — which factor of the parent was followed here
        img_factors         : (N, K) — per-stimulus NMF loadings at this node
        lambdas             : (K,)   — factor importances
        stimulus_weights_in : (N,)   — importance weights propagated from parent layer
        path                : list[int]

    Parameters
    ----------
    root_node : dict — root node returned by bft()

    Returns
    -------
    list[dict]  BFS order (root first, leaves last)
    """
    nodes = []
    queue = [(root_node, None)]
    while queue:
        node, parent_id = queue.pop(0)
        nid = tuple(node['path'])
        nodes.append({
            'node_id':              nid,
            'parent_id':            parent_id,
            'layer_idx':            node['layer_idx'],
            'layer_name':           node['layer_name'],
            'layer_type':           node['layer_type'],
            'factor_idx':           node['factor_idx'],
            'img_factors':          node['img_factors'],
            'lambdas':              node['lambdas'],
            'stimulus_weights_in':  node['stimulus_weights_in'],
            'path':                 node['path'],
        })
        for child in node['children']:
            queue.append((child, nid))
    return nodes


# ── 2. Node activations ───────────────────────────────────────────────────────

def compute_node_activations(tree_nodes, stimulus_indices):
    """Compute a scalar per-node activation for a set of stimuli.

    Root node:      mean img_factors[stimulus_indices, 0].
    Non-root nodes: mean stimulus_weights_in[stimulus_indices].

    stimulus_weights_in captures the importance propagated from the parent branch —
    the direct measure of "how relevant is this branch to these stimuli?"  For the
    root the weights are uniform (1.0), so the top-factor loading is used instead.

    Parameters
    ----------
    tree_nodes       : list[dict]  from extract_tree_nodes()
    stimulus_indices : array-like of int — indices into the N-sample axis

    Returns
    -------
    dict  {node_id: float}
    """
    s_idx = np.asarray(stimulus_indices)
    activations = {}
    for e in tree_nodes:
        nid = e['node_id']
        if e['parent_id'] is None:
            vals = e['img_factors'][s_idx, 0]
        else:
            vals = e['stimulus_weights_in'][s_idx]
        activations[nid] = float(np.mean(vals))
    return activations


# ── 3. Tree layout & visualisation ───────────────────────────────────────────

def _hierarchical_layout(tree_nodes):
    """Compute (x, y) positions for a BFT tree.

    Root is at the top (highest y = layer_idx of root).
    Leaves are at the bottom (y = 0).
    Leaf x positions are assigned sequentially; parent x is the mean of children.
    Supports multiple roots (e.g. factor-level trees).

    Returns
    -------
    dict  {node_id: (x, y)}
    """
    by_id       = {e['node_id']: e for e in tree_nodes}
    children_of = {e['node_id']: [] for e in tree_nodes}
    for e in tree_nodes:
        if e['parent_id'] is not None:
            children_of[e['parent_id']].append(e['node_id'])

    positions    = {}
    leaf_counter = [0]

    def _assign(nid):
        children = children_of[nid]
        if not children:
            x = float(leaf_counter[0])
            leaf_counter[0] += 1
        else:
            for c in children:
                _assign(c)
            x = float(np.mean([positions[c][0] for c in children]))
        y = float(by_id[nid]['layer_idx'])
        positions[nid] = (x, y)

    for e in tree_nodes:
        if e['parent_id'] is None:
            _assign(e['node_id'])
    return positions


def plot_factor_tree(
    tree_nodes,
    node_activations,
    ax=None,
    cmap='viridis',
    node_size=1400,
    font_size=7,
    title=None,
):
    """Draw the BFT factor tree coloured by per-node stimulus activations.

    Each node is a (layer, factor) pair identified by its path tuple.
    Node colour encodes the activation value from node_activations.
    Edges run from parent factor to child factor (output layer → input layer,
    top to bottom).

    Parameters
    ----------
    tree_nodes       : list[dict]  from extract_tree_nodes()
    node_activations : dict  {node_id: float}  from compute_node_activations()
    ax               : matplotlib Axes or None  (creates a new figure if None)
    cmap             : str  — colormap name
    node_size        : int  — scatter marker size
    font_size        : int  — node label font size
    title            : str or None

    Returns
    -------
    fig, ax
    """
    G = nx.DiGraph()
    for e in tree_nodes:
        G.add_node(e['node_id'])
        if e['parent_id'] is not None:
            G.add_edge(e['parent_id'], e['node_id'])

    pos  = _hierarchical_layout(tree_nodes)
    vals = np.array([node_activations[e['node_id']] for e in tree_nodes])
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    norm     = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)
    node_colors = [colormap(norm(node_activations[nid])) for nid in G.nodes()]

    labels = {}
    for e in tree_nodes:
        nid   = e['node_id']
        lname = e['layer_name']
        if 'factor_k' in e:
            labels[nid] = f"L{e['layer_idx']}\nf{e['factor_k']}"
        else:
            depth = len(e['path'])
            if depth == 0:
                labels[nid] = f"L{e['layer_idx']}\n{lname}"
            else:
                labels[nid] = f"L{e['layer_idx']}\nf{e['factor_idx']}"

    n_nodes = len(tree_nodes)
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6.0, n_nodes * 0.9), 4.0))
    else:
        fig = ax.get_figure()

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors,
        node_size=node_size,
        labels=labels,
        font_size=font_size,
        font_color='white',
        edge_color='#555555',
        arrows=True,
        arrowsize=15,
    )

    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Activation', shrink=0.6)
    ax.set_axis_off()
    if title:
        ax.set_title(title)

    return fig, ax


# ── 4. Factor-level tree (one node per (path, k)) ────────────────────────────

def extract_factor_tree_nodes(root_node):
    """Expand BFT tree into factor-level nodes — one node per (path_tuple, factor_k).

    Each of the K NMF factors at every BFT path node becomes a separate visual
    node, making all leaf factors visible in tree diagrams.

    Connectivity:
        Root factors  (path=[]):  parent_id = None
        Other factors (path=P):   parent_id = (P[:-1], P[-1])
            — i.e. the specific factor at the parent path that spawned this branch.

    Parameters
    ----------
    root_node : dict — BFT root returned by bft()

    Returns
    -------
    list[dict] with keys: node_id, parent_id, layer_idx, layer_name, layer_type,
        path, factor_k, img_factors, lambdas, stimulus_weights_in
    """
    path_nodes = {}
    queue = [root_node]
    while queue:
        node = queue.pop(0)
        path_nodes[tuple(node['path'])] = node
        queue.extend(node['children'])

    factor_nodes = []
    for pt, node in path_nodes.items():
        K = node['img_factors'].shape[1]
        parent_id = None if len(pt) == 0 else (pt[:-1], pt[-1])
        for k in range(K):
            factor_nodes.append({
                'node_id':             (pt, k),
                'parent_id':           parent_id,
                'layer_idx':           node['layer_idx'],
                'layer_name':          node['layer_name'],
                'layer_type':          node['layer_type'],
                'path':                node['path'],
                'factor_k':            k,
                'img_factors':         node['img_factors'],
                'lambdas':             node['lambdas'],
                'stimulus_weights_in': node['stimulus_weights_in'],
            })
    return factor_nodes


def compute_factor_activations(factor_nodes, stimulus_indices):
    """Compute per-node activation for a factor-level tree.

    For each node the activation is the mean of img_factors[stimulus_indices, factor_k].

    Parameters
    ----------
    factor_nodes     : list[dict]  from extract_factor_tree_nodes()
    stimulus_indices : array-like of int

    Returns
    -------
    dict  {node_id: float}
    """
    s_idx = np.asarray(stimulus_indices)
    return {
        e['node_id']: float(e['img_factors'][s_idx, e['factor_k']].mean())
        for e in factor_nodes
    }


# ── 6. Factor fingerprints ────────────────────────────────────────────────────

def extract_factor_fingerprint(root_node, stimulus_index):
    """Concatenate img_factors[s, :] for every node in the BFT tree (BFS order).

    The fingerprint length = sum of K_i over all tree nodes.

    Parameters
    ----------
    root_node      : dict  — BFT root returned by bft()
    stimulus_index : int   — index into the N-sample axis

    Returns
    -------
    np.ndarray  shape (fingerprint_dim,)
    """
    parts = []
    queue = [root_node]
    while queue:
        node = queue.pop(0)
        parts.append(node['img_factors'][stimulus_index, :])
        queue.extend(node['children'])
    return np.concatenate(parts)


def extract_fingerprint_matrix(root_node, stimulus_indices):
    """Build a (n_stimuli, fingerprint_dim) fingerprint matrix.

    Parameters
    ----------
    root_node        : dict  — BFT root returned by bft()
    stimulus_indices : array-like of int

    Returns
    -------
    np.ndarray  shape (n_stimuli, fingerprint_dim)
    """
    return np.stack([
        extract_factor_fingerprint(root_node, int(s)) for s in stimulus_indices
    ])


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


# ── 7. Tree traversal helpers ────────────────────────────────────────────────

def nodes_at_layer(root_node, target_layer_idx):
    """Return all BFT nodes whose layer_idx equals target_layer_idx.

    Parameters
    ----------
    root_node        : dict — BFT root returned by bft()
    target_layer_idx : int — 0 = input-side (leaves), max = output (root)

    Returns
    -------
    list[dict]  matching nodes in BFS order
    """
    from collections import deque
    result, queue = [], deque([root_node])
    while queue:
        n = queue.popleft()
        if n['layer_idx'] == target_layer_idx:
            result.append(n)
        queue.extend(n['children'])
    return result


def top_stimuli_factor_activations(factor_nodes, source_path_node, factor_k, top_n):
    """Select top-N stimuli for a given factor and compute factor-level tree activations.

    Ranks stimuli by img_factors[:, factor_k] at source_path_node, selects the top_n,
    and returns their mean activation across all factor nodes in the tree.

    Parameters
    ----------
    factor_nodes     : list[dict]  from extract_factor_tree_nodes()
    source_path_node : BFT path node dict whose img_factors column is used for ranking
    factor_k         : int — which factor column to rank by
    top_n            : int — number of top stimuli to include

    Returns
    -------
    acts    : dict {node_id: float}  — mean factor activation per node for top stimuli
    top_idx : np.ndarray             — indices of the top_n stimuli
    """
    loadings = source_path_node['img_factors'][:, factor_k]
    top_idx  = np.argsort(loadings)[::-1][:top_n]
    acts     = compute_factor_activations(factor_nodes, top_idx)
    return acts, top_idx


# ── 8. OOD / adversarial projection ──────────────────────────────────────────

def _nnls_project(neural_factors, pos_joint):
    """Solve NNLS per row of pos_joint against the fixed neural_factors basis.

    Solves:  min_{x >= 0}  ||neural_factors @ x − pos_joint[s]||²
    for each row s.

    Parameters
    ----------
    neural_factors : (n_features, K)  — fixed NMF basis from bft()
    pos_joint      : (n_new, n_features)  — clipped-positive joint arbor for new stimuli

    Returns
    -------
    np.ndarray  (n_new, K)
    """
    n_new = pos_joint.shape[0]
    K     = neural_factors.shape[1]
    img_f = np.zeros((n_new, K))
    for s in range(n_new):
        img_f[s], _ = scipy_nnls(neural_factors, pos_joint[s])
    return img_f


def _joint_for_node(node, new_input):
    """Compute the raw signed joint arbor for new_input at a given BFT node.

    Parameters
    ----------
    node      : BFT node dict (contains 'W', 'layer_type')
    new_input : For 'fc' / 'conv': numpy array of activations.
                For 'attn': dict with keys 'x_tokens' (N,T,d_model)
                            and 'attn_weights' (N,T).

    Returns
    -------
    np.ndarray  (n_new, n_out * n_in)  raw (signed) joint arbor
    """
    ltype = node['layer_type']
    W     = node['W']
    if ltype == 'conv':
        return compute_conv_joint_arbors(W, new_input, stimulus_weights=None)
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
            stimulus_weights=None,
        )
    else:
        return compute_joint_arbors_normalized(W, new_input, stimulus_weights=None)


def project_stimuli_onto_tree(root_node, new_layer_inputs):
    """Project new stimuli onto fixed BFT factors via per-node NNLS.

    For every node in the BFT tree the joint arbor is recomputed from the new
    stimuli's layer activations and then projected onto the stored neural_factors
    with non-negative least squares to obtain img_factors for the new stimuli.

    This is an *approximation*: the original neural_factors were fitted on
    stimulus-weighted arbors (weights propagated from the layer above), but here
    all stimuli are treated with uniform weights.  The resulting img_factors are
    still interpretable as "how much does this stimulus express each factor?" and
    are compatible with extract_factor_fingerprint / plot_factor_tree.

    Parameters
    ----------
    root_node        : dict  — BFT tree root from bft()
    new_layer_inputs : list  — one entry per layer in FORWARD order (index 0 =
                       input-side layer, matching the layer_inputs_list argument
                       to bft()).

                       FC / conv layers: numpy array (n_new, n_in) or
                           (n_new, C_in, H, W).
                       Attention layers: dict with keys
                           'x_tokens'    — (n_new, T, d_model)
                           'attn_weights' — (n_new, T)

    Returns
    -------
    dict  — deep copy of the BFT tree with 'img_factors' (and 'neg_img_factors'
            if present) replaced by NNLS-projected values for the new stimuli,
            and 'stimulus_weights_in' set to uniform ones.
    """
    new_root = copy.deepcopy(root_node)

    def _project(node):
        l_idx     = node['layer_idx']
        new_input = new_layer_inputs[l_idx]
        raw_joint = _joint_for_node(node, new_input)

        pos_joint = np.clip(raw_joint,  0, None)
        neg_joint = np.clip(-raw_joint, 0, None)

        node['img_factors'] = _nnls_project(node['neural_factors'], pos_joint)

        if node.get('neg_neural_factors') is not None:
            node['neg_img_factors'] = _nnls_project(node['neg_neural_factors'], neg_joint)

        n_new = pos_joint.shape[0]
        node['stimulus_weights_in'] = np.ones(n_new)

        for child in node['children']:
            _project(child)

    _project(new_root)
    return new_root
