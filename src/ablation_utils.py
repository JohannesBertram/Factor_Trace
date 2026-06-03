import copy

import numpy as np
import torch


# ── Factor-to-class assignment ────────────────────────────────────────────────

def assign_factors_to_classes(root_node, all_targets, n_classes):
    """
    Assign each of the root node's children to one output class (greedy argmax).

    For each class d we pick the child whose img_factors[:, 0] has the highest
    mean over class-d samples.  No double-assignment.

    Parameters
    ----------
    root_node   : BFTNode or BFTResult — bft root
    all_targets : (n_samples,) int array of class labels
    n_classes   : number of classes

    Returns
    -------
    class_to_child : dict {class_d: child_node}
    """
    from .types import BFTResult
    if isinstance(root_node, BFTResult):
        root_node = root_node.root
    children = root_node.children
    assigned = {}
    used = set()
    for d in range(n_classes):
        mask = (all_targets == d)
        scores = [
            float(child.img_factors[mask, 0].mean()) if i not in used else -np.inf
            for i, child in enumerate(children)
        ]
        best = int(np.argmax(scores))
        assigned[d] = children[best]
        used.add(best)
    return assigned


def path_from_child(root_node, child_node):
    """
    Build a node list [root_node, child_node, child_node's child, ...] down to L1.

    Inner nodes are traced with n_branches=1, so each has at most one child.

    Returns
    -------
    nodes : list[BFTNode] ordered output-layer first
    """
    nodes = [root_node, child_node]
    cur = child_node
    while cur.children:
        cur = cur.children[0]
        nodes.append(cur)
    return nodes


# ── Importance extraction ─────────────────────────────────────────────────────

def extract_importance_scores(path_nodes, selectivity_weights=None, neg_selectivity_weights=None):
    """
    Extract per-weight importance from neural_factors along a traced path.

    Parameters
    ----------
    path_nodes : list[dict] — ordered output-layer first (from path_from_child)
    selectivity_weights : array of shape (K,) or None
        When provided, the root node's importance is computed as a weighted sum
        across all K factors (weights clipped to ≥0 and normalised to sum=1).
        Inner nodes always use fi=0 regardless of this parameter.
        If None, falls back to the single-factor (k_star) behaviour.

    Returns
    -------
    scores : dict {(layer_idx, i, j): float}

    Notes
    -----
    factor_idx on a node records which column of the *parent's* NMF spawned this
    branch, not which column to use from this node's own neural_factors.

    For path_nodes[0] (the root / output-side node), the relevant column is
    path_nodes[1]['factor_idx'] — i.e. which root factor this path follows.
    For all deeper nodes, inner BFT always branches on factor 0, so column 0 is used.
    A safety clamp handles any edge case where fi >= node's actual k.

    Inhibitory factors (neg_neural_factors) are added to the excitatory score so
    that weights acting through the inhibitory pathway are not systematically
    under-ranked.
    """
    scores = {}
    for idx, node in enumerate(path_nodes):
        l_idx  = node.layer_idx
        ltype  = node.layer_type
        W      = node.weight
        nf     = node.connection_factors         # (n_out*n_in_flat, K_pos)
        K      = nf.shape[1]

        # --- excitatory importance -------------------------------------------
        if idx == 0 and selectivity_weights is not None:
            # Weighted sum across all K root factors
            w = np.clip(np.asarray(selectivity_weights, dtype=float)[:K], 0, None)
            w = w / (w.sum() + 1e-8)
            flat = nf @ w                        # (n_out*n_in_flat,)
        else:
            if idx == 0 and len(path_nodes) > 1:
                fi = path_nodes[1].factor_idx
            else:
                fi = 0
            fi = min(fi, K - 1)
            flat = nf[:, fi]                     # (n_out*n_in_flat,)

        # --- inhibitory importance (add, not replace) ------------------------
        neg_nf = node.neg_connection_factors
        if neg_nf is not None and neg_nf.shape[0] == flat.shape[0]:
            if idx == 0 and selectivity_weights is not None:
                K_neg = neg_nf.shape[1]
                if neg_selectivity_weights is not None:
                    w_neg = np.clip(np.asarray(neg_selectivity_weights, dtype=float)[:K_neg], 0, None)
                else:
                    w_neg = np.ones(K_neg, dtype=float)
                w_neg = w_neg / (w_neg.sum() + 1e-8)
                flat = flat + neg_nf @ w_neg
            else:
                fi_neg = min(fi, neg_nf.shape[1] - 1)
                flat = flat + neg_nf[:, fi_neg]

        if ltype == 'conv':
            C_out, C_in, kH, kW = W.shape
            imp = flat.reshape(C_out, C_in * kH * kW)
            n_out, n_in = C_out, C_in * kH * kW
        else:
            n_out, n_in = W.shape
            imp = flat.reshape(n_out, n_in)
        for i in range(n_out):
            for j in range(n_in):
                scores[(l_idx, i, j)] = float(imp[i, j])
    return scores


def all_weight_keys(model):
    """Return a list of all (layer_idx, i, j) weight keys in model order."""
    keys = []
    for l_idx, li in enumerate(model.linear_layer_indices()):
        W = model.layers[li].weight.detach().numpy()
        n_out, n_in = W.shape
        for i in range(n_out):
            for j in range(n_in):
                keys.append((l_idx, i, j))
    return keys


# ── Model ablation ────────────────────────────────────────────────────────────

def ablate_model(model_orig, weight_keys_to_zero):
    """
    Return a deep copy of model_orig with the specified weights set to 0.

    Parameters
    ----------
    weight_keys_to_zero : iterable of (layer_idx, i, j)

    Returns
    -------
    model_copy : nn.Module (eval mode)
    """
    model = copy.deepcopy(model_orig)
    linear_indices = model.linear_layer_indices()
    with torch.no_grad():
        for (l_idx, i, j) in weight_keys_to_zero:
            li = linear_indices[l_idx]
            model.layers[li].weight[i, j] = 0.0
    model.eval()
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def per_class_accuracy(model, loader, label_transform, device):
    """
    Evaluate per-class accuracy on the test set.

    Parameters
    ----------
    model            : nn.Module
    loader           : DataLoader
    label_transform  : callable or None
    device           : torch device

    Returns
    -------
    acc : dict {class_d: float} — accuracy in [0, 1]
    """
    correct_counts = {}
    total_counts = {}
    model.eval()
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            if label_transform is not None:
                target = label_transform(target)
            result = model(data)
            output = result[0] if isinstance(result, (tuple, list)) else result
            preds = output.argmax(1)
            for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
                t = int(t)
                total_counts[t] = total_counts.get(t, 0) + 1
                correct_counts[t] = correct_counts.get(t, 0) + int(t == p)
    return {d: correct_counts.get(d, 0) / total_counts[d]
            for d in sorted(total_counts)}


# ── Baseline importance scores ────────────────────────────────────────────────

def magnitude_scores(model):
    """Return {(l_idx, i, j): |W[i,j]|} for all weights."""
    scores = {}
    for l_idx, li in enumerate(model.linear_layer_indices()):
        W = model.layers[li].weight.detach().cpu().numpy()
        n_out, n_in = W.shape
        for i in range(n_out):
            for j in range(n_in):
                scores[(l_idx, i, j)] = float(abs(W[i, j]))
    return scores


def act_magnitude_scores(model, layer_inputs_list):
    """
    Return {(l_idx, i, j): |W[i,j]| * mean_act[j]} where mean_act is the
    mean absolute activation over all samples at the layer's input.
    """
    scores = {}
    for l_idx, li in enumerate(model.linear_layer_indices()):
        W = model.layers[li].weight.detach().cpu().numpy()
        n_out, n_in = W.shape
        mean_act = np.abs(layer_inputs_list[l_idx]).mean(axis=0)  # (n_in,)
        for i in range(n_out):
            for j in range(n_in):
                scores[(l_idx, i, j)] = float(abs(W[i, j]) * mean_act[j])
    return scores


def taylor_scores(model, loader, label_transform, device, target_class):
    """
    Compute Taylor-criterion importance |grad(L) * W| for class-d samples only.

    Uses a single forward+backward pass.  The loss is cross-entropy over
    class-target_class samples.

    Returns {(l_idx, i, j): float}
    """
    model_tmp = copy.deepcopy(model)
    model_tmp.train()
    linear_indices = model_tmp.linear_layer_indices()

    optimizer = torch.optim.SGD(model_tmp.parameters(), lr=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    n_seen = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        if label_transform is not None:
            target = label_transform(target)
        mask = (target == target_class)
        if not mask.any():
            continue
        result = model_tmp(data[mask])
        output = result[0] if isinstance(result, tuple) else result
        loss = criterion(output, target[mask])
        loss.backward()
        n_seen += int(mask.sum())

    scores = {}
    for l_idx, li in enumerate(linear_indices):
        W = model_tmp.layers[li].weight
        grad = W.grad
        if grad is None:
            W_np = W.detach().numpy()
            n_out, n_in = W_np.shape
            for i in range(n_out):
                for j in range(n_in):
                    scores[(l_idx, i, j)] = 0.0
            continue
        taylor = (grad * W).abs().detach().cpu().numpy()
        n_out, n_in = taylor.shape
        for i in range(n_out):
            for j in range(n_in):
                scores[(l_idx, i, j)] = float(taylor[i, j])
    return scores


# ── Score normalisation ───────────────────────────────────────────────────────

def normalize_scores_per_layer(scores):
    """
    Min-max normalise importance scores within each layer to [0, 1].

    Use this before `run_ablation_sweep` with method='algo_bottom' to prevent
    wide early layers (with many near-zero NMF values) from dominating the
    low-importance pool at the expense of narrower output layers.

    Parameters
    ----------
    scores : dict {(layer_idx, i, j): float}

    Returns
    -------
    normalised : dict {(layer_idx, i, j): float}
    """
    by_layer = {}
    for key in scores:
        by_layer.setdefault(key[0], []).append(key)
    result = {}
    for l_idx, keys in by_layer.items():
        vals = np.array([scores[k] for k in keys])
        lo, hi = float(vals.min()), float(vals.max())
        rng = hi - lo
        for k in keys:
            result[k] = float((scores[k] - lo) / rng) if rng > 0 else 0.0
    return result


# ── Ablation sweep ────────────────────────────────────────────────────────────

def select_class_circuit(root_node, targets, class_d):
    """
    Return importance scores for the output factor most selective for class d.

    Selectivity of factor k for class d is defined as:
        selectivity_k = mean(img_factors[targets==d, k])
                      / (mean(img_factors[targets!=d, k]) + 1e-6)

    The factor k* with the highest selectivity is chosen. If k* < 1 (no factor
    prefers class d), the factor with the highest raw mean activation for class d
    is used instead, and ``info['is_selective']`` is False.

    Parameters
    ----------
    root_node : BFT root node dict (has img_factors, children)
    targets   : (N,) int array of task-class labels
    class_d   : int — which class to select a circuit for

    Returns
    -------
    importance_scores : dict {(layer_idx, i, j): float}
    info : dict with keys 'k_star', 'selectivity', 'all_selectivities',
           'is_selective', 'warning' (str or None)
    """
    from .types import BFTResult
    if isinstance(root_node, BFTResult):
        root_node = root_node.root
    img_factors = root_node.img_factors         # (N, K)
    K = img_factors.shape[1]
    tgt = np.asarray(targets)
    mask_d = (tgt == class_d)
    mask_o = ~mask_d

    selectivities = []
    for k in range(K):
        mean_d = float(img_factors[mask_d, k].mean()) if mask_d.any() else 0.0
        mean_o = float(img_factors[mask_o, k].mean()) if mask_o.any() else 0.0
        selectivities.append(mean_d / (mean_o + 1e-6))

    k_star = int(np.argmax(selectivities))
    is_selective = selectivities[k_star] > 1.0
    warning = None if is_selective else (
        f'No factor with selectivity > 1.0 for class {class_d}; '
        f'using factor {k_star} (highest raw mean activation for this class).'
    )

    # Find the child of root that followed factor k_star
    children = root_node.children
    child = None
    for c in children:
        if c.factor_idx == k_star:
            child = c
            break
    if child is None and children:
        child = children[0]  # fallback: first child

    if child is not None:
        path_nodes = path_from_child(root_node, child)
    else:
        path_nodes = [root_node]

    neg_selectivities = None
    neg_img_f = root_node.neg_img_factors
    if neg_img_f is not None:
        neg_selectivities = [
            (float(neg_img_f[mask_d, k].mean()) if mask_d.any() else 0.0)
            / ((float(neg_img_f[mask_o, k].mean()) if mask_o.any() else 0.0) + 1e-6)
            for k in range(neg_img_f.shape[1])
        ]

    importance_scores = extract_importance_scores(
        path_nodes,
        selectivity_weights=np.array(selectivities),
        neg_selectivity_weights=np.array(neg_selectivities) if neg_selectivities is not None else None,
    )

    info = {
        'k_star': k_star,
        'selectivity': selectivities[k_star],
        'all_selectivities': selectivities,
        'is_selective': is_selective,
        'warning': warning,
    }
    return importance_scores, info


def run_ablation_sweep(model_orig, importance_scores, ablation_fractions,
                       test_loader, label_transform, device,
                       method='algo_top', n_random_repeats=10,
                       layer_inputs_list=None):
    """
    For each fraction f in ablation_fractions, zero the top f fraction of
    weights ranked by `importance_scores` (or randomly), then evaluate per-class
    accuracy.

    Parameters
    ----------
    model_orig         : original (unablated) model
    importance_scores  : dict {(l_idx, i, j): float} — higher = more important
    ablation_fractions : list of floats in (0, 1)
    test_loader        : DataLoader
    label_transform    : callable or None
    device             : torch device
    method             : 'algo_top' | 'algo_bottom' | 'random' | (ignored for
                         magnitude/taylor — caller pre-computes importance_scores)
    n_random_repeats   : number of random draws to average when method='random'
    layer_inputs_list  : unused placeholder (kept for API consistency)

    Returns
    -------
    results : dict {fraction: {class_d: accuracy}}
    """
    keys = list(importance_scores.keys())
    vals = np.array([importance_scores[k] for k in keys])

    if method == 'algo_top':
        ranked = [keys[i] for i in np.argsort(vals)[::-1]]
    elif method == 'algo_bottom':
        ranked = [keys[i] for i in np.argsort(vals)]
    elif method == 'random':
        ranked = None  # handled below
    else:
        ranked = [keys[i] for i in np.argsort(vals)[::-1]]

    n_total = len(keys)
    results = {}
    for f in ablation_fractions:
        n_ablate = max(1, int(round(f * n_total)))
        if method == 'random':
            accs_list = []
            for _ in range(n_random_repeats):
                chosen = list(np.random.choice(n_total, n_ablate, replace=False))
                ablated = ablate_model(model_orig, [keys[c] for c in chosen])
                accs_list.append(per_class_accuracy(ablated, test_loader, label_transform, device))
            classes = sorted(accs_list[0].keys())
            results[f] = {d: float(np.mean([a[d] for a in accs_list])) for d in classes}
        else:
            ablated = ablate_model(model_orig, ranked[:n_ablate])
            results[f] = per_class_accuracy(ablated, test_loader, label_transform, device)
    return results
