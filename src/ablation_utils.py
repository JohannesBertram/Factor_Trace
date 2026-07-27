import copy
from dataclasses import dataclass

import numpy as np
import torch


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


def default_prunable_names(model, layer_filter=None):
    """Names of all Conv2d/Linear modules in named_modules() order.

    This is the order collect_layer_dicts captures layers in, so the returned
    list aligns index-for-index with BFT layer_idx when the trace used no
    layer_filter (pass the same layer_filter here otherwise).
    """
    import torch.nn as nn
    return [name for name, mod in model.named_modules()
            if isinstance(mod, (nn.Conv2d, nn.Linear))
            and (layer_filter is None or layer_filter(name, mod))]


def _prunable_modules(model, layer_names=None):
    """Resolve (layer_names, modules); defaults to all Conv2d/Linear modules."""
    if layer_names is None:
        layer_names = default_prunable_names(model)
    return layer_names, [model.get_submodule(n) for n in layer_names]


def _flat_weight(mod):
    """Module weight as a 2-D (n_out, n_in_flat) numpy array (conv kernels flattened)."""
    W = mod.weight.detach().cpu().numpy()
    return W.reshape(W.shape[0], -1)


def all_weight_keys(model, layer_names=None):
    """Return a list of all (layer_idx, i, j) weight keys in model order.

    j indexes the flattened non-output weight dims (conv: C_in*kH*kW), matching
    extract_importance_scores.
    """
    _, mods = _prunable_modules(model, layer_names)
    keys = []
    for l_idx, mod in enumerate(mods):
        n_out, n_in = _flat_weight(mod).shape
        for i in range(n_out):
            for j in range(n_in):
                keys.append((l_idx, i, j))
    return keys


def ablate_model(model_orig, weight_keys_to_zero, layer_names=None):
    """
    Return a deep copy of model_orig with the specified weights set to 0.

    Parameters
    ----------
    weight_keys_to_zero : iterable of (layer_idx, i, j) — j indexes the
        flattened non-output weight dims (conv: C_in*kH*kW)
    layer_names : list[str] or None — module names aligned with layer_idx;
        None resolves to all Conv2d/Linear modules in named_modules() order

    Returns
    -------
    model_copy : nn.Module (eval mode)
    """
    model = copy.deepcopy(model_orig)
    _, mods = _prunable_modules(model, layer_names)
    by_layer = {}
    for (l_idx, i, j) in weight_keys_to_zero:
        ii, jj = by_layer.setdefault(l_idx, ([], []))
        ii.append(i)
        jj.append(j)
    with torch.no_grad():
        for l_idx, (ii, jj) in by_layer.items():
            W = mods[l_idx].weight
            W.view(W.shape[0], -1)[torch.as_tensor(ii), torch.as_tensor(jj)] = 0.0
    model.eval()
    return model


def per_class_accuracy(model, loader, label_transform, device, pred_transform=None):
    """
    Evaluate per-class accuracy on the test set.

    Parameters
    ----------
    model            : nn.Module
    loader           : DataLoader
    label_transform  : callable or None — maps raw labels to task classes
    device           : torch device
    pred_transform   : callable or None — maps argmax predictions to task
                       classes (e.g. ImageNet index -> super-category; return
                       -1 for out-of-task predictions so they count as wrong)

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
            if pred_transform is not None:
                preds = pred_transform(preds.cpu())
            for t, p in zip(target.cpu().numpy(), preds.cpu().numpy()):
                t = int(t)
                total_counts[t] = total_counts.get(t, 0) + 1
                correct_counts[t] = correct_counts.get(t, 0) + int(t == p)
    return {d: correct_counts.get(d, 0) / total_counts[d]
            for d in sorted(total_counts)}


def magnitude_scores(model, layer_names=None):
    """Return {(l_idx, i, j): |W[i,j]|} for all weights in the prunable layers."""
    scores = {}
    _, mods = _prunable_modules(model, layer_names)
    for l_idx, mod in enumerate(mods):
        Wf = np.abs(_flat_weight(mod))
        n_out, n_in = Wf.shape
        for i in range(n_out):
            for j in range(n_in):
                scores[(l_idx, i, j)] = float(Wf[i, j])
    return scores


def act_magnitude_scores(model, layer_inputs_list, layer_names=None):
    """
    Return {(l_idx, i, j): |W[i,j]| * mean_act[j]} where mean_act is the
    mean absolute activation over all samples at the layer's input.

    For conv layers (input_fmap (N, C_in, H, W)) the mean is per input channel,
    broadcast over the kH*kW kernel positions of that channel.
    """
    scores = {}
    _, mods = _prunable_modules(model, layer_names)
    for l_idx, mod in enumerate(mods):
        Wf = np.abs(_flat_weight(mod))
        n_out, n_in = Wf.shape
        fmap = np.abs(np.asarray(layer_inputs_list[l_idx]))
        if fmap.ndim == 4:
            mean_act = np.repeat(fmap.mean(axis=(0, 2, 3)), n_in // fmap.shape[1])
        else:
            mean_act = fmap.reshape(fmap.shape[0], -1).mean(axis=0)
        for i in range(n_out):
            for j in range(n_in):
                scores[(l_idx, i, j)] = float(Wf[i, j] * mean_act[j])
    return scores


def taylor_scores(model, loader, label_transform, device, target_class,
                  layer_names=None, loss_on_raw_labels=False):
    """
    Compute Taylor-criterion importance |grad(L) * W| for class-d samples only.

    Uses a single forward+backward pass.  The loss is cross-entropy over
    class-target_class samples.  The model copy is kept in eval mode so
    BatchNorm/Dropout behave as at test time (gradients still flow).

    Parameters
    ----------
    loss_on_raw_labels : if True, samples are selected by label_transform(target)
        == target_class but the CE loss uses the raw loader labels — for models
        whose output space is finer than the task classes (e.g. 1000 ImageNet
        logits scored on 8 super-categories).

    Returns {(l_idx, i, j): float}
    """
    model_tmp = copy.deepcopy(model)
    model_tmp.eval()
    layer_names, mods = _prunable_modules(model_tmp, layer_names)

    criterion = torch.nn.CrossEntropyLoss()
    model_tmp.zero_grad()
    n_seen = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device)
        t_task = label_transform(target) if label_transform is not None else target
        mask = (t_task == target_class).to(device)
        if not mask.any():
            continue
        result = model_tmp(data[mask])
        output = result[0] if isinstance(result, (tuple, list)) else result
        loss_target = target[mask] if loss_on_raw_labels else t_task.to(device)[mask]
        loss = criterion(output, loss_target)
        loss.backward()
        n_seen += int(mask.sum())

    scores = {}
    for l_idx, mod in enumerate(mods):
        W = mod.weight
        if W.grad is None:
            taylor = np.zeros(_flat_weight(mod).shape)
        else:
            taylor = (W.grad * W).abs().detach().cpu().numpy()
            taylor = taylor.reshape(taylor.shape[0], -1)
        n_out, n_in = taylor.shape
        for i in range(n_out):
            for j in range(n_in):
                scores[(l_idx, i, j)] = float(taylor[i, j])
    return scores


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


def filter_scores_by_layer(scores, layer_indices):
    """Return a new scores dict restricted to the given layer indices.

    Parameters
    ----------
    scores        : dict {(layer_idx, i, j): float}
    layer_indices : iterable of int

    Returns
    -------
    filtered : dict {(layer_idx, i, j): float}
    """
    s = set(layer_indices)
    return {k: v for k, v in scores.items() if k[0] in s}


def run_ablation_sweep(model_orig, importance_scores, ablation_fractions,
                       test_loader, label_transform, device,
                       method='algo_top', n_random_repeats=10,
                       layer_indices=None, layer_names=None, pred_transform=None):
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
    method             : 'algo_top' | 'algo_bottom' | 'random'
    n_random_repeats   : number of random draws to average when method='random'
    layer_indices      : list[int] or None — if set, only weights in these layers
                         are candidates for ablation
    layer_names        : list[str] or None — module names aligned with layer_idx
                         (see ablate_model)
    pred_transform     : callable or None — see per_class_accuracy

    Returns
    -------
    results : dict {fraction: {class_d: accuracy}}
    """
    if layer_indices is not None:
        importance_scores = filter_scores_by_layer(importance_scores, layer_indices)

    keys = list(importance_scores.keys())
    vals = np.array([importance_scores[k] for k in keys])

    if method == 'algo_top':
        ranked = [keys[i] for i in np.argsort(vals)[::-1]]
    elif method == 'algo_bottom':
        ranked = [keys[i] for i in np.argsort(vals)]
    elif method == 'random':
        ranked = None  # handled below
    else:
        raise ValueError(f"unknown method {method!r}")

    n_total = len(keys)
    results = {}
    for f in ablation_fractions:
        n_ablate = max(1, int(round(f * n_total)))
        if method == 'random':
            accs_list = []
            for _ in range(n_random_repeats):
                chosen = list(np.random.choice(n_total, n_ablate, replace=False))
                ablated = ablate_model(model_orig, [keys[c] for c in chosen], layer_names)
                accs_list.append(per_class_accuracy(ablated, test_loader, label_transform,
                                                    device, pred_transform))
            classes = sorted(accs_list[0].keys())
            results[f] = {d: float(np.mean([a[d] for a in accs_list])) for d in classes}
        else:
            ablated = ablate_model(model_orig, ranked[:n_ablate], layer_names)
            results[f] = per_class_accuracy(ablated, test_loader, label_transform,
                                            device, pred_transform)
    return results


def ablation_sweep(
    model, bft_result, test_loader, *,
    target_class,
    fractions=(0.05, 0.10, 0.20, 0.30, 0.50),
    layer_indices=None,
    methods=('bft_top', 'bft_bottom', 'magnitude', 'random'),
    label_transform=None,
    device=None,
    n_random_repeats=10,
    normalize_bft=True,
    layer_inputs_list=None,
    layer_names=None,
    targets=None,
    pred_transform=None,
    taylor_on_raw_labels=False,
    verbose=0,
):
    """Single entrypoint for a pruning experiment on one target class.

    Computes importance scores from BFT and magnitude baselines, then runs
    run_ablation_sweep for each method and returns an AblationResult.

    Parameters
    ----------
    model         : nn.Module — any model whose BFT layer_idx maps onto
                    Conv2d/Linear modules (see layer_names)
    bft_result    : BFTResult from bft()
    test_loader   : DataLoader — evaluation set
    target_class  : int — which class circuit to extract
    fractions     : pruning fractions to sweep
    layer_indices : list[int] or None — restrict pruning to these layer indices
                    (0 = input-side layer, n_layers-1 = output layer).
                    None means all layers.
    methods       : subset of ('bft_top', 'bft_bottom', 'magnitude', 'random',
                    'act_magnitude', 'taylor'). 'act_magnitude' requires
                    layer_inputs_list; 'taylor' uses test_loader + target_class.
    label_transform : callable or None
    device        : torch device or None
    n_random_repeats : repeats for the 'random' method
    normalize_bft : if True, min-max normalise BFT scores per layer before using
                    them for BOTH 'bft_top' and 'bft_bottom' (prevents wide early
                    layers from dominating and keeps top/bottom selection symmetric)
    layer_inputs_list : per-layer inputs (fc: (N, n_in); conv: (N, C_in, H, W)),
                    required for 'act_magnitude'
    layer_names   : list[str] or None — module names aligned with BFT layer_idx.
                    None resolves to all Conv2d/Linear modules in named_modules()
                    order, which is correct whenever the trace hooked every such
                    module. When the trace used a layer_filter (e.g. the SqueezeNet
                    spine), pass [d['name'] for d in collected 'layer_data'].
    targets       : (N,) array or None — task-class labels of the traced samples.
                    None uses bft_result.targets, which is only valid for
                    primary-mode BFT; layer-dict traces return all-zeros targets,
                    so pass the labels of the traced sample set explicitly there.
    pred_transform : callable or None — maps argmax predictions to task classes
                    (see per_class_accuracy); needed when the model's output
                    space is finer than the task classes (e.g. ImageNet
                    super-categories).
    taylor_on_raw_labels : compute the Taylor loss on raw loader labels instead
                    of transformed ones (see taylor_scores).
    verbose       : 0 = silent, 1 = print method progress

    Returns
    -------
    AblationResult
    """
    from .types import BFTResult, AblationResult

    if not isinstance(bft_result, BFTResult):
        raise TypeError('bft_result must be a BFTResult')
    if targets is None:
        targets = bft_result.targets
    targets = np.asarray(targets)

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')

    # Guard the label-space invariant: select_class_circuit compares target_class
    # against the trace targets, while per_class_accuracy/taylor_scores compare
    # against label_transform(target). They only agree if the targets live in the
    # transformed task space (primary mode: trace on the label-transformed loader;
    # layer-dict mode: pass targets= with the traced samples' task labels).
    uniq = set(int(t) for t in np.unique(targets))
    assert target_class in uniq, (
        f"target_class={target_class} not in trace targets {sorted(uniq)}; "
        "trace BFT on the label-transformed loader (primary mode) or pass "
        "targets= (layer-dict mode) so targets match the space used by "
        "per_class_accuracy(label_transform=...).")

    # Alignment guard: every traced node must map onto the module its layer_idx
    # resolves to, else the wrong weights would be pruned silently. Fails for
    # traces over non-module weights (e.g. the ViT attn slice) — those cannot be
    # weight-pruned through this entrypoint.
    _, _mods = _prunable_modules(model, layer_names)
    for nd in bft_result.nodes():
        if (nd.layer_idx >= len(_mods) or
                tuple(_mods[nd.layer_idx].weight.shape) != tuple(nd.weight.shape)):
            raise ValueError(
                f'BFT node at layer_idx={nd.layer_idx} (weight {nd.weight.shape}) does '
                f'not match the resolved prunable module list ({len(_mods)} modules); '
                'pass layer_names= with the module names the trace was collected from '
                "([d['name'] for d in layer_data]).")

    # Compute importance scores
    bft_sc, bft_info = select_class_circuit(bft_result, targets, target_class)
    if bft_info['warning'] and verbose:
        print(f'[ablation_sweep] {bft_info["warning"]}')

    # Use per-layer-normalized BFT scores for BOTH bft_top and bft_bottom so that,
    # when a layer-depth sweep pools layers of different widths/NMF scales, top- and
    # bottom-selection are on the same footing (fixes the raw-vs-normalized asymmetry).
    bft_sc_norm = normalize_scores_per_layer(bft_sc) if normalize_bft else bft_sc
    mag_sc = magnitude_scores(model, layer_names)

    _score_map = {
        'bft_top':    (bft_sc_norm, 'algo_top'),
        'bft_bottom': (bft_sc_norm, 'algo_bottom'),
        'magnitude':  (mag_sc,      'algo_top'),
        'random':     (mag_sc,      'random'),
    }
    if 'act_magnitude' in methods:
        if layer_inputs_list is None:
            raise ValueError("method 'act_magnitude' requires layer_inputs_list=")
        _score_map['act_magnitude'] = (
            act_magnitude_scores(model, layer_inputs_list, layer_names), 'algo_top')
    if 'taylor' in methods:
        _score_map['taylor'] = (
            taylor_scores(model, test_loader, label_transform, device, target_class,
                          layer_names=layer_names,
                          loss_on_raw_labels=taylor_on_raw_labels),
            'algo_top')

    baseline = per_class_accuracy(model, test_loader, label_transform, device,
                                  pred_transform)

    results = {}
    for method in methods:
        if method not in _score_map:
            raise ValueError(f'Unknown method: {method!r}. '
                             f'Choose from {list(_score_map)}.')
        scores, run_method = _score_map[method]
        if verbose:
            print(f'[ablation_sweep] method={method}  layers={layer_indices}')
        results[method] = run_ablation_sweep(
            model, scores, list(fractions),
            test_loader, label_transform, device,
            method=run_method,
            n_random_repeats=n_random_repeats,
            layer_indices=layer_indices,
            layer_names=layer_names,
            pred_transform=pred_transform,
        )

    all_layer_indices = (layer_indices if layer_indices is not None
                         else sorted({k[0] for k in bft_sc}))

    return AblationResult(
        results=results,
        baseline=baseline,
        target_class=target_class,
        layer_indices=all_layer_indices,
        bft_info=bft_info,
    )


def ablation_layer_sweep(
    model, bft_result, test_loader, *,
    target_class,
    fractions=(0.10, 0.20, 0.30),
    methods=('bft_top', 'bft_bottom', 'magnitude', 'random'),
    label_transform=None,
    device=None,
    n_random_repeats=10,
    normalize_bft=True,
    layer_inputs_list=None,
    layer_names=None,
    targets=None,
    pred_transform=None,
    taylor_on_raw_labels=False,
    verbose=0,
):
    """Sweep over layer depth: run ablation_sweep for last 1, 2, …, L layers.

    At depth d, the pruning set contains the last d layers (output-side first).
    For a 3-layer network: depth=1 prunes only layer 2, depth=2 prunes layers
    1 and 2, depth=3 prunes all layers.

    Parameters
    ----------
    (same as ablation_sweep, minus layer_indices)

    Returns
    -------
    dict {depth: AblationResult}
        depth 1 = last layer only, depth n_layers = all layers
    """
    from .types import BFTResult

    if not isinstance(bft_result, BFTResult):
        raise TypeError('bft_result must be a BFTResult')

    # Compute BFT scores once to determine which layer indices exist
    if targets is None:
        targets = bft_result.targets
    bft_sc, _ = select_class_circuit(bft_result, np.asarray(targets), target_class)
    all_layers = sorted({k[0] for k in bft_sc})
    n_layers = len(all_layers)

    layer_sweep = {}
    for depth in range(1, n_layers + 1):
        # last `depth` layers (output-side)
        layer_indices = all_layers[-depth:]
        if verbose:
            print(f'[ablation_layer_sweep] depth={depth}  layers={layer_indices}')
        layer_sweep[depth] = ablation_sweep(
            model, bft_result, test_loader,
            target_class=target_class,
            fractions=fractions,
            layer_indices=layer_indices,
            methods=methods,
            label_transform=label_transform,
            device=device,
            n_random_repeats=n_random_repeats,
            normalize_bft=normalize_bft,
            layer_inputs_list=layer_inputs_list,
            layer_names=layer_names,
            targets=targets,
            pred_transform=pred_transform,
            taylor_on_raw_labels=taylor_on_raw_labels,
            verbose=max(0, verbose - 1),
        )

    return layer_sweep
