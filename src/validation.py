"""Full validation suite → the nb09 results schema, as a library.

Assembles the appendix validation figure's data (figP_validation) for one model
from its already-built trees, reusing the existing src pieces:
  - faithfulness: causal reconstruction (from a validate=True circuit tree),
    NNLS round-trip fidelity, NMF init-stability + K*±1 sensitivity;
  - class-relevant structure: fingerprint vs activation separability (native,
    PCA-matched, random-projection null, shuffled-label null) and the weight-term
    control (arbor-NMF vs activation-NMF);
  - rank robustness: the per-layer arbor-R² sweep (FU1).

`run_validation(...)` returns a dict in the exact schema `build_validation_bundles.py`
re-encodes into `nb09_<exp>_validation`, so a model notebook writes it to
`logs/results/nb09_<exp>.json` and the existing build script + figP work unchanged.

Recon-control tick marks (exact ceiling / random-rank floor) are computed only at
the output fc node (where they are well defined without the traced connection
weights); deeper controls are left out and figP simply omits those ticks.
"""
import numpy as np

from .arbors import node_pos_arbor, activation_matrix, nodes_by_layer, nodes_per_layer
from .robustness_utils import compute_nmf_stability, compute_k_sensitivity
from .recon_validation import summarize_validation, reconstruct_preactivation
from .bft import run_nmf_minibatch, full_nmf_pipeline
from .fingerprint_utils import project_stimuli_onto_tree, extract_fingerprint_matrix
from . import separability as _sep


# ── faithfulness: causal reconstruction (read off the validate=True tree) ──────

def _recon_from_tree(tree):
    """recon dict {per_node, overall, per_layer} from a tree traced with validate=True.

    Returns None when no node carries a causal-reconstruction result (conv/attn or
    a tree built with validate=False)."""
    summ = summarize_validation(tree.nodes())
    if summ is None:
        return None
    per_node = [{'layer_idx': int(d['layer_idx']), 'preact_r2': float(d['preact_r2'])}
                for d in summ['individual'] if d.get('preact_r2') is not None]
    return {'per_node': per_node, 'overall': summ['overall'],
            'per_layer': {str(k): v for k, v in summ['per_layer'].items()}}


def _recon_controls_root(tree, layer_inputs, n_seeds=3, max_iter=300, seed=0):
    """Exact-ceiling and random-rank-floor preact-R² at the output fc node.

    Well defined only where the traced connection weights are uniform (the root),
    so we compute it there; figP reads just `exact` and `random_R`."""
    root = tree.root
    if root.layer_type != 'fc':
        return {}
    li = root.layer_idx
    W = root.weight
    act = layer_inputs[li]
    z_true = act @ W.T
    ss_tot = float(((z_true - z_true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return {}
    K = int(root.img_factors.shape[1])
    Xpos = node_pos_arbor(root, act)
    node_id = f'L{li}:{root.layer_name}:root'.replace('.', '_')
    ctrl = {'exact': {'preact_r2': 1.0}}
    # random-rank floor: NMF at K from random inits, reconstruct, median preact-R²
    r2s = []
    rng = np.random.default_rng(seed)
    for s in range(n_seeds):
        W_s, H_s, _ = run_nmf_minibatch(Xpos, K, random_state=int(rng.integers(1 << 30)),
                                        init='random', max_iter=max_iter)
        Xhat = (W_s @ H_s.T).reshape(len(act), W.shape[0], W.shape[1]).sum(2)  # (N, n_out)
        # de-normalize: root arbor is unnormalized (pi=1), so Xhat is already z-scale
        ss_res = float(((z_true - Xhat) ** 2).sum())
        r2s.append(1.0 - ss_res / ss_tot)
    ctrl['random_R'] = {'preact_r2': float(np.median(r2s))}
    return {node_id: ctrl}


# ── faithfulness: NNLS round-trip fidelity ─────────────────────────────────────

def _roundtrip_hist(tree, layer_inputs, bins=30):
    """Histogram + summary of per-stimulus NNLS re-projection cosine.

    Re-projects the traced population onto the fixed factors and measures how well
    each stimulus's recovered fingerprint matches its original. Skipped (None) when
    the tree has an attn node (projection is not defined through attention)."""
    if any(nd.layer_type == 'attn' for nd in tree.nodes()):
        return None
    try:
        proj = project_stimuli_onto_tree(tree, layer_inputs)
    except Exception:
        return None
    n = tree.root.img_factors.shape[0]
    F0 = extract_fingerprint_matrix(tree, np.arange(n))
    F1 = extract_fingerprint_matrix(proj, np.arange(n))
    u = lambda X: X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cos = (u(F0) * u(F1)).sum(1)
    counts, edges = np.histogram(cos, bins=bins)
    return {'counts': counts.tolist(), 'bin_edges': edges.tolist(), 'n': int(cos.size),
            'mean': float(cos.mean()), 'std': float(cos.std()),
            'min': float(cos.min()), 'max': float(cos.max()),
            'q25': float(np.percentile(cos, 25)), 'median': float(np.percentile(cos, 50)),
            'q75': float(np.percentile(cos, 75))}


# ── faithfulness: NMF stability + K*±1 sensitivity ─────────────────────────────

def _stability(tree, layer_inputs, n_seeds=5, max_iter=300, sub=800):
    nbl = nodes_by_layer(tree)
    per_layer, ksens = {}, {'k_star': [], 'k_minus1': [], 'k_plus1': []}
    rng = np.random.default_rng(0)
    for li in sorted(nbl):
        nd = nbl[li]
        X = node_pos_arbor(nd, layer_inputs[li])
        if X.shape[0] > sub:
            X = X[rng.choice(X.shape[0], sub, replace=False)]
        k = int(nd.img_factors.shape[1])
        sim, _ = compute_nmf_stability(X, k, n_seeds=n_seeds, max_iter=max_iter)
        m = sim[~np.eye(len(sim), dtype=bool)]
        per_layer[str(li)] = {'mean': float(m.mean()), 'std': float(m.std()), 'k': k}
        ks, _ = compute_k_sensitivity(X, k, n_seeds=max(2, n_seeds // 2), max_iter=max_iter)
        _m = lambda a: float(np.nanmean(a)) if a is not None else float('nan')
        ksens['k_star'].append(_m(ks.get('k_star')))
        ksens['k_minus1'].append(_m(ks.get('k_minus1')))
        ksens['k_plus1'].append(_m(ks.get('k_plus1')))
    return {'per_layer': per_layer, 'k_sensitivity': ksens}


# ── class-relevant structure: separability with all baselines ──────────────────

def _sep_block(fp_tree, layer_inputs, labels, seed=0):
    """{bft_fingerprint, raw_activations, bft_matched, act_matched, act_randproj}."""
    y = np.asarray(labels).astype(int)
    n = fp_tree.root.img_factors.shape[0]
    fp = _sep.fingerprint_slices(fp_tree, n, with_per_layer=False)['full']
    acts = _sep.activation_reps(layer_inputs)['full']
    d = int(min(fp.shape[1], acts.shape[1]))
    out = {'bft_fingerprint': _mk(_sep.metrics(fp, y)),
           'raw_activations': _mk(_sep.metrics(acts, y)),
           'bft_matched': _mk(_sep.metrics(_sep._match(fp, d, 'pca'), y)),
           'act_matched': _mk(_sep.metrics(_sep._match(acts, d, 'pca'), y)),
           'act_randproj': _mk(_sep.metrics(_sep._match(acts, d, 'grp', seed), y))}
    return out, fp.shape[1], acts.shape[1], d


def _mk(m):
    return {'silhouette': m['sil'], 'knn_acc': m['knn']}


def _separability(fp_tree, layer_inputs, labels_task, labels_fine, seed=0):
    by_task, fpd, actd, md = _sep_block(fp_tree, layer_inputs, labels_task, seed)
    by_fine, _, _, _ = _sep_block(fp_tree, layer_inputs, labels_fine, seed)
    # shuffled-label null on the fingerprint
    n = fp_tree.root.img_factors.shape[0]
    fp = _sep.fingerprint_slices(fp_tree, n, with_per_layer=False)['full']
    rng = np.random.default_rng(seed + 7)
    yl = rng.permutation(np.asarray(labels_fine).astype(int))
    null = _sep.metrics(fp, yl)
    return {'by_task': by_task, 'by_fine': by_fine,
            'dims': {'fingerprint': int(fpd), 'activations': int(actd), 'matched': int(md)},
            'null_shuffled_labels': {'silhouette': null['sil'], 'knn_acc': null['knn']}}


# ── rank robustness: FU1 per-layer arbor-R² sweep ──────────────────────────────

def _fu1(tree, layer_inputs, cap_extra=4, max_iter=200):
    nbl = nodes_by_layer(tree)
    per_layer = {}
    for li in sorted(nbl):
        nd = nbl[li]
        X = node_pos_arbor(nd, layer_inputs[li]).astype(np.float32)
        kdef = int(nd.img_factors.shape[1])
        cap = int(min(12, X.shape[1], max(8, kdef + cap_extra)))
        ss = float(((X - X.mean()) ** 2).sum())
        sweep = []
        for K in range(1, cap + 1):
            W, H, _ = full_nmf_pipeline(X, K, init='random', max_iter=max_iter)
            Xhat = W @ H.T
            r2 = 1.0 - float(((X - Xhat) ** 2).sum()) / ss if ss > 0 else float('nan')
            sweep.append({'K': int(K), 'recon_r2': float(r2)})
        per_layer[str(li)] = {'default_k': kdef, 'cap': cap, 'sweep': sweep}
    return {'r2_targets': [0.90, 0.95], 'per_layer': per_layer}


# ── top-level ─────────────────────────────────────────────────────────────────

def run_validation(exp, circuit_tree, fp_tree, layer_inputs, labels_task, labels_fine,
                   *, caps=None, stab_seeds=5, seed=0):
    """Assemble the full nb09-format validation dict for one model.

    circuit_tree : C0 tree (traced with validate=True for causal recon on fc layers).
    fp_tree      : the fingerprint tree (paper HPs) — separability/weight-term.
    layer_inputs : per-layer input feature maps aligned to circuit_tree's rows.
    labels_task/labels_fine : class labels (task = trained label, fine = e.g. digit).
    """
    caps = caps or {}
    recon = _recon_from_tree(circuit_tree)
    # In the model notebooks both trees are traced on the same population, so the
    # circuit layer inputs align to the fp tree's rows too.
    fp_li = layer_inputs
    out = {
        'experiment': exp, 'mode': 'cluster', 'notebook': '01-05-merged',
        'caps': {'recon': recon is not None,
                 'roundtrip': not any(n.layer_type == 'attn' for n in circuit_tree.nodes()),
                 **{k: caps[k] for k in caps}},
        'stability': _stability(circuit_tree, layer_inputs, n_seeds=stab_seeds),
        'recon': recon,
        'recon_controls': _recon_controls_root(circuit_tree, layer_inputs, seed=seed),
        'roundtrip': _roundtrip_hist(circuit_tree, layer_inputs),
        'separability': _separability(fp_tree, fp_li, labels_task, labels_fine, seed),
        'A1_weight_vs_activation': {
            'fingerprint_separability': {
                'arbor_nmf': _mk(_wt(fp_tree, fp_li, labels_fine)['arbor_nmf']),
                'activation_nmf': _mk(_wt(fp_tree, fp_li, labels_fine)['activation_nmf'])}},
        'FU1_rank_sweep': _fu1(circuit_tree, layer_inputs),
    }
    return out


def _wt(fp_tree, layer_inputs, labels):
    return _sep.weight_term_control(fp_tree, labels, layer_inputs)


def _fp_layer_inputs(fp_tree, circuit_layer_inputs):
    """Layer inputs aligned to the FP tree. If the fp tree was traced on the same
    population (same n), reuse; else fall back to circuit inputs (separability then
    uses whatever the fp tree stored). Callers pass matching inputs where possible."""
    n_fp = fp_tree.root.img_factors.shape[0]
    if circuit_layer_inputs and len(circuit_layer_inputs[0]) == n_fp:
        return circuit_layer_inputs
    return circuit_layer_inputs
