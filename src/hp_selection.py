"""Hyperparameter (per-layer rank) selection by held-out arbor reconstruction.

The C0 procedure (notebook 15), as a library. Selects each layer's NMF rank as the
smallest K whose *held-out* arbor reconstruction R² is within ``plateau_eps`` of that
layer's best, then imposes three structural constraints so the circuit tree exposes
the class structure:

  1. output-layer rank >= n_classes + last_extra;
  2. every output factor is traced (n_branches[out] = k_max[out]);
  3. every output factor splits one layer back (n_branches[out-1] >= 2).

The rule reads no fingerprint / separability metric, so selecting on it cannot
inflate the reported separability. Held-out ≈ in-sample in practice (the arbor NMF
does not overfit at these ranks); the held-out split is what *demonstrates* that.

``select_ranks`` consumes arbors already built by the caller (via ``src.arbors``),
keeping this module free of model/context specifics.
"""
import numpy as np
from sklearn.decomposition import MiniBatchNMF

from .bft import _safe_init


def stratified_split(y, frac, seed=0):
    """Disjoint (fit, holdout) row indices, stratified so every class is in both."""
    rng = np.random.default_rng(seed)
    fit, ho = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_ho = max(1, int(round(frac * len(idx)))) if len(idx) > 1 else 0
        ho.extend(idx[:n_ho]); fit.extend(idx[n_ho:])
    return np.sort(np.array(fit, int)), np.sort(np.array(ho, int))


def _mb_nmf(X, K, max_iter):
    n_s, n_f = X.shape
    return MiniBatchNMF(n_components=K, init=_safe_init(K, n_s, n_f), random_state=0,
                        max_iter=max_iter, batch_size=max(64, min(1024, n_s // 4)),
                        tol=1e-3, max_no_improvement=5, l1_ratio=0)


def _k_grid(cap, n_cols, wide_thresh=100_000):
    if n_cols < wide_thresh:
        return list(range(1, cap + 1))
    coarse = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24]
    return [k for k in coarse if k <= cap] or [cap]


def heldout_curve(X, y, cap, max_iter=300, holdout_frac=0.30, sweep_rows=900,
                  max_cols=None, split_seed=0):
    """Per-K in-sample and held-out arbor reconstruction R² on one positive arbor.

    Returns (rows, meta). Each row: {K, heldout_r2, insample_r2, heldout_rel_err}.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    rng = np.random.default_rng(split_seed)
    sub_cols = None
    if max_cols and X.shape[1] > max_cols:
        cols = np.sort(rng.choice(X.shape[1], max_cols, replace=False))
        X, sub_cols = X[:, cols], int(max_cols)
    if X.shape[0] > sweep_rows:
        keep, _ = stratified_split(y, 1.0 - sweep_rows / X.shape[0], split_seed)
        X, y = X[keep], y[keep]

    fit, ho = stratified_split(y, holdout_frac, split_seed)
    if len(ho) < 4 or len(fit) < 4:
        return [], {'sub_cols': sub_cols, 'n_fit': len(fit), 'n_ho': len(ho),
                    'skipped': 'too few rows to split'}
    Xf, Xh = X[fit], X[ho]
    ss_h = float(((Xh - Xh.mean()) ** 2).sum())
    ss_f = float(((Xf - Xf.mean()) ** 2).sum())
    rows = []
    for K in _k_grid(min(cap, min(Xf.shape) - 1), X.shape[1]):
        if K < 1:
            continue
        m = _mb_nmf(Xf, K, max_iter)
        Wf = m.fit_transform(Xf)
        Wh = m.transform(Xh)
        res_h = float(((Xh - Wh @ m.components_) ** 2).sum())
        res_f = float(((Xf - Wf @ m.components_) ** 2).sum())
        rows.append({'K': int(K),
                     'heldout_r2':  1.0 - res_h / ss_h if ss_h > 0 else float('nan'),
                     'insample_r2': 1.0 - res_f / ss_f if ss_f > 0 else float('nan'),
                     'heldout_rel_err': float(np.sqrt(res_h) / (np.linalg.norm(Xh) + 1e-12))})
    return rows, {'sub_cols': sub_cols, 'n_fit': int(len(fit)), 'n_ho': int(len(ho)),
                  'n_cols': int(X.shape[1])}


def pool_curves(curves):
    """Average several nodes' curves at each K they share."""
    if not curves:
        return []
    common = sorted(set.intersection(*[{r['K'] for r in c} for c in curves]))
    out = []
    for K in common:
        vals = [[r for r in c if r['K'] == K][0] for c in curves]
        out.append({'K': K,
                    'heldout_r2':  float(np.mean([v['heldout_r2'] for v in vals])),
                    'insample_r2': float(np.mean([v['insample_r2'] for v in vals])),
                    'n_nodes': len(vals)})
    return out


def pick_K(curve, plateau_eps=0.01, r2_targets=(0.90, 0.95)):
    """Smallest K within ``plateau_eps`` of the best held-out R², plus alternatives."""
    if not curve:
        return {'K': None, 'reason': 'empty curve'}
    r2 = np.array([r['heldout_r2'] for r in curve], float)
    Ks = np.array([r['K'] for r in curve], int)
    best = float(np.nanmax(r2))
    within = Ks[r2 >= best - plateau_eps]
    out = {'K': int(within.min()), 'best_heldout_r2': best,
           'K_argmax': int(Ks[int(np.nanargmax(r2))]),
           'selected_at_sweep_cap': bool(int(within.min()) == int(Ks.max())),
           'still_rising': bool(len(r2) >= 2 and (r2[-1] - r2[-2]) > plateau_eps)}
    for t in r2_targets:
        hit = Ks[r2 >= t]
        out[f'K_at_r2_{int(t * 100)}'] = int(hit.min()) if len(hit) else None
    return out


def build_profile(k_sel, n_classes, layer_ids, last_extra=4, b_second_last=2,
                  cap_hit=()):
    """Assemble (k_max, n_branches) from per-layer K*, applying the 3 constraints."""
    k_sel = [int(k) if k else 1 for k in k_sel]
    k_max = list(k_sel)
    notes = []
    L = len(layer_ids) - 1
    floor = int(n_classes + last_extra)
    output_floor_applied = k_max[L] < floor
    if output_floor_applied:
        notes.append(f'output k_max {k_max[L]} -> {floor} (n_classes {n_classes} + {last_extra})')
        k_max[L] = floor
    else:
        notes.append(f'output k_max {k_max[L]} from criterion (>= floor {floor})')
    n_branches = [1] * len(layer_ids)
    n_branches[L] = k_max[L]
    notes.append(f'n_branches[out] = k_max[out] = {k_max[L]} (every output factor traced)')
    if L - 1 >= 0:
        n_branches[L - 1] = max(2, int(b_second_last))
        notes.append(f'n_branches[out-1] = {n_branches[L - 1]} (every circuit splits)')
    return {'k_max': k_max, 'n_branches': n_branches, 'k_from_criterion': k_sel,
            'notes': notes, 'output_floor_applied': bool(output_floor_applied),
            'k_cap_hit': list(cap_hit)}


def select_ranks(arbors_by_layer, labels, k_cap, *, n_classes, last_extra=4,
                 b_second_last=2, max_iter=300, holdout_frac=0.30, sweep_rows=900,
                 max_cols=None, plateau_eps=0.01, split_seed=0):
    """Full held-out rank selection over all layers.

    Parameters
    ----------
    arbors_by_layer : {layer_idx: [positive arbor (N, features), ...]}  (1+ nodes/layer)
    labels          : (N,) class labels aligned to the arbor rows
    k_cap           : sweep ceiling
    n_classes       : for the output-rank floor

    Returns a dict with per-layer sweep + selection, and the assembled ``profile``.
    """
    labels = np.asarray(labels)
    layer_ids = sorted(arbors_by_layer)
    sweep, k_sel, cap_hit = {}, [], []
    for li in layer_ids:
        curves = []
        for X in arbors_by_layer[li]:
            keep = np.where(np.abs(X).sum(1) > 0)[0]
            if len(keep) < 8:
                continue
            c, _ = heldout_curve(X[keep], labels[keep], k_cap, max_iter, holdout_frac,
                                 sweep_rows, max_cols, split_seed)
            if c:
                curves.append(c)
        pooled = pool_curves(curves)
        sel = pick_K(pooled, plateau_eps)
        sweep[li] = {'curve': pooled, 'selection': sel, 'n_nodes': len(curves)}
        k_sel.append(sel.get('K'))
        if sel.get('selected_at_sweep_cap') or sel.get('still_rising'):
            cap_hit.append(li)
    profile = build_profile(k_sel, n_classes, layer_ids, last_extra, b_second_last, cap_hit)
    return {'sweep': sweep, 'profile': profile, 'layer_ids': layer_ids}
