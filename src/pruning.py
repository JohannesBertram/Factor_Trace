"""Causal pruning validation — the per-class ablation sweep + aggregation.

Wraps ``ablation_sweep`` (src.ablation_utils) into a per-model driver and lifts the
observation packing, aggregation and paired significance tests out of notebooks
13/14 so a model notebook's pruning section is a single call.

An *observation* is one (seed, target-class) pruning curve. ``run_pruning`` produces
a list of them across the given trained replicates and target classes; ``aggregate``
turns that list into the plot arrays (per-method target/bystander accuracy vs pruned
fraction) and the specificity / vs-baseline tests.
"""
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

from .ablation_utils import ablation_sweep

DEFAULT_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
DEFAULT_METHODS = ('bft_top', 'bft_bottom', 'random')


def pack_obs(ab, seed, target):
    """AblationResult -> one observation record."""
    return {'seed': int(seed), 'target_class': int(target),
            'baseline': {int(k): float(v) for k, v in ab.baseline.items()},
            'bft_info': {k: ab.bft_info.get(k) for k in
                         ('k_star', 'selectivity', 'is_selective', 'warning')},
            'curves': {m: {float(f): {int(c): float(a) for c, a in per.items()}
                           for f, per in fr.items()}
                       for m, fr in ab.results.items()}}


def obs_curves(o, method, n_classes, fractions):
    """(target_accs, bystander_accs) over [0]+fractions for one observation."""
    d = o['target_class']
    others = [c for c in range(n_classes) if c != d]
    cur = o['curves'][method]
    t = [o['baseline'][d]] + [cur[f][d] for f in fractions]
    b = [np.mean([o['baseline'][c] for c in others])] + \
        [np.mean([cur[f][c] for c in others]) for f in fractions]
    return np.array(t), np.array(b)


def aggregate(per_obs, n_classes, fractions=DEFAULT_FRACTIONS, methods=DEFAULT_METHODS,
              frac_stat=0.20):
    """Plot arrays + paired tests from a list of observations (notebook 13/14 §3)."""
    fractions = list(fractions)
    agg = {'fractions': [0.0] + fractions, 'n_obs': len(per_obs), 'frac_stat': frac_stat,
           'n_classes': n_classes, 'methods': {}}
    for m in methods:
        T = np.stack([obs_curves(o, m, n_classes, fractions)[0] for o in per_obs])
        B = np.stack([obs_curves(o, m, n_classes, fractions)[1] for o in per_obs])
        agg['methods'][m] = {'target_mean': T.mean(0), 'target_sd': T.std(0),
                             'bystander_mean': B.mean(0), 'bystander_sd': B.std(0),
                             'target_all': T, 'bystander_all': B}
    drops = {m: {'target': [], 'bystander': [], 'auc': []} for m in methods}
    for o in per_obs:
        d = o['target_class']
        others = [c for c in range(n_classes) if c != d]
        for m in methods:
            cur = o['curves'][m]
            drops[m]['target'].append(o['baseline'][d] - cur[frac_stat][d])
            drops[m]['bystander'].append(np.mean(
                [o['baseline'][c] - cur[frac_stat][c] for c in others]))
            drops[m]['auc'].append(np.mean(
                [o['baseline'][d] - cur[f][d] for f in fractions]))
    agg['drops'] = drops
    tests = {}
    t_bt, b_bt = np.array(drops['bft_top']['target']), np.array(drops['bft_top']['bystander'])
    if len(t_bt) >= 2 and np.any(t_bt != b_bt):
        w, p = wilcoxon(t_bt, b_bt)
        tests['bft_top_target_vs_bystander'] = {'wilcoxon_stat': float(w), 'p': float(p)}
    for m in methods:
        if m == 'bft_top' or len(per_obs) < 2:
            continue
        t, p = ttest_rel(drops['bft_top']['auc'], drops[m]['auc'])
        tests[f'bft_top_vs_{m}_target_auc'] = {'t': float(t), 'p': float(p)}
    agg['tests'] = tests
    return agg


def pruning_results_dict(experiment, run_result, fractions=DEFAULT_FRACTIONS,
                         frac_stat=0.20):
    """Shape a run_pruning() result into the nb13/nb14 results-JSON schema that
    `scripts/build_pruning_bundle.py` consumes (works for both build paths).

    Written to data/results/<name>.json with mode='cluster'; the build script then
    re-encodes it into the figdata bundle the pruning panels read."""
    agg = run_result['aggregate']
    return {'experiment': f'{experiment}_pruning', 'mode': 'cluster',
            'per_obs': run_result['per_obs'],
            'aggregate': {'methods': agg['methods'], 'fractions': agg['fractions'],
                          'n_obs': agg['n_obs']},
            'stats': {'frac_stat': agg['frac_stat'], 'drops': agg['drops'],
                      'tests': agg['tests'], 'n_obs': agg['n_obs']},
            'config': {'fractions': list(fractions), 'frac_stat': frac_stat}}


def run_pruning(replicates, eval_loader, target_classes, *, n_classes,
                fractions=DEFAULT_FRACTIONS, methods=DEFAULT_METHODS,
                label_transform=None, device=None, layer_indices=None,
                n_random_repeats=10, frac_stat=0.20, verbose=1):
    """Run the pruning sweep over trained replicates × target classes.

    Parameters
    ----------
    replicates : list of dict, each {'seed', 'model', 'tree', 'layer_names', 'targets'} —
                 one trained model, its BFT circuit tree, the prunable layer names
                 (from collect_layer_dicts), and the traced samples' task labels.
                 ``targets`` is required for layer-dict traces (their
                 ``bft_result.targets`` is all-zeros); primary-mode traces may omit it.
    eval_loader : DataLoader — held-out set scored per class after each prune.
    target_classes : which class circuits to prune.
    layer_indices : restrict pruning to these layer_idx (None = all). Pass a subset
                    for per-layer sparsity experiments.

    Returns {'per_obs': [...], 'aggregate': {...}}.
    """
    per_obs = []
    for rep in replicates:
        for d in target_classes:
            ab = ablation_sweep(rep['model'], rep['tree'], eval_loader, target_class=d,
                                fractions=fractions, methods=methods,
                                label_transform=label_transform, device=device,
                                layer_indices=layer_indices,
                                layer_names=rep.get('layer_names'),
                                targets=rep.get('targets'),
                                n_random_repeats=n_random_repeats, verbose=0)
            per_obs.append(pack_obs(ab, rep['seed'], d))
            if verbose:
                td = ab.baseline[d] - ab.results['bft_top'][frac_stat][d]
                print(f'  seed {rep["seed"]} class {d}: baseline={ab.baseline[d]:.3f} '
                      f'bft_top drop@{frac_stat}={td:+.3f}'
                      + ('' if ab.bft_info.get('is_selective') else '  [no selective factor]'))
    agg = aggregate(per_obs, n_classes, fractions, methods, frac_stat)
    return {'per_obs': per_obs, 'aggregate': agg}
