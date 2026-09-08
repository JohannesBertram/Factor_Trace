"""Causal pruning validation — the per-class ablation sweep driver.

Wraps ``ablation_sweep`` (src.ablation_utils) into a per-model driver and lifts the
observation packing out of the notebooks so a model notebook's pruning section is a
single call. An *observation* is one (seed, target-class) pruning curve;
``run_pruning`` produces the list of them across the given trained replicates and
target classes. All aggregation and significance testing lives in
``src.bundles.pruning_bundle``, the single place the plotted stats are computed.
"""
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


def pruning_results_dict(experiment, run_result, fractions=DEFAULT_FRACTIONS,
                         frac_stat=0.20, methods=DEFAULT_METHODS, class_names=None):
    """Shape a run_pruning() result into the results-JSON schema that
    `scripts/build_pruning_bundle.py` consumes (works for both build paths).

    Written to data/results/<name>.json with mode='cluster'; the build script then
    re-encodes it into the figdata bundle the pruning panels read."""
    per_obs = run_result['per_obs']
    nc = len(per_obs[0]['baseline']) if per_obs else 0
    if class_names is None:
        class_names = {str(i): str(i) for i in range(nc)}
    else:
        class_names = {str(i): str(class_names[i]) for i in range(nc)}
    return {'experiment': f'{experiment}_pruning', 'mode': 'cluster',
            'per_obs': per_obs,
            'config': {'fractions': list(fractions), 'frac_stat': frac_stat,
                       'methods': list(methods), 'class_names': class_names}}


def run_pruning(replicates, eval_loader, target_classes, *,
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

    Returns {'per_obs': [...]}; feed it to ``src.bundles.pruning_bundle`` (stats)
    or ``pruning_results_dict`` (results JSON).
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
    return {'per_obs': per_obs}
