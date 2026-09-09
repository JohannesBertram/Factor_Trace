"""Figure-bundle builders — turn analysis results into figdata bundles in place.

The model notebooks (01-05) call these at the end of their validation (§7) and
pruning (§8) sections so that a single notebook run writes *every* figure input
into ``figures/figdata/`` — nothing to convert afterwards; only ``figdata/``
needs to move off the cluster. ``scripts/build_validation_bundles.py`` and
``scripts/build_pruning_bundle.py`` wrap the same functions for rebuilding a
bundle from a stored results JSON.
"""
import json

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

from . import figdata

# what the paper calls each experiment, and the architecture line under the title
VALIDATION_LABELS = {
    'mlp_even_odd':  ('MLP even/odd', r'$784\to8\to4\to2$, MNIST parity'),
    'mlp_digit':     ('MLP digits', r'$784\to40\to20\to10$, MNIST digits'),
    'cnn_cifar':     ('CNN', 'SmallCNN, CIFAR-10'),
    'vit_mnist':     ('ViT', 'TinyViT $d{=}32$, MNIST parity'),
    'imagenet_cnn':  ('ImageNet CNN', 'SqueezeNet 1.1 spine, ImageNet (8 categories)'),
}

# exp -> (bundle name, n_classes, minimum observations for a committable run)
PRUNING_BUNDLES = {
    'mlp_even_odd': ('nb13_pruning_mlp_even_odd', 2, 10),
    'mlp_digit':    ('nb14_pruning_mlp_digit', 10, 10),
    'cnn_cifar':    ('nb14_pruning_cnn_cifar', 10, 10),
    # 7 accepts a preliminary 7/8-category run; the floor still rejects smoke runs
    'imagenet_cnn': ('nb14_pruning_imagenet_cnn', 8, 7),
}


def _sanitize(obj):
    """figdata flattens nested keys on '.', so no dict key may contain one."""
    if isinstance(obj, dict):
        return {str(k).replace('.', '_'): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def validation_bundle(exp, results, source=None):
    """Re-encode one nb09-schema validation results dict into its figdata bundle.

    ``results`` is exactly what ``run_validation`` returns (or the parsed
    ``logs/results/nb09_<exp>.json``). Pure re-encoding: nothing recomputed.
    """
    D = dict(results)
    D.pop('figures', None)
    D = _sanitize(D)
    D['label'], D['arch'] = VALIDATION_LABELS[exp]
    if source:
        D['source_json'] = str(source)
    figdata.save(f'nb09_{exp}_validation', D)
    return f'nb09_{exp}_validation'


def _curves(o, method, fractions, n_classes):
    """(target, bystander) accuracy over [0]+fractions for one observation."""
    d = int(o['target_class'])
    others = [c for c in range(n_classes) if c != d]
    cur = o['curves'][method]
    t = [o['baseline'][str(d)]] + [cur[str(f)][str(d)] for f in fractions]
    b = [np.mean([o['baseline'][str(c)] for c in others])] + \
        [np.mean([cur[str(f)][str(c)] for c in others]) for f in fractions]
    return np.array(t), np.array(b)


def pruning_bundle(exp, results, min_obs=None):
    """Build the pruning figdata bundle from a nb13/nb14-schema results dict.

    Aggregate and tests are recomputed from ``per_obs`` (so partial runs with a
    checkpointed per_obs work); returns the bundle name, or None when the run is
    below the committable-observation floor (protects committed bundles from
    smoke runs).
    """
    name, n_classes, floor = PRUNING_BUNDLES[exp]
    if min_obs is not None:
        floor = min_obs
    # normalize key types: in-memory dicts carry int/float keys, JSON strings
    obs = json.loads(json.dumps(results['per_obs']))
    if len(obs) < floor:
        print(f'  [pruning_bundle] {exp}: only {len(obs)} observations '
              f'(< {floor}) — bundle NOT written')
        return None
    cfg = results['config']
    fractions = [float(f) for f in cfg['fractions']]
    frac_stat = float(cfg['frac_stat'])
    fi_stat = fractions.index(frac_stat)
    meths = list(cfg['methods'])

    def _p(test, a, b):
        try:
            with np.errstate(invalid='ignore'):
                return float(test(a, b).pvalue)
        except ValueError:
            return float('nan')

    methods, drops = {}, {}
    for m in meths:
        T = np.stack([_curves(o, m, fractions, n_classes)[0] for o in obs])
        B = np.stack([_curves(o, m, fractions, n_classes)[1] for o in obs])
        methods[m] = {'target_mean': T.mean(0).astype(np.float32),
                      'target_sd': T.std(0).astype(np.float32),
                      'bystander_mean': B.mean(0).astype(np.float32),
                      'bystander_sd': B.std(0).astype(np.float32),
                      'target_all': T.astype(np.float32),
                      'bystander_all': B.astype(np.float32)}
        drops[m] = {'target': (T[:, 0] - T[:, 1 + fi_stat]).astype(np.float32),
                    'bystander': (B[:, 0] - B[:, 1 + fi_stat]).astype(np.float32),
                    'auc': (T[:, 0] - T[:, 1:].mean(1)).astype(np.float32)}

    tests = {'p_target_vs_bystander': _p(
        wilcoxon, drops['bft_top']['target'], drops['bft_top']['bystander'])}
    for m in meths:
        if m == 'bft_top':
            continue
        tests[f'p_top_vs_{m}_auc_t'] = _p(
            ttest_rel, drops['bft_top']['auc'], drops[m]['auc'])
        tests[f'p_top_vs_{m}_auc_wilcoxon'] = _p(
            wilcoxon, drops['bft_top']['auc'], drops[m]['auc'])

    cls = {str(k): v for k, v in cfg['class_names'].items()}
    figdata.save(name, {
        'experiment': results['experiment'],
        'fractions': np.asarray([0.0] + fractions, np.float32),
        'frac_stat': frac_stat,
        'n_obs': len(obs),
        'n_seeds': len({o['seed'] for o in obs}),
        'n_classes': n_classes,
        'class_names': [cls[str(c)] for c in range(n_classes)],
        'obs_seed': np.asarray([o['seed'] for o in obs], np.int32),
        'obs_class': np.asarray([o['target_class'] for o in obs], np.int32),
        'methods': methods,          # all pruning methods, mean/sd accuracy curves
        'drops': drops,              # per-observation drops at frac_stat + AUC
        **tests,
    })
    print(f'  [pruning_bundle] {name}: n_obs={len(obs)} '
          f'p_tvb={tests["p_target_vs_bystander"]:.2g}')
    return name
