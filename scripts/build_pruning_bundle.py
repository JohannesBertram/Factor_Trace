#!/usr/bin/env python3
"""Turn the notebook-13/14 pruning results into figdata bundles.

    python scripts/build_pruning_bundle.py                # nb14: mlp_digit cnn_cifar
    python scripts/build_pruning_bundle.py mlp_even_odd   # rebuild the nb13 bundle
    python scripts/build_pruning_bundle.py imagenet_cnn   # once its cluster run lands

Notebook 13 (even/odd MLP) and notebook 14 (digit MLP, CIFAR CNN, ImageNet CNN)
write raw JSON dumps to ``data/results/`` on the machine that ran them; the
cluster copies live in ``logs/results/``. The paper panels (Fig. 2e and the new
pruning panels of Fig. 6 / Fig. B) are drawn from
``figures/figdata/nb1N_pruning_<exp>.{npz,json}`` so they redraw anywhere with
numpy + matplotlib only.

For the nb13 experiment this is a pure re-encoding of the stored aggregate,
as before. For the nb14 experiments the aggregate and tests are recomputed
here from ``per_obs`` — deliberately, because a partial cluster run (e.g.
CIFAR with seeds still training) checkpoints per_obs but not the final
aggregate section, and the recomputation is the same arithmetic the notebook
runs. Only cluster-mode runs are accepted; the notebook's ``local`` smoke
profile refuses to overwrite a committed bundle.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import figdata                                    # noqa: E402

CURVE_KEYS = ('target_mean', 'target_sd', 'bystander_mean', 'bystander_sd')

# exp -> (bundle/raw name, n_classes, minimum observations for a committable run)
EXPERIMENTS = {
    'mlp_even_odd': ('nb13_pruning_mlp_even_odd', 2, 10),
    'mlp_digit':    ('nb14_pruning_mlp_digit', 10, 10),
    'cnn_cifar':    ('nb14_pruning_cnn_cifar', 10, 10),
    # 7 accepts the preliminary 7/8-category run (cat7=bird still on the
    # cluster); the floor still rejects smoke runs
    'imagenet_cnn': ('nb14_pruning_imagenet_cnn', 8, 7),
}


def find_source(name):
    """Newest cluster-mode raw JSON among logs/results and data/results."""
    hits = []
    for d in (REPO / 'logs' / 'results', REPO / 'data' / 'results'):
        p = d / f'{name}.json'
        if p.exists():
            raw = json.load(open(p))
            if raw.get('mode') == 'cluster':
                hits.append((p.stat().st_mtime, p, raw))
            else:
                print(f'  [skip] {p} is a {raw.get("mode")!r}-mode run, not cluster')
    if not hits:
        sys.exit(f'no cluster-mode {name}.json under logs/results/ or data/results/ '
                 f'— run the notebook with the cluster profile first.')
    hits.sort()
    return hits[-1][1], hits[-1][2]


def curves(o, method, fractions, n_classes):
    """(target, bystander) accuracy over [0]+fractions for one JSON observation."""
    d = o['target_class']
    others = [c for c in range(n_classes) if c != d]
    t = [o['baseline'][str(d)]] + [o['curves'][method][str(f)][str(d)]
                                   for f in fractions]
    b = [np.mean([o['baseline'][str(c)] for c in others])] + \
        [np.mean([o['curves'][method][str(f)][str(c)] for c in others])
         for f in fractions]
    return np.array(t), np.array(b)


def build_nb13(name, raw):
    """Pure re-encoding of the stored aggregate — unchanged from the original."""
    agg, st, cfg = raw['aggregate'], raw['stats'], raw['config']
    methods = {m: {k: np.asarray(d[k], np.float32) for k in CURVE_KEYS}
               for m, d in agg['methods'].items()}
    drops = {m: {k: np.asarray(v, np.float32) for k, v in st['drops'][m].items()}
             for m in st['drops']}
    p_tvb = float(st['tests'].get('bft_top_target_vs_bystander',
                                  {}).get('p', float('nan')))
    figdata.save(name, {
        'experiment': raw['experiment'],
        'fractions': np.asarray(agg['fractions'], np.float32),
        'frac_stat': float(st['frac_stat']),
        'n_obs': int(agg['n_obs']),
        'class_names': [cfg['class_names']['0'], cfg['class_names']['1']],
        'methods': methods,          # all pruning methods, mean/sd accuracy curves
        'drops': drops,              # per-observation drops at frac_stat
        'p_target_vs_bystander': p_tvb,
    })


def build_nb14(name, raw, n_classes):
    """Aggregate + tests from per_obs (works for partial runs); slim superset."""
    obs = raw['per_obs']
    cfg = raw['config']
    fractions = [float(f) for f in cfg['fractions']]
    frac_stat = float(cfg['frac_stat'])
    fi_stat = fractions.index(frac_stat)
    meths = list(cfg['methods'])

    def _p(test, a, b):
        """p-value, or nan when the paired test is degenerate (e.g. the ImageNet
        spine run, where every saliency baseline floors the network at the
        smallest fraction and the paired differences are identically zero)."""
        try:
            with np.errstate(invalid='ignore'):
                return float(test(a, b).pvalue)
        except ValueError:
            return float('nan')

    methods, drops = {}, {}
    for m in meths:
        T = np.stack([curves(o, m, fractions, n_classes)[0] for o in obs])
        B = np.stack([curves(o, m, fractions, n_classes)[1] for o in obs])
        methods[m] = {'target_mean': T.mean(0).astype(np.float32),
                      'target_sd': T.std(0).astype(np.float32),
                      'bystander_mean': B.mean(0).astype(np.float32),
                      'bystander_sd': B.std(0).astype(np.float32),
                      # per-observation curves, so a panel can show the
                      # per-class structure where the mean would mislead
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

    cls = cfg['class_names']
    figdata.save(name, {
        'experiment': raw['experiment'],
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
    print(f'  n_obs={len(obs)}  p_top_vs_bottom(auc)='
          f'{tests["p_top_vs_bft_bottom_auc_wilcoxon"]:.2g}  '
          f'p_tvb={tests["p_target_vs_bystander"]:.2g}')


def main(exps):
    for exp in exps:
        if exp not in EXPERIMENTS:
            sys.exit(f'unknown experiment {exp!r}; choose from {list(EXPERIMENTS)}')
        name, n_classes, min_obs = EXPERIMENTS[exp]
        src, raw = find_source(name)
        n_obs = len(raw.get('per_obs', []))
        if n_obs < min_obs:
            sys.exit(f'{src} has only {n_obs} observations (< {min_obs}) — '
                     f'not a committable run.')
        print(f'{exp}: {src.relative_to(REPO)}')
        if exp == 'mlp_even_odd':
            build_nb13(name, raw)
        else:
            build_nb14(name, raw, n_classes)


if __name__ == '__main__':
    main(sys.argv[1:] or ['mlp_digit', 'cnn_cifar'])
