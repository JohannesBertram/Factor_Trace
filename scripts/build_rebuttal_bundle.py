#!/usr/bin/env python3
"""Pack the rebuttal experiments' results into the figdata bundle ``rebuttal``.

    .venv/bin/python scripts/build_rebuttal_bundle.py

Reads the stored JSONs of ``rebuttal/exps/results/`` (see ``rebuttal/rebuttal_exps.md``) and
writes ``figures/figdata/rebuttal.{npz,json}``, so the slides in ``figures/results.py`` draw the
rebuttal results without the experiment code. Nothing is re-run. Sections:

- ``link_mlp`` / ``link_cnn``: drop of every parent factor when a child's top weights are pruned
  (8x4x2 MLP L2 -> L1 at 1% of W1; CIFAR-10 CNN classifier -> conv4 at 0.5% of conv4).
- ``sub_mlp``: per-digit accuracy drop when a digit sub-circuit of the 784 -> 8 layer is pruned.
- ``sub_cnn``: per-group accuracy drop when a conv4 orientation sub-factor is pruned.
- ``baselines``: class specificity of BFT vs magnitude / |Wa| / Taylor / LRP pruning.
- ``imagenet``: the frozen ImageNet tree on random held-out images.
- ``tau``: pruning and fingerprint separability across the gate tau.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXPS = REPO / 'rebuttal' / 'exps'
RES = EXPS / 'results'
sys.path[:0] = [str(REPO), str(EXPS)]

from src import figdata                                                       # noqa: E402
from summarize_prune_baselines import metrics                                 # noqa: E402
from summarize_tau import prune_drops                                         # noqa: E402

CIFAR = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def read(name):
    return json.load(open(RES / f'{name}.json'))


def link_mlp():
    """8x4x2 MLP, L2 -> L1: rows = pruned child, columns = parent factor (odd, 4, 0)."""
    (R,) = read('link_mlp0').values()
    m4 = R['M4']['pos']
    n = str(m4['counts'][0])
    perm = [np.mean([r[i] for r in m4['perm'][n][i]]) for i in range(len(m4['perm'][n]))]
    return {'drop': np.asarray(m4['actual'][n]), 'random': np.asarray(m4['random'][n]['mean']),
            'perm_own': np.asarray(perm), 'labels': ['odd', '4', '0'],
            'frac': int(n) / m4['n_weights']}


def link_cnn():
    """CIFAR-10 CNN, classifier -> conv4: 13 x 13, labeled by each root factor's top class."""
    R = read('link_cnn')
    n = str(R['M4']['counts'][0])
    M = R['M4'][n]
    A = np.asarray(M['actual'])
    perm = [np.mean([r[i] for r in M['perm'][i]]) for i in range(len(A))]
    return {'drop': A, 'random': np.asarray(M['random_mean']), 'perm_own': np.asarray(perm),
            'top_class': np.array([m['top_class'] for m in R['M0']]), 'class_names': CIFAR,
            'frac': int(n) / R['M4']['n_weights']}


def sub_mlp():
    """Accuracy drop per digit (0, 4, 1, 3) for each destroyed digit sub-circuit of L0, test set."""
    R = read('subcircuit_prune_mlp')
    base, fr, dig = R['baseline']['test'], R['fractions'], ['0', '4', '1', '3']
    out = {'fractions': np.asarray(fr), 'digits': np.array([0, 4, 1, 3])}
    for key in ('0_not_4', '4_not_0', '1_not_3', '3_not_1'):
        runs = R['runs'][f'{key}|L0']
        out[key] = {m: np.array([[base[d] - runs[m]['test'][str(f)][d] for d in dig] for f in fr])
                    for m in ('factor_own', 'attr_own', 'random')}
    return out


def sub_cnn():
    """Accuracy drop per orientation group for the horse (f1 vs f2) and ship (f1 vs f3) pairs.

    Columns: own orientation = mean(a, flip_b), mirror orientation = mean(b, flip_a), other
    classes. Rows of 'own' prune sub-factor a, rows of 'sibling' prune b (so for b the 'own'
    orientation is the second column).
    """
    out = {}
    for name, run in (('horse', 'cnn_conv4_prune_2_1v2'), ('ship', 'cnn_conv4_prune_10_1v3')):
        R = read(run)
        base = R['baseline']
        fr = sorted(R['runs']['own'], key=float)

        def drops(m):
            rows = []
            for f in fr:
                d = {g: base[g] - v for g, v in R['runs'][m][f].items()}
                rows.append([(d['a'] + d['flip_b']) / 2, (d['b'] + d['flip_a']) / 2,
                             d['other_classes']])
            return np.array(rows)
        out[name] = {'node': R['node'], 'a': R['a'], 'b': R['b'], 'fractions': np.array(fr, float),
                     **{m: drops(m) for m in ('own', 'sibling', 'class', 'random')}}
    return out


def baselines():
    """Whole-network specificity (target - bystander drop) per model, each baseline in its better
    normalization, as in summarize_prune_baselines.py."""
    out = {'methods': ['bft', 'magnitude', 'wxa', 'wxa_ctr', 'taylor', 'taylor_ctr', 'lrp',
                       'lrp_ctr', 'random']}
    for model in ('mlp_even_odd', 'mlp_digit', 'cnn_cifar'):
        R = read(f'prune_baselines_{model}')
        fr, obs = R['fractions'], R['per_obs']
        spec = {m: np.mean([metrics(o, m, fr)['spec'] for o in obs])
                for m in R['methods'] if '/L' not in m}
        out[model] = np.array([max(spec[v] for v in (m, m + '@global') if v in spec)
                               for m in out['methods']])
    return out


def imagenet():
    R = read('imagenet_heldout')
    pools = ['traced', 'val_rest', 'val_new', 'train16']
    get = lambda key: np.array([R['pools'][p][key] for p in pools])          # (pool, mean/lo/hi)
    return {'pools': pools, 'n': np.array([R['pools'][p]['n'] for p in pools]),
            'network': get('net_cat_acc'), 'tree': get('root_call_acc'), 'chance': 1 / 8}


def tau():
    out = {}
    for model in ('mlp_even_odd', 'mlp_digit', 'cnn_cifar'):
        R = read(f'tau_sensitivity_{model}')
        taus = [str(t) for t in R['taus']]
        per = lambda f: np.array([np.mean([f(r, r['per_tau'][t]) for r in R['runs']]) for t in taus])
        rnd = np.mean([np.mean([r['baseline'][c] - np.mean([r['random'][f][c] for f in r['random']])
                                for c in r['baseline']]) for r in R['runs']])
        out[model] = {'taus': np.asarray(R['taus']), 'paper_tau': R['paper_tau'],
                      'target': per(lambda r, x: prune_drops(r, x)[0]),
                      'bystander': per(lambda r, x: prune_drops(r, x)[1]),
                      'silhouette': per(lambda r, x: x['separability']['task']['silhouette']),
                      'random': float(rnd)}
    return out


if __name__ == '__main__':
    figdata.save('rebuttal', {'link_mlp': link_mlp(), 'link_cnn': link_cnn(), 'sub_mlp': sub_mlp(),
                              'sub_cnn': sub_cnn(), 'baselines': baselines(),
                              'imagenet': imagenet(), 'tau': tau()})
