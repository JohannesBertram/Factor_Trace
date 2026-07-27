#!/usr/bin/env python3
"""Turn the notebook-13 pruning results into a figdata bundle.

    python scripts/build_pruning_bundle.py

Notebook 13 (``notebooks/13_pruning_mlp_even_odd.ipynb``) writes a raw JSON dump
to ``data/results/nb13_pruning_mlp_even_odd.json``. Fig. 2's causal-validation
panel (e) is drawn from ``figures/figdata/nb13_pruning_mlp_even_odd.{npz,json}``,
so it redraws anywhere with numpy + matplotlib only, like every other paper
figure. This is a pure re-encoding into a slim superset — the per-method
accuracy curves, the per-observation drops at ``frac_stat``, and the key test —
nothing is recomputed.

The notebook has a ``quick`` mode (few seeds, 3 fractions) and a full ``cluster``
mode (10 seed x class observations, 8 fractions). Only the full run belongs in
the committed bundle; this script refuses to overwrite it with a quick run.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import figdata                                    # noqa: E402

NAME = 'nb13_pruning_mlp_even_odd'
SRC = REPO / 'data' / 'results' / f'{NAME}.json'
CURVE_KEYS = ('target_mean', 'target_sd', 'bystander_mean', 'bystander_sd')
MIN_OBS = 10          # a full run: 5 seeds x 2 target classes


def main():
    if not SRC.exists():
        sys.exit(f'{SRC} not found — run notebook 13 (full/cluster mode) first.')
    raw = json.load(open(SRC))
    agg, st, cfg = raw['aggregate'], raw['stats'], raw['config']
    n_obs = int(agg['n_obs'])
    if n_obs < MIN_OBS:
        sys.exit(f'{SRC} has only {n_obs} observations — this looks like the '
                 f'notebook\'s quick mode. Run the full sweep before rebuilding '
                 f'the committed bundle (which was built from a {MIN_OBS}-obs run).')

    methods = {m: {k: np.asarray(d[k], np.float32) for k in CURVE_KEYS}
               for m, d in agg['methods'].items()}
    drops = {m: {k: np.asarray(v, np.float32) for k, v in st['drops'][m].items()}
             for m in st['drops']}
    p_tvb = float(st['tests'].get('bft_top_target_vs_bystander', {}).get('p', float('nan')))

    figdata.save(NAME, {
        'experiment': raw['experiment'],
        'fractions': np.asarray(agg['fractions'], np.float32),
        'frac_stat': float(st['frac_stat']),
        'n_obs': n_obs,
        'class_names': [cfg['class_names']['0'], cfg['class_names']['1']],
        'methods': methods,          # all pruning methods, mean/sd accuracy curves
        'drops': drops,              # per-observation drops at frac_stat
        'p_target_vs_bystander': p_tvb,
    })


if __name__ == '__main__':
    main()
