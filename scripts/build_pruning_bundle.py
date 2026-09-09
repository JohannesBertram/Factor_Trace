#!/usr/bin/env python3
"""Rebuild the pruning figdata bundles from stored results JSONs.

    python scripts/build_pruning_bundle.py                # mlp_digit cnn_cifar
    python scripts/build_pruning_bundle.py mlp_even_odd imagenet_cnn

The model notebooks (01-05 §8) now write the bundles directly via
``src.bundles.pruning_bundle``; this script only re-encodes an existing
cluster-mode ``nb13/nb14_pruning_<exp>.json`` (from ``logs/results/`` or
``data/results/``, newest wins) without re-running anything. Aggregate and
tests are recomputed from ``per_obs``, so partial runs with a checkpointed
per_obs work; the per-experiment observation floor rejects smoke runs.
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bundles import pruning_bundle, PRUNING_BUNDLES    # noqa: E402


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
                 f'— run the notebook first.')
    hits.sort()
    return hits[-1][1], hits[-1][2]


def main(exps):
    for exp in exps:
        if exp not in PRUNING_BUNDLES:
            sys.exit(f'unknown experiment {exp!r}; choose from {list(PRUNING_BUNDLES)}')
        src, raw = find_source(PRUNING_BUNDLES[exp][0])
        print(f'{exp}: {src.relative_to(REPO)}')
        pruning_bundle(exp, raw)


if __name__ == '__main__':
    main(sys.argv[1:] or ['mlp_digit', 'cnn_cifar'])
