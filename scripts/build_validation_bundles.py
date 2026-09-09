#!/usr/bin/env python3
"""Rebuild the validation figdata bundles from stored results JSONs.

    python scripts/build_validation_bundles.py [exp ...]

The model notebooks (01-05 §7) now write the bundles directly via
``src.bundles.validation_bundle``; this script only re-encodes an existing
``logs/results/nb09_<exp>.json`` (e.g. one copied from the cluster) without
re-running anything.
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.bundles import validation_bundle, VALIDATION_LABELS   # noqa: E402


def main(exps):
    results_dir = REPO / 'logs' / 'results'
    n = 0
    for exp in exps:
        src = results_dir / f'nb09_{exp}.json'
        if not src.exists():
            print(f'  skip {exp}: {src} not found')
            continue
        with open(src) as f:
            D = json.load(f)
        validation_bundle(exp, D, source=os.path.relpath(src, REPO))
        print(f'  built nb09_{exp}_validation')
        n += 1
    print(f'\nbuilt {n}/{len(exps)} validation bundles')


if __name__ == '__main__':
    main(sys.argv[1:] or list(VALIDATION_LABELS))
