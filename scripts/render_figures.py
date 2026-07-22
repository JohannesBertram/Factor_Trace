#!/usr/bin/env python3
"""Redraw paper figures from the figdata bundles — no models, no datasets.

    python scripts/render_figures.py                 # every figure with a bundle
    python scripts/render_figures.py fig3_digit_mlp_circuits figB_digit_mlp_details
    python scripts/render_figures.py --list

The bundles are written by the notebooks (which run where the data is) and live in
figures/figdata/. This is the loop for iterating on figure *design* off-cluster:
edit src/paper_figures.py, re-run this, look at figures/preview/*.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import figstyle                                    # noqa: E402
from src import figdata                            # noqa: E402
from src.paper_figures import FIGURES              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*', help='figure names (default: all available)')
    ap.add_argument('--list', action='store_true', help='list figures and bundle status')
    args = ap.parse_args()

    have = set(figdata.available())
    if args.list:
        for name, (bundle, _, mode) in FIGURES.items():
            print(f'{"ok " if bundle in have else "MISSING"}  {name:38s} '
                  f'{bundle:18s} {mode}')
        return

    names = args.names or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        sys.exit(f'unknown figure(s): {", ".join(unknown)}\n'
                 f'available: {", ".join(FIGURES)}')

    cache, done, skipped = {}, [], []
    for name in names:
        bundle, render, mode = FIGURES[name]
        if bundle not in have:
            skipped.append((name, bundle))
            continue
        if bundle not in cache:
            cache[bundle] = figdata.load(bundle)
        fig = render(cache[bundle])
        figstyle.save_fig(fig, name)
        matplotlib.pyplot.close(fig)
        done.append(name)

    print(f'\nrendered {len(done)}/{len(names)} figures')
    for name, bundle in skipped:
        print(f'  skipped {name}: bundle {bundle!r} not in figures/figdata/ — '
              'run the notebook section that builds it')


if __name__ == '__main__':
    main()
