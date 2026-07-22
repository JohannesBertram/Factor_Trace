#!/usr/bin/env python3
"""Turn the notebook-09 cluster results into figdata bundles.

    python scripts/build_validation_bundles.py

Notebook 09 runs on the cluster and writes one JSON per experiment to
``logs/results/nb09_<exp>.json``. The appendix validation figures are drawn from
``figures/figdata/nb09_<exp>_validation.{npz,json}`` so they redraw anywhere with
numpy + matplotlib only, exactly like every other paper figure.

This is a pure re-encoding: nothing is recomputed, nothing is dropped except the
``figures`` key (paths to the notebook's own throwaway PDFs).
"""
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src import figdata                                    # noqa: E402

EXPERIMENTS = ['mlp_even_odd', 'mlp_digit', 'cnn_cifar', 'vit_mnist', 'imagenet_cnn']

# what the paper calls each experiment, and the architecture line under the title
LABELS = {
    'mlp_even_odd':  ('MLP even/odd', r'$784\to8\to4\to2$, MNIST parity'),
    'mlp_digit':     ('MLP digits', r'$784\to40\to20\to10$, MNIST digits'),
    'cnn_cifar':     ('CNN', 'SmallCNN, CIFAR-10'),
    'vit_mnist':     ('ViT', 'TinyViT $d{=}32$, MNIST parity'),
    'imagenet_cnn':  ('ImageNet CNN', 'SqueezeNet 1.1 spine, ImageNet (8 categories)'),
}


def sanitize(obj):
    """figdata flattens nested keys on '.', so no dict key may contain one.

    Node ids are module paths ('L0:layers.1:0-0'), so they would otherwise be
    split into a spurious extra level on load.
    """
    if isinstance(obj, dict):
        return {str(k).replace('.', '_'): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def build(exp: str, results_dir: Path) -> bool:
    src = results_dir / f'nb09_{exp}.json'
    if not src.exists():
        print(f'  skip {exp}: {src} not found')
        return False
    with open(src) as f:
        D = json.load(f)
    D.pop('figures', None)                    # paths to the notebook's own PDFs
    D = sanitize(D)
    D['label'], D['arch'] = LABELS[exp]
    D['source_json'] = os.path.relpath(src, REPO)
    figdata.save(f'nb09_{exp}_validation', D)
    return True


def main():
    results_dir = REPO / 'logs' / 'results'
    n = sum(build(exp, results_dir) for exp in EXPERIMENTS)
    print(f'\nbuilt {n}/{len(EXPERIMENTS)} validation bundles')


if __name__ == '__main__':
    main()
