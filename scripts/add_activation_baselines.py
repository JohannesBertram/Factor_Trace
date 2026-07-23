#!/usr/bin/env python3
"""Add the network's own activations to the fingerprint bundles.

The fingerprint figures compare the BFT fingerprint against the representation the
network itself computes — same stimuli, same 2-D PCA treatment, dots colored by
class. The bundles carry the fingerprints but not the activations, so this script
recomputes the activations from the committed seed-0 checkpoints and writes them
back into `figures/figdata/nb0N_fingerprints.npz` as an `act` entry:

    act.index          int   (n,)        rows of fp.id these activations belong to
    act.reps.<i>.X     f32   (n, dim)    one activation matrix
    act.reps.<i>.label str                as named in the separability panel
    act.reps.<i>.dim   int

It reproduces each notebook's stimulus pipeline exactly and then *verifies* the
alignment against the labels already in the bundle — if a single label disagrees
the experiment is skipped rather than written, because a silent misalignment
would produce a plausible-looking but wrong figure.

    python scripts/add_activation_baselines.py            # all experiments
    python scripts/add_activation_baselines.py nb01 nb03
    python scripts/add_activation_baselines.py --check    # verify, write nothing
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src import figdata                                            # noqa: E402
from src.bft import collect_layer_dicts                            # noqa: E402
from src.checkpoint import load_experiment                         # noqa: E402
from src.checkpoint import get_loaders_from_config, get_transform  # noqa: E402
from src.data_utils import get_mnist_loaders                       # noqa: E402
from src.training import label_transform_even_odd                  # noqa: E402

MODELS = os.path.join(REPO, 'data', 'models')
DATA = os.path.join(REPO, 'data')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_STIM = 1500          # stimuli kept for the embedding panels, class-balanced


def balanced_index(labels, n_max, seed=0):
    """Class-balanced subsample of row indices, deterministic."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    per = max(1, n_max // len(classes))
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(labels == c)[0],
                                     min(per, int((labels == c).sum())), replace=False)
                          for c in classes])
    return np.sort(idx)


# ── one loader per experiment, reproducing the analysis notebook's §2 ─────────

def collect_mlp(exp, batch=None):
    """Notebooks 01 / 02: collect_layer_dicts over the config's test loader."""
    model, config = load_experiment(os.path.join(MODELS, exp), DEVICE)
    config = dict(config)
    config['dataset_kwargs'] = dict(config['dataset_kwargs'], root=DATA + os.sep)
    _, test_loader = get_loaders_from_config(config)
    lt = get_transform(config['label_transform'])
    c = collect_layer_dicts(model, test_loader, label_transform=lt, device=DEVICE)
    return c, [d['input_fmap'] for d in c['layer_data']]


def nb01():
    c, li = collect_mlp('mnist_even_odd_mlp_8_4_0134_seed0')
    return dict(targets=c['targets'], digits=c['digits'],
                reps=[(r'$L_2$ act.', li[1]), (r'$L_3$ act.', li[2])])


def nb02():
    c, li = collect_mlp('mnist_digit_mlp_40_20_seed0')
    return dict(targets=c['targets'], digits=c['targets'],
                reps=[(r'$L_2$ act.', li[1]), (r'$L_3$ act.', li[2])])


def nb03():
    """Notebook 03: only_correct, then the top-60-per-class confidence filter."""
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    test_ds = torchvision.datasets.CIFAR10(
        DATA, False, download=False,
        transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    loader = DataLoader(test_ds, 256, shuffle=False, num_workers=0)
    model, _ = load_experiment(os.path.join(MODELS, 'cifar10_cnn_seed0'), DEVICE)
    raw = collect_layer_dicts(model, loader, DEVICE, only_correct=True)
    n_top, n_classes = 60, 10
    keep = np.sort(np.concatenate([
        np.where(raw['targets'] == c)[0][
            np.argsort(raw['confidences'][raw['targets'] == c])[::-1][:n_top]]
        for c in range(n_classes)]))
    li = [ld['input_fmap'][keep] for ld in raw['layer_data']]
    flat = [x.reshape(len(x), x.shape[1], -1).mean(2) if x.ndim == 4 else x for x in li]
    return dict(targets=raw['targets'][keep], digits=raw['targets'][keep],
                reps=[('conv4 act.', flat[-2]), ('classifier act.', flat[-1])])


def nb04():
    """Notebook 04: the TinyViT's captured CLS-token activations."""
    from src.models import TinyViT                                  # noqa: F401
    model, _ = load_experiment(os.path.join(MODELS, 'mnist_even_odd_vit_tiny_seed0'),
                               DEVICE)
    _, test_loader = get_mnist_loaders(batch_size=64, root=DATA + os.sep)
    model.eval()
    out = {k: [] for k in ('attn_out_cls', 'ffn1_in', 'ffn2_in')}
    targets, digits = [], []
    with torch.no_grad():
        for imgs, digs in test_loader:
            imgs = imgs.to(DEVICE)
            lab = label_transform_even_odd(digs).to(DEVICE)
            mask = model(imgs, capture=True).argmax(1) == lab
            if not mask.any():
                continue
            blk = model.block
            out['attn_out_cls'].append(blk._attn_out[mask, 0].cpu().numpy())
            out['ffn1_in'].append(blk._ffn1_in[mask, 0].cpu().numpy())
            out['ffn2_in'].append(blk._ffn2_in[mask, 0].cpu().numpy())
            targets.append(lab[mask].cpu().numpy())
            digits.append(digs[mask.cpu()].numpy())
    out = {k: np.concatenate(v) for k, v in out.items()}
    return dict(targets=np.concatenate(targets), digits=np.concatenate(digits),
                reps=[('attn out', out['attn_out_cls']), ('FFN act.', out['ffn2_in'])])


EXPERIMENTS = {
    'nb01': dict(bundle='nb01_fingerprints', collect=nb01, label_key='id_digits'),
    'nb02': dict(bundle='nb02_fingerprints', collect=nb02, label_key='id_targets'),
    'nb03': dict(bundle='nb03_fingerprints', collect=nb03, label_key='id_targets'),
    'nb04': dict(bundle='nb04_fingerprints', collect=nb04, label_key='id_digits'),
}


def run(name, spec, write=True):
    """Recompute the activations and write them into the bundle.

    Preferred mode is *aligned*: the recomputed stimuli are the very rows the
    bundle's fingerprints came from, verified label by label, so the two
    embedding panels show the same stimuli. When the pipeline cannot be
    reproduced exactly (a model whose correctness mask shifts by a few samples
    moves every later index), fall back to an *unaligned* sample of the same
    test set and record its own labels, so the panel can say so.
    """
    D = figdata.load(spec['bundle'])
    fp = D['fp']
    ref = fp[spec['label_key']]
    n_fp = len(fp['id'])
    got = spec['collect']()
    pool = got['digits'] if spec['label_key'].endswith('digits') else got['targets']
    pool = np.asarray(pool)
    rows = fp['id_index'] if 'id_index' in fp else np.arange(n_fp)

    aligned = len(pool) > rows.max() and np.array_equal(pool[rows], np.asarray(ref))
    if aligned:
        idx = balanced_index(ref, MAX_STIM)
        take = rows[idx]
        act = dict(index=idx.astype(np.int64), aligned=1)
        note = f'{len(pool)} stimuli verified against the bundle'
    else:
        n_diff = (int((pool[rows] != np.asarray(ref)).sum())
                  if len(pool) > rows.max() else n_fp)
        idx = balanced_index(pool, MAX_STIM)
        take = idx
        act = dict(labels=pool[idx].astype(np.int64), aligned=0)
        note = (f'{len(pool)} stimuli recomputed but NOT aligned to the bundle '
                f'({n_diff} of {n_fp} labels differ) — storing an independent '
                'sample of the same test set')

    act['reps'] = [dict(label=lab, X=np.asarray(X)[take].astype(np.float32),
                        dim=int(np.asarray(X).shape[1])) for lab, X in got['reps']]
    print(f'  {name}: {note}; keeping {len(take)}; '
          + ', '.join(f"{r['label']} ({r['dim']}d)" for r in act['reps']))
    if write:
        D['act'] = act
        figdata.save(spec['bundle'], D)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='*', default=None)
    ap.add_argument('--check', action='store_true', help='verify only, write nothing')
    args = ap.parse_args()
    names = args.names or list(EXPERIMENTS)
    ok = 0
    for name in names:
        if name not in EXPERIMENTS:
            sys.exit(f'unknown experiment {name!r}; have {", ".join(EXPERIMENTS)}')
        print(f'{name} ({EXPERIMENTS[name]["bundle"]})')
        try:
            ok += bool(run(name, EXPERIMENTS[name], write=not args.check))
        except Exception as exc:                                   # noqa: BLE001
            print(f'  {name}: FAILED — {type(exc).__name__}: {exc}')
    print(f'\n{ok}/{len(names)} bundles updated' if not args.check else
          f'\n{ok}/{len(names)} bundles verified')


if __name__ == '__main__':
    main()
