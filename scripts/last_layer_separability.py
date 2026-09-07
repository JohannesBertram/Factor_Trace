"""nb17 — single-tree, last-layer fingerprint experiment.

Question: can the separate fingerprint tree be dropped, reading the fingerprint
from the CIRCUIT tree's output-layer factors only (optionally + one layer back)?
And what is the fair activation comparison for that code?

Per model this script builds the circuit-tree top (the root factorization, and
its children one layer back — for CIFAR only the last two layers are traced,
which reproduces the full circuit tree's top exactly since a node's NMF depends
only on its own arbor and the weights passed down from above), then compares:

  fingerprint codes                     activation baselines (pooled)
  ---------------------                 ------------------------------
  fp_out    root img_factors            act_out     root-layer output z = a @ W^T
  fp_top2   root + children             act_penult  root-layer input
  fp_full   whole tree (where traced)   act_last2   last two traced layers' inputs
                                        act_full    all traced layers' inputs

Every fp x act pair is compared native + PCA-matched + GRP-matched at min dim
(src.separability.paired_matched), under the task labels and, where different,
fine labels (digits for the parity models). Cosine silhouette + 3-fold kNN.

Writes data/results/nb17_last_layer_sep_<exp>.json
Run:   .venv/bin/python scripts/last_layer_separability.py [mlp_even_odd mlp_digit vit_mnist cnn_cifar]
"""
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from src import bft, collect_layer_dicts, cached_tree
from src.checkpoint import load_experiment, get_loaders_from_config, get_transform
from src.data_utils import get_mnist_loaders
from src.training import label_transform_even_odd
from src import separability as sep
from src.arbors import activation_matrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      ('mps' if torch.backends.mps.is_available() else 'cpu'))
MODEL_ROOT = os.path.join(REPO, 'data', 'models')
OUT_DIR = os.path.join(REPO, 'data', 'results')
os.makedirs(OUT_DIR, exist_ok=True)


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


# ── model contexts ────────────────────────────────────────────────────────────

def ctx_mlp(exp_dir):
    model, config = load_experiment(os.path.join(MODEL_ROOT, exp_dir), DEVICE)
    config['dataset_kwargs']['root'] = os.path.join(REPO, 'data')
    _, test_loader = get_loaders_from_config(config)
    lt = get_transform(config['label_transform'])
    c = collect_layer_dicts(model, test_loader, label_transform=lt, device=DEVICE)
    return model, c


def build_mlp_even_odd():
    model, c = ctx_mlp('mnist_even_odd_mlp_8_4_0134_seed0')
    n = len(c['targets'])
    tree = cached_tree('nb01_circuit', lambda: bft(
        c['layer_data'], k_max=[7, 6, 4], n_branches=[1, 2, 4],
        stimulus_threshold=0.5, weighting='img_selectivity', n_jobs=3),
        params=dict(k=[7, 6, 4], b=[1, 2, 4], tau=0.5, n=n))
    # the cached tree was traced in primary mode and stores its targets — verify
    # the locally rebuilt population matches it row for row
    tt = np.asarray(tree.targets)
    if tt.any() and not np.array_equal(tt, np.asarray(c['targets'])):
        raise RuntimeError('nb01: rebuilt population does not match cached tree rows')
    return dict(exp='mlp_even_odd', tree=tree, full_tree=True,
                layer_inputs=[d['input_fmap'] for d in c['layer_data']],
                labels_task=np.asarray(c['targets']).astype(int),
                labels_fine=np.asarray(c['digits']).astype(int))


def build_mlp_digit():
    model, c = ctx_mlp('mnist_digit_mlp_40_20_seed0')
    n = len(c['targets'])
    tree = cached_tree('nb02_circuit', lambda: bft(
        c['layer_data'], k_max=[10, 11, 14], n_branches=[1, 2, 14],
        stimulus_threshold=0.7, weighting='img_selectivity', n_jobs=3),
        params=dict(k=[10, 11, 14], b=[1, 2, 14], tau=0.7, n=n))
    tt = np.asarray(tree.targets)
    if tt.any() and not np.array_equal(tt, np.asarray(c['targets'])):
        raise RuntimeError('nb02: rebuilt population does not match cached tree rows')
    y = np.asarray(c['targets']).astype(int)
    return dict(exp='mlp_digit', tree=tree, full_tree=True,
                layer_inputs=[d['input_fmap'] for d in c['layer_data']],
                labels_task=y, labels_fine=y)


def build_cnn_cifar():
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    ds = torchvision.datasets.CIFAR10(os.path.join(REPO, 'data'), False, download=True,
                                      transform=T.Compose([T.ToTensor(), T.Normalize(mean, std)]))
    loader = DataLoader(ds, 256, shuffle=False, num_workers=0)
    model, _ = load_experiment(os.path.join(MODEL_ROOT, 'cifar10_cnn_seed0'), DEVICE)
    raw = collect_layer_dicts(model, loader, device=DEVICE, only_correct=True)
    # top-200 most confident correct per class (nb03 §1 config)
    keep = np.sort(np.concatenate([
        np.where(raw['targets'] == cl)[0][
            np.argsort(raw['confidences'][raw['targets'] == cl])[::-1][:200]]
        for cl in range(10)]))
    layer_data = [{**ld, 'input_fmap': ld['input_fmap'][keep],
                   'output_fmap': ld['output_fmap'][keep]} for ld in raw['layer_data']]
    y = raw['targets'][keep].astype(int)
    n = len(y)
    # last two layers only: reproduces the full circuit tree's root + conv4 children
    tree = cached_tree('nb17_cnn_top2', lambda: bft(
        layer_data[-2:], k_max=[10, 14], n_branches=[1, 14],
        conv_pool_method='avg', stimulus_threshold=0.0,
        weighting='img_selectivity', verbose=1, n_jobs=3),
        params=dict(k=[10, 14], b=[1, 14], tau=0.0, n=n))
    return dict(exp='cnn_cifar', tree=tree, full_tree=False,
                layer_inputs=[d['input_fmap'] for d in layer_data],
                labels_task=y, labels_fine=y)


def build_vit_mnist():
    model, config = load_experiment(os.path.join(MODEL_ROOT, 'mnist_even_odd_vit_tiny_seed0'), DEVICE)
    _, test_loader = get_mnist_loaders(root=os.path.join(REPO, 'data'), batch_size=64)
    model.eval()
    acts = {'attn_in': [], 'attn_w': [], 'attn_out_cls': [], 'ffn1_in': [], 'ffn2_in': []}
    targets, digits = [], []
    with torch.no_grad():
        for imgs, digs in test_loader:
            imgs = imgs.to(DEVICE)
            labels = label_transform_even_odd(digs).to(DEVICE)
            logits = model(imgs, capture=True)
            mask = (logits.argmax(1) == labels)
            if not mask.any():
                continue
            blk = model.block
            acts['attn_in'].append(blk._attn_in[mask].cpu().numpy())
            acts['attn_w'].append(blk._attn_w[mask].mean(1)[:, 0, :].cpu().numpy())
            acts['attn_out_cls'].append(blk._attn_out[mask, 0].cpu().numpy())
            acts['ffn1_in'].append(blk._ffn1_in[mask, 0].cpu().numpy())
            acts['ffn2_in'].append(blk._ffn2_in[mask, 0].cpu().numpy())
            targets.append(labels[mask].cpu().numpy())
            digits.append(digs[mask.cpu()].numpy())
    acts = {k: np.concatenate(v) for k, v in acts.items()}
    targets = np.concatenate(targets).astype(int)
    digits = np.concatenate(digits).astype(int)
    D = model.block.attn.embed_dim if hasattr(model.block.attn, 'embed_dim') else 32
    W_V = model.block.attn.in_proj_weight[2 * D:, :].detach().cpu().numpy()
    layer_dicts = [
        {'type': 'attn', 'name': 'B0-V', 'weight': W_V,
         'input_fmap': acts['attn_in'], 'attn_weights': acts['attn_w']},
        {'type': 'fc', 'name': 'B0-O',
         'weight': model.block.attn.out_proj.weight.detach().cpu().numpy(),
         'input_fmap': acts['attn_out_cls']},
        {'type': 'fc', 'name': 'B0-FFN1',
         'weight': model.block.ffn1.weight.detach().cpu().numpy(),
         'input_fmap': acts['ffn1_in']},
        {'type': 'fc', 'name': 'B0-FFN2',
         'weight': model.block.ffn2.weight.detach().cpu().numpy(),
         'input_fmap': acts['ffn2_in']},
    ]
    n = len(targets)
    tree = cached_tree('nb17_vit_full', lambda: bft(
        layer_dicts, k_max=[7, 10, 7, 10], n_branches=[1, 1, 2, 10],
        stimulus_threshold=0.0, weighting='img_selectivity', n_jobs=3),
        params=dict(k=[7, 10, 7, 10], b=[1, 1, 2, 10], tau=0.0, n=n))
    return dict(exp='vit_mnist', tree=tree, full_tree=True,
                layer_inputs=[d['input_fmap'] for d in layer_dicts],
                labels_task=targets, labels_fine=digits)


BUILDERS = {'mlp_even_odd': build_mlp_even_odd, 'mlp_digit': build_mlp_digit,
            'vit_mnist': build_vit_mnist, 'cnn_cifar': build_cnn_cifar}


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(ctx):
    tree = ctx['tree']
    root = tree.root
    li = ctx['layer_inputs']
    n = root.img_factors.shape[0]

    fps = {'fp_out': np.asarray(root.img_factors)}
    if root.children:
        fps['fp_top2'] = np.concatenate(
            [root.img_factors] + [np.asarray(c.img_factors) for c in root.children], axis=1)
    if ctx['full_tree']:
        fps['fp_full'] = sep.fingerprint_slices(tree, n, with_per_layer=False)['full']

    pooled = [sep._pool(x) for x in li]
    acts = {'act_penult': pooled[-1],
            'act_last2': np.concatenate(pooled[-2:], axis=1),
            'act_full': np.concatenate(pooled, axis=1)}
    if root.layer_type in ('fc', 'attn'):
        # the root is the last traced layer, so its input is the last entry of
        # the (possibly longer than the trace) layer_inputs list
        a_in = activation_matrix(root, li[-1])
        acts['act_out'] = a_in @ np.asarray(root.weight).T

    label_sets = {'task': ctx['labels_task']}
    if not np.array_equal(ctx['labels_task'], ctx['labels_fine']):
        label_sets['fine'] = ctx['labels_fine']

    res = {'exp': ctx['exp'], 'n': int(n), 'root_k': int(root.img_factors.shape[1]),
           'dims': {k: int(v.shape[1]) for k, v in {**fps, **acts}.items()},
           'native': {}, 'paired': {}}
    for lname, y in label_sets.items():
        res['native'][lname] = {name: sep.metrics(X, y)
                                for name, X in {**fps, **acts}.items()}
        res['paired'][lname] = {}
        for fname, F in fps.items():
            for aname, A in acts.items():
                res['paired'][lname][f'{fname}__vs__{aname}'] = sep.paired_matched(F, A, y)
    return res


def main():
    models = sys.argv[1:] or ['mlp_even_odd', 'mlp_digit', 'vit_mnist', 'cnn_cifar']
    for m in models:
        t0 = time.time()
        print(f'=== {m} (device {DEVICE}) ===', flush=True)
        ctx = BUILDERS[m]()
        res = evaluate(ctx)
        res['wall_s'] = round(time.time() - t0, 1)
        out = os.path.join(OUT_DIR, f'nb17_last_layer_sep_{m}.json')
        with open(out, 'w') as f:
            json.dump(_json(res), f, indent=1)
        print(f'--- {m}: wrote {out} ({res["wall_s"]}s)')
        for lname, block in res['native'].items():
            row = '  '.join(f'{k}={v["sil"]:.3f}' for k, v in block.items())
            print(f'  [{lname}] {row}', flush=True)


if __name__ == '__main__':
    main()
