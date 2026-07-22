"""Plot-data bundles: the hand-off between where the data is and where figures are made.

The analysis notebooks run where the checkpoints and datasets live (cluster). The
paper figures are drawn from a *bundle* — a nested dict of plain numpy arrays and
scalars written to ``figures/figdata/`` — so ``src/paper_figures.py`` needs only
numpy + matplotlib and can run anywhere the repo is checked out.

    # notebook, next to the analysis state
    D = figdata.save('nb01_circuits', dict(layer_sizes=[8, 4, 2], circuits=[...]))

    # anywhere else
    D = figdata.load('nb01_circuits')
    figdata.summary('nb01_circuits')        # keys, shapes, dtypes — inspect blind

Format: ``<name>.npz`` for arrays plus ``<name>.json`` for scalars, strings and
labels. Deliberately not pickle — the bundle must not depend on this package's
classes, must stay inspectable (``np.load`` / ``jq``), and must be safe to commit.

Contract: bundles hold ndarrays, python scalars, strings, None, and lists/dicts of
those. Anything else (a ``BFTNode``, a DataLoader, a matplotlib object) raises, so
the export can never silently drag the analysis stack into the figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

FIGDATA_DIR = Path(__file__).resolve().parents[1] / 'figures' / 'figdata'

_SCALARS = (str, bool, int, float, np.integer, np.floating, np.bool_)


# ── flatten / unflatten ───────────────────────────────────────────────────────

def _flatten(obj, prefix, arrays, meta):
    """Split a nested structure into {key: ndarray} and a JSON-able {key: value}."""
    if obj is None or isinstance(obj, _SCALARS):
        meta[prefix] = obj.item() if isinstance(obj, np.generic) else obj
    elif isinstance(obj, np.ndarray):
        arrays[prefix] = obj.astype(np.float32) if obj.dtype == np.float64 else obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(v, f'{prefix}.{k}' if prefix else str(k), arrays, meta)
    elif isinstance(obj, (list, tuple)):
        # homogeneous numeric list -> one array; ragged / structured -> index keys
        try:
            arr = np.asarray(obj)
            homogeneous = arr.dtype != object
        except ValueError:
            homogeneous = False
        if homogeneous and arr.dtype.kind in 'fiub':
            _flatten(arr, prefix, arrays, meta)
        elif all(isinstance(o, str) for o in obj):
            meta[prefix] = list(obj)
        else:
            meta[f'{prefix}.__list__'] = len(obj)
            for i, v in enumerate(obj):
                _flatten(v, f'{prefix}.{i}', arrays, meta)
    else:
        raise TypeError(
            f"figdata: key {prefix!r} holds {type(obj).__name__}, which cannot go in a "
            "bundle. Export plain arrays/scalars instead (precompute in the notebook).")


def _unflatten(flat):
    """Rebuild the nested structure; keys recorded as lists come back as lists."""
    out: dict = {}
    lists = {k[: -len('.__list__')]: v for k, v in flat.items() if k.endswith('.__list__')}
    for key, val in flat.items():
        if key.endswith('.__list__'):
            continue
        node, parts = out, key.split('.')
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    def _relist(node, prefix=''):
        if not isinstance(node, dict):
            return node
        node = {k: _relist(v, f'{prefix}.{k}' if prefix else k) for k, v in node.items()}
        if prefix in lists:
            return [node[str(i)] for i in range(lists[prefix])]
        return node
    return _relist(out)


# ── public API ────────────────────────────────────────────────────────────────

def path(name: str, root: Path | str | None = None) -> Path:
    return Path(root or FIGDATA_DIR) / name


def save(name: str, data: dict, root: Path | str | None = None) -> dict:
    """Write a bundle and return `data` unchanged (so cells can do D = figdata.save(...))."""
    arrays: dict = {}
    meta: dict = {}
    _flatten(data, '', arrays, meta)
    p = path(name, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p.with_suffix('.npz'), **arrays)
    with open(p.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    kb = p.with_suffix('.npz').stat().st_size / 1024
    print(f'[figdata] {name}: {len(arrays)} arrays, {len(meta)} scalars, {kb:.0f} kB')
    return data


def load(name: str, root: Path | str | None = None) -> dict:
    p = path(name, root)
    if not p.with_suffix('.npz').exists():
        raise FileNotFoundError(
            f'{p.with_suffix(".npz")} not found — run the notebook section that builds '
            f'{name!r} (on the machine that has the data) and commit the bundle.')
    flat = dict(np.load(p.with_suffix('.npz')))
    with open(p.with_suffix('.json')) as f:
        flat.update(json.load(f))
    return _unflatten(flat)


def summary(name: str, root: Path | str | None = None) -> None:
    """Print every leaf of a bundle — enough to design a panel without the notebook."""
    p = path(name, root)
    with np.load(p.with_suffix('.npz')) as z:
        rows = [(k, f'{z[k].dtype}{list(z[k].shape)}') for k in sorted(z.files)]
        total = sum(z[k].nbytes for k in z.files)
    with open(p.with_suffix('.json')) as f:
        meta = json.load(f)
    print(f'── {name}  ({len(rows)} arrays, {total / 1024:.0f} kB in memory)')
    for k, v in rows:
        print(f'   {k:52s} {v}')
    for k in sorted(meta):
        if not k.endswith('__list__'):
            v = meta[k]
            v = f'[{len(v)} items]' if isinstance(v, list) and len(v) > 6 else repr(v)
            print(f'   {k:52s} {v}')


def available(root: Path | str | None = None) -> list[str]:
    d = Path(root or FIGDATA_DIR)
    return sorted(p.stem for p in d.glob('*.npz')) if d.exists() else []
