"""Disk caching for expensive analysis artifacts (BFT trees and JSON results).

A BFT trace over a scaled population is minutes to hours; the validation / pruning
/ separability analyses that follow all consume the *same* fitted tree. This module
lets a notebook fit a tree once, cache it to ``data/cache/``, and reuse it on the
next run — keyed by an explicit tag plus a hash of the hyperparameters, so changing
the HPs (or passing ``force=True``) transparently invalidates the cache.

Usage
-----
    from src.cache_utils import cached_tree, cached_result

    tree = cached_tree('nb03_circuit_seed0',
                       lambda: bft(layer_dicts, k_max=K_MAX, n_branches=N_BRANCHES, ...),
                       params=dict(k_max=K_MAX, n_branches=N_BRANCHES, tau=STIM_THRESHOLD,
                                   n_stim=len(targets)))

    res = cached_result('nb03_validation_seed0',
                        lambda: run_validation(tree, ...),
                        params=dict(...))

Set ``NB_NOCACHE=1`` in the environment to bypass all caching (always recompute,
never write). Pass ``force=True`` to recompute a single entry and overwrite it.
"""
import os
import json
import pickle
import hashlib

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(_REPO, 'data', 'cache')


def _nocache():
    return os.environ.get('NB_NOCACHE', '0') == '1'


def _hp_hash(params):
    """Short stable hash of a params dict (order-independent, ndarray-aware)."""
    if not params:
        return 'none'

    def canon(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, dict):
            return {str(k): canon(v) for k, v in sorted(o.items())}
        if isinstance(o, (list, tuple)):
            return [canon(v) for v in o]
        return o

    blob = json.dumps(canon(params), sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def cache_path(tag, params=None, ext='pkl'):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f'{tag}__{_hp_hash(params)}.{ext}')


def cached_tree(tag, build_fn, *, params=None, force=False, verbose=True):
    """Return a BFT tree from cache, or build it with ``build_fn()`` and cache it.

    ``params`` should capture everything that determines the tree (HPs, population
    size, seed); its hash is part of the cache filename, so a change recomputes.
    """
    p = cache_path(tag, params, 'pkl')
    if not _nocache() and not force and os.path.exists(p):
        if verbose:
            print(f'  [cache] load tree {os.path.relpath(p, _REPO)}')
        with open(p, 'rb') as f:
            return pickle.load(f)
    if verbose:
        print(f'  [cache] build tree {tag} …')
    tree = build_fn()
    if not _nocache():
        tmp = p + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
        if verbose:
            print(f'  [cache] saved {os.path.relpath(p, _REPO)}')
    return tree


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o if isinstance(o, (float, int, str)) or o is None else str(o)


def cached_result(tag, compute_fn, *, params=None, force=False, verbose=True):
    """Return a JSON-serializable result dict from cache, or compute + cache it."""
    p = cache_path(tag, params, 'json')
    if not _nocache() and not force and os.path.exists(p):
        if verbose:
            print(f'  [cache] load result {os.path.relpath(p, _REPO)}')
        with open(p) as f:
            return json.load(f)
    if verbose:
        print(f'  [cache] compute result {tag} …')
    res = compute_fn()
    if not _nocache():
        tmp = p + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(_jsonable(res), f, indent=1)
        os.replace(tmp, p)
        if verbose:
            print(f'  [cache] saved {os.path.relpath(p, _REPO)}')
    return _jsonable(res)


def clear_cache(tag_prefix=''):
    """Delete cached files whose tag starts with ``tag_prefix`` (all if empty)."""
    if not os.path.isdir(CACHE_DIR):
        return 0
    n = 0
    for fn in os.listdir(CACHE_DIR):
        fp = os.path.join(CACHE_DIR, fn)
        if fn.startswith(tag_prefix) and os.path.isfile(fp):
            os.remove(fp)
            n += 1
    return n
