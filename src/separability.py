"""Fingerprint vs. activation separability — the C1.8 analysis, as a library.

Answers: is the BFT factor fingerprint a more class-separable stimulus code than
the network's own activations, and *where in the tree* does that separability live?
Lifted from notebook 16 so it can run inside the per-model notebooks (01-05).

Key entry points
----------------
    metrics(X, y)                 -> {'sil', 'knn', 'dim'}   cosine silhouette + 3-fold kNN
    activation_reps(layer_inputs) -> {'penult', 'full'}      the two activation baselines
    fingerprint_slices(tree, ...) -> {name: (N,d) array}     whole tree + upper/lower slices
    paired_matched(A, B, y)       -> dim-matched (PCA & GRP) comparison of two codes
    evaluate(tree, y, layer_inputs) -> full per-tree result dict

All silhouettes are cosine silhouettes on L2-normalized rows; matching reduces the
*larger* code to the smaller's width by PCA and by Gaussian random projection, so a
win cannot be a dimensionality artifact.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics import silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from .arbors import activation_matrix
from .fingerprint_utils import extract_fingerprint_matrix


# ── metrics ───────────────────────────────────────────────────────────────────

def knn_cv(X, y, kmax=5, cv_max=3):
    """3-fold CV kNN accuracy, robust to small / non-contiguous classes."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    mn = int(counts.min())
    if len(classes) < 2 or mn < 2:
        return float('nan')
    cv = int(min(cv_max, mn))
    k = int(max(1, min(kmax, mn)))
    return float(cross_val_score(KNeighborsClassifier(k), X, y, cv=cv).mean())


def metrics(X, y):
    """Cosine silhouette + kNN accuracy of code X under labels y."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] == 0:
        return {'sil': float('nan'), 'knn': float('nan'), 'dim': 0}
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 12:
        return {'sil': float('nan'), 'knn': knn_cv(Xn, y), 'dim': int(X.shape[1])}
    return {'sil': float(silhouette_score(Xn, y, metric='cosine')),
            'knn': knn_cv(Xn, y), 'dim': int(X.shape[1])}


# ── representations ───────────────────────────────────────────────────────────

def _pool(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 4:
        return a.mean((2, 3))
    if a.ndim == 3:
        return a.mean(1)
    return a.reshape(len(a), -1)


def activation_reps(layer_inputs):
    """penultimate (classifier input) and full-concatenation activation baselines."""
    layers = [_pool(li) for li in layer_inputs]
    return {'penult': layers[-1], 'full': np.concatenate(layers, axis=1)}


def _slice_predicates(L_out):
    half = L_out // 2
    return {
        'full':        lambda li, p: True,
        'output_only': lambda li, p: li == L_out,
        'top2':        lambda li, p: li >= L_out - 1,
        'top_half':    lambda li, p: li >= L_out - half,
        'no_output':   lambda li, p: li != L_out,
        'bottom_half': lambda li, p: li <= half,
        'input_only':  lambda li, p: li == 0,
        'spine':       lambda li, p: all(int(x) == 0 for x in p),
    }


def _fp_by_predicate(tree, indices, keep):
    from .types import BFTResult
    root = tree.root if isinstance(tree, BFTResult) else tree
    parts, queue = [], [root]
    while queue:
        nd = queue.pop(0)
        if keep(int(nd.layer_idx), tuple(nd.path)):
            parts.append(np.asarray(nd.img_factors)[indices, :])
        queue.extend(nd.children)
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(indices), 0))


def fingerprint_slices(tree, n_rows, with_per_layer=True):
    """{slice_name: (n_rows, d)} — whole tree, upper/lower slices, and per-layer."""
    from .types import BFTResult
    root = tree.root if isinstance(tree, BFTResult) else tree
    L_out = max(int(nd.layer_idx) for nd in _iter_nodes(root))
    idx = np.arange(n_rows)
    out = {name: _fp_by_predicate(tree, idx, pred)
           for name, pred in _slice_predicates(L_out).items()}
    if with_per_layer:
        for li in range(L_out + 1):
            out[f'L{li}'] = _fp_by_predicate(tree, idx,
                                             (lambda l_: (lambda l, p: l == l_))(li))
    return {k: v for k, v in out.items() if v.shape[1] > 0}


def _iter_nodes(root):
    queue = [root]
    while queue:
        nd = queue.pop(0)
        yield nd
        queue.extend(nd.children)


# ── dim-matched comparison ────────────────────────────────────────────────────

def _match(X, d, how, seed=0):
    X = np.asarray(X, dtype=np.float32)
    d = int(min(d, X.shape[1], max(1, X.shape[0] - 1)))
    if d >= X.shape[1]:
        return X
    if how == 'pca':
        return PCA(n_components=d, random_state=0).fit_transform(X)
    return GaussianRandomProjection(n_components=d, random_state=seed).fit_transform(X)


def paired_matched(A, B, y, grp_seed=0):
    """A vs B at native + PCA-matched + GRP-matched to min(dim). A=fingerprint, B=activation."""
    dm = int(min(A.shape[1], B.shape[1]))
    out = {'match_dim': dm, 'A_native': metrics(A, y), 'B_native': metrics(B, y)}
    for how in ('pca', 'grp'):
        out[f'A_{how}'] = metrics(_match(A, dm, how, grp_seed), y)
        out[f'B_{how}'] = metrics(_match(B, dm, how, grp_seed), y)
    return out


# ── top-level ─────────────────────────────────────────────────────────────────

def evaluate(tree, y, layer_inputs, grp_seed=0):
    """Native sep of every activation baseline + fingerprint slice, plus the paired
    dim-matched fingerprint-vs-activation comparisons. Mirrors nb16's per-tree row."""
    y = np.asarray(y).astype(int)
    n_rows = tree.root.img_factors.shape[0]
    acts = activation_reps(layer_inputs)
    slices = fingerprint_slices(tree, n_rows)

    native = {}
    for name, X in {**{f'act_{k}': v for k, v in acts.items()},
                    **{f'fp_{k}': v for k, v in slices.items()}}.items():
        native[name] = metrics(X, y)

    pairs = {}
    for aname, A in acts.items():
        pairs[f'fp_full__vs__act_{aname}'] = paired_matched(slices['full'], A, y, grp_seed)
    for sname in ('output_only', 'top_half', 'spine', 'bottom_half'):
        if sname in slices:
            pairs[f'fp_{sname}__vs__act_penult'] = paired_matched(
                slices[sname], acts['penult'], y, grp_seed)

    return {'n_rows': int(n_rows), 'fp_full_dim': int(slices['full'].shape[1]),
            'penult_dim': int(acts['penult'].shape[1]),
            'full_dim': int(acts['full'].shape[1]),
            'native': native, 'paired': pairs}


def weight_term_control(tree, y, layer_inputs, pool_method='avg'):
    """Weight-term control: NMF on the arbor vs on activations alone, at matched rank.

    Concatenates, per traced layer, the loadings of an NMF fit on the arbor and,
    separately, on the activation-only matrix at the same rank, and reports the
    separability of each concatenation. Isolates what the *weight* term buys.
    """
    from .arbors import nodes_by_layer
    from .bft import run_nmf_minibatch, normalize_factors
    y = np.asarray(y).astype(int)
    nbl = nodes_by_layer(tree)
    arbor_parts, act_parts = [], []
    for li in sorted(nbl):
        nd = nbl[li]
        k = int(nd.img_factors.shape[1])
        A = activation_matrix(nd, layer_inputs[li])
        Aclip = np.clip(A, 0, None).astype(np.float32)
        Wa, _, _ = run_nmf_minibatch(Aclip, k, init='random', max_iter=300)
        act_parts.append(Wa)
        arbor_parts.append(np.asarray(nd.img_factors))          # arbor loadings, already fit
    fp_arbor = np.concatenate(arbor_parts, axis=1)
    fp_act = np.concatenate(act_parts, axis=1)
    return {'arbor_nmf': metrics(fp_arbor, y), 'activation_nmf': metrics(fp_act, y)}
