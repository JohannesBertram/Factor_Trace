"""Hypothesis-tailored stimulus sets for BFT ("dataset probes").

Standard BFT factorizes each layer's joint arbor matrix over a broad,
class-balanced stimulus set — the SAE end of the interpretability spectrum:
unsupervised, and the NMF rank budget goes to the dominant global structure.
This module moves BFT toward the linear-probe / TCAV end *without changing the
method*: the only supervision is the choice of stimuli. Because the joint arbor
rows are stimuli, enriching the set with a hypothesis class (all bears, one dog
breed group, "striped things") reallocates the factorization's reconstruction
budget onto the circuitry that processes those stimuli, so sub-structure the
broad trace blurs into one factor becomes resolvable.

Pipeline (see notebooks/06_tailored_imagenet.ipynb and TAILORED_BFT.md):
    hyp   = HYPOTHESES['bears']                        # or build your own
    data  = collect_hypothesis_data(model, val_ds, hyp, device=DEVICE,
                                    layer_filter=spine_filter)
    tree  = bft(data['layer_data'], ...)
    prof  = factor_label_profile(tree.root.img_factors, data['fine'])
    corr  = match_factors(tree.root, broad_tree.root)  # refinement vs rotation
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .bft import collect_layer_dicts

__all__ = [
    'imagenet_class_names', 'find_imagenet_classes',
    'collect_hypothesis_data', 'subsample_data', 'mixture_indices',
    'factor_label_profile', 'substructure_scores',
    'match_factors', 'refinement_score',
]


# ── ImageNet class lookup (torchvision metadata, no download) ─────────────────

def imagenet_class_names():
    """The 1000 human-readable ImageNet class names, indexed by class id."""
    from torchvision.models import SqueezeNet1_1_Weights
    return list(SqueezeNet1_1_Weights.IMAGENET1K_V1.meta['categories'])


def find_imagenet_classes(query, class_names=None):
    """Case-insensitive substring search -> {class_id: name}.

    >>> find_imagenet_classes('terrier')   # doctest: +SKIP
    {158: 'toy terrier', 179: 'Staffordshire bullterrier', ...}
    """
    if class_names is None:
        class_names = imagenet_class_names()
    q = query.lower()
    return {i: c for i, c in enumerate(class_names) if q in c.lower()}


# ── Hypothesis data collection ────────────────────────────────────────────────

def _hyp_tables(hypothesis):
    """(fine_id -> group_idx, group_names, all fine ids sorted)."""
    groups = hypothesis['groups']
    group_names = list(groups)
    fine_to_group = {int(c): gi for gi, ids in enumerate(groups.values())
                     for c in ids}
    return fine_to_group, group_names, sorted(fine_to_group)


def collect_hypothesis_data(model, val_ds, hypothesis, device, layer_filter,
                            n_per_class=40, correctness='strict', select='confident',
                            batch_size=64, num_workers=4, seed=0, verbose=True):
    """Collect spine layer dicts for a tailored (hypothesis-defined) stimulus set.

    Parameters
    ----------
    hypothesis  : dict with 'groups': {group_name: [ImageNet class ids]}.
                  Groups encode the *expected* sub-structure — they are used only
                  for evaluation and coloring, never by the trace itself.
    n_per_class : max stimuli kept per fine ImageNet class (val split has ~50).
    correctness : 'strict' — keep only 1000-way top-1 correct samples;
                  'group'  — keep samples whose top-1 prediction falls anywhere
                             inside the hypothesis's class set (the model "ran
                             the right circuit" even if it missed the species);
                  'none'   — keep everything.
    select      : 'confident' — per fine class, keep the top-confidence samples
                  (matches nb05's convention); 'random' — seeded random sample.

    Returns a dict like ``collect_layer_dicts`` plus:
        'targets'     : group labels 0..G-1 (what plots/metrics group by)
        'fine'        : original ImageNet class ids
        'group_names' : list of group names
        'fine_names'  : {fine_id: readable name}
        'class_names' : {group_idx: group name} (for plot utilities)
    """
    fine_to_group, group_names, fine_ids = _hyp_tables(hypothesis)
    names = imagenet_class_names()

    val_targets = np.array(
        val_ds.targets if hasattr(val_ds, 'targets')
        else [s[1] for s in val_ds.samples])
    keep = np.where(np.isin(val_targets, fine_ids))[0]
    loader = DataLoader(Subset(val_ds, keep), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    if verbose:
        print(f"hypothesis '{hypothesis.get('name', '?')}': "
              f"{len(fine_ids)} fine classes in {len(group_names)} groups, "
              f'{len(keep)} val candidates')

    # Collect everything, filter afterwards — we need the model's predictions
    # (from the classifier layer's output fmap) for correctness='group'.
    raw = collect_layer_dicts(model, loader, device, only_correct=False,
                              layer_filter=layer_filter)

    out_fmap = raw['layer_data'][-1]['output_fmap']       # classifier conv
    preds = (out_fmap.mean(axis=(2, 3)) if out_fmap.ndim == 4
             else out_fmap).argmax(1)
    fine = raw['targets']

    if correctness == 'strict':
        ok = preds == fine
    elif correctness == 'group':
        ok = np.isin(preds, fine_ids)
    elif correctness == 'none':
        ok = np.ones(len(fine), bool)
    else:
        raise ValueError(f'unknown correctness mode: {correctness!r}')

    # Per-fine-class subsampling to n_per_class.
    rng = np.random.default_rng(seed)
    chosen = []
    for c in fine_ids:
        ci = np.where(ok & (fine == c))[0]
        if select == 'confident':
            ci = ci[np.argsort(raw['confidences'][ci])[::-1]]
        else:
            ci = rng.permutation(ci)
        chosen.append(ci[:n_per_class])
    chosen = np.sort(np.concatenate(chosen))

    data = subsample_data(raw, chosen)
    data['fine'] = data['targets']
    data['targets'] = np.array([fine_to_group[int(c)] for c in data['fine']])
    data['preds_1000'] = preds[chosen]
    data['group_names'] = group_names
    data['fine_names'] = {int(c): names[c] for c in fine_ids}
    data['class_names'] = {gi: g for gi, g in enumerate(group_names)}
    if verbose:
        cnt = {names[c]: int((data['fine'] == c).sum()) for c in fine_ids}
        print(f"kept {len(chosen)} stimuli (correctness='{correctness}'): {cnt}")
    return data


def subsample_data(data, keep):
    """Index a collected data dict (images/targets/.../layer_data) by ``keep``."""
    keep = np.asarray(keep)
    out = {k: v[keep] for k, v in data.items()
           if isinstance(v, np.ndarray) and v.ndim >= 1
           and len(v) == len(data['targets'])}
    out['layer_data'] = [{**ld, 'input_fmap': ld['input_fmap'][keep],
                          'output_fmap': ld['output_fmap'][keep]}
                         for ld in data['layer_data']]
    return out


def mixture_indices(fine_labels, hyp_ids, ref_ids, rho, n_total, seed=0):
    """Indices for a dose-response mixture: fraction ``rho`` hypothesis stimuli.

    Draws round(rho * n_total) samples from the hypothesis classes and the rest
    from the reference classes, each spread as evenly as possible across their
    fine classes. Total N stays fixed so enrichment is not confounded with
    sample size.
    """
    fine_labels = np.asarray(fine_labels)
    rng = np.random.default_rng(seed)

    def draw(ids, n):
        ids = [c for c in ids if (fine_labels == c).sum() > 0]
        per, extra = divmod(n, len(ids))
        take = []
        for j, c in enumerate(rng.permutation(ids)):
            ci = np.where(fine_labels == c)[0]
            k = min(per + (j < extra), len(ci))
            take.append(rng.choice(ci, k, replace=False))
        return np.concatenate(take) if take else np.array([], int)

    n_hyp = int(round(rho * n_total))
    idx = np.concatenate([draw(hyp_ids, n_hyp), draw(ref_ids, n_total - n_hyp)])
    return np.sort(idx)


# ── Sub-structure metrics ─────────────────────────────────────────────────────

def factor_label_profile(img_factors, labels):
    """Per-factor loading mass over labels.

    Returns dict with:
        'mass'    : (K, n_labels) — column j of factor k = share of factor k's
                    total loading carried by label j (rows sum to 1)
        'labels'  : sorted unique labels (columns of 'mass')
        'purity'  : (K,) max share — 1.0 means the factor loads on one label only
        'entropy' : (K,) normalized entropy of the mass distribution (0 = pure)
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    W = np.asarray(img_factors, float)
    mass = np.stack([W[labels == u].sum(0) for u in uniq], axis=1)   # (K, L)
    tot = mass.sum(1, keepdims=True)
    mass = mass / np.maximum(tot, 1e-12)
    ent = -(mass * np.log(np.maximum(mass, 1e-12))).sum(1)
    ent = ent / np.log(len(uniq)) if len(uniq) > 1 else ent * 0
    return {'mass': mass, 'labels': uniq,
            'purity': mass.max(1), 'entropy': ent}


def substructure_scores(F, labels, n_clusters=None, knn=5, seed=0):
    """How well a representation ``F`` (n, D) resolves ``labels``.

    Returns dict(silhouette, knn_acc, kmeans_ari, n, d). Cosine metric
    throughout (fingerprints are non-negative and scale-free).
    """
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from sklearn.cluster import KMeans
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    F = np.asarray(F, float)
    labels = np.asarray(labels)
    keep = np.linalg.norm(F, axis=1) > 0
    F, labels = F[keep], labels[keep]
    uniq = np.unique(labels)
    if n_clusters is None:
        n_clusters = len(uniq)

    sil = float(silhouette_score(F, labels, metric='cosine')) \
        if len(uniq) > 1 else float('nan')
    Fn = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(Fn)
    ari = float(adjusted_rand_score(labels, km.labels_))
    knn_acc = float(np.mean(cross_val_score(
        KNeighborsClassifier(n_neighbors=knn, metric='cosine'),
        F, labels, cv=5))) if len(uniq) > 1 else float('nan')
    return {'silhouette': sil, 'knn_acc': knn_acc, 'kmeans_ari': ari,
            'n': int(len(F)), 'd': int(F.shape[1])}


# ── Factor correspondence between two trees ───────────────────────────────────

def _cf_matrix(node):
    """Connection factors as (K, n_features), L2-normalized rows."""
    H = np.asarray(node.connection_factors, float).T
    return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-12)


def match_factors(node_a, node_b):
    """Cosine correspondence between two nodes' connection factors.

    Both nodes must sit at the same layer of the same model (the arbor feature
    space must match). Typical use: the tailored tree's root vs the broad
    (nb05) tree's root.

    Returns dict with:
        'sim'    : (K_a, K_b) cosine similarity matrix
        'match'  : (K_a,) index of the best-matching b-factor per a-factor
        'assign' : Hungarian assignment [(i_a, i_b), ...] on the square part
    """
    A, B = _cf_matrix(node_a), _cf_matrix(node_b)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f'arbor feature spaces differ: {A.shape[1]} vs '
                         f'{B.shape[1]} — nodes are not at the same layer')
    sim = A @ B.T
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-sim)
    return {'sim': sim, 'match': sim.argmax(1),
            'assign': list(zip(r.tolist(), c.tolist()))}


def refinement_score(sim):
    """Is the a-factorization a *refinement* of b (each a-factor descends from
    one b-factor) or a *rotation* (a-factors mix several b-factors)?

    Per a-factor: the top-vs-second-best margin rescaled by the row's cosine
    range, (top - second) / (top - min). This removes the cosine floor that
    non-negative factor vectors share (random non-negative vectors already have
    high mutual cosine). 1.0 = clean refinement, ~0 = rotation. Compare against
    the 'random_control' hypothesis rather than reading the value absolutely.
    Returns (per_factor, mean).
    """
    s = np.sort(np.asarray(sim), axis=1)[:, ::-1]
    if s.shape[1] < 2:
        return np.ones(len(s)), 1.0
    per = (s[:, 0] - s[:, 1]) / np.maximum(s[:, 0] - s[:, -1], 1e-12)
    return per, float(per.mean())
