"""Master statistics/bootstrap script for the reproducibility checklist.

Recomputes every headline number from the committed figdata bundles, attaches a
95% percentile bootstrap CI (resample stimuli with replacement), and runs the
paired cross-model and permutation tests. Writes results/stats.json.

No BFT, no models, no datasets — bundles only.
"""
import sys, json, os
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import wilcoxon, binomtest, norm

from src import figdata
from src.paper_figures import (cosine_silhouette, fp_vs_act, support_overlap,
                               unit_support, nodes_by_path, _circuit_colors,
                               lam_weighted, unit)

RNG_SEED = 0
B = 2000                      # bootstrap resamples
OUT = {}

# ── fast vectorized cosine silhouette + weighted bootstrap ────────────────────
#
# Bootstrap with replacement = draw a multiset of rows. A row drawn w times gives
# identical fingerprints (distance 0 among its copies), so the whole resample is
# summarized by the per-row draw count w (bincount of the index draw). The
# silhouette of the resample is then a weighted average of per-row silhouettes,
# each computed from cluster-distance sums Dm @ Wmat — one BLAS matmul per draw,
# no O(n^2) fancy-index gather. This is exact (identical to slicing Dm[idx][:,idx]
# and calling cosine_silhouette), just far faster.

def _cos_dist(X):
    U = unit(np.asarray(X, float))
    Dm = 1.0 - U @ U.T
    np.clip(Dm, 0, 2, out=Dm)
    np.fill_diagonal(Dm, 0.0)
    return Dm


def _sil_weighted(Dm, yidx, C, w):
    """Mean cosine silhouette of a bootstrap resample given per-row draw counts w.
    yidx: integer label indices (0..C-1). Returns weighted mean over drawn rows."""
    Wmat = np.zeros((len(w), C))
    for c in range(C):                        # Wmat[q,c] = w_q if row q is class c
        Wmat[yidx == c, c] = w[yidx == c]
    M = Dm @ Wmat                             # M[p,c] = sum_q w_q [y_q=c] Dm[p,q]
    Nc = Wmat.sum(0)                          # drawn count per cluster
    own = M[np.arange(len(w)), yidx]
    n_own = Nc[yidx]
    a = np.where(n_own > 1, own / np.maximum(n_own - 1, 1), 0.0)
    Mmean = M / np.maximum(Nc[None, :], 1e-12)
    Mmean[np.arange(len(w)), yidx] = np.inf
    b = Mmean.min(1)
    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / denom, 0.0)
    tot = w.sum()
    return float((w * s).sum() / tot) if tot > 0 else 0.0


def _prep(X, y):
    Dm = _cos_dist(X)
    uniq, yidx = np.unique(np.asarray(y), return_inverse=True)
    return Dm, yidx, len(uniq)


def boot_silhouette(X, y, n_boot=B, seed=RNG_SEED):
    Dm, yidx, C = _prep(X, y)
    n = len(yidx)
    point = _sil_weighted(Dm, yidx, C, np.ones(n))
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for it in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        vals[it] = _sil_weighted(Dm, yidx, C, w)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return dict(point=point, lo=float(lo), hi=float(hi), mean=float(vals.mean()),
                std=float(vals.std()))


def boot_silhouette_paired(Xf, yf, Xa, ya, n_boot=B, seed=RNG_SEED):
    """Paired bootstrap: same row-draw counts drive both fp and act silhouettes."""
    Df, yfi, Cf = _prep(Xf, yf)
    Da, yai, Ca = _prep(Xa, ya)
    n = len(yfi)
    pf = _sil_weighted(Df, yfi, Cf, np.ones(n))
    pa = _sil_weighted(Da, yai, Ca, np.ones(n))
    rng = np.random.default_rng(seed)
    fp_v, act_v, diff_v = (np.empty(n_boot) for _ in range(3))
    for it in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        sf = _sil_weighted(Df, yfi, Cf, w)
        sa = _sil_weighted(Da, yai, Ca, w)
        fp_v[it], act_v[it], diff_v[it] = sf, sa, sf - sa
    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return dict(lo=float(lo), hi=float(hi), mean=float(v.mean()), std=float(v.std()))
    return dict(fp=dict(point=pf, **ci(fp_v)), act=dict(point=pa, **ci(act_v)),
                diff=dict(point=pf - pa, **ci(diff_v)),
                frac_fp_gt_act=float(np.mean(diff_v > 0)))


# ══ 1. Native-dim fingerprint silhouettes (figfp_structure a-e; Fig 4) ═════════
print('=== 1. native-dim fingerprint silhouettes ===')
FP = {}
sil_res = {}

# parity models: parity (id_targets) + digit (id_digits)
for key, bundle, name in [('mlp_even_odd', 'nb01_fingerprints', '8x4 MLP'),
                          ('vit', 'nb04_fingerprints', 'TinyViT')]:
    D = figdata.load(bundle); fp = D['fp']
    r_par = boot_silhouette(fp['id'], fp['id_targets'])
    r_dig = boot_silhouette(fp['id'], fp['id_digits'])
    sil_res[key] = dict(name=name, parity=r_par, digit=r_dig)
    print(f'{name:10s} parity {r_par["point"]:.3f} [{r_par["lo"]:.3f},{r_par["hi"]:.3f}]'
          f'  digit {r_dig["point"]:.3f} [{r_dig["lo"]:.3f},{r_dig["hi"]:.3f}]')

# class models: id_targets
for key, bundle, name in [('mlp_digit', 'nb02_fingerprints', 'digit MLP'),
                          ('cifar', 'nb03_fingerprints', 'CIFAR-10 CNN'),
                          ('imagenet', 'nb05_fingerprints', 'ImageNet')]:
    D = figdata.load(bundle); fp = D['fp']
    r = boot_silhouette(fp['id'], fp['id_targets'])
    sil_res[key] = dict(name=name, cls=r)
    print(f'{name:12s} class {r["point"]:.3f} [{r["lo"]:.3f},{r["hi"]:.3f}]')
OUT['fingerprint_silhouette'] = sil_res

# ══ 2. fp vs act native-dim paired (CIFAR & ImageNet; Fig 4 d/e/g/h/i) ═════════
print('\n=== 2. fp vs act native-dim (paired) ===')
fpact = {}
for key, bundle, name in [('cifar', 'nb03_fingerprints', 'CIFAR-10 CNN'),
                          ('imagenet', 'nb05_fingerprints', 'ImageNet')]:
    D = figdata.load(bundle)
    Xf, lf, Xa, la, rep = fp_vs_act(D, D['fp']['id_targets'])
    r = boot_silhouette_paired(Xf, lf, Xa, la)
    r['aligned'] = int(rep['aligned']); r['name'] = name
    r['fp_dim'] = int(Xf.shape[1]); r['act_dim'] = int(Xa.shape[1])
    fpact[key] = r
    print(f'{name:12s} fp {r["fp"]["point"]:.3f} [{r["fp"]["lo"]:.3f},{r["fp"]["hi"]:.3f}]'
          f'  act {r["act"]["point"]:.3f} [{r["act"]["lo"]:.3f},{r["act"]["hi"]:.3f}]'
          f'  diff {r["diff"]["point"]:.3f} [{r["diff"]["lo"]:.3f},{r["diff"]["hi"]:.3f}]'
          f'  P(fp>act)={r["frac_fp_gt_act"]:.3f}')
OUT['fp_vs_act_native'] = fpact

# ══ 3. within-class / between-class cosine (Fig 4f, CIFAR) ═════════════════════
print('\n=== 3. within/between-class cosine (CIFAR Fig 4f) ===')

def boot_within_between(X, y, n_boot=B, seed=RNG_SEED):
    X = np.asarray(X, float); y = np.asarray(y)
    U = unit(X); S = U @ U.T
    uniq = np.unique(y)
    def wb(SS, yy):
        within = np.mean([SS[np.ix_(yy == c, yy == c)].mean() for c in uniq
                          if (yy == c).sum() > 0])
        between = SS[yy[:, None] != yy[None, :]].mean()
        return within, between
    w0, b0 = wb(S, y)
    rng = np.random.default_rng(seed); n = len(y)
    wv, bv = np.empty(n_boot), np.empty(n_boot)
    for it in range(n_boot):
        idx = rng.integers(0, n, n)
        wv[it], bv[it] = wb(S[np.ix_(idx, idx)], y[idx])
    def ci(v, p):
        lo, hi = np.percentile(v, [2.5, 97.5]); return dict(point=float(p), lo=float(lo), hi=float(hi))
    return dict(within=ci(wv, w0), between=ci(bv, b0))

Dc = figdata.load('nb03_fingerprints')
r = boot_within_between(Dc['fp']['id'], Dc['fp']['id_targets'])
OUT['within_between_cosine_cifar'] = r
print(f'CIFAR within {r["within"]["point"]:.3f} [{r["within"]["lo"]:.3f},{r["within"]["hi"]:.3f}]'
      f'  between {r["between"]["point"]:.3f} [{r["between"]["lo"]:.3f},{r["between"]["hi"]:.3f}]')

# ══ 4. class/category purity per layer + output-layer vs chance ════════════════
print('\n=== 4. lambda-weighted purity per layer ===')

def layer_purity_ci(circ_bundle, fp_bundle, chance, n_boot=B, seed=RNG_SEED):
    """Per BFT layer: lambda-weighted class purity + bootstrap CI over stimuli.
    Reconstructs class_profile from per-stimulus img_factors + labels so the
    resample is honest. Labels come from the aligned fingerprint bundle."""
    Dc = figdata.load(circ_bundle); Df = figdata.load(fp_bundle)
    labels_full = np.asarray(Df['fp']['id_targets'])
    classes = np.unique(labels_full)
    # group nodes by layer_idx (output layer = max idx)
    nodes = Dc['nodes']
    layer_ids = sorted({int(n['layer_idx']) for n in nodes}, reverse=True)  # output first
    out = []
    for li in layer_ids:
        lnodes = [n for n in nodes if int(n['layer_idx']) == li]
        packs = []                       # (H, lam_share, labels) per node
        for nd in lnodes:
            H = np.asarray(nd['img_factors'], float)     # (n_kept, K)
            lam = np.asarray(nd['lam_share'], float)
            if 'stim_idx' in nd and H.shape[0] != len(labels_full):
                lab = labels_full[np.asarray(nd['stim_idx'])]
            else:
                lab = labels_full
            packs.append((H, lam, lab))
        lengths = {H.shape[0] for H, _, _ in packs}
        joint = len(lengths) == 1        # all nodes share one stimulus population

        def weighted_purity(ridx_shared=None):
            purs, lams = [], []
            for H, lam, lab in packs:
                if ridx_shared is None:
                    Hs, labs = H, lab
                else:
                    ridx = ridx_shared if joint else \
                        rng_local.integers(0, len(lab), len(lab))
                    Hs, labs = H[ridx], lab[ridx]
                prof = np.zeros((H.shape[1], len(classes)))
                for j, c in enumerate(classes):
                    mm = labs == c
                    if mm.any():
                        prof[:, j] = Hs[mm].mean(0)
                prof = prof / (prof.sum(1, keepdims=True) + 1e-12)
                purs.append(prof.max(1)); lams.append(lam)
            purs = np.concatenate(purs); lams = np.concatenate(lams)
            return lam_weighted(purs, lams)

        point = weighted_purity(None)
        rng_local = np.random.default_rng(seed + li)
        n0 = packs[0][0].shape[0]
        vals = np.empty(n_boot)
        for it in range(n_boot):
            ridx = rng_local.integers(0, n0, n0) if joint else True
            vals[it] = weighted_purity(ridx)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        rec = dict(layer_idx=li, point=float(point), lo=float(lo), hi=float(hi),
                   n_factors=int(sum(len(p[1]) for p in packs)), joint=int(joint))
        # shuffled-label null (the "vs chance" test), output layer only
        if li == layer_ids[0]:
            rng_s = np.random.default_rng(seed + 999)
            null = np.empty(n_boot)
            for it in range(n_boot):
                purs, lams = [], []
                for H, lam, lab in packs:
                    labs = rng_s.permutation(lab)
                    prof = np.zeros((H.shape[1], len(classes)))
                    for j, c in enumerate(classes):
                        mm = labs == c
                        if mm.any():
                            prof[:, j] = H[mm].mean(0)
                    prof = prof / (prof.sum(1, keepdims=True) + 1e-12)
                    purs.append(prof.max(1)); lams.append(lam)
                null[it] = lam_weighted(np.concatenate(purs), np.concatenate(lams))
            rec['null_mean'] = float(null.mean()); rec['null_hi'] = float(np.percentile(null, 97.5))
            rec['p_vs_chance'] = float((null >= point).mean())
        out.append(rec)
    return dict(chance=chance, layers=out)

for key, cb, fb, chance, name in [
        ('cifar', 'nb03_circuits', 'nb03_fingerprints', 0.10, 'CIFAR'),
        ('imagenet', 'nb05_circuits', 'nb05_fingerprints', 0.125, 'ImageNet')]:
    r = layer_purity_ci(cb, fb, chance)
    OUT.setdefault('purity', {})[key] = r
    print(f'-- {name} (chance {chance}) output-first:')
    for L in r['layers']:
        print(f'   L_idx {L["layer_idx"]}: {L["point"]:.3f} [{L["lo"]:.3f},{L["hi"]:.3f}]')

# ══ 5. paired cross-model tests: fp vs act (Fig P-d) & weight term (Fig P-e) ═══
print('\n=== 5. paired cross-model tests (5 models) ===')
VAL = [('MLP even/odd', 'nb09_mlp_even_odd_validation'),
       ('MLP digit', 'nb09_mlp_digit_validation'),
       ('CIFAR-10 CNN', 'nb09_cnn_cifar_validation'),
       ('TinyViT', 'nb09_vit_mnist_validation'),
       ('ImageNet', 'nb09_imagenet_cnn_validation')]

def paired_test(pairs, names, tag):
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    d = a - b
    n_pos = int((d > 0).sum()); n = len(d)
    sgn = binomtest(n_pos, n, 0.5, alternative='greater').pvalue
    w_g = wilcoxon(a, b, alternative='greater', zero_method='wilcox')
    try:
        w_two = wilcoxon(a, b, alternative='two-sided', zero_method='wilcox').pvalue
    except Exception:
        w_two = None
    res = dict(names=names, a=a.tolist(), b=b.tolist(), diff=d.tolist(),
               n_pos=n_pos, n=n, sign_p_greater=float(sgn),
               wilcoxon_stat=float(w_g.statistic), wilcoxon_p_greater=float(w_g.pvalue),
               wilcoxon_p_two=(float(w_two) if w_two is not None else None),
               mean_diff=float(d.mean()))
    print(f'-- {tag}: {n_pos}/{n} models fp>act; sign p(greater)={sgn:.4f}; '
          f'Wilcoxon p(greater)={w_g.pvalue:.4f}; mean diff={d.mean():.3f}')
    for nm, x, y in zip(names, a, b):
        print(f'     {nm:14s} {x:.3f} vs {y:.3f}  (diff {x-y:+.3f})')
    return res

fpd, wtc, names = [], [], []
for name, b in VAL:
    D = figdata.load(b)
    sep = D['separability']['by_fine']
    a1 = D['A1_weight_vs_activation']['fingerprint_separability']
    fpd.append((float(sep['bft_fingerprint']['silhouette']),
                float(sep['act_matched']['silhouette'])))
    wtc.append((float(a1['arbor_nmf']['silhouette']),
                float(a1['activation_nmf']['silhouette'])))
    names.append(name)
OUT['paired_fp_vs_act_dimmatched'] = paired_test(fpd, names, 'fp vs dim-matched act (Fig P-d)')
OUT['paired_weight_term_control'] = paired_test(wtc, names, 'arbor NMF vs activation NMF (Fig P-e)')

# ══ 6. support-cosine permutation tests (Fig A-c 8x4 MLP, Fig B-c digit MLP) ═══
print('\n=== 6. support-cosine permutation tests ===')
# 8x4 MLP: two circuits' L1 unit support (figA)
D1 = figdata.load('nb01_circuits')
CIRC = _circuit_colors(D1); NODES = nodes_by_path(D1)
S1 = np.stack([unit_support(NODES[(c['k'],)]) for c in CIRC])
ov1, null1 = support_overlap(S1, n_shuffle=20000)
p_disjoint = float((null1 <= ov1).mean())         # circuits MORE disjoint than chance
z1 = (ov1 - null1.mean()) / null1.std()
OUT['support_overlap_8x4'] = dict(observed=float(ov1), null_mean=float(null1.mean()),
    null_std=float(null1.std()), p_more_disjoint=p_disjoint, z=float(z1))
print(f'8x4 MLP  support cos {ov1:.3f}  null {null1.mean():.3f}±{null1.std():.3f}  '
      f'z={z1:.2f}  p(<=obs)={p_disjoint:.4f}')

# digit MLP: precomputed support matrix (figB)
D2 = figdata.load('nb02_circuits')
S2 = np.asarray(D2['support'])
ov2, null2 = support_overlap(S2, n_shuffle=20000)
p_overlap = float((null2 >= ov2).mean())          # circuits MORE overlapping than chance
z2 = (ov2 - null2.mean()) / null2.std()
OUT['support_overlap_digit'] = dict(observed=float(ov2), null_mean=float(null2.mean()),
    null_std=float(null2.std()), p_more_overlapping=p_overlap, z=float(z2))
print(f'digit MLP support cos {ov2:.3f}  null {null2.mean():.3f}±{null2.std():.3f}  '
      f'z={z2:.2f}  p(>=obs)={p_overlap:.4f}')

# ══ 7. surface existing spreads: stability, k-sensitivity, refit ══════════════
print('\n=== 7. stability / k-sensitivity / refit spreads ===')
stab_all = []; stab_rows = {}; ksens = {}
for name, b in VAL:
    D = figdata.load(b)
    pl = D['stability']['per_layer']
    means = [pl[k]['mean'] for k in sorted(pl, key=int)]
    stab_all.extend(means)
    stab_rows[name] = dict(layer_means=[float(m) for m in means],
                           overall=float(np.mean(means)), min=float(np.min(means)))
    ks = D['stability']['k_sensitivity']
    ksens[name] = dict(k_star=float(np.mean(ks['k_star'])),
                       k_minus1=float(np.mean(ks['k_minus1'])),
                       k_plus1=float(np.mean(ks['k_plus1'])))
stab_all = np.array(stab_all)
OUT['stability'] = dict(per_model=stab_rows, k_sensitivity=ksens,
    all_layers_mean=float(stab_all.mean()), all_layers_std=float(stab_all.std()),
    all_layers_min=float(stab_all.min()), n_layers=int(len(stab_all)),
    gate=0.85, frac_above_gate=float((stab_all >= 0.85).mean()))
print(f'stability across {len(stab_all)} layers: {stab_all.mean():.3f}±{stab_all.std():.3f} '
      f'(min {stab_all.min():.3f}); all >= 0.85 gate: {(stab_all>=0.85).all()}')
kk = np.array([[ksens[n]['k_minus1'], ksens[n]['k_star'], ksens[n]['k_plus1']] for n in names])
print(f'k-sensitivity (mean over models): K*-1={kk[:,0].mean():.3f} K*=1.000 '
      f'K*+1={kk[:,2].mean():.3f}  min(K*-1)={kk[:,0].min():.3f} min(K*+1)={kk[:,2].min():.3f}')

# refit spread (CIFAR root, the causal recon number 0.937)
Dc = figdata.load('nb09_cnn_cifar_validation')
for k, v in Dc['recon_controls'].items():
    if isinstance(v, dict) and '_refit_spread' in v:
        rs = v['_refit_spread']
        runs = [round(float(x), 3) for x in rs['preact_r2_runs']]
        rng_v = float(rs['range'])
        OUT['refit_spread_cifar_root'] = dict(node=k, preact_r2_runs=list(map(float, rs['preact_r2_runs'])),
            median=float(rs['median']), rng=rng_v,
            arbor_r2_pos_range=float(rs['arbor_r2_pos_range']))
        print(f'refit spread @{k}: preact R2 runs {runs} (range {rng_v:.3f})')

_OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'stats_checklist.json')
with open(_OUT_PATH, 'w') as f:
    json.dump(OUT, f, indent=1)
print('\nwrote logs/stats_checklist.json (complete)')
