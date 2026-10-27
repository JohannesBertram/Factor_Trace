"""Not reviewed code. The paper's key results as dark slides, drawn from the figure bundles.

Writes figures/slides/<name>.pdf; a slide with builds writes <name>_1, <name>_2, ... and is
centered as one group (style.publish), so the builds overlay. Reads only figures/figdata/ bundles,
through the data helpers of src/paper_figures.py.

    PYTHONPATH=. .venv/bin/python figures/results.py
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from figures.style import COLOR, publish, slide
from src import figdata
from src.paper_figures import (_VAL_MODELS, _OOD_MODELS, _cnn_fp_blocks, _faith_row, _sil_row,
                               cnn_depth_stats, cnn_layers, cosine_silhouette, fp_vs_act,
                               lam_weighted, montage, nodes_by_path, ood_signature_one, pca2,
                               top_stims, unit)

AR = 0.5
SHORT = {'airplane': 'plane', 'automobile': 'car'}            # CIFAR names that fit a montage


# ---------------------------------------------------------------- canvas and axes
def canvas():
    """Slide with a full-bleed drawing canvas in inches (y down) behind the plot axes."""
    fig = slide(AR)
    plt.rcParams['savefig.bbox'] = 'standard'
    W, H = fig.get_size_inches()
    cv = fig.add_axes([0, 0, 1, 1], zorder=-1)
    cv.set_xlim(0, W)
    cv.set_ylim(H, 0)
    cv.set_aspect('equal')
    cv.axis('off')
    return fig, cv


def ax_at(fig, x, y, w, h, frame=True):
    """Axes at x, y (inches from the top-left corner) of size w x h inches."""
    W, H = fig.get_size_inches()
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, length=2, pad=1.5)
    if not frame:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax


def mix(c, v, base=None):
    b = np.array(mcolors.to_rgb(base or COLOR['bg']))
    return tuple(b + (np.array(mcolors.to_rgb(c)) - b) * float(np.clip(v, 0, 1)))


def txt(ax, x, y, s, size=7, color=None, **kw):
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    return ax.text(x, y, s, fontsize=size, color=color or COLOR['fg'], zorder=8, **kw)


def class_color(i):
    return COLOR[f'class_{int(i) % 10}']


def image(fig, x, y, s, img, ec=None, lw=0.6, h=None):
    """An RGB image s inches wide (and h high, square by default) at (x, y), framed."""
    ax = ax_at(fig, x, y, s, h or s, frame=False)
    ax.imshow(img, interpolation='nearest', aspect='auto')
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(ec or COLOR['edge'])
        sp.set_linewidth(lw)
    return ax


def pill(cv, x, y, s, color):
    t = txt(cv, x, y, s, size=7, color=COLOR['bg'])
    t.set_bbox(dict(boxstyle='round,pad=0.25', fc=color, ec='none'))
    return t


def style_plot(ax, ylabel=None, xlabel=None):
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.5, labelpad=2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5, labelpad=2)


# ---------------------------------------------------------------- MLP circuits
class MLPGraph:
    """Both class circuits of the even/odd MLP in one graph, units grouped by circuit."""

    def __init__(self):
        D = figdata.load('nb01_circuits')
        self.circuits, self.sizes = D['circuits'], [int(n) for n in D['layer_sizes']]
        self.order = [int(d) for d in D['digit_order']]
        self.col = {c['name']: COLOR[c['color_key']] for c in self.circuits}
        load = np.stack([np.asarray(c['scaffold']['loading'], float) for c in self.circuits])
        self.load = load / (load.max(1, keepdims=True) + 1e-12)
        self.off = np.cumsum([0] + self.sizes)
        self.perm = []                                          # each circuit's units together
        for li, n in enumerate(self.sizes):
            V = self.load[:, self.off[li]:self.off[li + 1]]
            owner = np.where(V.max(0) < 0.05, len(self.circuits), V.argmax(0))
            self.perm.append(np.arange(n) if li == len(self.sizes) - 1
                             else np.lexsort((-V.max(0), owner)))
        self.ys = [np.linspace(0.45, 1.95, n) if n > 2 else np.array([0.95, 1.45])
                   for n in self.sizes]

    def draw(self, cv, x0=0.45, exc=True, inh=True, inh_by_circuit=False):
        """Edges (excitatory solid, inhibitory dashed), nodes and labels, L1 at x0."""
        xs = [x0, x0 + 0.65, x0 + 1.3]
        dash = (0, (1.5, 1.2))
        for c in self.circuits:
            sc, cc = c['scaffold'], self.col[c['name']]
            for bi, (E, N) in enumerate(zip(sc['edges'], sc['neg_edges'])):
                E = np.asarray(E)[np.ix_(self.perm[bi + 1], self.perm[bi])]
                N = np.asarray(N)[np.ix_(self.perm[bi + 1], self.perm[bi])]
                emax = max(E.max(), N.max(), 1e-12)
                kinds = ([(N, cc if inh_by_circuit else COLOR['inhibitory'], dash,
                           2.2 if inh_by_circuit else 1.0)] if inh else []) + \
                        ([(E, cc, '-', 2.2)] if exc else [])
                for M, color, ls, lw0 in kinds:
                    for i, j in np.ndindex(M.shape):
                        w = M[i, j] / emax
                        if w >= 0.06:
                            cv.plot([xs[bi], xs[bi + 1]], [self.ys[bi][j], self.ys[bi + 1][i]],
                                    color=color, lw=0.2 + lw0 * w, ls=ls,
                                    alpha=0.9 if ls == '-' or inh_by_circuit else 0.6,
                                    zorder=2 if ls == '-' else 1, solid_capstyle='round')
        for li, n in enumerate(self.sizes):
            V = self.load[:, self.off[li]:self.off[li + 1]][:, self.perm[li]]
            V = V / (V.max() + 1e-12)
            for u in range(n):
                v, owner = V[:, u].max(), int(V[:, u].argmax())
                c = self.col[self.circuits[owner]['name']]
                cv.add_patch(Circle((xs[li], self.ys[li][u]), 0.035 + 0.045 * v, zorder=4,
                                    fc=mix(c, 0.25 + 0.75 * v) if v >= 0.05 else COLOR['bg'],
                                    ec=COLOR['muted'] if v < 0.05 else c, lw=0.8))
        for x, s in zip(xs, ['$L_1$', '$L_2$', '$L_3$']):
            txt(cv, x, 2.17, s, size=8, color=COLOR['muted'])
        for y, c in zip(self.ys[2], self.circuits):
            txt(cv, xs[2] + 0.13, y, c['name'], size=8, color=self.col[c['name']], ha='left')


def slide_mlp_circuits():
    """Build 1: both class circuits in one graph. 2: the L1 factors of each, as pixel arbors."""
    G = MLPGraph()
    for step in (1, 2):
        fig, cv = canvas()
        G.draw(cv)
        for k, (lab, ls) in enumerate([('excitatory', '-'), ('inhibitory', (0, (1.5, 1.2)))]):
            y = 0.18 + 0.15 * k
            cv.plot([0.2, 0.42], [y, y], color=COLOR['muted'], ls=ls, lw=1.0)
            txt(cv, 0.47, y, lab, size=6.5, color=COLOR['muted'], ha='left')

        if step == 2:                                            # pixel arbors of the L1 factors
            s, x0 = 0.3, 2.45
            for r, c in enumerate(G.circuits):
                y0 = 0.62 + r * 0.82
                prof = np.asarray(c['l1_profiles'])
                rgb = np.array(mcolors.to_rgb(G.col[c['name']]))
                for k, A in enumerate(np.asarray(c['l1_arbors'])):
                    A = np.clip(np.clip(A, 0, None) / (np.percentile(A, 99.3) + 1e-12), 0, 1)
                    ax = image(fig, x0 + k * (s + 0.03), y0, s, A[..., None] ** 0.62 * rgb,
                               ec=COLOR['edge'])
                    txt(ax, 0.92, 0.9, str(G.order[int(prof[k].argmax())]), size=6.5,
                        ha='right', va='top', transform=ax.transAxes)
                txt(cv, x0, y0 - 0.12, f'{c["name"]} circuit, $L_1$ factors', size=7,
                    color=G.col[c['name']], ha='left')
        yield fig, f'mlp_circuits_{step}'


def slide_mlp_inhibition():
    """The same graph split by sign: each circuit inhibits exactly where the other one excites."""
    G = MLPGraph()
    fig, cv = canvas()
    G.draw(cv, exc=True, inh=False)
    G.draw(cv, x0=2.75, exc=False, inh=True, inh_by_circuit=True)
    for x, s in [(1.1, 'excitatory'), (3.4, 'inhibitory')]:
        txt(cv, x, 0.22, s, size=8)
    yield fig, 'mlp_inhibition'


# ---------------------------------------------------------------- CNN / ImageNet circuits
def slide_circuits(bundle, name, show, sub, n_top):
    """Build 1: output factors, each its top stimuli and class. 2: a few traced one layer back."""
    D = figdata.load(bundle)
    names = [SHORT.get(c, c) for c in D['class_names']]
    by_path = {tuple(int(i) for i in n['path']): n for n in D['nodes']}
    root = D['nodes'][0]
    n = min(n_top, int(root['n_factors']))
    s = min(0.42, 4.45 / n - 0.04)
    x0 = (4.685 - n * (s + 0.04) + 0.04) / 2
    ours = COLOR['ours']
    sb, gin, gout = 0.5, 0.06, 0.32
    xb0 = (4.685 - len(show) * (2 * sb + gin) - (len(show) - 1) * gout) / 2
    for step in (1, 2):
        fig, cv = canvas()
        for k in range(n):
            x = x0 + k * (s + 0.04)
            prof = root['class_profile'][k]
            hl = step == 2 and k in show
            image(fig, x, 0.22, s, montage(D, top_stims(D, root, k), 2, 2, bg=0.0),
                  ec=ours if hl else None, lw=1.4 if hl else 0.6)
            txt(cv, x + s / 2, 0.22 + s + 0.1, names[int(np.argmax(prof))], size=6.5,
                color=ours if hl else COLOR['fg'])
        if step == 2:
            for g, r in enumerate(show):
                node = by_path[(r,)]
                xg = xb0 + g * (2 * sb + gin + gout)
                xp, yp = xg + sb + gin / 2, 1.08
                xr = x0 + r * (s + 0.04) + s / 2
                cv.plot([xr, xp], [0.22 + s + 0.18, yp - 0.08], color=ours, lw=0.8)
                pill(cv, xp, yp, f'$f_{{{r}}}$ · {names[int(np.argmax(root["class_profile"][r]))]}',
                     ours)
                for slot, k in enumerate(sub[r]):
                    x = xg + slot * (sb + gin)
                    cv.plot([xp, x + sb / 2], [yp + 0.09, 1.3], color=COLOR['muted'], lw=0.6)
                    image(fig, x, 1.3, sb, montage(D, top_stims(D, node, k), 2, 2, bg=0.0))
                    txt(cv, x + sb / 2, 1.3 + sb + 0.1, f'$f_{{{k}}}$', size=7, color=ours)
        yield fig, f'{name}_{step}'


# ---------------------------------------------------------------- depth
def slide_depth():
    """Class purity of every factor by depth, CIFAR-10 CNN and ImageNet SqueezeNet."""
    fig, cv = canvas()
    ours = COLOR['ours']
    for i, (bundle, title, w) in enumerate([('nb03_circuits', 'CIFAR-10 CNN', 1.35),
                                            ('nb05_circuits', 'ImageNet SqueezeNet', 2.1)]):
        D = figdata.load(bundle)
        layers = cnn_layers(D['nodes'])
        ax = ax_at(fig, [0.7, 2.4][i], 0.4, w, 1.45)
        for li, (_, nodes) in enumerate(layers):
            pur = np.concatenate([n['class_profile'].max(1) for n in nodes])
            ax.scatter(np.full(len(pur), li), pur, s=3, color=ours, alpha=0.45, lw=0)
        mean = [lam_weighted(np.concatenate([n['class_profile'].max(1) for n in nodes]),
                             np.concatenate([n['lam_share'] for n in nodes]))
                for _, nodes in layers]
        ax.plot(range(len(layers)), mean, '-o', color=ours, lw=1.6, ms=3.5)
        chance = 1 / len(D['class_names'])
        ax.axhline(chance, color=COLOR['muted'], lw=0.7, ls=(0, (3, 2)))
        txt(ax, -0.3, chance - 0.03, 'chance', size=6.5, color=COLOR['muted'], ha='left',
            va='top')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f'$L_{{{len(layers) - li}}}$' for li in range(len(layers))])
        ax.set_xlim(-0.4, len(layers) - 0.6)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(title, fontsize=8, pad=4)
        style_plot(ax, ylabel='class purity' if i == 0 else None, xlabel='output  →  input')
    yield fig, 'depth'


# ---------------------------------------------------------------- pruning
def slide_pruning():
    """Pruning the weights BFT ranks most important for a class, on three models."""
    fig, cv = canvas()
    ours, bot, rnd = COLOR['ours'], COLOR['bft_bottom'], COLOR['random']
    cases = [('nb13_pruning_mlp_even_odd', 'MLP even/odd', 50),
             ('nb14_pruning_cnn_cifar', 'CIFAR-10 CNN', 20),
             ('nb14_pruning_imagenet_cnn', 'ImageNet SqueezeNet', 20)]
    for i, (bundle, title, xmax) in enumerate(cases):
        P = figdata.load(bundle)
        fr = np.asarray(P['fractions']) * 100
        m = P['methods']
        ax = ax_at(fig, 0.55 + i * 1.45, 0.3, 1.15, 1.15)
        ax.plot(fr, m['random']['target_mean'], ':', color=rnd, lw=1.3)
        if 'target_all' in m['bft_bottom'] and 'imagenet' in bundle:     # bimodal: one per category
            for row in m['bft_bottom']['target_all']:
                ax.plot(fr, row, ':', color=bot, lw=0.9, alpha=0.8)
        else:
            ax.plot(fr, m['bft_bottom']['target_mean'], ':', color=bot, lw=1.4)
        ax.plot(fr, m['bft_top']['bystander_mean'], '--', color=ours, lw=1.3)
        ax.plot(fr, m['bft_top']['target_mean'], '-o', color=ours, lw=1.6, ms=3)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([0, xmax / 2, xmax])
        ax.set_xticklabels([f'{v:g}' for v in (0, xmax / 2, xmax)])
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(title, fontsize=8, pad=4)
        style_plot(ax, ylabel='class accuracy' if i == 0 else None,
                   xlabel='% weights pruned' if i == 1 else None)
    hs = [plt.Line2D([], [], color=ours, lw=1.6, marker='o', ms=3, label='most important · its class'),
          plt.Line2D([], [], color=ours, lw=1.3, ls='--', label='most important · other classes'),
          plt.Line2D([], [], color=bot, lw=1.4, ls=':', label='least important · its class'),
          plt.Line2D([], [], color=rnd, lw=1.3, ls=':', label='random · its class')]
    ax.legend(handles=hs, ncol=2, fontsize=6.5, frameon=False, loc='upper center',
              bbox_to_anchor=(-0.75, -0.34), columnspacing=1.5, handlelength=2.2)
    yield fig, 'pruning'


# ---------------------------------------------------------------- fingerprints
def bft_cmap():
    return mcolors.LinearSegmentedColormap.from_list('bft', [COLOR['bg'], COLOR['ours'], 'white'],
                                                     N=256)


def slide_fingerprint_classes():
    """Mean CIFAR-10 fingerprint per class, factors grouped by output circuit."""
    D = figdata.load('nb03_fingerprints')
    cols, edges, _, _ = _cnn_fp_blocks(D)
    # _cnn_fp_blocks keeps one circuit per class; the circuits beyond that go after them
    co, sizes = list(D['col_order']), [int(b) for b in D['block_sizes']]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    used = set(cols)
    for st, n in zip(starts, sizes):
        block = co[st:st + n]
        if block[0] not in used:
            edges = list(edges) + [len(cols)]
            cols = list(cols) + block
    M = np.asarray(D['fp_mean_by_class'], float)
    M = M / M.sum(1, keepdims=True)
    M = M[:, cols] / (M[:, cols].max(0, keepdims=True) + 1e-12)
    fig, cv = canvas()
    ax = ax_at(fig, 0.85, 0.3, 3.6, 1.6)
    ax.imshow(M, cmap=bft_cmap(), aspect='auto', interpolation='nearest',
              norm=mcolors.PowerNorm(0.8, vmin=0, vmax=1.2))
    for e in edges:
        ax.axvline(e - 0.5, color=COLOR['edge'], lw=0.6)
    names = list(D['class_names'])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    for t, i in zip(ax.get_yticklabels(), range(len(names))):
        t.set_color(class_color(i))
    ax.set_xticks([])
    ax.tick_params(length=0)
    ax.set_xlabel(f'{M.shape[1]} factors, grouped by output circuit', fontsize=7.5, labelpad=4)
    yield fig, 'fingerprint_classes'


def slide_fingerprint_separability():
    """CIFAR-10 fingerprint vs. the network's own penultimate activations, and both models' silhouettes."""
    D3, D5 = figdata.load('nb03_fingerprints'), figdata.load('nb05_fingerprints')
    F, yf, A, ya, rep = fp_vs_act(D3, D3['fp']['id_targets'])
    fig, cv = canvas()
    for i, (X, y, title) in enumerate([(F, yf, f'BFT fingerprint, {F.shape[1]}-d'),
                                       (A, ya, f'network activations, {A.shape[1]}-d')]):
        ax = ax_at(fig, 0.3 + i * 1.5, 0.4, 1.35, 1.35, frame=False)
        P, _ = pca2(X)
        for c in np.unique(y):
            m = y == c
            ax.scatter(P[m, 0], P[m, 1], s=1.6, color=class_color(c), alpha=0.75, lw=0,
                       rasterized=True)
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(COLOR['edge'])
        ax.set_title(title, fontsize=7.5, pad=3)
        txt(ax, 0.04, 0.96, f'sil {cosine_silhouette(X, y):.2f}', size=7, ha='left', va='top',
            transform=ax.transAxes)
    txt(cv, 1.8, 1.95, 'CIFAR-10 CNN, test stimuli colored by class', size=7,
        color=COLOR['muted'])

    F5, y5, A5, ya5, _ = fp_vs_act(D5, D5['fp']['id_targets'])
    sil = [(cosine_silhouette(F, yf), cosine_silhouette(A, ya)),
           (cosine_silhouette(F5, y5), cosine_silhouette(A5, ya5))]
    ax = ax_at(fig, 3.55, 0.4, 1.0, 1.35)
    for g, (a, b) in enumerate(sil):
        ax.bar(g - 0.18, a, 0.34, color=COLOR['ours'])
        ax.bar(g + 0.18, b, 0.34, color=COLOR['baseline'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['CIFAR-10', 'ImageNet'], fontsize=7)
    ax.set_yticks([0, 0.3, 0.6])
    style_plot(ax, ylabel='class silhouette')
    hs = [Rectangle((0, 0), 1, 1, color=COLOR['ours'], label='fingerprint'),
          Rectangle((0, 0), 1, 1, color=COLOR['baseline'], label='activations')]
    ax.legend(handles=hs, fontsize=6.5, frameon=False, loc='lower center',
              bbox_to_anchor=(0.5, 1.0), ncol=1, handlelength=1, handleheight=0.8)
    yield fig, 'fingerprint_separability'


def slide_weight_term():
    """The same NMF on the arbor (weight x activation) vs. on the activations alone."""
    rows = [_sil_row(name, figdata.load(b)) for name, b in _VAL_MODELS]
    fig, cv = canvas()
    ax = ax_at(fig, 1.0, 0.35, 2.9, 1.4)
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [r['arbor_nmf'] for r in rows], 0.36, color=COLOR['ours'],
           label=r'arbor NMF ($W \cdot a$)')
    ax.bar(x + 0.19, [r['act_nmf'] for r in rows], 0.36, color=COLOR['baseline'],
           label='activation-only NMF')
    ax.set_xticks(x)
    ax.set_xticklabels(['MLP\neven/odd', 'MLP\ndigit', 'CIFAR-10\nCNN', 'TinyViT', 'ImageNet'],
                       fontsize=7)
    ax.set_yticks([0, 0.3, 0.6])
    style_plot(ax, ylabel='fingerprint class silhouette')
    ax.legend(fontsize=7, frameon=False, loc='upper right', bbox_to_anchor=(1.05, 1.12),
              handlelength=1, handleheight=0.8)
    n = sum(r['arbor_nmf'] > r['act_nmf'] for r in rows)
    txt(ax, 0.0, 1.06, f'arbor ahead on {n} of {len(rows)} models', size=7, ha='left',
        color=COLOR['muted'], transform=ax.transAxes)
    yield fig, 'weight_term'


# ---------------------------------------------------------------- OOD
def slide_ood():
    """Held-out digits follow the network's own P(odd); synthetic inputs collapse onto one fingerprint."""
    D = figdata.load('nb01_fingerprints')
    fig, cv = canvas()
    even, odd = COLOR['even'], COLOR['odd']
    ax = ax_at(fig, 0.55, 0.35, 1.3, 1.3)
    ax.plot([0, 1], [0, 1], color=COLOR['faint'], lw=0.7, ls=(0, (3, 2)))
    for d, x, y in D['id_pts']:
        ax.scatter(x, y, s=22, facecolor='none', edgecolor=even if int(d) % 2 == 0 else odd, lw=1.0)
    nudge = {5: (2, -6.5), 7: (-1.5, -6.5)}                     # points; keeps 5 off 8
    for d, x, y in D['ood_pts']:
        c = even if int(d) % 2 == 0 else odd
        ax.scatter(x, y, s=24, color=c, lw=0)
        ax.annotate(str(int(d)), (x, y), textcoords='offset points', fontsize=7, color=c,
                    xytext=nudge.get(int(d), (4, -1)), va='center')
    ax.set_xlim(-0.06, 1.08)
    ax.set_ylim(-0.06, 1.08)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    txt(ax, 0.03, 0.97, f'$r = {float(D["r_ood"]):.3f}$', size=7, ha='left', va='top',
        transform=ax.transAxes)
    style_plot(ax, ylabel='odd-circuit share', xlabel='P(network says odd)')
    ax.set_title('even/odd MLP: held-out digits', fontsize=7.5, pad=4)

    rows = [ood_signature_one(name, figdata.load(b)) for name, b in _OOD_MODELS]
    ax = ax_at(fig, 2.5, 0.35, 1.85, 1.3)
    x = np.arange(len(rows))
    for off, key, c, lab in [(-0.26, 'collapse_id', COLOR['baseline'], 'in-distribution'),
                             (0, 'collapse_near', COLOR['near_ood'], 'held-out data'),
                             (0.26, 'collapse_far', COLOR['far_ood'], 'synthetic')]:
        v = [r[key] if r[key] is not None else np.nan for r in rows]
        ax.bar(x + off, v, 0.26, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(['8x4\nMLP', 'digit\nMLP', 'CNN', 'ViT', 'Image-\nNet'], fontsize=6.5)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1])
    style_plot(ax, ylabel='cosine to condition mean')
    ax.set_title('all models: collapse onto one fingerprint', fontsize=7.5, pad=4)
    ax.legend(fontsize=6.5, frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.3),
              ncol=3, handlelength=1, handleheight=0.8, columnspacing=0.8, handletextpad=0.4)
    yield fig, 'ood'


# ---------------------------------------------------------------- backup slides
MODELS = ['MLP\neven/odd', 'MLP\ndigit', 'CIFAR-10\nCNN', 'TinyViT', 'ImageNet']


def arbor_rgb(A, color):
    A = np.clip(np.clip(A, 0, None) / (np.percentile(A, 99.3) + 1e-12), 0, 1)
    return A[..., None] ** 0.62 * np.array(mcolors.to_rgb(color))


def slide_backup_digit_mlp():
    """Fig. B: the ten-digit MLP's output factors pool digits; traced back they un-pool."""
    D = figdata.load('nb02_circuits')
    C, nodes, n_show = D['circuits'], nodes_by_path(D), int(D['n_show'])
    ours = COLOR['ours']
    fig, cv = canvas()
    ax = ax_at(fig, 0.55, 0.35, 1.35, 1.45)
    M = np.stack([c['profile'] for c in C])
    ax.imshow(M, cmap=bft_cmap(), aspect='auto', vmin=0, vmax=M.max() * 1.15,
              interpolation='nearest')
    ax.set_xticks(range(10))
    ax.set_yticks(range(len(C)))
    ax.set_yticklabels([f'$f_{{{int(c["k"])}}}$' for c in C])
    ax.tick_params(length=0)
    ax.set_title('output factors pool digits', fontsize=7.5, pad=4)
    style_plot(ax, xlabel='digit')

    ax = ax_at(fig, 2.3, 0.35, 0.95, 1.45)
    best = np.array([[D['root_pur'][int(c['k'])],
                      nodes[(int(c['k']),)]['class_profile'].max(1).max(),
                      nodes[(int(c['k']), 0)]['class_profile'].max(1).max()] for c in C])
    for b in best:
        ax.plot(range(3), b, color=mix(ours, 0.5), lw=0.8)
    ax.plot(range(3), best.mean(0), '-o', color=ours, lw=1.8, ms=3.5)
    ax.axhline(0.1, color=COLOR['muted'], lw=0.7, ls=(0, (3, 2)))
    txt(ax, 2.3, 0.12, 'chance', size=6.5, color=COLOR['muted'], ha='right', va='bottom')
    ax.set_xticks(range(3))
    ax.set_xticklabels(['$L_3$', '$L_2$', '$L_1$'])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_title('traced back: un-pooled', fontsize=7.5, pad=4)
    style_plot(ax, ylabel='digit purity')

    s_ = 0.3                                                       # the detectors of two circuits
    for r, c in enumerate(C[:2]):
        y0 = 0.5 + r * 0.72
        P = np.asarray(c['l1_profiles'])
        top = np.argsort(-c['profile'])[:n_show]
        txt(cv, 3.55, y0 - 0.1, f'$f_{{{int(c["k"])}}}$ pools ' + ', '.join(map(str, top)),
            size=7, color=ours, ha='left')
        for q, d in enumerate(top):
            ax = image(fig, 3.55 + q * (s_ + 0.04), y0, s_,
                       arbor_rgb(c['l1_arbors'][int(P[:, d].argmax())], ours))
            txt(ax, 0.92, 0.9, str(int(d)), size=6.5, ha='right', va='top',
                transform=ax.transAxes)
    yield fig, 'backup_digit_mlp'


def slide_backup_vit():
    """Fig. G: a TinyViT block traces like an MLP; output circuits pool similar digits along parity."""
    D = figdata.load('nb04_circuits')
    C, root = D['circuits'], D['nodes'][0]
    even, odd = COLOR['even'], COLOR['odd']
    fig, cv = canvas()
    ax = ax_at(fig, 0.5, 0.35, 1.55, 1.55)
    M = np.stack([c['digit_profile'] for c in C])
    ax.imshow(M, cmap=bft_cmap(), aspect='auto', vmin=0, vmax=M.max() * 1.15,
              interpolation='nearest')
    ax.set_xticks(range(10))
    ax.set_yticks(range(len(C)))
    ax.set_yticklabels([f'$f_{{{int(c["k"])}}}$' for c in C])
    for t, c in zip(ax.get_yticklabels(), C):
        t.set_color(even if c['profile'][0] > c['profile'][1] else odd)
    ax.tick_params(length=0)
    ax.set_title('output circuits, by digit', fontsize=7.5, pad=4)
    style_plot(ax, xlabel='digit')

    s_, n = 0.15, 8                                                # strongest stimuli per circuit
    x0 = 2.45
    for r, c in enumerate(C):
        imgs = top_stims(D, root, int(c['k']))[:n]
        y = 0.35 + r * (s_ + 0.005)
        image(fig, x0, y, n * s_, montage(D, imgs, 1, n, bg=0.0), ec=COLOR['bg'], lw=0, h=s_)
        txt(cv, x0 - 0.06, y + s_ / 2, f'$f_{{{int(c["k"])}}}$', size=6.5, ha='right',
            color=even if c['profile'][0] > c['profile'][1] else odd)
    txt(cv, x0 + n * s_ / 2, 0.22, 'strongest stimuli per circuit', size=7.5)
    yield fig, 'backup_vit'


def slide_backup_gallery(bundle, name, n_show):
    """Figs. E, N: every class circuit with its two strongest sub-factors one layer back."""
    D = figdata.load(bundle)
    names = [SHORT.get(c, c) for c in D['class_names']]
    by_path = {tuple(int(i) for i in n['path']): n for n in D['nodes']}
    root = D['nodes'][0]
    seen, show = set(), []
    for k in range(int(root['n_factors'])):                    # one circuit per class, strongest first
        c = int(np.argmax(root['class_profile'][k]))
        if c not in seen and (k,) in by_path:
            seen.add(c)
            show.append(k)
    show = show[:n_show]
    s_ = min(0.4, 3.9 / len(show) - 0.05)
    x0 = (4.685 - len(show) * (s_ + 0.05) + 0.05) / 2
    ours = COLOR['ours']
    fig, cv = canvas()
    for g, k in enumerate(show):
        x = x0 + g * (s_ + 0.05)
        c = int(np.argmax(root['class_profile'][k]))
        txt(cv, x + s_ / 2, 0.2, names[c], size=6.5, color=ours)
        image(fig, x, 0.3, s_, montage(D, top_stims(D, root, k), 2, 2, bg=0.0), ec=ours, lw=1.0)
        node = by_path[(k,)]
        for q in range(min(2, int(node['n_factors']))):
            y = 0.3 + s_ + 0.18 + q * (s_ + 0.06)
            image(fig, x, y, s_, montage(D, top_stims(D, node, q), 2, 2, bg=0.0))
        cv.plot([x + s_ / 2] * 2, [0.3 + s_ + 0.02, 0.3 + s_ + 0.16], color=COLOR['muted'], lw=0.7)
    txt(cv, x0 - 0.08, 0.3 + s_ / 2, 'output', size=6.5, color=COLOR['muted'], ha='right')
    txt(cv, x0 - 0.08, 0.3 + 2 * s_ + 0.21, 'one layer\nback', size=6.5, color=COLOR['muted'],
        ha='right', linespacing=1.1)
    yield fig, name


def slide_backup_conv1():
    """Fig. 6c, e: at conv1 the factors are color channels, shared by every circuit."""
    D = figdata.load('nb03_circuits')
    by_path = {tuple(int(i) for i in n['path']): n for n in D['nodes']}
    leaf = by_path[(9, 0, 0, 0)]
    fig, cv = canvas()
    s_ = 0.42
    rgb = [COLOR['red'], COLOR['green'], COLOR['blue']]
    for k in range(int(leaf['n_factors'])):
        x = 0.4 + k * (s_ + 0.06)
        image(fig, x, 0.55, s_, montage(D, top_stims(D, leaf, k), 2, 2, bg=0.0))
        share = np.asarray(leaf['conn']['in_mass'][k], float)
        share = share / share.sum()
        xb = x
        for v, c in zip(share, rgb):                               # input mass per color channel
            cv.add_patch(Rectangle((xb, 0.55 + s_ + 0.05), s_ * v, 0.07, fc=c, lw=0))
            xb += s_ * v
        txt(cv, x + s_ / 2, 0.42, f'$f_{{{k}}}$', size=7, color=COLOR['ours'])
    txt(cv, 0.4 + 2.5 * (s_ + 0.06) - 0.03, 1.35, 'conv1 factors of the cat circuit,\n'
        'with their input per color channel', size=7, color=COLOR['muted'])

    stats, rand = cnn_depth_stats(D)
    ax = ax_at(fig, 3.2, 0.45, 1.25, 1.3)
    ax.plot(range(len(stats)), [st['spread'].mean() for st in stats], '-s', color=COLOR['ours'],
            lw=1.6, ms=3.5)
    ax.axhline(rand, color=COLOR['muted'], lw=0.7, ls=(0, (3, 2)))
    txt(ax, len(stats) - 0.6, rand + 0.01, 'random', size=6.5, color=COLOR['muted'],
        ha='right', va='bottom')
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([f'$L_{{{len(stats) - i}}}$' for i in range(len(stats))])
    ax.set_ylim(0, 0.32)
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    style_plot(ax, ylabel='color spread of top stimuli')
    yield fig, 'backup_conv1'


def slide_backup_validation():
    """Fig. P a-c: faithful (causal reconstruction, NNLS round-trip) and stable across NMF seeds."""
    rows = [_faith_row(name, figdata.load(b)) for name, b in _VAL_MODELS]
    fig, cv = canvas()
    y = np.arange(len(rows))[::-1]
    for i, (key, title) in enumerate([('recon', 'causal reconstruction $R^2$'),
                                      ('rt', 'NNLS round-trip'), ('stab', 'NMF-seed stability')]):
        ax = ax_at(fig, 0.95 + i * 1.25, 0.4, 1.0, 1.35)
        for yy, r in zip(y, rows):
            if r[key] is None:
                txt(ax, 0.03, yy, 'n/a', size=6.5, color=COLOR['muted'], ha='left')
                continue
            ax.barh(yy, r[key], 0.6, color=COLOR['ours'],
                    xerr=r['stab_sd'] if key == 'stab' else None,
                    error_kw=dict(ecolor=COLOR['fg'], lw=0.8, capsize=1.5))
            txt(ax, r[key] + (r['stab_sd'] if key == 'stab' else 0) + 0.05, yy, f'{r[key]:.2f}',
                size=6.5, ha='left')
        ax.set_xlim(0, 1.3)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks(y)
        ax.set_yticklabels([r['name'] for r in rows] if i == 0 else [], fontsize=7)
        ax.tick_params(axis='y', length=0)
        ax.set_title(title, fontsize=7.5, pad=4)
    yield fig, 'backup_validation'


def slide_backup_rank():
    """Fig. P f: the factors barely move when the rank is one off the chosen K*."""
    fig, cv = canvas()
    ours = COLOR['ours']
    rng = np.random.default_rng(1)
    for i, (name, b) in enumerate(_VAL_MODELS):
        ks = figdata.load(b)['stability']['k_sensitivity']
        ax = ax_at(fig, 0.6 + i * 0.82, 0.5, 0.65, 1.2)
        for j, key in enumerate(['k_minus1', 'k_star', 'k_plus1']):
            v = np.asarray(ks[key], float)
            ax.scatter(j + (rng.random(len(v)) - 0.5) * 0.3, v, s=7, lw=0,
                       color=ours if j == 1 else mix(ours, 0.6, base='white'))
            ax.plot([j - 0.3, j + 0.3], [np.median(v)] * 2, color=COLOR['fg'], lw=1.0)
        ax.set_xticks(range(3))
        ax.set_xticklabels(['$-1$', '$K^*$', '$+1$'], fontsize=6.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '.5', '1'] if i == 0 else [])
        ax.set_title(MODELS[i].replace('\n', ' '), fontsize=7, pad=4)
        if i == 0:
            style_plot(ax, ylabel='cosine to the $K^*$ factors')
    yield fig, 'backup_rank'


def slide_backup_fp_baselines():
    """Fig. P d: the fingerprint against activation baselines at native and matched width."""
    rows = [_sil_row(name, figdata.load(b)) for name, b in _VAL_MODELS]
    fig, cv = canvas()
    ax = ax_at(fig, 0.9, 0.5, 3.3, 1.25)
    bars = [('fp_sil', COLOR['ours'], 'BFT fingerprint'),
            ('act_raw_sil', COLOR['baseline'], 'activations, native width'),
            ('act_sil', mix(COLOR['baseline'], 0.55), 'activations, matched width'),
            ('act_rand_sil', COLOR['random'], 'random projection, matched')]
    x = np.arange(len(rows))
    for q, (key, c, lab) in enumerate(bars):
        v = [r[key] if r[key] is not None else np.nan for r in rows]
        ax.bar(x + (q - 1.5) * 0.2, v, 0.19, color=c, label=lab)
    ax.axhline(0, color=COLOR['fg'], lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=7)
    ax.set_yticks([0, 0.3, 0.6])
    style_plot(ax, ylabel='class silhouette')
    ax.legend(fontsize=6.5, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1, handleheight=0.8, columnspacing=1.2)
    yield fig, 'backup_fp_baselines'


def slide_backup_fp_geometry():
    """Fig. fp_structure a-e: pairwise fingerprint cosine is block-diagonal by class, every model."""
    fig, cv = canvas()
    cmap = bft_cmap()
    for i, ((name, b), key) in enumerate(zip(_OOD_MODELS, ['id_digits', 'id_targets', 'id_targets',
                                                            'id_digits', 'id_targets'])):
        D = figdata.load(b)
        X, y = np.asarray(D['fp']['id'], float), np.asarray(D['fp'][key])
        rng = np.random.default_rng(0)
        sel = np.concatenate([np.sort(rng.choice(np.where(y == c)[0],
                                                 min(30, int((y == c).sum())), replace=False))
                              for c in np.unique(y)])
        U = unit(X[sel])
        ax = ax_at(fig, 0.3 + i * 0.85, 0.5, 0.75, 0.75, frame=False)
        ax.imshow(U @ U.T, cmap=cmap, vmin=0, vmax=1.15, interpolation='nearest')
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(COLOR['edge'])
        ax.set_title(MODELS[i].replace('\n', ' '), fontsize=7, pad=4)
        txt(ax, 0.5, -0.12, f'silhouette {cosine_silhouette(X, y):.2f}', size=6.5,
            color=COLOR['muted'], va='top', transform=ax.transAxes)
    txt(cv, 2.34, 1.65, 'cosine between test fingerprints, sorted by class', size=7,
        color=COLOR['muted'])
    yield fig, 'backup_fp_geometry'


def slide_backup_ood_conditions():
    """Fig. fp_ood f: CIFAR-10 mean fingerprint per condition; synthetic inputs use a few factors."""
    D = figdata.load('nb03_fingerprints')
    fp = D['fp']
    rows = [('test images', np.asarray(fp['id'], float).mean(0), COLOR['baseline']),
            ('CIFAR-100', np.asarray(fp['ood'], float).mean(0), COLOR['near_ood'])]
    rows += [(f['label'], np.asarray(f['F'], float).mean(0), COLOR['far_ood']) for f in fp['far']]
    M = np.stack([m for _, m, _ in rows])
    M = M / M.sum(1, keepdims=True)
    fig, cv = canvas()
    ax = ax_at(fig, 0.95, 0.4, 3.4, 1.3)
    ax.imshow(M[:, list(D['col_order'])], cmap=bft_cmap(), aspect='auto', interpolation='nearest',
              norm=mcolors.PowerNorm(0.7, vmin=0, vmax=M.max() * 1.15))
    for e in D['blk_edge']:
        ax.axvline(e - 0.5, color=COLOR['edge'], lw=0.6)
    ax.set_xticks([])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(r[2])
    ax.tick_params(length=0)
    ax.set_xlabel(f'{M.shape[1]} factors, grouped by output circuit', fontsize=7.5, labelpad=4)
    yield fig, 'backup_ood_conditions'


if __name__ == '__main__':
    publish([*slide_mlp_circuits(), *slide_mlp_inhibition()],
            [*slide_circuits('nb03_circuits', 'cnn_circuits', [0, 2, 12],
                             {0: [0, 1], 2: [1, 2], 12: [0, 1]}, 13)],
            [*slide_circuits('nb05_circuits', 'imagenet_circuits', [0, 2],
                             {0: [0, 1], 2: [1, 2]}, 10)],
            [*slide_depth()],
            [*slide_pruning()],
            [*slide_fingerprint_classes()],
            [*slide_fingerprint_separability()],
            [*slide_weight_term()],
            [*slide_ood()],
            [*slide_backup_digit_mlp()],
            [*slide_backup_vit()],
            [*slide_backup_gallery('nb03_circuits', 'backup_cnn_gallery', 10)],
            [*slide_backup_gallery('nb05_circuits', 'backup_imagenet_gallery', 10)],
            [*slide_backup_conv1()],
            [*slide_backup_validation()],
            [*slide_backup_rank()],
            [*slide_backup_fp_baselines()],
            [*slide_backup_fp_geometry()],
            [*slide_backup_ood_conditions()])
