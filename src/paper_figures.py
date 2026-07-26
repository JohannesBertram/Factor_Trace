"""Paper figures, drawn from figdata bundles.

Every function here takes a bundle (see ``src/figdata.py``) and returns a matplotlib
figure. Nothing in this module touches a model, a dataset or sklearn — that work
happens in the notebooks, which export the bundles. So the figures can be
regenerated on any checkout:

    from src import figdata
    from src.paper_figures import FIGURES
    name = 'fig2_mlp_circuits'
    bundle, render, mode = FIGURES[name]
    fig = render(figdata.load(bundle))
    figstyle.save_fig(fig, name)          # or: python scripts/render_figures.py

Colors always resolve through ``.figstyle/colors.yaml`` (bundles store the semantic
key, e.g. 'even', never a hex value), so the paper's palette stays in one place.
"""

from __future__ import annotations

import re

import numpy as np
import matplotlib
import matplotlib.patches
import matplotlib.patheffects
import matplotlib.pyplot as plt

import figstyle

IMAGE_SIDE = 28


# ── shared style helpers ──────────────────────────────────────────────────────

def tint(hex_color, amount):
    """Blend a color toward white (0 keeps it, 1 makes it white)."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1.0 - rgb) * amount)


def seq_cmap(hex_color, name):
    """white -> color -> near-black sequential map (keeps class identity)."""
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        name, [(1, 1, 1), tuple(rgb + (1 - rgb) * 0.45), tuple(rgb), tuple(rgb * 0.35)])


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def digit_colors(digit_order, c_even, c_odd):
    """Parity gives the hue, order within a parity gives the shade."""
    out, seen = {}, {0: 0, 1: 0}
    for d in digit_order:
        base = c_even if d % 2 == 0 else c_odd
        out[d] = base if seen[d % 2] == 0 else tint(base, 0.55)
        seen[d % 2] += 1
    return out


def show_map(ax, M, cmap, pct=99.3, gamma=0.62):
    ax.imshow(M, cmap=cmap, interpolation='nearest',
              norm=matplotlib.colors.PowerNorm(gamma, vmin=0,
                                               vmax=np.percentile(M, pct)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)


def row_label(fig, spec, text, color='k'):
    """Axis-off cell used as a row label (keeps an image grid perfectly aligned)."""
    ax = fig.add_subplot(spec)
    ax.set_axis_off()
    ax.text(1.0, 0.5, text, transform=ax.transAxes, ha='right', va='center',
            color=color, fontsize=6.5)
    return ax


def spacer_label(fig, anchor, spacer, text, color='k', dx=0.0):
    """Panel label in a reserved spacer row, anchored to an axes. Use after freeze."""
    fig.text(max(anchor.get_position().x0 + dx, 0.002), spacer.get_position().y0,
             text, ha='left', va='bottom', color=color, fontweight='bold')


def bar_panel(ax, values, digit_order, digit_color, *, xticks=True,
              yticklabels=False, ylab=None, title=None, tcolor='k', color=None):
    """Mini bar chart: per-digit share of one factor's stimulus loading.

    `color` overrides the per-digit hues with one flat color for every bar (used
    at the top of the tree, where the factors are not yet digit-specific)."""
    bar_colors = [color] * len(digit_order) if color is not None \
        else [digit_color[d] for d in digit_order]
    ax.bar(np.arange(len(digit_order)), values,
           color=bar_colors, edgecolor='0.25', linewidth=0.3, width=0.8)
    ax.set_xlim(-0.65, len(digit_order) - 0.35)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(len(digit_order)))
    ax.set_xticklabels(digit_order if xticks else [])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['0', '1'] if (yticklabels or ylab) else [])
    ax.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if ylab:
        ax.set_ylabel(ylab, labelpad=1)
    if title:
        ax.set_title(title, color=tcolor, pad=1.5)


def draw_scaffold(ax, edges, neg_edges, loading, layer_sizes, color, c_inh,
                  out_labels=('even', 'odd'), highlight_out=0, legend=False):
    """Neuron-level circuit graph L1 -> L2 -> L3, excitatory and inhibitory edges."""
    ys = [np.linspace(0.98, 0.02, n) if n > 1 else np.array([0.5]) for n in layer_sizes]
    xs = [np.full(n, float(li)) for li, n in enumerate(layer_sizes)]
    off = np.cumsum([0] + list(layer_sizes))
    load = np.asarray(loading, float)

    for bi, (E, N) in enumerate(zip(edges, neg_edges)):
        emax = max(E.max(), N.max(), 1e-12)              # per-boundary normalization
        for M, col, lsty, z, al, sc in ((N, c_inh, (0, (1.4, 1.1)), 1, .55, 0.9),
                                        (E, color, '-', 2, .95, 1.8)):
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    w = M[i, j] / emax
                    if w < 0.03:
                        continue
                    ax.plot([xs[bi][j], xs[bi + 1][i]], [ys[bi][j], ys[bi + 1][i]],
                            lw=0.12 + sc * w, color=col, linestyle=lsty,
                            alpha=al, zorder=z, solid_capstyle='round')

    for li, n in enumerate(layer_sizes):
        v = load[off[li]:off[li + 1]]
        v = v / (v.max() + 1e-12)                        # per-layer normalization
        on = v >= 0.05
        ax.scatter(xs[li][on], ys[li][on], s=7 + 30 * v[on],
                   c=[matplotlib.colors.to_rgba(color, 0.3 + 0.7 * vi) for vi in v[on]],
                   edgecolors='0.25', linewidths=0.4, zorder=3)
        ax.scatter(xs[li][~on], ys[li][~on], s=6, c='none',
                   edgecolors='0.72', linewidths=0.4, zorder=3)

    ax.annotate('', xy=(-0.18, 0.5), xytext=(-0.62, 0.5),
                arrowprops=dict(arrowstyle='-|>', lw=0.5, color='0.45',
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text(-0.66, 0.5, '784 px', ha='right', va='center', fontsize=6, color='0.35')
    for li, lab in enumerate([r'$L_1$', r'$L_2$', r'$L_3$']):
        ax.text(li, -0.10, lab, ha='center', va='top', fontsize=6.5)
    for i, lab in enumerate(out_labels):
        ax.text(2.12, ys[2][i], lab, ha='left', va='center', fontsize=6.5,
                color=color if i == highlight_out else '0.55',
                fontweight='bold' if i == highlight_out else 'normal')
    if legend:
        for yy, (col, lsty, lab) in enumerate(((color, '-', 'excitatory'),
                                               (c_inh, (0, (1.4, 1.1)), 'inhibitory'))):
            y = 0.20 - 0.14 * yy
            ax.plot([-1.33, -1.05], [y, y], color=col, linestyle=lsty, lw=0.9,
                    solid_capstyle='round')
            ax.text(-1.0, y, lab, fontsize=6.2, ha='left', va='center', color='0.3')
    ax.set_xlim(-1.35, 2.8)
    ax.set_ylim(-0.2, 1.06)
    ax.axis('off')


def _circuit_colors(D):
    """Resolve every circuit's semantic color key once, in place."""
    for c in D['circuits']:
        c['color'] = figstyle.color(c['color_key'])
    return D['circuits']


# ── quantities derived from a circuits bundle (numpy only — see figdata.py) ───

def nodes_by_path(D):
    return {tuple(int(i) for i in n['path']): n for n in D['nodes']}


def unit_support(node):
    """lambda-weighted share of a node's incoming connection mass per input unit."""
    m = (node['lam_share'][:, None] * node['conn']['in_mass']).sum(0)
    return m / (m.sum() + 1e-12)


def support_overlap(S, n_shuffle=2000, seed=0):
    """Mean pairwise cosine between circuit supports, and a within-row shuffle null.

    The null keeps each circuit's mass distribution and only moves it to other
    units, so it answers "do the circuits avoid each other more than mass of this
    shape would by chance".
    """
    def mean_cos(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        C = Mn @ Mn.T
        iu = np.triu_indices(len(M), 1)
        return float(C[iu].mean())

    rng = np.random.default_rng(seed)
    null = np.array([mean_cos(np.stack([rng.permutation(r) for r in S]))
                     for _ in range(n_shuffle)])
    return mean_cos(S), null


def purity_by_layer(D):
    """[(layer_idx, purity, lambda-share, circuit index)] for every factor of a trace.

    Purity is the largest class share of a factor's stimulus loading: 1/n_classes
    if the factor is class-agnostic, 1 if it fires for a single class.
    """
    out = []
    for node in D['nodes']:
        path = [int(i) for i in node['path']]
        P, lam = node['class_profile'], node['lam_share']
        for k in range(len(P)):
            circuit = path[0] if path else k
            out.append((int(node['layer_idx']), float(P[k].max()),
                        float(lam[k]), circuit))
    return out


def draw_scaffold_pair(ax, circuits, layer_sizes, c_inh, out_labels, in_label='784 px'):
    """Both class circuits in ONE neuron graph — the direct view of how much of the
    hidden layer they share, and of the push-pull at the output."""
    ys = [np.linspace(0.96, 0.04, n) if n > 1 else np.array([0.5]) for n in layer_sizes]
    xs = [np.full(n, float(li)) for li, n in enumerate(layer_sizes)]
    off = np.cumsum([0] + list(layer_sizes))

    # Hidden units are re-ordered so each circuit's units sit together: whether the
    # circuits share the layer is then legible from the picture, not just the stat.
    # The output layer keeps its order so the class labels stay put.
    load = np.stack([np.asarray(c['scaffold']['loading'], float) for c in circuits])
    load = load / (load.max(axis=1, keepdims=True) + 1e-12)
    perm = []
    for li, n in enumerate(layer_sizes):
        V = load[:, off[li]:off[li + 1]]
        if li == len(layer_sizes) - 1:
            perm.append(np.arange(n))
            continue
        owner = V.argmax(0)
        owner[V.max(0) < 0.05] = len(circuits)              # unused units last
        perm.append(np.lexsort((-V.max(0), owner)))
    load = np.concatenate([load[:, off[li]:off[li + 1]][:, p]
                           for li, p in enumerate(perm)], axis=1)

    for c in circuits:
        sc = c['scaffold']
        edges = [E[np.ix_(perm[bi + 1], perm[bi])] for bi, E in enumerate(sc['edges'])]
        negs = [E[np.ix_(perm[bi + 1], perm[bi])] for bi, E in enumerate(sc['neg_edges'])]
        for bi, (E, N) in enumerate(zip(edges, negs)):
            emax = max(E.max(), N.max(), 1e-12)
            for M, col, lsty, z, al, w0 in ((N, c_inh, (0, (1.3, 1.0)), 1, .5, 0.8),
                                            (E, c['color'], '-', 2, .9, 1.7)):
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        w = M[i, j] / emax
                        if w < 0.06:
                            continue
                        ax.plot([xs[bi][j], xs[bi + 1][i]],
                                [ys[bi][j], ys[bi + 1][i]], lw=0.12 + w0 * w,
                                color=col, linestyle=lsty, alpha=al, zorder=z,
                                solid_capstyle='round')

    # a unit's color is the circuit that claims most of its loading; grey if unused
    for li, n in enumerate(layer_sizes):
        V = load[:, off[li]:off[li + 1]]
        V = V / (V.max() + 1e-12)
        for u in range(n):
            v, owner = V[:, u].max(), int(V[:, u].argmax())
            if v < 0.05:
                ax.scatter([xs[li][u]], [ys[li][u]], s=6, c='none', edgecolors='0.72',
                           linewidths=0.4, zorder=3)
                continue
            col = matplotlib.colors.to_rgba(circuits[owner]['color'], 0.3 + 0.7 * v)
            ax.scatter([xs[li][u]], [ys[li][u]], s=7 + 30 * v, c=[col],
                       edgecolors='0.25', linewidths=0.4, zorder=3)

    ax.annotate('', xy=(-0.18, 0.5), xytext=(-0.60, 0.5),
                arrowprops=dict(arrowstyle='-|>', lw=0.5, color='0.45',
                                shrinkA=0, shrinkB=0, mutation_scale=5))
    ax.text(-0.64, 0.5, in_label, ha='right', va='center', fontsize=6, color='0.35')
    for li, lab in enumerate([r'$L_1$', r'$L_2$', r'$L_3$']):
        ax.text(li, -0.11, lab, ha='center', va='top', fontsize=6.5)
    for i, lab in enumerate(out_labels):
        col = next((c['color'] for c in circuits if c['name'] == lab), '0.4')
        ax.text(2.13, ys[2][i], lab, ha='left', va='center', fontsize=6.5, color=col)
    for yy, (col, lsty, lab) in enumerate((('0.45', '-', 'excitatory'),
                                           (c_inh, (0, (1.3, 1.0)), 'inhibitory'))):
        y = 0.17 - 0.13 * yy
        ax.plot([-1.30, -1.03], [y, y], color=col, linestyle=lsty, lw=0.9,
                solid_capstyle='round')
        ax.text(-0.98, y, lab, fontsize=6.2, ha='left', va='center', color='0.3')
    ax.set_xlim(-1.32, 2.75)
    ax.set_ylim(-0.22, 1.04)
    ax.axis('off')


# ── Figure 2 — class and sub-class circuits in the 8x4 even/odd MLP ───────────

def fig2_mlp_circuits(D):
    """Message: BFT splits the parity net into two class circuits that barely share
    hidden units and push-pull at the output, and each circuit refines into
    digit-selective sub-circuits on the way to the pixels."""
    CIRCUITS = _circuit_colors(D)
    LAYER_SIZES, OUT_LABELS = list(D['layer_sizes']), list(D['out_labels'])
    DIGIT_ORDER = list(D['digit_order'])
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    C_INH = figstyle.color('inhibitory')
    DIGIT_COLOR = digit_colors(DIGIT_ORDER, C_EVEN, C_ODD)

    # 0.49 (from 0.52) + a tighter inter-row gap trims a bit of height; the
    # aspect-locked pixel montages still fill their cells at this ratio.
    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='paper',
                   height_to_width_ratio=0.49)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1.36],
                          height_ratios=[1.0, 1.30], hspace=0.12)

    # ── (a) both circuits in one graph: shared units and output push-pull ────
    ax_a = fig.add_subplot(gs[0, 0])
    draw_scaffold_pair(ax_a, CIRCUITS, LAYER_SIZES, C_INH, OUT_LABELS)
    ax_a.set_title('(a) the two class circuits', pad=2, loc='left', fontweight='bold')

    # ── (b) the top of the trace tree, as per-digit loadings ────────────────
    gsb = gs[0, 1].subgridspec(3, 7, height_ratios=[1.0, 0.24, 1.0])
    # L3 even sits centered above its two L2 children (cols 0:2 and 2:4); L3 odd
    # above its single L2 child (cols 4:6). Bars carry the per-digit hues (as in
    # c,d) so the loadings read the same everywhere; the circuit-colored titles
    # sit over the plots. The black "(b)" panel label is placed after freeze.
    ax_b = [fig.add_subplot(gsb[0, 1:3]), fig.add_subplot(gsb[0, 4:6]),
            fig.add_subplot(gsb[2, 0:2]), fig.add_subplot(gsb[2, 2:4]),
            fig.add_subplot(gsb[2, 4:6])]
    fig.add_subplot(gsb[1, :]).set_axis_off()          # room for the tree connectors
    bp = dict(digit_order=DIGIT_ORDER, digit_color=DIGIT_COLOR)
    bar_panel(ax_b[0], CIRCUITS[0]['l3_profile'], xticks=False, ylab='loading',
              title=r'$L_3$ even', tcolor=C_EVEN, **bp)
    bar_panel(ax_b[1], CIRCUITS[1]['l3_profile'], xticks=False,
              title=r'$L_3$ odd', tcolor=C_ODD, **bp)
    bar_panel(ax_b[2], CIRCUITS[0]['l2_profiles'][0], yticklabels=True,
              title=r'$L_2\ f_0$', tcolor=C_EVEN, **bp)
    bar_panel(ax_b[3], CIRCUITS[0]['l2_profiles'][1], title=r'$L_2\ f_1$',
              tcolor=C_EVEN, **bp)
    bar_panel(ax_b[4], CIRCUITS[1]['l2_profiles'][0], title=r'$L_2\ f_0$',
              tcolor=C_ODD, **bp)

    # ── (d), (e) layer-1 sub-circuits: pixel arbor + per-digit loading ───────
    gs_bot = gs[1, :].subgridspec(2, 1, height_ratios=[0.13, 1.0])
    sp_de = fig.add_subplot(gs_bot[0]); sp_de.set_axis_off()   # room for the labels
    n_l1 = len(CIRCUITS[0]['l1_arbors'])                       # 4 factors per circuit
    gsd = gs_bot[1].subgridspec(2, 2 * n_l1 + 1, height_ratios=[1.0, 0.45],
                                width_ratios=[1] * n_l1 + [0.4] + [1] * n_l1)
    img_axes = {}
    for ci, c in enumerate(CIRCUITS):
        base = 0 if ci == 0 else n_l1 + 1
        cmap = seq_cmap(c['color'], c['name'])
        img_axes[ci] = []
        for k, M in enumerate(c['l1_arbors']):
            axi = fig.add_subplot(gsd[0, base + k])
            show_map(axi, M, cmap)
            top = DIGIT_ORDER[int(np.argmax(c['l1_profiles'][k]))]
            axi.text(0.05, 0.97, rf'$f_{k}$', transform=axi.transAxes, ha='left',
                     va='top', color=c['color'], fontsize=6.5)
            axi.text(0.95, 0.97, f'{top}', transform=axi.transAxes, ha='right',
                     va='top', color='0.3', fontsize=6.5, fontweight='bold')
            img_axes[ci].append(axi)
            bar_panel(fig.add_subplot(gsd[1, base + k]), c['l1_profiles'][k],
                      xticks=(k == 0), ylab='loading' if k == 0 else None, **bp)

    # ── settle constrained_layout, then freeze and add cross-axes annotation ──
    figstyle.freeze(fig)          # draw once, then switch the layout engine off

    # (b) panel label — black, at the top-left corner of the panel-b region (left
    # edge of the leftmost L2 axis, top of the L3 row), kept clear of the blue
    # "L3 even" title that sits over the loading plot.
    fig.text(ax_b[2].get_position().x0, ax_b[0].get_position().y1, '(b)',
             ha='left', va='bottom', color='k', fontweight='bold')

    for src, dsts in ((ax_b[0], (ax_b[2], ax_b[3])), (ax_b[1], (ax_b[4],))):  # tree
        b0 = src.get_position()
        x0, ytop = b0.x0 + b0.width / 2, b0.y0 - 0.012
        for d in dsts:
            b1 = d.get_position()
            x1, ybot = b1.x0 + b1.width / 2, b1.y1 + 0.058
            ymid = (ytop + ybot) / 2
            fig.add_artist(matplotlib.lines.Line2D([x0, x0, x1, x1],
                                                   [ytop, ymid, ymid, ybot],
                                                   color='0.5', lw=1.0, zorder=0))

    for ci, c in enumerate(CIRCUITS):                                    # block labels
        fig.text(max(img_axes[ci][0].get_position().x0 - 0.008, 0.002),
                 sp_de.get_position().y0,
                 f"({'cd'[ci]}) {c['name']} circuit — $L_1$ factors below $L_2\\,f_0$",
                 ha='left', va='bottom', color=c['color'], fontweight='bold')
    return fig


# ── Appendix A — decomposition details for the 8x4 MLP ────────────────────────

def figA_mlp_details(D):
    """Message: the Fig. 2 decomposition is well conditioned — a graded spectrum per
    node, an output layer wired as clean push-pull, hidden-unit supports that barely
    touch, inhibitory maps that mirror the excitatory ones, and driving stimuli that
    confirm what each L1 factor detects."""
    CIRCUITS = _circuit_colors(D)
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    C_INH = figstyle.color('inhibitory')
    CMAP_INH = seq_cmap(C_INH, 'inh')
    DIGIT_ORDER = list(D['digit_order'])
    OUT_LABELS = list(D['out_labels'])
    NODES = nodes_by_path(D)
    root = NODES[()]
    root_colors = [figstyle.color(k) for k in D['root_factor_color_keys']]

    anchors = {}
    figstyle.apply(venue='aaai2024', width='full', nrows=3, ncols=3, mode='appendix',
                   height_to_width_ratio=0.50)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    # thin spacer rows reserve room for panel labels, which are placed after the
    # layout is frozen (a long label inside an axes would inflate its grid cell).
    gs = fig.add_gridspec(6, 1, height_ratios=[0.10, 0.62, 0.11, 0.62, 0.11, 0.42],
                          hspace=0.03)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4)]
    for sp in spacers:
        sp.set_axis_off()
    gs_top = gs[1].subgridspec(1, 3, width_ratios=[1.25, 0.72, 1.00])

    # ── (a) lambda spectrum of every node in the trace ────────────────────────
    gsa = gs_top[0, 0].subgridspec(1, 5)
    nodes_a = [('$L_3$', D['root_lam'], root_colors),
               ('$L_2$ even', CIRCUITS[0]['l2_lam'], [C_EVEN] * 5),
               ('$L_2$ odd',  CIRCUITS[1]['l2_lam'], [C_ODD] * 5),
               ('$L_1$ even', CIRCUITS[0]['l1_lam'], [C_EVEN] * 5),
               ('$L_1$ odd',  CIRCUITS[1]['l1_lam'], [C_ODD] * 5)]
    # every panel shares the same x-scale, so a bar is the same physical width in
    # all of them — the single-factor L2-odd node then reads as one normal bar,
    # not a panel-filling block.
    kmax_a = max(len(lam) for _, lam, _ in nodes_a)
    for i, (name, lam, cols) in enumerate(nodes_a):
        ax = fig.add_subplot(gsa[0, i])
        ax.bar(np.arange(len(lam)), lam, color=cols[:len(lam)], edgecolor='0.25',
               linewidth=0.3, width=0.75)
        ax.set_xticks(np.arange(len(lam)))
        ax.set_xticklabels(np.arange(len(lam)))
        ax.set_xlim(-0.6, kmax_a - 0.4)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '', '1'] if i == 0 else [])
        ax.tick_params(length=1.5, pad=1)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_title(name, pad=1.5, fontsize=7)
        if i == 2:
            ax.set_xlabel('factor $f$', labelpad=1)
        if i == 0:
            ax.set_ylabel(r'$\lambda$ share', labelpad=1)
            anchors['a'] = ax

    # ── (b) output push-pull: each class factor drives one logit, suppresses the
    #        other. Mass is normalized per factor, so the two bars are comparable.
    gsb = gs_top[0, 1].subgridspec(1, 2)
    for i, c in enumerate(CIRCUITS):
        ax = fig.add_subplot(gsb[0, i])
        k = c['k']
        exc, inh = root['conn']['out_mass'][k], root['neg_conn']['out_mass'][k]
        tot = exc.sum() + inh.sum()
        x = np.arange(len(OUT_LABELS))
        ax.bar(x - 0.19, exc / tot, width=0.36, color=c['color'], edgecolor='0.25',
               linewidth=0.3, label='excites')
        ax.bar(x + 0.19, inh / tot, width=0.36, color=C_INH, edgecolor='0.25',
               linewidth=0.3, label='inhibits')
        ax.set_xticks(x); ax.set_xticklabels(OUT_LABELS)
        ax.set_ylim(0, 0.62); ax.set_yticks([0, 0.25, 0.5])
        ax.set_yticklabels(['0', '', '0.5'] if i == 0 else [])
        ax.tick_params(length=1.5, pad=1)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{c['name']} factor", color=c['color'], pad=1.5, fontsize=7)
        if i == 0:
            ax.set_ylabel('share of\narbor mass', labelpad=1)
            anchors['b'] = ax
            ax.legend(loc='upper center', frameon=False, fontsize=6.0,
                      handlelength=0.9, handletextpad=0.4, borderpad=0.1,
                      labelspacing=0.15, bbox_to_anchor=(1.06, 1.08))
        ax.set_xlabel('output unit', labelpad=1)

    # ── (c) how the two circuits divide the eight L1 units ───────────────────
    ax_c = fig.add_subplot(gs_top[0, 2])
    S = np.stack([unit_support(NODES[(c['k'],)]) for c in CIRCUITS])
    x = np.arange(S.shape[1])
    for i, c in enumerate(CIRCUITS):
        ax_c.bar(x + (i - 0.5) * 0.38, S[i], width=0.36, color=c['color'],
                 edgecolor='0.25', linewidth=0.3, label=f"{c['name']} circuit")
    ax_c.set_xticks(x); ax_c.set_xlabel('$L_1$ unit', labelpad=1)
    ax_c.set_ylabel('share of\ncircuit mass', labelpad=1)
    ax_c.set_ylim(0, S.max() * 1.28)                    # headroom for the legend
    ax_c.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_c.spines[s].set_visible(False)
    ax_c.legend(loc='upper center', ncol=2, frameon=False, fontsize=6.0,
                handlelength=0.9, handletextpad=0.4, borderpad=0.1,
                columnspacing=1.0, labelspacing=0.15, bbox_to_anchor=(0.5, 1.06))
    anchors['c'] = ax_c

    # ── (d) connection maps: excitatory vs inhibitory, L3 and L2 ─────────────
    gsd = gs[3].subgridspec(2, 5, width_ratios=[0.26] + [1] * 4)
    cols_d = [(rf"$L_3\ f_{{{CIRCUITS[0]['k']}}}$ (even)", CIRCUITS[0], 'l3_conn',
               C_EVEN),
              (rf"$L_3\ f_{{{CIRCUITS[1]['k']}}}$ (odd)", CIRCUITS[1], 'l3_conn',
               C_ODD),
              (r'$L_2$ even $f_0$', CIRCUITS[0], 'l2_conn', C_EVEN),
              (r'$L_2$ odd $f_0$',  CIRCUITS[1], 'l2_conn', C_ODD)]
    for r, lab in enumerate(('excitatory', 'inhibitory')):
        row_label(fig, gsd[r, 0], lab)
    for j, (name, c, key, col) in enumerate(cols_d):
        for r, (M, cmap) in enumerate(((c[key], seq_cmap(col, f'c{j}')),
                                       (c[key + '_neg'], CMAP_INH))):
            n_out, n_in = M.shape
            ax = fig.add_subplot(gsd[r, j + 1])
            ax.imshow(M, cmap=cmap, aspect='auto', interpolation='nearest',
                      vmin=0, vmax=M.max() or 1)
            ax.set_xticks(np.arange(n_in)); ax.set_yticks(np.arange(n_out))
            ax.set_xticklabels(np.arange(n_in) if r == 1 else [], fontsize=6)
            ax.set_yticklabels(np.arange(n_out), fontsize=6)
            ax.tick_params(length=1.2, pad=0.8)
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(name, color=col, pad=1.5, fontsize=7)
                if j == 0:
                    anchors['d'] = ax
            elif j == 1:
                ax.set_xlabel('input unit', labelpad=1)

    # ── (e, f) what drives each L1 factor ────────────────────────────────────
    n_l1 = len(CIRCUITS[0]['l1_wavg'])                   # 4 factors per circuit
    gse = gs[5].subgridspec(1, 2 * n_l1 + 2,
                            width_ratios=[0.42] + [1] * n_l1 + [0.22] + [1] * n_l1)
    row_label(fig, gse[0, 0], 'weighted avg.\nstimulus')
    first_img = {}
    for ci, c in enumerate(CIRCUITS):
        base = 1 if ci == 0 else n_l1 + 2
        for k, wavg in enumerate(c['l1_wavg']):
            ax = fig.add_subplot(gse[0, base + k])
            ax.imshow(wavg, cmap='gray_r', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            top = DIGIT_ORDER[int(np.argmax(c['l1_profiles'][k]))]
            ax.text(0.05, 0.97, rf'$f_{k}$', transform=ax.transAxes, ha='left',
                    va='top', color=c['color'], fontsize=6.5)
            ax.text(0.95, 0.97, f'{top}', transform=ax.transAxes, ha='right',
                    va='top', color='0.3', fontsize=6.5, fontweight='bold')
            if k == 0:
                first_img[ci] = ax

    # ── settle constrained_layout, then freeze and add the panel labels ──────
    figstyle.freeze(fig)          # draw once, then switch the layout engine off
    spacer_label(fig, anchors['a'], spacers[0], '(a) factor spectra', dx=-0.030)
    spacer_label(fig, anchors['b'], spacers[0], '(b) output push-pull', dx=-0.036)
    spacer_label(fig, anchors['c'], spacers[0], '(c) $L_1$ unit support', dx=-0.036)
    spacer_label(fig, anchors['d'], spacers[1],
                 r'(d) connection maps (out $\times$ in)', dx=-0.030)
    for ci, c in enumerate(CIRCUITS):
        spacer_label(fig, first_img[ci], spacers[2],
                     f"({'ef'[ci]}) {c['name']} circuit — $L_1$ factors",
                     color=c['color'], dx=-0.006)
    return fig


def pca_fit(X):
    """2-D PCA basis of the (cosine-normalized) rows, via SVD. numpy only.

    Sign is canonicalized on each component's largest-magnitude loading, so the
    embedding of a given bundle always comes out the same way up.
    """
    U = unit(np.asarray(X, dtype=float))
    mu = U.mean(0)
    _, S, Vt = np.linalg.svd(U - mu, full_matrices=False)
    Vt = Vt[:2].copy()
    for j in range(len(Vt)):
        if Vt[j][np.argmax(np.abs(Vt[j]))] < 0:
            Vt[j] = -Vt[j]
    return mu, Vt, (S ** 2 / (S ** 2).sum())[:2]


def pca_apply(X, mu, Vt):
    return (unit(np.asarray(X, dtype=float)) - mu) @ Vt.T


def pca2(X):
    mu, Vt, evr = pca_fit(X)
    return pca_apply(X, mu, Vt), evr


def class_colors(classes):
    """Per-class hues for classes that have no color of their own (ten CIFAR
    categories, ten digits). Defined in .figstyle/colors.yaml as class_0..9."""
    return {int(c): figstyle.color(f'class_{i % 10}') for i, c in enumerate(classes)}


def pca_panel(ax, X, labels, colors, *, title=None, order=None, s=2.2, alpha=0.75,
              sil=True, note=None):
    """One 2-D embedding panel: dots colored by class, no ticks (PCA units carry
    no meaning), explained variance on the axes."""
    coords, evr = pca2(X)
    labels = np.asarray(labels)
    for c in (order if order is not None else np.unique(labels)):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=s, color=colors[int(c)],
                   edgecolor='none', alpha=alpha, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f'PC 1 (var {evr[0]:.2f})', labelpad=1, fontsize=6)
    ax.set_ylabel(f'PC 2 (var {evr[1]:.2f})', labelpad=1, fontsize=6)
    for s_ in ax.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    if title:
        ax.set_title(title, pad=1.5, fontsize=6.5)
    txt = note
    if sil:
        txt = (f'sil {cosine_silhouette(X, labels):.2f}'
               + (f'\n{note}' if note else ''))
    if txt:
        ax.text(0.035, 0.975, txt, transform=ax.transAxes, ha='left', va='top',
                fontsize=6, linespacing=1.2, color='0.15',
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.8))
    return coords


def cond_embedding(ax, F_id, labels, colors, *, near=None, far=(), c_near=None,
                   c_far=None, order=None, near_label='near-OOD', s=3.5,
                   legend=True, id_label=None):
    """The in-distribution fingerprints in their own PCA plane, with the OOD
    conditions projected into the same plane — where does OOD input land?"""
    mu, Vt, evr = pca_fit(F_id)
    if len(far):
        P = np.concatenate([pca_apply(X, mu, Vt) for X in far])
        ax.scatter(P[:, 0], P[:, 1], s=s, color=c_far, alpha=0.5, edgecolor='none',
                   zorder=1, rasterized=True)
    if near is not None:
        P = pca_apply(near, mu, Vt)
        ax.scatter(P[:, 0], P[:, 1], s=s, color=c_near, alpha=0.55, edgecolor='none',
                   zorder=2, rasterized=True)
    P = pca_apply(F_id, mu, Vt)
    labels = np.asarray(labels)
    for c in (order if order is not None else np.unique(labels)):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=s, color=colors[int(c)], alpha=0.8,
                   edgecolor='none', zorder=3, rasterized=True)
    if legend:
        for lab, col in ((id_label or 'in-distribution', '0.35'),
                         (near_label, c_near), ('far-OOD', c_far)):
            ax.scatter([], [], s=8, color=col, label=lab)
        ax.legend(fontsize=6, frameon=False, loc='upper left', handlelength=0.7,
                  handletextpad=0.25, borderpad=0.1, labelspacing=0.18,
                  borderaxespad=0.15, scatterpoints=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f'PC 1 (var {evr[0]:.2f})', labelpad=1, fontsize=6)
    ax.set_ylabel(f'PC 2 (var {evr[1]:.2f})', labelpad=1, fontsize=6)
    for s_ in ax.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    return evr


def act_rep(D, which=-1):
    """The activation baseline stored next to the fingerprints, or None.

    Written by `scripts/add_activation_baselines.py`; `aligned` says whether the
    rows are the very stimuli the fingerprints came from (then `index` selects
    them) or an independent sample of the same test set (then `labels` are its
    own).
    """
    act = D.get('act')
    if not act:
        return None
    rep = act['reps'][which]
    # the bundle names a hidden layer one index higher than the circuit figures do
    # (it counts the pixels as $L_1$), so shift it down to match what the reader sees
    label = re.sub(r'\$L_(\d)\$', lambda m: f'$L_{int(m.group(1)) - 1}$',
                   str(rep['label']))
    return dict(X=rep['X'], label=label, dim=int(rep['dim']),
                aligned=bool(act.get('aligned', 0)),
                index=act.get('index'), labels=act.get('labels'))


def fp_vs_act(D, fp_labels, which=-1):
    """(fingerprint X, labels) and (activation X, labels) over the same stimuli."""
    rep = act_rep(D, which)
    if rep is None:
        return None
    if rep['aligned']:
        idx = np.asarray(rep['index'])
        return (D['fp']['id'][idx], np.asarray(fp_labels)[idx],
                rep['X'], np.asarray(fp_labels)[idx], rep)
    return (D['fp']['id'], np.asarray(fp_labels), rep['X'],
            np.asarray(rep['labels']), rep)


def ramp_cmap(hex_color, name):
    """white -> color, gentle 5-stop ramp (similarity matrices)."""
    p = np.array(matplotlib.colors.to_rgb(hex_color))
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        name, [(1, 1, 1), tuple(p + (1 - p) * 0.78), tuple(p + (1 - p) * 0.42),
               tuple(p), tuple(p * 0.55)])


def resolve_color(key, digit_color=None):
    """Bundle color keys -> matplotlib colors ('digit:4' and grey levels included)."""
    if not isinstance(key, str):
        return key
    if key.startswith('digit:'):
        return digit_color[int(key.split(':')[1])]
    if key.startswith('#') or key.replace('.', '').isdigit():
        return key
    return figstyle.color(key)


def draw_scaffold_backbone(ax, edges, neg_edges, loading, layer_sizes, color, c_inh,
                           top_in=1):
    """Circuit backbone for a wider net: each unit's top-`top_in` incoming edges.

    A 40->20->10 arbor has ~1000 edges; drawing them all is an unreadable
    hairball, so only the strongest inputs of each active unit are shown.
    """
    ys = [np.linspace(0.98, 0.02, n) if n > 1 else np.array([0.5]) for n in layer_sizes]
    xs = [np.full(n, float(li)) for li, n in enumerate(layer_sizes)]
    off = np.cumsum([0] + list(layer_sizes))
    load = np.asarray(loading, float)

    for bi, (E, N) in enumerate(zip(edges, neg_edges)):
        emax = max(E.max(), N.max(), 1e-12)
        tgt = load[off[bi + 1]:off[bi + 2]]
        tgt = tgt / (tgt.max() + 1e-12)
        for M, col, lsty, z, al, sc in ((N, c_inh, (0, (1.2, 1.0)), 1, .5, 0.9),
                                        (E, color, '-', 2, .85, 1.4)):
            for i in range(M.shape[0]):
                if tgt[i] < 0.05:                      # inactive target unit
                    continue
                for j in np.argsort(M[i])[::-1][:top_in]:
                    w = M[i, j] / emax
                    if w < 0.05:
                        continue
                    ax.plot([xs[bi][j], xs[bi + 1][i]], [ys[bi][j], ys[bi + 1][i]],
                            lw=0.08 + sc * w, color=col, linestyle=lsty, alpha=al,
                            zorder=z, solid_capstyle='round')

    for li, n in enumerate(layer_sizes):
        v = load[off[li]:off[li + 1]]
        v = v / (v.max() + 1e-12)
        on = v >= 0.05
        ax.scatter(xs[li][on], ys[li][on], s=1.5 + 9 * v[on],
                   c=[matplotlib.colors.to_rgba(color, 0.3 + 0.7 * vi) for vi in v[on]],
                   edgecolors='0.3', linewidths=0.25, zorder=3)
        ax.scatter(xs[li][~on], ys[li][~on], s=1.2, c='none', edgecolors='0.75',
                   linewidths=0.25, zorder=3)
    ax.set_xlim(-0.35, len(layer_sizes) - 0.65)
    ax.set_ylim(-0.04, 1.04)
    ax.axis('off')


def style_matrix_axes(ax):
    for s in ax.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax.tick_params(length=1.5, pad=1)


# ── shared fingerprint panel drawers (used by the main figure and appendices) ─

def mlp_factor_tree(ax, D, digit=None, amax=None, label=None, label_color='0.2',
                    show_layers=True):
    """The 13 factors of the 8x4 MLP trace as the tree they form: the output
    layer's two factors branch into L2 factors and those into L1 factors.

    One node per column of the fingerprint heatmap, laid out left to right in the
    same L3->L2->L1 order — the picture of where the heatmap's columns come from.

    Without `digit` the nodes are shaded by each factor's mean loading over all
    digits. With one, they are shaded by how strongly *that* digit drives the
    factor, so drawing the same tree for two digits shows the fingerprint changing
    with the stimulus: an even digit lights the even branch, an odd digit the odd
    one. Pass a shared `amax` so the two copies stay on one intensity scale.
    """
    dims = D['dims']
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    CMAP = {'even': seq_cmap(C_EVEN, 'ev'), 'odd': seq_cmap(C_ODD, 'od')}

    # each factor's share of a fingerprint — the columns of the heatmap
    M = D['fp_mean_by_digit']
    Mn = M / M.sum(1, keepdims=True)
    if digit is None:
        act = Mn.mean(0)
    else:
        act = Mn[[int(d) for d in D['digit_order']].index(int(digit))]
    amax = float(act.max()) if amax is None else float(amax)

    LAYERS = [2, 1, 0]                      # left to right: the trace's direction
    BLOCKS = [('even', C_EVEN, list(D['cols_even']), 0.755),
              ('odd', C_ODD, list(D['cols_odd']), 0.215)]
    SPREAD = {2: 0.0, 1: 0.115, 0: 0.115}   # vertical step between sibling factors

    for name, col, cols, yc in BLOCKS:
        # columns of this circuit, grouped by layer and kept in bundle order
        by_layer = {li: [c for c in cols if int(dims[c, 0]) == li] for li in LAYERS}
        pos = {}
        for xi, li in enumerate(LAYERS):
            ks = by_layer[li]
            step = SPREAD[li] if len(ks) > 1 else 0.0
            ys = yc - step * (np.arange(len(ks)) - (len(ks) - 1) / 2)
            for c, y in zip(ks, ys):
                pos[c] = (float(xi), float(y))

        # edges: only the top-lambda factor of a node branches (n_branches=1 below
        # the output), so every factor of a layer hangs off its parent's factor 0
        for li_par, li_ch in ((2, 1), (1, 0)):
            if not by_layer[li_par] or not by_layer[li_ch]:
                continue
            x0, y0 = pos[by_layer[li_par][0]]
            for c in by_layer[li_ch]:
                x1, y1 = pos[c]
                ax.plot([x0, x1], [y0, y1], color=col, lw=0.55, alpha=0.55,
                        zorder=1, solid_capstyle='round')

        for c, (x, y) in pos.items():
            v = float(act[c]) / (amax + 1e-12)
            ax.scatter([x], [y], s=11 + 62 * v, zorder=3, linewidths=0.4,
                       edgecolors='0.3', c=[CMAP[name](0.16 + 0.72 * v)])
        ax.text(-0.17, yc, name, ha='right', va='center', fontsize=6.5, color=col)

    if label:
        ax.text(-0.78, 1.20, label, ha='left', va='top', fontsize=6.5,
                color=label_color)
    if not show_layers:
        ax.set_xlim(-0.78, 2.30)
        ax.set_ylim(-0.06, 1.30 if label else 1.06)
        ax.axis('off')
        return
    for xi, li in enumerate(LAYERS):
        n = int((dims[:, 0] == li).sum())
        # layer and factor count on one line — stacked they collide once the tree
        # is drawn at half height (two copies in one cell)
        ax.text(xi, -0.045, rf'$L_{{{li + 1}}}$ ({n})', ha='center', va='top',
                fontsize=6.5)
    ax.set_xlim(-0.78, 2.30)
    ax.set_ylim(-0.24, 1.30 if label else 1.06)
    ax.axis('off')


def mlp_fp_heatmap(ax, D, digit_color):
    """Mean fingerprint per digit for the 8x4 MLP, columns grouped by circuit
    (blue even, red odd) then layer — the introduction to what a fingerprint is."""
    DIGIT_ORDER = list(D['digit_order'])
    COL_ORDER, N_EV = list(D['col_order']), int(D['n_ev'])
    COLS_EVEN, COLS_ODD = list(D['cols_even']), list(D['cols_odd'])
    dims = D['dims']
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    CMAP_EVEN, CMAP_ODD = seq_cmap(C_EVEN, 'ev'), seq_cmap(C_ODD, 'od')

    def layer_groups(cols):
        out, s = [], 0
        for i in range(1, len(cols) + 1):
            if i == len(cols) or dims[cols[i], 0] != dims[cols[s], 0]:
                out.append((s, i, int(dims[cols[s], 0])))
                s = i
        return out

    M = D['fp_mean_by_digit']
    M = M / M.sum(1, keepdims=True)
    Mo = M[:, COL_ORDER]
    _col = np.arange(len(COL_ORDER))
    ax.imshow(np.where(_col < N_EV, Mo, np.nan), cmap=CMAP_EVEN, vmin=0, vmax=Mo.max(),
              aspect='auto', interpolation='nearest')
    ax.imshow(np.where(_col >= N_EV, Mo, np.nan), cmap=CMAP_ODD, vmin=0, vmax=Mo.max(),
              aspect='auto', interpolation='nearest')
    ax.axvline(N_EV - 0.5, color='0.25', lw=0.8)
    for off, cols in ((0, COLS_EVEN), (N_EV, COLS_ODD)):
        for s0, s1, lay in layer_groups(cols):
            if s0:
                ax.axvline(off + s0 - 0.5, color='0.75', lw=0.4)
            ax.text(off + (s0 + s1 - 1) / 2, len(DIGIT_ORDER) - 0.35,
                    rf'$L_{{{lay + 1}}}$', ha='center', va='top', fontsize=6.0)
    ax.set_xticks([])
    ax.set_yticks(range(len(DIGIT_ORDER))); ax.set_yticklabels(DIGIT_ORDER)
    for t, d in zip(ax.get_yticklabels(), DIGIT_ORDER):
        t.set_color(digit_color[d])
    ax.set_ylabel('stimulus digit', labelpad=1)
    ax.set_xlabel('factor, by circuit and layer', labelpad=2)
    ax.tick_params(length=1.5, pad=1)
    for s in ax.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax.text((N_EV - 1) / 2, -0.72, 'even circuit', ha='center', va='bottom',
            color=C_EVEN, fontsize=6.5)
    ax.text((N_EV + len(COL_ORDER) - 1) / 2, -0.72, 'odd circuit', ha='center',
            va='bottom', color=C_ODD, fontsize=6.5)
    ax.set_ylim(len(DIGIT_ORDER) + 0.25, -1.1)


def mlp_heldout(ax, D, digit_color, c_even, c_odd, r_ood):
    """Held-out digits: the odd-circuit share of the fingerprint tracks the
    network's own P(odd), not the true parity."""
    ax.plot([0, 1], [0, 1], color='0.7', lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    for d, x, y in D['id_pts']:
        ax.scatter(x, y, s=13, facecolor='none', edgecolor=digit_color[int(d)],
                   linewidths=0.8, zorder=2)
    LBL_OFF = {2: (3.4, -1.0), 5: (2.0, -6.4), 6: (3.4, -1.0),
               7: (-1.5, -6.4), 8: (3.4, -1.0), 9: (3.4, -1.0)}
    for d, x, y in D['ood_pts']:
        d = int(d)
        c = c_even if d % 2 == 0 else c_odd
        ax.scatter(x, y, s=15, color=c, edgecolor='none', zorder=3)
        ax.annotate(str(d), (x, y), textcoords='offset points',
                    xytext=LBL_OFF.get(d, (3.4, -1.0)), fontsize=6.5, color=c)
    ax.set_xlim(-0.06, 1.06); ax.set_ylim(-0.06, 1.06)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel('P(network says odd)', labelpad=1)
    ax.set_ylabel('odd-circuit share', labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.text(0.04, 0.97, rf'$r={r_ood:.3f}$', transform=ax.transAxes,
            ha='left', va='top', fontsize=6.5)
    ax.scatter([], [], s=13, facecolor='none', edgecolor='0.45', linewidths=0.8,
               label='trained digits')
    ax.scatter([], [], s=15, color='0.45', edgecolor='none', label='held-out digits')
    ax.legend(fontsize=6, frameon=False, loc='lower right', handlelength=0.9,
              handletextpad=0.3, borderpad=0.1, labelspacing=0.2, borderaxespad=0.2)


# ── the single fingerprint main figure (8x4 MLP intro + CIFAR-10 CNN result) ──

def fig4_fingerprints_main(D):
    """Message: a BFT fingerprint is the vector of factor loadings a stimulus
    evokes; on the legible 8x4 MLP (a) it is a class-structured code and (b) its
    columns are the circuits BFT finds; on the CIFAR-10 CNN (c-f) the 156-d
    fingerprint stays class-structured and is as class-separable as the network's
    own 256-d penultimate activations."""
    from src import figdata
    D_mlp = figdata.load('nb01_fingerprints')

    # -- CNN (this bundle) --
    N_CLASSES = int(D['n_classes']); CLS = list(D['class_names'])
    COL_ORDER, BLK_EDGE, BLK_START, BLOCK_SIZES = _cnn_fp_blocks(D)
    C_BFT = figstyle.color('ours')
    CMAP = ramp_cmap(C_BFT, 'bft')
    CCOL = class_colors(range(N_CLASSES))
    n_factors = int(D['n_factors'])
    fp = D['fp']
    EMB = fp_vs_act(D, fp['id_targets'])

    # -- MLP intro --
    DIGIT_ORDER = list(D_mlp['digit_order'])
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    DIGIT_COLOR = digit_colors(DIGIT_ORDER, C_EVEN, C_ODD)

    # -- ImageNet (third row): the same code on a pretrained 1000-way SqueezeNet --
    D_img = figdata.load('nb05_fingerprints')
    IMG_CLS = list(D_img['class_names'])
    N_IMG = int(D_img['n_classes'])
    ICOL = class_colors(range(N_IMG))
    IMG_EMB = fp_vs_act(D_img, D_img['fp']['id_targets'])

    W, SP = 6.975, 0.13
    # rows 2-3 (the CIFAR and ImageNet embedding panels) are stretch-to-fit
    # scatters/heatmaps, so they can be shorter than row 1 without losing detail.
    rows = [SP, 1.30, SP, 1.12, SP, 1.12]
    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='paper',
                   height_to_width_ratio=sum(rows) / W)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.02, wspace=0.03)
    gs = fig.add_gridspec(6, 1, height_ratios=rows, hspace=0.02)
    sp1 = fig.add_subplot(gs[0]); sp1.set_axis_off()
    sp2 = fig.add_subplot(gs[2]); sp2.set_axis_off()
    sp3 = fig.add_subplot(gs[4]); sp3.set_axis_off()
    gs_top = gs[1].subgridspec(1, 3, width_ratios=[1.10, 1.00, 2.10])
    gs_bot = gs[3].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.12])
    gs_img = gs[5].subgridspec(1, 3, width_ratios=[1.0, 1.0, 1.30])

    # (a) where a fingerprint's entries come from: the same factor tree lit up by
    # two different stimuli — a 0 (even) and a 1 (odd). Node shade is how strongly
    # that digit drives the factor, on one shared scale, so the pair *is* the
    # fingerprint: the even branch lights for the 0, the odd branch for the 1.
    A_DIGITS = [0, 1]
    _Mn = D_mlp['fp_mean_by_digit']
    _Mn = _Mn / _Mn.sum(1, keepdims=True)
    _dord = [int(d) for d in D_mlp['digit_order']]
    A_MAX = max(float(_Mn[_dord.index(d)].max()) for d in A_DIGITS)
    gsa = gs_top[0, 0].subgridspec(len(A_DIGITS), 1)
    ax_a = None
    for _i, _dg in enumerate(A_DIGITS):
        _axt = fig.add_subplot(gsa[_i, 0])
        mlp_factor_tree(_axt, D_mlp, digit=_dg, amax=A_MAX,
                        label=f'digit {_dg}', label_color=DIGIT_COLOR[_dg],
                        show_layers=(_i == len(A_DIGITS) - 1))
        if _i == 0:
            ax_a = _axt

    # (b) the fingerprint itself: mean loading per digit over those same factors
    ax_b = fig.add_subplot(gs_top[0, 1])
    mlp_fp_heatmap(ax_b, D_mlp, DIGIT_COLOR)

    # (c) the CNN's mean fingerprint per class, columns grouped by output circuit
    ax_c = fig.add_subplot(gs_top[0, 2])
    cnn_fp_heatmap(ax_c, D, COL_ORDER, BLK_EDGE, CMAP, CCOL, CLS, N_CLASSES,
                   n_factors)

    # (d, e) the CNN embedding, against the network's own representation
    ax_d = fig.add_subplot(gs_bot[0, 0])
    ax_e = fig.add_subplot(gs_bot[0, 1])
    if EMB is not None:
        Xf, lf, Xa, la, rep = EMB
        pca_panel(ax_d, Xf, lf, CCOL)
        pca_panel(ax_e, Xa, la, CCOL,
                  note=None if rep['aligned'] else 'independent sample')
    else:
        for ax in (ax_d, ax_e):
            _val_na(ax, 'no activation baseline\nin this bundle')

    # (f) the class geometry the fingerprint induces
    ax_f = fig.add_subplot(gs_bot[0, 2])
    cnn_class_geometry(ax_f, D, CMAP, CCOL, CLS, N_CLASSES, show_between=True)

    # ── row 3: the same two codes on ImageNet, and the head-to-head ──────────
    # (g) the ImageNet fingerprint embedding
    ax_g = fig.add_subplot(gs_img[0, 0])
    pca_panel(ax_g, D_img['fp']['id'], D_img['fp']['id_targets'], ICOL)

    # (h) the network's own activations on the same stimuli. The ImageNet bundle
    # carries no activation baseline (add_activation_baselines.py needs the val
    # images, which the export does not ship), so the panel says so rather than
    # quietly disappearing.
    ax_h = fig.add_subplot(gs_img[0, 1])
    if IMG_EMB is not None:
        Xf, lf, Xa, la, rep = IMG_EMB
        pca_panel(ax_h, Xa, la, ICOL,
                  note=None if rep['aligned'] else 'independent sample')
    else:
        _val_na(ax_h, 'no ImageNet activation\nbaseline in this bundle\n'
                      '(see Fig. M for the\ndimension-matched score)')

    # (i) fingerprint vs. the network's own activations — a compact recap of the
    # silhouettes the embeddings above already carry (d/e for CIFAR-10, g/h for
    # ImageNet), each code measured at its own native dimension.
    ax_i = fig.add_subplot(gs_img[0, 2])
    C_BASE = figstyle.color('baseline')
    # point + 95% stimulus-bootstrap CI for each native-dimension silhouette
    cnn_fp = silhouette_ci(EMB[0], EMB[1]) if EMB is not None else None
    cnn_act = silhouette_ci(EMB[2], EMB[3]) if EMB is not None else None
    img_fp = silhouette_ci(D_img['fp']['id'], D_img['fp']['id_targets'])
    img_act = silhouette_ci(IMG_EMB[2], IMG_EMB[3]) if IMG_EMB is not None else None
    SETS = [('CIFAR-10\nCNN', cnn_fp, cnn_act),
            ('ImageNet\nSqueezeNet', img_fp, img_act)]
    SETS = [s for s in SETS if s[1] is not None and s[2] is not None]
    labels = [s[0] for s in SETS]
    fp_s = [s[1][0] for s in SETS];  act_s = [s[2][0] for s in SETS]
    fp_err = np.array([[s[1][0] - s[1][1], s[1][2] - s[1][0]] for s in SETS]).T
    act_err = np.array([[s[2][0] - s[2][1], s[2][2] - s[2][0]] for s in SETS]).T
    xg = np.arange(len(SETS))
    ax_i.bar(xg - 0.19, fp_s, width=0.36, color=C_BFT, edgecolor='0.25',
             linewidth=0.3, label='BFT fingerprint')
    ax_i.bar(xg + 0.19, act_s, width=0.36, color=C_BASE, edgecolor='0.25',
             linewidth=0.3, label='network activations')
    ax_i.errorbar(xg - 0.19, fp_s, yerr=fp_err, fmt='none', ecolor='0.2',
                  elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=5)
    ax_i.errorbar(xg + 0.19, act_s, yerr=act_err, fmt='none', ecolor='0.2',
                  elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=5)
    ax_i.set_xticks(xg)
    ax_i.set_xticklabels(labels, fontsize=6, linespacing=1.15)
    ax_i.set_ylim(0, max(fp_s + act_s) * 1.34)
    ax_i.set_yticks([0, 0.2, 0.4])
    ax_i.set_ylabel('silhouette', labelpad=1)
    ax_i.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_i.spines[s_].set_visible(False)
    ax_i.legend(fontsize=6, frameon=False, loc='upper right', handlelength=0.8,
                handletextpad=0.35, borderpad=0.1, labelspacing=0.18,
                borderaxespad=0.0)

    figstyle.freeze(fig)
    # Every label names the dataset it comes from — the rows are not one model
    # each (row 1 crosses from MNIST to CIFAR-10), so the provenance has to be on
    # the panel rather than implied by position.
    emb_lab = ((f'(d) CIFAR-10 · fingerprint, {n_factors}-d',
                f"(e) CIFAR-10 · {EMB[4]['label'].replace(chr(10), ' ')}, "
                f"{EMB[4]['dim']}-d")
               if EMB is not None
               else ('(d) CIFAR-10 · fingerprint', '(e) CIFAR-10 · activations'))
    img_lab = ((f"(h) ImageNet · {IMG_EMB[4]['label'].replace(chr(10), ' ')}, "
                f"{IMG_EMB[4]['dim']}-d") if IMG_EMB is not None
               else '(h) ImageNet · activations')
    for ax, lab, anchor in ((ax_a, '(a) MNIST · 8×4 MLP trace', sp1),
                            (ax_b, '(b) MNIST · its fingerprint', sp1),
                            (ax_c, '(c) CIFAR-10 · fingerprint per class', sp1),
                            (ax_d, emb_lab[0], sp2), (ax_e, emb_lab[1], sp2),
                            (ax_f, '(f) CIFAR-10 · class geometry', sp2),
                            (ax_g, f"(g) ImageNet · fingerprint, "
                                   f"{int(D_img['n_factors'])}-d", sp3),
                            (ax_h, img_lab, sp3),
                            (ax_i, '(i) both · fingerprint vs. activations', sp3)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002),
                 anchor.get_position().y0, lab, ha='left', va='bottom', fontweight='bold')
    return fig


# ── Appendix B — class circuits in the 40x20 digit MLP (merged main + details) ─

def figB_digit_mlp_details(D):
    """Message: the ten-class net does not factorize into digits at the output — it
    factorizes into seven overlapping, distributed circuits that each pool several
    digits, and tracing a circuit backward un-pools it into digit-selective factors.
    Merges the former main figure (output factors, un-pooling, distributed support,
    the digit-detector gallery) with its decomposition detail (spectra, the traced
    circuit of every output factor)."""
    CIRCUITS = D['circuits']
    n_c, N_SHOW, N_DIGITS = len(CIRCUITS), int(D['n_show']), int(D['n_digits'])
    LAYER_SIZES = list(D['layer_sizes'])
    Sup, root_pur = D['support'], D['root_pur']
    NODES = nodes_by_path(D)
    C_BFT, C_INH = figstyle.color('ours'), figstyle.color('inhibitory')
    C_L3, C_L1 = C_BFT, tint(C_BFT, 0.45)
    CMAP = seq_cmap(C_BFT, 'bft')

    W = 6.975
    s_g = W / (0.30 + n_c)                              # gallery arbor side, square
    rows = [0.12, 1.30, 0.12, 0.72, 0.12, 1.45, 0.14, N_SHOW * s_g + 0.02]
    anchors = {}
    figstyle.apply(venue='aaai2024', width='full', nrows=4, ncols=3, mode='appendix',
                   height_to_width_ratio=sum(rows) / W)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    gs = fig.add_gridspec(8, 1, height_ratios=rows, hspace=0.03)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4, 6)]
    for sp in spacers:
        sp.set_axis_off()
    gs_top = gs[1].subgridspec(1, 3, width_ratios=[1.20, 0.86, 1.26])

    # ── (a) what each output-layer factor stands for: a group of digits, not one ─
    ax_a = fig.add_subplot(gs_top[0, 0])
    Mprof = np.stack([c['profile'] for c in CIRCUITS])
    im = ax_a.imshow(Mprof, cmap=CMAP, aspect='auto', vmin=0, vmax=Mprof.max())
    ax_a.set_xticks(range(N_DIGITS)); ax_a.set_xticklabels(range(N_DIGITS))
    ax_a.set_yticks(range(n_c))
    ax_a.set_yticklabels([f"$f_{c['k']}$" for c in CIRCUITS])
    ax_a.set_xlabel('digit', labelpad=1)
    ax_a.set_ylabel('output factor', labelpad=1)
    style_matrix_axes(ax_a)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.038, pad=0.03)
    cb.set_label('digit share', labelpad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1)
    anchors['a'] = ax_a

    # ── (b) un-pooling: each circuit's purest factor gets purer toward the input ─
    ax_b = fig.add_subplot(gs_top[0, 1])
    best = np.zeros((n_c, 3))
    for j, c in enumerate(CIRCUITS):
        k = int(c['k'])
        best[j] = [root_pur[k],
                   NODES[(k,)]['class_profile'].max(1).max(),
                   NODES[(k, 0)]['class_profile'].max(1).max()]
        for xi, node in enumerate((NODES[(k,)], NODES[(k, 0)]), start=1):
            pur = node['class_profile'].max(1)
            ax_b.scatter(np.full(len(pur), xi) + (j - 3) * 0.028, pur, s=3.5,
                         color='0.72', edgecolor='none', zorder=1)
    for j in range(n_c):
        ax_b.plot([0, 1, 2], best[j], color=C_L1, lw=0.6, alpha=0.85, zorder=2)
    ax_b.plot([0, 1, 2], best.mean(0), color=C_L3, lw=1.4, marker='o', ms=2.6,
              zorder=3)
    ax_b.axhline(1 / N_DIGITS, color='0.6', lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax_b.text(2.42, 1 / N_DIGITS + 0.01, 'chance', fontsize=6, color='0.45',
              ha='right', va='bottom')
    for x, m in zip([0, 1, 2], best.mean(0)):
        ax_b.text(x, m + 0.045, f'{m:.2f}', ha='center', va='bottom', fontsize=6,
                  color=C_L3)
    ax_b.set_xticks([0, 1, 2]); ax_b.set_xticklabels([r'$L_3$', r'$L_2$', r'$L_1$'])
    ax_b.set_xlim(-0.45, 2.45); ax_b.set_ylim(0, 1.06)
    ax_b.set_yticks([0, 0.5, 1.0])
    ax_b.set_ylabel('digit purity', labelpad=1)
    ax_b.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_b.spines[s].set_visible(False)
    anchors['b'] = ax_b

    def covered(i, j):
        top = np.argsort(-CIRCUITS[i]['profile'])[:N_SHOW]
        P = CIRCUITS[j]['l1_profiles']
        return sum(int(P[int(P[:, d].argmax())].argmax()) == d for d in top)

    own = sum(covered(i, i) for i in range(n_c))
    ctrl = np.mean([covered(i, j) for i in range(n_c) for j in range(n_c) if i != j])
    ax_b.text(0.5, -0.22, f'{own} of {n_c * N_SHOW} pooled digits get their own\n'
              f'$L_1$ factor (other circuits: {ctrl * n_c:.0f})',
              transform=ax_b.transAxes, ha='center', va='top', fontsize=6,
              color='0.35', linespacing=1.2)

    # ── (c) the circuits are distributed: most L1 units serve most circuits ────
    ax_c = fig.add_subplot(gs_top[0, 2])
    unit_order = np.argsort(-Sup.mean(0))
    ax_c.imshow((Sup / Sup.max(1, keepdims=True))[:, unit_order], cmap=CMAP,
                aspect='auto', vmin=0, vmax=1)
    ax_c.set_xticks([])
    ax_c.set_yticks(range(n_c))
    ax_c.set_yticklabels([f"$f_{c['k']}$" for c in CIRCUITS])
    ax_c.set_xlabel(f'{LAYER_SIZES[0]} $L_1$ units, sorted by mean use', labelpad=2)
    style_matrix_axes(ax_c)
    ov, null = support_overlap(Sup)
    pr = 1 / ((Sup / Sup.sum(1, keepdims=True)) ** 2).sum(1)
    ax_c.text(0.5, -0.30, f'{pr.mean():.0f} of {LAYER_SIZES[0]} units per circuit; '
              f'pairwise overlap {ov:.2f} (shuffled {null.mean():.2f})',
              transform=ax_c.transAxes, ha='center', va='top', fontsize=6,
              color='0.35')
    anchors['c'] = ax_c

    # ── (d) informativity spectra of every node of the trace ─────────────────
    gsd = gs[3].subgridspec(1, 1 + n_c)
    ax = fig.add_subplot(gsd[0, 0])
    lam = D['root_lam']
    ax.bar(range(len(lam)), lam, color=C_BFT, edgecolor='0.25', linewidth=0.3,
           width=0.75)
    ax.set_title('$L_3$ (output)', pad=1.5, fontsize=7)
    ax.set_ylabel(r'$\lambda$ share', labelpad=1)
    anchors['d'] = ax
    axes_d = [ax]
    for j, c in enumerate(CIRCUITS):
        axj = fig.add_subplot(gsd[0, j + 1], sharey=ax)
        lam = c['l1_lam']
        axj.bar(range(len(lam)), lam, color=tint(C_BFT, 0.45), edgecolor='0.25',
                linewidth=0.3, width=0.75)
        axj.set_title(rf"$L_1$ of $f_{c['k']}$", pad=1.5, fontsize=7)
        axes_d.append(axj)
    for i, axj in enumerate(axes_d):
        axj.set_ylim(0, 0.65)
        axj.set_yticks([0, 0.5])
        axj.set_yticklabels(['0', '.5'] if i == 0 else [])
        axj.set_xticks([0, 5] if i else [0, 3, 6])
        axj.tick_params(length=1.5, pad=1)
        if i == len(axes_d) // 2:
            axj.set_xlabel('factor $f$', labelpad=1)
        for s in ('top', 'right'):
            axj.spines[s].set_visible(False)

    # ── (e) the circuit of every output factor, through the full network ─────
    gse = gs[5].subgridspec(1, n_c)
    for j, c in enumerate(CIRCUITS):
        ax = fig.add_subplot(gse[0, j])
        sc = c['scaffold']
        draw_scaffold_backbone(ax, sc['edges'], sc['neg_edges'], sc['loading'],
                               list(sc['layer_sizes']), C_BFT, C_INH)
        top = np.argsort(-c['profile'])[:N_SHOW]
        ax.set_title(rf"$f_{c['k']}$: " + ','.join(str(int(d)) for d in top),
                     pad=1.5, fontsize=7)
        if j == 0:
            anchors['e'] = ax

    # ── (f) the un-pooled sub-circuits themselves, in pixel space ────────────
    gsf = gs[7].subgridspec(N_SHOW, n_c + 1, width_ratios=[0.30] + [1] * n_c)
    for r in range(N_SHOW):
        row_label(fig, gsf[r, 0], f'{["1st", "2nd", "3rd"][r]}\npooled\ndigit')
    arbor_axes = []
    for j, c in enumerate(CIRCUITS):
        P = c['l1_profiles']
        top = np.argsort(-c['profile'])[:N_SHOW]      # the digits this factor pools
        for r, d in enumerate(top):
            k = int(P[:, d].argmax())                 # its best detector for that digit
            ax = fig.add_subplot(gsf[r, j + 1])
            M = c['l1_arbors'][k]
            ax.imshow(M, cmap=CMAP, interpolation='nearest',
                      norm=matplotlib.colors.PowerNorm(0.62, vmin=0,
                                                       vmax=np.percentile(M, 99.3)))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            ax.text(0.05, 0.97, rf'$f_{{{k}}}$', transform=ax.transAxes, ha='left',
                    va='top', fontsize=6.5, color=C_BFT)
            ax.text(0.95, 0.97, f'{int(d)}', transform=ax.transAxes, ha='right',
                    va='top', fontsize=7, color='0.15', fontweight='bold')
            if r == 0:
                ax.set_title(r'$f_{' + str(c['k']) + r'}$ pools ' +
                             ','.join(str(int(t)) for t in top), pad=2, fontsize=7)
                if j == 0:
                    arbor_axes.append(ax)

    figstyle.freeze(fig)

    def _label(key, spacer, text, dx=0.0):
        fig.text(max(anchors[key].get_position().x0 + dx, 0.002),
                 spacers[spacer].get_position().y0, text, ha='left', va='bottom',
                 fontsize=7, fontweight='bold')

    _label('a', 0, '(a) output-layer factors', dx=-0.030)
    _label('b', 0, '(b) un-pooling', dx=-0.030)
    _label('c', 0, '(c) $L_1$ units per circuit', dx=-0.030)
    _label('d', 1, '(d) informativity spectra', dx=-0.030)
    _label('e', 2, '(e) traced circuit of each output factor '
                   '(40 $L_1$ / 20 $L_2$ / 10 output units; '
                   'each unit\'s strongest input)', dx=0.0)
    fig.text(max(arbor_axes[0].get_position().x0 - 0.036, 0.002),
             spacers[3].get_position().y0,
             r'(f) the $L_1$ factor that detects each pooled digit '
             r'(bold: the digit; $f$: which factor of the circuit)',
             ha='left', va='bottom', fontsize=7)
    return fig


# ── Figure 6 / Appendix E — the CNN on CIFAR-10, read through its stimuli ─────

CNN_LAYER_LABEL = {'classifier': 'classifier', 'features.12': 'conv4',
                   'features.8': 'conv3', 'features.4': 'conv2',
                   'features.0': 'conv1'}

# depth labels L_1 (pixels) .. L_5 (classifier), matching Fig. 6's convention
CNN_LI_LABEL = {'classifier': r'$L_5$', 'features.12': r'$L_4$',
                'features.8': r'$L_3$', 'features.4': r'$L_2$',
                'features.0': r'$L_1$'}


def cifar_rgb(D, x):
    """One (3, H, W) normalized tensor -> (H, W, 3) in [0, 1]."""
    return np.clip(x * D['image_std'][:, None, None] + D['image_mean'][:, None, None],
                   0, 1).transpose(1, 2, 0)


def montage(D, imgs, nrow, ncol, pad=1, bg=1.0):
    """Tile the first nrow*ncol images of (T, 3, H, W) into one RGB array."""
    h, w = imgs.shape[-2:]
    out = np.full((nrow * h + (nrow - 1) * pad, ncol * w + (ncol - 1) * pad, 3), bg,
                  dtype=np.float32)
    for i in range(min(nrow * ncol, len(imgs))):
        r, c = divmod(i, ncol)
        out[r * (h + pad):r * (h + pad) + h, c * (w + pad):c * (w + pad) + w] = \
            cifar_rgb(D, imgs[i])
    return out


def stim_panel(fig, spec, D, imgs, nrow, ncol, *, ec='0.6', lw=0.4):
    ax = fig.add_subplot(spec)
    ax.imshow(montage(D, imgs, nrow, ncol), interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(ec); s.set_linewidth(lw)
    return ax


def rgb_strip(ax, shares, h=0.055):
    """Thin R/G/B bar across the bottom of an image axes: how much of the
    factor's input arbor sits on each color channel."""
    x = 0.0
    for frac, col in zip(shares / shares.sum(), ('#D62728', '#2CA02C', '#1F77B4')):
        ax.add_patch(matplotlib.patches.Rectangle(
            (x, 0.0), frac, h, transform=ax.transAxes, facecolor=col,
            edgecolor='none', zorder=5, clip_on=False))
        x += frac


def color_arbor_recurrence(NODES, layer='features.0'):
    """Do the conv1 nodes of different circuits find the same color factors?
    Greedy-match each pair's factors by RGB profile and return the mean cosine."""
    P = []
    for n in NODES:
        if n['layer_name'] == layer and 'conn' in n:
            M = n['conn']['in_mass']
            P.append(M / (M.sum(1, keepdims=True) + 1e-12))
    vals = []
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            A, B = unit(P[a]), unit(P[b])
            S = A @ B.T
            free = list(range(S.shape[1]))
            for i in np.argsort(-S.max(1)):
                j = free[int(np.argmax(S[i, free]))]
                vals.append(float(S[i, j])); free.remove(j)
    return float(np.mean(vals))


def hcat(tiles, gaps, bg=1.0):
    """Lay (h, w, 3) tiles of equal height side by side, with per-gap spacing."""
    h = tiles[0].shape[0]
    W = sum(t.shape[1] for t in tiles) + int(sum(gaps))
    out = np.full((h, W, 3), bg, np.float32)
    x = 0
    for i, t in enumerate(tiles):
        out[:, x:x + t.shape[1]] = t
        x += t.shape[1] + (int(gaps[i]) if i < len(gaps) else 0)
    return out


def vcat(rows, gap, bg=1.0):
    w = max(r.shape[1] for r in rows)
    H = sum(r.shape[0] for r in rows) + gap * (len(rows) - 1)
    out = np.full((H, w, 3), bg, np.float32)
    y = 0
    for i, r in enumerate(rows):
        out[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + gap
    return out


def strip_axes(fig, spec, comp, *, top=0, bottom=0):
    """One axes holding a composite image, with room reserved for text above
    and below it — keeps image blocks out of nested aspect-locked gridspecs."""
    ax = fig.add_subplot(spec)
    ax.imshow(comp, interpolation='nearest')
    ax.set_xlim(-0.5, comp.shape[1] - 0.5)
    ax.set_ylim(comp.shape[0] - 0.5 + bottom, -0.5 - top)
    ax.set_axis_off()
    return ax


def cnn_layers(NODES):
    """(layer_name, [nodes]) in trace order: output first, input last."""
    order, groups = [], {}
    for n in NODES:
        if n['layer_name'] not in groups:
            order.append(n['layer_name']); groups[n['layer_name']] = []
        groups[n['layer_name']].append(n)
    return [(name, groups[name]) for name in order]


def rgb_spread(D, imgs):
    """Mean pairwise distance between the mean colors of a set of stimuli."""
    c = np.stack([cifar_rgb(D, im).mean((0, 1)) for im in imgs])
    d = np.linalg.norm(c[:, None] - c[None, :], axis=-1)
    return d[np.triu_indices(len(c), 1)].mean()


def cnn_depth_stats(D, seed=0):
    """Per layer: class purity of every factor and color spread of its top set,
    plus the matching chance / random-set references."""
    rows = []
    for name, nodes in cnn_layers(D['nodes']):
        pur = np.concatenate([n['class_profile'].max(1) for n in nodes])
        lam = np.concatenate([n['lam_share'] for n in nodes])
        spread = np.array([rgb_spread(D, n['top_images'][k])
                           for n in nodes for k in range(n['n_factors'])])
        rows.append(dict(layer=name, purity=pur, lam=lam, spread=spread))
    pool = np.concatenate([n['top_images'].reshape(-1, *n['top_images'].shape[-3:])
                           for n in D['nodes']])
    rng = np.random.default_rng(seed)
    n_top = D['nodes'][0]['top_images'].shape[1]
    rand = np.mean([rgb_spread(D, pool[rng.choice(len(pool), n_top, replace=False)])
                    for _ in range(300)])
    return rows, float(rand)


def sibling_overlap(NODES, layer):
    """Mean Jaccard overlap of the top-stimulus sets of sibling factors."""
    vals = []
    for n in NODES:
        if n['layer_name'] != layer:
            continue
        T = n['top_idx']
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                a, b = set(T[i].tolist()), set(T[j].tolist())
                vals.append(len(a & b) / len(a | b))
    return float(np.mean(vals)) if vals else float('nan')


def lam_weighted(v, w):
    return float((v * w).sum() / w.sum())


def purity_ci_by_layer(D, labels, n_boot=800, seed=0):
    """Per layer (output first): (point, lo, hi) of the lambda-weighted class
    purity with a 95% stimulus-bootstrap CI. Reconstructs each factor's class
    profile from per-stimulus loadings + labels so the resample is honest; labels
    come from the aligned fingerprint bundle. Numpy only (Appendix stats)."""
    labels = np.asarray(labels); classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    out = []
    for _name, nodes in cnn_layers(D['nodes']):
        packs = []
        for nd in nodes:
            H = np.asarray(nd['img_factors'], float)
            lam = np.asarray(nd['lam_share'], float)
            lab = (labels[np.asarray(nd['stim_idx'])]
                   if 'stim_idx' in nd and H.shape[0] != len(labels) else labels)
            packs.append((H, lam, lab))
        joint = len({p[0].shape[0] for p in packs}) == 1

        def wp(ridx):
            purs, lams = [], []
            for H, lam, lab in packs:
                if ridx is None:
                    Hs, labs = H, lab
                else:
                    r = ridx if joint else rng.integers(0, len(lab), len(lab))
                    Hs, labs = H[r], lab[r]
                prof = np.zeros((H.shape[1], len(classes)))
                for j, c in enumerate(classes):
                    m = labs == c
                    if m.any():
                        prof[:, j] = Hs[m].mean(0)
                prof = prof / (prof.sum(1, keepdims=True) + 1e-12)
                purs.append(prof.max(1)); lams.append(lam)
            return lam_weighted(np.concatenate(purs), np.concatenate(lams))

        point = wp(None)
        n0 = packs[0][0].shape[0]
        vals = np.array([wp(rng.integers(0, n0, n0) if joint else True)
                         for _ in range(n_boot)])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out.append((point, float(lo), float(hi)))
    return out


def fig6_cnn_circuits(D):
    NODES, CLS = D['nodes'], list(D['class_names'])
    by_path = {tuple(n['path'].tolist()): n for n in NODES}
    root = NODES[0]
    C_BFT, C_X = figstyle.color('ours'), figstyle.color('cross_class')
    C_BASE = figstyle.color('baseline')
    n_root = root['n_factors']
    CLASS_COLOR = class_colors(range(len(CLS)))

    SHOW = [0, 1, 9]           # trees in (b): car (automobile), horse, airplane
    # which conv4 sub-factors to open per circuit: the horse's f0/f1 are near
    # duplicates, so f1/f2 make the split between the two groups legible.
    SUB = {0: [0, 1], 1: [1, 2], 9: [0, 1]}
    LEAF = (9, 0, 0, 0)        # the conv1 node panel (c) opens
    N_TREE = 2                 # two conv4 sub-factors per circuit

    # row heights in inches, derived from the image grids they hold
    W = 6.975
    s_a = W / n_root                                   # (a) montage side, square
    ha = s_a + 0.30 + 0.13                             # montage + class bars + label
    # wider side/inter-tree gaps hold the sub-factor montages to a size closer to
    # the (a) and (c) montages (they were the largest images in the figure), which
    # trims the row's height; tighter label rows save a little more.
    PAD_B, GAP_B, GAP_IN = 0.24, 0.75, 0.14            # side / inter-tree / intra-pair
    NODE_H, FLAB_H = 0.24, 0.13                        # node-label / f-label rows
    s_b = W / (N_TREE * len(SHOW) + (len(SHOW) - 1) * GAP_B
               + len(SHOW) * GAP_IN + 2 * PAD_B)
    hb = NODE_H + s_b + FLAB_H                          # node label + image + f label
    n_leaf = by_path[LEAF]['n_factors']
    s_c = 0.60                                         # (c) conv1 montage side
    hbot = s_c + 0.17                                  # montage + purity number
    sp = 0.13                                          # spacer rows
    rows = [sp, ha, sp, hb, sp, hbot]
    total = sum(rows)

    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='paper',
                   height_to_width_ratio=total / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.008, w_pad=0.008,
                          hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(6, 1, height_ratios=rows, hspace=0.0)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4)]
    for s_ in spacers:
        s_.set_axis_off()
    anchors = {}

    def tag(ax, text, x=0.05, y=0.97, size=6.5, color='white'):
        ax.text(x, y, text, transform=ax.transAxes, ha='left', va='top',
                fontsize=size, color=color,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2,
                                                                foreground='0.15')])

    # ── (a) output factors: stimuli and the full class distribution ──────────
    gsa = gs[1].subgridspec(3, n_root, height_ratios=[s_a, 0.30, 0.13])
    pmax = root['class_profile'].max()                  # common bar scale (honest)
    for k in range(n_root):
        ax = stim_panel(fig, gsa[0, k], D, root['top_images'][k], 2, 2)
        prof = root['class_profile'][k]
        c = int(np.argmax(prof))
        axb = fig.add_subplot(gsa[1, k])                # class distribution, 10 bars
        axb.bar(range(len(CLS)), prof, width=0.9, edgecolor='none',
                color=[CLASS_COLOR[i] for i in range(len(CLS))])
        axb.set_xlim(-0.6, len(CLS) - 0.4); axb.set_ylim(0, pmax * 1.05)
        axb.set_xticks([]); axb.set_yticks([])
        for s_ in axb.spines.values():
            s_.set_visible(False)
        # f label rides below the stimuli (on the bar chart), not over a
        # photograph where it is unreadable — matching Fig. 8(a)
        axb.text(0.0, 1.02, rf'$f_{{{k}}}$', transform=axb.transAxes, ha='left',
                 va='top', fontsize=6.5, color=C_BFT,
                 bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.4))
        lab = fig.add_subplot(gsa[2, k]); lab.set_axis_off()
        lab.text(0.5, 1.0, f'{CLS[c]} {prof[c]:.2f}', ha='center', va='top',
                 fontsize=6, color='0.2')
        if k == 0:
            anchors['a'] = ax

    # ── (b) traceback: each output factor splits into conv4 sub-factors ──────
    # columns per circuit: two image cells with a small gap between them (so the
    # two sub-factors read as distinct), circuits separated by a wider gap.
    widths, col_of = [PAD_B], {}
    for j_ in range(len(SHOW)):
        col_of[j_] = len(widths)
        widths += [1.0, GAP_IN, 1.0]
        widths += [GAP_B] if j_ < len(SHOW) - 1 else [PAD_B]
    gsb = gs[3].subgridspec(3, len(widths), height_ratios=[NODE_H, s_b, FLAB_H],
                            width_ratios=widths)
    tree = []                                           # (node_ax, [sub_ax, ...])
    for j_, r in enumerate(SHOW):
        node = by_path[(r,)]
        own = int(np.argmax(root['class_profile'][r]))
        base = col_of[j_]
        node_ax = fig.add_subplot(gsb[0, base:base + 3]); node_ax.set_axis_off()
        node_ax.text(0.5, 0.32, rf'$f_{{{r}}}$ · {CLS[own]}', ha='center', va='center',
                     fontsize=6.5, color='white',
                     bbox=dict(boxstyle='round,pad=0.28', fc=C_BFT, ec='none'))
        subs = []
        for slot, k in enumerate(SUB[r]):               # slot 0/1 -> columns base, base+2
            col = base + 2 * slot
            prof = node['class_profile'][k]
            c = int(np.argmax(prof))
            off = c != own and prof[c] - prof[own] > 0.10
            ax = stim_panel(fig, gsb[1, col], D, node['top_images'][k], 2, 2,
                            ec=C_X if off else '0.6', lw=1.1 if off else 0.4)
            if off:                                     # flag only a cross-class group
                tag(ax, CLS[c][:5], x=0.5, y=0.04, size=6, color=C_X)
                ax.texts[-1].set(ha='center', va='bottom')
            flab = fig.add_subplot(gsb[2, col]); flab.set_axis_off()
            flab.text(0.5, 0.92, rf'$f_{{{k}}}$', ha='center', va='top', fontsize=6.5,
                      color=C_BFT)
            subs.append(ax)
        tree.append((node_ax, subs))
        if j_ == 0:
            anchors['b'] = node_ax

    # ── (c) at conv1 the groups are appearance, not class ────────────────────
    w_de = (W - n_leaf * s_c) / 2                        # width left for (d) and (e)
    gsc = gs[5].subgridspec(1, 3, width_ratios=[n_leaf * s_c, w_de, w_de])
    leaf = by_path[LEAF]
    gscl = gsc[0, 0].subgridspec(2, n_leaf, height_ratios=[s_c, 0.17])
    for k in range(n_leaf):
        ax = stim_panel(fig, gscl[0, k], D, leaf['top_images'][k], 2, 2)
        rgb_strip(ax, leaf['conn']['in_mass'][k])
        tag(ax, rf'$f_{{{k}}}$')
        lab = fig.add_subplot(gscl[1, k]); lab.set_axis_off()
        lab.text(0.5, 1.0, f'{leaf["class_profile"][k].max():.2f}', ha='center',
                 va='top', fontsize=6, color='0.2')
        if k == 0:
            anchors['c'] = ax

    # ── (d) class purity falls toward the input; (e) color grows coherent ────
    stats, rand = cnn_depth_stats(D)
    x = np.arange(len(stats))
    chance = 1.0 / D['n_classes']
    # generic depth labels L_1 (nearest the pixels) .. L_N (output), matching the
    # convention of Fig. 2; stats run output-first, so the ticks descend
    n_lay = len(stats)
    xt = [rf'$L_{{{n_lay - i}}}$' for i in range(n_lay)]

    ax_d = fig.add_subplot(gsc[0, 1])
    pur = [lam_weighted(st['purity'], st['lam']) for st in stats]
    ax_d.axhline(chance, color='0.65', lw=0.6, ls=(0, (3, 2)), zorder=0)
    for i_, st in enumerate(stats):
        ax_d.scatter(np.full(len(st['purity']), i_), st['purity'], s=2.0, color=C_BFT,
                     alpha=0.28, edgecolor='none', zorder=2)
    # 95% stimulus-bootstrap CI on the lambda-weighted purity (Appendix stats)
    from src import figdata as _fd
    _lab = np.asarray(_fd.load('nb03_fingerprints')['fp']['id_targets'])
    _ci = purity_ci_by_layer(D, _lab)
    _yerr = np.array([[max(p - lo, 0), max(hi - p, 0)] for p, lo, hi in _ci]).T
    ax_d.errorbar(x, pur, yerr=_yerr, fmt='none', ecolor='0.25', elinewidth=0.7,
                  capsize=1.6, capthick=0.7, zorder=4)
    ax_d.plot(x, pur, color=C_BFT, marker='o', ms=3.0, lw=1.4, zorder=3)
    ax_d.set_ylim(0, 1.0); ax_d.set_yticks([0, 0.5, 1.0])
    ax_d.set_yticklabels(['0', '.5', '1'])
    ax_d.set_ylabel('class purity', color=C_BFT, labelpad=1, fontsize=6)
    ax_d.text(len(stats) - 0.9, chance + 0.03, 'chance', fontsize=6, color='0.45',
              ha='right', va='bottom')
    ax_d.set_xlim(-0.35, len(stats) - 0.65); ax_d.set_xticks(x)
    ax_d.set_xticklabels(xt, fontsize=6)
    ax_d.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_d.spines[s_].set_visible(False)
    anchors['d'] = ax_d

    ax_e = fig.add_subplot(gsc[0, 2])
    spread = [st['spread'].mean() for st in stats]
    ax_e.axhline(rand, color=tint(C_BASE, 0.45), lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax_e.plot(x, spread, color=C_BASE, marker='s', ms=2.6, lw=1.4, zorder=3)
    ax_e.set_ylim(0, 0.34); ax_e.set_yticks([0, 0.1, 0.2, 0.3])
    ax_e.set_yticklabels(['0', '', '.2', ''])
    ax_e.set_ylabel('color spread', color=C_BASE, labelpad=1, fontsize=6)
    ax_e.text(len(stats) - 0.1, rand - 0.012, 'random', fontsize=6,
              color=tint(C_BASE, 0.3), ha='right', va='top')
    ax_e.set_xlim(-0.35, len(stats) - 0.65); ax_e.set_xticks(x)
    ax_e.set_xticklabels(xt, fontsize=6)
    ax_e.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_e.spines[s_].set_visible(False)
    anchors['e'] = ax_e

    figstyle.freeze(fig)

    # tree connectors: from each output-factor node down to its two sub-factors
    for node_ax, subs in tree:
        nb = node_ax.get_position()
        x0, y0 = (nb.x0 + nb.x1) / 2, nb.y0 + 0.10 * nb.height
        for sub in subs:
            sb = sub.get_position()
            fig.add_artist(matplotlib.lines.Line2D(
                [x0, (sb.x0 + sb.x1) / 2], [y0, sb.y1], color='0.6', lw=0.5,
                zorder=0))

    def _label(key, spacer, text, dx=0.0, **kw):
        fig.text(max(anchors[key].get_position().x0 + dx, 0.002),
                 spacers[spacer].get_position().y0 + 0.004, text, ha='left',
                 va='bottom', fontsize=7, fontweight='bold', **kw)

    _label('a', 0, '(a) output factors, with class distribution', dx=-0.004)
    _label('b', 1, r'(b) traceback to $L_4$', dx=-0.010)
    _label('c', 2, r'(c) $L_1$ factors of $f_9$ (class purity below)', dx=-0.004)
    _label('d', 2, '(d) class purity', dx=-0.030)
    _label('e', 2, '(e) color spread', dx=-0.030)
    return fig


def figE_cnn_details(D):
    NODES, CLS = D['nodes'], list(D['class_names'])
    by_path = {tuple(n['path'].tolist()): n for n in NODES}
    root = NODES[0]
    C_BFT, C_X = figstyle.color('ours'), figstyle.color('cross_class')
    n_root = root['n_factors']
    LAYERS = cnn_layers(NODES)
    CONV4 = [by_path[(r,)] for r in range(n_root)]
    CASCADE = [(9,), (9, 0), (9, 0, 0), (9, 0, 0, 0)]      # the traced spine of f9
    chance = 1.0 / D['n_classes']

    # the gallery shows six circuits at four sub-factors each; the four animal
    # circuits below (cat/deer/dog/frog) are dropped to make room
    SKIP = {'cat', 'deer', 'dog', 'frog'}
    GAL = [r for r in range(n_root)
           if CLS[int(np.argmax(root['class_profile'][r]))] not in SKIP]

    N_SHOW = 4                     # (b): sub-factors per circuit
    EX_R, EX_C = 2, 3              # example stimuli per sub-factor (6 of the 8 stored)
    PER_ROW = 1                    # circuits per gallery row (four sub-factors is wide)
    LAB_W, AVG_W = 0.55, 1.0       # circuit-label / weighted-average column widths
    EX_W = EX_C / EX_R             # keep every example tile square
    SUBGAP, CIRCGAP, FLAB = 0.14, 0.42, 0.20   # gaps, and the f-label row height

    # row heights in inches, derived from the image grids they hold
    W, SP = 6.975, 0.15
    n_gal_rows = -(-len(GAL) // PER_ROW)
    # columns of one circuit: label, then [weighted-avg, examples] per sub-factor
    circ_cols = [LAB_W]
    for _s in range(N_SHOW):
        circ_cols += [AVG_W, EX_W] + ([SUBGAP] if _s < N_SHOW - 1 else [])
    col_w, col_base = [], []
    for _b in range(PER_ROW):
        col_base.append(len(col_w))
        col_w += circ_cols + ([CIRCGAP] if _b < PER_ROW - 1 else [])
    s_g = W / sum(col_w)                                    # montage side
    h_b = n_gal_rows * (s_g + FLAB) + 0.04
    n_spine = 2 + sum(by_path[p_]['n_factors'] for p_ in CASCADE[1:])
    h_c = W * (65.0 + 33.0) / (n_spine * 65.0 + (n_spine - 5) * 6.0 + 4 * 24.0)
    rows = [SP, 1.05, SP, h_b, SP, h_c]

    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='appendix',
                   height_to_width_ratio=sum(rows) / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.008, w_pad=0.008,
                          hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(6, 1, height_ratios=rows, hspace=0.0)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4)]
    for s_ in spacers:
        s_.set_axis_off()
    anchors = {}

    def tag(ax, text, x=0.05, y=0.97, size=6, color='white', ha='left', va='top'):
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=size, color=color,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2,
                                                                foreground='0.15')])

    # ── (a) informativity spectrum of every node, by layer ───────────────────
    gsa = gs[1].subgridspec(1, len(LAYERS))
    for i, (name, nodes) in enumerate(LAYERS):
        ax = fig.add_subplot(gsa[0, i])
        kmax = max(n['n_factors'] for n in nodes)
        L = np.full((len(nodes), kmax), np.nan)
        for j_, n in enumerate(nodes):
            L[j_, :n['n_factors']] = n['lam_share']
        x = np.arange(kmax)
        ax.bar(x, np.nanmean(L, 0), color=tint(C_BFT, 0.45), edgecolor='0.25',
               linewidth=0.3, width=0.75, zorder=1)
        if len(nodes) > 1:
            for row in L:
                ax.scatter(x, row, s=1.6, color=C_BFT, zorder=2, edgecolor='none')
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0', '.5', '1'] if i == 0 else [])
        ax.set_xticks(x)
        step = 2 if kmax > 6 else 1
        ax.set_xticklabels([str(v) if v % step == 0 else '' for v in x], fontsize=6)
        ax.tick_params(length=1.5, pad=1)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.set_title(CNN_LI_LABEL[name], pad=1.5, fontsize=6.5)
        if i == 0:
            ax.set_ylabel(r'$\lambda$ share', labelpad=1)
            anchors['a'] = ax
        if i == len(LAYERS) // 2:
            ax.set_xlabel('factor $f$', labelpad=1)

    # ── (b) gallery: each output circuit's strongest L4 sub-factors, shown as
    #        the weighted-average stimulus (framed, the factor's prototype)
    #        beside real top stimuli; all factor labels sit below the images ────
    row_h = []
    for _ in range(n_gal_rows):
        row_h += [s_g, FLAB]
    gsb = gs[3].subgridspec(2 * n_gal_rows, len(col_w), width_ratios=col_w,
                            height_ratios=row_h)
    for idx, r in enumerate(GAL):
        grow, blk = idx // PER_ROW, idx % PER_ROW
        irow, lrow = 2 * grow, 2 * grow + 1
        base = col_base[blk]
        own = int(np.argmax(root['class_profile'][r]))
        lab = fig.add_subplot(gsb[irow, base]); lab.set_axis_off()
        lab.text(1.0, 0.5, rf'$f_{{{r}}}$' + '\n' + CLS[own][:6], ha='right',
                 va='center', fontsize=6, linespacing=1.2)
        if idx == 0:
            anchors['b'] = lab
        node = CONV4[r]
        for k in range(N_SHOW):
            ac, ec = base + 1 + 3 * k, base + 2 + 3 * k        # avg / examples cols
            prof = node['class_profile'][k]
            c = int(np.argmax(prof))
            off = c != own and prof[c] - prof[own] > 0.10
            axa = fig.add_subplot(gsb[irow, ac])               # weighted-avg stimulus
            axa.imshow(cifar_rgb(D, node['wavg'][k]), interpolation='nearest')
            axa.set_xticks([]); axa.set_yticks([])
            for s_ in axa.spines.values():
                s_.set_color(C_BFT); s_.set_linewidth(1.0)
            stim_panel(fig, gsb[irow, ec], D, node['top_images'][k], EX_R, EX_C,
                       ec=C_X if off else '0.6', lw=1.1 if off else 0.4)
            la = fig.add_subplot(gsb[lrow, ac]); la.set_axis_off()
            la.text(0.5, 0.95, 'avg', ha='center', va='top', fontsize=6,
                    color='0.45')
            le = fig.add_subplot(gsb[lrow, ec]); le.set_axis_off()
            le.text(0.5, 0.95, rf'$f_{{{k}}}$' + (f' → {CLS[c][:5]}' if off else ''),
                    ha='center', va='top', fontsize=6, color=C_X if off else '0.2')

    # ── (c) the traced spine of one circuit, root to conv1 ───────────────────
    spine = [root] + [by_path[p_] for p_ in CASCADE]
    n_show = [1, 1] + [n['n_factors'] for n in spine[2:]]   # root and conv4: the
    tiles, gaps, meta = [], [], []                          # traced factor only
    for i, n in enumerate(spine):
        for k in range(n_show[i]):
            kk = 9 if i == 0 else k                    # the root column is f9 itself
            tiles.append(montage(D, n['top_images'][kk], 2, 2))
            gaps.append(24 if k == n_show[i] - 1 else 6)
            meta.append((i, n, kk))
    comp_c = hcat(tiles, gaps[:-1])
    side = tiles[0].shape[0]
    ax_c = strip_axes(fig, gs[5], comp_c, top=15, bottom=18)
    x = 0
    for j_, (i, n, kk) in enumerate(meta):
        ax_c.text(x + 2, 2, rf'$f_9$' if i == 0 else rf'$f_{{{kk}}}$', ha='left',
                  va='top', fontsize=6, color='white',
                  path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2,
                                                                  foreground='0.15')])
        ax_c.text(x + side / 2, side + 3, f"{n['class_profile'][kk].max():.2f}",
                  ha='center', va='top', fontsize=6, color='0.2')
        if 'conn' in n and n['conn']['in_mass'].shape[1] == 3:
            sh = n['conn']['in_mass'][kk]
            sh = sh / sh.sum()
            xs = x
            for frac, colr in zip(sh, ('#D62728', '#2CA02C', '#1F77B4')):
                ax_c.add_patch(matplotlib.patches.Rectangle(
                    (xs, side - 4), frac * side, 4, facecolor=colr, edgecolor='none',
                    zorder=5))
                xs += frac * side
        if j_ == 0 or meta[j_ - 1][0] != i:
            first_x = x
        if j_ == len(meta) - 1 or meta[j_ + 1][0] != i:
            name = r'$L_5$' if i == 0 else CNN_LI_LABEL[n['layer_name']]
            ax_c.text((first_x + x + side) / 2, -3, name, ha='center', va='bottom',
                      fontsize=6.5, color='0.35')
        x += side + gaps[j_]
    anchors['c'] = ax_c

    figstyle.freeze(fig)

    def _label(key, spacer, text, dx=0.0):
        fig.text(max(anchors[key].get_position().x0 + dx, 0.002),
                 spacers[spacer].get_position().y0 + 0.002, text, ha='left',
                 va='bottom', fontsize=7, fontweight='bold')

    _label('a', 0, r'(a) $\lambda$ spectra by layer', dx=-0.026)
    _label('b', 1, r'(b) $L_4$ gallery: each output circuit $f_r$ traced back to its '
                   r'four strongest $L_4$ sub-factors $f_k$ — weighted-avg. stimulus '
                   r'(framed) then top stimuli', dx=-0.004)
    _label('c', 2, r'(c) traced spine of $f_9$, $L_5\to L_1$ — class purity below '
                   f'each factor (chance {chance:.2f})', dx=-0.004)
    return fig


def _cnn_fp_blocks(D):
    """Fingerprint columns grouped into the ten output circuits, the circuits
    ordered by the class whose mean fingerprint loads them most — so the block
    strip of a class-by-circuit panel is diagonal iff every class picks its own
    circuit. Returns (column order, block edges, block starts, block sizes)."""
    co = list(D['col_order'])
    starts = np.concatenate([[0], list(D['blk_edge'])])
    sizes = list(D['block_sizes'])
    blocks = [co[starts[b]:starts[b] + sizes[b]] for b in range(len(sizes))]
    M = D['fp_mean_by_class']
    M = M / M.sum(1, keepdims=True)
    B = np.stack([M[:, b].sum(1) for b in blocks], axis=1)     # class x circuit
    order = B.argmax(1)                                        # circuit of each class
    if len(set(order.tolist())) == len(order):                 # 1:1, so reorder
        blocks = [blocks[i] for i in order]
        sizes = [sizes[i] for i in order]
    edges = list(np.cumsum(sizes)[:-1])
    return [c for b in blocks for c in b], edges, \
        np.concatenate([[0], edges]), sizes


def cnn_fp_heatmap(ax_a, D, COL_ORDER, BLK_EDGE, CMAP, CCOL, CLS, N_CLASSES,
                   n_factors):
    """Mean fingerprint per class, columns grouped by output circuit and each
    scaled to its top class. Returns the axes; its y-tick colors are the class
    legend the embedding panels reuse."""
    M = D['fp_mean_by_class']
    M = M / M.sum(1, keepdims=True)
    Mc = M[:, COL_ORDER]
    Mc = Mc / (Mc.max(0, keepdims=True) + 1e-12)
    ax_a.imshow(Mc, cmap=CMAP, aspect='auto', interpolation='nearest',
                norm=matplotlib.colors.PowerNorm(0.8, vmin=0, vmax=1))
    for e in BLK_EDGE:
        ax_a.axvline(e - 0.5, color='0.25', lw=0.7)
    ax_a.set_xticks([])
    ax_a.set_yticks(range(N_CLASSES)); ax_a.set_yticklabels(CLS, fontsize=6)
    for t_, c_ in zip(ax_a.get_yticklabels(), range(N_CLASSES)):
        t_.set_color(CCOL[c_])
    ax_a.set_xlabel(f'{n_factors} factors', labelpad=2)
    ax_a.tick_params(length=1.5, pad=1)
    for s_ in ax_a.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    return ax_a


def cnn_class_geometry(ax, D, CMAP, CCOL, CLS, N_CLASSES, show_between=False):
    """Pairwise cosine of the test fingerprints, sorted by class — the block
    diagonal behind the class silhouette. With `show_between` the panel also
    reports the mean between-class cosine, and both numbers move to the top
    right (where the matrix is emptiest)."""
    fp = D['fp']
    order = np.argsort(fp['id_targets'], kind='stable')
    U = unit(fp['id'][order])
    S = U @ U.T
    ax.imshow(S, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
    n_per = np.bincount(fp['id_targets'])
    cuts = np.cumsum(n_per)
    for c_ in cuts[:-1]:
        ax.axhline(c_ - 0.5, color='white', lw=0.4)
        ax.axvline(c_ - 0.5, color='white', lw=0.4)
    ax.set_xticks([]); ax.set_yticks(cuts - n_per / 2)
    ax.set_yticklabels(CLS, fontsize=6)
    for t_, c_ in zip(ax.get_yticklabels(), range(N_CLASSES)):
        t_.set_color(CCOL[c_])
    ax.set_xlabel(f'{len(U)} test stimuli', labelpad=1)
    ax.tick_params(length=0, pad=1)
    for s_ in ax.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    starts = np.concatenate([[0], cuts[:-1]])
    within = np.mean([S[a:b, a:b].mean() for a, b in zip(starts, cuts)])
    if show_between:
        cls_of = np.repeat(np.arange(len(n_per)), n_per)   # class of each sorted row
        between = float(S[cls_of[:, None] != cls_of[None, :]].mean())
        txt = f'within-class $\\cos$ {within:.2f}\nbetween-class {between:.2f}'
        ax.text(0.97, 0.975, txt, transform=ax.transAxes, ha='right', va='top',
                fontsize=6, color='0.15', linespacing=1.25,
                bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))
    else:
        ax.text(0.97, 0.03, f'within-class $\\cos$ {within:.2f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=6,
                color='0.15', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))


# ── Appendix G — the whole BFT trace of the TinyViT ───────────────────────────

def _vit_stages(NODES):
    """Trace nodes grouped by layer, in backward (trace) order."""
    order, groups = [], {}
    for n in NODES:
        if n['layer_name'] not in groups:
            order.append(n['layer_name'])
            groups[n['layer_name']] = []
        groups[n['layer_name']].append(n)
    return [(name, groups[name]) for name in order]


def _parity_bars(ax, profile, digits, c_even, c_odd, *, xticks=True, ylab=None):
    """Per-digit loading of one factor, colored by the parity of the digit."""
    cols = [c_even if int(d) % 2 == 0 else c_odd for d in digits]
    ax.bar(np.arange(len(digits)), profile, color=cols, edgecolor='0.25',
           linewidth=0.25, width=0.85)
    ax.set_xlim(-0.7, len(digits) - 0.3)
    ax.set_ylim(0, max(profile.max() * 1.12, 1e-6))
    ax.set_xticks(np.arange(len(digits)))
    ax.set_xticklabels([str(int(d)) for d in digits] if xticks else [], fontsize=6)
    ax.set_yticks([])
    ax.tick_params(length=0, pad=0.6)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    if ylab:
        ax.set_ylabel(ylab, labelpad=1, fontsize=6)


def figG_vit_circuits(D):
    NODES, CIRC = D['nodes'], D['circuits']
    DIGITS = [int(d) for d in D['digits']]
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    C_BFT = figstyle.color('ours')
    CMAP = seq_cmap(C_BFT, 'bft')
    STAGES = _vit_stages(NODES)
    FFN1 = [n for n in NODES if n['layer_name'] == 'B0-FFN1']
    ATTN = [n for n in NODES if n['layer_type'] == 'attn']
    n_c = len(CIRC)
    side = int(np.sqrt(ATTN[0]['attn_mean'].shape[1] - 1))     # 4x4 patch grid

    anchors = {}
    figstyle.apply(venue='aaai2024', width='full', nrows=3, ncols=3, mode='appendix',
                   height_to_width_ratio=0.60)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    # thin spacer rows reserve room for the panel labels added after the freeze
    gs = fig.add_gridspec(6, 1, height_ratios=[0.10, 1.15, 0.11, 0.70, 0.11, 1.10],
                          hspace=0.03)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4)]
    for sp in spacers:
        sp.set_axis_off()
    gs_top = gs[1].subgridspec(1, 3, width_ratios=[1.55, 1.0, 1.0])
    gs_bot = gs[5].subgridspec(1, 2, width_ratios=[1.15, 1.0])

    # ── (a) informativity spectrum of every node, by layer of the block ──────
    gsa = gs_top[0, 0].subgridspec(1, len(STAGES))
    for i, (name, nodes) in enumerate(STAGES):
        ax = fig.add_subplot(gsa[0, i])
        L = np.stack([n['lam_share'] for n in nodes])
        x = np.arange(L.shape[1])
        ax.bar(x, L.mean(0), color=tint(C_BFT, 0.45), edgecolor='0.25',
               linewidth=0.3, width=0.75, zorder=1)
        if len(nodes) > 1:
            for row in L:
                ax.scatter(x, row, s=1.6, color=C_BFT, zorder=2, edgecolor='none')
        ax.set_ylim(0, 0.72)
        ax.set_yticks([0, 0.25, 0.5])
        ax.set_yticklabels(['0', '', '.5'] if i == 0 else [])
        ax.set_xticks(x); ax.set_xticklabels(x, fontsize=6)
        ax.tick_params(length=1.5, pad=1)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_title(f"{name}\n({len(nodes)} node{'s' if len(nodes) > 1 else ''})",
                     pad=1.5, fontsize=6.5, linespacing=1.1)
        ax.set_xlabel('factor $f$', labelpad=1)
        if i == 0:
            ax.set_ylabel(r'$\lambda$ share', labelpad=1)
            anchors['a'] = ax

    # ── (b) what each output-layer circuit responds to: digits and parity ────
    gsb = gs_top[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.30])
    ax_b = fig.add_subplot(gsb[0, 0])
    Mprof = np.stack([c['digit_profile'] for c in CIRC])
    ax_b.imshow(Mprof, cmap=CMAP, aspect='auto', vmin=0, vmax=Mprof.max(),
                interpolation='nearest')
    ax_b.set_xticks(range(len(DIGITS))); ax_b.set_xticklabels(DIGITS, fontsize=6)
    ax_b.set_yticks(range(n_c))
    ax_b.set_yticklabels([rf"$f_{{{int(c['k'])}}}$" for c in CIRC])
    ax_b.set_xlabel('digit', labelpad=1)
    ax_b.set_ylabel('output factor', labelpad=1)
    style_matrix_axes(ax_b)
    anchors['b'] = ax_b
    ax_bp = fig.add_subplot(gsb[0, 1])                 # parity share of each circuit
    for i, c in enumerate(CIRC):
        ev, od = float(c['profile'][0]), float(c['profile'][1])
        ax_bp.barh(i, ev, color=C_EVEN, edgecolor='none', height=0.7)
        ax_bp.barh(i, od, left=ev, color=C_ODD, edgecolor='none', height=0.7)
        ax_bp.text(0.5, i, f'{max(ev, od):.2f}', ha='center', va='center',
                   fontsize=6, color='white')
    ax_bp.set_xlim(0, 1); ax_bp.set_ylim(n_c - 0.5, -0.5)
    ax_bp.set_xticks([]); ax_bp.set_yticks([])
    for s in ax_bp.spines.values():
        s.set_visible(False)
    ax_bp.set_xlabel('parity', labelpad=1, fontsize=6)
    ax_bp.set_title('even / odd', pad=1.5, fontsize=6,
                    color='0.35')

    # ── (c) which FFN units each circuit recruits ────────────────────────────
    ax_c = fig.add_subplot(gs_top[0, 2])
    Use = NODES[0]['conn']['in_mass']                  # (n_circuits, ffn_dim)
    Un = Use / (Use.max(1, keepdims=True) + 1e-12)
    order_u = np.argsort(-Un.mean(0))
    ax_c.imshow(Un[:, order_u], cmap=CMAP, aspect='auto', vmin=0, vmax=1,
                interpolation='nearest')
    Uu = unit(Use)
    ov = (Uu @ Uu.T)[np.triu_indices(n_c, 1)].mean()
    ax_c.set_xticks([]); ax_c.set_yticks(range(n_c))
    ax_c.set_yticklabels([rf"$f_{{{int(c['k'])}}}$" for c in CIRC])
    ax_c.set_xlabel(f'{Use.shape[1]} FFN units, sorted by mean use', labelpad=2)
    style_matrix_axes(ax_c)
    ax_c.text(0.98, 0.06, f'mean overlap {ov:.2f}', transform=ax_c.transAxes,
              ha='right', va='bottom', fontsize=6, color='0.15',
              bbox=dict(fc='white', ec='none', alpha=0.8, pad=1.0))
    anchors['c'] = ax_c

    # ── (d) the FFN1 sub-factors of each circuit: driving input and digits ───
    n_k = FFN1[0]['digit_profile'].shape[0]
    widths, gap = [], 0.45
    for j in range(n_c):
        widths += [1.0] * n_k + ([gap] if j < n_c - 1 else [])
    gsd = gs[3].subgridspec(2, len(widths), width_ratios=widths,
                            height_ratios=[1.0, 0.78])
    img_first = {}
    for j, node in enumerate(FFN1):
        base = j * (n_k + 1)
        for k in range(n_k):
            ax = fig.add_subplot(gsd[0, base + k])
            ax.imshow(node['wavg'][k, 0], cmap='gray_r', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            ax.text(0.05, 0.97, rf'$f_{k}$', transform=ax.transAxes, ha='left',
                    va='top', color=C_BFT, fontsize=6.5)
            prof = node['digit_profile'][k]
            top = int(np.argmax(prof))
            ax.text(0.95, 0.97, str(top), transform=ax.transAxes, ha='right', va='top',
                    fontsize=7, fontweight='bold',
                    color=C_EVEN if top % 2 == 0 else C_ODD)
            if k == 0:
                img_first[j] = ax
            _parity_bars(fig.add_subplot(gsd[1, base + k]), prof, DIGITS,
                         C_EVEN, C_ODD, ylab='loading' if k == 0 else None)
        if j < n_c - 1:
            fig.add_subplot(gsd[:, base + n_k]).set_axis_off()
    anchors['d'] = img_first[0]

    # ── (e) CLS attention: shared across factors ─────────────────────────────
    A_all = np.stack([n['attn_mean'] for n in ATTN])          # (nodes, K, T)
    A_flat = A_all.reshape(-1, A_all.shape[-1])
    A_flat = A_flat / A_flat.sum(1, keepdims=True)
    Ua = unit(A_flat)
    cos_min = (Ua @ Ua.T).min()
    n_dev = len(ATTN)
    gse = gs_bot[0, 0].subgridspec(1, n_dev + 2, width_ratios=[1.0, 0.30] + [1] * n_dev)
    ax_e = fig.add_subplot(gse[0, 0])
    mean_patch = A_flat[:, 1:].mean(0)
    show_map(ax_e, mean_patch.reshape(side, side), CMAP, pct=100, gamma=1.0)
    ax_e.set_title('mean', pad=1.5, fontsize=6.5)
    anchors['e'] = ax_e
    fig.add_subplot(gse[:, 1]).set_axis_off()
    dev = A_flat[:, 1:] - mean_patch
    vmax = np.abs(dev).max()
    for i, node in enumerate(ATTN):
        ax = fig.add_subplot(gse[0, i + 2])
        ax.imshow(dev[i * node['attn_mean'].shape[0]].reshape(side, side), cmap='RdBu_r',
                  vmin=-vmax, vmax=vmax, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color('0.6'); s_.set_linewidth(0.4)
        p = node['path']
        ax.set_title(rf'$f_{{{int(p[0])}}}b_{{{int(p[1])}}}$', pad=1.5, fontsize=6,
                     color='0.35')
    attn_caption = (f'CLS-token weight {A_flat[:, 0].mean():.3f}; deviation of each '
                    rf'leaf $k_0$ from the mean' + '\n'
                    rf'(all {len(A_flat)} factor maps: pairwise $\cos\geq{cos_min:.2f}$)')

    # ── (f) the strongest real stimuli of each output circuit ────────────────
    n_top = min(8, NODES[0]['top_images'].shape[1])
    gsf = gs_bot[0, 1].subgridspec(n_c, n_top + 1, width_ratios=[0.42] + [1] * n_top)
    for j, c in enumerate(CIRC):
        row_label(fig, gsf[j, 0], rf"$f_{{{int(c['k'])}}}$",
                  color=C_EVEN if c['profile'][0] > 0.5 else C_ODD)
        for t in range(n_top):
            ax = fig.add_subplot(gsf[j, t + 1])
            ax.imshow(NODES[0]['top_images'][j, t, 0], cmap='gray_r',
                      interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            if j == 0 and t == 0:
                anchors['f'] = ax

    figstyle.freeze(fig)

    def _label(key, spacer, text, dx=0.0):
        fig.text(max(anchors[key].get_position().x0 + dx, 0.002),
                 spacers[spacer].get_position().y0, text, ha='left', va='bottom',
                 fontsize=7, fontweight='bold')

    _label('a', 0, '(a) informativity spectra, by layer of the block', dx=-0.030)
    _label('b', 0, '(b) output circuits', dx=-0.028)
    _label('c', 0, '(c) FFN units per circuit', dx=-0.020)
    _label('d', 1, '(d) $L_{\\mathrm{FFN1}}$ sub-factors of each output circuit — '
                   'weighted-average input (top, bold: dominant digit) and per-digit '
                   'loading (bottom; blue even, red odd)', dx=-0.006)
    _label('e', 2, '(e) CLS attention per factor', dx=-0.022)
    p_e = ax_e.get_position()
    fig.text(max(p_e.x0 - 0.022, 0.002), p_e.y0 - 0.028, attn_caption, ha='left',
             va='top', fontsize=6, color='0.35', linespacing=1.25)
    _label('f', 2, '(f) strongest stimuli per circuit', dx=-0.030)
    return fig


# ── shared helper: cosine silhouette (used by the fingerprint figures) ────────

def cosine_silhouette(X, labels):
    """Mean silhouette under cosine distance — numpy only (no sklearn in figures)."""
    U = unit(np.asarray(X, float))
    Dm = 1.0 - U @ U.T
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    out = np.empty(len(labels))
    for i in range(len(labels)):
        same = labels == labels[i]
        n_same = same.sum()
        a = (Dm[i, same].sum() / (n_same - 1)) if n_same > 1 else 0.0
        b = min(Dm[i, labels == c].mean() for c in uniq if c != labels[i])
        out[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(out.mean())


# ── stimulus-bootstrap confidence intervals (numpy only; Appendix "Statistical
# Reporting"). A row drawn w times in a resample gives identical fingerprints, so
# the whole resample is summarized by the per-row draw count and the silhouette is
# a weighted mean of per-row silhouettes computed from Dm @ Wmat — one BLAS matmul
# per resample, exact, and fast enough to run at figure-render time. ─────────────

def _sil_from_counts(Dm, yidx, C, w):
    Wmat = np.zeros((len(w), C))
    for c in range(C):
        Wmat[yidx == c, c] = w[yidx == c]
    M = Dm @ Wmat
    Nc = Wmat.sum(0)
    own = M[np.arange(len(w)), yidx]
    a = np.where(Nc[yidx] > 1, own / np.maximum(Nc[yidx] - 1, 1), 0.0)
    Mmean = M / np.maximum(Nc[None, :], 1e-12)
    Mmean[np.arange(len(w)), yidx] = np.inf
    b = Mmean.min(1)
    denom = np.maximum(a, b)
    s = np.where(denom > 0, (b - a) / denom, 0.0)
    tot = w.sum()
    return float((w * s).sum() / tot) if tot > 0 else 0.0


def silhouette_ci(X, labels, n_boot=1000, seed=0):
    """(point, lo, hi): cosine silhouette with a 95% percentile stimulus-bootstrap
    CI. Matches cosine_silhouette exactly at w=1."""
    U = unit(np.asarray(X, float))
    Dm = 1.0 - U @ U.T
    np.clip(Dm, 0, 2, out=Dm); np.fill_diagonal(Dm, 0.0)
    uniq, yidx = np.unique(np.asarray(labels), return_inverse=True)
    C, n = len(uniq), len(yidx)
    point = _sil_from_counts(Dm, yidx, C, np.ones(n))
    rng = np.random.default_rng(seed)
    vals = np.array([_sil_from_counts(Dm, yidx, C,
                     np.bincount(rng.integers(0, n, n), minlength=n).astype(float))
                     for _ in range(n_boot)])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


# ── validation figures (notebook 09) ──────────────────────────────────────────
#
# One figure per model, all nine panels identical across models so the five can be
# read as a table. Rows are the three claims: the decomposition is faithful, it is
# robust, and it beats the controls. A panel a model cannot supply says why.

VAL_NA = {                    # why a panel is empty, keyed by the capability flag
    'recon': 'no causal reconstruction:\nlayer-dict trace has no model to re-run',
    'roundtrip': 'no NNLS round-trip:\nnot defined through attention nodes',
}


def _val_na(ax, msg):
    """A panel this model cannot supply — say so rather than leaving a hole.

    The dashed frame is what makes the gap read as deliberate: the five model
    figures share one layout, so an empty cell has to look empty *on purpose*.
    """
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color('0.85'); s.set_linewidth(0.5); s.set_linestyle((0, (2, 2)))
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha='center', va='center',
            fontsize=6, color='0.5', style='italic', linespacing=1.35)


def _val_causal_nodes(D):
    """(layer index, pre-activation R²) for every causally reconstructed node.

    S2 walks the traced tree; FU2 covers fc nodes S2 could not reach, which in
    layer-dict mode (ViT) is the only causal measurement there is. Node ids carry
    the layer as their 'L<i>:' prefix.
    """
    if isinstance(D.get('recon'), dict):
        nodes = D['recon']['per_node']
        return (np.array([int(n['layer_idx']) for n in nodes]),
                np.array([float(n['preact_r2']) for n in nodes]), 'S2')
    fu2 = D.get('FU2_recon_fc')
    if isinstance(fu2, dict) and fu2:
        items = sorted(fu2.items())
        return (np.array([int(re.match(r'L(\d+)', k).group(1)) for k, _ in items]),
                np.array([float(v['preact_r2']) for _, v in items]), 'FU2')
    return None, None, None


def _cfg_label(name):
    """Sweep row id -> something a caption can carry: 'rank=rank x1.3' -> 'rank ×1.3'."""
    return (str(name).replace('rank=rank ', 'rank ').replace('=', ' ')
            .replace(' x', r' $\times$'))


def _executed_k_max(k_max):
    """The profile that actually runs: ``_auto_k_factorize`` floors ``k_max`` at 2.

    The sweep emits raw ``round(m * K)`` profiles, so a requested rank of 1 is
    printed here as the 2 that ``src/bft.py`` substitutes. Printing the request
    instead would put a number in the figure that no notebook can be set to.
    """
    return [max(2, int(v)) for v in k_max]


def _layer_shades(n, base):
    """Input layer darkest -> output layer lightest, so depth is readable."""
    return [tint(base, 0.62 * i / max(n - 1, 1)) for i in range(n)]


# ── Figure 8 / Appendix N — SqueezeNet on ImageNet, read through its stimuli ──
#
# Same conv-trace export schema as the CIFAR-10 CNN (Fig. 6 / App. E), so these
# reuse its stimulus helpers (montage, stim_panel, cifar_rgb, rgb_strip). What
# differs is the model: a pretrained 1000-way SqueezeNet 1.1, traced along its
# squeeze spine (conv1 -> fire2..fire9 -> classifier), 8 held-out categories.

IMAGENET_LAYER_LABEL = {
    'classifier.1': 'classifier', 'classifier': 'classifier',
    'features.12.squeeze': 'fire9', 'features.11.squeeze': 'fire8',
    'features.10.squeeze': 'fire7', 'features.9.squeeze': 'fire6',
    'features.7.squeeze': 'fire5', 'features.6.squeeze': 'fire4',
    'features.4.squeeze': 'fire3', 'features.3.squeeze': 'fire2',
    'features.0': 'conv1'}


def imagenet_layer_label(name):
    return IMAGENET_LAYER_LABEL.get(str(name), str(name))


# depth labels L_1 (pixels) .. L_10 (classifier), matching Fig. 8's convention
IMAGENET_LI_LABEL = {
    'classifier.1': r'$L_{10}$', 'classifier': r'$L_{10}$',
    'features.12.squeeze': r'$L_9$', 'features.11.squeeze': r'$L_8$',
    'features.10.squeeze': r'$L_7$', 'features.9.squeeze': r'$L_6$',
    'features.7.squeeze': r'$L_5$', 'features.6.squeeze': r'$L_4$',
    'features.4.squeeze': r'$L_3$', 'features.3.squeeze': r'$L_2$',
    'features.0': r'$L_1$'}


def imagenet_li_label(name):
    return IMAGENET_LI_LABEL.get(str(name), str(name))


# fingerprint dims store the BFT layer index; map it to the same fire-module name.
IMAGENET_FP_LAYER = {9: 'classifier', 8: 'fire9', 7: 'fire8', 6: 'fire7', 5: 'fire6',
                     4: 'fire5', 3: 'fire4', 2: 'fire3', 1: 'fire2', 0: 'conv1'}


def imagenet_depth_purity(D):
    """Per layer (output first): class purity of every factor and its lambda share.

    Purity is a factor's largest category share; 1/n_categories if it fires for
    every category equally, 1 if for a single one.
    """
    rows = []
    for name, nodes in cnn_layers(D['nodes']):
        pur = np.concatenate([n['class_profile'].max(1) for n in nodes])
        lam = np.concatenate([n['lam_share'] for n in nodes])
        rows.append(dict(layer=name, purity=pur, lam=lam))
    return rows


def spatial_overlay(D, img, amap, gamma=0.7):
    """RGB stimulus dimmed and tinted where a factor's channel-weighted activation
    map is high — 'where in the image the factor fires'. amap is the coarse
    (h, w) feature-map response, upsampled by nearest-neighbor to the image."""
    rgb = cifar_rgb(D, img)
    H, W = rgb.shape[:2]
    a = amap.astype(float)
    a = a / (a.max() + 1e-12)
    a = np.kron(a, np.ones((H // a.shape[0] + 1, W // a.shape[1] + 1)))[:H, :W]
    a = a[..., None] ** gamma
    hot = np.array(matplotlib.colors.to_rgb(figstyle.color('ours')))
    return np.clip(rgb * (1 - 0.55 * a) + hot * (0.55 * a), 0, 1)


def fig8_imagenet_circuits(D):
    """Message: the stimuli that drive each factor make a pretrained 1000-way
    SqueezeNet legible from the logits to the pixels — at the classifier every
    factor is one category, one fire module back each circuit un-pools into
    appearance sub-groups, and at conv1 the factors are color/texture channels
    shared across circuits."""
    NODES = D['nodes']
    CLS = list(D['class_names'])
    N_CLASSES = len(D['classes'])
    by_path = {tuple(int(i) for i in n['path']): n for n in NODES}
    root = NODES[0]
    C_BFT, C_X = figstyle.color('ours'), figstyle.color('cross_class')
    n_root = root['n_factors']
    CLASS_COLOR = class_colors(range(N_CLASSES))

    # Circuits traced in (b): f0, f1, f2 -- bear, airplane, dog. Only root factors
    # with a sub-tree can be traced, and the root entry of n_branches (5) decides
    # how many there are, so f0-f4 are the candidates.
    SHOW = [0, 1, 2]
    N_TREE = 2                 # two fire8 sub-factors per circuit

    # row heights in inches, derived from the image grids they hold
    W = 6.975
    s_a = W / n_root                                   # (a) montage side, square
    ha = s_a + 0.30 + 0.13                             # montage + class bars + label
    W_PUR = 1.50                                       # (c) purity plot width
    # a narrower (c) plus tighter inter-tree/side gaps hand the sub-factor
    # montages more width, so the stimuli in (b) read larger; a short pill row
    # and a small a->b spacer then pull (b) up against (a) so the bigger images
    # cost no height.
    PAD_B, GAP_B, GAP_IN = 0.10, 0.38, 0.11            # side / inter-tree / intra-pair
    NODE_H, FLAB_H = 0.16, 0.13                        # node-label / f-label rows
    s_b = (W - W_PUR) / (N_TREE * len(SHOW) + (len(SHOW) - 1) * GAP_B
                         + len(SHOW) * GAP_IN + 2 * PAD_B)
    hb = NODE_H + s_b + FLAB_H                          # node label + image + f label
    sp, sp_ab = 0.13, 0.10                             # top label / a->b gap
    rows = [sp, ha, sp_ab, hb]
    total = sum(rows)

    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='paper',
                   height_to_width_ratio=total / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.008, w_pad=0.008,
                          hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(4, 1, height_ratios=rows, hspace=0.0)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2)]
    for s_ in spacers:
        s_.set_axis_off()
    anchors = {}

    def tag(ax, text, x=0.05, y=0.96, size=6.5, color='white'):
        ax.text(x, y, text, transform=ax.transAxes, ha='left', va='top',
                fontsize=size, color=color,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2,
                                                                foreground='0.15')])

    # ── (a) output factors: stimuli and the full category distribution ───────
    # The f label rides on the bar chart, not the stimulus montage — over a
    # photograph it is unreadable whatever the stroke.
    gsa = gs[1].subgridspec(3, n_root, height_ratios=[s_a, 0.30, 0.13])
    pmax = root['class_profile'].max()
    for k in range(n_root):
        ax = stim_panel(fig, gsa[0, k], D, root['top_images'][k], 2, 2)
        prof = root['class_profile'][k]
        c = int(np.argmax(prof))
        axb = fig.add_subplot(gsa[1, k])
        axb.bar(range(N_CLASSES), prof, width=0.9, edgecolor='none',
                color=[CLASS_COLOR[i] for i in range(N_CLASSES)])
        axb.set_xlim(-0.6, N_CLASSES - 0.4); axb.set_ylim(0, pmax * 1.05)
        axb.set_xticks([]); axb.set_yticks([])
        for s_ in axb.spines.values():
            s_.set_visible(False)
        axb.text(0.0, 1.02, rf'$f_{{{k}}}$', transform=axb.transAxes, ha='left',
                 va='top', fontsize=6.5, color=C_BFT,
                 bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.4))
        lab = fig.add_subplot(gsa[2, k]); lab.set_axis_off()
        lab.text(0.5, 1.0, f'{CLS[c][:8]} {prof[c]:.2f}', ha='center', va='top',
                 fontsize=6.0, color='0.2')
        if k == 0:
            anchors['a'] = ax

    # ── row 2: (b) traceback | (c) category purity by depth ──────────────────
    gs_row = gs[3].subgridspec(1, 2, width_ratios=[W - W_PUR, W_PUR])
    # two image cells per circuit with a small gap between them (so the two
    # sub-factors read as distinct), circuits separated by a wider gap.
    widths, col_of = [PAD_B], {}
    for j_ in range(len(SHOW)):
        col_of[j_] = len(widths)
        widths += [1.0, GAP_IN, 1.0]
        widths += [GAP_B] if j_ < len(SHOW) - 1 else [PAD_B]
    gsb = gs_row[0, 0].subgridspec(3, len(widths),
                                   height_ratios=[NODE_H, s_b, FLAB_H],
                                   width_ratios=widths)
    tree = []
    for j_, r in enumerate(SHOW):
        node = by_path[(r,)]
        own = int(np.argmax(root['class_profile'][r]))
        base = col_of[j_]
        node_ax = fig.add_subplot(gsb[0, base:base + 3]); node_ax.set_axis_off()
        node_ax.text(0.5, 0.32, rf'$f_{{{r}}}$ · {CLS[own][:8]}', ha='center',
                     va='center', fontsize=6.5, color='white',
                     bbox=dict(boxstyle='round,pad=0.28', fc=C_BFT, ec='none'))
        subs = []
        for slot, k in enumerate(range(N_TREE)):        # slot -> columns base, base+2
            col = base + 2 * slot
            prof = node['class_profile'][k]
            c = int(np.argmax(prof))
            off = c != own and prof[c] - prof[own] > 0.10
            ax = stim_panel(fig, gsb[1, col], D, node['top_images'][k], 2, 2,
                            ec=C_X if off else '0.6', lw=1.1 if off else 0.4)
            if off:
                tag(ax, CLS[c][:6], x=0.5, y=0.04, size=6, color=C_X)
                ax.texts[-1].set(ha='center', va='bottom')
            flab = fig.add_subplot(gsb[2, col]); flab.set_axis_off()
            flab.text(0.5, 0.92, rf'$f_{{{k}}}$', ha='center', va='top', fontsize=6.5,
                      color=C_BFT)
            subs.append(ax)
        tree.append((node_ax, subs))
        if j_ == 0:
            anchors['b'] = node_ax

    # ── (c) category purity falls toward the input ───────────────────────────
    stats = imagenet_depth_purity(D)
    x = np.arange(len(stats))
    chance = 1.0 / N_CLASSES
    # generic depth labels L_1 (nearest the pixels) .. L_N (output), matching the
    # convention of Fig. 2; stats run output-first, so the ticks descend
    n_lay = len(stats)
    xt = [rf'$L_{{{n_lay - i}}}$' for i in range(n_lay)]
    ax_c = fig.add_subplot(gs_row[0, 1])
    pur = [lam_weighted(st['purity'], st['lam']) for st in stats]
    ax_c.axhline(chance, color='0.65', lw=0.6, ls=(0, (3, 2)), zorder=0)
    for i_, st in enumerate(stats):
        ax_c.scatter(np.full(len(st['purity']), i_), st['purity'], s=1.8, color=C_BFT,
                     alpha=0.28, edgecolor='none', zorder=2)
    # 95% stimulus-bootstrap CI on the lambda-weighted purity (Appendix stats)
    from src import figdata as _fd
    _lab = np.asarray(_fd.load('nb05_fingerprints')['fp']['id_targets'])
    _ci = purity_ci_by_layer(D, _lab)
    _yerr = np.array([[max(p - lo, 0), max(hi - p, 0)] for p, lo, hi in _ci]).T
    ax_c.errorbar(x, pur, yerr=_yerr, fmt='none', ecolor='0.25', elinewidth=0.7,
                  capsize=1.5, capthick=0.7, zorder=4)
    ax_c.plot(x, pur, color=C_BFT, marker='o', ms=2.6, lw=1.3, zorder=3)
    ax_c.set_ylim(0, 1.0); ax_c.set_yticks([0, 0.5, 1.0])
    ax_c.set_yticklabels(['0', '.5', '1'])
    ax_c.set_ylabel('category purity', color=C_BFT, labelpad=1)
    ax_c.text(-0.3, chance + 0.03, 'chance', fontsize=6, color='0.45',
              ha='left', va='bottom')
    ax_c.set_xlim(-0.5, len(stats) - 0.5); ax_c.set_xticks(x)
    ax_c.set_xticklabels(xt, fontsize=6.0)
    ax_c.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_c.spines[s_].set_visible(False)
    anchors['c'] = ax_c

    figstyle.freeze(fig)

    for node_ax, subs in tree:
        nb = node_ax.get_position()
        x0, y0 = (nb.x0 + nb.x1) / 2, nb.y0 + 0.10 * nb.height
        for sub in subs:
            sb = sub.get_position()
            fig.add_artist(matplotlib.lines.Line2D(
                [x0, (sb.x0 + sb.x1) / 2], [y0, sb.y1], color='0.6', lw=0.5,
                zorder=0))

    def _label(key, spacer, text, dx=0.0, **kw):
        fig.text(max(anchors[key].get_position().x0 + dx, 0.002),
                 spacers[spacer].get_position().y0 + 0.004, text, ha='left',
                 va='bottom', fontsize=7, fontweight='bold', **kw)

    _label('a', 0, '(a) output factors, with category distribution', dx=-0.004)
    _label('b', 1, r'(b) traceback to $L_9$', dx=-0.010)
    _label('c', 1, '(c) category purity by depth', dx=-0.030)
    return fig


def imagenet_spine(NODES, circuit_k):
    """The traced spine of one circuit, output first: root -> circuit node ->
    follow the 0-branch down to conv1."""
    by = {tuple(int(i) for i in n['path']): n for n in NODES}
    path, chain = (int(circuit_k),), [by[()], by[(int(circuit_k),)]]
    while tuple(list(path) + [0]) in by:
        path = tuple(list(path) + [0])
        chain.append(by[path])
    return chain


def figN_imagenet_details(D):
    """Message: the Fig. 8 trace in full — every node's informativity spectrum, a
    wider gallery of each circuit's fire8 sub-factors, one circuit followed all the
    way from the classifier to conv1, and the spatial activation maps that show each
    output factor fires on its object."""
    NODES, CLS = D['nodes'], list(D['class_names'])
    N_CLASSES = len(D['classes'])
    by_path = {tuple(int(i) for i in n['path']): n for n in NODES}
    root = NODES[0]
    C_BFT, C_X = figstyle.color('ours'), figstyle.color('cross_class')
    LAYERS = cnn_layers(NODES)
    CIRC = [c['k'] for c in D['circuits']]                 # 5 traced circuits
    SPINE_CIRC = 1                                          # airplane, for (c)
    chance = 1.0 / N_CLASSES

    N_SHOW = 3                     # (b): sub-factors per circuit
    EX_R, EX_C = 2, 2              # example stimuli per sub-factor (4 of the 5 stored,
                                   # so the montage is full — no empty cell)
    PER_ROW = 1                    # one circuit per gallery row (wavg + examples wide)
    LAB_W, AVG_W = 0.55, 1.0       # circuit-label / weighted-average column widths
    EX_W = EX_C / EX_R             # keep every example tile square
    SUBGAP, CIRCGAP, FLAB = 0.14, 0.42, 0.20   # gaps, and the f-label row height
    S_MAX = 0.75                   # cap the gallery montage side (else the row is huge)

    n_c = len(CIRC)
    W, SP = 6.975, 0.15
    n_gal_rows = -(-n_c // PER_ROW)
    # columns of one circuit: label, then [weighted-avg, examples] per sub-factor
    circ_cols = [LAB_W]
    for _s in range(N_SHOW):
        circ_cols += [AVG_W, EX_W] + ([SUBGAP] if _s < N_SHOW - 1 else [])
    inner = []
    for _b in range(PER_ROW):
        inner += circ_cols + ([CIRCGAP] if _b < PER_ROW - 1 else [])
    s_g = min(W / sum(inner), S_MAX)                       # montage side
    pad = max((W / s_g - sum(inner)) / 2, 0.0)             # side padding to centre
    col_w, col_base = [pad], []
    for _b in range(PER_ROW):
        col_base.append(len(col_w))
        col_w += circ_cols + ([CIRCGAP] if _b < PER_ROW - 1 else [])
    col_w += [pad]
    h_b = n_gal_rows * (s_g + FLAB) + 0.04
    spine = imagenet_spine(NODES, SPINE_CIRC)
    n_sp = len(spine)
    s_sp = W / (n_sp + (n_sp - 1) * 0.14)                   # spine montage side
    h_c = s_sp + 0.34
    rows = [SP, 1.05, SP, h_b, SP, h_c]

    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='appendix',
                   height_to_width_ratio=sum(rows) / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.008, w_pad=0.008,
                          hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(6, 1, height_ratios=rows, hspace=0.0)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4)]
    for s_ in spacers:
        s_.set_axis_off()
    anchors = {}

    def tag(ax, text, x=0.05, y=0.96, size=6, color='white', ha='left', va='top'):
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size,
                color=color, path_effects=[matplotlib.patheffects.withStroke(
                    linewidth=1.2, foreground='0.15')])

    # ── (a) informativity spectrum of every node, by layer ───────────────────
    gsa = gs[1].subgridspec(1, len(LAYERS))
    for i, (name, nodes) in enumerate(LAYERS):
        ax = fig.add_subplot(gsa[0, i])
        kmax = max(n['n_factors'] for n in nodes)
        L = np.full((len(nodes), kmax), np.nan)
        for j_, n in enumerate(nodes):
            L[j_, :n['n_factors']] = n['lam_share']
        x = np.arange(kmax)
        # a single-factor node otherwise draws one panel-filling block; keep that
        # bar as narrow as the multi-factor bars elsewhere
        ax.bar(x, np.nanmean(L, 0), color=tint(C_BFT, 0.45), edgecolor='0.25',
               linewidth=0.3, width=(0.3 if kmax == 1 else 0.75), zorder=1)
        if len(nodes) > 1:
            for row in L:
                ax.scatter(x, row, s=1.4, color=C_BFT, zorder=2, edgecolor='none')
        ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1.0])
        ax.set_yticklabels(['0', '.5', '1'] if i == 0 else [])
        ax.set_xticks(x); ax.set_xlim(-0.6, kmax - 0.4)
        step = 2 if kmax > 6 else 1
        ax.set_xticklabels([str(v) if v % step == 0 else '' for v in x], fontsize=6)
        ax.tick_params(length=1.5, pad=1)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.set_title(imagenet_li_label(name), pad=1.5, fontsize=6.5)
        if i == 0:
            ax.set_ylabel(r'$\lambda$ share', labelpad=1)
            anchors['a'] = ax
        if i == len(LAYERS) // 2:
            ax.set_xlabel('factor $f$', labelpad=1)

    # ── (b) gallery: each output circuit's strongest L8 sub-factors, shown as
    #        the weighted-average stimulus (framed, the factor's prototype)
    #        beside real top stimuli; all factor labels sit below the images ────
    row_h = []
    for _ in range(n_gal_rows):
        row_h += [s_g, FLAB]
    gsb = gs[3].subgridspec(2 * n_gal_rows, len(col_w), width_ratios=col_w,
                            height_ratios=row_h)
    for bi, r in enumerate(CIRC):
        grow, blk = bi // PER_ROW, bi % PER_ROW
        irow, lrow = 2 * grow, 2 * grow + 1
        base = col_base[blk]
        own = int(np.argmax(root['class_profile'][r]))
        lab = fig.add_subplot(gsb[irow, base]); lab.set_axis_off()
        lab.text(1.0, 0.5, rf'$f_{{{r}}}$' + '\n' + CLS[own][:8], ha='right',
                 va='center', fontsize=6, linespacing=1.2)
        if bi == 0:
            anchors['b'] = lab
        cnode = by_path[(r,)]
        for k in range(N_SHOW):
            ac, ec = base + 1 + 3 * k, base + 2 + 3 * k        # avg / examples cols
            prof = cnode['class_profile'][k]
            c = int(np.argmax(prof))
            off = c != own and prof[c] - prof[own] > 0.10
            axa = fig.add_subplot(gsb[irow, ac])               # weighted-avg stimulus
            axa.imshow(cifar_rgb(D, cnode['wavg'][k]), interpolation='nearest')
            axa.set_xticks([]); axa.set_yticks([])
            for s_ in axa.spines.values():
                s_.set_color(C_BFT); s_.set_linewidth(1.0)
            stim_panel(fig, gsb[irow, ec], D, cnode['top_images'][k], EX_R, EX_C,
                       ec=C_X if off else '0.6', lw=1.1 if off else 0.4)
            la = fig.add_subplot(gsb[lrow, ac]); la.set_axis_off()
            la.text(0.5, 0.95, 'avg', ha='center', va='top', fontsize=6, color='0.45')
            le = fig.add_subplot(gsb[lrow, ec]); le.set_axis_off()
            le.text(0.5, 0.95, rf'$f_{{{k}}}$' + (f' → {CLS[c][:6]}' if off else ''),
                    ha='center', va='top', fontsize=6, color=C_X if off else '0.2')

    # ── (c) the traced spine of one circuit, classifier to conv1 ─────────────
    tiles, gaps, meta = [], [], []
    for i, n in enumerate(spine):
        kk = SPINE_CIRC if i == 0 else 0                   # root: the circuit factor
        tiles.append(montage(D, n['top_images'][kk], 2, 2))
        gaps.append(10)
        meta.append((n, kk))
    comp_c = hcat(tiles, gaps[:-1])
    side = tiles[0].shape[0]
    ax_c = strip_axes(fig, gs[5], comp_c, top=15, bottom=20)
    x = 0
    for n, kk in meta:
        ax_c.text(x + 2, 2, rf'$f_{{{kk}}}$', ha='left', va='top', fontsize=6,
                  color='white', path_effects=[matplotlib.patheffects.withStroke(
                      linewidth=1.2, foreground='0.15')])
        ax_c.text(x + side / 2, side + 3, f"{n['class_profile'][kk].max():.2f}",
                  ha='center', va='top', fontsize=6, color='0.2')
        if 'conn' in n and n['conn']['in_mass'].shape[1] == 3:      # conv1 RGB arbor
            sh = n['conn']['in_mass'][kk]; sh = sh / sh.sum()
            xs = x
            for frac, colr in zip(sh, ('#D62728', '#2CA02C', '#1F77B4')):
                ax_c.add_patch(matplotlib.patches.Rectangle(
                    (xs, side - 4), frac * side, 4, facecolor=colr, edgecolor='none',
                    zorder=5))
                xs += frac * side
        ax_c.text(x + side / 2, -3, imagenet_li_label(n['layer_name']),
                  ha='center', va='bottom', fontsize=6.5, color='0.35', rotation=0)
        x += side + 10
    anchors['c'] = ax_c

    figstyle.freeze(fig)

    def _label(key, spacer, text, dx=0.0):
        fig.text(max(anchors[key].get_position().x0 + dx, 0.002),
                 spacers[spacer].get_position().y0 + 0.002, text, ha='left',
                 va='bottom', fontsize=7, fontweight='bold')

    _label('a', 0, r'(a) $\lambda$ spectra by layer', dx=-0.026)
    _label('b', 1, r'(b) $L_8$ gallery: each output circuit $f_r$ traced back to its '
                   r'strongest $L_8$ sub-factors $f_k$ — weighted-avg. stimulus '
                   r'(framed) then top stimuli', dx=-0.004)
    _label('c', 2, r'(c) airplane circuit traced $L_{10}\to L_1$ '
                   rf'(purity below, $\rightarrow$ chance {chance:.2f})', dx=-0.004)
    return fig


# fig9_imagenet_fingerprints was merged into fig4_fingerprints_main (row 3:
# the ImageNet embedding, the activation slot and the silhouette head-to-head).
# Its remaining panels live in figO: the per-category means in (b), the class
# geometry in (c), far-OOD in (d)/(e)/(g) and the round-trip in (a).


# ── Appendix P — cross-model validation (merges the former Figs. P/Q/R) ───────
#
# One full-page figure over all five nb09 validation bundles: faithfulness (a–c),
# class separability (d,e) and rank robustness (f,g). The helpers below each draw
# one band from one model's bundle; figP_validation lays out the four bands.

def _faith_row(name, D):
    """One model's faithfulness numbers from its nb09 bundle: causal
    reconstruction (with its random-rank floor and exact ceiling), NNLS
    round-trip, and NMF-seed stability. A metric a model cannot supply is None
    (no causal reconstruction for the pretrained SqueezeNet, no round-trip for
    the ViT's attention nodes)."""
    caps = D['caps']
    li, r2, src = _val_causal_nodes(D)
    rc = D['recon_controls']

    def cv(key):
        vs = ([float(n[key]['preact_r2']) for n in rc.values() if key in n]
              if isinstance(rc, dict) else [])
        return float(np.mean(vs)) if vs else None

    stab = D['stability']['per_layer']
    rt = D.get('roundtrip')
    return dict(name=name, src=src,
                recon=float(np.median(r2)) if li is not None else None,
                floor=cv('random_R'), ceil=cv('exact'),
                rt=(float(rt['median']) if caps['roundtrip'] and isinstance(rt, dict)
                    else None),
                stab=float(np.mean([stab[k]['mean'] for k in stab])),
                stab_sd=float(np.std([stab[k]['mean'] for k in stab])))


def figP_validation(D):
    """Message: the cross-model validation figure — on every model the
    decomposition is faithful (a–c: causal reconstruction between a random-rank
    floor and the exact ceiling, NNLS round-trip, NMF-seed stability), the BFT
    fingerprint is the more class-separable code (d,e: it out-silhouettes the
    dimension-matched activations, and the weight term earns its place), and the
    factors are a property of the arbor rather than the exact NMF rank (f,g:
    K*±1 robustness and the rank sweep). Merges the former Figs. P/Q/R; the two
    honest gaps in (a,b) are shown as n/a rather than hidden.

    Ignores its D argument: loads all five nb09 validation bundles once and reads
    faithfulness (a–c), separability (d,e) and rank robustness (f,g) from them."""
    from src import figdata
    Ds = [(name, figdata.load(b)) for name, b in _VAL_MODELS]
    frows = [_faith_row(name, Dm) for name, Dm in Ds]
    srows = [_sil_row(name, Dm) for name, Dm in Ds]
    n = len(Ds)

    C_BFT, C_ACT = figstyle.color('ours'), figstyle.color('baseline')
    C_RAND, C_CEIL = figstyle.color('random'), figstyle.color('ceiling')
    y = np.arange(n)[::-1]

    # row heights in inches — generous, so nothing is squished (four bands:
    # faithfulness bars, separability bars, rank sensitivity, rank sweep)
    W = 6.975
    H = [0.24,   # 0  faithfulness legend
         1.42,   # 1  band A — (a–c) faithfulness, 3 horizontal-bar panels
         0.22,   # 2  spacer
         1.74,   # 3  band B — (d,e) separability, 2 grouped-bar panels
         0.24,   # 4  spacer
         0.16,   # 5  label (f)
         1.08,   # 6  band C — (f) rank sensitivity, 5 small multiples
         0.16,   # 7  label (g)
         1.12]   # 8  band D — (g) rank sweep, 5 small multiples
    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='appendix',
                   height_to_width_ratio=sum(H) / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.03, w_pad=0.05, hspace=0.03,
                          wspace=0.06)
    gs = fig.add_gridspec(len(H), 1, height_ratios=H, hspace=0.0)
    sp = {r: fig.add_subplot(gs[r]) for r in (0, 2, 4, 5, 7)}
    for s_ in sp.values():
        s_.set_axis_off()

    # ── (a–c) the decomposition is faithful ──────────────────────────────────
    gsa = gs[1].subgridspec(1, 3, width_ratios=[1.32, 1.0, 1.0], wspace=0.08)
    ax_r, ax_t, ax_s = [fig.add_subplot(gsa[0, c]) for c in range(3)]

    def barpanel(ax, key, title, *, gate=None, left=False, err_key=None):
        for yi, r in zip(y, frows):
            v = r[key]
            if v is None:
                ax.text(0.03, yi, 'n/a', va='center', ha='left', fontsize=6.4,
                        color='0.6', style='italic')
                continue
            ax.barh(yi, v, height=0.62, color=C_BFT, edgecolor='0.25', lw=0.3, zorder=3)
            if err_key is not None and r.get(err_key):
                ax.errorbar(v, yi, xerr=r[err_key], fmt='none', ecolor='0.2',
                            elinewidth=0.7, capsize=1.6, capthick=0.7, zorder=4)
            ax.text(min(v, 1.0) + 0.02, yi, f'{v:.2f}', va='center', ha='left',
                    fontsize=6.4, color='0.3')
        if gate is not None:
            ax.axvline(gate, color='0.4', lw=0.8, ls=':', zorder=2)
            ax.text(gate - 0.03, -0.55, f'{gate:g} gate', ha='right', va='center',
                    fontsize=6.0, color='0.4')
        ax.set_xlim(0, 1.2); ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(['0', '', '1'])
        ax.set_ylim(-0.8, n - 0.2)
        ax.set_title(title, fontsize=7.5, pad=3, fontweight='bold')
        ax.tick_params(length=1.5, pad=1.5)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.set_yticks(y)
        ax.set_yticklabels([r['name'] for r in frows] if left else [], fontsize=6.6)

    barpanel(ax_r, 'recon', r'(a) causal recon. $R^2$', left=True)
    for yi, r in zip(y, frows):
        if r['recon'] is None:
            continue
        if r['floor'] is not None:
            ax_r.scatter(max(r['floor'], 0.0), yi, marker='|', s=46, color=C_RAND,
                         linewidths=1.2, zorder=5)
        if r['ceil'] is not None:
            ax_r.scatter(min(r['ceil'], 1.19), yi, marker='|', s=46, color=C_CEIL,
                         linewidths=1.2, zorder=5)
    barpanel(ax_t, 'rt', '(b) NNLS round-trip')
    barpanel(ax_s, 'stab', '(c) NMF-seed stability', gate=0.85, err_key='stab_sd')

    handles = [matplotlib.patches.Patch(fc=C_BFT, ec='0.25', lw=0.3, label='BFT'),
               matplotlib.lines.Line2D([], [], color=C_RAND, lw=1.2, marker='|',
                                       ls='none', ms=7, label='random-rank floor (a)'),
               matplotlib.lines.Line2D([], [], color=C_CEIL, lw=1.2, marker='|',
                                       ls='none', ms=7, label='exact ceiling (a)'),
               matplotlib.lines.Line2D([], [], color='0.4', lw=0.8, ls=':',
                                       label='0.85 stability gate (c)')]
    sp[0].legend(handles=handles, loc='center', ncol=4, fontsize=6.8, frameon=False,
                 handlelength=1.2, handletextpad=0.4, columnspacing=1.6, borderpad=0.0)

    # ── (d,e) the fingerprint is the more class-separable code ───────────────
    gsb = gs[3].subgridspec(1, 2, wspace=0.14)
    ax_d, ax_e = fig.add_subplot(gsb[0, 0]), fig.add_subplot(gsb[0, 1])
    xg = np.arange(n); wbar = 0.36

    from math import comb

    def _sign_p(n_pos, n_tot):                     # one-sided sign test p-value
        return sum(comb(n_tot, k) for k in range(n_pos, n_tot + 1)) / 2 ** n_tot

    def grouped(ax, k0, k1, c0, c1, l0, l1):
        top = max(max(r[k0], r[k1]) for r in srows)
        ax.bar(xg - wbar / 2, [r[k0] for r in srows], wbar, color=c0, edgecolor='0.25',
               lw=0.3, zorder=3, label=l0)
        ax.bar(xg + wbar / 2, [r[k1] for r in srows], wbar, color=c1, edgecolor='0.25',
               lw=0.3, zorder=3, label=l1)
        ax.set_ylim(0, top * 1.20); ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel('cosine silhouette', fontsize=7, labelpad=2)
        ax.set_xticks(xg)
        ax.set_xticklabels([r['name'] for r in srows], fontsize=6.4, rotation=20,
                           ha='right')
        ax.tick_params(length=1.5, pad=1.5)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        ax.legend(fontsize=6.6, frameon=False, loc='upper right', handlelength=0.9,
                  handletextpad=0.35, borderpad=0.1, labelspacing=0.25)
        n_pos = int(sum(r[k0] > r[k1] for r in srows))
        ax.text(0.02, 0.98, f'{n_pos}/{len(srows)} models,\n$p$={_sign_p(n_pos, len(srows)):.3f}',
                transform=ax.transAxes, ha='left', va='top', fontsize=6.0, color='0.2',
                linespacing=1.1)

    grouped(ax_d, 'fp_sil', 'act_sil', C_BFT, C_ACT,
            'BFT fingerprint', 'dim-matched act.')
    ax_d.set_title('(d) fingerprint vs. network activations', fontsize=7.5, pad=3,
                   fontweight='bold')
    grouped(ax_e, 'arbor_nmf', 'act_nmf', C_BFT, tint(C_ACT, 0.5),
            r'arbor NMF ($W\!\cdot\!a$)', 'activation-only NMF')
    ax_e.set_title('(e) the weight term earns its place', fontsize=7.5, pad=3,
                   fontweight='bold')

    # ── (f) sensitivity to the rank, (g) rank vs. arbor reconstruction ───────
    gsc = gs[6].subgridspec(1, n, wspace=0.10)
    gsd = gs[8].subgridspec(1, n, wspace=0.10)
    for c, (name, Dm) in enumerate(Ds):
        ac = fig.add_subplot(gsc[0, c])
        _draw_ranksens(ac, Dm, ylab=(c == 0))
        ac.set_title(name, fontsize=7, pad=2)
        ad = fig.add_subplot(gsd[0, c])
        _draw_ranksweep(ad, Dm, ylab=(c == 0))

    figstyle.freeze(fig)
    for row, txt in ((5, r'(f) sensitivity to the rank ($K^{*}\!\pm\!1$; each dot a '
                          r'node, bar the median)'),
                     (7, '(g) rank vs. arbor reconstruction (dot: rank used, line '
                         'shade: layer depth)')):
        fig.text(0.006, sp[row].get_position().y0, txt, ha='left', va='bottom',
                 fontsize=7.5, fontweight='bold')
    return fig


def _draw_ranksens(ax, D, *, ylab=False):
    """Panel-e drawing for one model: how close the re-run factors stay to the
    K* factors when the rank is set to K*-1, K* and K*+1 — the factors are a
    property of the arbor, not of the exact rank."""
    C_BFT = figstyle.color('ours')
    ks = D['stability']['k_sensitivity']
    COLS = [(r'$-1$', np.asarray(ks['k_minus1']), tint(C_BFT, 0.45)),
            (r'$K^{*}$', np.asarray(ks['k_star']), C_BFT),
            (r'$+1$', np.asarray(ks['k_plus1']), tint(C_BFT, 0.45))]
    rng = np.random.default_rng(1)
    for i, (lab, v, col) in enumerate(COLS):
        x = i + (rng.random(len(v)) - 0.5) * 0.3
        ax.scatter(x, v, s=7, color=col, edgecolor='none', alpha=0.8, zorder=3)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color='0.2', lw=0.9, zorder=4)
    ax.set_xticks(range(3)); ax.set_xticklabels([c[0] for c in COLS], fontsize=6)
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '.5', '1'] if ylab else [])
    ax.tick_params(length=1.5, pad=1.5)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    if ylab:
        ax.set_ylabel(r'cosine to $K^{*}$', fontsize=6.5, labelpad=1)


def _draw_ranksweep(ax, D, *, ylab=False):
    """Panel-f drawing for one model: arbor reconstruction R^2 against the NMF
    rank K for every layer (dot: the rank used) — the choice sits on a plateau,
    not a cliff."""
    C_BFT = figstyle.color('ours')
    fu = D['FU1_rank_sweep']['per_layer']
    fkeys = sorted(fu, key=int)
    fshades = _layer_shades(len(fkeys), C_BFT)
    for k, col in zip(fkeys, fshades):
        sw = fu[k]['sweep']
        K = np.array([s['K'] for s in sw]); R = np.array([s['recon_r2'] for s in sw])
        ax.plot(K, R, color=col, lw=0.8, alpha=0.9, zorder=2)
        dk = int(fu[k]['default_k'])
        if dk in K:
            ax.scatter([dk], [R[list(K).index(dk)]], s=11, color=col, zorder=4,
                       edgecolor='white', linewidth=0.4)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '.5', '1'] if ylab else [])
    ax.tick_params(length=1.5, pad=1.5)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    ax.set_xlabel(r'rank $K$', fontsize=6.5, labelpad=1)
    if ylab:
        ax.set_ylabel(r'arbor $R^2$', fontsize=6.5, labelpad=1)


_VAL_MODELS = [
    ('MLP even/odd', 'nb09_mlp_even_odd_validation'),
    ('MLP digit', 'nb09_mlp_digit_validation'),
    ('CIFAR-10 CNN', 'nb09_cnn_cifar_validation'),
    ('TinyViT', 'nb09_vit_mnist_validation'),
    ('ImageNet', 'nb09_imagenet_cnn_validation'),
]


def _sil_row(name, D):
    """One model's silhouette numbers from its nb09 bundle: the BFT fingerprint
    against the dimension-matched activations (the separability claim), and the
    weight-term advantage — the same NMF run on the arbor ($W\\!\\cdot\\!a$) vs on
    the activations alone."""
    sep = D['separability']['by_fine']
    a1 = D['A1_weight_vs_activation']['fingerprint_separability']
    return dict(name=name,
                fp_sil=float(sep['bft_fingerprint']['silhouette']),
                act_sil=float(sep['act_matched']['silhouette']),
                arbor_nmf=float(a1['arbor_nmf']['silhouette']),
                act_nmf=float(a1['activation_nmf']['silhouette']))


# ── Appendix fingerprint figures: two cross-model figures (OOD / structure) ───
#
# figfp_ood and figfp_structure below merge the former per-model fingerprint
# figures C/D/F/H/O and the cross-model figS into two: everything OOD (figfp_ood)
# and everything else (figfp_structure). The NNLS round-trip is dropped — it is
# Fig. P. Both load all five nb0N_fingerprints bundles internally.

_OOD_MODELS = [
    ('$8\\times4$ MLP', 'nb01_fingerprints'),
    ('digit MLP', 'nb02_fingerprints'),
    ('CIFAR-10 CNN', 'nb03_fingerprints'),
    ('TinyViT', 'nb04_fingerprints'),
    ('ImageNet', 'nb05_fingerprints'),
]


def _cos_to_mean(X):
    """Mean cosine of each fingerprint to its set's mean — how tightly a condition
    collapses onto a single fingerprint (1 = one point, lower = spread out)."""
    U = unit(np.asarray(X, float))
    m = unit(U.mean(0, keepdims=True))
    return float((U @ m.T).mean())


def ood_signature_one(name, D):
    """The OOD signature of one model: how far each condition collapses, and how
    close its mean fingerprint sits to a trained-class centroid. Near-OOD is the
    model's held-out real distribution (held-out digits / Fashion-MNIST /
    CIFAR-100); the pretrained SqueezeNet ships none, so its near bars are absent."""
    fp = D['fp']
    y = np.asarray(fp['id_targets']); X = unit(np.asarray(fp['id'], float))
    cents = unit(np.stack([X[y == c].mean(0) for c in np.unique(y)]))

    def maxc(v):
        return float((unit(np.asarray(v, float)[None]) @ cents.T).max())

    near = fp.get('ood')
    if near is None:
        near = fp.get('fmnist')
    far = fp['far']
    out = dict(name=name,
               collapse_id=_cos_to_mean(fp['id']),
               collapse_far=float(np.mean([_cos_to_mean(f['F']) for f in far])),
               cent_far=float(np.mean([maxc(np.asarray(f['F']).mean(0)) for f in far])),
               cent_id=float(np.mean([maxc(X[i])
                                      for i in range(0, len(X), max(1, len(X) // 80))])))
    if near is not None:
        near = np.asarray(near, float)
        out['collapse_near'] = _cos_to_mean(near)
        out['cent_near'] = maxc(near.mean(0))
    else:
        out['collapse_near'] = out['cent_near'] = None
    return out


def _ood_cond_heatmap(ax, D, CMAP, C_ID, C_NEAR, C_FAR, *, xlabel=False):
    """One model's mean fingerprint per stimulus condition (in-distribution /
    near-OOD / the four synthetic far-OOD), each row normalized, factors in the
    bundle's circuit order with dividers — far-OOD concentrates on a sparse
    subset of factors while in-distribution spreads across its circuits."""
    fp = D['fp']
    col_order = list(D['col_order'])
    edges = (list(D['blk_edge']) if 'blk_edge' in D
             else ([int(D['n_ev'])] if 'n_ev' in D else []))
    Xid = np.asarray(fp['id'], float)
    rows = [('in-distribution', Xid.mean(0), C_ID)]
    near = fp.get('ood')
    if near is None:
        near = fp.get('fmnist')
    if near is not None:
        rows.append((D['cond'][1]['label'], np.asarray(near, float).mean(0), C_NEAR))
    for f in fp['far']:
        rows.append((f['label'], np.asarray(f['F'], float).mean(0), C_FAR))
    M = np.stack([m for _, m, _ in rows]); M = M / M.sum(1, keepdims=True)
    ax.imshow(M[:, col_order], cmap=CMAP, aspect='auto', interpolation='nearest',
              norm=matplotlib.colors.PowerNorm(0.7, vmin=0, vmax=M.max()))
    for e in edges:
        ax.axvline(e - 0.5, color='0.30', lw=0.5)
    ax.set_xticks([])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(r[2])
    ax.tick_params(length=1.5, pad=1)
    for s_ in ax.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    if xlabel:
        ax.set_xlabel(f'{len(col_order)} factors, grouped by output circuit',
                      labelpad=2, fontsize=6.5)


def figfp_ood(D):
    """Message: every model shows the same OOD signature (merges the OOD panels of
    the former Figs. C/D/F/H/O and all of Fig. S). In-distribution and near-OOD
    (real held-out data) fingerprints stay spread out and land on a trained class;
    the four synthetic far-OOD conditions collapse onto essentially one fingerprint
    (a) and, for the deeper nets, sit off every class (b). The 8x4 MLP's held-out
    digits track the network's own P(odd), not the true parity (c). Per model, the
    mean fingerprint per condition (d-h) shows far-OOD concentrating on a sparse
    factor subset. Ignores its D argument; loads all five nb0N_fingerprints."""
    from src import figdata
    Ds = [(name, figdata.load(b)) for name, b in _OOD_MODELS]
    rows = [ood_signature_one(name, Dm) for name, Dm in Ds]
    C_ID = figstyle.color('baseline')
    C_NEAR, C_FAR = figstyle.color('near_ood'), figstyle.color('far_ood')
    C_BFT = figstyle.color('ours'); CMAP = ramp_cmap(C_BFT, 'bft')
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')

    W = 6.975
    H = [0.20,   # 0 legend
         1.55,   # 1 band A: collapse / distance / held-out
         0.20,   # 2 spacer
         0.30,   # 3 band-B label
         5.20]   # 4 band B: five per-model condition heatmaps
    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='appendix',
                   height_to_width_ratio=sum(H) / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.03, w_pad=0.05, hspace=0.03,
                          wspace=0.05)
    gs = fig.add_gridspec(len(H), 1, height_ratios=H, hspace=0.0)
    sp_leg = fig.add_subplot(gs[0]); sp_leg.set_axis_off()
    sp_lbl = fig.add_subplot(gs[3]); sp_lbl.set_axis_off()

    # ── (a,b) cross-model collapse and distance, (c) held-out tracking ───────
    xg = np.arange(len(rows)); wbar = 0.26
    SHORT = [r'$8{\times}4$', 'digit', 'CNN', 'ViT', 'ImageNet']
    gsa = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.0, 0.94], wspace=0.30)
    ax_a, ax_b, ax_c = [fig.add_subplot(gsa[0, c]) for c in range(3)]

    def triplet(ax, id_key, near_key, far_key):
        ax.bar(xg - wbar, [r[id_key] for r in rows], wbar, color=C_ID,
               edgecolor='0.25', lw=0.3, zorder=3)
        for i, r in enumerate(rows):                # near-OOD, skipped where absent
            if r[near_key] is not None:
                ax.bar(i, r[near_key], wbar, color=C_NEAR, edgecolor='0.25', lw=0.3,
                       zorder=3)
        ax.bar(xg + wbar, [r[far_key] for r in rows], wbar, color=C_FAR,
               edgecolor='0.25', lw=0.3, zorder=3)
        ax.set_xticks(xg)
        ax.set_xticklabels(SHORT, fontsize=6.2, rotation=30, ha='right')
        ax.set_ylim(0, 1.04); ax.set_yticks([0, 0.5, 1.0])
        ax.tick_params(length=1.5, pad=1.5)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)

    triplet(ax_a, 'collapse_id', 'collapse_near', 'collapse_far')
    ax_a.set_ylabel('mean cosine to condition mean', fontsize=7, labelpad=2)
    ax_a.set_title('(a) collapse onto one fingerprint', fontsize=7.5, pad=3,
                   fontweight='bold')
    triplet(ax_b, 'cent_id', 'cent_near', 'cent_far')
    ax_b.set_ylabel('max cosine to a class centroid', fontsize=7, labelpad=2)
    ax_b.set_title('(b) distance to any trained class', fontsize=7.5, pad=3,
                   fontweight='bold')

    D01 = Ds[0][1]
    DIGIT_COLOR = digit_colors(list(D01['digit_order']), C_EVEN, C_ODD)
    mlp_heldout(ax_c, D01, DIGIT_COLOR, C_EVEN, C_ODD, float(D01['r_ood']))
    ax_c.set_title(r'(c) held-out digits track P(odd)',
                   fontsize=7.5, pad=3, fontweight='bold')
    ax_c.text(0.04, 0.86, r'$8\times4$ MLP', transform=ax_c.transAxes, ha='left',
              va='top', fontsize=6.5, color='0.35')

    handles = [matplotlib.patches.Patch(fc=C_ID, ec='0.25', lw=0.3,
                                        label='in-distribution'),
               matplotlib.patches.Patch(fc=C_NEAR, ec='0.25', lw=0.3,
                                        label='near-OOD (real held-out data)'),
               matplotlib.patches.Patch(fc=C_FAR, ec='0.25', lw=0.3,
                                        label='far-OOD (synthetic)')]
    sp_leg.legend(handles=handles, loc='center', ncol=3, fontsize=6.8, frameon=False,
                  handlelength=1.1, handletextpad=0.4, columnspacing=1.8, borderpad=0.0)

    # ── (d–h) per-model mean fingerprint per condition ───────────────────────
    # labels go in the inter-heatmap gaps (placed after freeze) so no cell space
    # is spent on titles and each heatmap stays tall enough to read its rows
    gsb = gs[4].subgridspec(len(Ds), 1, hspace=0.22)
    letters = 'defgh'
    heat_ax = []
    for i, (name, Dm) in enumerate(Ds):
        axh = fig.add_subplot(gsb[i, 0])
        _ood_cond_heatmap(axh, Dm, CMAP, C_ID, C_NEAR, C_FAR,
                          xlabel=(i == len(Ds) - 1))
        heat_ax.append((axh, f'({letters[i]}) {name}'))

    figstyle.freeze(fig)
    fig.text(0.006, sp_lbl.get_position().y1, '(d–h) mean fingerprint per condition '
             '— far-OOD concentrates on a sparse subset of factors',
             ha='left', va='top', fontsize=7.5, fontweight='bold')
    for axh, lab in heat_ax:
        p = axh.get_position()
        fig.text(p.x0, p.y1 + 0.004, lab, ha='left', va='bottom', fontsize=7.5,
                 fontweight='bold')
    return fig


def _fp_sim_matrix(ax, X, labels, names, colors, CMAP, *, coarse=None, per=40,
                   seed=0):
    """One model's pairwise-cosine class geometry: a balanced subsample of the
    in-distribution fingerprints sorted by class, its block-diagonal cosine
    matrix and the cosine silhouette (parity + digit for the parity models). The
    silhouette is computed on the full id set so it matches the native-dimension
    numbers reported elsewhere; only the drawn matrix is subsampled."""
    X = np.asarray(X, float); labels = np.asarray(labels)
    sil = cosine_silhouette(X, labels)
    sil_coarse = (cosine_silhouette(X, np.asarray(coarse)) if coarse is not None
                  else None)
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    sel = []
    for c in uniq:
        w = np.where(labels == c)[0]
        if len(w) > per:
            w = np.sort(rng.choice(w, per, replace=False))
        sel.append(w)
    sel = np.concatenate(sel)
    Xs = X[sel]; ys = labels[sel]
    U = unit(Xs)
    ax.imshow(U @ U.T, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest',
              aspect='auto')
    counts = np.array([int(np.sum(ys == c)) for c in uniq])
    cuts = np.cumsum(counts)
    for c_ in cuts[:-1]:
        ax.axhline(c_ - 0.5, color='white', lw=0.4)
        ax.axvline(c_ - 0.5, color='white', lw=0.4)
    ax.set_xticks([])
    ax.set_yticks(cuts - counts / 2); ax.set_yticklabels(names, fontsize=6)
    for t, c in zip(ax.get_yticklabels(), colors):
        t.set_color(c)
    ax.tick_params(length=0, pad=1)
    for s_ in ax.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    if coarse is not None:
        txt = f'silhouette\n{sil_coarse:.2f} parity\n{sil:.2f} digit'
    else:
        txt = f'silhouette {sil:.2f}'
    ax.text(0.955, 0.955, txt, transform=ax.transAxes, ha='right', va='top',
            fontsize=6, linespacing=1.15, color='0.15',
            bbox=dict(fc='white', ec='none', alpha=0.82, pad=1.0))
    ax.set_xlabel(f'{len(Xs)} stimuli', labelpad=1, fontsize=6.5)


def _repro_matrix(ax, S, names, colors, CMAP, xlabel):
    """Draw a class x class reproduction matrix (train vs test, or split A vs B)
    with white dividers, colored row names and the within/off-diagonal summary."""
    n = len(names)
    ax.imshow(S, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
    for e in range(1, n):
        ax.axhline(e - 0.5, color='white', lw=0.4)
        ax.axvline(e - 0.5, color='white', lw=0.4)
    ax.set_xticks([]); ax.set_yticks(range(n)); ax.set_yticklabels(names, fontsize=6)
    for t, c in zip(ax.get_yticklabels(), colors):
        t.set_color(c)
    ax.tick_params(length=0, pad=1)
    for s_ in ax.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    ax.set_xlabel(xlabel, labelpad=2, fontsize=6, color='0.35', linespacing=1.25)


def figfp_structure(D):
    """Message: everything about the fingerprint that is not OOD (merges the
    remaining panels of the former Figs. C/D/F/H/O; the NNLS round-trip is dropped
    — it is Fig. P). On every model the fingerprint is a class-structured code:
    its pairwise-cosine geometry is block-diagonal by class (a-e), and for the two
    models that ship a second sample the code reproduces across the train/test
    split (f) and an independent val split (g). Ignores its D argument; loads all
    five nb0N_fingerprints."""
    from src import figdata
    DD = {name: figdata.load(b) for name, b in _OOD_MODELS}
    C_BFT = figstyle.color('ours'); CMAP = ramp_cmap(C_BFT, 'bft')
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    DIGCOL = digit_colors(list(range(10)), C_EVEN, C_ODD)

    W = 6.975
    H = [0.16,   # 0 band-A label
         1.55,   # 1 band A: five class-geometry matrices
         0.28,   # 2 spacer
         0.16,   # 3 band-B label
         2.00]   # 4 band B: two reproduction matrices
    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='appendix',
                   height_to_width_ratio=sum(H) / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.03, w_pad=0.06, hspace=0.03,
                          wspace=0.10)
    gs = fig.add_gridspec(len(H), 1, height_ratios=H, hspace=0.0)
    sp_a = fig.add_subplot(gs[0]); sp_a.set_axis_off()
    sp_b = fig.add_subplot(gs[3]); sp_b.set_axis_off()

    # ── (a–e) class geometry, one matrix per model ───────────────────────────
    def digit_cfg(Dm, label_key, *, parity=False):
        labs = np.asarray(Dm['fp'][label_key])
        uniq = np.unique(labs)
        coarse = np.asarray(Dm['fp']['id_targets']) % 2 if parity else None
        return (Dm['fp']['id'], labs, [str(int(d)) for d in uniq],
                [DIGCOL[int(d)] for d in uniq], coarse)

    def class_cfg(Dm):
        labs = np.asarray(Dm['fp']['id_targets'])
        uniq = np.unique(labs); CLS = list(Dm['class_names'])
        return (Dm['fp']['id'], labs, [CLS[int(c)] for c in uniq],
                list(class_colors(range(len(uniq))).values()), None)

    GEO = [('(a) $8\\times4$ MLP',
            digit_cfg(DD['$8\\times4$ MLP'], 'id_digits', parity=True)),
           ('(b) digit MLP', digit_cfg(DD['digit MLP'], 'id_targets')),
           ('(c) CIFAR-10 CNN', class_cfg(DD['CIFAR-10 CNN'])),
           ('(d) TinyViT', digit_cfg(DD['TinyViT'], 'id_digits', parity=True)),
           ('(e) ImageNet', class_cfg(DD['ImageNet']))]
    gsa = gs[1].subgridspec(1, 5, wspace=0.16)
    for i, (title, (X, labs, names, cols, coarse)) in enumerate(GEO):
        ax = fig.add_subplot(gsa[0, i])
        _fp_sim_matrix(ax, X, labs, names, cols, CMAP, coarse=coarse)
        ax.set_title(title, fontsize=7.5, pad=3, fontweight='bold')

    # ── (f) CIFAR-10 train vs test, (g) ImageNet independent val split ───────
    gsb = gs[4].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)
    D3 = DD['CIFAR-10 CNN']; fp3 = D3['fp']
    CLS3 = list(D3['class_names']); n3 = int(D3['n_classes'])
    Ttr = unit(np.stack([fp3['train'][fp3['train_targets'] == c].mean(0)
                         for c in range(n3)]))
    Tte = unit(np.stack([fp3['id'][fp3['id_targets'] == c].mean(0)
                         for c in range(n3)]))
    S3 = Ttr @ Tte.T
    n_hit = int((S3.argmax(1) == np.arange(n3)).sum())
    ax_f = fig.add_subplot(gsb[0, 0])
    _repro_matrix(ax_f, S3, CLS3, list(class_colors(range(n3)).values()), CMAP,
                  'rows: train, columns: test\n'
                  f'diag {np.diag(S3).mean():.2f}, off-diag '
                  f'{S3[~np.eye(n3, dtype=bool)].mean():.2f}, {n_hit}/{n3} nearest')
    ax_f.set_title('(f) CIFAR-10: train vs test', fontsize=7.5, pad=3,
                   fontweight='bold')

    D5 = DD['ImageNet']; sc = D5['split_cross']
    M5 = np.asarray(sc['matrix']); bs = [int(b) for b in sc['block_sizes']]
    CLS5 = list(D5['class_names']); n5 = len(bs)
    edges = np.cumsum(bs); starts = np.concatenate([[0], edges])
    within = float(np.mean([M5[starts[i]:edges[i], starts[i]:edges[i]].mean()
                            for i in range(n5)]))
    off = float((M5.sum() - sum(M5[starts[i]:edges[i], starts[i]:edges[i]].sum()
                                for i in range(n5))) /
                (M5.size - sum(b * b for b in bs)))
    ax_g = fig.add_subplot(gsb[0, 1])
    ax_g.imshow(M5, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
    for e in edges[:-1]:
        ax_g.axhline(e - 0.5, color='white', lw=0.4)
        ax_g.axvline(e - 0.5, color='white', lw=0.4)
    ax_g.set_xticks([]); ax_g.set_yticks(edges - np.array(bs) / 2)
    ax_g.set_yticklabels(CLS5, fontsize=6)
    for t, c in zip(ax_g.get_yticklabels(), class_colors(range(n5)).values()):
        t.set_color(c)
    ax_g.tick_params(length=0, pad=1)
    for s_ in ax_g.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    ax_g.set_xlabel(f'val split A $\\times$ B, by category\n'
                    f'within {within:.2f}, across {off:.2f}', labelpad=2, fontsize=6,
                    color='0.35', linespacing=1.25)
    ax_g.set_title('(g) ImageNet: independent val split', fontsize=7.5, pad=3,
                   fontweight='bold')

    figstyle.freeze(fig)
    fig.text(0.006, sp_a.get_position().y0, '(a–e) class geometry: pairwise cosine '
             'of the fingerprints, sorted by class', ha='left', va='bottom',
             fontsize=7.5, fontweight='bold')
    fig.text(0.006, sp_b.get_position().y0, '(f,g) the code is a property of the '
             'class, not the sample', ha='left', va='bottom', fontsize=7.5,
             fontweight='bold')
    return fig


# ── registry: figure name -> (bundle, render function, save mode) ─────────────

FIGURES = {
    'fig2_mlp_circuits':   ('nb01_circuits', fig2_mlp_circuits, 'paper'),
    'figA_mlp_details':    ('nb01_circuits', figA_mlp_details, 'appendix'),
    'figB_digit_mlp_details':  ('nb02_circuits', figB_digit_mlp_details, 'appendix'),
    'fig4_fingerprints_main':  ('nb03_fingerprints', fig4_fingerprints_main, 'paper'),
    'fig6_cnn_circuits': ('nb03_circuits', fig6_cnn_circuits, 'paper'),
    'figE_cnn_details': ('nb03_circuits', figE_cnn_details, 'appendix'),
    'fig8_imagenet_circuits': ('nb05_circuits', fig8_imagenet_circuits, 'paper'),
    'figN_imagenet_details': ('nb05_circuits', figN_imagenet_details, 'appendix'),
    'figG_vit_circuits': ('nb04_circuits', figG_vit_circuits, 'appendix'),
    # cross-model validation: one full-page figure (merges the former figP/figQ/
    # figR); loads all five nb09 bundles internally, so the registry bundle is only
    # the render gate.
    'figP_validation': ('nb09_mlp_even_odd_validation', figP_validation, 'appendix'),
    # appendix fingerprint analysis: two cross-model figures (merge the former
    # figC/D/F/H/O + figS); each loads all five nb0N_fingerprints internally.
    'figfp_ood': ('nb01_fingerprints', figfp_ood, 'appendix'),
    'figfp_structure': ('nb01_fingerprints', figfp_structure, 'appendix'),
}
