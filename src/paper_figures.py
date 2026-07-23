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
             text, ha='left', va='bottom', color=color)


def bar_panel(ax, values, digit_order, digit_color, *, xticks=True,
              yticklabels=False, ylab=None, title=None, tcolor='k'):
    """Mini bar chart: per-digit share of one factor's stimulus loading."""
    ax.bar(np.arange(len(digit_order)), values,
           color=[digit_color[d] for d in digit_order],
           edgecolor='0.25', linewidth=0.3, width=0.8)
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
    NODES = nodes_by_path(D)
    n_classes = len(DIGIT_ORDER)

    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='paper',
                   height_to_width_ratio=0.545)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.18, 1.40, 0.82],
                          height_ratios=[1.0, 1.30], hspace=0.16)

    # ── (a) both circuits in one graph: shared units and output push-pull ────
    ax_a = fig.add_subplot(gs[0, 0])
    draw_scaffold_pair(ax_a, CIRCUITS, LAYER_SIZES, C_INH, OUT_LABELS)
    ax_a.set_title('(a) the two class circuits', pad=2, loc='left')

    S = np.stack([unit_support(NODES[(c['k'],)]) for c in CIRCUITS])
    ov, null = support_overlap(S)
    ax_a.text(0.5, -0.055, rf'$L_1$ overlap {ov:.2f} (shuffled {null.mean():.2f}, '
              rf'$p={max((null <= ov).mean(), 1 / len(null)):.3f}$)',
              transform=ax_a.transAxes, ha='center', va='top', fontsize=6,
              color='0.35')

    # ── (b) the top of the trace tree, as per-digit loadings ────────────────
    gsb = gs[0, 1].subgridspec(3, 7, height_ratios=[1.0, 0.24, 1.0])
    ax_b = [fig.add_subplot(gsb[0, 0:2]), fig.add_subplot(gsb[0, 4:6]),
            fig.add_subplot(gsb[2, 0:2]), fig.add_subplot(gsb[2, 2:4]),
            fig.add_subplot(gsb[2, 4:6])]
    fig.add_subplot(gsb[1, :]).set_axis_off()          # room for the tree connectors
    bp = dict(digit_order=DIGIT_ORDER, digit_color=DIGIT_COLOR)
    bar_panel(ax_b[0], CIRCUITS[0]['l3_profile'], xticks=False, ylab='loading',
              title=r'(b) $L_3$ even', tcolor=C_EVEN, **bp)
    bar_panel(ax_b[1], CIRCUITS[1]['l3_profile'], xticks=False,
              title=r'$L_3$ odd', tcolor=C_ODD, **bp)
    bar_panel(ax_b[2], CIRCUITS[0]['l2_profiles'][0], yticklabels=True,
              title=r'$L_2\ k_0$', tcolor=C_EVEN, **bp)
    bar_panel(ax_b[3], CIRCUITS[0]['l2_profiles'][1], title=r'$L_2\ k_1$',
              tcolor=C_EVEN, **bp)
    bar_panel(ax_b[4], CIRCUITS[1]['l2_profiles'][0], title=r'$L_2\ k_0$',
              tcolor=C_ODD, **bp)

    # ── (c) digit purity of every factor, layer by layer ─────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    rows = purity_by_layer(D)
    xs_of = {2: 0, 1: 1, 0: 2}                              # L3, L2, L1 left to right
    for li, pur, lam, ci in rows:
        col = CIRCUITS[0]['color'] if ci == CIRCUITS[0]['k'] else CIRCUITS[1]['color']
        ax_c.scatter(xs_of[li] + (0.11 if ci == CIRCUITS[1]['k'] else -0.11), pur,
                     s=4 + 26 * lam, color=col, alpha=0.85, edgecolors='0.25',
                     linewidths=0.3, zorder=3)
    def lam_weighted_purity(li):
        sel = [(p, l) for lay, p, l, _ in rows if lay == li]
        return sum(p * l for p, l in sel) / sum(l for _, l in sel)

    means = [lam_weighted_purity(li) for li in (2, 1, 0)]
    ax_c.plot([0, 1, 2], means, color='0.25', lw=0.9, marker='_', ms=9, zorder=4)
    for x, m, va, dy in zip([0, 1, 2], means, ('top', 'top', 'bottom'),
                            (-0.04, -0.04, 0.035)):
        ax_c.text(x - 0.34, m + dy, f'{m:.2f}', ha='left', va=va, fontsize=6,
                  color='0.25')
    ax_c.axhline(1 / n_classes, color='0.6', lw=0.5, ls=(0, (2.5, 2)))
    ax_c.text(2.42, 1 / n_classes, 'chance', ha='right', va='bottom', fontsize=6,
              color='0.5')
    ax_c.set_xlim(-0.45, 2.45); ax_c.set_ylim(0.15, 1.03)
    ax_c.set_xticks([0, 1, 2]); ax_c.set_xticklabels([r'$L_3$', r'$L_2$', r'$L_1$'])
    ax_c.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_c.set_ylabel('digit purity', labelpad=1)
    ax_c.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_c.spines[s].set_visible(False)
    ax_c.set_title('(c) class $\\rightarrow$ sub-class', pad=2, loc='left')

    # ── (d), (e) layer-1 sub-circuits: pixel arbor + per-digit loading ───────
    gs_bot = gs[1, :].subgridspec(2, 1, height_ratios=[0.13, 1.0])
    sp_de = fig.add_subplot(gs_bot[0]); sp_de.set_axis_off()   # room for the labels
    gsd = gs_bot[1].subgridspec(2, 11, height_ratios=[1.0, 0.45],
                                width_ratios=[1] * 5 + [0.4] + [1] * 5)
    img_axes = {}
    for ci, c in enumerate(CIRCUITS):
        base = 0 if ci == 0 else 6
        cmap = seq_cmap(c['color'], c['name'])
        img_axes[ci] = []
        for k, M in enumerate(c['l1_arbors']):
            axi = fig.add_subplot(gsd[0, base + k])
            show_map(axi, M, cmap)
            top = DIGIT_ORDER[int(np.argmax(c['l1_profiles'][k]))]
            axi.text(0.05, 0.97, rf'$k_{k}$', transform=axi.transAxes, ha='left',
                     va='top', color=c['color'], fontsize=6.5)
            axi.text(0.95, 0.97, f'{top}', transform=axi.transAxes, ha='right',
                     va='top', color='0.3', fontsize=6.5, fontweight='bold')
            img_axes[ci].append(axi)
            bar_panel(fig.add_subplot(gsd[1, base + k]), c['l1_profiles'][k],
                      xticks=(k == 0), ylab='loading' if k == 0 else None, **bp)

    # ── settle constrained_layout, then freeze and add cross-axes annotation ──
    figstyle.freeze(fig)          # draw once, then switch the layout engine off

    for src, dsts in ((ax_b[0], (ax_b[2], ax_b[3])), (ax_b[1], (ax_b[4],))):  # tree
        b0 = src.get_position()
        x0, ytop = b0.x0 + b0.width / 2, b0.y0 - 0.012
        for d in dsts:
            b1 = d.get_position()
            x1, ybot = b1.x0 + b1.width / 2, b1.y1 + 0.058
            ymid = (ytop + ybot) / 2
            fig.add_artist(matplotlib.lines.Line2D([x0, x0, x1, x1],
                                                   [ytop, ymid, ymid, ybot],
                                                   color='0.65', lw=0.4, zorder=0))

    for ci, c in enumerate(CIRCUITS):                                    # block labels
        fig.text(max(img_axes[ci][0].get_position().x0 - 0.008, 0.002),
                 sp_de.get_position().y0,
                 f"({'de'[ci]}) {c['name']} circuit — $L_1$ factors below $L_2\\,k_0$",
                 ha='left', va='bottom', color=c['color'])
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
                   height_to_width_ratio=0.60)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    # thin spacer rows reserve room for panel labels, which are placed after the
    # layout is frozen (a long label inside an axes would inflate its grid cell).
    gs = fig.add_gridspec(6, 1, height_ratios=[0.10, 0.62, 0.11, 0.62, 0.11, 0.74],
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
    for i, (name, lam, cols) in enumerate(nodes_a):
        ax = fig.add_subplot(gsa[0, i])
        ax.bar(np.arange(len(lam)), lam, color=cols[:len(lam)], edgecolor='0.25',
               linewidth=0.3, width=0.75)
        ax.set_xticks(np.arange(len(lam)))
        ax.set_xticklabels(np.arange(len(lam)))
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(['0', '', '1'] if i == 0 else [])
        ax.tick_params(length=1.5, pad=1)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.set_title(name, pad=1.5, fontsize=7)
        if i == 2:
            ax.set_xlabel('factor $k$', labelpad=1)
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
    ov, null = support_overlap(S)
    x = np.arange(S.shape[1])
    for i, c in enumerate(CIRCUITS):
        ax_c.bar(x + (i - 0.5) * 0.38, S[i], width=0.36, color=c['color'],
                 edgecolor='0.25', linewidth=0.3, label=f"{c['name']} circuit")
    ax_c.set_xticks(x); ax_c.set_xlabel('$L_1$ unit', labelpad=1)
    ax_c.set_ylabel('share of\ncircuit mass', labelpad=1)
    ax_c.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_c.spines[s].set_visible(False)
    ax_c.legend(loc='upper right', frameon=False, fontsize=6.0, handlelength=0.9,
                handletextpad=0.4, borderpad=0.1, labelspacing=0.15)
    ax_c.text(0.02, 0.98, rf'overlap {ov:.2f}' '\n'
              rf'shuffled {null.mean():.2f}$\pm${null.std():.2f}',
              transform=ax_c.transAxes, ha='left', va='top', fontsize=6.0,
              color='0.35')
    anchors['c'] = ax_c

    # ── (d) connection maps: excitatory vs inhibitory, L3 and L2 ─────────────
    gsd = gs[3].subgridspec(2, 5, width_ratios=[0.26] + [1] * 4)
    cols_d = [(rf"$L_3\ k_{{{CIRCUITS[0]['k']}}}$ (even)", CIRCUITS[0], 'l3_conn',
               C_EVEN),
              (rf"$L_3\ k_{{{CIRCUITS[1]['k']}}}$ (odd)", CIRCUITS[1], 'l3_conn',
               C_ODD),
              (r'$L_2$ even $k_0$', CIRCUITS[0], 'l2_conn', C_EVEN),
              (r'$L_2$ odd $k_0$',  CIRCUITS[1], 'l2_conn', C_ODD)]
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

    # ── (e, f) what drives each L1 factor, and what it suppresses ────────────
    gse = gs[5].subgridspec(2, 12, width_ratios=[0.42] + [1] * 5 + [0.4] + [1] * 5)
    for r, lab in enumerate(('weighted avg.\nstimulus', 'inhibitory\narbor')):
        row_label(fig, gse[r, 0], lab)
    first_img = {}
    for ci, c in enumerate(CIRCUITS):
        base = 1 if ci == 0 else 7
        for k, wavg in enumerate(c['l1_wavg']):
            ax = fig.add_subplot(gse[0, base + k])
            ax.imshow(wavg, cmap='gray_r', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            top = DIGIT_ORDER[int(np.argmax(c['l1_profiles'][k]))]
            ax.text(0.05, 0.97, rf'$k_{k}$', transform=ax.transAxes, ha='left',
                    va='top', color=c['color'], fontsize=6.5)
            ax.text(0.95, 0.97, f'{top}', transform=ax.transAxes, ha='right',
                    va='top', color='0.3', fontsize=6.5, fontweight='bold')
            if k == 0:
                first_img[ci] = ax
            show_map(fig.add_subplot(gse[1, base + k]), c['l1_neg_arbors'][k], CMAP_INH)

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
    # same layer-index shift sep_controls applies, so a layer is named the same
    # in the separability panel and in the embedding panel
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


def sep_controls(D):
    """Activation-control rows of a fingerprints bundle, as plot-ready tuples.

    The bundle names a hidden layer's activations one index higher than the circuit
    figures do (it counts the pixels as $L_1$), so a 784->40->20->10 net exports its
    40-unit layer as "$L_2$ act." while Fig. 3 draws that same layer as $L_1$. The
    figures are what the reader sees side by side, so the index is shifted down here.
    """
    out = []
    for s in D['sep']:
        lab = re.sub(r'\$L_(\d)\$', lambda m: f'$L_{int(m.group(1)) - 1}$', s['label'])
        out.append((lab, s['silhouette'], s['knn'], int(s['dim']),
                    resolve_color(s['color_key'])))
    return out


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


# ── Figure 4 — factor fingerprints in the 8x4 even/odd MLP ────────────────────

def fig4_mlp_fingerprints(D):
    """Message: the 13-d BFT fingerprint is a class-structured code that is more
    digit-separable than any raw activation layer, reports the network's own decision
    on digits the factorization never saw, and collapses on far-OOD input."""
    DIGIT_ORDER = list(D['digit_order'])
    COL_ORDER, N_EV = list(D['col_order']), int(D['n_ev'])
    COLS_EVEN, COLS_ODD = list(D['cols_even']), list(D['cols_odd'])
    dims = D['dims']
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    DIGIT_COLOR = digit_colors(DIGIT_ORDER, C_EVEN, C_ODD)
    CMAP_EVEN, CMAP_ODD = seq_cmap(C_EVEN, 'ev'), seq_cmap(C_ODD, 'od')
    CMAP_SIM = ramp_cmap(figstyle.color('ours'), 'sim')
    SIL_CLASS, SIL_DIGIT, R_OOD = D['sil_class'], D['sil_digit'], D['r_ood']
    COND = [(c['label'], c['values'], resolve_color(c['color_key'])) for c in D['cond']]
    SEP = sep_controls(D)

    def layer_groups(cols):
        """Contiguous runs of one layer within a column block -> (start, stop, layer)."""
        out, s = [], 0
        for i in range(1, len(cols) + 1):
            if i == len(cols) or dims[cols[i], 0] != dims[cols[s], 0]:
                out.append((s, i, int(dims[cols[s], 0])))
                s = i
        return out

    EMB = fp_vs_act(D, D['fp']['id_digits'])
    W, SP = 6.975, 0.13
    rows = [SP, 1.32, SP, 1.30]
    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='paper',
                   height_to_width_ratio=sum(rows) / W)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.02, wspace=0.03)
    gs = fig.add_gridspec(4, 1, height_ratios=rows, hspace=0.02)
    sp = fig.add_subplot(gs[0]); sp.set_axis_off()         # room for the panel labels
    sp2 = fig.add_subplot(gs[2]); sp2.set_axis_off()
    gs_top = gs[1].subgridspec(1, 3, width_ratios=[2.10, 1.0, 1.0])
    gs_bot = gs[3].subgridspec(1, 3, width_ratios=[1.20, 1.0, 1.06])

    # (a) mean fingerprint per digit, columns grouped by circuit then layer
    ax_a = fig.add_subplot(gs_top[0, 0])
    M = D['fp_mean_by_digit']
    M = M / M.sum(1, keepdims=True)                        # per-digit loading profile
    Mo = M[:, COL_ORDER]
    _col = np.arange(len(COL_ORDER))
    ax_a.imshow(np.where(_col < N_EV, Mo, np.nan), cmap=CMAP_EVEN, vmin=0, vmax=Mo.max(),
                aspect='auto', interpolation='nearest')
    ax_a.imshow(np.where(_col >= N_EV, Mo, np.nan), cmap=CMAP_ODD, vmin=0, vmax=Mo.max(),
                aspect='auto', interpolation='nearest')
    ax_a.axvline(N_EV - 0.5, color='0.25', lw=0.8)
    for off, cols in ((0, COLS_EVEN), (N_EV, COLS_ODD)):
        for s0, s1, lay in layer_groups(cols):
            if s0:
                ax_a.axvline(off + s0 - 0.5, color='0.75', lw=0.4)
            ax_a.text(off + (s0 + s1 - 1) / 2, len(DIGIT_ORDER) - 0.35,
                      rf'$L_{{{lay + 1}}}$', ha='center', va='top', fontsize=6.0)
    ax_a.set_xticks([])
    ax_a.set_yticks(range(len(DIGIT_ORDER))); ax_a.set_yticklabels(DIGIT_ORDER)
    for t, d in zip(ax_a.get_yticklabels(), DIGIT_ORDER):
        t.set_color(DIGIT_COLOR[d])
    ax_a.set_ylabel('stimulus digit', labelpad=1)
    ax_a.set_xlabel('factor, by circuit and layer', labelpad=2)
    ax_a.tick_params(length=1.5, pad=1)
    for s in ax_a.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_a.text((N_EV - 1) / 2, -0.72, 'even circuit', ha='center', va='bottom',
              color=C_EVEN, fontsize=6.5)
    ax_a.text((N_EV + len(COL_ORDER) - 1) / 2, -0.72, 'odd circuit', ha='center',
              va='bottom', color=C_ODD, fontsize=6.5)
    ax_a.set_ylim(len(DIGIT_ORDER) + 0.25, -1.1)

    # (b, c) the embedding, against the network's own representation
    ax_pf = fig.add_subplot(gs_top[0, 1])
    ax_pa = fig.add_subplot(gs_top[0, 2])
    if EMB is not None:
        Xf, lf, Xa, la, rep = EMB
        pca_panel(ax_pf, Xf, lf, DIGIT_COLOR, order=DIGIT_ORDER)
        pca_panel(ax_pa, Xa, la, DIGIT_COLOR, order=DIGIT_ORDER,
                  note=None if rep['aligned'] else 'independent sample')
    else:
        for ax in (ax_pf, ax_pa):
            _val_na(ax, 'no activation baseline\nin this bundle')

    # (d) is the fingerprint worth it? — digit separability against the raw
    #     activations of the same network, at 1/60 of the dimensions
    ax_b = fig.add_subplot(gs_bot[0, 0])
    y = np.arange(len(SEP))[::-1]
    ax_b.barh(y + 0.2, [s[1] for s in SEP], height=0.38, color=[s[4] for s in SEP],
              edgecolor='none')
    ax_b.barh(y - 0.2, [s[2] for s in SEP], height=0.38, color=[s[4] for s in SEP],
              edgecolor='none', alpha=0.45)
    _h = [matplotlib.patches.Patch(fc='0.5', ec='none', label='silhouette'),
          matplotlib.patches.Patch(fc='0.5', ec='none', alpha=0.45,
                                   label='5-NN accuracy')]
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([f"{s[0].replace(chr(10), ' ')} ({s[3]}d)" for s in SEP],
                         fontsize=6)
    ax_b.get_yticklabels()[0].set_color(SEP[0][4])
    ax_b.set_xlim(0, 1.04); ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_ylim(-0.6, len(SEP) - 0.05)
    ax_b.set_xlabel('digit separability', labelpad=1)
    ax_b.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_b.spines[s].set_visible(False)
    ax_b.legend(handles=_h, fontsize=6, frameon=False, loc='lower right',
                handlelength=0.8, handletextpad=0.35, borderpad=0.1, labelspacing=0.2,
                borderaxespad=0.0)

    # (e) held-out digits: the fingerprint tracks the network's decision, not the label
    ax_c = fig.add_subplot(gs_bot[0, 1])
    ax_c.plot([0, 1], [0, 1], color='0.7', lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    for d, x, y in D['id_pts']:
        ax_c.scatter(x, y, s=13, facecolor='none', edgecolor=DIGIT_COLOR[int(d)],
                     linewidths=0.8, zorder=2)
    LBL_OFF = {2: (3.4, -1.0), 5: (2.0, -6.4), 6: (3.4, -1.0),
               7: (-1.5, -6.4), 8: (3.4, -1.0), 9: (3.4, -1.0)}
    for d, x, y in D['ood_pts']:
        d = int(d)
        c = C_EVEN if d % 2 == 0 else C_ODD
        ax_c.scatter(x, y, s=15, color=c, edgecolor='none', zorder=3)
        ax_c.annotate(str(d), (x, y), textcoords='offset points',
                      xytext=LBL_OFF.get(d, (3.4, -1.0)), fontsize=6.5, color=c)
    ax_c.set_xlim(-0.06, 1.06); ax_c.set_ylim(-0.06, 1.06)
    ax_c.set_xticks([0, 0.5, 1]); ax_c.set_yticks([0, 0.5, 1])
    ax_c.set_xlabel('P(network says odd)', labelpad=1)
    ax_c.set_ylabel('odd-circuit share', labelpad=1)
    ax_c.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_c.spines[s].set_visible(False)
    ax_c.text(0.04, 0.97, rf'$r={R_OOD:.3f}$', transform=ax_c.transAxes,
              ha='left', va='top', fontsize=6.5)
    ax_c.scatter([], [], s=13, facecolor='none', edgecolor='0.45', linewidths=0.8,
                 label='trained digits')
    ax_c.scatter([], [], s=15, color='0.45', edgecolor='none', label='held-out digits')
    ax_c.legend(fontsize=6, frameon=False, loc='lower right', handlelength=0.9,
                handletextpad=0.3, borderpad=0.1, labelspacing=0.2, borderaxespad=0.2)

    # (f) far-OOD: every stimulus collapses onto one fingerprint
    ax_d = fig.add_subplot(gs_bot[0, 2])
    pos = np.arange(len(COND))[::-1]
    for p, (lab, vals, c) in zip(pos, COND):
        parts = ax_d.violinplot([vals], positions=[p], vert=False, widths=0.82,
                                showextrema=False, showmedians=False)
        for b in parts['bodies']:
            b.set_facecolor(c); b.set_edgecolor('none'); b.set_alpha(0.5)
        ax_d.scatter(np.median(vals), p, s=9, marker='D', color=c, zorder=3,
                     edgecolor='none')
    ax_d.set_yticks(pos); ax_d.set_yticklabels([c[0] for c in COND])
    for t, c in zip(ax_d.get_yticklabels(), COND):
        t.set_color(c[2])
    ax_d.set_xlim(0.42, 1.045); ax_d.set_xticks([0.5, 0.75, 1.0])
    ax_d.set_xlabel('cos. to condition mean', labelpad=1)
    ax_d.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_d.spines[s].set_visible(False)

    # ── settle constrained_layout, then freeze and add the panel labels ──────
    figstyle.freeze(fig)
    emb_lab = ((f"(b) fingerprint, {D['fp']['id'].shape[1]}-d",
                f"(c) {EMB[4]['label'].replace(chr(10), ' ')}, {EMB[4]['dim']}-d")
               if EMB is not None else ('(b) fingerprint', '(c) activations'))
    for ax, lab, anchor in ((ax_a, '(a) mean fingerprint', sp),
                            (ax_pf, emb_lab[0], sp), (ax_pa, emb_lab[1], sp),
                            (ax_b, '(d) digit separability', sp2),
                            (ax_c, '(e) held-out digits', sp2),
                            (ax_d, '(f) far-OOD collapse', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002),
                 anchor.get_position().y0, lab, ha='left', va='bottom')
    return fig


# ── Appendix C — fingerprint details for the 8x4 MLP ──────────────────────────

def figC_mlp_fingerprint_details(D):
    """Message: the fingerprint machinery behind Fig. 4 is well conditioned — the NNLS
    projection recovers the NMF's own fingerprint, every condition's mean fingerprint is
    readable, and the digit blocks that produce the Fig. 4(b) silhouette are visible."""
    DIGIT_ORDER = list(D['digit_order'])
    COL_ORDER, N_EV = list(D['col_order']), int(D['n_ev'])
    dims = D['dims']
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    C_BFT, C_NEAR, C_FAR = (figstyle.color('ours'), figstyle.color('near_ood'),
                            figstyle.color('far_ood'))
    DIGIT_COLOR = digit_colors(DIGIT_ORDER, C_EVEN, C_ODD)
    CMAP_BFT = ramp_cmap(C_BFT, 'bft')
    rt_sims = D['rt_sims']
    ROWS = [(r['label'], r['mean'], resolve_color(r['color_key'], DIGIT_COLOR))
            for r in D['rows']]
    GROUP = [(g['label'], int(g['start']), int(g['stop'])) for g in D['group']]
    LIKE = [(l['label'], l['values'], resolve_color(l['color_key'])) for l in D['like']]
    pca = D['pca']

    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='appendix',
                   height_to_width_ratio=0.80)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.03, wspace=0.03)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.10, 1.0, 0.10, 1.0], hspace=0.10)
    sp1 = fig.add_subplot(gs[0, :]); sp1.set_axis_off()
    sp2 = fig.add_subplot(gs[2, :]); sp2.set_axis_off()
    gs_bot = gs[3, :].subgridspec(1, 3, width_ratios=[1.20, 0.72, 1.08])

    # (a) NNLS round-trip: projecting held-out stimuli onto the fixed factors
    #     recovers the fingerprint the NMF itself produced
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.hist(rt_sims, bins=np.linspace(0.6, 1.0, 60), color=C_BFT, alpha=0.85, lw=0)
    ax_a.set_yscale('log')
    ax_a.axvline(rt_sims.mean(), color='0.25', lw=0.7, ls=(0, (2.5, 2)))
    ax_a.text(0.03, 0.97, f'mean {rt_sims.mean():.3f}\nmin {rt_sims.min():.3f}\n'
              rf'{np.mean(rt_sims > 0.99):.1%} above 0.99'.replace('%', r'\%'),
              transform=ax_a.transAxes, ha='left', va='top', fontsize=6.5,
              linespacing=1.25)
    ax_a.set_xlabel('cosine(NMF fingerprint, NNLS fingerprint)', labelpad=1)
    ax_a.set_ylabel('test stimuli', labelpad=1)
    ax_a.set_xlim(0.6, 1.005)
    ax_a.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_a.spines[s].set_visible(False)

    # (b) mean fingerprint of every stimulus condition — the full version of Fig. 4(a)
    ax_b = fig.add_subplot(gs[1, 1:])
    Mb = np.stack([m for _, m, _ in ROWS])
    Mb = Mb / Mb.sum(1, keepdims=True)
    ax_b.imshow(Mb[:, COL_ORDER], cmap=CMAP_BFT, aspect='auto', interpolation='nearest',
                norm=matplotlib.colors.PowerNorm(0.75, vmin=0, vmax=Mb.max()))
    ax_b.axvline(N_EV - 0.5, color='0.25', lw=0.8)
    for _, _s0, _s1 in GROUP[:-1]:
        ax_b.axhline(_s1 - 0.5, color='0.25', lw=0.8)
    ax_b.set_xticks(range(len(COL_ORDER)))
    ax_b.set_xticklabels([rf'$L_{{{dims[i, 0] + 1}}}k_{{{dims[i, 2]}}}$'
                          for i in COL_ORDER], rotation=90, fontsize=6)
    ax_b.set_yticks(range(len(ROWS)))
    ax_b.set_yticklabels([r[0] for r in ROWS], fontsize=6)
    for t, r in zip(ax_b.get_yticklabels(), ROWS):
        t.set_color(r[2])
    ax_b.tick_params(length=1.5, pad=1)
    for s in ax_b.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_b.text((N_EV - 1) / 2, -0.75, 'even circuit', ha='center', va='bottom',
              color=C_EVEN, fontsize=6.5)
    ax_b.text((N_EV + len(COL_ORDER) - 1) / 2, -0.75, 'odd circuit', ha='center',
              va='bottom', color=C_ODD, fontsize=6.5)
    ax_b.set_ylim(len(ROWS) - 0.5, -1.15)
    for lab, _s0, _s1 in GROUP:
        ax_b.text(1.012, (_s0 + _s1 - 1) / 2, lab, transform=ax_b.get_yaxis_transform(),
                  rotation=90, ha='left', va='center', fontsize=6.5, color='0.35')

    # (c) the same fingerprints in the plane of their first two principal
    #     components — where the held-out and far-OOD stimuli land
    ax_c = fig.add_subplot(gs_bot[0, 0])
    _fp = D['fp']
    cond_embedding(ax_c, _fp['id'], _fp['id_digits'], DIGIT_COLOR,
                   near=_fp['ood'], far=[f['F'] for f in _fp['far']],
                   c_near=C_NEAR, c_far=C_FAR, order=DIGIT_ORDER,
                   near_label='held-out digits', id_label='trained digits')

    # (d) pairwise fingerprint similarity, stimuli sorted by digit — the geometry
    #     behind the silhouette numbers of Fig. 4(b)
    ax_d = fig.add_subplot(gs_bot[0, 1])
    PER = int(D['sel_per_digit'])
    N_VIZ = len(D['fp_sel'])
    _U = unit(D['fp_sel'])
    ax_d.imshow(_U @ _U.T, cmap=CMAP_BFT, vmin=0, vmax=1, interpolation='nearest')
    for b in range(PER, N_VIZ, PER):
        ax_d.axhline(b - 0.5, color='0.25', lw=0.5)
        ax_d.axvline(b - 0.5, color='0.25', lw=0.5)
    ctr = (np.arange(len(DIGIT_ORDER)) + 0.5) * PER
    ax_d.set_xticks(ctr); ax_d.set_xticklabels(DIGIT_ORDER)
    ax_d.set_yticks(ctr); ax_d.set_yticklabels(DIGIT_ORDER)
    for tt in (ax_d.get_xticklabels(), ax_d.get_yticklabels()):
        for t, d in zip(tt, DIGIT_ORDER):
            t.set_color(DIGIT_COLOR[d])
    ax_d.tick_params(length=0, pad=1)
    for s in ax_d.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_d.set_xlabel('stimulus digit', labelpad=1)
    ax_d.text(0.96, 0.96,
              f"silhouette\n{D['sil_class']:.2f} class\n{D['sil_digit']:.2f} digit",
              transform=ax_d.transAxes, ha='right', va='top', fontsize=6,
              linespacing=1.15, color='0.15',
              bbox=dict(fc='white', ec='none', alpha=0.82, pad=1.0))

    # (e) how close each condition gets to any trained digit
    ax_e = fig.add_subplot(gs_bot[0, 2])
    pos = np.arange(len(LIKE))[::-1]
    for p, (lab, vals, c) in zip(pos, LIKE):
        parts = ax_e.violinplot([vals], positions=[p], vert=False, widths=0.82,
                                showextrema=False, showmedians=False)
        for b in parts['bodies']:
            b.set_facecolor(c); b.set_edgecolor('none'); b.set_alpha(0.5)
        ax_e.scatter(np.median(vals), p, s=9, marker='D', color=c, zorder=3,
                     edgecolor='none')
    ax_e.set_yticks(pos); ax_e.set_yticklabels([l[0] for l in LIKE])
    for t, l in zip(ax_e.get_yticklabels(), LIKE):
        t.set_color(l[2])
    ax_e.set_xlabel('max cosine to a trained digit', labelpad=1)
    ax_e.set_xlim(0.42, 1.03); ax_e.set_xticks([0.5, 0.75, 1.0])
    ax_e.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_e.spines[s].set_visible(False)

    figstyle.freeze(fig)
    for ax, lab, anchor in ((ax_a, '(a) NNLS round-trip', sp1),
                            (ax_b, '(b) mean fingerprint per condition', sp1),
                            (ax_c, '(c) fingerprint PCA', sp2),
                            (ax_d, '(d) fingerprint similarity', sp2),
                            (ax_e, '(e) distance to the trained digits', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), anchor.get_position().y0,
                 lab, ha='left', va='bottom')
    return fig


# ── Figure 3 — class circuits in the 40x20 digit MLP ──────────────────────────

def fig3_digit_mlp_circuits(D):
    """Message: the ten-class net does not factorize into digits at the output — it
    factorizes into seven overlapping, distributed circuits that each pool several
    digits, and tracing a circuit backward un-pools it into digit-selective factors."""
    CIRCUITS = D['circuits']
    n_c, N_SHOW, N_DIGITS = len(CIRCUITS), int(D['n_show']), int(D['n_digits'])
    LAYER_SIZES = list(D['layer_sizes'])
    Sup, root_pur = D['support'], D['root_pur']
    NODES = nodes_by_path(D)
    C_BFT = figstyle.color('ours')
    C_L3, C_L1 = C_BFT, tint(C_BFT, 0.45)
    CMAP = seq_cmap(C_BFT, 'bft')

    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='paper',
                   height_to_width_ratio=0.88)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.72, 1.62, 0.06], hspace=0.22)
    gs_top = gs[0].subgridspec(1, 3, width_ratios=[1.20, 0.86, 1.26])
    fig.add_subplot(gs[2]).set_axis_off()   # room for the bottom row's tick labels

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
    ax_a.set_title('(a) output-layer factors', loc='left', pad=2)
    cb = fig.colorbar(im, ax=ax_a, fraction=0.038, pad=0.03)
    cb.set_label('digit share', labelpad=1)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1)

    # ── (b) un-pooling: each circuit's purest factor gets purer toward the input,
    #        while the spread over all its factors (grey) stays wide ─────────────
    ax_b = fig.add_subplot(gs_top[0, 1])
    layers = [(2, root_pur[None, :].T), (1, None), (0, None)]
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
    ax_b.set_title('(b) un-pooling', loc='left', pad=2)

    # how many of the pooled digits actually get a dedicated L1 factor, against the
    # matched control of scoring a circuit's digits with another circuit's factors
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
    ax_c.set_title('(c) $L_1$ units per circuit', loc='left', pad=2)
    ov, null = support_overlap(Sup)
    pr = 1 / ((Sup / Sup.sum(1, keepdims=True)) ** 2).sum(1)
    ax_c.text(0.5, -0.30, f'{pr.mean():.0f} of {LAYER_SIZES[0]} units per circuit; '
              f'pairwise overlap {ov:.2f} (shuffled {null.mean():.2f})',
              transform=ax_c.transAxes, ha='center', va='top', fontsize=6,
              color='0.35')

    # ── (d) the un-pooled sub-circuits themselves, in pixel space ─────────────
    gs_bot = gs[1].subgridspec(2, 1, height_ratios=[0.12, 1.0])
    sp_d = fig.add_subplot(gs_bot[0]); sp_d.set_axis_off()   # room for the (d) label
    gsd = gs_bot[1].subgridspec(N_SHOW, n_c + 1, width_ratios=[0.30] + [1] * n_c)
    for r in range(N_SHOW):
        row_label(fig, gsd[r, 0], f'{["1st", "2nd", "3rd"][r]}\npooled\ndigit')
    arbor_axes = []
    for j, c in enumerate(CIRCUITS):
        P = c['l1_profiles']
        top = np.argsort(-c['profile'])[:N_SHOW]     # the digits this factor pools
        for r, d in enumerate(top):
            k = int(P[:, d].argmax())                # its best detector for that digit
            ax = fig.add_subplot(gsd[r, j + 1])
            M = c['l1_arbors'][k]
            ax.imshow(M, cmap=CMAP, interpolation='nearest',
                      norm=matplotlib.colors.PowerNorm(0.62, vmin=0,
                                                       vmax=np.percentile(M, 99.3)))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            ax.text(0.05, 0.97, rf'$k_{{{k}}}$', transform=ax.transAxes, ha='left',
                    va='top', fontsize=6.5, color=C_BFT)
            ax.text(0.95, 0.97, f'{int(d)}', transform=ax.transAxes, ha='right',
                    va='top', fontsize=7, color='0.15', fontweight='bold')
            if r == 0:
                ax.set_title(r'$f_{' + str(c['k']) + r'}$ pools ' +
                             ','.join(str(int(t)) for t in top), pad=2, fontsize=7)
                arbor_axes.append(ax)

    figstyle.freeze(fig)
    fig.text(max(arbor_axes[0].get_position().x0 - 0.036, 0.002),
             sp_d.get_position().y0,
             r'(d) the $L_1$ factor that detects each pooled digit '
             r'(bold: the digit; $k$: which factor of the circuit)',
             ha='left', va='bottom')
    return fig


# ── Appendix B — decomposition details for the 40x20 digit MLP ────────────────

def figB_digit_mlp_details(D):
    """Message: the Fig. 3 decomposition rests on a graded spectrum per node, circuits
    that span the whole network, and the digit profile of *every* L1 factor — not only
    the ones the main figure has room for."""
    CIRCUITS = D['circuits']
    n_c, N_SHOW = len(CIRCUITS), int(D['n_show'])
    C_BFT, C_INH = figstyle.color('ours'), figstyle.color('inhibitory')
    CMAP = seq_cmap(C_BFT, 'bft')

    anchors = {}
    figstyle.apply(venue='aaai2024', width='full', nrows=3, ncols=3, mode='appendix',
                   height_to_width_ratio=0.72)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    # thin spacer rows reserve room for the panel labels added after the freeze
    gs = fig.add_gridspec(6, 1, height_ratios=[0.10, 0.55, 0.10, 0.85, 0.11, 1.0],
                          hspace=0.03)
    gs_bot = gs[5].subgridspec(1, 2, width_ratios=[1.0, 1.0])
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4)]
    for sp in spacers:
        sp.set_axis_off()

    # ── (a) informativity spectra of every node of the trace ────────────────
    gsa = gs[1].subgridspec(1, 1 + n_c)
    ax = fig.add_subplot(gsa[0, 0])
    lam = D['root_lam']
    ax.bar(range(len(lam)), lam, color=C_BFT, edgecolor='0.25', linewidth=0.3,
           width=0.75)
    ax.set_title('$L_3$ (output)', pad=1.5, fontsize=7)
    ax.set_ylabel(r'$\lambda$ share', labelpad=1)
    anchors['a'] = ax
    axes_a = [ax]
    for j, c in enumerate(CIRCUITS):
        axj = fig.add_subplot(gsa[0, j + 1], sharey=ax)
        lam = c['l1_lam']
        axj.bar(range(len(lam)), lam, color=tint(C_BFT, 0.45), edgecolor='0.25',
                linewidth=0.3, width=0.75)
        axj.set_title(rf"$L_1$ of $f_{c['k']}$", pad=1.5, fontsize=7)
        axes_a.append(axj)
    for i, axj in enumerate(axes_a):
        axj.set_ylim(0, 0.65)
        axj.set_yticks([0, 0.5])
        axj.set_yticklabels(['0', '.5'] if i == 0 else [])
        axj.set_xticks([0, 5] if i else [0, 3, 6])
        axj.tick_params(length=1.5, pad=1)
        if i == len(axes_a) // 2:
            axj.set_xlabel('factor $k$', labelpad=1)
        for s in ('top', 'right'):
            axj.spines[s].set_visible(False)

    # ── (b) the circuit of every output factor, through the full network ────
    gsb = gs[3].subgridspec(1, n_c)
    for j, c in enumerate(CIRCUITS):
        ax = fig.add_subplot(gsb[0, j])
        sc = c['scaffold']
        draw_scaffold_backbone(ax, sc['edges'], sc['neg_edges'], sc['loading'],
                               list(sc['layer_sizes']), C_BFT, C_INH)
        top = np.argsort(-c['profile'])[:N_SHOW]
        ax.set_title(rf"$f_{c['k']}$: " + ','.join(str(int(d)) for d in top),
                     pad=1.5, fontsize=7)
        if j == 0:
            anchors['b'] = ax

    # ── (c) digit profile of every layer-1 factor, per circuit ──────────────
    gsc = gs_bot[0, 0].subgridspec(1, 2 * n_c - 1,
                                   width_ratios=[1, 0.22] * (n_c - 1) + [1])
    for j, c in enumerate(CIRCUITS):
        ax = fig.add_subplot(gsc[0, 2 * j])
        P = c['l1_profiles']
        ax.imshow(P, cmap=CMAP, aspect='auto', vmin=0, vmax=P.max())
        ax.set_xticks([0, 9]); ax.set_xticklabels([0, 9])
        ax.set_yticks(range(0, P.shape[0], 3))
        ax.set_yticklabels(range(0, P.shape[0], 3) if j == 0 else [])
        ax.tick_params(length=1.5, pad=1)
        for s in ax.spines.values():
            s.set_color('0.6'); s.set_linewidth(0.4)
        ax.set_title(rf"$f_{c['k']}$", pad=1.5, fontsize=7)
        if j == 0:
            ax.set_xlabel('digit', labelpad=1)
            ax.set_ylabel('$L_1$ factor $k$', labelpad=1)
            anchors['c'] = ax

    # ── (d) stimulus-weighted average input for the arbors of the main figure
    gsd = gs_bot[0, 1].subgridspec(N_SHOW, 1 + n_c, width_ratios=[0.42] + [1] * n_c)
    for r in range(N_SHOW):
        row_label(fig, gsd[r, 0], f'{["1st", "2nd", "3rd"][r]}\npooled digit')
    for j, c in enumerate(CIRCUITS):
        P = c['l1_profiles']
        top = np.argsort(-c['profile'])[:N_SHOW]
        for r, d in enumerate(top):
            k = int(P[:, d].argmax())            # same factor Fig. 3(d) draws
            ax = fig.add_subplot(gsd[r, j + 1])
            ax.imshow(c['l1_wavg'][k], cmap='gray_r', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(rf"$f_{c['k']}$", pad=1.5, fontsize=7)
                if j == 0:
                    anchors['d'] = ax

    figstyle.freeze(fig)

    def _label(key, spacer, text, dx=0.0):
        x = max(anchors[key].get_position().x0 + dx, 0.002)   # keep inside the canvas
        fig.text(x, spacers[spacer].get_position().y0, text,
                 ha='left', va='bottom', fontsize=7)

    _label('a', 0, '(a) informativity spectra', dx=-0.030)
    _label('b', 1, '(b) traced circuit of each output factor '
                   '(nodes: 40 $L_1$ / 20 $L_2$ / 10 output units; '
                   "solid excitatory, dashed inhibitory; each unit's "
                   'strongest input)', dx=0.0)
    _label('c', 2, '(c) digit profile of every $L_1$ factor', dx=-0.030)
    _label('d', 2, '(d) weighted-average input of the $L_1$ factors of Fig. 3(d)',
           dx=-0.030)
    return fig


# ── Figure 5 — factor fingerprints in the 40x20 digit MLP ─────────────────────

def fig5_digit_mlp_fingerprints(D):
    """Message: at ten classes the fingerprint stays class-structured and still reports
    the network's own decision on Fashion-MNIST, but it is no longer the most separable
    code — the 20-d penultimate activations are. It buys traceability, not separability."""
    N_CLASSES = int(D['n_classes'])
    COL_ORDER, BLK_EDGE = list(D['col_order']), list(D['blk_edge'])
    BLOCK_SIZES = list(D['block_sizes'])
    C_BFT, C_NEAR = figstyle.color('ours'), figstyle.color('near_ood')
    CMAP = ramp_cmap(C_BFT, 'bft')
    SIL, R_OOD, AGREE = D['sil'], D['r_ood'], D['agree']
    P_model, P_fprint = D['p_model'], D['p_fprint']
    SEP = sep_controls(D)
    BLABELS = D['block_labels']
    COND = [(c['label'], c['values'], resolve_color(c['color_key'])) for c in D['cond']]
    n_factors = int(D['n_factors'])
    DCOL = class_colors(range(N_CLASSES))
    EMB = fp_vs_act(D, D['fp']['id_targets'])

    W, SP = 6.975, 0.13
    rows = [SP, 1.32, SP, 1.30]
    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='paper',
                   height_to_width_ratio=sum(rows) / W)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.02, wspace=0.03)
    gs = fig.add_gridspec(4, 1, height_ratios=rows, hspace=0.02)
    sp = fig.add_subplot(gs[0]); sp.set_axis_off()         # room for the panel labels
    sp2 = fig.add_subplot(gs[2]); sp2.set_axis_off()
    gs_top = gs[1].subgridspec(1, 3, width_ratios=[2.10, 1.0, 1.0])
    gs_bot = gs[3].subgridspec(1, 3, width_ratios=[1.20, 1.0, 1.06])

    # (a) mean fingerprint per digit, columns grouped by circuit
    ax_a = fig.add_subplot(gs_top[0, 0])
    M = D['fp_mean_by_digit']
    M = M / M.sum(1, keepdims=True)                        # per-digit loading profile
    ax_a.imshow(M[:, COL_ORDER], cmap=CMAP, aspect='auto', interpolation='nearest',
                norm=matplotlib.colors.PowerNorm(0.7, vmin=0, vmax=M.max()))
    for e in BLK_EDGE:
        ax_a.axvline(e - 0.5, color='0.25', lw=0.7)
    for bi, blen in enumerate(BLOCK_SIZES):
        s0 = int(np.concatenate([[0], BLK_EDGE])[bi])
        ax_a.text(s0 + blen / 2 - 0.5, N_CLASSES - 0.3, rf'$f_{bi}$', ha='center',
                  va='top', fontsize=6.5, color=C_BFT)
    ax_a.set_xticks([])
    ax_a.set_yticks(range(N_CLASSES)); ax_a.set_yticklabels(range(N_CLASSES))
    for t_, c_ in zip(ax_a.get_yticklabels(), range(N_CLASSES)):
        t_.set_color(DCOL[c_])                        # the legend for (b) and (c)
    ax_a.set_ylabel('stimulus digit', labelpad=1)
    ax_a.set_xlabel(f'{n_factors} factors, grouped by circuit', labelpad=7)
    ax_a.tick_params(length=1.5, pad=1)
    for s in ax_a.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_a.set_ylim(N_CLASSES + 0.55, -0.5)

    # (b, c) the embedding, against the network's own representation
    ax_pf = fig.add_subplot(gs_top[0, 1])
    ax_pa = fig.add_subplot(gs_top[0, 2])
    if EMB is not None:
        Xf, lf, Xa, la, rep = EMB
        pca_panel(ax_pf, Xf, lf, DCOL)
        pca_panel(ax_pa, Xa, la, DCOL,
                  note=None if rep['aligned'] else 'independent sample')
    else:
        for ax in (ax_pf, ax_pa):
            _val_na(ax, 'no activation baseline\nin this bundle')

    # (d) is the fingerprint worth it? — at ten classes, not on separability: the
    #     20-d penultimate activations beat it. Reported as it is, not spun.
    ax_b = fig.add_subplot(gs_bot[0, 0])
    y = np.arange(len(SEP))[::-1]
    ax_b.barh(y + 0.2, [s[1] for s in SEP], height=0.38, color=[s[4] for s in SEP],
              edgecolor='none')
    ax_b.barh(y - 0.2, [s[2] for s in SEP], height=0.38, color=[s[4] for s in SEP],
              edgecolor='none', alpha=0.45)
    _h = [matplotlib.patches.Patch(fc='0.5', ec='none', label='silhouette'),
          matplotlib.patches.Patch(fc='0.5', ec='none', alpha=0.45,
                                   label='5-NN accuracy')]
    ax_b.set_yticks(y)
    ax_b.set_yticklabels([f"{s[0].replace(chr(10), ' ')} ({s[3]}d)" for s in SEP],
                         fontsize=6)
    ax_b.get_yticklabels()[0].set_color(SEP[0][4])
    ax_b.set_xlim(0, 1.04); ax_b.set_xticks([0, 0.5, 1.0])
    ax_b.set_ylim(-0.6, len(SEP) - 0.05)
    ax_b.set_xlabel('digit separability', labelpad=1)
    ax_b.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_b.spines[s_].set_visible(False)
    ax_b.legend(handles=_h, fontsize=6, frameon=False, loc='lower right',
                handlelength=0.8, handletextpad=0.35, borderpad=0.1, labelspacing=0.2,
                borderaxespad=0.0)

    # (e) Fashion-MNIST: the fingerprint reports the network's own (wrong) digit
    ax_c = fig.add_subplot(gs_bot[0, 1])
    ax_c.plot([0, 1], [0, 1], color='0.7', lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    ax_c.scatter(P_model.ravel(), P_fprint.ravel(), s=8, color=C_NEAR,
                 edgecolor='none', alpha=0.85, zorder=2)
    ax_c.set_xlim(-0.06, 1.06); ax_c.set_ylim(-0.06, 1.06)
    ax_c.set_xticks([0, 0.5, 1]); ax_c.set_yticks([0, 0.5, 1])
    ax_c.set_xlabel('P(network says digit $d$)', labelpad=1)
    ax_c.set_ylabel('P(fingerprint says digit $d$)', labelpad=1)
    ax_c.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_c.spines[s].set_visible(False)
    ax_c.text(0.04, 0.97, f'$r={R_OOD:.2f}$\nagree {AGREE:.2f}',
              transform=ax_c.transAxes, ha='left', va='top', fontsize=6.5,
              linespacing=1.2)
    # what the agreement is agreement *about*: the network funnels clothing into a
    # couple of digits, and the fingerprint names the same ones
    pref = P_model.argmax(1)
    top = np.argsort(-np.bincount(pref, minlength=N_CLASSES))
    parts = [f'{int((pref == d).sum())}' r'$\to$' f'{d}' for d in top
             if (pref == d).sum()]
    ax_c.text(0.04, 0.79, 'clothing:\n' + ', '.join(parts),
              transform=ax_c.transAxes, ha='left', va='top', fontsize=6,
              color=C_NEAR, linespacing=1.2)

    # (f) far-OOD: every stimulus collapses onto one fingerprint
    ax_d = fig.add_subplot(gs_bot[0, 2])
    pos = np.arange(len(COND))[::-1]
    for p, (lab, vals, c) in zip(pos, COND):
        parts = ax_d.violinplot([vals], positions=[p], vert=False, widths=0.82,
                                showextrema=False, showmedians=False)
        for b in parts['bodies']:
            b.set_facecolor(c); b.set_edgecolor('none'); b.set_alpha(0.5)
        ax_d.scatter(np.median(vals), p, s=9, marker='D', color=c, zorder=3,
                     edgecolor='none')
    ax_d.set_yticks(pos); ax_d.set_yticklabels([c[0] for c in COND])
    for t, c in zip(ax_d.get_yticklabels(), COND):
        t.set_color(c[2])
    ax_d.set_xlim(0.42, 1.045); ax_d.set_xticks([0.5, 0.75, 1.0])
    ax_d.set_xlabel('cos. to condition mean', labelpad=1)
    ax_d.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_d.spines[s].set_visible(False)

    # ── settle constrained_layout, then freeze and add the panel labels ──────
    figstyle.freeze(fig)
    emb_lab = ((f'(b) fingerprint, {n_factors}-d',
                f"(c) {EMB[4]['label'].replace(chr(10), ' ')}, {EMB[4]['dim']}-d")
               if EMB is not None else ('(b) fingerprint', '(c) activations'))
    for ax, lab, anchor in ((ax_a, '(a) mean fingerprint', sp),
                            (ax_pf, emb_lab[0], sp), (ax_pa, emb_lab[1], sp),
                            (ax_b, '(d) digit separability', sp2),
                            (ax_c, '(e) Fashion-MNIST', sp2),
                            (ax_d, '(f) far-OOD collapse', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002),
                 anchor.get_position().y0, lab, ha='left', va='bottom')
    return fig


# ── Appendix D — fingerprint details for the 40x20 digit MLP ──────────────────

def figD_digit_mlp_fingerprint_details(D):
    """Message: details for Fig. 5 — the NNLS round-trip is looser here than in the
    8x4 net (125 factors, mean cosine 0.92), every condition's mean fingerprint is
    readable, and Fashion-MNIST falls *inside* the digit geometry rather than outside."""
    N_CLASSES = int(D['n_classes'])
    COL_ORDER, BLK_EDGE = list(D['col_order']), list(D['blk_edge'])
    BLOCK_SIZES = list(D['block_sizes'])
    C_BFT = figstyle.color('ours')
    CMAP = ramp_cmap(C_BFT, 'bft')
    rt_sims, N_BLOCK = D['rt_sims'], int(D['n_block'])
    _Ub = unit(D['block_fp'])          # block fingerprints -> cross-similarity
    S_blk = _Ub @ _Ub.T
    n_factors = int(D['n_factors'])
    ROWS = [(r['label'], r['mean'], resolve_color(r['color_key'])) for r in D['rows']]
    GROUP = [(g['label'], int(g['start']), int(g['stop'])) for g in D['group']]
    LIKE = [(l['label'], l['values'], resolve_color(l['color_key'])) for l in D['like']]
    blabels = D['block_labels']
    bcolors = [resolve_color(k) for k in D['block_color_keys']]
    n_blocks = len(blabels)

    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='appendix',
                   height_to_width_ratio=0.86)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.03, wspace=0.03)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.09, 1.0, 0.09, 1.0], hspace=0.10)
    sp1 = fig.add_subplot(gs[0, :]); sp1.set_axis_off()
    sp2 = fig.add_subplot(gs[2, :]); sp2.set_axis_off()
    gs_bot = gs[3, :].subgridspec(1, 4, width_ratios=[1.00, 1.14, 0.68, 1.02])

    # (a) NNLS round-trip: projecting stimuli onto the fixed factors recovers the
    #     fingerprint the NMF itself produced
    ax_a = fig.add_subplot(gs[1, 0])
    ax_a.hist(rt_sims, bins=np.linspace(0.5, 1.0, 60), color=C_BFT, alpha=0.85, lw=0)
    ax_a.set_yscale('log')
    ax_a.axvline(rt_sims.mean(), color='0.25', lw=0.7, ls=(0, (2.5, 2)))
    ax_a.text(0.03, 0.97, f'mean {rt_sims.mean():.3f}\nmin {rt_sims.min():.3f}\n'
              rf'{np.mean(rt_sims > 0.95):.0%} above 0.95'.replace('%', r'\%'),
              transform=ax_a.transAxes, ha='left', va='top', fontsize=6.5,
              linespacing=1.25)
    ax_a.set_xlabel('cosine(NMF fingerprint, NNLS fingerprint)', labelpad=1)
    ax_a.set_ylabel('test stimuli', labelpad=1)
    ax_a.set_xlim(0.5, 1.005)
    ax_a.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_a.spines[s].set_visible(False)

    # (b) mean fingerprint of every stimulus condition — full version of Fig. 5(a)
    ax_b = fig.add_subplot(gs[1, 1:])
    Mb = np.stack([m for _, m, _ in ROWS])
    Mb = Mb / Mb.sum(1, keepdims=True)
    ax_b.imshow(Mb[:, COL_ORDER], cmap=CMAP, aspect='auto', interpolation='nearest',
                norm=matplotlib.colors.PowerNorm(0.7, vmin=0, vmax=Mb.max()))
    for e in BLK_EDGE:
        ax_b.axvline(e - 0.5, color='0.25', lw=0.7)
    ax_b.axhline(N_CLASSES - 0.5, color='0.25', lw=0.8)
    for bi, blen in enumerate(BLOCK_SIZES):
        s0 = int(np.concatenate([[0], BLK_EDGE])[bi])
        ax_b.text(s0 + blen / 2 - 0.5, -0.75, rf'$f_{bi}$', ha='center', va='bottom',
                  fontsize=6.5, color=C_BFT)
    ax_b.set_xticks([])
    ax_b.set_yticks(range(len(ROWS)))
    ax_b.set_yticklabels([r[0] for r in ROWS], fontsize=6)
    for t, r in zip(ax_b.get_yticklabels(), ROWS):
        t.set_color(r[2])
    ax_b.set_xlabel(f'{n_factors} factors, grouped by circuit', labelpad=2)
    ax_b.tick_params(length=1.5, pad=1)
    for s in ax_b.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_b.set_ylim(len(ROWS) - 0.5, -1.15)
    for lab, _s0, _s1 in GROUP:
        ax_b.text(1.008, (_s0 + _s1 - 1) / 2, lab, transform=ax_b.get_yaxis_transform(),
                  rotation=90, ha='left', va='center', fontsize=6.5, color='0.35')

    # (c) ID digits vs Fashion-MNIST classes, block cross-similarity
    # (c) where Fashion-MNIST and the far-OOD stimuli land in the digit plane
    ax_emb = fig.add_subplot(gs_bot[0, 0])
    _fp = D['fp']
    cond_embedding(ax_emb, _fp['id'], _fp['id_targets'], class_colors(range(N_CLASSES)),
                   near=_fp['ood'], far=[f['F'] for f in _fp['far']],
                   c_near=figstyle.color('near_ood'), c_far=figstyle.color('far_ood'),
                   near_label='Fashion-MNIST', id_label='digits')

    ax_c = fig.add_subplot(gs_bot[0, 1])
    ax_c.imshow(S_blk, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    for b in range(N_BLOCK, N_BLOCK * n_blocks, N_BLOCK):
        lw = 0.8 if b == N_BLOCK * N_CLASSES else 0.25
        ax_c.axhline(b - 0.5, color='0.3', lw=lw)
        ax_c.axvline(b - 0.5, color='0.3', lw=lw)
    ctr = (np.arange(n_blocks) + 0.5) * N_BLOCK
    ax_c.set_xticks(ctr[:N_CLASSES])
    ax_c.set_xticklabels(range(N_CLASSES), fontsize=6)
    ax_c.set_yticks(ctr); ax_c.set_yticklabels(blabels, fontsize=6)
    for t, c in zip(ax_c.get_yticklabels(), bcolors):
        t.set_color(c)
    ax_c.tick_params(length=0, pad=1)
    for s in ax_c.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_c.set_xlabel('ID digit', labelpad=1)

    # (d) pairwise fingerprint similarity, stimuli sorted by digit — the geometry
    #     behind the silhouette of Fig. 5(b)
    ax_d = fig.add_subplot(gs_bot[0, 2])
    PER = int(D['sel_per_digit'])
    _U = unit(D['fp_sel'])
    ax_d.imshow(_U @ _U.T, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    for b in range(PER, PER * N_CLASSES, PER):
        ax_d.axhline(b - 0.5, color='0.35', lw=0.3)
        ax_d.axvline(b - 0.5, color='0.35', lw=0.3)
    ctr = (np.arange(N_CLASSES) + 0.5) * PER
    ax_d.set_xticks(ctr); ax_d.set_xticklabels(range(N_CLASSES), fontsize=6)
    ax_d.set_yticks(ctr); ax_d.set_yticklabels(range(N_CLASSES), fontsize=6)
    ax_d.tick_params(length=0, pad=1)
    for sp_ in ax_d.spines.values():
        sp_.set_color('0.6'); sp_.set_linewidth(0.4)
    ax_d.set_xlabel('stimulus digit', labelpad=1)
    ax_d.text(0.96, 0.96, f"silhouette {D['sil']:.2f}", transform=ax_d.transAxes,
              ha='right', va='top', fontsize=6, color='0.15',
              bbox=dict(fc='white', ec='none', alpha=0.82, pad=1.0))

    # (e) how close each condition gets to any trained digit
    ax_e = fig.add_subplot(gs_bot[0, 3])
    pos = np.arange(len(LIKE))[::-1]
    for p, (lab, vals, c) in zip(pos, LIKE):
        parts = ax_e.violinplot([vals], positions=[p], vert=False, widths=0.82,
                                showextrema=False, showmedians=False)
        for b in parts['bodies']:
            b.set_facecolor(c); b.set_edgecolor('none'); b.set_alpha(0.5)
        ax_e.scatter(np.median(vals), p, s=9, marker='D', color=c, zorder=3,
                     edgecolor='none')
    ax_e.set_yticks(pos); ax_e.set_yticklabels([l[0] for l in LIKE])
    for t, l in zip(ax_e.get_yticklabels(), LIKE):
        t.set_color(l[2])
    ax_e.set_xlabel('max cosine to a trained digit', labelpad=1)
    ax_e.set_xlim(0.32, 1.03); ax_e.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax_e.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_e.spines[s].set_visible(False)

    figstyle.freeze(fig)
    for ax, lab, anchor in ((ax_a, '(a) NNLS round-trip', sp1),
                            (ax_b, '(b) mean fingerprint per condition', sp1),
                            (ax_emb, '(c) fingerprint PCA', sp2),
                            (ax_c, '(d) digits vs Fashion-MNIST', sp2),
                            (ax_d, '(e) fingerprint similarity', sp2),
                            (ax_e, '(f) distance to the trained digits', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), anchor.get_position().y0,
                 lab, ha='left', va='bottom')
    return fig


# ── Figure 6 / Appendix E — the CNN on CIFAR-10, read through its stimuli ─────

CNN_LAYER_LABEL = {'classifier': 'classifier', 'features.12': 'conv4',
                   'features.8': 'conv3', 'features.4': 'conv2',
                   'features.0': 'conv1'}


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


def fig6_cnn_circuits(D):
    NODES, CLS = D['nodes'], list(D['class_names'])
    by_path = {tuple(n['path'].tolist()): n for n in NODES}
    root = NODES[0]
    C_BFT, C_X = figstyle.color('ours'), figstyle.color('cross_class')
    C_BASE = figstyle.color('baseline')
    n_root = root['n_factors']

    SHOW = [0, 9, 5]            # circuits in panel (b): automobile, airplane, bird
    LEAF = (9, 0, 0, 0)         # the conv1 node panel (c) opens
    N_SUB, GAP = 3, 0.42        # stimuli per conv4 sub-factor; gap between circuits

    # panel heights follow from the image grids: a cell of the (a) row is
    # 1/n_root of the text width and square, a (b) column is 1/(3*6+2*GAP).
    W = 6.975
    h_a = W / n_root
    n_sub = by_path[(SHOW[0],)]['n_factors']
    h_b = N_SUB * W / (len(SHOW) * n_sub + (len(SHOW) - 1) * GAP)
    w_left, w_right = 1.02, 1.30
    n_leaf = by_path[LEAF]['n_factors']
    h_c = w_left / (w_left + w_right) * W / n_leaf
    lab_a, lab_b, lab_c = 0.19, 0.07, 0.28     # label sub-rows, in units of h_*
    sp = 0.13                                  # spacer rows, inches
    rows = [(sp, 0), (h_a * (1 + lab_a) + 0.05, 0), (sp, 0),
            (h_b * (1 + lab_b) + 0.02, 0), (sp, 0), (h_c * (1 + lab_c) + 0.05, 0)]
    total = sum(r[0] for r in rows)

    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='paper',
                   height_to_width_ratio=total / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.008, w_pad=0.008,
                          hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(6, 1, height_ratios=[r[0] for r in rows], hspace=0.0)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4)]
    for s_ in spacers:
        s_.set_axis_off()
    anchors = {}

    def tag(ax, text, x=0.05, y=0.97, size=6.5, color='white'):
        ax.text(x, y, text, transform=ax.transAxes, ha='left', va='top',
                fontsize=size, color=color,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2,
                                                                foreground='0.15')])

    # ── (a) every output-layer factor is a class detector ────────────────────
    gsa = gs[1].subgridspec(2, n_root, height_ratios=[1.0, lab_a])
    for k in range(n_root):
        ax = stim_panel(fig, gsa[0, k], D, root['top_images'][k], 2, 2)
        prof = root['class_profile'][k]
        c = int(np.argmax(prof))
        tag(ax, rf'$f_{{{k}}}$')
        lab = fig.add_subplot(gsa[1, k]); lab.set_axis_off()
        lab.text(0.5, 1.0, f'{CLS[c]} {prof[c]:.2f}', ha='center', va='top',
                 fontsize=6, color='0.2')
        if k == 0:
            anchors['a'] = ax

    # ── (b) one layer back: each circuit splits into distinct stimulus groups ─
    widths = []
    for j_ in range(len(SHOW)):
        widths += [1.0] * n_sub + ([GAP] if j_ < len(SHOW) - 1 else [])
    gsb = gs[3].subgridspec(2, len(widths), width_ratios=widths,
                            height_ratios=[1.0, lab_b])
    for j_, r in enumerate(SHOW):
        node = by_path[(r,)]
        own = int(np.argmax(root['class_profile'][r]))
        base = j_ * (n_sub + 1)
        for k in range(n_sub):
            prof = node['class_profile'][k]
            c = int(np.argmax(prof))
            # a group is called cross-class only when another class clearly leads
            off = c != own and prof[c] - prof[own] > 0.10
            ax = stim_panel(fig, gsb[0, base + k], D, node['top_images'][k], N_SUB, 1,
                            ec=C_X if off else '0.6', lw=1.1 if off else 0.4)
            tag(ax, rf'$k_{{{k}}}$', size=6)
            if off:
                tag(ax, CLS[c], x=0.5, y=0.035, size=6, color=C_X)
                ax.texts[-1].set(ha='center', va='bottom')
            if j_ == 0 and k == 0:
                anchors['b'] = ax
        if j_ < len(SHOW) - 1:
            fig.add_subplot(gsb[:, base + n_sub]).set_axis_off()
        head = fig.add_subplot(gsb[0, base:base + n_sub]); head.set_axis_off()
        head.set_zorder(-5)
        anchors[f'head{j_}'] = head

    # ── (c) at conv1 the groups are appearance, not class ────────────────────
    gsc = gs[5].subgridspec(1, 2, width_ratios=[w_left, w_right])
    leaf = by_path[LEAF]
    gscl = gsc[0, 0].subgridspec(3, n_leaf, height_ratios=[1.0, 0.13, lab_c - 0.13])
    for k in range(n_leaf):
        ax = stim_panel(fig, gscl[0, k], D, leaf['top_images'][k], 2, 2)
        rgb_strip(ax, leaf['conn']['in_mass'][k])
        tag(ax, rf'$k_{{{k}}}$')
        lab = fig.add_subplot(gscl[1, k]); lab.set_axis_off()
        lab.text(0.5, 1.0, f'{leaf["class_profile"][k].max():.2f}', ha='center',
                 va='top', fontsize=6, color='0.2')
        if k == 0:
            anchors['c'] = ax
    rec = color_arbor_recurrence(NODES)
    note = fig.add_subplot(gscl[2, :]); note.set_axis_off()
    note.text(0.0, 0.9, f'the same four color factors in all {n_root} circuits '
              f'(cosine {rec:.2f})', ha='left', va='top', fontsize=6, color='0.35')

    # ── (d) class purity falls, appearance coherence rises, toward the input ─
    stats, rand = cnn_depth_stats(D)
    ax_d = fig.add_subplot(gsc[0, 1])
    x = np.arange(len(stats))
    pur = [lam_weighted(st['purity'], st['lam']) for st in stats]
    chance = 1.0 / D['n_classes']
    ax_d.axhline(chance, color='0.65', lw=0.6, ls=(0, (3, 2)), zorder=0)
    for i_, st in enumerate(stats):
        ax_d.scatter(np.full(len(st['purity']), i_), st['purity'], s=2.2, color=C_BFT,
                     alpha=0.30, edgecolor='none', zorder=2)
    ax_d.plot(x, pur, color=C_BFT, marker='o', ms=3.2, lw=1.4, zorder=3)
    ax_d.set_ylim(0, 1.0); ax_d.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_d.set_yticklabels(['0', '', '.5', '', '1'])
    ax_d.set_ylabel('class purity', color=C_BFT, labelpad=1)
    ax_d.text(-0.28, chance + 0.03, 'chance', fontsize=6, color='0.45', ha='left',
              va='bottom')
    ax_t = figstyle.dual_axis(ax_d, left_color=C_BFT, right_color=C_BASE)
    spread = [st['spread'].mean() for st in stats]
    ax_t.axhline(rand, color=tint(C_BASE, 0.45), lw=0.6, ls=(0, (3, 2)), zorder=0)
    ax_t.plot(x, spread, color=C_BASE, marker='s', ms=2.8, lw=1.4, ls=(0, (4, 1.4)),
              zorder=3)
    ax_t.set_ylim(0, 0.34)
    ax_t.set_yticks([0, 0.1, 0.2, 0.3]); ax_t.set_yticklabels(['0', '', '.2', ''])
    ax_t.set_ylabel('color spread', color=C_BASE, labelpad=1)
    ax_t.text(len(stats) - 1.1, rand - 0.012, 'random stimuli', fontsize=6,
              color=tint(C_BASE, 0.3), ha='right', va='top')
    ax_d.set_xlim(-0.35, len(stats) - 0.55)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([CNN_LAYER_LABEL[st['layer']] for st in stats], fontsize=6.2)
    ax_d.tick_params(length=1.5, pad=1)
    ax_t.tick_params(length=1.5, pad=1)
    ax_d.spines['top'].set_visible(False); ax_t.spines['top'].set_visible(False)
    anchors['d'] = ax_d

    figstyle.freeze(fig)

    def _label(key, spacer, text, dx=0.0, **kw):
        fig.text(max(anchors[key].get_position().x0 + dx, 0.002),
                 spacers[spacer].get_position().y0 + 0.004, text, ha='left',
                 va='bottom', fontsize=7, **kw)

    _label('a', 0, '(a) output layer: the four strongest stimuli of every factor, '
                   "with its dominant class and that class's share of the factor",
           dx=-0.004)
    jac = sibling_overlap(NODES, 'features.12')
    _label('b', 1, '(b) one layer back (conv4): the six sub-factors of a circuit '
                   'take near-disjoint groups of its stimuli '
                   f'(sibling top sets overlap {jac:.2f})', dx=-0.004)
    _label('c', 2, '(c) conv1 factors of $f_9$, with class purity and RGB input '
                   'arbor', dx=-0.004)
    _label('d', 2, '(d) across depth', dx=-0.032)
    for j_, r in enumerate(SHOW):
        p = anchors[f'head{j_}'].get_position()
        fig.text((p.x0 + p.x1) / 2, p.y0 - 0.010,
                 rf'$f_{{{r}}}$ {CLS[int(np.argmax(root["class_profile"][r]))]}',
                 ha='center', va='top', fontsize=6.5)
    return fig


def figE_cnn_details(D):
    NODES, CLS = D['nodes'], list(D['class_names'])
    by_path = {tuple(n['path'].tolist()): n for n in NODES}
    root = NODES[0]
    C_BFT, C_X = figstyle.color('ours'), figstyle.color('cross_class')
    CMAP = seq_cmap(C_BFT, 'bft')
    n_root = root['n_factors']
    LAYERS = cnn_layers(NODES)
    CONV4 = [by_path[(r,)] for r in range(n_root)]
    n_sub = CONV4[0]['n_factors']
    CASCADE = [(9,), (9, 0), (9, 0, 0), (9, 0, 0, 0)]      # the traced spine of f9
    chance = 1.0 / D['n_classes']

    # row heights in inches, derived from the image grids they hold
    W, SP = 6.975, 0.15
    h_d = 5 * W / (2 * (0.24 + n_sub) + 0.30)
    n_spine = 2 + sum(by_path[p_]['n_factors'] for p_ in CASCADE[1:])
    h_e = W * (65.0 + 33.0) / (n_spine * 65.0 + (n_spine - 5) * 6.0 + 4 * 24.0)
    rows = [SP, 1.45, SP, h_d, SP, h_e, SP, 1.10]

    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='appendix',
                   height_to_width_ratio=sum(rows) / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.008, w_pad=0.008,
                          hspace=0.02, wspace=0.02)
    gs = fig.add_gridspec(8, 1, height_ratios=rows, hspace=0.0)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4, 6)]
    for s_ in spacers:
        s_.set_axis_off()
    anchors = {}

    def tag(ax, text, x=0.05, y=0.97, size=6, color='white', ha='left', va='top'):
        ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=size, color=color,
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2,
                                                                foreground='0.15')])

    gs1 = gs[1].subgridspec(1, 3, width_ratios=[1.20, 1.0, 0.80])

    # ── (a) informativity spectrum of every node, by layer ───────────────────
    gsa = gs1[0, 0].subgridspec(1, len(LAYERS))
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
        ax.set_title(f'{CNN_LAYER_LABEL[name]}\n({len(nodes)} node'
                     f"{'s' if len(nodes) > 1 else ''})", pad=1.5, fontsize=6.5,
                     linespacing=1.1)
        if i == 0:
            ax.set_ylabel(r'$\lambda$ share', labelpad=1)
            anchors['a'] = ax
        if i == len(LAYERS) // 2:
            ax.set_xlabel('factor $k$', labelpad=1)

    # ── (b) class profile of every output-layer factor ───────────────────────
    ax_b = fig.add_subplot(gs1[0, 1])
    M = root['class_profile']
    ax_b.imshow(M, cmap=CMAP, aspect='auto', vmin=0, vmax=M.max(),
                interpolation='nearest')
    ax_b.set_xticks(range(len(CLS)))
    ax_b.set_xticklabels([c[:5] for c in CLS], rotation=90, fontsize=6)
    ax_b.set_yticks(range(n_root))
    ax_b.set_yticklabels([rf'$f_{{{k}}}$' for k in range(n_root)], fontsize=6)
    style_matrix_axes(ax_b)
    anchors['b'] = ax_b

    # ── (c) class profile of every conv4 sub-factor ──────────────────────────
    ax_c = fig.add_subplot(gs1[0, 2])
    P = np.concatenate([n['class_profile'] for n in CONV4])
    ax_c.imshow(P, cmap=CMAP, aspect='auto', vmin=0, vmax=M.max(),
                interpolation='nearest')
    for r in range(1, n_root):
        ax_c.axhline(r * n_sub - 0.5, color='white', lw=0.6)
    ax_c.set_xticks(range(len(CLS)))
    ax_c.set_xticklabels([c[:5] for c in CLS], rotation=90, fontsize=6)
    ax_c.set_yticks(np.arange(n_root) * n_sub + (n_sub - 1) / 2)
    ax_c.set_yticklabels([rf'$f_{{{k}}}$' for k in range(n_root)], fontsize=6)
    ax_c.tick_params(length=0, pad=1)
    style_matrix_axes(ax_c)
    anchors['c'] = ax_c

    # ── (d) the six conv4 sub-factors of all ten circuits ────────────────────
    per_row = 2
    n_rows = n_root // per_row
    widths = []
    for b in range(per_row):
        widths += [0.24] + [1.0] * n_sub + ([0.30] if b < per_row - 1 else [])
    gsd = gs[3].subgridspec(n_rows, len(widths), width_ratios=widths)
    for r in range(n_root):
        row, blk = r % n_rows, r // n_rows
        base = blk * (n_sub + 2)
        own = int(np.argmax(root['class_profile'][r]))
        lab = fig.add_subplot(gsd[row, base]); lab.set_axis_off()
        lab.text(1.0, 0.5, rf'$f_{{{r}}}$' + '\n' + CLS[own][:5], ha='right',
                 va='center', fontsize=6, linespacing=1.2)
        node = CONV4[r]
        for k in range(n_sub):
            prof = node['class_profile'][k]
            c = int(np.argmax(prof))
            off = c != own and prof[c] - prof[own] > 0.10
            ax = stim_panel(fig, gsd[row, base + 1 + k], D, node['top_images'][k], 1, 1,
                            ec=C_X if off else '0.6', lw=1.1 if off else 0.4)
            tag(ax, rf'$k_{{{k}}}$')
            tag(ax, f'{CLS[c][:5]} {prof[c]:.2f}', x=0.5, y=0.045,
                color=C_X if off else 'white', ha='center', va='bottom')
            if r == 0 and k == 0:
                anchors['d'] = ax
        if blk < per_row - 1:
            fig.add_subplot(gsd[row, base + n_sub + 1]).set_axis_off()

    # ── (e) the traced spine of one circuit, root to conv1 ───────────────────
    spine = [root] + [by_path[p_] for p_ in CASCADE]
    n_show = [1, 1] + [n['n_factors'] for n in spine[2:]]   # root and conv4: the
    tiles, gaps, meta = [], [], []                          # traced factor only
    for i, n in enumerate(spine):
        for k in range(n_show[i]):
            kk = 9 if i == 0 else k                    # the root column is f9 itself
            tiles.append(montage(D, n['top_images'][kk], 2, 2))
            gaps.append(24 if k == n_show[i] - 1 else 6)
            meta.append((i, n, kk))
    comp_e = hcat(tiles, gaps[:-1])
    side = tiles[0].shape[0]
    ax_e = strip_axes(fig, gs[5], comp_e, top=15, bottom=18)
    x = 0
    for j_, (i, n, kk) in enumerate(meta):
        ax_e.text(x + 2, 2, rf'$f_9$' if i == 0 else rf'$k_{{{kk}}}$', ha='left',
                  va='top', fontsize=6, color='white',
                  path_effects=[matplotlib.patheffects.withStroke(linewidth=1.2,
                                                                  foreground='0.15')])
        ax_e.text(x + side / 2, side + 3, f"{n['class_profile'][kk].max():.2f}",
                  ha='center', va='top', fontsize=6, color='0.2')
        if 'conn' in n and n['conn']['in_mass'].shape[1] == 3:
            sh = n['conn']['in_mass'][kk]
            sh = sh / sh.sum()
            xs = x
            for frac, colr in zip(sh, ('#D62728', '#2CA02C', '#1F77B4')):
                ax_e.add_patch(matplotlib.patches.Rectangle(
                    (xs, side - 4), frac * side, 4, facecolor=colr, edgecolor='none',
                    zorder=5))
                xs += frac * side
        if j_ == 0 or meta[j_ - 1][0] != i:
            first_x = x
        if j_ == len(meta) - 1 or meta[j_ + 1][0] != i:
            name = 'classifier' if i == 0 else CNN_LAYER_LABEL[n['layer_name']]
            ax_e.text((first_x + x + side) / 2, -3, name, ha='center', va='bottom',
                      fontsize=6.5, color='0.35')
        x += side + gaps[j_]
    anchors['e'] = ax_e

    gs4 = gs[7].subgridspec(1, 2, width_ratios=[1.0, 1.0])

    # ── (f) the conv1 color arbor of every circuit ───────────────────────────
    ax_f = fig.add_subplot(gs4[0, 0])
    conv1 = [by_path[(r, 0, 0, 0)] for r in range(n_root)]
    n_k = conv1[0]['n_factors']
    for r, n in enumerate(conv1):
        Mi = n['conn']['in_mass']
        Mi = Mi / Mi.sum(1, keepdims=True)
        for k in range(n_k):
            x = r * (n_k + 1) + k
            bot = 0.0
            for ci, colr in enumerate(('#D62728', '#2CA02C', '#1F77B4')):
                ax_f.bar(x, Mi[k, ci], bottom=bot, color=colr, width=0.92,
                         edgecolor='none')
                bot += Mi[k, ci]
    ax_f.set_xlim(-0.9, n_root * (n_k + 1) - 1.1)
    ax_f.set_ylim(0, 1); ax_f.set_yticks([0, 0.5, 1])
    ax_f.set_yticklabels(['0', '.5', '1'])
    ax_f.set_xticks(np.arange(n_root) * (n_k + 1) + (n_k - 1) / 2)
    ax_f.set_xticklabels([rf'$f_{{{r}}}$' for r in range(n_root)], fontsize=6)
    ax_f.set_ylabel('input arbor share', labelpad=1)
    ax_f.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_f.spines[s_].set_visible(False)
    ax_f.text(0.5, 1.02, f'R / G / B, {n_k} factors per circuit — '
              f'cosine {color_arbor_recurrence(NODES):.2f} between circuits',
              transform=ax_f.transAxes, ha='center', va='bottom', fontsize=6,
              color='0.35')
    anchors['f'] = ax_f

    # ── (g) where a factor fires: stimulus and its activation map ────────────
    SPAT = [((9,), 'conv4'), ((9, 0, 0, 0), 'conv1')]
    n_im = 4
    rows_g, gaps_g = [], []
    for i, (p_, name) in enumerate(SPAT):
        sp_ = by_path[p_]['spatial']['top']
        for t in range(n_im):
            gaps_g.append(22 if t == n_im - 1 else 5)
    stim, maps = [], []
    for i, (p_, name) in enumerate(SPAT):
        sp_ = by_path[p_]['spatial']['top']
        for t in range(n_im):
            im = cifar_rgb(D, sp_['images'][t])
            m = sp_['maps'][t]
            m = m / (m.max() + 1e-12)
            m = np.repeat(np.repeat(m, im.shape[0] // m.shape[0], 0),
                          im.shape[1] // m.shape[1], 1)
            stim.append(im)
            maps.append(CMAP(m)[..., :3].astype(np.float32))
    comp_g = vcat([hcat(stim, gaps_g[:-1]), hcat(maps, gaps_g[:-1])], 4)
    ax_g = strip_axes(fig, gs4[0, 1], comp_g, top=14, bottom=2)
    side_g = stim[0].shape[0]
    x = 0
    for i in range(len(SPAT)):
        first_x = x
        for t in range(n_im):
            x += side_g + (gaps_g[i * n_im + t] if i * n_im + t < len(gaps_g) - 1 else 0)
        ax_g.text((first_x + x - gaps_g[i * n_im + n_im - 1]) / 2 if i < len(SPAT) - 1
                  else (first_x + x) / 2, -2, SPAT[i][1], ha='center', va='bottom',
                  fontsize=6.5, color='0.35')
    x = 0
    for j_ in range(len(stim)):
        for y0 in (0, side_g + 4):
            ax_g.add_patch(matplotlib.patches.Rectangle(
                (x - 0.5, y0 - 0.5), side_g, side_g, fill=False, edgecolor='0.6',
                linewidth=0.4, zorder=6))
        x += side_g + (gaps_g[j_] if j_ < len(gaps_g) - 1 else 0)
    ax_g.text(-4, side_g / 2, 'stimulus', ha='right', va='center', fontsize=6,
              color='0.35', rotation=90)
    ax_g.text(-4, side_g + 4 + side_g / 2, 'map', ha='right', va='center', fontsize=6,
              color='0.35', rotation=90)
    anchors['g'] = ax_g

    figstyle.freeze(fig)

    def _label(key, spacer, text, dx=0.0):
        fig.text(max(anchors[key].get_position().x0 + dx, 0.002),
                 spacers[spacer].get_position().y0 + 0.002, text, ha='left',
                 va='bottom', fontsize=7)

    _label('a', 0, r'(a) $\lambda$ spectra by layer', dx=-0.026)
    _label('b', 0, '(b) output factors', dx=-0.020)
    _label('c', 0, '(c) conv4 sub-factors, same scale', dx=-0.020)
    _label('d', 1, '(d) strongest stimulus of every conv4 sub-factor of every '
                   'circuit, with its dominant class (rose: a class other than the '
                   "circuit's)", dx=-0.020)
    _label('e', 2, r'(e) the traced spine of $f_9$, root to conv1 — class purity '
                   f'below each factor (chance {chance:.2f})', dx=-0.004)
    _label('f', 3, '(f) conv1 input arbor per circuit', dx=-0.030)
    _label('g', 3, '(g) stimulus and activation map', dx=-0.030)
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


def _cnn_fp_ood(D):
    """CIFAR-100: what the network says vs what the fingerprint says, per
    CIFAR-100 class (rows) and CIFAR-10 class (columns)."""
    fp = D['fp']
    cent = unit(D['fp_mean_by_class'])
    near = (unit(fp['ood']) @ cent.T).argmax(1)
    pred, tgt = fp['ood_preds'], fp['ood_targets']
    labels = np.unique(tgt)
    n_c = int(D['n_classes'])
    P_model = np.zeros((len(labels), n_c))
    P_fprint = np.zeros((len(labels), n_c))
    for j, t in enumerate(labels):
        m = tgt == t
        P_model[j] = np.bincount(pred[m], minlength=n_c) / m.sum()
        P_fprint[j] = np.bincount(near[m], minlength=n_c) / m.sum()
    return P_model, P_fprint, float(np.mean(near == pred)), near


def fig7_cnn_fingerprints(D):
    """Message: the 240-d fingerprint of the CNN is a class-structured code — as
    class-separable as the network's own penultimate activations at a comparable
    dimension — that reports the network's own CIFAR-100 decision and collapses
    onto a single point on far-OOD input."""
    N_CLASSES = int(D['n_classes'])
    CLS = list(D['class_names'])
    COL_ORDER, BLK_EDGE, BLK_START, BLOCK_SIZES = _cnn_fp_blocks(D)
    C_BFT, C_NEAR = figstyle.color('ours'), figstyle.color('near_ood')
    CMAP = ramp_cmap(C_BFT, 'bft')
    CCOL = class_colors(range(N_CLASSES))
    n_factors = int(D['n_factors'])
    fp = D['fp']
    COND = [(c['label'], c['values'], resolve_color(c['color_key'])) for c in D['cond']]
    P_model, P_fprint, agree, _ = _cnn_fp_ood(D)
    r_ood = float(np.corrcoef(P_model.ravel(), P_fprint.ravel())[0, 1])
    EMB = fp_vs_act(D, fp['id_targets'])

    W, SP = 6.975, 0.13
    rows = [SP, 1.32, SP, 1.30]
    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=1, mode='paper',
                   height_to_width_ratio=sum(rows) / W)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.02, wspace=0.03)
    gs = fig.add_gridspec(4, 1, height_ratios=rows, hspace=0.02)
    sp1 = fig.add_subplot(gs[0]); sp1.set_axis_off()
    sp2 = fig.add_subplot(gs[2]); sp2.set_axis_off()
    gs_top = gs[1].subgridspec(1, 3, width_ratios=[2.10, 1.0, 1.0])
    gs_bot = gs[3].subgridspec(1, 3, width_ratios=[1.05, 1.0, 1.15])

    # ── (a) mean fingerprint per class, columns grouped by output circuit ────
    gsa = gs_top[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.20])
    ax_a = fig.add_subplot(gsa[0, 0])
    M = D['fp_mean_by_class']
    M = M / M.sum(1, keepdims=True)
    Mc = M[:, COL_ORDER]
    Mc = Mc / (Mc.max(0, keepdims=True) + 1e-12)      # each factor to its top class
    ax_a.imshow(Mc, cmap=CMAP, aspect='auto', interpolation='nearest',
                norm=matplotlib.colors.PowerNorm(0.8, vmin=0, vmax=1))
    for e in BLK_EDGE:
        ax_a.axvline(e - 0.5, color='0.25', lw=0.7)
    ax_a.set_xticks([])
    ax_a.set_yticks(range(N_CLASSES)); ax_a.set_yticklabels(CLS, fontsize=6)
    for t_, c_ in zip(ax_a.get_yticklabels(), range(N_CLASSES)):
        t_.set_color(CCOL[c_])                        # the legend for (b) and (c)
    ax_a.set_xlabel(f'{n_factors} factors', labelpad=2)
    ax_a.tick_params(length=1.5, pad=1)
    for s_ in ax_a.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)

    # each class against each circuit block: the code's diagonal
    ax_ab = fig.add_subplot(gsa[0, 1])
    B = np.stack([M[:, COL_ORDER][:, BLK_START[b]:BLK_START[b] + BLOCK_SIZES[b]].sum(1)
                  for b in range(len(BLOCK_SIZES))], axis=1)
    ax_ab.imshow(B, cmap=CMAP, aspect='auto', interpolation='nearest',
                 vmin=B.min(), vmax=B.max())
    ax_ab.set_xticks([]); ax_ab.set_yticks([])
    for s_ in ax_ab.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    # frame each class's own circuit: with the blocks ordered by the circuit a
    # class loads most, a full diagonal means the class -> circuit map is 1:1
    for c_ in range(N_CLASSES):
        if B.argmax(1)[c_] == c_:
            ax_ab.add_patch(matplotlib.patches.Rectangle(
                (c_ - 0.5, c_ - 0.5), 1, 1, fill=False, edgecolor='white',
                linewidth=0.7, zorder=4))
    ax_ab.set_xlabel('own circuit', labelpad=2, fontsize=6, color=C_BFT)

    # ── (b, c) the embedding, against the network's own representation ──────
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_top[0, 2])
    if EMB is not None:
        Xf, lf, Xa, la, rep = EMB
        pca_panel(ax_b, Xf, lf, CCOL)
        pca_panel(ax_c, Xa, la, CCOL,
                  note=None if rep['aligned'] else 'independent sample')
    else:
        for ax in (ax_b, ax_c):
            _val_na(ax, 'no activation baseline\nin this bundle')

    # ── (d) the class geometry of the code ──────────────────────────────────
    ax_d = fig.add_subplot(gs_bot[0, 0])
    order = np.argsort(fp['id_targets'], kind='stable')
    U = unit(fp['id'][order])
    S = U @ U.T
    ax_d.imshow(S, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
    n_per = np.bincount(fp['id_targets'])
    cuts = np.cumsum(n_per)
    for c_ in cuts[:-1]:
        ax_d.axhline(c_ - 0.5, color='white', lw=0.4)
        ax_d.axvline(c_ - 0.5, color='white', lw=0.4)
    ax_d.set_xticks([]); ax_d.set_yticks(cuts - n_per / 2)
    ax_d.set_yticklabels(CLS, fontsize=6)
    for t_, c_ in zip(ax_d.get_yticklabels(), range(N_CLASSES)):
        t_.set_color(CCOL[c_])
    ax_d.set_xlabel(f'{len(U)} test stimuli', labelpad=1)
    ax_d.tick_params(length=0, pad=1)
    for s_ in ax_d.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)
    within = np.mean([S[a:b, a:b].mean() for a, b in
                      zip(np.concatenate([[0], cuts[:-1]]), cuts)])
    ax_d.text(0.97, 0.03, f'within-class $\\cos$ {within:.2f}',
              transform=ax_d.transAxes, ha='right', va='bottom', fontsize=6,
              color='0.15', bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0))

    # ── (e) CIFAR-100: the fingerprint names the class the network names ─────
    ax_e = fig.add_subplot(gs_bot[0, 1])
    ax_e.plot([0, 1], [0, 1], color='0.7', lw=0.6, ls=(0, (2.5, 2)), zorder=1)
    ax_e.scatter(P_model.ravel(), P_fprint.ravel(), s=3.5, color=C_NEAR,
                 edgecolor='none', alpha=0.55, zorder=2)
    ax_e.set_xlim(-0.06, 1.06); ax_e.set_ylim(-0.06, 1.06)
    ax_e.set_xticks([0, 0.5, 1]); ax_e.set_yticks([0, 0.5, 1])
    ax_e.set_xlabel('P(network says class $c$)', labelpad=1)
    ax_e.set_ylabel('P(fingerprint says class $c$)', labelpad=1)
    ax_e.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_e.spines[s_].set_visible(False)
    ax_e.text(0.04, 0.97, f'$r={r_ood:.2f}$\nagree {agree:.2f}',
              transform=ax_e.transAxes, ha='left', va='top', fontsize=6.5,
              linespacing=1.2)
    ax_e.text(0.04, 0.775, f'{len(P_model)} CIFAR-100\nclasses '
              rf'$\times$ {N_CLASSES}', transform=ax_e.transAxes, ha='left',
              va='top', fontsize=6, color=C_NEAR, linespacing=1.2)

    # ── (f) far-OOD: every stimulus collapses onto one fingerprint ───────────
    ax_f = fig.add_subplot(gs_bot[0, 2])
    pos = np.arange(len(COND))[::-1]
    for p_, (lab, vals, c_) in zip(pos, COND):
        parts = ax_f.violinplot([vals], positions=[p_], vert=False, widths=0.82,
                                showextrema=False, showmedians=False)
        for b_ in parts['bodies']:
            b_.set_facecolor(c_); b_.set_edgecolor('none'); b_.set_alpha(0.5)
        ax_f.scatter(np.median(vals), p_, s=9, marker='D', color=c_, zorder=3,
                     edgecolor='none')
    ax_f.set_yticks(pos); ax_f.set_yticklabels([c_[0] for c_ in COND])
    for t_, c_ in zip(ax_f.get_yticklabels(), COND):
        t_.set_color(c_[2])
    ax_f.set_xlim(0.42, 1.045); ax_f.set_xticks([0.5, 0.75, 1.0])
    ax_f.set_xlabel('cos. to condition mean', labelpad=1)
    ax_f.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_f.spines[s_].set_visible(False)

    figstyle.freeze(fig)
    emb_lab = ((f'(b) fingerprint, {n_factors}-d',
                f"(c) {EMB[4]['label'].replace(chr(10), ' ')}, {EMB[4]['dim']}-d")
               if EMB is not None else ('(b) fingerprint', '(c) activations'))
    for ax, lab, sp_ in ((ax_a, '(a) mean fingerprint per class', sp1),
                         (ax_b, emb_lab[0], sp1),
                         (ax_c, emb_lab[1], sp1),
                         (ax_d, '(d) class geometry', sp2),
                         (ax_e, '(e) CIFAR-100', sp2),
                         (ax_f, '(f) far-OOD collapse', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), sp_.get_position().y0,
                 lab, ha='left', va='bottom')
    return fig


CNN_FP_LAYER = {0: 'conv1', 1: 'conv2', 2: 'conv3', 3: 'conv4', 4: 'classifier'}


def figF_cnn_fingerprint_details(D):
    """Message: the fingerprint machinery behind Fig. 7, including where it is
    weakest — with 240 factors the NNLS round-trip is the loosest of the paper's
    models, while the code itself is stable across the train/test split."""
    N_CLASSES = int(D['n_classes'])
    CLS = list(D['class_names'])
    COL_ORDER, BLK_EDGE, BLK_START, BLOCK_SIZES = _cnn_fp_blocks(D)
    C_BFT = figstyle.color('ours')
    C_NEAR, C_FAR = figstyle.color('near_ood'), figstyle.color('far_ood')
    CMAP = ramp_cmap(C_BFT, 'bft')
    fp, dims = D['fp'], D['dims']
    n_factors = int(D['n_factors'])
    FAR = list(fp['far'])
    LIKE = [(c['label'], c['values'], resolve_color(c['color_key'])) for c in D['like']]

    # every condition's mean fingerprint, in one matrix
    rows = [(CLS[c], D['fp_mean_by_class'][c], '0.2') for c in range(N_CLASSES)]
    rows.append(('CIFAR-100', D['fp_mean_ood'], C_NEAR))
    rows += [(f['label'], f['F'].mean(0), C_FAR) for f in FAR]
    R = np.stack([r[1] for r in rows])
    R = R / R.sum(1, keepdims=True)

    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='appendix',
                   height_to_width_ratio=0.62)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.02, wspace=0.03)
    gs = fig.add_gridspec(4, 1, height_ratios=[0.10, 1.0, 0.10, 1.0], hspace=0.02)
    sp1 = fig.add_subplot(gs[0]); sp1.set_axis_off()
    sp2 = fig.add_subplot(gs[2]); sp2.set_axis_off()
    gs_top = gs[1].subgridspec(1, 3, width_ratios=[0.72, 1.60, 0.78])
    gs_bot = gs[3].subgridspec(1, 4, width_ratios=[1.0, 0.85, 1.05, 1.15])

    # ── (a) NNLS round-trip: the honest number ──────────────────────────────
    ax_a = fig.add_subplot(gs_top[0, 0])
    v = D['rt_sims']
    ax_a.hist(v, bins=24, color=tint(C_BFT, 0.4), edgecolor=C_BFT, linewidth=0.4)
    ax_a.axvline(v.mean(), color=C_BFT, lw=1.0)
    ax_a.set_xlabel('cos(NMF, NNLS re-fit)', labelpad=1)
    ax_a.set_ylabel('stimuli', labelpad=1)
    ax_a.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_a.spines[s_].set_visible(False)
    ax_a.text(0.03, 0.97, f'mean {v.mean():.3f}\nmin {v.min():.3f}\n'
              f'{int((v > 0.95).sum())} of {len(v)} above 0.95',
              transform=ax_a.transAxes, ha='left', va='top', fontsize=6,
              linespacing=1.25)

    # ── (b) mean fingerprint of every condition ─────────────────────────────
    ax_b = fig.add_subplot(gs_top[0, 1])
    Rc = R[:, COL_ORDER]
    Rc = Rc / (Rc.max(0, keepdims=True) + 1e-12)
    ax_b.imshow(Rc, cmap=CMAP, aspect='auto', interpolation='nearest',
                norm=matplotlib.colors.PowerNorm(0.8, vmin=0, vmax=1))
    for e in BLK_EDGE:
        ax_b.axvline(e - 0.5, color='0.25', lw=0.6)
    ax_b.axhline(N_CLASSES - 0.5, color='0.25', lw=0.8)
    ax_b.axhline(N_CLASSES + 0.5, color='0.25', lw=0.8)
    ax_b.set_xticks([]); ax_b.set_yticks(range(len(rows)))
    ax_b.set_yticklabels([r[0] for r in rows], fontsize=6)
    for t_, r_ in zip(ax_b.get_yticklabels(), rows):
        t_.set_color(r_[2])
    ax_b.set_xlabel(f'{n_factors} factors, grouped into the {N_CLASSES} circuits',
                    labelpad=2)
    ax_b.tick_params(length=1.5, pad=1)
    for s_ in ax_b.spines.values():
        s_.set_color('0.6'); s_.set_linewidth(0.4)

    # ── (c) where the OOD conditions land in the plane of the classes ───────
    ax_emb = fig.add_subplot(gs_top[0, 2])
    cond_embedding(ax_emb, fp['id'], fp['id_targets'], class_colors(range(N_CLASSES)),
                   near=fp['ood'], far=[f['F'] for f in FAR],
                   c_near=C_NEAR, c_far=C_FAR, near_label='CIFAR-100',
                   id_label='CIFAR-10')

    # ── (d) the code survives the train/test split ──────────────────────────
    # (c) where Fashion-MNIST and the far-OOD stimuli land in the digit plane
    ax_c = fig.add_subplot(gs_bot[0, 0])
    Ttr = unit(np.stack([fp['train'][fp['train_targets'] == c].mean(0)
                         for c in range(N_CLASSES)]))
    Tte = unit(np.stack([fp['id'][fp['id_targets'] == c].mean(0)
                         for c in range(N_CLASSES)]))
    S = Ttr @ Tte.T
    ax_c.imshow(S, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
    ax_c.set_xticks(range(N_CLASSES)); ax_c.set_xticklabels(CLS, rotation=90,
                                                            fontsize=6)
    ax_c.set_yticks(range(N_CLASSES)); ax_c.set_yticklabels(CLS, fontsize=6)
    ax_c.tick_params(length=0, pad=1)
    style_matrix_axes(ax_c)
    n_hit = int((S.argmax(1) == np.arange(N_CLASSES)).sum())
    ax_c.set_xlabel('rows: train, columns: test\n'
                    f'diag {np.diag(S).mean():.2f}, off-diag '
                    f'{S[~np.eye(N_CLASSES, dtype=bool)].mean():.2f}, '
                    f'{n_hit}/{N_CLASSES} nearest',
                    labelpad=2, fontsize=6, color='0.35', linespacing=1.25)

    # ── (d) how close each OOD condition sits to each class ─────────────────
    ax_d = fig.add_subplot(gs_bot[0, 1])
    CENT = unit(D['fp_mean_by_class'])
    conds = [('CIFAR-100', D['fp_mean_ood'])] + [(f['label'], f['F'].mean(0))
                                                 for f in FAR]
    Sc = CENT @ unit(np.stack([c[1] for c in conds])).T
    ax_d.imshow(Sc, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest', aspect='auto')
    ax_d.set_xticks(range(len(conds)))
    ax_d.set_xticklabels([c[0] for c in conds], rotation=90, fontsize=6)
    for t_, c_ in zip(ax_d.get_xticklabels(), conds):
        t_.set_color(C_NEAR if c_[0] == 'CIFAR-100' else C_FAR)
    ax_d.set_yticks(range(N_CLASSES)); ax_d.set_yticklabels(CLS, fontsize=6)
    ax_d.tick_params(length=0, pad=1)
    style_matrix_axes(ax_d)
    ax_d.set_xlabel(f'CIFAR-100 up to {Sc[:, 0].max():.2f},\n'
                    f'far-OOD up to {Sc[:, 1:].max():.2f}', labelpad=2, fontsize=6,
                    color='0.35', linespacing=1.25)

    # ── (e) distance to the trained classes, per condition ──────────────────
    ax_e = fig.add_subplot(gs_bot[0, 2])
    pos = np.arange(len(LIKE))[::-1]
    for p_, (lab, vals, c_) in zip(pos, LIKE):
        parts = ax_e.violinplot([vals], positions=[p_], vert=False, widths=0.82,
                                showextrema=False, showmedians=False)
        for b_ in parts['bodies']:
            b_.set_facecolor(c_); b_.set_edgecolor('none'); b_.set_alpha(0.5)
        ax_e.scatter(np.median(vals), p_, s=9, marker='D', color=c_, zorder=3,
                     edgecolor='none')
    ax_e.set_yticks(pos); ax_e.set_yticklabels([c_[0] for c_ in LIKE])
    for t_, c_ in zip(ax_e.get_yticklabels(), LIKE):
        t_.set_color(c_[2])
    ax_e.set_xlim(0.15, 1.03); ax_e.set_xticks([0.25, 0.5, 0.75, 1.0])
    ax_e.set_xlabel('max cos. to a class', labelpad=1)
    ax_e.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_e.spines[s_].set_visible(False)

    # ── (f) where in the network the fingerprint mass sits ──────────────────
    ax_f = fig.add_subplot(gs_bot[0, 3])
    lay = dims[:, 0]
    layers = sorted(np.unique(lay))[::-1]                  # output first
    shades = [tint(C_BFT, 0.72 * i / max(len(layers) - 1, 1)) for i in
              range(len(layers))]
    sets = [('ID test', fp['id'], '0.2'), ('CIFAR-100', fp['ood'], C_NEAR)] + \
           [(f['label'], f['F'], C_FAR) for f in FAR]
    x = np.arange(len(sets))
    bot = np.zeros(len(sets))
    for li, l_ in enumerate(layers):
        share = np.array([float((X / (X.sum(1, keepdims=True) + 1e-12))[:, lay == l_]
                                .sum(1).mean()) for _, X, _ in sets])
        ax_f.bar(x, share, bottom=bot, color=shades[li], width=0.78,
                 edgecolor='white', linewidth=0.3,
                 label=f'{CNN_FP_LAYER[int(l_)]} ({int((lay == l_).sum())})')
        bot += share
    ax_f.set_xticks(x)
    ax_f.set_xticklabels([s_[0] for s_ in sets], rotation=90, fontsize=6)
    for t_, s_ in zip(ax_f.get_xticklabels(), sets):
        t_.set_color(s_[2])
    ax_f.set_ylim(0, 1); ax_f.set_yticks([0, 0.5, 1])
    ax_f.set_yticklabels(['0', '.5', '1'])
    ax_f.set_ylabel('mass share', labelpad=1)
    ax_f.tick_params(length=1.5, pad=1)
    for s_ in ('top', 'right'):
        ax_f.spines[s_].set_visible(False)
    ax_f.legend(fontsize=6, frameon=False, loc='center left',
                bbox_to_anchor=(1.0, 0.5), handlelength=0.7, handletextpad=0.35,
                borderpad=0.1, labelspacing=0.25, borderaxespad=0.0)

    figstyle.freeze(fig)
    for ax, lab, sp_ in ((ax_a, '(a) NNLS round-trip', sp1),
                         (ax_b, '(b) mean fingerprint per condition', sp1),
                         (ax_emb, '(c) fingerprint PCA', sp1),
                         (ax_c, '(d) train vs test', sp2),
                         (ax_d, '(e) classes vs OOD', sp2),
                         (ax_e, '(f) distance to the classes', sp2),
                         (ax_f, '(g) mass by layer', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002),
                 sp_.get_position().y0, lab, ha='left', va='bottom')
    return fig


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
        ax.set_xlabel('factor $k$', labelpad=1)
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
            ax.text(0.05, 0.97, rf'$k_{k}$', transform=ax.transAxes, ha='left',
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
                 fontsize=7)

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


# ── Appendix H — the whole fingerprint analysis of the TinyViT ────────────────

def _digit_shades(digits, c_even, c_odd):
    """Parity gives the hue; position within the parity gives the shade."""
    out, seen = {}, {0: 0, 1: 0}
    for d in sorted(int(x) for x in digits):
        base = c_even if d % 2 == 0 else c_odd
        out[d] = tint(base, 0.62 * seen[d % 2] / 4)
        seen[d % 2] += 1
    return out


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


def _violin_rows(ax, rows, xlabel, xlim, xticks):
    """Horizontal violins, one per condition — the OOD panels of every fingerprint fig."""
    pos = np.arange(len(rows))[::-1]
    for p, (lab, vals, c) in zip(pos, rows):
        parts = ax.violinplot([vals], positions=[p], vert=False, widths=0.82,
                              showextrema=False, showmedians=False)
        for b in parts['bodies']:
            b.set_facecolor(c); b.set_edgecolor('none'); b.set_alpha(0.5)
        ax.scatter(np.median(vals), p, s=9, marker='D', color=c, zorder=3,
                   edgecolor='none')
    ax.set_yticks(pos); ax.set_yticklabels([r[0] for r in rows])
    for t, r in zip(ax.get_yticklabels(), rows):
        t.set_color(r[2])
    ax.set_xlim(*xlim); ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def figH_vit_fingerprints(D):
    COL_ORDER = list(D['col_order'])
    BLK_EDGE, BLOCK_SIZES = list(D['blk_edge']), list(D['block_sizes'])
    n_factors = int(D['n_factors'])
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    C_BFT, C_NEAR, C_FAR = (figstyle.color('ours'), figstyle.color('near_ood'),
                            figstyle.color('far_ood'))
    CMAP = ramp_cmap(C_BFT, 'bft')
    DIGITS = list(range(D['fp_mean_by_digit'].shape[0]))
    DIGIT_COLOR = _digit_shades(DIGITS, C_EVEN, C_ODD)
    rt = D['roundtrip_corrs']
    COND = [(c['label'], c['values'], resolve_color(c['color_key'])) for c in D['cond']]
    LIKE = [(l['label'], l['values'], resolve_color(l['color_key'])) for l in D['like']]
    SIM, SIM_NAMES = D['condition_similarity']['matrix'], list(
        D['condition_similarity']['names'])

    fp = D['fp']
    F_id, id_digits, id_targets = fp['id'], np.asarray(fp['id_digits']), np.asarray(
        fp['id_targets'])
    F_fm = fp['fmnist']
    FAR = [(f['label'], f['F']) for f in fp['far']]

    # (b) needs a balanced, plottable subset of the in-distribution fingerprints
    PER = 30
    sel = np.concatenate([np.where(id_digits == d)[0][:PER] for d in DIGITS])
    F_sel, sel_digits = F_id[sel], id_digits[sel]
    sil_class = cosine_silhouette(F_sel, sel_digits % 2)
    sil_digit = cosine_silhouette(F_sel, sel_digits)

    def _block_edges():
        return np.concatenate([[0], BLK_EDGE, [len(COL_ORDER)]])

    def _draw_blocks(ax, n_rows, y_txt):
        for e in BLK_EDGE:
            ax.axvline(e - 0.5, color='0.25', lw=0.7)
        ed = _block_edges()
        for bi in range(len(BLOCK_SIZES)):
            ax.text((ed[bi] + ed[bi + 1] - 1) / 2, y_txt, rf'$f_{bi}$', ha='center',
                    va='bottom', fontsize=6.5, color=C_BFT)

    figstyle.apply(venue='aaai2024', width='full', nrows=3, ncols=3, mode='appendix',
                   height_to_width_ratio=0.86)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.03, wspace=0.03)
    gs = fig.add_gridspec(6, 3, height_ratios=[0.10, 1.0, 0.10, 1.0, 0.10, 1.0],
                          width_ratios=[1.28, 1.0, 1.0], hspace=0.08)
    spacers = [fig.add_subplot(gs[r, :]) for r in (0, 2, 4)]
    for sp in spacers:
        sp.set_axis_off()

    # (a) mean fingerprint per digit — the code is blocked by output circuit
    gsa = gs[1, :2].subgridspec(1, 2, width_ratios=[1.0, 0.10])
    ax_a = fig.add_subplot(gsa[0, 0])
    M = D['fp_mean_by_digit']
    M = M / M.sum(1, keepdims=True)
    ax_a.imshow(M[:, COL_ORDER], cmap=CMAP, aspect='auto', interpolation='nearest',
                norm=matplotlib.colors.PowerNorm(0.7, vmin=0, vmax=M.max()))
    _draw_blocks(ax_a, len(DIGITS), -0.75)
    ax_a.set_xticks([])
    ax_a.set_yticks(DIGITS); ax_a.set_yticklabels(DIGITS)
    for t, d in zip(ax_a.get_yticklabels(), DIGITS):
        t.set_color(DIGIT_COLOR[d])
    ax_a.set_ylabel('stimulus digit', labelpad=1)
    ax_a.set_xlabel(f'{n_factors} factors, grouped by output circuit', labelpad=2)
    ax_a.tick_params(length=1.5, pad=1)
    for s in ax_a.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_a.set_ylim(len(DIGITS) - 0.5, -1.2)

    # per-digit share of each output circuit — the same numbers, summed per block
    ax_as = fig.add_subplot(gsa[0, 1])
    ed = _block_edges()
    Mo = M[:, COL_ORDER]
    blk_cols = [tint(C_BFT, f) for f in (0.0, 0.35, 0.68)]
    for d in DIGITS:
        left = 0.0
        for bi in range(len(BLOCK_SIZES)):
            w = float(Mo[d, ed[bi]:ed[bi + 1]].sum())
            ax_as.barh(d, w, left=left, color=blk_cols[bi], edgecolor='none',
                       height=0.8)
            left += w
    ax_as.set_xlim(0, 1); ax_as.set_ylim(len(DIGITS) - 0.5, -1.2)
    ax_as.set_xticks([]); ax_as.set_yticks([])
    for s in ax_as.spines.values():
        s.set_visible(False)
    ax_as.set_xlabel('circuit share', labelpad=2, fontsize=6)
    for bi, c in enumerate(blk_cols):
        ax_as.text((bi + 0.5) / 3, -0.9, rf'$f_{bi}$', ha='center', va='bottom',
                   fontsize=6, color=c if bi < 2 else '0.45')

    # (b) pairwise similarity of the in-distribution fingerprints, sorted by digit
    ax_b = fig.add_subplot(gs[1, 2])
    U = unit(F_sel)
    ax_b.imshow(U @ U.T, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    for b in range(PER, PER * len(DIGITS), PER):
        ax_b.axhline(b - 0.5, color='0.35', lw=0.3)
        ax_b.axvline(b - 0.5, color='0.35', lw=0.3)
    ctr = (np.arange(len(DIGITS)) + 0.5) * PER
    ax_b.set_xticks(ctr); ax_b.set_xticklabels(DIGITS, fontsize=6)
    ax_b.set_yticks(ctr); ax_b.set_yticklabels(DIGITS, fontsize=6)
    for tt in (ax_b.get_xticklabels(), ax_b.get_yticklabels()):
        for t, d in zip(tt, DIGITS):
            t.set_color(DIGIT_COLOR[d])
    ax_b.tick_params(length=0, pad=1)
    for s in ax_b.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_b.set_xlabel('stimulus digit', labelpad=1)
    ax_b.text(0.96, 0.96, f'silhouette\n{sil_class:.2f} parity\n{sil_digit:.2f} digit',
              transform=ax_b.transAxes, ha='right', va='top', fontsize=6,
              linespacing=1.15, color='0.15',
              bbox=dict(fc='white', ec='none', alpha=0.82, pad=1.0))

    # (c) mean fingerprint of every stimulus condition
    ax_c = fig.add_subplot(gs[3, :2])
    ROWS = ([('even (ID)', F_id[id_targets == 0].mean(0), C_EVEN),
             ('odd (ID)', F_id[id_targets == 1].mean(0), C_ODD),
             ('Fashion-MNIST', F_fm.mean(0), C_NEAR)] +
            [(lab, X.mean(0), C_FAR) for lab, X in FAR])
    Mc = np.stack([m for _, m, _ in ROWS])
    Mc = Mc / Mc.sum(1, keepdims=True)
    ax_c.imshow(Mc[:, COL_ORDER], cmap=CMAP, aspect='auto', interpolation='nearest',
                norm=matplotlib.colors.PowerNorm(0.7, vmin=0, vmax=Mc.max()))
    _draw_blocks(ax_c, len(ROWS), -0.85)
    ax_c.axhline(1.5, color='0.25', lw=0.8)
    ax_c.axhline(2.5, color='0.25', lw=0.8)
    ax_c.set_xticks([])
    ax_c.set_yticks(range(len(ROWS)))
    ax_c.set_yticklabels([r[0] for r in ROWS], fontsize=6)
    for t, r in zip(ax_c.get_yticklabels(), ROWS):
        t.set_color(r[2])
    ax_c.set_xlabel(f'{n_factors} factors, grouped by output circuit', labelpad=2)
    ax_c.tick_params(length=1.5, pad=1)
    for s in ax_c.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_c.set_ylim(len(ROWS) - 0.5, -1.3)

    # (d) mean fingerprint similarity between conditions
    ax_d = fig.add_subplot(gs[3, 2])
    ax_d.imshow(SIM, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    for i in range(len(SIM)):
        for j in range(len(SIM)):
            ax_d.text(j, i, f'{SIM[i, j]:.2f}'.lstrip('0'), ha='center', va='center',
                      fontsize=6, color='white' if SIM[i, j] > 0.6 else '0.2')
    short = [n.replace(' (ID)', '').replace('FashionMNIST', 'F-MNIST')
             for n in SIM_NAMES]
    cols_d = [C_EVEN, C_ODD, C_NEAR] + [C_FAR] * (len(short) - 3)
    ax_d.set_xticks(range(len(short)))
    ax_d.set_xticklabels(short, rotation=90, fontsize=6)
    ax_d.set_yticks(range(len(short))); ax_d.set_yticklabels(short, fontsize=6)
    for tt in (ax_d.get_xticklabels(), ax_d.get_yticklabels()):
        for t, c in zip(tt, cols_d):
            t.set_color(c)
    ax_d.tick_params(length=0, pad=1)
    for s in ax_d.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)

    # (e) the same fingerprints in the plane of their first two principal
    #     components, and the network's own representation beside it
    gse = gs[5, 0].subgridspec(1, 2, width_ratios=[1.0, 0.78])
    ax_e = fig.add_subplot(gse[0, 0])
    ax_e2 = fig.add_subplot(gse[0, 1])
    EMB = fp_vs_act(D, id_digits)
    if EMB is not None:
        _Xf, _lf, _Xa, _la, _rep = EMB
        pca_panel(ax_e2, _Xa, _la, DIGIT_COLOR, sil=False,
                  note=None if _rep['aligned'] else 'independent sample')
    else:
        _val_na(ax_e2, 'no activation baseline\nin this bundle')
    X_all = np.concatenate([unit(F_sel), unit(F_fm)] + [unit(X) for _, X in FAR])
    Xc = X_all - X_all.mean(0)
    _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    evr = sv ** 2 / (sv ** 2).sum()
    P = Xc @ vt[:2].T
    n_sel, n_fm = len(F_sel), len(F_fm)
    ax_e.scatter(*P[n_sel + n_fm:].T, s=3.5, color=C_FAR, alpha=0.5, edgecolor='none',
                 zorder=1)
    ax_e.scatter(*P[n_sel:n_sel + n_fm].T, s=3.5, color=C_NEAR, alpha=0.6,
                 edgecolor='none', zorder=2)
    for d in DIGITS:
        m = sel_digits == d
        ax_e.scatter(*P[:n_sel][m].T, s=3.5, color=DIGIT_COLOR[d], alpha=0.8,
                     edgecolor='none', zorder=3)
    for lab, c in (('even (ID)', C_EVEN), ('odd (ID)', C_ODD),
                   ('Fashion-MNIST', C_NEAR), ('far-OOD', C_FAR)):
        ax_e.scatter([], [], s=8, color=c, label=lab)
    ax_e.set_xlabel(f'PC 1 (var {evr[0]:.2f})', labelpad=1, fontsize=6)
    ax_e.set_ylabel(f'PC 2 (var {evr[1]:.2f})', labelpad=1, fontsize=6)
    ax_e.set_xticks([]); ax_e.set_yticks([])
    for s in ax_e.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_e.legend(fontsize=6, frameon=False, loc='upper left', handlelength=0.7,
                handletextpad=0.25, borderpad=0.1, labelspacing=0.18, ncol=2,
                columnspacing=0.6, borderaxespad=0.15, scatterpoints=1)

    # (f) far-OOD: every stimulus collapses onto one fingerprint
    ax_f = fig.add_subplot(gs[5, 1])
    _violin_rows(ax_f, COND, 'cos. to condition mean', (0.42, 1.045), [0.5, 0.75, 1.0])

    # (g) how close each condition gets to a trained class
    ax_g = fig.add_subplot(gs[5, 2])
    _violin_rows(ax_g, LIKE, 'max cosine to a trained class', (0.32, 1.03),
                 [0.4, 0.6, 0.8, 1.0])

    figstyle.freeze(fig)
    rt_txt = (rf'NNLS round-trip $r={rt.min():.3f}$--${rt.max():.3f}$'
              if rt.min() < rt.max() else rf'NNLS round-trip $r={rt.min():.3f}$')
    for ax, lab, anchor, dx in (
            (ax_a, '(a) mean fingerprint per digit', spacers[0], 0.0),
            (ax_b, '(b) fingerprint similarity', spacers[0], -0.030),
            (ax_c, f'(c) mean fingerprint per condition — {rt_txt}', spacers[1], 0.0),
            (ax_d, '(d) condition similarity', spacers[1], -0.030),
            (ax_e, f"(e) fingerprint PCA, {n_factors}-d", spacers[2], 0.0),
            (ax_e2, (f"{EMB[4]['label']}, {EMB[4]['dim']}-d"
                     if EMB is not None else 'activations'), spacers[2], 0.0),
            (ax_f, '(f) far-OOD collapse', spacers[2], 0.0),
            (ax_g, '(g) distance to a trained class', spacers[2], -0.030)):
        fig.text(max(ax.get_position().x0 - 0.004 + dx, 0.002),
                 anchor.get_position().y0, lab, ha='left', va='bottom')
    return fig


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


def _val_axes(ax, *, xlab=None, ylab=None):
    ax.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    if xlab:
        ax.set_xlabel(xlab, labelpad=1.5)
    if ylab:
        ax.set_ylabel(ylab, labelpad=1.5)


def _val_note(ax, text, *, loc='lower right'):
    va, y = ('bottom', 0.03) if 'lower' in loc else ('top', 0.97)
    ha, x = ('right', 0.97) if 'right' in loc else ('left', 0.03)
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6,
            color='0.25', linespacing=1.2,
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=0.8))


def _seed_spread(D, key):
    """Model-seed std of a headline number, or None when only one model was traced."""
    sr = D.get('seed_repeats')
    if not isinstance(sr, dict) or not isinstance(sr.get('model_seed'), dict):
        return None
    m = sr['model_seed'].get(key)
    if not isinstance(m, dict) or int(m.get('n', 0)) < 2:
        return None
    return float(m['mean']), float(m['std']), int(m['n'])


def _boot_incumbent(D):
    """Notebook 10's paired stimulus-subsampling record for the *incumbent* config.

    The incumbent row is ``bft()`` at the registry defaults, which is exactly the
    configuration every measurement panel of this figure was traced at — so its
    interval is the one that may be drawn on those panels, never the selected
    configuration's. Two models (pretrained SqueezeNet, TinyViT) have a single
    checkpoint and can never have a model-seed error bar; this is the only spread
    they will ever get. Returns None when notebook 10 never ran for this model.

    The bundle writer replaces '.' in dict keys, so the incumbent's name has to be
    sanitized the same way before it is looked up.
    """
    bt = D.get('bootstrap')
    if not isinstance(bt, dict) or not isinstance(bt.get('per_config'), dict):
        return None
    rec = bt['per_config'].get(str(bt.get('incumbent', '')).replace('.', '_'))
    return rec if isinstance(rec, dict) else None


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


def _val_header_lines(D, boot, g_boot):
    """Subtitle lines under a validation figure's title.

    The nine panels are traced at the registry defaults, while the settings in the
    subtitle are what the sweep *selected* — a distinction that only starts to
    matter once notebook 10 both finished the grid and put an interval on it, so
    the extra lines appear exactly then. Built before the layout because the
    header row has to be sized for however many lines come back.
    """
    hp = D.get('final_hp') if isinstance(D.get('final_hp'), dict) else {}
    kw = hp.get('bft_kwargs', {})
    lines = [f"$n={int(D['config']['n_samples'])}$ stimuli"
             + (f", selected $K_{{\\max}}={_executed_k_max(kw['k_max'])}$"
                if 'k_max' in kw else '')
             + (f", threshold ${float(kw['stimulus_threshold']):g}$"
                if 'stimulus_threshold' in kw else '')
             + (f", {int(hp['fingerprint_dim'])}-d fingerprint"
                if 'fingerprint_dim' in hp else '')]
    if not boot:
        return lines
    if hp.get('selected_config'):
        n_cfg = len(D.get('HP_sweep', {}).get('configs', []) or [])
        ci = hp.get('silhouette_ci')
        lines.append(
            f"selection: {_cfg_label(hp['selected_config'])}"
            + (f' of {n_cfg} swept configurations' if n_cfg else '')
            + f", silhouette {float(hp['silhouette']):.3f}"
            + (f" [{float(ci[0]):.3f}, {float(ci[1]):.3f}]"
               if ci is not None and ci[0] is not None else '')
            + f" against {float(boot['mean']):.3f} at the registry defaults, "
              'which is what every panel below is traced at')
    if g_boot:                                  # usetex: a bare % comments the line
        lines.append('error bar in (g): paired 80\\,\\% stimulus subsample — this model '
                     'has a single checkpoint, so no model-seed spread exists')
    return lines


def _layer_shades(n, base):
    """Input layer darkest -> output layer lightest, so depth is readable."""
    return [tint(base, 0.62 * i / max(n - 1, 1)) for i in range(n)]


def fig_validation(D):
    """Appendix validation figure — faithfulness, robustness, baselines.

    Drawn from a ``nb09_<exp>_validation`` bundle; the same function renders all
    five models (the bundle carries the model's name, its capability flags and
    every number notebook 09 measured).
    """
    C_BFT, C_BASE = figstyle.color('ours'), figstyle.color('baseline')
    C_CEIL, C_RAND = figstyle.color('ceiling'), figstyle.color('random')
    caps = D['caps']
    boot = _boot_incumbent(D)         # notebook 10's interval, or None if it never ran
    sil_sp = _seed_spread(D, 'silhouette')
    # one checkpoint => no model-seed spread ever; fall back to notebook 10's paired
    # stimulus subsample, which is measured at exactly the configuration (g) shows
    g_boot = boot if not sil_sp else None
    sub_lines = _val_header_lines(D, boot, g_boot)

    figstyle.apply(venue='aaai2024', width='full', nrows=3, ncols=3, mode='appendix',
                   height_to_width_ratio=0.80)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.014, w_pad=0.016,
                          hspace=0.03, wspace=0.03)
    # the header row has to grow with the subtitle, or an extra line lands on (a)
    gs = fig.add_gridspec(7, 3,
                          height_ratios=[0.20 + 0.11 * (len(sub_lines) - 1),
                                         0.10, 1.0, 0.10, 1.0, 0.10, 1.0],
                          hspace=0.06)
    header = fig.add_subplot(gs[0, :]); header.set_axis_off()
    spacers = [fig.add_subplot(gs[r, :]) for r in (1, 3, 5)]
    for sp in spacers:
        sp.set_axis_off()
    ax = {(r, c): fig.add_subplot(gs[2 + 2 * r, c]) for r in range(3) for c in range(3)}

    # ── (a) causal reconstruction against a floor and two ceilings ────────────
    a = ax[(0, 0)]
    if caps['recon'] and isinstance(D['recon_controls'], dict):
        VARIANTS = [('random, rank $R$', 'random_R', C_RAND),
                    ('act.-only NMF', 'act_nmf_K', C_BASE),
                    ('BFT', 'bft', C_BFT),
                    ('SVD, rank $R$', 'svd_R', C_CEIL),
                    ('exact', 'exact', tint(C_CEIL, 0.45))]
        nodes = list(D['recon_controls'].values())
        pos = np.arange(len(VARIANTS))[::-1]
        for p, (lab, key, c) in zip(pos, VARIANTS):
            v = [float(n[key]['preact_r2']) for n in nodes if key in n]
            a.barh(p, np.mean(v), height=0.66, color=c, edgecolor='none',
                   alpha=0.85 if key != 'bft' else 1.0)
            if len(v) > 1:
                a.scatter(np.clip(v, -1.02, 1.02), np.full(len(v), p), s=4.5, color='0.15',
                          zorder=3, edgecolor='none', alpha=0.75)
        a.axvline(0, color='0.4', lw=0.5)
        a.set_yticks(pos); a.set_yticklabels([v[0] for v in VARIANTS])
        for t, v in zip(a.get_yticklabels(), VARIANTS):
            t.set_color(v[2] if v[1] != 'exact' else C_CEIL)
        a.set_xlim(-1.05, 1.05); a.set_xticks([-1, -0.5, 0, 0.5, 1])
        _val_axes(a, xlab=r'pre-activation $R^2$')
        a.spines['left'].set_visible(False); a.tick_params(axis='y', length=0)
    else:
        _val_na(a, VAL_NA['recon'])

    # ── (b) every reconstructable node, by depth ──────────────────────────────
    b = ax[(0, 1)]
    li, r2, src = _val_causal_nodes(D)
    if li is not None:
        FLOOR = -0.25                       # a node can reconstruct arbitrarily badly
        layers = sorted(set(li.tolist()))
        rng = np.random.default_rng(0)
        for j, l in enumerate(layers):
            y = r2[li == l]
            x = j + (rng.random(len(y)) - 0.5) * (0.24 if len(y) > 1 else 0.0)
            ok = y >= FLOOR
            b.scatter(x[ok], y[ok], s=11, color=C_BFT, edgecolor='none', alpha=0.85,
                      zorder=3)
            b.scatter(x[~ok], np.full((~ok).sum(), FLOOR), s=13, marker='v',
                      color=C_BFT, edgecolor='none', alpha=0.85, zorder=3)
            b.plot([j - 0.28, j + 0.28], [np.median(y)] * 2, color='0.2', lw=0.9,
                   zorder=4)
        b.set_xticks(range(len(layers)))
        b.set_xticklabels([f'$L_{{{l}}}$' for l in layers])
        b.set_xlim(-0.5, len(layers) - 0.5)
        if r2.min() < 0:                    # only spend the space when it is used
            b.axhline(0, color='0.6', lw=0.5, zorder=1)
            b.set_ylim(FLOOR - 0.04, 1.02)
        else:
            b.set_ylim(0, 1.02)
        b.set_yticks([0, 0.5, 1])
        _val_axes(b, xlab='layer (input $\\to$ output)',
                  ylab=r'pre-activation $R^2$')
        txt = (f'median {np.median(r2):.3f}, min {r2.min():.3f}\n'
               f"{len(r2)} node{'s' if len(r2) != 1 else ''}")
        if src == 'FU2':
            txt += ' (fc nodes only)'
        sp = _seed_spread(D, 'median_preact_r2')
        if sp:
            txt += f'\n{sp[2]} model seeds: {sp[0]:.3f}$\\pm${sp[1]:.3f}'
        ctrl = D['recon_controls']
        ho = [(n['refit_heldout']['preact_r2'], n['refit_insample']['preact_r2'])
              for n in ctrl.values() if 'refit_heldout' in n] if isinstance(
                  ctrl, dict) else []
        if ho:
            txt += (f'\nheld-out {np.mean([x[0] for x in ho]):.3f} / in-sample '
                    f'{np.mean([x[1] for x in ho]):.3f}')
        if (r2 < FLOOR).any():
            txt += (f'\n{int((r2 < FLOOR).sum())} off scale below, '
                    f'min {r2.min():.1f}')
        _val_note(b, txt, loc='upper right' if r2.max() < 0.75 else 'lower right')
    else:
        _val_na(b, VAL_NA['recon'])

    # ── (c) NNLS round-trip: do the fixed factors re-explain held-out stimuli ──
    c = ax[(0, 2)]
    rt = D.get('roundtrip')
    if caps['roundtrip'] and isinstance(rt, dict):
        edges, cnt = np.asarray(rt['bin_edges']), np.asarray(rt['counts'])
        c.bar(edges[:-1], cnt, width=np.diff(edges), align='edge', color=C_BFT,
              edgecolor='none', alpha=0.9)
        c.axvline(rt['median'], color='0.2', lw=0.8, ls='--')
        c.set_yscale('log')
        _val_axes(c, xlab='round-trip cosine', ylab='stimuli')
        c.set_xlim(min(float(edges[0]), 0.75), 1.002)
        _val_note(c, f"median {rt['median']:.3f}\nmin {rt['min']:.3f}\n"
                     f"$n={int(rt['n'])}$", loc='upper left')
    else:
        _val_na(c, VAL_NA['roundtrip'])

    # ── (d) do the factors survive a different NMF seed ───────────────────────
    d = ax[(1, 0)]
    per_layer = D['stability']['per_layer']
    keys = sorted(per_layer, key=int)
    shades = _layer_shades(len(keys), C_BFT)
    mu = np.array([per_layer[k]['mean'] for k in keys])
    sd = np.array([per_layer[k]['std'] for k in keys])
    d.bar(np.arange(len(keys)), mu, yerr=sd, color=shades, edgecolor='none',
          width=0.7, error_kw=dict(ecolor='0.25', elinewidth=0.7, capsize=1.4))
    for i, k in enumerate(keys):
        d.text(i, 0.035, f"$K{{=}}{int(per_layer[k]['k'])}$", ha='center', va='bottom',
               fontsize=6, color='white' if mu[i] > 0.25 else '0.3', rotation=90)
    d.axhline(0.85, color='0.35', lw=0.7, ls=':')     # the 0.85 stability gate
    d.set_xticks(range(len(keys)))
    # a 10-layer spine cannot carry 10 tick labels at this width
    step = 1 if len(keys) <= 6 else 2
    d.set_xticklabels([f'$L_{{{k}}}$' if i % step == 0 else ''
                       for i, k in enumerate(keys)])
    d.set_ylim(0, 1.06); d.set_yticks([0, 0.5, 1])
    _val_axes(d, xlab='layer (input $\\to$ output)',
              ylab='matched cosine, %d NMF seeds' % int(D['config']['stab_seeds']))

    # ── (e) and a different rank ──────────────────────────────────────────────
    e = ax[(1, 1)]
    ks = D['stability']['k_sensitivity']
    COLS = [('$K^{*}-1$', np.asarray(ks['k_minus1']), tint(C_BFT, 0.45)),
            ('$K^{*}$', np.asarray(ks['k_star']), C_BFT),
            ('$K^{*}+1$', np.asarray(ks['k_plus1']), tint(C_BFT, 0.45))]
    rng = np.random.default_rng(1)
    for i, (lab, v, col) in enumerate(COLS):
        x = i + (rng.random(len(v)) - 0.5) * 0.3
        e.scatter(x, v, s=9, color=col, edgecolor='none', alpha=0.8, zorder=3)
        e.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color='0.2', lw=0.9, zorder=4)
    e.set_xticks(range(3)); e.set_xticklabels([c[0] for c in COLS])
    e.set_xlim(-0.5, 2.5); e.set_ylim(0, 1.05); e.set_yticks([0, 0.5, 1])
    _val_axes(e, xlab='rank of the re-run NMF',
              ylab='cosine to the $K^{*}$ factors')
    _val_note(e, f"{len(COLS[0][1])} node{'s' if len(COLS[0][1]) != 1 else ''}",
              loc='lower left')

    # ── (f) how much of the arbor each rank explains ──────────────────────────
    f = ax[(1, 2)]
    fu = D['FU1_rank_sweep']['per_layer']
    fkeys = sorted(fu, key=int)
    fshades = _layer_shades(len(fkeys), C_BFT)
    for k, col in zip(fkeys, fshades):
        sw = fu[k]['sweep']
        K = np.array([s['K'] for s in sw]); R = np.array([s['recon_r2'] for s in sw])
        f.plot(K, R, color=col, lw=0.8, alpha=0.9, zorder=2)
        dk = int(fu[k]['default_k'])
        if dk in K:
            f.scatter([dk], [R[list(K).index(dk)]], s=13, color=col, zorder=4,
                      edgecolor='white', linewidth=0.4)
    f.set_ylim(0, 1.05); f.set_yticks([0, 0.5, 1])
    _val_axes(f, xlab='NMF rank $K$', ylab=r'arbor reconstruction $R^2$')
    f.text(0.97, 0.06, 'dot: rank used\nline shade: layer depth', transform=f.transAxes,
           ha='right', va='bottom', fontsize=6, color='0.25', linespacing=1.2)

    # ── (g) is the fingerprint a better code than the activations ─────────────
    g = ax[(2, 0)]
    sep = D['separability']['by_fine']
    dims = D['separability']['dims']
    ROWS = [('BFT fingerprint', 'bft_fingerprint', C_BFT, dims['fingerprint']),
            ('BFT, dim-matched', 'bft_matched', tint(C_BFT, 0.42), dims['matched']),
            ('activations', 'raw_activations', C_BASE, dims['activations']),
            ('act., dim-matched', 'act_matched', tint(C_BASE, 0.42), dims['matched']),
            ('act., rand. proj.', 'act_randproj', tint(C_BASE, 0.68), dims['matched'])]
    ROWS = [r for r in ROWS if r[1] in sep]
    ROWS.append(('shuffled labels', None, C_RAND, None))
    pos = np.arange(len(ROWS))[::-1]
    sil_err = sil_sp[1] if sil_sp else (float(g_boot['sd']) if g_boot else None)
    for p, (lab, key, col, dim) in zip(pos, ROWS):
        rec = sep[key] if key else D['separability']['null_shuffled_labels']
        s, kn = float(rec['silhouette']), float(rec['knn_acc'])
        err = sil_err if key == 'bft_fingerprint' else None
        g.barh(p, s, height=0.66, color=col, edgecolor='none',
               xerr=err, error_kw=dict(ecolor='0.25', elinewidth=0.7, capsize=1.4))
        g.text(max(s, 0) + (err or 0) + 0.03, p, f'{kn:.2f}', va='center', ha='left', fontsize=6,
               color='0.3')
    g.axvline(0, color='0.4', lw=0.5)
    g.set_yticks(pos)
    g.set_yticklabels([f'{r[0]} ({r[3]}d)' if r[3] else r[0] for r in ROWS])
    for t, r in zip(g.get_yticklabels(), ROWS):
        t.set_color(r[2])
    g.set_xlim(-0.08, 1.0)
    _val_axes(g, xlab='silhouette (bar) and 5-NN accuracy (number),\nfine-grained class')
    g.spines['left'].set_visible(False); g.tick_params(axis='y', length=0)
    # What kind of interval that error bar is goes in the header — this panel is
    # a third of the figure wide and every corner of it already holds a bar or a
    # k-NN number.

    # ── (h) does multiplying the weights in earn its place ────────────────────
    h = ax[(2, 1)]
    a1 = D['A1_weight_vs_activation']
    fps = a1['fingerprint_separability']
    pl = a1['per_layer']
    GROUPS = [('fingerprint silhouette', 'o', 22,
               [(fps['activation_nmf']['silhouette'], fps['arbor_nmf']['silhouette'])]),
              ('class selectivity', 's', 11,
               [(v['selectivity_act'], v['selectivity_arbor'])
                for v in pl.values()]),
              ('NMF stability', '^', 11,
               [(v['stability_act'], v['stability_arbor']) for v in pl.values()])]
    h.plot([0, 1], [0, 1], color='0.6', lw=0.6, ls='--', zorder=1)
    for lab, mk, size, pts in GROUPS:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        h.scatter(xs, ys, s=size, marker=mk, color=C_BFT, edgecolor='white',
                  linewidth=0.4, zorder=3, label=lab)
    h.set_xlim(0, 1.03); h.set_ylim(0, 1.03)
    h.set_xticks([0, 0.5, 1]); h.set_yticks([0, 0.5, 1])
    _val_axes(h, xlab='activation-only NMF',
              ylab=r'BFT arbor NMF ($W\!\cdot\!a$)')
    h.legend(fontsize=6, frameon=False, loc='lower right', handlelength=0.6,
             handletextpad=0.2, borderpad=0.1, labelspacing=0.2, borderaxespad=0.2,
             scatterpoints=1)
    # mid-left, above the diagonal: empty in all five models
    h.text(0.04, 0.58, 'above the diagonal:\nthe weight term helps',
           transform=h.transAxes, ha='left', va='bottom', fontsize=6, color='0.3',
           linespacing=1.2)

    # ── (i) against pixel-attribution methods ─────────────────────────────────
    i_ax = ax[(2, 2)]
    att = D['attribution']
    ORDER = [('BFT input map', 'BFT', C_BFT), ('integrated grad.', 'IG', C_BASE),
             ('saliency', 'Saliency', tint(C_BASE, 0.42)),
             ('input magnitude', 'input_mag', tint(C_BASE, 0.68))]
    disc = att['discriminability_fine']
    rows = [(lab, disc[k], col) for lab, k, col in ORDER if k in disc]
    pos = np.arange(len(rows))[::-1]
    for p, (lab, v, col) in zip(pos, rows):
        i_ax.barh(p, v, height=0.66, color=col, edgecolor='none')
    chance = float(D['separability']['null_shuffled_labels']['knn_acc'])
    i_ax.axvline(chance, color='0.35', lw=0.7, ls=':')
    i_ax.text(chance + 0.012, pos[-1] - 0.55, 'chance', fontsize=6, color='0.35',
              ha='left', va='bottom')
    i_ax.set_yticks(pos); i_ax.set_yticklabels([r[0] for r in rows])
    for t, r in zip(i_ax.get_yticklabels(), rows):
        t.set_color(r[2])
    i_ax.set_xlim(0, 1.0)
    # room above the bars for the note when BFT has no pixel-shaped input layer
    i_ax.set_ylim(pos[-1] - 0.75, pos[0] + (1.3 if 'BFT' not in disc else 0.5))
    _val_axes(i_ax, xlab='3-NN accuracy')
    i_ax.spines['left'].set_visible(False); i_ax.tick_params(axis='y', length=0)
    if 'BFT' not in disc:
        _val_note(i_ax, 'BFT input map needs a pixel-shaped\ninput layer (MLPs only)',
                  loc='upper left')

    figstyle.freeze(fig)
    hp_pos = header.get_position()
    fig.text(0.002, hp_pos.y1, f"{D['label']} — {D['arch']}", ha='left', va='top',
             fontweight='bold')
    for i, line in enumerate(sub_lines):
        fig.text(0.002, hp_pos.y1 - 0.021 - 0.015 * i, line, ha='left', va='top',
                 fontsize=6, color='0.35')

    # the row's claim rides on its leading panel label — a separate row header
    # would either cost a text line or collide with the right-hand panel labels
    LABELS = [('(a) faithful: reconstruction controls',
               '(b) reconstruction per node', '(c) projection round-trip'),
              ('(d) robust: stability across NMF seeds',
               '(e) sensitivity to the rank', '(f) rank vs. arbor explained'),
              ('(g) vs. baselines: fingerprint or activations',
               '(h) weights vs. activations only', '(i) vs. pixel attribution')]
    for r, labs in enumerate(LABELS):
        y = spacers[r].get_position().y0
        for cix, lab in enumerate(labs):
            fig.text(max(ax[(r, cix)].get_position().x0 - 0.004, 0.002), y, lab,
                     ha='left', va='bottom', fontweight='bold')
    return fig


# ── registry: figure name -> (bundle, render function, save mode) ─────────────

FIGURES = {
    'fig2_mlp_circuits':   ('nb01_circuits', fig2_mlp_circuits, 'paper'),
    'figA_mlp_details':    ('nb01_circuits', figA_mlp_details, 'appendix'),
    'fig3_digit_mlp_circuits': ('nb02_circuits', fig3_digit_mlp_circuits, 'paper'),
    'figB_digit_mlp_details':  ('nb02_circuits', figB_digit_mlp_details, 'appendix'),
    'fig4_mlp_fingerprints':   ('nb01_fingerprints', fig4_mlp_fingerprints, 'paper'),
    'figC_mlp_fingerprint_details': ('nb01_fingerprints',
                                     figC_mlp_fingerprint_details, 'appendix'),
    'fig5_digit_mlp_fingerprints': ('nb02_fingerprints',
                                    fig5_digit_mlp_fingerprints, 'paper'),
    'figD_digit_mlp_fingerprint_details': ('nb02_fingerprints',
                                           figD_digit_mlp_fingerprint_details,
                                           'appendix'),
    'fig6_cnn_circuits': ('nb03_circuits', fig6_cnn_circuits, 'paper'),
    'figE_cnn_details': ('nb03_circuits', figE_cnn_details, 'appendix'),
    'fig7_cnn_fingerprints': ('nb03_fingerprints', fig7_cnn_fingerprints, 'paper'),
    'figF_cnn_fingerprint_details': ('nb03_fingerprints',
                                     figF_cnn_fingerprint_details, 'appendix'),
    'figG_vit_circuits': ('nb04_circuits', figG_vit_circuits, 'appendix'),
    'figH_vit_fingerprints': ('nb04_fingerprints', figH_vit_fingerprints, 'appendix'),
    'figI_validation_mlp_even_odd': ('nb09_mlp_even_odd_validation',
                                     fig_validation, 'appendix'),
    'figJ_validation_mlp_digit': ('nb09_mlp_digit_validation',
                                  fig_validation, 'appendix'),
    'figK_validation_cnn_cifar': ('nb09_cnn_cifar_validation',
                                  fig_validation, 'appendix'),
    'figL_validation_vit': ('nb09_vit_mnist_validation', fig_validation, 'appendix'),
    'figM_validation_imagenet': ('nb09_imagenet_cnn_validation',
                                 fig_validation, 'appendix'),
}
