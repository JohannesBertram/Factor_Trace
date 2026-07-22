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

    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=4, mode='paper',
                   height_to_width_ratio=0.94)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.02, wspace=0.03)
    gs = fig.add_gridspec(2, 4, height_ratios=[0.11, 1.0],
                          width_ratios=[1.42, 1.10, 0.96, 1.16])
    sp = fig.add_subplot(gs[0, :]); sp.set_axis_off()      # room for the panel labels

    # (a) mean fingerprint per digit, columns grouped by circuit then layer
    ax_a = fig.add_subplot(gs[1, 0])
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
    ax_a.set_xlabel('factor, by circuit and layer', labelpad=8)
    ax_a.tick_params(length=1.5, pad=1)
    for s in ax_a.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_a.text((N_EV - 1) / 2, -0.72, 'even circuit', ha='center', va='bottom',
              color=C_EVEN, fontsize=6.5)
    ax_a.text((N_EV + len(COL_ORDER) - 1) / 2, -0.72, 'odd circuit', ha='center',
              va='bottom', color=C_ODD, fontsize=6.5)
    ax_a.set_ylim(len(DIGIT_ORDER) + 0.25, -1.1)

    # (b) is the fingerprint worth it? — digit separability against the raw
    #     activations of the same network, at 1/60 of the dimensions
    ax_b = fig.add_subplot(gs[1, 1])
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

    # (c) held-out digits: the fingerprint tracks the network's decision, not the label
    ax_c = fig.add_subplot(gs[1, 2])
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

    # (d) far-OOD: every stimulus collapses onto one fingerprint
    ax_d = fig.add_subplot(gs[1, 3])
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
    for ax, lab in ((ax_a, '(a) mean fingerprint'), (ax_b, '(b) digit separability'),
                    (ax_c, '(c) held-out digits'), (ax_d, '(d) far-OOD collapse')):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), sp.get_position().y0,
                 lab, ha='left', va='bottom')
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

    # (c) the same fingerprints in the plane of their first two principal components
    ax_c = fig.add_subplot(gs_bot[0, 0])
    for xy in pca['far']:
        ax_c.scatter(*xy['coords'].T, s=3.5, color=C_FAR, alpha=0.55,
                     edgecolor='none', zorder=1)
    ax_c.scatter(*pca['ood_coords'].T, s=3.5, color=C_NEAR, alpha=0.5,
                 edgecolor='none', zorder=2)
    for g in pca['digits']:
        d = int(g['digit'])
        ax_c.scatter(*g['coords'].T, s=3.5, color=DIGIT_COLOR[d], alpha=0.75,
                     edgecolor='none', zorder=3, label=f'digit {d}')
    ax_c.scatter([], [], s=8, color=C_NEAR, label='held-out')
    ax_c.scatter([], [], s=8, color=C_FAR, label='far-OOD')
    ax_c.set_xlabel(f'PC1 ({pca["evr"][0] * 100:.0f}' + r'\%)', labelpad=1)
    ax_c.set_ylabel(f'PC2 ({pca["evr"][1] * 100:.0f}' + r'\%)', labelpad=1)
    ax_c.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_c.spines[s].set_visible(False)
    ax_c.legend(fontsize=6, frameon=False, loc='upper left', handlelength=0.7,
                handletextpad=0.25, borderpad=0.1, labelspacing=0.18, ncol=2,
                columnspacing=0.6, borderaxespad=0.15, scatterpoints=1)

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

    figstyle.apply(venue='aaai2024', width='full', nrows=1, ncols=4, mode='paper',
                   height_to_width_ratio=0.94)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.02, wspace=0.03)
    gs = fig.add_gridspec(2, 4, height_ratios=[0.11, 1.0],
                          width_ratios=[1.40, 1.12, 0.96, 1.18])
    sp = fig.add_subplot(gs[0, :]); sp.set_axis_off()      # room for the panel labels

    # (a) mean fingerprint per digit, columns grouped by circuit
    ax_a = fig.add_subplot(gs[1, 0])
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
    ax_a.set_ylabel('stimulus digit', labelpad=1)
    ax_a.set_xlabel(f'{n_factors} factors, grouped by circuit', labelpad=7)
    ax_a.tick_params(length=1.5, pad=1)
    for s in ax_a.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_a.set_ylim(N_CLASSES + 0.55, -0.5)

    # (b) is the fingerprint worth it? — at ten classes, not on separability: the
    #     20-d penultimate activations beat it. Reported as it is, not spun.
    ax_b = fig.add_subplot(gs[1, 1])
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

    # (c) Fashion-MNIST: the fingerprint reports the network's own (wrong) digit
    ax_c = fig.add_subplot(gs[1, 2])
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

    # (d) far-OOD: every stimulus collapses onto one fingerprint
    ax_d = fig.add_subplot(gs[1, 3])
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
    for ax, lab in ((ax_a, '(a) mean fingerprint'), (ax_b, '(b) digit separability'),
                    (ax_c, '(c) Fashion-MNIST'), (ax_d, '(d) far-OOD collapse')):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), sp.get_position().y0,
                 lab, ha='left', va='bottom')
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
    gs_bot = gs[3, :].subgridspec(1, 3, width_ratios=[1.20, 0.72, 1.08])

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
    ax_c = fig.add_subplot(gs_bot[0, 0])
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
    ax_d = fig.add_subplot(gs_bot[0, 1])
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
    ax_e.set_xlim(0.32, 1.03); ax_e.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax_e.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_e.spines[s].set_visible(False)

    figstyle.freeze(fig)
    for ax, lab, anchor in ((ax_a, '(a) NNLS round-trip', sp1),
                            (ax_b, '(b) mean fingerprint per condition', sp1),
                            (ax_c, '(c) digits vs Fashion-MNIST', sp2),
                            (ax_d, '(d) fingerprint similarity', sp2),
                            (ax_e, '(e) distance to the trained digits', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), anchor.get_position().y0,
                 lab, ha='left', va='bottom')
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

    # (e) the same fingerprints in the plane of their first two principal components
    ax_e = fig.add_subplot(gs[5, 0])
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
    ax_e.set_xlabel(f'PC1 ({evr[0] * 100:.0f}' + r'\%)', labelpad=1)
    ax_e.set_ylabel(f'PC2 ({evr[1] * 100:.0f}' + r'\%)', labelpad=1)
    ax_e.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_e.spines[s].set_visible(False)
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
            (ax_e, '(e) fingerprint PCA', spacers[2], 0.0),
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

    figstyle.apply(venue='aaai2024', width='full', nrows=3, ncols=3, mode='appendix',
                   height_to_width_ratio=0.80)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.014, w_pad=0.016,
                          hspace=0.03, wspace=0.03)
    gs = fig.add_gridspec(7, 3,
                          height_ratios=[0.20, 0.10, 1.0, 0.10, 1.0, 0.10, 1.0],
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
    sil_sp = _seed_spread(D, 'silhouette')
    for p, (lab, key, col, dim) in zip(pos, ROWS):
        rec = sep[key] if key else D['separability']['null_shuffled_labels']
        s, kn = float(rec['silhouette']), float(rec['knn_acc'])
        err = sil_sp[1] if (key == 'bft_fingerprint' and sil_sp) else None
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
    hp = D.get('final_hp') if isinstance(D.get('final_hp'), dict) else {}
    kw = hp.get('bft_kwargs', {})
    sub = (f"$n={int(D['config']['n_samples'])}$ stimuli"
           + (f", $K_{{\\max}}={[int(v) for v in kw['k_max']]}$" if 'k_max' in kw else '')
           + (f", threshold ${float(kw['stimulus_threshold']):g}$"
              if 'stimulus_threshold' in kw else '')
           + (f", {int(hp['fingerprint_dim'])}-d fingerprint"
              if 'fingerprint_dim' in hp else ''))
    hp_pos = header.get_position()
    fig.text(0.002, hp_pos.y1, f"{D['label']} — {D['arch']}", ha='left', va='top',
             fontweight='bold')
    fig.text(0.002, hp_pos.y1 - 0.021, sub, ha='left', va='top', fontsize=6,
             color='0.35')

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
