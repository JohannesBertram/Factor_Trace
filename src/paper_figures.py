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


# ── Figure 2 — class and sub-class circuits in the 8x4 even/odd MLP ───────────

def fig2_mlp_circuits(D):
    CIRCUITS = _circuit_colors(D)
    LAYER_SIZES, OUT_LABELS = list(D['layer_sizes']), list(D['out_labels'])
    DIGIT_ORDER = list(D['digit_order'])
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    C_INH = figstyle.color('inhibitory')
    DIGIT_COLOR = digit_colors(DIGIT_ORDER, C_EVEN, C_ODD)

    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='paper',
                   height_to_width_ratio=0.545)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.45],
                          height_ratios=[1.0, 1.32], hspace=0.14)

    # (a), (b) — neuron-level circuits
    for i, c in enumerate(CIRCUITS):
        ax = fig.add_subplot(gs[0, i])
        draw_scaffold(ax, c['scaffold']['edges'], c['scaffold']['neg_edges'],
                      c['scaffold']['loading'], LAYER_SIZES, c['color'], C_INH,
                      out_labels=OUT_LABELS, highlight_out=i, legend=(i == 0))
        ax.set_title(f"({'ab'[i]}) {c['name']} circuit", color=c['color'], pad=2,
                     loc='left')

    # (c) — output- and middle-layer factors, laid out as the top of the trace tree
    gsc = gs[0, 2].subgridspec(3, 8, height_ratios=[1.0, 0.22, 1.0])
    ax_c = [fig.add_subplot(gsc[0, 1:3]), fig.add_subplot(gsc[0, 5:7]),
            fig.add_subplot(gsc[2, 0:2]), fig.add_subplot(gsc[2, 2:4]),
            fig.add_subplot(gsc[2, 5:7])]
    fig.add_subplot(gsc[1, :]).set_axis_off()          # room for the tree connectors
    bp = dict(digit_order=DIGIT_ORDER, digit_color=DIGIT_COLOR)
    bar_panel(ax_c[0], CIRCUITS[0]['l3_profile'], xticks=False, ylab='loading',
              title=rf"(c) $L_3\ k_{{{CIRCUITS[0]['k']}}}$", tcolor=C_EVEN, **bp)
    bar_panel(ax_c[1], CIRCUITS[1]['l3_profile'], xticks=False,
              title=rf"$L_3\ k_{{{CIRCUITS[1]['k']}}}$", tcolor=C_ODD, **bp)
    bar_panel(ax_c[2], CIRCUITS[0]['l2_profiles'][0], yticklabels=True,
              title=r'$L_2\ k_0$', tcolor=C_EVEN, **bp)
    bar_panel(ax_c[3], CIRCUITS[0]['l2_profiles'][1], title=r'$L_2\ k_1$',
              tcolor=C_EVEN, **bp)
    bar_panel(ax_c[4], CIRCUITS[1]['l2_profiles'][0], title=r'$L_2\ k_0$',
              tcolor=C_ODD, **bp)

    # (d), (e) — layer-1 sub-circuits: pixel-space arbor + per-digit loading
    gs_bot = gs[1, :].subgridspec(2, 1, height_ratios=[0.13, 1.0])
    sp_de = fig.add_subplot(gs_bot[0]); sp_de.set_axis_off()   # room for block labels
    gsd = gs_bot[1].subgridspec(2, 11, height_ratios=[1.0, 0.45],
                                width_ratios=[1] * 5 + [0.4] + [1] * 5)
    img_axes = {}
    for ci, c in enumerate(CIRCUITS):
        base = 0 if ci == 0 else 6
        cmap = seq_cmap(c['color'], c['name'])
        img_axes[ci] = []
        for k, M in enumerate(c['l1_arbors']):
            axi = fig.add_subplot(gsd[0, base + k])
            axi.imshow(M, cmap=cmap, interpolation='nearest',
                       norm=matplotlib.colors.PowerNorm(0.62, vmin=0,
                                                        vmax=np.percentile(M, 99.3)))
            axi.set_xticks([]); axi.set_yticks([])
            for s in axi.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            axi.text(0.05, 0.97, rf'$k_{k}$', transform=axi.transAxes, ha='left',
                     va='top', color=c['color'], fontsize=6.5)
            img_axes[ci].append(axi)
            bar_panel(fig.add_subplot(gsd[1, base + k]), c['l1_profiles'][k],
                      xticks=(k == 0), ylab='loading' if k == 0 else None, **bp)

    # ── settle constrained_layout, then freeze and add cross-axes annotation ──
    figstyle.freeze(fig)          # draw once, then switch the layout engine off

    for src, dsts in ((ax_c[0], (ax_c[2], ax_c[3])), (ax_c[1], (ax_c[4],))):  # tree edges
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
                 f"({'de'[ci]}) {c['name']} circuit — $L_1$ sub-circuits",
                 ha='left', va='bottom', color=c['color'])
    return fig


# ── Appendix A — decomposition details for the 8x4 MLP ────────────────────────

def figA_mlp_details(D):
    CIRCUITS = _circuit_colors(D)
    C_EVEN, C_ODD = figstyle.color('even'), figstyle.color('odd')
    C_INH = figstyle.color('inhibitory')
    CMAP_INH = seq_cmap(C_INH, 'inh')
    root_colors = [figstyle.color(k) for k in D['root_factor_color_keys']]

    anchors = {}
    figstyle.apply(venue='aaai2024', width='full', nrows=3, ncols=3, mode='appendix',
                   height_to_width_ratio=0.62)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    # rows 0/2/4 are thin spacers that reserve room for the panel labels, which are
    # placed after the layout is frozen (a long label inside an axes would inflate
    # that grid cell and squeeze the image rows).
    gs = fig.add_gridspec(6, 1, height_ratios=[0.10, 0.60, 0.10, 0.90, 0.11, 0.70],
                          hspace=0.03)
    spacers = [fig.add_subplot(gs[r]) for r in (0, 2, 4)]
    for sp in spacers:
        sp.set_axis_off()
    gs_top = gs[1].subgridspec(1, 2, width_ratios=[1.0, 1.32])

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

    # ── (b) connection maps: excitatory vs inhibitory, L3 and L2 ─────────────
    gsb = gs_top[0, 1].subgridspec(2, 5, width_ratios=[0.40] + [1] * 4)
    cols_b = [(rf"$L_3\ k_{{{CIRCUITS[0]['k']}}}$", CIRCUITS[0], 'l3_conn', C_EVEN),
              (rf"$L_3\ k_{{{CIRCUITS[1]['k']}}}$", CIRCUITS[1], 'l3_conn', C_ODD),
              (r'$L_2$ even $k_0$', CIRCUITS[0], 'l2_conn', C_EVEN),
              (r'$L_2$ odd $k_0$',  CIRCUITS[1], 'l2_conn', C_ODD)]
    for r, lab in enumerate(('excitatory', 'inhibitory')):
        row_label(fig, gsb[r, 0], lab)
    for j, (name, c, key, col) in enumerate(cols_b):
        for r, (M, cmap) in enumerate(((c[key], seq_cmap(col, f'c{j}')),
                                       (c[key + '_neg'], CMAP_INH))):
            n_out, n_in = M.shape
            ax = fig.add_subplot(gsb[r, j + 1])
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
                    anchors['b'] = ax
            elif j == 1:
                ax.set_xlabel('input unit', labelpad=1)

    # ── (c) per-unit pixel receptive fields of the dominant L1 factor ─────────
    gsc = gs[3].subgridspec(2, 9, width_ratios=[0.30] + [1] * 8)
    for ci, c in enumerate(CIRCUITS):
        cmap = seq_cmap(c['color'], f'rf{ci}')
        share = c['l1_unit_share']
        row_label(fig, gsc[ci, 0], f"{c['name']}\ncircuit $k_0$", color=c['color'])
        for ni, rf in enumerate(c['l1_unit_rf']):
            ax = fig.add_subplot(gsc[ci, ni + 1])
            if ci == 0 and ni == 0:
                anchors['c'] = ax
            show_map(ax, rf, cmap, pct=99.5)
            ax.text(0.05, 0.97, f'{ni}', transform=ax.transAxes, ha='left', va='top',
                    color=c['color'], fontsize=6.5)
            ax.text(0.95, 0.97, rf'{100 * share[ni]:.0f}\%', transform=ax.transAxes,
                    ha='right', va='top', color='0.35', fontsize=6)

    # ── (d, e) weighted-average stimulus and inhibitory arbor per L1 factor ──
    gsd = gs[5].subgridspec(2, 12, width_ratios=[0.42] + [1] * 5 + [0.4] + [1] * 5)
    for r, lab in enumerate(('weighted avg.\nstimulus', 'inhibitory\narbor')):
        row_label(fig, gsd[r, 0], lab)
    first_img = {}
    for ci, c in enumerate(CIRCUITS):
        base = 1 if ci == 0 else 7
        for k, wavg in enumerate(c['l1_wavg']):
            ax = fig.add_subplot(gsd[0, base + k])
            ax.imshow(wavg, cmap='gray_r', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            ax.text(0.05, 0.97, rf'$k_{k}$', transform=ax.transAxes, ha='left',
                    va='top', color=c['color'], fontsize=6.5)
            if k == 0:
                first_img[ci] = ax
            show_map(fig.add_subplot(gsd[1, base + k]), c['l1_neg_arbors'][k], CMAP_INH)

    # ── settle constrained_layout, then freeze and add the panel labels ──────
    figstyle.freeze(fig)          # draw once, then switch the layout engine off
    spacer_label(fig, anchors['a'], spacers[0], '(a) factor spectra', dx=-0.030)
    spacer_label(fig, anchors['b'], spacers[0],
                 r'(b) connection maps (out $\times$ in)', dx=-0.035)
    spacer_label(fig, anchors['c'], spacers[1],
                 '(c) $L_1$ pixel receptive fields', dx=-0.030)
    for ci, c in enumerate(CIRCUITS):
        spacer_label(fig, first_img[ci], spacers[2],
                     f"({'de'[ci]}) {c['name']} circuit — $L_1$ factors",
                     color=c['color'], dx=-0.006)
    return fig




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
                          width_ratios=[1.28, 1.10, 1.04, 1.14])
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
                      rf'$L_{{{lay + 1}}}$', ha='center', va='top', fontsize=6.5)
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

    # (b) pairwise fingerprint similarity, stimuli sorted by digit
    ax_b = fig.add_subplot(gs[1, 1])
    PER = int(D['sel_per_digit'])
    N_VIZ = len(D['fp_sel'])
    _U = unit(D['fp_sel'])
    ax_b.imshow(_U @ _U.T, cmap=CMAP_SIM, vmin=0, vmax=1, interpolation='nearest')
    for b in range(PER, N_VIZ, PER):
        ax_b.axhline(b - 0.5, color='0.25', lw=0.5)
        ax_b.axvline(b - 0.5, color='0.25', lw=0.5)
    ctr = (np.arange(len(DIGIT_ORDER)) + 0.5) * PER
    ax_b.set_xticks(ctr); ax_b.set_xticklabels(DIGIT_ORDER)
    ax_b.set_yticks(ctr); ax_b.set_yticklabels(DIGIT_ORDER)
    for tt in (ax_b.get_xticklabels(), ax_b.get_yticklabels()):
        for t, d in zip(tt, DIGIT_ORDER):
            t.set_color(DIGIT_COLOR[d])
    ax_b.tick_params(length=0, pad=1)
    for s in ax_b.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_b.set_xlabel('stimulus digit', labelpad=1)
    ax_b.text(0.96, 0.96, f'silhouette\n{SIL_CLASS:.2f} class\n{SIL_DIGIT:.2f} digit',
              transform=ax_b.transAxes, ha='right', va='top', fontsize=6,
              linespacing=1.15, color='0.15',
              bbox=dict(fc='white', ec='none', alpha=0.82, pad=1.0))

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
    for ax, lab in ((ax_a, '(a) mean fingerprint'), (ax_b, '(b) fingerprint similarity'),
                    (ax_c, '(c) held-out digits'), (ax_d, '(d) far-OOD collapse')):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), sp.get_position().y0,
                 lab, ha='left', va='bottom')
    return fig


# ── Appendix C — fingerprint details for the 8x4 MLP ──────────────────────────

def figC_mlp_fingerprint_details(D):
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
    SEP = [(s['label'], s['silhouette'], s['knn'], int(s['dim']),
            resolve_color(s['color_key'])) for s in D['sep']]
    pca = D['pca']

    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='appendix',
                   height_to_width_ratio=0.80)
    fig = plt.figure()
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.014,
                          hspace=0.03, wspace=0.03)
    gs = fig.add_gridspec(4, 3, height_ratios=[0.10, 1.0, 0.10, 1.0], hspace=0.10)
    sp1 = fig.add_subplot(gs[0, :]); sp1.set_axis_off()
    sp2 = fig.add_subplot(gs[2, :]); sp2.set_axis_off()

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
    ax_c = fig.add_subplot(gs[3, 0])
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

    # (d) digit separability of the fingerprint vs raw activations
    ax_d = fig.add_subplot(gs[3, 1])
    x = np.arange(len(SEP))
    ax_d.bar(x - 0.2, [s[1] for s in SEP], width=0.38, color=[s[4] for s in SEP],
             edgecolor='none')
    ax_d.bar(x + 0.2, [s[2] for s in SEP], width=0.38, color=[s[4] for s in SEP],
             edgecolor='none', alpha=0.45)
    _h = [matplotlib.patches.Patch(fc='0.5', ec='none', label='silhouette'),
          matplotlib.patches.Patch(fc='0.5', ec='none', alpha=0.45,
                                   label='5-NN accuracy')]
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f'{s[0]}\n({s[3]}d)' for s in SEP], fontsize=6)
    ax_d.set_ylim(0, 1.05)
    ax_d.set_ylabel('digit separability', labelpad=1)
    ax_d.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_d.spines[s].set_visible(False)
    ax_d.legend(handles=_h, fontsize=6, frameon=False, loc='upper right',
                handlelength=0.8, handletextpad=0.35, borderpad=0.1, labelspacing=0.2,
                borderaxespad=0.2)

    # (e) how close each condition gets to any trained digit
    ax_e = fig.add_subplot(gs[3, 2])
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
                            (ax_d, '(d) digit separability', sp2),
                            (ax_e, '(e) distance to the trained digits', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), anchor.get_position().y0,
                 lab, ha='left', va='bottom')
    return fig


# ── Figure 3 — class circuits in the 40x20 digit MLP ──────────────────────────

def fig3_digit_mlp_circuits(D):
    CIRCUITS = D['circuits']
    n_c, N_SHOW, N_DIGITS = len(CIRCUITS), int(D['n_show']), int(D['n_digits'])
    LAYER_SIZES = list(D['layer_sizes'])
    Sup, root_pur = D['support'], D['root_pur']
    l1_pur = [c['l1_pur'] for c in CIRCUITS]
    l1_lam = [c['l1_lam'] for c in CIRCUITS]
    C_BFT = figstyle.color('ours')
    C_L3, C_L1 = C_BFT, tint(C_BFT, 0.45)
    CMAP = seq_cmap(C_BFT, 'bft')

    figstyle.apply(venue='aaai2024', width='full', nrows=2, ncols=3, mode='paper',
                   height_to_width_ratio=0.74)
    fig = plt.figure()
    # NOTE: wspace/hspace must not be passed to subgridspec() (silently disables
    # constrained_layout in mpl 3.10) — inner spacing is set on the layout engine.
    fig.set_layout_engine('constrained', h_pad=0.012, w_pad=0.012,
                          hspace=0.03, wspace=0.02)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.74, 1.46, 0.09], hspace=0.24)
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.45])
    gs_bot = gs[1].subgridspec(1, 2, width_ratios=[2.95, 1.0])
    fig.add_subplot(gs[2]).set_axis_off()   # room for (d)'s tick labels once aligned

    # (a) output-layer factor x digit loading
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
    cb = fig.colorbar(im, ax=ax_a, fraction=0.045, pad=0.04)
    cb.set_label('digit share', labelpad=3)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1)

    # (b) which layer-1 units each circuit recruits
    ax_b = fig.add_subplot(gs_top[0, 1])
    unit_order = np.argsort(-Sup.mean(0))
    ax_b.imshow((Sup / Sup.max(1, keepdims=True))[:, unit_order], cmap=CMAP,
                aspect='auto', vmin=0, vmax=1)
    ax_b.set_xticks([])
    ax_b.set_yticks(range(n_c))
    ax_b.set_yticklabels([f"$f_{c['k']}$" for c in CIRCUITS])
    ax_b.set_xlabel(f'{LAYER_SIZES[0]} $L_1$ units, sorted by mean use', labelpad=2)
    style_matrix_axes(ax_b)
    ax_b.set_title('(b) $L_1$ units per circuit', loc='left', pad=2)

    # (d) digit purity: output factor vs its layer-1 factors
    ax_d = fig.add_subplot(gs_bot[0, 1])
    for i, c in enumerate(CIRCUITS):
        ax_d.scatter(np.full(len(l1_pur[i]), i), l1_pur[i], s=4 + 90 * l1_lam[i],
                     facecolor=C_L1, edgecolor='none', alpha=0.85, zorder=2)
    ax_d.scatter(range(n_c), root_pur, marker='_', s=42, color=C_L3, linewidths=1.4,
                 zorder=3, label='output factor')
    ax_d.scatter([], [], s=18, facecolor=C_L1, edgecolor='none', label='$L_1$ factors')
    ax_d.axhline(0.1, color='0.6', lw=0.5, ls=(0, (2, 2)), zorder=1)
    ax_d.text(n_c - 0.6, 0.115, 'chance', fontsize=6, color='0.45', ha='right')
    ax_d.set_xticks(range(n_c))
    ax_d.set_xticklabels([f"$f_{c['k']}$" for c in CIRCUITS])
    ax_d.set_xlim(-0.6, n_c - 0.4)
    ax_d.set_ylim(0, 1.0)
    ax_d.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_d.spines[s].set_visible(False)
    ax_d.legend(fontsize=6, frameon=False, loc='upper left', handlelength=1.0,
                handletextpad=0.4, borderpad=0.1, labelspacing=0.25)

    # (c) pixel arbors of the strongest layer-1 factors, per circuit
    gs_left = gs_bot[0, 0].subgridspec(2, 1, height_ratios=[0.14, 1.0])
    sp_c = fig.add_subplot(gs_left[0]); sp_c.set_axis_off()   # room for the (c) label
    gsd = gs_left[1].subgridspec(N_SHOW, n_c)
    arbor_axes = []
    for j, c in enumerate(CIRCUITS):
        for r in range(N_SHOW):
            ax = fig.add_subplot(gsd[r, j])
            M = c['l1_arbors'][r]
            ax.imshow(M, cmap=CMAP, interpolation='nearest',
                      norm=matplotlib.colors.PowerNorm(0.62, vmin=0,
                                                       vmax=np.percentile(M, 99.3)))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color('0.6'); s.set_linewidth(0.4)
            p = c['l1_profiles'][r]
            ax.text(0.05, 0.97, rf'$k_{r}$', transform=ax.transAxes, ha='left',
                    va='top', fontsize=6.5, color=C_BFT)
            ax.text(0.95, 0.97, f'{p.argmax()}', transform=ax.transAxes, ha='right',
                    va='top', fontsize=7, color='0.15', fontweight='bold')
            if r == 0:
                ax.set_title(r'$f_{' + str(c['k']) + r'}$: ' +
                             ','.join(str(int(d)) for d in c['pooled']), pad=2,
                             fontsize=7)
                arbor_axes.append(ax)
            if r == N_SHOW - 1 and j == 0:
                arbor_bottom = ax

    figstyle.freeze(fig)
    # (c) and (d) sit side by side: give them the same top and bottom edge. The
    # arbor grid loses height to its own label row and column headers, so (d) is
    # matched to the image block explicitly.
    top, bot = arbor_axes[0].get_position().y1, arbor_bottom.get_position().y0
    p_d = ax_d.get_position()
    ax_d.set_position([p_d.x0, bot, p_d.width, top - bot])
    for anchor, label in ((arbor_axes[0], '(c) $L_1$ sub-circuits'),
                          (ax_d, '(d) digit purity')):
        fig.text(max(anchor.get_position().x0 - 0.006, 0.002),
                 sp_c.get_position().y0, label, ha='left', va='bottom')
    return fig


# ── Appendix B — decomposition details for the 40x20 digit MLP ────────────────

def figB_digit_mlp_details(D):
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
        ax.set_title(rf"$f_{c['k']}$: " + ','.join(str(int(d)) for d in c['pooled']),
                     pad=1.5, fontsize=7)
        if j == 0:
            anchors['b'] = ax

    # ── (c) digit profile of every layer-1 factor, per circuit ──────────────
    gsc = gs_bot[0, 0].subgridspec(1, n_c)
    for j, c in enumerate(CIRCUITS):
        ax = fig.add_subplot(gsc[0, j])
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
    gsd = gs_bot[0, 1].subgridspec(N_SHOW, 1 + n_c, width_ratios=[0.30] + [1] * n_c)
    for r in range(N_SHOW):
        row_label(fig, gsd[r, 0], rf'$k_{r}$')
    for j, c in enumerate(CIRCUITS):
        for r in range(N_SHOW):
            ax = fig.add_subplot(gsd[r, j + 1])
            ax.imshow(c['l1_wavg'][r], cmap='gray_r', interpolation='nearest')
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
    _label('d', 2, '(d) weighted-average input of the $L_1$ factors of Fig. 3(c)',
           dx=-0.022)
    return fig


# ── Figure 5 — factor fingerprints in the 40x20 digit MLP ─────────────────────

def fig5_digit_mlp_fingerprints(D):
    N_CLASSES = int(D['n_classes'])
    COL_ORDER, BLK_EDGE = list(D['col_order']), list(D['blk_edge'])
    BLOCK_SIZES = list(D['block_sizes'])
    C_BFT, C_NEAR = figstyle.color('ours'), figstyle.color('near_ood')
    CMAP = ramp_cmap(C_BFT, 'bft')
    SIL, R_OOD, AGREE = D['sil'], D['r_ood'], D['agree']
    P_model, P_fprint = D['p_model'], D['p_fprint']
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
                          width_ratios=[1.50, 1.04, 0.99, 1.20])
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

    # (b) pairwise fingerprint similarity, stimuli sorted by digit
    ax_b = fig.add_subplot(gs[1, 1])
    PER = int(D['sel_per_digit'])
    _U = unit(D['fp_sel'])
    ax_b.imshow(_U @ _U.T, cmap=CMAP, vmin=0, vmax=1, interpolation='nearest')
    for b in range(PER, PER * N_CLASSES, PER):
        ax_b.axhline(b - 0.5, color='0.35', lw=0.3)
        ax_b.axvline(b - 0.5, color='0.35', lw=0.3)
    ctr = (np.arange(N_CLASSES) + 0.5) * PER
    ax_b.set_xticks(ctr); ax_b.set_xticklabels(range(N_CLASSES), fontsize=6)
    ax_b.set_yticks(ctr); ax_b.set_yticklabels(range(N_CLASSES), fontsize=6)
    ax_b.tick_params(length=0, pad=1)
    for s in ax_b.spines.values():
        s.set_color('0.6'); s.set_linewidth(0.4)
    ax_b.set_xlabel('stimulus digit', labelpad=1)
    ax_b.text(0.96, 0.96, f'silhouette {SIL:.2f}', transform=ax_b.transAxes,
              ha='right', va='top', fontsize=6, color='0.15',
              bbox=dict(fc='white', ec='none', alpha=0.82, pad=1.0))

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
    for ax, lab in ((ax_a, '(a) mean fingerprint'), (ax_b, '(b) fingerprint similarity'),
                    (ax_c, '(c) Fashion-MNIST'), (ax_d, '(d) far-OOD collapse')):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), sp.get_position().y0,
                 lab, ha='left', va='bottom')
    return fig


# ── Appendix D — fingerprint details for the 40x20 digit MLP ──────────────────

def figD_digit_mlp_fingerprint_details(D):
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
    SEP = [(s['label'], s['silhouette'], s['knn'], int(s['dim']),
            resolve_color(s['color_key'])) for s in D['sep']]
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
    ax_c = fig.add_subplot(gs[3, 0])
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

    # (d) digit separability of the fingerprint vs raw activations
    ax_d = fig.add_subplot(gs[3, 1])
    x = np.arange(len(SEP))
    ax_d.bar(x - 0.2, [s[1] for s in SEP], width=0.38, color=[s[4] for s in SEP],
             edgecolor='none')
    ax_d.bar(x + 0.2, [s[2] for s in SEP], width=0.38, color=[s[4] for s in SEP],
             edgecolor='none', alpha=0.45)
    _h = [matplotlib.patches.Patch(fc='0.5', ec='none', label='silhouette'),
          matplotlib.patches.Patch(fc='0.5', ec='none', alpha=0.45, label='5-NN accuracy')]
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([f'{s[0]}\n({s[3]}d)' for s in SEP], fontsize=6)
    ax_d.set_ylim(0, 1.05)
    ax_d.set_ylabel('digit separability', labelpad=1)
    ax_d.tick_params(length=1.5, pad=1)
    for s in ('top', 'right'):
        ax_d.spines[s].set_visible(False)
    ax_d.legend(handles=_h, fontsize=6, frameon=False, loc='upper right',
                handlelength=0.8, handletextpad=0.35, borderpad=0.1, labelspacing=0.2,
                borderaxespad=0.2)

    # (e) how close each condition gets to any trained digit
    ax_e = fig.add_subplot(gs[3, 2])
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
                            (ax_d, '(d) digit separability', sp2),
                            (ax_e, '(e) distance to the trained digits', sp2)):
        fig.text(max(ax.get_position().x0 - 0.004, 0.002), anchor.get_position().y0,
                 lab, ha='left', va='bottom')
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
}
