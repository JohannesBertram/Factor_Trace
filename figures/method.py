"""Not reviewed code. The BFT method step by step, as dark slides on a toy even/odd network.

Writes figures/slides/<name>.pdf; a slide with builds writes <name>_1, <name>_2, ... on one shared
canvas, so the builds overlay. Each group of consecutive slides that show the same plot is centered
as one (style.publish), so its shared content stays put and the full last build is centered.
The network has the sizes of the paper's MLP (8 x 4 x 2 on digits 0, 4, 1, 3) but its activations
and weights are a seeded toy, not the trained model; the factorizations are real NMF fits on the
toy arbors.

    PYTHONPATH=. .venv/bin/python figures/method.py
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from figures.style import COLOR, publish, slide

AR = 0.5                                   # every slide, so the deck has one size
DIGITS = [0, 4, 1, 3]                      # stimulus rows, even first
CLASS = {0: 'even', 4: 'even', 1: 'odd', 3: 'odd'}
DETECTS = [0, 0, 4, 4, 1, 1, 3, 3]         # the digit each hidden-1 neuron of the toy responds to


# ---------------------------------------------------------------- toy network
def nmf(M: np.ndarray, k: int, seed: int = 0, iters: int = 3000):
    """Frobenius NMF by multiplicative updates; unit-norm factors sorted by informativity lambda."""
    rng = np.random.default_rng(seed)
    w, h = rng.random((M.shape[0], k)) + 0.1, rng.random((k, M.shape[1])) + 0.1
    for _ in range(iters):
        h *= (w.T @ M) / (w.T @ w @ h + 1e-12)
        w *= (M @ h.T) / (w @ h @ h.T + 1e-12)
    nw, nh = np.linalg.norm(w, axis=0), np.linalg.norm(h, axis=1)
    lam = nw * nh
    o = np.argsort(-lam)
    return (w / nw)[:, o], (h / nh[:, None])[o].T, lam[o]


def toy() -> dict:
    """Activations, weights, arbors and factorizations of the toy 8 x 4 x 2 network."""
    rng = np.random.default_rng(0)
    A1 = np.array([[0.85 if d == s else 0.05 for d in DETECTS] for s in DIGITS])
    A1 = A1 + rng.uniform(0, 0.08, A1.shape)
    W2 = np.array([[0.9, 0.8, 0.5, 0.4, 0, 0, -0.2, -0.1],      # hidden 2: two even, two odd
                   [0.4, 0.5, 0.9, 0.8, -0.1, -0.2, 0, 0],      # neurons, each preferring
                   [0, -0.1, -0.2, 0, 0.9, 0.8, 0.5, 0.4],      # one digit of its class
                   [-0.1, 0, 0, -0.2, 0.4, 0.5, 0.9, 0.8]])
    A2 = np.clip(A1 @ W2.T, 0, None)
    A2 /= A2.max()
    W3 = np.array([[1.0, 0.8, -0.5, -0.4], [-0.4, -0.5, 0.9, 1.0]])

    J3 = (np.clip(W3, 0, None)[None] * A2[:, None, :]).reshape(4, -1)
    w3, h3, lam3 = nmf(J3, 2)
    if w3[0, 0] < w3[0, 1]:                                    # factor 0 is the even one
        w3, h3, lam3 = w3[:, ::-1], h3[:, ::-1], lam3[::-1]

    wt = lam3 * w3
    share = wt[:, 0] / (wt.sum(1) + 1e-9)                      # selectivity for the even factor
    pi = np.where(share >= np.quantile(share, 0.5), share, 0)  # tau = 0.5 gate
    J2 = pi[:, None] * (np.clip(W2, 0, None)[None] * A1[:, None, :]).reshape(4, -1)
    w2, h2, lam2 = nmf(J2, 2)
    if w2[0, 0] < w2[0, 1]:                                    # child 0 is the "0" factor
        w2, h2, lam2 = w2[:, ::-1], h2[:, ::-1], lam2[::-1]
    return dict(A1=A1, A2=A2, W2=W2, W3=W3, J3=J3, w3=w3, h3=h3, share=share, pi=pi, J2=J2,
                w2=w2, h2=h2)


# ---------------------------------------------------------------- drawing
def canvas():
    """Full-bleed axes in inches, y pointing down; a fixed canvas so builds overlay."""
    fig = slide(AR)
    plt.rcParams['savefig.bbox'] = 'standard'
    W, H = fig.get_size_inches()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def mix(c, v: float, base=None):
    """Color c at intensity v in [0, 1], fading into the empty-cell grey."""
    b = np.array(mcolors.to_rgb(base or COLOR['grid']))
    return tuple(b + (np.array(mcolors.to_rgb(c)) - b) * float(np.clip(v, 0, 1)))


def digit_color(d: int):
    """0 and 1 in the full class hue, 4 and 3 in a lighter shade of it."""
    c = np.array(mcolors.to_rgb(COLOR[CLASS[d]]))
    return tuple(c + (1 - c) * 0.45) if d in (4, 3) else tuple(c)


def txt(ax, x, y, s, size=8, color=None, **kw):
    kw.setdefault('ha', 'center')
    kw.setdefault('va', 'center')
    return ax.text(x, y, s, fontsize=size, color=color or COLOR['fg'], zorder=8, **kw)


def arrow(ax, p, q, color=None, rad=0.0, lw=1.0, ms=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, lw=lw,
                                 color=color or COLOR['fg'], zorder=6, shrinkA=0, shrinkB=0,
                                 connectionstyle=f'arc3,rad={rad}'))


def box(ax, x0, y0, x1, y1, color, lw=1.2, z=5):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=lw, zorder=z))


def cells(ax, x0, y0, M, color, cw, ch=None, vmax=None, gap=0.012):
    """Matrix of cells, top-left at (x0, y0); color is one color or a function of (row, col)."""
    ch = ch or cw
    M = np.asarray(M, float)
    vmax = vmax or M.max()
    for i, j in np.ndindex(M.shape):
        c = color(i, j) if callable(color) else color
        ax.add_patch(Rectangle((x0 + j * cw + gap / 2, y0 + i * ch + gap / 2), cw - gap, ch - gap,
                               fc=mix(c, M[i, j] / vmax), lw=0, zorder=3))
    return x0, y0, x0 + M.shape[1] * cw, y0 + M.shape[0] * ch


def tile(ax, x, y, s, d, border=None, dim=False, lw=1.0):
    """A digit chip of width s centred at (x, y)."""
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fc=COLOR['bg'], lw=lw, zorder=5,
                           ec=border or COLOR['edge']))
    txt(ax, x, y, str(d), size=s * 72 * 0.62, color=COLOR['faint'] if dim else COLOR['fg'],
        family='monospace')


def node(ax, p, r, ec=None, fc=None, lw=1.0):
    ax.add_patch(Circle(p, r, fc=fc or COLOR['bg'], ec=ec or COLOR['muted'], lw=lw, zorder=4))


def edge(ax, p, q, color, lw, z=2):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, solid_capstyle='round', zorder=z)


class Net:
    """The 8 x 4 x 2 network: node positions, faint edges, and overlays."""

    def __init__(self, ax, xs, yc, dy=0.19, r=0.065, sizes=(8, 4, 2)):
        self.ax, self.r = ax, r
        self.pos = [np.c_[np.full(n, x), yc + (np.arange(n) - (n - 1) / 2) * dy]
                    for x, n in zip(xs, sizes)]
        for a, b in zip(self.pos[:-1], self.pos[1:]):
            for p in a:
                for q in b:
                    edge(ax, p, q, COLOR['edge'], 0.4, z=1)

    def nodes(self, fills=None, rings=None):
        """fills / rings: {(layer, unit): color}."""
        fills, rings = fills or {}, rings or {}
        for l, P in enumerate(self.pos):
            for u, p in enumerate(P):
                node(self.ax, p, self.r, ec=rings.get((l, u)), fc=fills.get((l, u)),
                     lw=1.4 if (l, u) in rings else 0.9)

    def weights(self, l, M, color, wmax=None, lo=0.3, hi=3.0):
        """Edges from layer l into layer l + 1, width proportional to M[out, in]."""
        wmax = wmax or M.max()
        for i, j in np.ndindex(M.shape):
            if M[i, j] > 0.02 * wmax:
                edge(self.ax, self.pos[l][j], self.pos[l + 1][i], color,
                     lo + (hi - lo) * M[i, j] / wmax)


def out_labels(ax, net, dx=0.14):
    for (x, y), c in zip(net.pos[2], ('even', 'odd')):
        txt(ax, x + dx, y, c, color=COLOR[c], ha='left')


# ---------------------------------------------------------------- slides
def slide_activations():
    """The usual reading: activations of one layer, mapped to concepts."""
    fig, ax = canvas()
    for o in (0.09, 0.06, 0.03, 0):                            # a stack of inputs
        ax.add_patch(Rectangle((0.2 + o, 0.98 - o), 0.46, 0.46, fc=COLOR['bg'], ec=COLOR['muted'],
                               lw=1.0, zorder=4))
    txt(ax, 0.43, 1.21, 'images,\ntext, ...', size=6.5)
    arrow(ax, (0.76, 1.2), (0.98, 1.2), lw=0.9)
    net = Net(ax, [1.2, 1.75, 2.3], 1.2, dy=0.38, r=0.1, sizes=(3, 3, 2))
    net.nodes(fills={(1, u): mix(COLOR['fg'], 0.25) for u in range(3)},
              rings={(1, u): COLOR['fg'] for u in range(3)})

    # read out the middle layer as a vector of activations
    xv, yv, cv = 2.85, 0.75, 0.3
    top = net.pos[1][0] - [0, 0.1]
    ax.plot([top[0], top[0], xv + cv / 2], [top[1], 0.38, 0.38], color=COLOR['fg'], lw=0.9)
    arrow(ax, (xv + cv / 2, 0.38), (xv + cv / 2, yv - 0.1), lw=0.9)
    for o in (0.06, 0.03):
        ax.add_patch(Rectangle((xv + o, yv - o), cv, 3 * cv, fc=COLOR['bg'], ec=COLOR['edge'],
                               lw=1.0, zorder=2))
    for u, v in enumerate([0.9, 0.1, 0.2]):
        ax.add_patch(Rectangle((xv, yv + u * cv), cv, cv, fc=mix(COLOR['fg'], v), ec=COLOR['bg'],
                               lw=1.0, zorder=3))
        txt(ax, xv + cv / 2, yv + (u + 0.5) * cv, f'{v:.1f}', size=7,
            color=COLOR['bg'] if v > 0.5 else COLOR['fg'])

    for k, s in enumerate(['dog', 'Golden Gate Bridge', 'unsafe behavior', 'neuroscience']):
        y = 0.72 + k * 0.32
        ax.add_patch(FancyBboxPatch((3.55, y - 0.11), 1.1, 0.22, fc=COLOR['bg'], ec=COLOR['fg'],
                                    lw=1.0, zorder=4, boxstyle='round,pad=0,rounding_size=0.04'))
        txt(ax, 4.1, y, s, size=7)
        arrow(ax, (xv + cv + 0.08, yv + 1.5 * cv), (3.5, y), lw=0.8, ms=5)
    yield fig, 'activations'


def slide_arbor():
    """One neuron, the same activation from two stimuli through different input terms."""
    fig, ax = canvas()
    w, base, k, shades = np.array([0.6, 0.6, 0.4]), 0.95, 0.9, (1.0, 0.7, 0.45)
    for x0, d, a, side in [(1.0, 0, np.array([0.9, 0.1, 0.3]), 1),
                           (3.2, 1, np.array([0.1, 0.9, 0.3]), -1)]:
        c = COLOR[CLASS[d]]
        n, ins = (x0, 0.5), [(x0 + dx, 1.3) for dx in (-0.45, 0, 0.45)]
        for p, aj, t, f in zip(ins, a, w * a, shades):
            edge(ax, n, p, mix(c, f, COLOR['bg']), 0.6 + 6 * t)
            node(ax, p, 0.09, fc=mix(COLOR['fg'], aj))
        node(ax, n, 0.17, ec=COLOR['fg'], fc=mix(COLOR['fg'], 0.72), lw=1.2)
        txt(ax, *n, '$n$', size=10, color=COLOR['bg'])
        tile(ax, x0, 1.82, 0.32, d, border=c, lw=1.4)
        arrow(ax, (x0, 1.64), (x0, 1.45), color=COLOR['muted'], lw=0.8, ms=5)

        # the input terms stacked into the sum, on the side facing the other neuron
        bx, y = x0 + side * 0.55 - (side < 0) * 0.15, base
        for t, f in zip(w * a, shades):
            ax.add_patch(Rectangle((bx, y - t * k), 0.15, t * k, fc=mix(c, f, COLOR['bg']),
                                   ec=COLOR['bg'], lw=1.0, zorder=3))
            y -= t * k
        arrow(ax, (x0 + side * 0.25, 0.92), (bx + 0.075 - side * 0.11, 0.82),
              color=COLOR['muted'], rad=side * 0.3, lw=0.8, ms=5)
        txt(ax, bx + 0.075, base + 0.13, r'$\Sigma$', size=8, color=COLOR['muted'])
    txt(ax, 2.1, 0.62, '=', size=22)
    yield fig, 'arbor'


def toy_network(ax, lit=None):
    """The toy task: digit images into the 8 x 4 x 2 network; lit = {digit: border color}."""
    lit = lit or {}
    for y, d in zip([0.6, 0.98, 1.36, 1.74], [0, 1, 3, 4]):
        tile(ax, 0.55, y, 0.3, d, border=lit.get(d), dim=bool(lit) and d not in lit,
             lw=1.4 if d in lit else 1.0)
    arrow(ax, (0.85, 1.17), (1.25, 1.17), lw=1.0)
    net = Net(ax, [1.6, 2.5, 3.4], 1.17, dy=0.21, r=0.07)
    for x, s in zip([0.55, 1.6, 2.5, 3.4], ['28 x 28', '8', '4', '2']):
        txt(ax, x, 2.12, s, size=7, color=COLOR['muted'])
    return net


def slide_toy_task():
    fig, ax = canvas()
    net = toy_network(ax)
    net.nodes(rings={(2, 0): COLOR['even'], (2, 1): COLOR['odd']})
    out_labels(ax, net)
    yield fig, 'toy_task'


def slide_overview(t: dict):
    """BFT on the toy task's network, no machinery. 1: what drives each output splits into
    circuits. 2: the even circuit is traced back, splitting again into sub-circuits."""
    green, even, odd = COLOR['green'], COLOR['even'], COLOR['odd']
    h3 = t['h3'].reshape(2, 4, 2)                              # (output, hidden 2, factor)
    for step in (1, 2):
        fig, ax = canvas()
        net = toy_network(ax, lit={0: digit_color(0), 4: digit_color(4)} if step == 2 else None)
        fills = {}
        for k, c in enumerate([even, odd]):
            net.weights(1, h3[..., k], mix(c, 1.0 if step == 1 or k == 0 else 0.45),
                        wmax=h3.max(), hi=3.0)
            mass = h3[..., k].sum(0)
            fills.update({(1, u): mix(c, mass[u] / mass.max()) for u in range(4)
                          if mass[u] > 0.3 * mass.max()})
        if step == 2:
            for k, d in enumerate([0, 4]):
                M = t['h2'][:, k].reshape(4, 8)
                M = np.where(M > 0.15 * M.max(), M, 0)          # the sub-circuit, not its noise
                net.weights(0, M, digit_color(d), wmax=t['h2'].max(), hi=3.0)
                mass = M.sum(0)
                fills.update({(0, u): mix(digit_color(d), mass[u] / mass.max()) for u in range(8)
                              if mass[u] > 0.3 * mass.max()})
            arrow(ax, tuple(net.pos[2][0] + [0, -0.11]), tuple(net.pos[1][0] + [0.02, -0.11]),
                  color=green, rad=0.5, lw=1.3)
            arrow(ax, tuple(net.pos[1][0] + [-0.02, -0.11]), tuple(net.pos[0][0] + [0.02, -0.11]),
                  color=green, rad=0.5, lw=1.3)
        net.nodes(fills=fills, rings={(2, 0): even, (2, 1): odd})
        out_labels(ax, net)
        yield fig, f'overview_{step}'


def slide_arbor_matrix(t: dict):
    """Build 1: one stimulus, one neuron. 2: all stimuli. 3: both output neurons."""
    W3p = np.clip(t['W3'], 0, None)
    J = t['J3']
    x0, y0, c = 2.75, 0.75, 0.21
    for step in (1, 2, 3):
        fig, ax = canvas()
        net = Net(ax, [0.8, 1.4, 2.0], 1.15)
        rows = [0] if step == 1 else range(4)
        outs = [0, 1] if step == 3 else [0]
        a2 = t['A2'][0] if step == 1 else t['A2'].mean(0)
        for i in outs:
            M = np.zeros_like(W3p)
            M[i] = W3p[i] * a2
            net.weights(1, M, COLOR[('even', 'odd')[i]], wmax=1.0)
        net.nodes(fills={(1, u): mix(COLOR['fg'], a2[u] / a2.max()) for u in range(4)},
                  rings={(2, i): COLOR[('even', 'odd')[i]] for i in outs})
        if step > 1:                                           # the rest of the stimuli, stacked
            for o in (0.09, 0.06, 0.03):
                ax.add_patch(Rectangle((0.18 + o, 1.03 - o), 0.24, 0.24, fc=COLOR['bg'],
                                       ec=COLOR['edge'], lw=1.0, zorder=4))
        tile(ax, 0.3, 1.15, 0.24, 0)
        arrow(ax, (0.47, 1.15), (0.66, 1.15), color=COLOR['muted'], lw=0.8, ms=5)

        shown = np.zeros_like(J)
        for s in rows:
            for i in outs:
                shown[s, 4 * i:4 * i + 4] = J[s, 4 * i:4 * i + 4]
        cells(ax, x0, y0, shown, lambda i, j: COLOR[('even', 'odd')[j // 4]], c, vmax=J.max())
        for s, d in enumerate(DIGITS):
            tile(ax, x0 - 0.16, y0 + (s + 0.5) * c, 0.16, d, dim=s not in rows)
        for i in outs:
            name = ('even', 'odd')[i]
            box(ax, x0 + 4 * i * c, y0 + min(rows) * c, x0 + 4 * (i + 1) * c,
                y0 + (max(rows) + 1) * c, COLOR[name])
            txt(ax, x0 + (4 * i + 2) * c, y0 - 0.13, name, size=7, color=COLOR[name])
        txt(ax, x0 + 4 * c, 0.3, r'weight $\times$ activation   $W_{ij}\,a_{s,j}$', size=8)
        txt(ax, x0 + 4 * c, y0 + 4 * c + 0.2, r'arbor matrix $J$', size=8, color=COLOR['muted'])
        arrow(ax, net.pos[2][0] + [0.06, -0.08], (x0 + 0.1, y0 - 0.25), rad=-0.3, lw=0.9)
        yield fig, f'arbor_matrix_{step}'


def slide_nmf(t: dict):
    """J = sum over factors of a stimulus factor times a connection factor."""
    fig, ax = canvas()
    c = 0.17
    x0, y0 = 0.5, 0.62                                         # as on the simplification slide
    cells(ax, x0, y0, t['J3'], lambda i, j: COLOR[('even', 'odd')[j // 4]], c)
    for s, d in enumerate(DIGITS):
        tile(ax, x0 - 0.16, y0 + (s + 0.5) * c, 0.16, d)
    txt(ax, x0 + 4 * c, y0 + 4 * c + 0.14, '$J$', size=9)
    txt(ax, 2.0, y0 + 2 * c, r'$\approx$', size=18)

    xs, xh = 2.45, 2.85
    txt(ax, xs + 0.5 * c, 0.22, 'stimulus\nfactor', size=6.5, color=COLOR['muted'])
    txt(ax, xh + 4 * c, 0.22, 'connection factor', size=6.5, color=COLOR['muted'])
    for k, (yk, name) in enumerate([(0.45, 'even'), (1.28, 'odd')]):
        col = COLOR[name]
        cells(ax, xs, yk, t['w3'][:, k:k + 1], col, c, vmax=1)
        txt(ax, xs + c + 0.1, yk + 2 * c, r'$\times$', size=10)
        cells(ax, xh, yk + 1.5 * c, t['h3'][:, k:k + 1].T, col, c, vmax=t['h3'].max())
        txt(ax, xs - 0.13, yk + 2 * c, f'$f_{k}$', size=8, color=col)
    txt(ax, xh + 4 * c, 1.07, '+', size=14)
    yield fig, 'nmf'


def slide_factor(t: dict):
    """A factor read back onto the network: which stimuli, which connections."""
    for k, name in enumerate(['even', 'odd']):
        fig, ax = canvas()
        col = COLOR[name]
        w, h = t['w3'][:, k], t['h3'][:, k].reshape(2, 4)
        c = 0.2
        for s, d in enumerate(DIGITS):
            y = 0.75 + s * 0.3
            tile(ax, 0.55, y, 0.22, d, dim=w[s] < 0.3 * w.max())
            ax.add_patch(Rectangle((0.72, y - c / 2), c, c, fc=mix(col, w[s] / w.max()), lw=0,
                                   zorder=3))
        txt(ax, 0.72 + c / 2, 0.45, f'$f_{k}$', size=9, color=col)
        net = Net(ax, [1.6, 2.4, 3.2], 1.2)
        net.weights(1, h, col, wmax=h.max(), hi=3.5)
        mass = h.sum(0) / h.sum(0).max()
        net.nodes(fills={(1, u): mix(col, mass[u]) for u in range(4)},
                  rings={(2, i): COLOR[n] for i, n in enumerate(['even', 'odd'])})
        out_labels(ax, net)
        txt(ax, 0.82, 2.05, 'stimulus factor', size=7, color=COLOR['muted'])
        txt(ax, 2.8, 2.05, 'connection factor', size=7, color=COLOR['muted'])
        yield fig, f'factor_{k + 1}'


def slide_trace(t: dict):
    """The backward step, in four builds on one canvas.

    1: the even factor traces back into the layer below. 2: each stimulus's share of its loadings
    on that factor is pi, the tau gate drops the rest. 3: pi scales the rows of the arbor matrix
    one layer down. 4: NMF of that matrix gives the "0" and the "4" factor.
    """
    green, even, odd = COLOR['green'], COLOR['even'], COLOR['odd']
    share, pi, w2, h2 = t['share'], t['pi'], t['w2'], t['h2']
    wt = t['w3'] / t['w3'].sum(1, keepdims=True)              # each stimulus's loadings, summed to 1
    raw = (np.clip(t['W2'], 0, None)[None] * t['A1'][:, None, :]).reshape(4, -1)
    y0, ch, xb, bw = 0.55, 0.22, 2.0, 0.42
    x0 = xb + bw + 0.18
    cw = (4.6 - x0) / 32
    for step in (1, 2, 3, 4):
        fig, ax = canvas()
        net = Net(ax, [0.3, 0.85, 1.4], 1.2, dy=0.18, r=0.055)
        net.weights(1, t['h3'][:, 0].reshape(2, 4), even, hi=2.2)
        child = {}
        if step < 4:
            net.weights(0, np.ones((4, 8)), mix(green, 0.55), lo=0.6, hi=0.6)
        else:
            for k in range(2):
                col = digit_color((0, 4)[k])
                net.weights(0, h2[:, k].reshape(4, 8), col, hi=2.2)
                mass = h2[:, k].reshape(4, 8).sum(0)
                child.update({(0, u): mix(col, mass[u] / mass.max())
                              for u in range(8) if mass[u] > 0.3 * mass.max()})
        net.nodes(fills={(2, 0): mix(even, 0.8), **child}, rings={(2, 0): even, (2, 1): odd})
        arrow(ax, net.pos[2][0] + [0.03, -0.08], net.pos[1][0] + [0.03, -0.08], color=green,
              rad=0.7, lw=1.2)
        if step == 1:
            yield fig, 'trace_1'
            continue

        # pi: the stimulus's share of its loadings that falls on the even factor
        arrow(ax, net.pos[2][0] + [0.09, -0.04], (xb - 0.02, y0 - 0.12), color=green, rad=-0.3,
              lw=1.0)
        txt(ax, xb + bw / 2, y0 - 0.12, r'$\pi$', size=10, color=green)
        for s, d in enumerate(DIGITS):
            y, dim = y0 + (s + 0.5) * ch, pi[s] == 0
            tile(ax, xb - 0.13, y, 0.17, d, dim=dim)
            for x, wd, c in [(xb, bw * wt[s, 0], even), (xb + bw * wt[s, 0], bw * wt[s, 1], odd)]:
                ax.add_patch(Rectangle((x, y - 0.06), wd, 0.12, lw=0, zorder=3,
                                       fc=mix(c, 0.35 if dim else 1.0, COLOR['bg'])))
            if not dim:
                box(ax, xb, y - 0.06, xb + bw * share[s], y + 0.06, green, lw=1.2)
        if step == 2:
            yield fig, 'trace_2'
            continue

        cells(ax, x0, y0, pi[:, None] / pi.max() * raw, COLOR['fg'], cw, ch, vmax=raw.max())
        for i in range(1, 4):
            ax.plot([x0 + 8 * i * cw] * 2, [y0 - 0.03, y0 + 4 * ch + 0.03], color=COLOR['bg'],
                    lw=1.2, zorder=4)
        txt(ax, x0 + 16 * cw, y0 - 0.15, 'arbor matrix one layer down', size=7,
            color=COLOR['muted'])
        if step == 3:
            yield fig, 'trace_3'
            continue

        xm = x0 + 16 * cw
        arrow(ax, (xm, y0 + 4 * ch + 0.05), (xm, 1.62), lw=0.8)
        txt(ax, xm + 0.07, 1.55, 'NMF', size=7, color=COLOR['muted'], ha='left')
        for k in range(2):
            d, yk = (0, 4)[k], 1.72 + k * 0.3
            col = digit_color(d)
            tile(ax, x0 - 0.15, yk + 0.09, 0.18, d, border=col, lw=1.2)
            cells(ax, x0, yk, w2[:, k:k + 1].T, col, 0.1, 0.18, vmax=1)
            cells(ax, x0 + 0.5, yk, h2[:, k:k + 1].T, col, (32 * cw - 0.5) / 32, 0.18,
                  vmax=h2.max())
        yield fig, 'trace_4'


def tree(ax, xs, ys, r, fills=None, fs=1.0):
    """Two-level factor tree, even and odd above their digit factors; returns the node centres."""
    fills = fills or {}
    top = {'even': (xs[0], ys[0]), 'odd': (xs[1], ys[0])}
    kids = {d: (x, ys[1]) for d, x in zip(DIGITS, xs[2:])}
    apex = ((xs[0] + xs[1]) / 2, ys[0] - 0.33 * fs)
    for p in top.values():
        edge(ax, apex, p, COLOR['muted'], 0.9)
    for d, p in kids.items():
        edge(ax, top[CLASS[d]], p, COLOR['muted'], 0.9)
        for dx in (-0.12, 0.12):
            edge(ax, p, (p[0] + dx * fs, p[1] + 0.38 * fs), COLOR['faint'], 0.8)
    for name, p in {**top, **kids}.items():
        col = COLOR[name] if isinstance(name, str) else digit_color(name)
        node(ax, p, r, ec=col, fc=fills.get(name), lw=1.4)
        txt(ax, *p, str(name), size=7 if isinstance(name, str) else 9 * fs)
    txt(ax, apex[0], ys[1] + 0.55 * fs, '...', size=10, color=COLOR['faint'])
    return {**top, **kids}


FP = np.array([[0.9, 0.05, 0.9, 0.2, 0, 0],            # stimulus x tree node: toy fingerprints
               [0.8, 0.1, 0.25, 0.85, 0, 0],
               [0.05, 0.9, 0, 0, 0.9, 0.2],
               [0.1, 0.85, 0, 0, 0.2, 0.85]])
FP_NODES = ['even', 'odd', 0, 4, 1, 3]
TREE = ([0.85, 2.05, 0.5, 1.2, 1.7, 2.4], [0.7, 1.35], 0.15)   # x, y, radius


def slide_tree():
    fig, ax = canvas()
    tree(ax, *TREE)
    yield fig, 'tree'


def slide_fingerprint():
    """Build 1: one stimulus's loadings over the tree. 2: every stimulus, one row each."""
    def color(name):
        return COLOR[name] if isinstance(name, str) else digit_color(name)

    for step in (1, 2):
        fig, ax = canvas()
        fills = {n: mix(color(n), FP[0, i]) for i, n in enumerate(FP_NODES)} if step == 1 else {}
        tree(ax, *TREE, fills=fills)
        arrow(ax, (2.65, 1.0), (2.9, 1.0), lw=1.0)

        x0, y0, c = 3.25, 0.62, 0.2
        shown = FP if step == 2 else FP * (np.arange(4) == 0)[:, None]
        cells(ax, x0, y0, shown, lambda i, j: color(FP_NODES[j]), c, vmax=1)
        for j, n in enumerate(FP_NODES):
            ax.add_patch(Circle((x0 + (j + 0.5) * c, y0 - 0.1), 0.04, fc=color(n), lw=0, zorder=4))
        for s, d in enumerate(DIGITS):
            tile(ax, x0 - 0.17, y0 + (s + 0.5) * c, 0.16, d,
                 border=COLOR['fg'] if s == 0 and step == 1 else None, dim=step == 1 and s > 0)
        if step == 1:
            tile(ax, 0.55, 0.3, 0.26, 0, border=COLOR['fg'])
        yield fig, f'fingerprint_{step}'


def slide_simplification(t: dict):
    """The nmf slide's arbor matrix: its columns are scaled copies of the activations, and the
    small matrix of those activations gives the same stimulus factors."""
    W3p, A2 = np.clip(t['W3'], 0, None), t['A2']
    c_j = np.linalg.norm(W3p, axis=0)
    At = A2 * c_j
    wA, _, _ = nmf(At, 2)
    if wA[0, 0] < wA[0, 1]:
        wA = wA[:, ::-1]
    fig, ax = canvas()
    green, c = COLOR['green'], 0.17

    x0, y0 = 0.5, 0.62                                         # as on the nmf slide
    cells(ax, x0, y0, t['J3'], lambda i, j: COLOR[('even', 'odd')[j // 4]], c)
    xa, ya = x0 + 2 * c, 1.6
    cells(ax, xa, ya, At, COLOR['fg'], c)
    for (x, y) in [(x0, y0), (xa, ya)]:
        for s, d in enumerate(DIGITS):
            tile(ax, x - 0.15, y + (s + 0.5) * c, 0.16, d)
    for i in range(2):
        txt(ax, x0 + (4 * i + 2) * c, y0 - 0.12, ('even', 'odd')[i], size=7,
            color=COLOR[('even', 'odd')[i]])
        for j in range(4):                                     # copy (i, j) of activation column j
            ax.add_patch(FancyArrowPatch(
                (x0 + (4 * i + j + 0.5) * c, y0 + 4 * c + 0.04), (xa + (j + 0.5) * c, ya - 0.04),
                arrowstyle='-|>', mutation_scale=5, lw=0.7, zorder=6, shrinkA=0, shrinkB=0,
                color=green if W3p[i, j] > 0 else COLOR['faint'],
                ls='-' if W3p[i, j] > 0 else (0, (2, 2))))
    txt(ax, x0 + 8 * c + 0.08, y0 + 2 * c - 0.09, r'$J^+$', size=9, ha='left')
    txt(ax, x0 + 8 * c + 0.08, y0 + 2 * c + 0.11, r'$S \times n_l n_{l-1}$', size=7,
        color=COLOR['muted'], ha='left')
    txt(ax, xa + 4 * c + 0.08, ya + 2 * c - 0.09, r'$\tilde A$', size=9, ha='left')
    txt(ax, xa + 4 * c + 0.08, ya + 2 * c + 0.11, r'$S \times n_{l-1}$', size=7,
        color=COLOR['muted'], ha='left')

    # both factorizations give the same stimulus factors
    xs = 2.75
    for (y, x_from, w) in [(y0, x0 + 8 * c + 0.62, t['w3']), (ya, xa + 4 * c + 0.62, wA)]:
        arrow(ax, (x_from, y + 2 * c), (xs - 0.08, y + 2 * c), lw=0.8)
        txt(ax, (x_from + xs) / 2, y + 2 * c - 0.1, 'NMF', size=6.5, color=COLOR['muted'])
        cells(ax, xs, y, w, lambda i, j: COLOR[('even', 'odd')[j]], c, vmax=1)
    txt(ax, xs + c, y0 - 0.12, r'$W_\mathrm{stim}$', size=8)
    txt(ax, xs + c, (y0 + 4 * c + ya) / 2, '=', size=14)

    xt = 3.3
    txt(ax, xt, ya + 0.08, r'$\tilde A = A\,\mathrm{diag}(c)$', size=7.5, ha='left')
    txt(ax, xt, ya + 0.34, r'$c_j = \|W^+_{:,j}\|_2$', size=7.5, ha='left')
    txt(ax, xt, ya + 0.6, r'$H_{(i,j),k} = W^+_{ij}\,G_{jk}/c_j$', size=7.5, ha='left')
    yield fig, 'simplification'


if __name__ == '__main__':
    t = toy()
    publish([*slide_activations()],
            [*slide_arbor()],
            [*slide_toy_task(), *slide_overview(t)],
            [*slide_arbor_matrix(t)],
            [*slide_nmf(t), *slide_simplification(t)],
            [*slide_factor(t)],
            [*slide_trace(t)],
            [*slide_tree(), *slide_fingerprint()])
