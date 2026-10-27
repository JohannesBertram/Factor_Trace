"""Not reviewed code. Shared style of every new figure: dark slides, light paper figures.

Slides go to figures/slides/<name>.pdf, paper figures to figures/paper/<name>.pdf, each with a
preview PNG in preview/. Every color comes from COLOR at draw time, never a hex literal.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import figstyle

VENUE = 'aaai2024'
W = 6.975                                 # AAAI text width, inches
BIG, SMALL = 7, 6                         # the two text sizes of every paper figure


def _components() -> dict:
    """The palette names plus every component of .figstyle/colors.yaml, resolved to hex by figstyle."""
    path = Path(figstyle.__file__).parent / '.figstyle' / 'colors.yaml'
    names = yaml.safe_load(path.read_text())['components']
    return {**figstyle.PALETTE, **{k: figstyle.color(k) for k in names}}


def _on_black(c: str, min_light: float = 0.6) -> str:
    """Same hue, lifted to at least min_light lightness so it reads on the black slide."""
    h, l, s = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return mcolors.to_hex(colorsys.hls_to_rgb(h, max(l, min_light), s))


# The paper keeps the colors.yaml hues as they are; the slides lift the dark ones
_PAPER = _components()
THEMES = {
    'dark': {'bg': 'black', 'fg': 'white', 'muted': '#B0B0B0', 'faint': '#707070',
             'edge': '#555555', 'grid': '#333333', **{k: _on_black(v) for k, v in _PAPER.items()}},
    'light': {'bg': 'white', 'fg': 'black', 'muted': '#555555', 'faint': '#9A9A9A',
              'edge': '#AAAAAA', 'grid': '#DDDDDD', **_PAPER},
}
COLOR = dict(THEMES['dark'])


def _theme(name: str):
    """Switch COLOR in place, so modules that imported it by name follow."""
    COLOR.clear()
    COLOR.update(THEMES[name])


# ---------------------------------------------------------------- slides
def slide(ar: float):
    """A dark, engine-less slide figure of the given height / width.

    Notes
    -----
    figstyle has no dark mode, so the rc overrides live here. savefig.facecolor must be set too:
    save_fig calls fig.savefig with no facecolor, so a dark figure would land on a white page.
    Place axes with fig.add_axes; subplots_adjust is silently undone by the slide rc.
    """
    _theme('dark')
    plt.rcdefaults()
    figstyle.apply(venue=VENUE, mode='slides', figs_per_slide=1)
    bg, fg = COLOR['bg'], COLOR['fg']
    plt.rcParams.update({
        'figure.facecolor': bg, 'axes.facecolor': bg, 'savefig.facecolor': bg,
        'savefig.edgecolor': bg, 'text.color': fg, 'axes.labelcolor': fg,
        'axes.edgecolor': fg, 'xtick.color': fg, 'ytick.color': fg, 'legend.labelcolor': fg,
        'grid.color': COLOR['grid'], 'patch.edgecolor': fg, 'figure.constrained_layout.use': False})
    w = plt.rcParams['figure.figsize'][0]
    fig = plt.figure(figsize=(w, w * ar))
    fig.set_layout_engine('none')
    return fig


def note(fig, text: str, y: float = 0.0):
    """The single grey line of context under a slide; slides carry no titles."""
    fig.text(0.5, y, text, ha='center', va='bottom', fontsize=8, color=COLOR['muted'])


def slide_arrow(fig, a, b, color, style='-|>', lw=1.2, ms=8):
    """Arrow between two points in figure fractions."""
    fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle=style, color=color,
                                   lw=lw, mutation_scale=ms, shrinkA=0, shrinkB=0))


def slide_box(fig, x, y, w, h, color, lw=1.2):
    """Rounded outline box in figure fractions."""
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.012',
                                  transform=fig.transFigure, fc='none', ec=color, lw=lw))


# ---------------------------------------------------------------- paper
def paper(height: float):
    """A light, engine-less figure of the full text width and the given height in inches."""
    _theme('light')
    plt.rcdefaults()
    figstyle.apply(venue=VENUE, width='full', mode='paper', usetex=False)
    plt.rcParams.update({'font.size': SMALL, 'axes.labelsize': BIG, 'axes.titlesize': BIG,
                         'xtick.labelsize': SMALL, 'ytick.labelsize': SMALL,
                         'legend.fontsize': SMALL, 'mathtext.fontset': 'stix',
                         'figure.constrained_layout.use': False, 'savefig.bbox': 'standard'})
    fig = plt.figure(figsize=(W, height))
    fig.set_layout_engine('none')
    fig.H = height
    return fig


def ax_in(fig, x, y, w, h):
    """Axes at x, y (inches from the top-left corner) of size w x h inches."""
    return fig.add_axes([x / W, 1 - (y + h) / fig.H, w / W, h / fig.H])


def fx(fig, x, y):
    """Inch position from the top-left corner in figure fractions."""
    return x / W, 1 - y / fig.H


def text(fig, x, y, s, **kw):
    """Text at an inch position from the top-left corner."""
    kw.setdefault('fontsize', SMALL)
    kw.setdefault('color', COLOR['fg'])
    return fig.text(*fx(fig, x, y), s, **kw)


def label(fig, x, y, s):
    """Bold panel letter at an inch position."""
    return text(fig, x, y, s, fontsize=figstyle.PANEL_LABEL_SIZE, fontweight='bold', ha='left',
                va='top')


def arrow(fig, a, b, color=None, style='-|>', lw=0.9, ms=6):
    """Arrow between two inch positions."""
    fig.add_artist(FancyArrowPatch(fx(fig, *a), fx(fig, *b), transform=fig.transFigure,
                                   arrowstyle=style, color=color or COLOR['fg'], lw=lw,
                                   mutation_scale=ms, shrinkA=0, shrinkB=0))


def box(fig, x, y, w, h, color, lw=0.9):
    """Rounded outline box at an inch position."""
    fig.add_artist(FancyBboxPatch(fx(fig, x, y + h), w / W, h / fig.H,
                                  boxstyle='round,pad=0,rounding_size=0.008',
                                  transform=fig.transFigure, fc='none', ec=color, lw=lw))


# ---------------------------------------------------------------- both
def strip(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def extent(fig):
    """Left and right edge of everything drawn, in inches: the tight box of every axes with its axis
    on, the artists of every axes without (drawing canvases, images)."""
    r = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(r) for t in fig.texts if t.get_text()]
    for ax in fig.axes:
        if ax.axison:
            boxes.append(ax.get_tightbbox(r))
            continue
        skip = {ax.patch, ax.xaxis, ax.yaxis, *ax.spines.values()}
        boxes += [b for a in ax.get_children() if a.get_visible() and a not in skip
                  for b in [a.get_window_extent(r)] if b.width > 0 and np.isfinite(b.x0)]
    return min(b.x0 for b in boxes) / fig.dpi, max(b.x1 for b in boxes) / fig.dpi


def publish(*groups):
    """Save each group of (fig, name) slides with one horizontal shift that centers the union of
    their content, so consecutive slides of one plot stay aligned and a growing slide's last build
    is centered."""
    for group in groups:
        group = list(group)
        lo, hi = zip(*(extent(fig) for fig, _ in group))
        for fig, name in group:
            W = fig.get_size_inches()[0]
            dx = (W - min(lo) - max(hi)) / 2 / W
            for ax in fig.axes:
                p = ax.get_position()
                ax.set_position([p.x0 + dx, p.y0, p.width, p.height])
            for t in fig.texts:
                t.set_x(t.get_position()[0] + dx)
            save(fig, name)


def save(fig, name: str):
    """PDF plus preview PNG, in figures/slides/ for a slide and figures/paper/ for a paper figure."""
    if COLOR['bg'] == 'black':
        path = figstyle.save_fig(fig, name, figdir='figures', preview_dpi=220)   # adds slides/
    else:
        path = figstyle.save_fig(fig, name, figdir='figures/paper', preview_dpi=250)
    plt.close(fig)
    return path
