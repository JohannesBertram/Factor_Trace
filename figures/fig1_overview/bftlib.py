"""Shared drawing primitives for the BFT Figure-1 schematic.

Everything is drawn on a single full-bleed axes in *inch* coordinates
(aspect='equal'), so circles are round and sizes are physical. Colors come from
figstyle so the schematic matches the paper (even=blue, odd=red, BFT=purple).
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import figstyle

# ------------------------------------------------------------------ colors
EVEN = figstyle.color("even")     # blue
ODD  = figstyle.color("odd")      # red
OURS = figstyle.color("ours")     # purple (BFT)
GREEN = figstyle.color("green")   # the "reweight" accent
INK  = "#2b2b2b"
GREY = "#9a9a9a"


def _to_rgb(c):
    return np.array(plt.matplotlib.colors.to_rgb(c))


def tint(c, f):
    """Blend color c toward white by fraction f in [0,1]."""
    rgb = _to_rgb(c)
    return tuple(rgb * (1 - f) + np.ones(3) * f)


def shade(c, f):
    """Blend color c toward black by fraction f."""
    return tuple(_to_rgb(c) * (1 - f))


# ------------------------------------------------------------------ canvas
def canvas(ax, W, H):
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()          # y grows downward -> matches reading order


# ------------------------------------------------------------------ nodes
def node(ax, xy, r, color, *, active=True, lw=1.0, z=5):
    """Filled (active) or open (inactive) circle. r in inches."""
    if active:
        fc, ec = tint(color, 0.15), shade(color, 0.25)
    else:
        fc, ec = "white", color
    ax.add_patch(Circle(xy, r, facecolor=fc, edgecolor=ec, lw=lw, zorder=z))


def _lw_from_w(w, wmax, lo=0.4, hi=3.0):
    return lo + (hi - lo) * (abs(w) / wmax)


def edge(ax, p0, p1, w, wmax, color, *, sign=1, z=1, alpha=0.95):
    """Weighted edge p0->p1. width∝|w|; solid=excitatory, dashed=inhibitory."""
    style = "-" if sign > 0 else (0, (2.2, 1.6))
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=_lw_from_w(w, wmax),
            ls=style, solid_capstyle="round", zorder=z, alpha=alpha)


# ------------------------------------------------------------------ arrows
def arrow(ax, p0, p1, *, color=INK, lw=1.3, z=6, rad=0.0, mut=9):
    ap = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
                         lw=lw, color=color, zorder=z,
                         connectionstyle=f"arc3,rad={rad}",
                         shrinkA=0, shrinkB=0)
    ax.add_patch(ap)
    return ap


# ------------------------------------------------------------------ boxes
def rbox(ax, x0, y0, x1, y1, *, fc="none", ec=INK, lw=1.0, pad=0.02, z=0,
         rounding=0.03, ls="-"):
    b = FancyBboxPatch((x0 + pad, y0 + pad), (x1 - x0) - 2 * pad,
                       (y1 - y0) - 2 * pad, zorder=z,
                       boxstyle=f"round,pad={pad},rounding_size={rounding}",
                       facecolor=fc, edgecolor=ec, lw=lw, linestyle=ls)
    ax.add_patch(b)
    return b


# ------------------------------------------------------------------ text
def text(ax, x, y, s, *, size=9, color=INK, ha="center", va="center",
         weight="normal", style="normal", z=8, math=False):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontweight=weight, fontstyle=style, zorder=z)


# ------------------------------------------------------------------ digits
def digit_field(kind, n=20, noise=0.0, rng=None):
    """Coarse intensity map (n,n) in [0,1] for a stylized '0' or '1'."""
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) * 2 - 1     # [-1,1]
    if kind == "0":
        r = np.sqrt((xx / 0.62) ** 2 + (yy / 0.86) ** 2)
        f = np.exp(-((r - 1.0) / 0.28) ** 2)
    elif kind == "1":
        stem = np.exp(-((xx - 0.12 * yy - 0.05) / 0.20) ** 2)
        stem *= (np.abs(yy) < 0.92)
        serif = np.exp(-((xx + 0.35) / 0.22) ** 2) * np.exp(-((yy + 0.55) / 0.22) ** 2)
        base = np.exp(-((yy - 0.86) / 0.14) ** 2) * (np.abs(xx) < 0.5)
        f = np.clip(stem + 0.7 * serif + 0.6 * base, 0, 1)
    else:
        f = np.zeros_like(xx)
    if noise:
        rng = rng or np.random.default_rng(0)
        f = np.clip(f + rng.normal(0, noise, f.shape) * (f > 0.05), 0, 1)
        f *= (rng.random(f.shape) > noise * 0.7)          # speckle dropout
    return f


def glyph_field(char, n=30, pad=0.14, weight="bold"):
    """Rasterize a character to an (n,n) [0,1] mask via its font outline
    (aspect-preserving, centered, origin='upper'). Works for any digit."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    tp = TextPath((0, 0), str(char), size=1, prop=FontProperties(weight=weight))
    ext = tp.get_extents()
    if ext.width <= 0 or ext.height <= 0:
        return np.zeros((n, n))
    s = ext.height / (1 - 2 * pad)
    cx, cy = ext.x0 + ext.width / 2, ext.y0 + ext.height / 2
    u = np.linspace(0, 1, n)
    gu, gv = np.meshgrid(u, u)                     # gv: 0=top row
    px = cx + (gu - 0.5) * s
    py = cy + (0.5 - gv) * s
    mask = tp.contains_points(np.column_stack([px.ravel(), py.ravel()]))
    return mask.reshape(n, n).astype(float)


def digit_rgb(kind, tintcol, noise=0.0, seed=0):
    f = glyph_field(kind)
    if noise:
        rng = np.random.default_rng(seed)
        f = np.clip(f + rng.normal(0, noise, f.shape) * (f > 0.05), 0, 1)
        f *= (rng.random(f.shape) > noise * 0.6)      # speckle dropout
    ink = _to_rgb(tintcol)
    img = np.ones((*f.shape, 3))
    for c in range(3):
        img[..., c] = 1 - f * (1 - ink[c])
    return img


def place_digit(ax, kind, center, size, tintcol, *, noise=0.0, seed=0,
                border=None, blw=1.0, z=7):
    """Draw a small digit image centered at `center` (inches), width=size."""
    x, y = center
    ext = [x - size / 2, x + size / 2, y - size / 2, y + size / 2]
    ins = ax.inset_axes([ext[0], ext[2], size, size], transform=ax.transData,
                        zorder=z)
    # inset uses standard (y-up) image coords; our main axis is y-down, so flip
    ins.imshow(digit_rgb(kind, tintcol, noise=noise, seed=seed),
               interpolation="bilinear", origin="upper")
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(border is not None)
        if border is not None:
            sp.set_edgecolor(border); sp.set_linewidth(blw)
    return ins


# ------------------------------------------------------------------ heatmaps
def cell_grid(ax, x0, y0, cell, M, color_fn, *, gap=0.0, ec="white", elw=0.5, z=3):
    """Draw matrix M as a grid of rectangles. Top-left at (x0,y0), row-major.
    color_fn(value, i, j) -> facecolor."""
    M = np.asarray(M, float)
    nr, nc = M.shape
    for i in range(nr):
        for j in range(nc):
            fc = color_fn(M[i, j], i, j)
            ax.add_patch(Rectangle((x0 + j * cell, y0 + i * cell),
                                    cell - gap, cell - gap, facecolor=fc,
                                    edgecolor=ec, lw=elw, zorder=z))
    return (x0, y0, x0 + nc * cell, y0 + nr * cell)


def gray_fn(vmax=1.0):
    def f(v, i, j):
        t = np.clip(v / vmax, 0, 1)
        g = 1 - 0.85 * t
        return (g, g, g)
    return f


def tintcols_fn(col_colors, vmax=1.0):
    """Per-column tint: intensity=value, hue from col_colors[j]."""
    def f(v, i, j):
        return tint(col_colors[j], 1 - np.clip(v / vmax, 0, 1) * 0.9)
    return f


def tintrows_fn(row_colors, vmax=1.0):
    def f(v, i, j):
        return tint(row_colors[i], 1 - np.clip(v / vmax, 0, 1) * 0.9)
    return f


# ------------------------------------------------------------------ templates
def curved(ax, p0, p1, *, color=INK, lw=1.4, rad=-0.3, z=6, mut=10):
    ap = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut, lw=lw,
                         color=color, zorder=z, shrinkA=0, shrinkB=0,
                         connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(ap)
    return ap

