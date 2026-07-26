"""Draft B — restructured 2-row overview: WHY (a) + HOW one step (b) on top,
the RECURSION into a factor tree (c) across the full width below."""
import sys, numpy as np
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE)))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle
import bftlib as B

W, H = 6.975, 3.25
figstyle.apply(venue="aaai2024", width="full", nrows=1, ncols=1, mode="paper",
               height_to_width_ratio=H / W)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
B.canvas(ax, W, H)

EVEN, ODD, INK, GREY, GREEN, OURS = B.EVEN, B.ODD, B.INK, B.GREY, B.GREEN, B.OURS
HDR, TXT, SML = 8.5, 7.5, 6.5


def header(x, y, letter, title):
    B.text(ax, x, y, f"({letter})", size=HDR, weight="bold", ha="left", va="center")
    B.text(ax, x + 0.15, y, title, size=HDR, ha="left", va="center")


def chip(x, y, ch, border, s=0.15):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, facecolor="black",
                           edgecolor=border, lw=1.3, zorder=7))
    ax.text(x, y, ch, color="white", fontsize=6.5, ha="center", va="center",
            zorder=8, family="monospace")


# ===================================================================== (a) why
header(0.05, 0.18, "a", "A neuron hides its inputs")
nx, ny = 0.78, 0.66
inx, iny = [0.48, 0.78, 1.08], 1.22
B.node(ax, (nx, ny), 0.115, INK); B.text(ax, nx, ny, "$n$", size=TXT, color="white", z=9)
for x in inx:
    B.node(ax, (x, iny), 0.06, GREY)
we, wo = [0.95, 0.30, 0.55], [0.30, 0.95, 0.55]
for x, a, b in zip(inx, we, wo):
    B.edge(ax, (nx - 0.04, ny + 0.03), (x - 0.03, iny - 0.05), a, 1.0, EVEN, z=2)
    B.edge(ax, (nx + 0.04, ny + 0.03), (x + 0.03, iny - 0.05), b, 1.0, ODD, z=2)
eqy = 1.62
B.text(ax, 0.24, eqy, "$n($", size=TXT, ha="left", va="center")
chip(0.45, eqy, "0", EVEN, 0.14)
B.text(ax, 0.535, eqy, "$)=n($", size=TXT, ha="left", va="center")
chip(0.94, eqy, "1", ODD, 0.14)
B.text(ax, 1.02, eqy, "$)$", size=TXT, ha="left", va="center")
B.text(ax, 0.78, 1.92, "same activation, different arbors", size=SML - 0.5,
       style="italic", color="0.4")

B.arrow(ax, (1.44, 1.02), (1.72, 1.02), lw=1.5)
B.text(ax, 1.58, 0.85, "BFT", size=SML, weight="bold", color=OURS)

# ============================================================ (b) one BFT step
header(1.86, 0.18, "b", "One backward factor-trace step")
MID = 0.98

# target node
tn = (2.12, MID)
B.node(ax, tn, 0.10, OURS); B.text(ax, tn[0], tn[1], "$n$", size=TXT, color="white", z=9)
B.text(ax, tn[0], tn[1] + 0.24, "neuron /", size=SML - 0.5, color="0.4")
B.text(ax, tn[0], tn[1] + 0.36, "factor", size=SML - 0.5, color="0.4")
B.arrow(ax, (2.28, MID), (2.52, MID), lw=1.1)

# arbor matrix J  (stimuli x input units)
jc = 0.15
jx, jy = 2.72, MID - jc
Jm = np.array([[0.9, 0.15, 0.6, 0.25, 0.4, 0.1],
               [0.2, 0.85, 0.55, 0.3, 0.15, 0.5]])
B.cell_grid(ax, jx, jy, jc, Jm, B.gray_fn(1.0))
chip(jx - 0.15, jy + jc * 0.5, "0", EVEN, 0.13)
chip(jx - 0.15, jy + jc * 1.5, "1", ODD, 0.13)
jcx = jx + 3 * jc
B.text(ax, jcx, jy - 0.13, "arbor matrix $J$", size=SML, color=INK)
B.text(ax, jcx, jy + 2 * jc + 0.14, r"weights $\times$ activations", size=6.0, color="0.45")
B.text(ax, jcx, jy + 2 * jc + 0.26, "over stimuli", size=6.0, color="0.45")

B.text(ax, jx + 6 * jc + 0.15, MID, r"$\approx$", size=12, va="center")

# NMF  H x W
hx = jx + 6 * jc + 0.38
Hm = np.array([[0.95, 0.12], [0.1, 0.9]])
B.cell_grid(ax, hx, jy, jc, Hm, B.tintcols_fn([EVEN, ODD]))
B.text(ax, hx + 0.5 * jc, jy - 0.11, "$f_0$", size=SML, color=EVEN)
B.text(ax, hx + 1.5 * jc, jy - 0.11, "$f_1$", size=SML, color=ODD)
B.text(ax, hx + 2 * jc + 0.14, MID, r"$\times$", size=9, va="center")
wx = hx + 2 * jc + 0.42
Wm = np.array([[0.85, 0.1, 0.55, 0.2, 0.45, 0.12],
               [0.15, 0.8, 0.5, 0.35, 0.1, 0.5]])
B.cell_grid(ax, wx, jy, jc, Wm, B.tintrows_fn([EVEN, ODD]))
B.text(ax, wx - 0.12, jy + 0.5 * jc, "$f_0$", size=SML, color=EVEN, ha="right")
B.text(ax, wx - 0.12, jy + 1.5 * jc, "$f_1$", size=SML, color=ODD, ha="right")
bx0, bx1 = hx - 0.24, wx + 6 * jc + 0.06
B.rbox(ax, bx0, jy - 0.28, bx1, jy + 2 * jc + 0.28, ec="0.6", lw=0.9, pad=0.0, rounding=0.03)
B.text(ax, (bx0 + bx1) / 2, jy - 0.20, "NMF", size=SML, weight="bold", color="0.45")
B.text(ax, hx + jc, jy + 2 * jc + 0.14, "stimulus", size=6.0, color="0.45")
B.text(ax, wx + 3 * jc, jy + 2 * jc + 0.14, "connection factors", size=6.0, color="0.45")
B.text(ax, (bx0 + bx1) / 2 - 0.15, jy + 2 * jc + 0.44,
       r"each factor $f_k$ $=$ a sub-circuit $+$ its stimuli",
       size=SML - 0.5, style="italic", color=B.shade(GREEN, 0.1))

# ============================================= (c) recurse -> factor tree (full width)
header(0.05, 2.06, "c", "Recurse backward through every layer")
ROOTY, MIDY, LEAFY = 2.52, 2.80, 3.06
# recursion narrative (top-left)
B.text(ax, 0.10, 2.40, "apply BFT to each", size=SML, color="0.35", ha="left")
B.text(ax, 0.10, 2.52, "factor, one layer back", size=SML, color="0.35", ha="left")
# layer axis just left of the tree, aligned to the tree rows
lax = 1.60
B.arrow(ax, (lax, ROOTY - 0.06), (lax, LEAFY + 0.12), lw=1.3, color="0.55")
for yy, lab in [(ROOTY, "$L_3$"), (MIDY, "$L_2$"), (LEAFY, "$L_1$")]:
    B.text(ax, lax - 0.07, yy, lab, size=SML, color="0.55", ha="right", va="center")

# curved arrow from the NMF output (b) arcing through the center into the tree
B.curved(ax, (4.55, jy + 2 * jc + 0.42), (3.05, ROOTY - 0.02),
         color=OURS, lw=1.5, rad=-0.30, mut=11)
B.text(ax, 3.55, 2.06, "recurse", size=SML, weight="bold", color=OURS, ha="left")

# the tree: output even/odd -> L2 -> L1 leaf templates (digits 0,4 even | 1,3 odd)
evx, odx = 2.75, 5.25
for (cx, hue, digs, sd, name) in [(evx, EVEN, ["0", "0", "4", "4"], 0, "even"),
                                  (odx, ODD, ["1", "1", "3", "3"], 10, "odd")]:
    root = (cx, ROOTY)
    l2 = [(cx - 0.50, MIDY), (cx + 0.50, MIDY)]
    l1x = [cx - 0.78, cx - 0.26, cx + 0.26, cx + 0.78]
    for m in l2:
        B.edge(ax, (root[0], root[1] + 0.05), (m[0], m[1] - 0.045), 0.85, 1.0, hue, z=2)
    for k, lx in enumerate(l1x):
        m = l2[0] if k < 2 else l2[1]
        B.edge(ax, (m[0], m[1] + 0.045), (lx, LEAFY - 0.13), 0.7, 1.0, hue, z=2)
    B.node(ax, root, 0.085, hue)
    for m in l2:
        B.node(ax, m, 0.052, hue)
    for k, lx in enumerate(l1x):
        B.place_digit(ax, digs[k], (lx, LEAFY), 0.22, hue, noise=0.12, seed=sd + k,
                      border="0.8", blw=0.6)
    B.text(ax, cx, ROOTY - 0.17, name, size=SML, color=hue)

figstyle.save_fig(fig, "fig1_overview_2row", figdir="figures")
