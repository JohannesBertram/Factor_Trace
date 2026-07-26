"""Draft A — faithful, polished 3-panel banner (improves the hand drawn draft)."""
import sys, numpy as np
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE)))
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle
import bftlib as B

W, H = 6.975, 2.04
figstyle.apply(venue="aaai2024", width="full", nrows=1, ncols=1, mode="paper",
               height_to_width_ratio=H / W)
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])
B.canvas(ax, W, H)

EVEN, ODD, INK, GREY, GREEN, OURS = B.EVEN, B.ODD, B.INK, B.GREY, B.GREEN, B.OURS
HDR, TXT, SML = 8.5, 7.5, 6.5


def header(x, letter, title):
    B.text(ax, x, 0.12, f"({letter})", size=HDR, weight="bold", ha="left", va="center")
    B.text(ax, x + 0.15, 0.12, title, size=HDR, ha="left", va="center")


def chipglyph(x, y, ch, border, s=0.15):
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, facecolor="black",
                           edgecolor=border, lw=1.3, zorder=7))
    ax.text(x, y, ch, color="white", fontsize=6.5, ha="center", va="center",
            zorder=8, family="monospace")


# ======================================================================= (a)
header(0.05, "a", "Neuron input arbor")
nx, ny = 0.60, 0.50
inx, iny = [0.33, 0.60, 0.87], 1.02
B.node(ax, (nx, ny), 0.10, INK, active=True)
B.text(ax, nx, ny, "$n$", size=TXT, color="white", z=9)
for x in inx:
    B.node(ax, (x, iny), 0.056, GREY, active=True)
w_even = [0.95, 0.30, 0.55]      # stimulus "0" (even, blue)
w_odd  = [0.30, 0.95, 0.55]      # stimulus "1" (odd, red)
for x, we, wo in zip(inx, w_even, w_odd):
    B.edge(ax, (nx - 0.035, ny + 0.02), (x - 0.028, iny - 0.05), we, 1.0, EVEN, z=2)
    B.edge(ax, (nx + 0.035, ny + 0.02), (x + 0.028, iny - 0.05), wo, 1.0, ODD, z=2)
eqy = 1.50
B.text(ax, 0.12, eqy, "$n($", size=TXT, ha="left", va="center")
chipglyph(0.315, eqy, "0", EVEN, s=0.13)
B.text(ax, 0.395, eqy, "$)=n($", size=TXT, ha="left", va="center")
chipglyph(0.775, eqy, "1", ODD, s=0.13)
B.text(ax, 0.85, eqy, "$)$", size=TXT, ha="left", va="center")
B.text(ax, 0.60, 1.82, "same activation, different arbors", size=SML - 0.5,
       style="italic", color="0.4")

B.arrow(ax, (1.28, 0.78), (1.60, 0.78), lw=1.4)
B.text(ax, 1.44, 0.62, "BFT", size=SML, weight="bold", color=OURS)

# ======================================================================= (b)
header(1.66, "b", "Backward Factor Trace")
MID = 0.99                       # vertical midline of the network / matrices

# --- network: trace the output layer's fan-in ---
oe, oo = (1.90, 0.48), (2.20, 0.48)
hid = [(1.80, MID), (2.05, MID), (2.30, MID)]
inp = [(1.90, 1.50), (2.20, 1.50)]
B.node(ax, oe, 0.082, EVEN); B.text(ax, oe[0], oe[1] - 0.17, "even", size=SML, color=EVEN)
B.node(ax, oo, 0.075, ODD);  B.text(ax, oo[0], oo[1] - 0.17, "odd", size=SML, color=ODD)
for h in hid:
    B.node(ax, h, 0.052, GREY)
for o in (oe, oo):
    for h in hid:
        B.edge(ax, (o[0], o[1] + 0.05), (h[0], h[1] - 0.05), 0.5, 1.0, "0.72", z=1)
for h in hid:
    for i in inp:
        B.edge(ax, (h[0], h[1] + 0.03), (i[0], i[1] - 0.06), 0.35, 1.0, "0.83", z=0)
chipglyph(inp[0][0], inp[0][1] + 0.02, "0", EVEN, s=0.15)
chipglyph(inp[1][0], inp[1][1] + 0.02, "1", ODD, s=0.15)

B.arrow(ax, (2.46, MID), (2.68, MID), lw=1.1)

# --- arbor matrix J ---
jc = 0.12
jx = 2.86
jy = MID - jc                    # centered on MID
Jm = np.array([[0.9, 0.15, 0.6, 0.25], [0.2, 0.85, 0.55, 0.3]])
B.cell_grid(ax, jx, jy, jc, Jm, B.gray_fn(1.0))
chipglyph(jx - 0.13, jy + jc * 0.5, "0", EVEN, s=0.115)
chipglyph(jx - 0.13, jy + jc * 1.5, "1", ODD, s=0.115)
B.text(ax, jx + 2 * jc, jy - 0.11, "arbor matrix $J$", size=SML, color=INK)

B.text(ax, jx + 4 * jc + 0.115, MID, r"$\approx$", size=11, va="center")

# --- NMF: H x W ---
hx = jx + 4 * jc + 0.30
Hm = np.array([[0.95, 0.12], [0.1, 0.9]])
B.cell_grid(ax, hx, jy, jc, Hm, B.tintcols_fn([EVEN, ODD]))
B.text(ax, hx + 0.5 * jc, jy - 0.10, "$f_0$", size=SML, color=EVEN)
B.text(ax, hx + 1.5 * jc, jy - 0.10, "$f_1$", size=SML, color=ODD)
B.text(ax, hx + 2 * jc + 0.115, MID, r"$\times$", size=8.5, va="center")
wx = hx + 2 * jc + 0.36
Wm = np.array([[0.85, 0.1, 0.55, 0.2], [0.15, 0.8, 0.5, 0.35]])
B.cell_grid(ax, wx, jy, jc, Wm, B.tintrows_fn([EVEN, ODD]))
B.text(ax, wx - 0.105, jy + 0.5 * jc, "$f_0$", size=SML, color=EVEN, ha="right")
B.text(ax, wx - 0.105, jy + 1.5 * jc, "$f_1$", size=SML, color=ODD, ha="right")
bx0, bx1 = hx - 0.20, wx + 4 * jc + 0.05
B.rbox(ax, bx0, jy - 0.26, bx1, jy + 2 * jc + 0.22, ec="0.6", lw=0.9, pad=0.0, rounding=0.03)
B.text(ax, (bx0 + bx1) / 2, jy - 0.18, "NMF", size=SML, weight="bold", color="0.4")
B.text(ax, hx + jc, jy + 2 * jc + 0.11, "stimulus", size=SML - 0.5, color="0.45")
B.text(ax, wx + 2 * jc, jy + 2 * jc + 0.11, "connection", size=SML - 0.5, color="0.45")
B.text(ax, (bx0 + bx1) / 2, jy + 2 * jc + 0.42,
       "each factor $=$ a sub-circuit", size=SML - 0.5, style="italic",
       color=B.shade(GREEN, 0.1))

# bridge arrow (b) -> (c)
B.arrow(ax, (bx1 + 0.08, MID), (bx1 + 0.38, MID), lw=1.4)
B.text(ax, bx1 + 0.23, MID - 0.16, "recurse", size=SML, weight="bold", color=OURS)

# ======================================================================= (c)
cx0 = bx1 + 0.46
header(cx0, "c", "Factor tree")
B.text(ax, cx0 + 0.02, 0.50, "$L_2$", size=SML, color="0.45", ha="left", va="center")
B.text(ax, cx0 + 0.02, 1.04, "$L_1$", size=SML, color="0.45", ha="left", va="center")
x0 = cx0 + 0.42
root = [(x0 + 0.10, 0.50, EVEN, "$f_0$"), (x0 + 0.92, 0.50, ODD, "$f_1$")]
child = [(x0 - 0.05, 1.04, EVEN), (x0 + 0.25, 1.04, EVEN),
         (x0 + 0.77, 1.04, ODD), (x0 + 1.07, 1.04, ODD)]
leaf = [("0", EVEN, 0), ("0", EVEN, 1), ("1", ODD, 2), ("1", ODD, 3)]
for rx, ry, rc, _ in root:
    for cx, cy, cc in child:
        if cc == rc:
            B.edge(ax, (rx, ry + 0.05), (cx, cy - 0.05), 0.8, 1.0, rc, z=2)
for (rx, ry, rc, lab) in root:
    B.node(ax, (rx, ry), 0.08, rc)
    B.text(ax, rx, ry, lab, size=SML, color="white", z=9)
for cx, cy, cc in child:
    B.node(ax, (cx, cy), 0.055, cc)
ly = 1.55
for (cx, cy, cc), (ch, lc, sd) in zip(child, leaf):
    B.edge(ax, (cx, cy + 0.05), (cx, ly - 0.17), 0.7, 1.0, lc, z=1)
    B.place_digit(ax, ch, (cx, ly), 0.24, lc, noise=0.13, seed=sd, border="0.8", blw=0.6)

figstyle.save_fig(fig, "fig1_overview_strip", figdir="figures")
