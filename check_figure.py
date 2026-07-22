"""Automated layout checks for publication figures.

Complements (does not replace) visual inspection of the preview PNG.
Called automatically by figstyle.save_fig(); can also be used directly:

    import check_figure
    check_figure.run_checks(fig, mode="paper")

Checks:
  1. Layout engine actually applied (constrained_layout silently bails out)
  2. Overlapping text elements (tick labels, titles, annotations, legends)
  3. Minimum font size at print scale
  4. Whitespace: how much of the canvas the tight bounding box actually uses
  5. Aspect-locked axes (imshow) leaving slack inside their grid cell
  6. Empty axes / axes without labels (informative)
Prints WARN/OK lines; returns a dict report.
"""

from __future__ import annotations

import itertools
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text

MIN_FONT_PT = {"paper": 6.0, "appendix": 6.0, "slides": 6.5}
# Slides use tueplots' beamer coordinate system (5.5in-wide slide,
# 9pt body / 7pt small fonts), so the floor is just below 7pt.
OVERLAP_FRAC = 0.25          # fraction of the smaller bbox that must intersect
COVERAGE_WARN = 0.85         # warn if the tight bbox covers < 85% of the canvas
ASPECT_FILL_WARN = 0.75      # warn if aspect-locked axes fill < 75% of their cell


def _visible_texts(fig):
    """Collect texts that are actually drawn. Tick labels whose tick location
    lies outside the axis limits exist as visible Text objects but are never
    rendered — including them causes false-positive overlap warnings."""
    out = []

    def _add(t):
        if t is not None and t.get_visible() and t.get_text().strip():
            out.append(t)

    for t in fig.texts:
        _add(t)
    for ax in fig.axes:
        _add(ax.title)
        _add(getattr(ax, "_left_title", None))
        _add(getattr(ax, "_right_title", None))
        _add(ax.xaxis.label)
        _add(ax.yaxis.label)
        for t in ax.texts:
            _add(t)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                _add(t)
        for axis, lims in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lims), max(lims)
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi:
                    for lab in (tick.label1, tick.label2):
                        _add(lab)
    for leg in fig.legends:
        for t in leg.get_texts():
            _add(t)
    return out


def _bbox(t, renderer):
    try:
        return t.get_window_extent(renderer=renderer)
    except Exception:
        return None


def _overlap_frac(b1, b2):
    x0, y0 = max(b1.x0, b2.x0), max(b1.y0, b2.y0)
    x1, y1 = min(b1.x1, b2.x1), min(b1.y1, b2.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    smaller = min(b1.width * b1.height, b2.width * b2.height)
    return inter / smaller if smaller > 0 else 0.0


def _at_default_margins(fig):
    """True if the axes sit exactly at matplotlib's default subplot margins —
    the signature of a layout engine that bailed out and left ~25% of the
    canvas as dead margin."""
    axs = [ax for ax in fig.axes if ax.get_visible()]
    if not axs:
        return False
    ps = [ax.get_position(original=True) for ax in axs]
    ref = (mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"],
           mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"])
    got = (min(p.x0 for p in ps), max(p.x1 for p in ps),
           min(p.y0 for p in ps), max(p.y1 for p in ps))
    return all(abs(a - b) < 1e-6 for a, b in zip(got, ref))


def run_checks(fig, *, mode: str = "paper", name: str = "figure") -> dict:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    report = {"name": name, "warnings": [], "info": []}

    # --- 0. layout engine applied? ----------------------------------------- #
    layout_msg = next((str(w.message) for w in caught
                       if "layout" in str(w.message).lower()), None)
    if layout_msg or _at_default_margins(fig):
        report["warnings"].append(
            (f"layout engine did not run ({layout_msg.splitlines()[0]})"
             if layout_msg else
             "axes sit at matplotlib's default margins — the layout engine "
             "silently bailed out") +
            ". The figure keeps default margins and wastes ~25% of the canvas. "
            "Most common cause: wspace/hspace passed to subgridspec() — set "
            "inner spacing on the layout engine instead (see subplots.md).")

    # --- 1. text overlaps -------------------------------------------------- #
    texts = _visible_texts(fig)
    boxes = [(t, _bbox(t, renderer)) for t in texts]
    boxes = [(t, b) for t, b in boxes if b is not None and b.width > 0]
    n_overlap = 0
    examples = []
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        # skip texts belonging to the same legend (they never truly overlap)
        if t1.get_figure() is None or t2.get_figure() is None:
            continue
        f = _overlap_frac(b1, b2)
        if f > OVERLAP_FRAC:
            n_overlap += 1
            if len(examples) < 3:
                examples.append(f"'{t1.get_text()[:25]}' <-> '{t2.get_text()[:25]}'")
    if n_overlap:
        report["warnings"].append(
            f"{n_overlap} overlapping text pair(s), e.g. {'; '.join(examples)}. "
            "Fix via rotation, fewer ticks, shorter labels, or more width.")

    # --- 2. minimum font size --------------------------------------------- #
    min_pt = min((t.get_fontsize() for t in texts), default=None)
    thr = MIN_FONT_PT.get(mode, 6.0)
    if min_pt is not None and min_pt < thr:
        report["warnings"].append(
            f"Smallest font is {min_pt:.1f}pt (< {thr}pt minimum for {mode}). "
            "Text this small is unreadable at print/projection size.")

    # --- 3. whitespace usage ----------------------------------------------- #
    try:
        tight = fig.get_tightbbox(renderer)  # inches (matplotlib >= 3.6)
        fw, fh = fig.get_size_inches()
        cover = (tight.width * tight.height) / (fw * fh)
        cover = min(cover, 1.0)
        if cover < COVERAGE_WARN:
            report["warnings"].append(
                f"Content covers only {cover:.0%} of the canvas. bbox='tight' "
                "crops the rest away, so the saved PDF ends up smaller than the "
                "venue width. Fix the layout (see check above) or the height "
                "ratio instead of letting it crop.")
        else:
            report["info"].append(f"content/canvas coverage: {cover:.0%}")
    except Exception:
        pass

    # --- 4. aspect-locked axes leaving slack in their cell ------------------ #
    fills = []
    for ax in fig.axes:
        if not ax.get_visible() or ax.get_aspect() == "auto":
            continue
        cell, drawn = ax.get_position(original=True), ax.get_position()
        cell_area = cell.width * cell.height
        if cell_area > 0:
            fills.append((drawn.width * drawn.height) / cell_area)
    tight_fills = [f for f in fills if f < ASPECT_FILL_WARN]
    if tight_fills:
        med = sorted(fills)[len(fills) // 2]
        report["warnings"].append(
            f"{len(tight_fills)} aspect-locked axes (imshow) fill < "
            f"{ASPECT_FILL_WARN:.0%} of their grid cell (median fill {med:.0%}). "
            "Square panels centre themselves in a taller/wider cell, so this "
            "space is invisible but wasted — tune height_to_width_ratio or the "
            "grid's height_ratios until the cells match the panels.")

    # --- 5. axes hygiene ---------------------------------------------------- #
    for i, ax in enumerate(fig.axes):
        if not ax.has_data() and not ax.get_images():
            report["info"].append(f"axes[{i}] has no data (intended?)")
        elif ax.get_visible() and not ax.get_xlabel() and not ax.get_ylabel() \
                and not getattr(ax, "_shared_axes", None):
            report["info"].append(f"axes[{i}] has no axis labels")

    # --- print report ------------------------------------------------------ #
    tag = f"[check:{name}]"
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"{tag} WARN: {w}")
    else:
        print(f"{tag} OK: no layout problems detected")
    for i in report["info"]:
        print(f"{tag} info: {i}")
    print(f"{tag} NOTE: automated checks are incomplete — always view the "
          "preview PNG and judge readability and message clarity visually.")
    return report
