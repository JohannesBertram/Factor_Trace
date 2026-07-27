"""figstyle: project-wide styling for publication-ready figures (papers + slides).

Built on tueplots. Copy this file (and check_figure.py) into the paper project
so figure scripts run standalone. Typical use in a figure script:

    import figstyle
    figstyle.apply(venue="neurips2024", width="half", nrows=1, ncols=2)
    fig, axs = plt.subplots(1, 2)
    ...
    figstyle.save_fig(fig, "main_comparison")

Modes:
    mode="paper"    (default) main-text figure
    mode="appendix" same width, relaxed height ratio
    mode="slides"   beamer 16:9 sizing; pass figs_per_slide=1|2|3

LaTeX is auto-detected; falls back to matplotlib mathtext ("cm") if absent.
"""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

try:
    import yaml
except ImportError:  # yaml only needed for .figstyle configs
    yaml = None

from tueplots import figsizes, fonts, fontsizes
from tueplots.constants.color import palettes

# --------------------------------------------------------------------------- #
# Colors: colorblind-safe Paul Tol palettes (via tueplots)
# --------------------------------------------------------------------------- #

def _hx(c):
    return f"#{c}" if not str(c).startswith("#") else str(c)

# Named colors from Paul Tol "bright" (colorblind-safe, 7 colors)
PALETTE = {
    "blue":   _hx(palettes.paultol_bright[0]),
    "red":    _hx(palettes.paultol_bright[1]),
    "green":  _hx(palettes.paultol_bright[2]),
    "yellow": _hx(palettes.paultol_bright[3]),
    "cyan":   _hx(palettes.paultol_bright[4]),
    "purple": _hx(palettes.paultol_bright[5]),
    "grey":   _hx(palettes.paultol_bright[6]),
}

# Panel labels — "(a)", "(b)", ... and the descriptive text next to them — use
# ONE size across every figure so the paper reads as a set. Sized for a full-width
# figure (the default here); pass an explicit size to label_axes/panel_label only
# for a genuinely narrower figure. Keeping it here (not per-figure) is what makes
# the labels match from figure to figure.
PANEL_LABEL_SIZE = 7.5

# Default categorical cycles (in priority order)
CYCLE_DEFAULT = [_hx(c) for c in palettes.paultol_bright]        # up to 7 series
CYCLE_MANY = [_hx(c) for c in palettes.paultol_muted]            # up to 10 series
CYCLE_HIGH_CONTRAST = [_hx(c) for c in palettes.paultol_high_contrast]  # up to 3

_SEMANTIC: dict = {}          # loaded from .figstyle/colors.yaml
# Config and output dirs resolve against this file's directory (the project
# root, where figstyle.py is copied), not the CWD — so a figure script run from
# a subdirectory or a notebook gets the same styling and the same figures/ dir.
_PROJECT_ROOT = Path(__file__).resolve().parent
_FIGSTYLE_DIR = _PROJECT_ROOT / ".figstyle"
_YAML_MISSING = False         # colors.yaml exists but PyYAML is not installed
_CURRENT = {"mode": "paper", "venue": None, "figs_per_slide": 1}


def color(name: str) -> str:
    """Semantic color lookup: project component name -> hex.

    Resolution order: .figstyle/colors.yaml -> PALETTE -> literal hex.
    Keeping all component->color assignments in colors.yaml is what makes
    colors consistent across every figure of the paper.
    """
    if not _SEMANTIC:
        _load_semantic()
    if name in _SEMANTIC:
        val = _SEMANTIC[name]
        return PALETTE.get(val, _hx(val))
    if name in PALETTE:
        return PALETTE[name]
    if str(name).lstrip("#").replace("_", "").isalnum() and len(str(name).lstrip("#")) in (3, 6):
        return _hx(name)
    hint = (" PyYAML is not installed, so colors.yaml was ignored — "
            "`pip install pyyaml`." if _YAML_MISSING else "")
    raise KeyError(
        f"Unknown color '{name}'. Add it to {_FIGSTYLE_DIR / 'colors.yaml'} "
        f"or use one of {sorted(PALETTE)}.{hint}"
    )


def _load_semantic():
    global _YAML_MISSING
    path = _FIGSTYLE_DIR / "colors.yaml"
    if not path.exists():
        return
    if yaml is None:
        _YAML_MISSING = True
        return
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    _SEMANTIC.update(data.get("components", data))


# --------------------------------------------------------------------------- #
# LaTeX detection
# --------------------------------------------------------------------------- #

def latex_available() -> bool:
    """True if a usable LaTeX toolchain for matplotlib usetex exists."""
    if os.environ.get("FIGSTYLE_NO_LATEX"):
        return False
    gs = shutil.which("gs") or shutil.which("gswin64c") or shutil.which("ghostscript")
    renderer = shutil.which("dvipng") or gs
    return shutil.which("latex") is not None and renderer is not None


def _font_config(venue: str, usetex: bool, slides: bool) -> dict:
    if slides:
        # Sans-serif for slides; avoid usetex so figures build anywhere fast.
        return {
            "font.family": "sans-serif",
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
        }
    if usetex:
        fn = getattr(fonts, f"{venue}_tex", None)
        if fn is not None:
            return fn(family="serif")
        return {"text.usetex": True, "font.family": "serif"}
    # No LaTeX: venue non-tex fonts if available, else Computer-Modern-like mathtext
    fn = getattr(fonts, venue, None)
    if fn is not None:
        cfg = dict(fn())
        cfg["text.usetex"] = False
        return cfg
    return {"font.family": "serif", "text.usetex": False, "mathtext.fontset": "cm"}


# --------------------------------------------------------------------------- #
# Venue resolution
# --------------------------------------------------------------------------- #

def _custom_venue(venue: str):
    """Load an unknown venue from .figstyle/venue.yaml (see references/venues.md)."""
    path = _FIGSTYLE_DIR / "venue.yaml"
    if yaml is None or not path.exists():
        return None
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("name", "").lower() != venue.lower() and venue != "custom":
        return None
    return cfg


def _figsize_from_widths(text_width_in, col_width_in, width, nrows, ncols,
                         height_to_width_ratio, pad_inches=0.015, rel_width=1.0):
    base = text_width_in if width == "full" else (col_width_in or text_width_in / 2.0)
    w = base * rel_width
    subplot_w = w / ncols
    h = subplot_w * height_to_width_ratio * nrows
    return {
        "figure.figsize": (w, h),
        "figure.constrained_layout.use": True,
        "figure.autolayout": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": pad_inches,
    }


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def apply(venue: str = "neurips2024", *, width: str = "full",
          nrows: int = 1, ncols: int = 1, mode: str = "paper",
          figs_per_slide: int = 1, height_to_width_ratio: float | None = None,
          rel_width: float = 1.0, usetex: str | bool = "auto",
          n_series: int | None = None) -> dict:
    """Configure matplotlib rcParams for a venue / mode. Returns the rc dict.

    venue: tueplots venue key (e.g. "neurips2024", "icml2024", "iclr2024",
           "colm2026", ...) or a custom name defined in .figstyle/venue.yaml.
    width: "full" (text width) or "half" (column / half text width).
    nrows/ncols: subplot grid — sizes each panel correctly.
    mode: "paper" | "appendix" | "slides".
    figs_per_slide: slides only; 1 (default), 2, or 3 figures on one slide.
    height_to_width_ratio: per-panel h/w. Defaults: golden ratio (paper),
        0.75 (appendix — taller is acceptable there).
    usetex: "auto" (detect), True, or False.
    n_series: if >7, switches to the 10-color muted cycle automatically.
    """
    if usetex == "auto":
        use_tex = latex_available()
    else:
        use_tex = bool(usetex)
    if use_tex and not latex_available():
        warnings.warn("usetex requested but LaTeX toolchain not found; "
                      "falling back to mathtext.")
        use_tex = False

    slides = mode == "slides"
    if height_to_width_ratio is None:
        height_to_width_ratio = 0.75 if mode == "appendix" else 0.6180339887498949

    rc: dict = {}

    if slides:
        if figs_per_slide not in (1, 2, 3):
            raise ValueError("figs_per_slide must be 1, 2, or 3 (default 1; "
                             "use >1 only for clearly linked figures).")
        # Width of *this one figure* as fraction of a 16:9 beamer slide.
        rel_w = {1: 0.85, 2: 0.46, 3: 0.31}[figs_per_slide]
        rel_h = {1: 0.62, 2: 0.55, 3: 0.50}[figs_per_slide]
        rc.update(figsizes.beamer_169(rel_width=rel_w, rel_height=rel_h))
        rc["savefig.bbox"] = "tight"
        rc.update(fontsizes.beamer())
        rc.update(_font_config(venue, usetex=False, slides=True))
    else:
        fs_fn = getattr(figsizes, f"{venue}_{width}", None) or getattr(figsizes, venue, None)
        if fs_fn is not None:
            kwargs = dict(nrows=nrows, ncols=ncols,
                          height_to_width_ratio=height_to_width_ratio,
                          rel_width=rel_width)
            if fs_fn.__name__ in ("iclr2023", "iclr2024", "neurips2021", "neurips2022",
                                  "neurips2023", "neurips2024", "tmlr2023", "jmlr2001",
                                  "colm2026", "eccv2024", "tue_ai_thesis") and width == "half":
                kwargs["rel_width"] = 0.5 * rel_width  # single-column venues: half = rel_width
            try:
                rc.update(fs_fn(**kwargs))
            except TypeError:
                # some presets (icml/aistats *_half) accept no rel_width
                kwargs.pop("rel_width", None)
                rc.update(fs_fn(**kwargs))
            ft_fn = getattr(fontsizes, venue, None)
            if ft_fn is not None:
                rc.update(ft_fn())
        else:
            cfg = _custom_venue(venue)
            if cfg is None:
                raise ValueError(
                    f"Venue '{venue}' not in tueplots and no .figstyle/venue.yaml found. "
                    "Follow references/venues.md to research and cache the venue."
                )
            rc.update(_figsize_from_widths(
                cfg["text_width_in"], cfg.get("column_width_in"), width,
                nrows, ncols, height_to_width_ratio, rel_width=rel_width))
            base = float(cfg.get("font_size_pt", 10))
            small = base - 2
            rc.update({
                "font.size": base - 1, "axes.labelsize": base - 1,
                "axes.titlesize": base - 1, "legend.fontsize": small,
                "xtick.labelsize": small, "ytick.labelsize": small,
            })
        rc.update(_font_config(venue, use_tex, slides=False))

    # Color cycle
    cycle = CYCLE_MANY if (n_series or 0) > 7 else CYCLE_DEFAULT
    rc["axes.prop_cycle"] = matplotlib.cycler(color=cycle)

    plt.rcParams.update(rc)
    _CURRENT.update(mode=mode, venue=venue, figs_per_slide=figs_per_slide)
    return rc


# --------------------------------------------------------------------------- #
# Helpers: subplot labels, shared legend, dual axis
# --------------------------------------------------------------------------- #

def label_axes(axs, *, style="(a)", loc="top-left", size=None, **text_kw):
    """Label subplots (a), (b), ... so captions can reference each panel.

    style: "(a)" | "a" | "A" | "(A)".  loc: "top-left" | "title-left".
    size: font size; defaults to PANEL_LABEL_SIZE so every figure matches.
    """
    import numpy as np
    axs = np.atleast_1d(axs).ravel()
    letters = "abcdefghijklmnopqrstuvwxyz"
    size = PANEL_LABEL_SIZE if size is None else size
    for i, ax in enumerate(axs):
        ch = letters[i]
        if "A" in style:
            ch = ch.upper()
        lab = f"({ch})" if "(" in style else ch
        kw = dict(fontweight="bold", va="top", ha="left", fontsize=size)
        kw.update(text_kw)
        if loc == "title-left":
            ax.set_title(lab, loc="left", fontweight="bold", fontsize=size)
        else:
            ax.text(0.02, 0.98, lab, transform=ax.transAxes, **kw)


def shared_legend(fig, axs=None, *, loc="above", ncol=None, **kw):
    """One legend for the whole figure (dedup labels) — saves vertical space.

    loc: "above" | "below" | "right".
    """
    import numpy as np
    if axs is None:
        axs = fig.axes
    axs = np.atleast_1d(axs).ravel()
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    if ncol is None:
        ncol = len(labels) if loc in ("above", "below") else 1
    anchors = {"above": (0.5, 1.02, "lower center"),
               "below": (0.5, -0.02, "upper center"),
               "right": (1.02, 0.5, "center left")}
    x, y, l = anchors[loc]
    return fig.legend(handles, labels, loc=l, bbox_to_anchor=(x, y),
                      ncol=ncol, frameon=False, **kw)


def _layout_engine_live(fig) -> bool:
    """True while a layout engine still repositions axes on every draw.
    set_layout_engine('none') leaves a PlaceHolderLayoutEngine, not None."""
    eng = fig.get_layout_engine()
    if eng is None:
        return False
    try:
        from matplotlib.layout_engine import PlaceHolderLayoutEngine
        return not isinstance(eng, PlaceHolderLayoutEngine)
    except ImportError:
        return True


def freeze(fig):
    """Run the layout engine once, then switch it off. Returns fig.

    Call this before adding anything that needs *final* axes positions —
    panel labels for nested/mixed grids, connector lines between panels,
    block headings. Adding such artists while the layout engine is live
    either moves them or (if they live inside an axes) inflates that grid
    cell. See references/subplots.md.
    """
    fig.canvas.draw()
    fig.set_layout_engine("none")
    return fig


def panel_label(anchor, text, *, dx=0.0, dy=0.012, loc="above", **text_kw):
    """Label a panel or a block of panels, anchored to an axes. Use after freeze().

    anchor: an Axes, or a sequence of Axes (the first one is used).
    loc:    "above" (default, top-left corner of the anchor) or "left"
            (left of the anchor, vertically centred — for row labels).
    dx/dy:  offsets in figure fractions.

    Works where label_axes() cannot: nested gridspecs, image grids, labels
    that span several axes.
    """
    ax = anchor[0] if isinstance(anchor, (list, tuple)) else anchor
    fig = ax.get_figure()
    if _layout_engine_live(fig):
        warnings.warn("panel_label() before figstyle.freeze(fig): the label "
                      "will be misplaced once the layout engine runs.")
    p = ax.get_position()
    kw = dict(fontweight="bold", fontsize=PANEL_LABEL_SIZE)
    kw.update(text_kw)
    if loc == "left":
        return fig.text(max(p.x0 - 0.008 + dx, 0.002), p.y0 + p.height / 2 + dy,
                        text, ha="right", va="center", **kw)
    # clamp inside the canvas: a label placed at x<0 or y>1 gets cropped away
    return fig.text(max(p.x0 + dx, 0.002), min(p.y1 + dy, 0.995), text,
                    ha="left", va="bottom", **kw)


def dual_axis(ax, *, left_color=None, right_color=None):
    """Create a twin y-axis. Dual axes are allowed; color-code the two axes
    (labels + ticks) to match their series so readers can tell them apart."""
    ax2 = ax.twinx()
    if left_color:
        c = color(left_color)
        ax.yaxis.label.set_color(c)
        ax.tick_params(axis="y", colors=c)
    if right_color:
        c = color(right_color)
        ax2.yaxis.label.set_color(c)
        ax2.tick_params(axis="y", colors=c)
    return ax2


# --------------------------------------------------------------------------- #
# Saving + preview + checks
# --------------------------------------------------------------------------- #

def save_fig(fig, name: str, *, figdir: str = "figures", preview: bool = True,
             checks: bool = True, preview_dpi: int = 200) -> Path:
    """Save PDF to <figdir>/<name>.pdf (+ preview PNG for visual inspection),
    then run automated layout checks and print a report. Returns the PDF path.
    """
    figdir = Path(figdir)
    if not figdir.is_absolute():
        figdir = _PROJECT_ROOT / figdir
    if _CURRENT["mode"] == "slides":
        figdir = figdir / "slides"
    elif _CURRENT["mode"] == "appendix":
        figdir = figdir / "appendix"
    figdir.mkdir(parents=True, exist_ok=True)
    pdf_path = figdir / f"{name}.pdf"
    fig.savefig(pdf_path)

    if preview:
        pdir = figdir / "preview"
        pdir.mkdir(exist_ok=True)
        fig.savefig(pdir / f"{name}.png", dpi=preview_dpi)

    if checks:
        try:
            import check_figure
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "check_figure", Path(__file__).parent / "check_figure.py")
            check_figure = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(check_figure)
        check_figure.run_checks(fig, mode=_CURRENT["mode"], name=name)

    print(f"[figstyle] saved {pdf_path}" + (f" (+ preview/{name}.png)" if preview else ""))
    return pdf_path
