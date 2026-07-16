"""
nature_style.py
===============
Nature-Physics-grade matplotlib styling for the Water ML-Clustering project.

Design goals (reverse-engineered from Nature Physics figures, e.g.
Li et al., Nat. Phys. 2026, s41567-026-03301-8):

  * Helvetica-like sans-serif (Nimbus Sans / Arial), *regular* weight body text.
    Only panel letters (a, b, c …) are bold.
  * Small, balanced fonts — no 16-pt bold everything.
  * Thin full-box spines (0.8 pt), ticks pointing *inward* on all four sides.
  * NO gridlines (Nature figures are clean white).
  * Translucent error bands, thin capped error bars.
  * A=high-density / disordered  -> RED ;  B=low-density / tetrahedral -> BLUE
    (matches the paper's HDL=red / LDL=blue convention).
  * Probability-density heatmaps in a jet-like `turbo` colormap (the paper's
    Fig. 2d / 3b look), with marginal projection histograms.

Import and call ``set_nature_style()`` once before plotting.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# ─────────────────────────────────────────────────────────────────────────────
# Fonts — prefer a Helvetica clone, fall back gracefully.
# ─────────────────────────────────────────────────────────────────────────────
_AVAILABLE = {f.name for f in fm.fontManager.ttflist}
_SANS_PREF = ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans",
              "FreeSans", "DejaVu Sans"]
SANS = next((f for f in _SANS_PREF if f in _AVAILABLE), "DejaVu Sans")


# ─────────────────────────────────────────────────────────────────────────────
# Colour system
# ─────────────────────────────────────────────────────────────────────────────
# Two local structures (paper convention: high-density / disordered = red,
# low-density / tetrahedral = blue).
#   LFTS = Locally Favoured Tetrahedral Structure  -> low density   -> BLUE
#   DNLS = Disordered Normal-Liquid Structure       -> high density  -> RED
C = {
    "LFTS":   "#2c5fa8",   # deep blue   (low-density, tetrahedral, LDL-like)
    "DNLS":   "#c0392b",   # brick red   (high-density, disordered, HDL-like)
    "LFTS_l": "#7ba3d6",   # light blue  (fills / bands)
    "DNLS_l": "#e08a80",   # light red
    "TOTAL":  "#444444",   # neutral grey for the combined distribution
    "NOISE":  "#b9c0c8",   # light grey for DBSCAN noise
    "ACCENT": "#e6a817",   # amber accent (reference lines, highlights)
    "FSDP":   "#2e7d32",   # green  — first sharp diffraction peak marker
    "D1":     "#c0392b",   # red    — D1 marker
}

# Per-water-model palette (model-comparison figures)
MODEL_COLOR = {
    "TIP4P/2005": "#0f766e",   # deep teal
    "TIP5P":      "#c2410c",   # terracotta
    "SWM4-NDP":   "#6d28d9",   # violet
}
MODEL_MARKER = {"TIP4P/2005": "o", "TIP5P": "s", "SWM4-NDP": "^"}

# Probability-density heatmap colormap (the paper's jet-like look).
CMAP_DENSITY = "turbo"


def temperature_cmap():
    """Blue (cold) -> red (warm) continuous colormap for multi-temperature
    overlays, echoing the isobar palette of the paper's Fig. 1."""
    return LinearSegmentedColormap.from_list(
        "cold_warm",
        ["#08306b", "#2c7fb8", "#41b6c4", "#7fcdbb",
         "#fee391", "#fb8d3c", "#e6550d", "#a50f15"],
    )


def temperature_colors(temps):
    """Map a list of temperatures (°C) to blue->red colours (cold->warm)."""
    temps = np.asarray(temps, dtype=float)
    cmap = temperature_cmap()
    lo, hi = temps.min(), temps.max()
    span = (hi - lo) or 1.0
    return {t: cmap((t - lo) / span) for t in temps}


# ─────────────────────────────────────────────────────────────────────────────
# rcParams
# ─────────────────────────────────────────────────────────────────────────────
def set_nature_style():
    """Apply the global Nature-Physics-style rcParams. Call once."""
    mpl.rcParams.update({
        # --- export ---
        "figure.dpi":        150,
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "figure.facecolor":  "white",
        "pdf.fonttype":      42,    # editable text in vector output
        "ps.fonttype":       42,

        # --- fonts: original (small) Nature sizes, but BOLD weight ---
        "font.family":       "sans-serif",
        "font.sans-serif":   [SANS, "Nimbus Sans", "Arial", "Liberation Sans",
                              "DejaVu Sans"],
        "font.size":         9,
        "axes.titlesize":    9,
        "axes.labelsize":    9.5,
        "axes.titleweight":  "bold",
        "axes.labelweight":  "bold",
        "font.weight":       "bold",
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.fontsize":   8,
        "figure.titlesize":  10,

        # --- math: same small size, but BOLD to match the body text ---
        "mathtext.fontset":  "dejavusans",
        "mathtext.default":  "bf",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,   # avoid missing-glyph boxes for "-"

        # --- axes: thin full box, no grid (unchanged from Nature style) ---
        "axes.linewidth":    0.8,
        "axes.edgecolor":    "#222222",
        "axes.labelcolor":   "#111111",
        "axes.titlecolor":   "#111111",
        "axes.grid":         False,
        "axes.axisbelow":    True,
        "axes.spines.top":   True,
        "axes.spines.right": True,

        # --- ticks: inward, all four sides, thin ---
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.top":         True,
        "ytick.right":       True,
        "xtick.major.size":  3.2,
        "ytick.major.size":  3.2,
        "xtick.minor.size":  1.8,
        "ytick.minor.size":  1.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.color":       "#222222",
        "ytick.color":       "#222222",
        "xtick.labelcolor":  "#111111",
        "ytick.labelcolor":  "#111111",

        # --- lines / legend ---
        "lines.linewidth":   1.4,
        "lines.markersize":  4.0,
        "lines.markeredgewidth": 0.8,
        "legend.frameon":    False,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.4,
        "errorbar.capsize":  2.0,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def panel_label(ax, letter, x=-0.16, y=1.04, **kw):
    """No-op: panel letters (a, b, c, ...) are added later by the user, so we
    deliberately do NOT draw them. Kept callable so existing call sites work."""
    return


def style_ax(ax, top=True, right=True):
    """Final tidy pass on a single axes (inward ticks, no grid)."""
    ax.tick_params(which="both", direction="in", top=top, right=right)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color("#222222")


def thin_colorbar(fig, mappable, ax, label="", pad=0.02, fraction=0.046,
                  **kw):
    """Slim colorbar with inward ticks and a regular-weight label."""
    cbar = fig.colorbar(mappable, ax=ax, pad=pad, fraction=fraction, **kw)
    cbar.outline.set_linewidth(0.8)
    cbar.outline.set_edgecolor("#222222")
    cbar.ax.tick_params(labelsize=8, width=0.8, direction="in")
    if label:
        cbar.set_label(label, fontsize=9)
    return cbar


def savefig(fig, path, **kw):
    """Save at high resolution with consistent padding."""
    fig.savefig(path, dpi=kw.pop("dpi", 600), bbox_inches="tight",
                pad_inches=kw.pop("pad_inches", 0.03), **kw)
    print(f"  saved  {path}")
