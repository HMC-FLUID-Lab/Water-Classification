"""Shared matplotlib style, colours, data loaders and output helper for the figure scripts.

Every panel is written as its own PNG (600 dpi) to figures/, named
main_fig<N>_<panel>.png or appendix_fig<N>_<panel>.png (appendix = Supporting
Information Figure S<N>). Panel letters and panel titles are not drawn.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

_AVAILABLE = {f.name for f in fm.fontManager.ttflist}
SANS = next((f for f in ["Helvetica", "Arial", "Nimbus Sans", "Liberation Sans",
                         "FreeSans", "DejaVu Sans"] if f in _AVAILABLE), "DejaVu Sans")

C = {
    "LFTS": "#2c5fa8",    # blue: locally favoured tetrahedral structure (H)
    "DNLS": "#c0392b",    # red: disordered normal-liquid structure (L)
    "LFTS_l": "#7ba3d6",
    "DNLS_l": "#e08a80",
    "TOTAL": "#444444",
    "NOISE": "#b9c0c8",   # grey: transitional / set aside
}
CMAP_DENSITY = "turbo"
LFTS_BAND, DNLS_BAND = (0.775, 0.825), (0.975, 1.025)   # k r_OO/2pi marks of k_T1, k_D1


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "savefig.facecolor": "white",
        "figure.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": [SANS, "Nimbus Sans", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9.5,
        "axes.titleweight": "bold", "axes.labelweight": "bold", "font.weight": "bold",
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.titlesize": 10,
        "mathtext.fontset": "dejavusans", "mathtext.default": "bf",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "axes.linewidth": 0.8, "axes.edgecolor": "#222222", "axes.labelcolor": "#111111",
        "axes.titlecolor": "#111111", "axes.grid": False, "axes.axisbelow": True,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
        "ytick.right": True, "xtick.major.size": 3.2, "ytick.major.size": 3.2,
        "xtick.minor.size": 1.8, "ytick.minor.size": 1.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.color": "#222222", "ytick.color": "#222222",
        "xtick.labelcolor": "#111111", "ytick.labelcolor": "#111111",
        "lines.linewidth": 1.4, "lines.markersize": 4.0, "lines.markeredgewidth": 0.8,
        "legend.frameon": False, "legend.handlelength": 1.5, "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.35, "legend.borderaxespad": 0.4, "errorbar.capsize": 2.0,
    })


def style_ax(ax, top=True, right=True):
    ax.tick_params(which="both", direction="in", top=top, right=right)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
        s.set_color("#222222")


def temperature_colors(temps):
    """Blue (cold) -> grey -> red (warm), coolwarm colormap."""
    temps = np.asarray(temps, dtype=float)
    lo, hi = temps.min(), temps.max()
    cmap = mpl.colormaps["coolwarm"]
    return {t: cmap((t - lo) / ((hi - lo) or 1.0)) for t in temps}


def single_axes(width, height, left=0.6, right=0.15, bottom=0.5, top=0.15):
    """Figure holding one axes of exactly width x height inches."""
    fig = plt.figure(figsize=(left + width + right, bottom + height + top))
    W, H = fig.get_size_inches()
    ax = fig.add_axes([left / W, bottom / H, width / W, height / H])
    return fig, ax


def joint_axes(fig, cell):
    """Main axes + top and right marginal axes inside one gridspec cell."""
    sub = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=cell, width_ratios=[4.2, 1.0],
                                           height_ratios=[1.0, 4.2], wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(sub[1, 0])
    axt = fig.add_subplot(sub[0, 0], sharex=ax)
    axr = fig.add_subplot(sub[1, 1], sharey=ax)
    for a in (axt, axr):
        a.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False,
                      top=False, right=False)
        for s in a.spines.values():
            s.set_visible(False)
    return ax, axt, axr


def save(fig, name, pad_inches=0.03, **kw):
    os.makedirs(config.FIGURES, exist_ok=True)
    path = os.path.join(config.FIGURES, name + ".png")
    fig.savefig(path, dpi=600, bbox_inches="tight", pad_inches=pad_inches, **kw)
    plt.close(fig)
    print(f"  wrote {path}")


# ── data ─────────────────────────────────────────────────────────────────────
CODE = {"LFTS": 0, "DNLS": 1, "transitional": -1}


def reference_data():
    """Order parameters + likelihood-ratio label (0 LFTS, 1 DNLS, -1 transitional)
    for the reference sample (TIP4P/2005, -20 C, Run01, first 20 frames)."""
    model, temp, run = config.REFERENCE
    n = config.N_FRAMES * config.N_MOL
    with np.load(config.order_parameter_file(model, temp, run)) as d:
        df = pd.DataFrame({k: d[k][:config.N_FRAMES].ravel() for k in d.files})
    df["label"] = pd.read_csv(config.label_file(model, temp, run))["label"].map(CODE).to_numpy()[:n]
    return df


def benchmark_data():
    """Reference order parameters + benchmark partitions (0 H, 1 L, -1 set aside)."""
    df = reference_data()
    path = config.label_file(*config.REFERENCE).replace("_labels.csv", "_benchmark_labels.csv")
    b = pd.read_csv(path)
    for col in ("kmeans", "gmm", "hdbscan_gmm"):
        df[col] = b[col].map({"H": 0, "L": 1, "set_aside": -1}).to_numpy()
    return df


def structure_factor(model, temp, run=None):
    run = run or config.MAIN_RUN[model]
    path = os.path.join(config.RESULTS, "structure_factors",
                        config.tag(model, temp, run) + "_Sk.npz")
    with np.load(path) as d:
        return {k: d[k] for k in d.files}
