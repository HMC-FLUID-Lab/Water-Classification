"""Shared builder for SI Figures S1-S3 (one benchmark pipeline each, TIP4P/2005, -20 C).

    <name>_a   per-group distributions of q, LSI, s_k and zeta
    <name>_b   (q, zeta) scatter of the partition
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from style import C, benchmark_data, save, set_style, style_ax

OPS = ["q", "LSI", "s_k", "zeta"]
OP_LABEL = {"q": r"$q$  (tetrahedral order)", "LSI": r"LSI  (nm$^2$)",
            "s_k": r"$S_k$", "zeta": r"$\zeta$  (nm)"}
LO_PCT = {"s_k": 4.0}


def benchmark_figure(method, h_color, l_color, name):
    set_style()
    df = benchmark_data()
    lab = df[method].values
    show_set_aside = (lab < 0).any()
    groups = [(0, "LFTS", h_color, 0.32), (1, "DNLS", l_color, 0.32)]
    if show_set_aside:
        groups.append((-1, "transition", C["NOISE"], 0.18))

    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.9))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.96, wspace=0.42, hspace=0.40)
    for ax, op in zip(axes.ravel(), OPS):
        v = df[op].values
        lo, hi = np.percentile(v, LO_PCT.get(op, 0.3)), np.percentile(v, 99.7)
        bins = np.linspace(lo, hi, 60)
        for cid, gname, col, alpha in groups:
            vv = df.loc[lab == cid, op].values
            w = np.ones_like(vv) / len(vv)
            ax.hist(vv, bins=bins, weights=w, color=col, alpha=alpha, lw=0)
            ax.hist(vv, bins=bins, weights=w, histtype="step", color=col, lw=1.4, label=gname)
        ax.set_xlabel(OP_LABEL[op]); ax.set_ylabel("probability"); ax.set_xlim(lo, hi)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        style_ax(ax)
    handles = [Patch(facecolor=col, edgecolor=col, alpha=alpha, linewidth=1.4)
               for _, _, col, alpha in groups]
    axes.ravel()[0].legend(handles, [g[1] for g in groups], loc="upper left", fontsize=8,
                           handlelength=1.3, handleheight=1.0, labelspacing=0.3,
                           borderpad=0.35, framealpha=0.9)
    save(fig, f"{name}_a")

    fig, ax = plt.subplots(figsize=(3.6, 4.9))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.10, top=0.96)
    x, y = df["q"].values, df["zeta"].values
    if show_set_aside:
        ax.scatter(x[lab < 0], y[lab < 0], s=3.4, c=C["NOISE"], alpha=0.45, lw=0,
                   rasterized=True, label="transition")
    for cid, gname, col in ((1, "DNLS", l_color), (0, "LFTS", h_color)):
        ax.scatter(x[lab == cid], y[lab == cid], s=4.0, c=col, alpha=0.45, lw=0,
                   rasterized=True, label=gname)
    ax.set_xlabel(r"$q$  (tetrahedral order)"); ax.set_ylabel(r"$\zeta$  (nm)")
    ax.set_xlim(np.percentile(x, 0.3), np.percentile(x, 99.7))
    ax.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.9))
    leg = ax.legend(loc="upper left", fontsize=8, markerscale=3, handletextpad=0.3, borderpad=0.3)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    style_ax(ax)
    save(fig, f"{name}_b")
