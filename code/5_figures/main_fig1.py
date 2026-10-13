#!/usr/bin/env python3
"""Main-text Figure 1: classification of TIP4P/2005 water at -20 C (reference sample).

    main_fig1_a  joint (zeta, q) density of all molecules, with marginals
    main_fig1_b  LFTS / DNLS density contours, transitional molecules in grey
    main_fig1_c  per-state distributions f(q), f(LSI), f(s_k), f(zeta)
    main_fig1_d  (q, zeta) scatter of the classified molecules
"""
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from style import C, CMAP_DENSITY, joint_axes, reference_data, save, set_style, style_ax

OPS = ["q", "LSI", "s_k", "zeta"]
OP_LABEL = {"q": r"$q$  (tetrahedral order)", "LSI": r"LSI  (nm$^2$)",
            "s_k": r"$s_k$", "zeta": r"$\zeta$  (nm)"}
OP_FLABEL = {"q": r"$f(q)$", "LSI": r"$f(\mathrm{LSI})$", "s_k": r"$f(s_k)$",
             "zeta": r"$f(\zeta)$"}
LO_PCT = {"s_k": 4.0}
LFTS, DNLS = 0, 1


def limits(df):
    x, y = df["zeta"].values, df["q"].values
    return (np.percentile(x, 0.5), np.percentile(x, 99.5),
            np.percentile(y, 1.0), np.percentile(y, 99.8))


def panel_a(df):
    xlo, xhi, ylo, yhi = limits(df)
    x, y = df["zeta"].values, df["q"].values
    fig = plt.figure(figsize=(4.0, 3.6))
    gs = gridspec.GridSpec(1, 1, left=0.15, right=0.86, bottom=0.14, top=0.94)
    ax, axt, axr = joint_axes(fig, gs[0, 0])
    h, xe, ye = np.histogram2d(x, y, bins=90, range=[[xlo, xhi], [ylo, yhi]])
    h = gaussian_filter(h.T, sigma=1.5)
    h /= h.max()                                       # relative density: mode = 1
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    pm = ax.contourf(xc, yc, h, levels=np.linspace(0.04, 1.0, 24), cmap=CMAP_DENSITY,
                     extend="max", antialiased=True)
    pm.set_edgecolor("face")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$\zeta$  (translational order)")
    ax.set_ylabel(r"$q$  (tetrahedral order)")
    style_ax(ax)
    axt.hist(x, bins=120, range=(xlo, xhi), color=C["TOTAL"], alpha=0.85, lw=0)
    axr.hist(y, bins=120, range=(ylo, yhi), color=C["TOTAL"], alpha=0.85, lw=0,
             orientation="horizontal")
    cb = fig.colorbar(pm, ax=axr, fraction=0.32, pad=0.22, aspect=16,
                      ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    cb.outline.set_linewidth(0.8); cb.ax.tick_params(labelsize=7, direction="in")
    cb.set_label("relative density", fontsize=7.5)
    save(fig, "main_fig1_a")


def panel_b(df):
    xlo, xhi, ylo, yhi = limits(df)
    x, y, L = df["zeta"].values, df["q"].values, df["label"].values
    fig = plt.figure(figsize=(4.0, 3.6))
    gs = gridspec.GridSpec(1, 1, left=0.15, right=0.97, bottom=0.14, top=0.94)
    ax, axt, axr = joint_axes(fig, gs[0, 0])
    mn = L < 0
    ax.scatter(x[mn], y[mn], s=1.5, c=C["NOISE"], alpha=0.30, lw=0, rasterized=True, zorder=0)
    axt.hist(x[mn], bins=110, range=(xlo, xhi), color=C["NOISE"], alpha=0.5, lw=0, density=True)
    axr.hist(y[mn], bins=110, range=(ylo, yhi), color=C["NOISE"], alpha=0.5, lw=0,
             density=True, orientation="horizontal")
    for cid, name in ((LFTS, "LFTS"), (DNLS, "DNLS")):
        col = C[name]
        xs, ys = x[L == cid], y[L == cid]
        sub = np.random.default_rng(0).choice(len(xs), size=min(6000, len(xs)), replace=False)
        kde = gaussian_kde(np.vstack([xs[sub], ys[sub]]))
        gx, gy = np.mgrid[xlo:xhi:120j, ylo:yhi:120j]
        zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        levels = np.linspace(zz.max() * 0.08, zz.max(), 6)
        ax.contourf(gx, gy, zz, levels=levels, colors=col, alpha=0.16)
        ax.contour(gx, gy, zz, levels=levels, colors=col, linewidths=0.7, alpha=0.9)
        axt.hist(xs, bins=110, range=(xlo, xhi), color=col, alpha=0.55, lw=0, density=True)
        axr.hist(ys, bins=110, range=(ylo, yhi), color=col, alpha=0.55, lw=0, density=True,
                 orientation="horizontal")
        ax.plot([], [], color=col, lw=4, alpha=0.6, label=name)
    ax.scatter([], [], s=10, c=C["NOISE"], label="transition")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$\zeta$  (translational order)")
    ax.set_ylabel(r"$q$  (tetrahedral order)")
    style_ax(ax)
    ax.legend(loc="lower right", fontsize=8, handlelength=1.1, borderpad=0.3)
    save(fig, "main_fig1_b")


def panel_c(df):
    groups = [(LFTS, "LFTS", C["LFTS"], 0.32), (DNLS, "DNLS", C["DNLS"], 0.32),
              (-1, "transition", C["NOISE"], 0.18)]
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.7))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.96, wspace=0.34, hspace=0.42)
    for ax, op in zip(axes.ravel(), OPS):
        v = df[op].values
        lo, hi = np.percentile(v, LO_PCT.get(op, 0.3)), np.percentile(v, 99.7)
        bins = np.linspace(lo, hi, 60)
        for cid, name, col, alpha in groups:
            vv = df.loc[df["label"] == cid, op].values
            w = np.ones_like(vv) / len(vv)
            ax.hist(vv, bins=bins, weights=w, color=col, alpha=alpha, lw=0)
            ax.hist(vv, bins=bins, weights=w, histtype="step", color=col, lw=1.4, label=name)
        ax.set_xlabel(OP_LABEL[op]); ax.set_ylabel(OP_FLABEL[op]); ax.set_xlim(lo, hi)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        style_ax(ax)
    handles = [Patch(facecolor=col, edgecolor=col, alpha=alpha, linewidth=1.4)
               for _, _, col, alpha in groups]
    axes.ravel()[0].legend(handles, [g[1] for g in groups], loc="upper left", fontsize=8,
                           handlelength=1.3, handleheight=1.0, labelspacing=0.3,
                           borderpad=0.35, framealpha=0.9)
    save(fig, "main_fig1_c")


def panel_d(df):
    fig, ax = plt.subplots(figsize=(4.3, 4.8))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.11, top=0.96)
    x, y, L = df["q"].values, df["zeta"].values, df["label"].values
    ax.scatter(x[L < 0], y[L < 0], s=3.4, c=C["NOISE"], alpha=0.6, lw=0, rasterized=True,
               label="transition")
    for cid, name in ((DNLS, "DNLS"), (LFTS, "LFTS")):
        ax.scatter(x[L == cid], y[L == cid], s=4.0, c=C[name], alpha=0.45, lw=0,
                   rasterized=True, label=name)
    ax.set_xlabel(r"$q$  (tetrahedral order)"); ax.set_ylabel(r"$\zeta$  (nm)")
    ax.set_xlim(np.percentile(x, 0.3), np.percentile(x, 99.7))
    ax.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.9))
    leg = ax.legend(loc="upper left", fontsize=8, markerscale=3, handletextpad=0.3, borderpad=0.3)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    style_ax(ax)
    save(fig, "main_fig1_d")


if __name__ == "__main__":
    set_style()
    df = reference_data()
    for panel in (panel_a, panel_b, panel_c, panel_d):
        panel(df)
