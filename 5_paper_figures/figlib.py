#!/usr/bin/env python3
"""
figlib.py
=========
Shared data loaders and reusable panel builders for the paper-figure scripts.

Consolidated from the old replot_nature_figures.py (data loaders + the
``fig1_two_states`` (zeta, q) panel used for main Fig. 1 top) and
fig6_distributions_scatter.py (the ``build`` order-parameter distribution +
scatter panel, reused for main Fig. 1 bottom AND the SI method figures).

Only the pieces actually feeding the manuscript images are kept here; the
superseded standalone figures (fig2..fig8 of the old replot script) were
archived under _archive/.

LFTS / DNLS conventions (see memory lfts-cluster-id-convention):
  * flat CSV  : LFTS = higher mean zeta cluster.
  * S(k) cache: LFTS = cluster with the larger (FSDP-window - D1-window) S(k),
                detected per condition (IDs are not stable across temperatures).
"""
from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from nature_style import (set_nature_style, C, CMAP_DENSITY, panel_label,
                          style_ax, savefig)

set_nature_style()

FIGDIR = os.path.join(_HERE, "figures")
FIGDIR_REDESIGN = os.path.join(_HERE, "figures_redesign")
CACHE = os.path.join(_HERE, "sk_cache")
CLUST = os.path.join(_ROOT, "results", "clustering")
PARAM = os.path.join(_ROOT, "data", "order_params")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(FIGDIR_REDESIGN, exist_ok=True)

R_OO = 0.285          # O-O nearest-neighbour distance (nm)
KN_RANGE = (0.6, 2.2)  # plotting window in k*r_OO/2pi
FSDP_KN = 0.78         # first sharp diffraction (tetrahedral) marker
D1_KN = 1.00           # principal-peak / disordered marker
D1_COLOR = "#555555"   # neutral grey for the D1 marker (NOT the DNLS-curve red)

OPS = ["q_all", "LSI_all", "Sk_all", "zeta_all"]
OP_LABEL = {"q_all": r"$q$  (tetrahedral order)", "LSI_all": r"LSI  (nm$^2$)",
            "Sk_all": r"$S_k$", "zeta_all": r"$\zeta$  (nm)"}
LO_PCT = {"Sk_all": 4.0}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def k_norm(k):
    return np.asarray(k) * R_OO / (2 * np.pi)


def load_flat(model_temp, method="dbscan_gmm"):
    """Load a flat per-molecule cluster CSV. Returns (df, label_col, lfts_id)."""
    path = os.path.join(CLUST, f"{model_temp}_{method}", "cluster_labels.csv")
    df = pd.read_csv(path)
    label_col = next(c for c in df.columns if c.startswith("label"))
    classified = df[df[label_col] >= 0]
    ids = sorted(classified[label_col].unique())
    means = {c: classified.loc[classified[label_col] == c, "zeta_all"].mean()
             for c in ids}
    lfts = max(means, key=means.get)
    return df, label_col, lfts


def load_cache(tag):
    """Load an S(k) cache. Returns dict with kn, and per-state S/std/pop."""
    path = os.path.join(CACHE, f"sk_{tag}.npz")
    if not os.path.isfile(path):
        return None
    d = np.load(path, allow_pickle=True)
    kn = k_norm(d["k_values"])
    cr = d["cluster_results"].item()
    pops = d["populations"].item()
    fsdp = (kn >= 0.72) & (kn <= 0.86)
    d1 = (kn >= 0.95) & (kn <= 1.10)
    score = {c: np.nanmean(np.asarray(v["S_k_avg"])[fsdp]) -
                np.nanmean(np.asarray(v["S_k_avg"])[d1]) for c, v in cr.items()}
    lfts = max(score, key=score.get)
    dnls = min(score, key=score.get)
    out = {"kn": kn, "n_frames": int(d["n_frames"])}
    for name, cid in (("LFTS", lfts), ("DNLS", dnls)):
        nf = np.asarray(cr[cid]["S_k_frames"])
        out[name] = dict(S=np.asarray(cr[cid]["S_k_avg"]),
                         std=np.asarray(cr[cid]["S_k_std"]),
                         sem=np.asarray(cr[cid]["S_k_std"]) / np.sqrt(max(len(nf), 1)),
                         pop=np.asarray(pops[cid], dtype=float))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Panel builder: two local structures in the (zeta, q) plane  (main Fig. 1 top)
# ─────────────────────────────────────────────────────────────────────────────
def _joint_axes(fig, cell):
    """Build main + top-marginal + right-marginal axes inside one gridspec cell."""
    sub = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=cell, width_ratios=[4.2, 1.0],
        height_ratios=[1.0, 4.2], wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(sub[1, 0])
    axt = fig.add_subplot(sub[0, 0], sharex=ax)
    axr = fig.add_subplot(sub[1, 1], sharey=ax)
    for a in (axt, axr):
        a.tick_params(labelbottom=False, labelleft=False,
                      bottom=False, left=False, top=False, right=False)
        for s in a.spines.values():
            s.set_visible(False)
    return ax, axt, axr


def fig1_two_states(method="dbscan_gmm", outname="fig1_two_states.png"):
    set_nature_style()
    df, lab, lfts = load_flat("tip4p2005_T-20", method=method)
    x = df["zeta_all"].values
    y = df["q_all"].values
    L = df[lab].values
    good = np.isfinite(x) & np.isfinite(y)
    x, y, L = x[good], y[good], L[good]

    xlo, xhi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    ylo, yhi = np.percentile(y, 1.0), np.percentile(y, 99.8)

    fig = plt.figure(figsize=(7.2, 3.5))
    outer = gridspec.GridSpec(1, 2, wspace=0.34, left=0.08, right=0.97,
                              bottom=0.14, top=0.90)

    # ---- panel a : joint density of all molecules ----
    # Smoothed, normalised 2D density drawn as FILLED CONTOURS (a continuous
    # field) so there is no per-cell speckle / isolated white holes.
    from scipy.ndimage import gaussian_filter
    ax, axt, axr = _joint_axes(fig, outer[0, 0])
    nb = 90
    h, xe, ye = np.histogram2d(x, y, bins=nb, range=[[xlo, xhi], [ylo, yhi]])
    h = gaussian_filter(h.T, sigma=1.5)
    h = h / h.max()                      # relative density: peak (mode) = 1, NOT a
                                         # true (area=1) probability density
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    levels = np.linspace(0.04, 1.0, 24)  # below 0.04 stays white (background)
    pm = ax.contourf(xc, yc, h, levels=levels, cmap=CMAP_DENSITY,
                     extend="max", antialiased=True)
    pm.set_edgecolor("face")             # kill faint seams between bands
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$\zeta$  (translational order)")
    ax.set_ylabel(r"$q$  (tetrahedral order)")
    style_ax(ax)
    axt.hist(x, bins=120, range=(xlo, xhi), color=C["TOTAL"], alpha=0.85, lw=0)
    axr.hist(y, bins=120, range=(ylo, yhi), color=C["TOTAL"], alpha=0.85, lw=0,
             orientation="horizontal")
    axt.set_title("All molecules", fontsize=9, pad=3)
    cb = fig.colorbar(pm, ax=axr, fraction=0.32, pad=0.22, aspect=16,
                      ticks=[0.2, 0.4, 0.6, 0.8, 1.0])
    cb.outline.set_linewidth(0.8); cb.ax.tick_params(labelsize=7, direction="in")
    cb.set_label("relative density", fontsize=7.5)
    panel_label(ax, "a", x=-0.20, y=1.32)

    # ---- panel b : resolved into LFTS / DNLS ----
    ax2, axt2, axr2 = _joint_axes(fig, outer[0, 1])
    # transition-zone (DBSCAN noise) molecules as a faint grey underlay
    mn = L < 0
    ax2.scatter(x[mn], y[mn], s=1.5, c=C["NOISE"], alpha=0.30, lw=0,
                rasterized=True, zorder=0)
    axt2.hist(x[mn], bins=110, range=(xlo, xhi), color=C["NOISE"], alpha=0.5,
              lw=0, density=True)
    axr2.hist(y[mn], bins=110, range=(ylo, yhi), color=C["NOISE"], alpha=0.5,
              lw=0, density=True, orientation="horizontal")
    for cid, name, col in ((lfts, "LFTS", C["LFTS"]),
                           ([c for c in np.unique(L) if c >= 0 and c != lfts][0],
                            "DNLS", C["DNLS"])):
        m = L == cid
        xs, ys = x[m], y[m]
        # filled KDE contours (two-blob "two structures" look)
        sub = np.random.default_rng(0).choice(len(xs),
                                               size=min(6000, len(xs)),
                                               replace=False)
        kde = gaussian_kde(np.vstack([xs[sub], ys[sub]]))
        gx, gy = np.mgrid[xlo:xhi:120j, ylo:yhi:120j]
        zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        levels = np.linspace(zz.max() * 0.08, zz.max(), 6)
        ax2.contourf(gx, gy, zz, levels=levels, colors=col, alpha=0.16)
        ax2.contour(gx, gy, zz, levels=levels, colors=col, linewidths=0.7, alpha=0.9)
        axt2.hist(xs, bins=110, range=(xlo, xhi), color=col, alpha=0.55, lw=0,
                  density=True)
        axr2.hist(ys, bins=110, range=(ylo, yhi), color=col, alpha=0.55, lw=0,
                  density=True, orientation="horizontal")
        ax2.plot([], [], color=col, lw=4, alpha=0.6, label=name)
    ax2.scatter([], [], s=10, c=C["NOISE"], label="transition")
    ax2.set_xlim(xlo, xhi); ax2.set_ylim(ylo, yhi)
    ax2.set_xlabel(r"$\zeta$  (translational order)")
    ax2.set_ylabel(r"$q$  (tetrahedral order)")
    style_ax(ax2)
    axt2.set_title("Resolved into two structures", fontsize=9, pad=3)
    ax2.legend(loc="lower right", fontsize=8, handlelength=1.1,
               borderpad=0.3)
    panel_label(ax2, "b", x=-0.20, y=1.32)

    savefig(fig, os.path.join(FIGDIR, outname))
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Panel builder: order-parameter distributions + (q, zeta) scatter
# (main Fig. 1 bottom, and every SI order-parameter figure)
# ─────────────────────────────────────────────────────────────────────────────
def build(method, lfts_col, dnls_col, noise_col, show_noise, condition, outname,
          scatter_method=None, pt_alpha=0.45, noise_alpha=0.25,
          pt_size=2.4, noise_size=2.0):
    set_nature_style()
    df, lab, lfts = load_flat("tip4p2005_T-20", method=method)
    dnls = [c for c in np.unique(df[lab]) if c >= 0 and c != lfts][0]
    # the (e) scatter may use a *different* clustering than the distributions
    if scatter_method and scatter_method != method:
        sdf, slab, slfts = load_flat("tip4p2005_T-20", method=scatter_method)
        sdnls = [c for c in np.unique(sdf[slab]) if c >= 0 and c != slfts][0]
    else:
        sdf, slab, slfts, sdnls = df, lab, lfts, dnls
    groups = [(lfts, "LFTS", lfts_col, 0.32), (dnls, "DNLS", dnls_col, 0.32)]
    if show_noise:
        groups.append((-1, "transition", noise_col, 0.18))

    fig = plt.figure(figsize=(9.6, 4.9))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1.35], wspace=0.38,
                           hspace=0.40, left=0.06, right=0.985, bottom=0.10, top=0.92)
    dist_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                 fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # ---- distributions ----
    for ax, op, letter in zip(dist_axes, OPS, "abcd"):
        v = df[op].values; v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, LO_PCT.get(op, 0.3)), np.percentile(v, 99.7)
        bins = np.linspace(lo, hi, 60)
        for cid, name, col, alpha in groups:
            sel = (df[lab] < 0) if cid == -1 else (df[lab] == cid)
            vv = df.loc[sel, op].values; vv = vv[np.isfinite(vv)]
            w = np.ones_like(vv) / len(vv)
            ax.hist(vv, bins=bins, weights=w, color=col, alpha=alpha, lw=0)
            ax.hist(vv, bins=bins, weights=w, histtype="step", color=col, lw=1.4,
                    label=name)
        ax.set_xlabel(OP_LABEL[op]); ax.set_ylabel("probability"); ax.set_xlim(lo, hi)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))
        style_ax(ax); panel_label(ax, letter, x=-0.20, y=1.04)
    # compact legend with filled handles that match the histogram style
    # (condition is shown as the scatter-panel title, so no bulky legend title)
    leg_h = [Patch(facecolor=col, edgecolor=col, alpha=alpha, linewidth=1.4)
             for _, _, col, alpha in groups]
    leg_l = [name for _, name, _, _ in groups]
    dist_axes[0].legend(leg_h, leg_l, loc="upper left", fontsize=8,
                        handlelength=1.3, handleheight=1.0, labelspacing=0.3,
                        borderpad=0.35, framealpha=0.9)

    # ---- (e) scatter on the right (q on x, zeta on y -- matches the
    #      dbscan_gmm_scatter diagnostic layout) ----
    axs = fig.add_subplot(gs[:, 2])
    x, y = sdf["q_all"].values, sdf["zeta_all"].values
    if show_noise:
        mn = sdf[slab].values < 0
        axs.scatter(x[mn], y[mn], s=noise_size, c=noise_col, alpha=noise_alpha,
                    lw=0, rasterized=True, label="transition")
    for cid, name, col in [(sdnls, "DNLS", dnls_col), (slfts, "LFTS", lfts_col)]:
        m = sdf[slab].values == cid
        axs.scatter(x[m], y[m], s=pt_size, c=col, alpha=pt_alpha, lw=0,
                    rasterized=True, label=name)
    axs.set_xlabel(r"$q$  (tetrahedral order)"); axs.set_ylabel(r"$\zeta$  (nm)")
    axs.set_xlim(np.percentile(x, 0.3), np.percentile(x, 99.7))
    axs.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.9))
    axs.set_title(condition, fontsize=9)
    leg = axs.legend(loc="upper left", fontsize=8, markerscale=3,
                     handletextpad=0.3, borderpad=0.3)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    style_ax(axs); panel_label(axs, "e", x=-0.12, y=1.02)

    savefig(fig, os.path.join(FIGDIR_REDESIGN, outname))
    plt.close(fig)
