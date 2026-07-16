#!/usr/bin/env python3
"""
make_si_figures.py
==================
Build the Supplementary-Information figures.

  cluster_number   BIC / AIC / silhouette vs number of GMM components
                   -> 6_paper_writing/paper_src/images/SI/cluster_number.png
  dbscan_heatmap   DBSCAN (eps, MinPts) silhouette / noise tradeoff (2 panels)
                   -> heatmap_redesign_preview/dbscan_hyperparam_2panel.png
  hdbscan_heatmap  HDBSCAN (min_cluster_size, min_samples) tradeoff (2 panels)
                   -> heatmap_redesign_preview/hdbscan_hyperparam_2panel.png
  method_comp      (zeta, q) overlay for K-means / GMM / HDBSCAN->GMM
                   -> figures_redesign/fig_si_method_comparison.png
  op_figures       per-method order-parameter distributions + scatter
                   -> figures_redesign/fig_si_op_{kmeans,gmm,hdbscan_gmm}.png

Consolidated from si_cluster_number.py, si_hyperparam_tradeoff.py,
si_hyperparam_tradeoff_hdbscan.py, si_method_comparison.py, si_method_figures.py
and the shared hyperparam_panels.py renderer.

Usage:  python make_si_figures.py [name ...] [--recompute]   (default: all)
        names: cluster_number dbscan_heatmap hdbscan_heatmap method_comp op_figures
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from nature_style import (set_nature_style, C, style_ax, panel_label, savefig,
                          thin_colorbar)
from figlib import build, FIGDIR_REDESIGN

set_nature_style()

CLUST = os.path.join(_ROOT, "results", "clustering")
HEATMAP_OUT = os.path.join(_HERE, "heatmap_redesign_preview")
os.makedirs(HEATMAP_OUT, exist_ok=True)
FEATS = ["q_all", "Q6_all", "LSI_all", "Sk_all", "zeta_all"]


# ═════════════════════════════════════════════════════════════════════════════
# Shared two-panel renderer for the density-clustering hyperparameter figures
# (was hyperparam_panels.py)
# ═════════════════════════════════════════════════════════════════════════════
_CONTOUR_TITLE = r"colour = silhouette $s$    lines = % discarded"
_PARETO_TITLE = "higher silhouette costs more discarded data"
_REMOVAL_LABEL = "molecules discarded as noise (%)"


def _smooth_fill(grid, fill, sigma):
    z = np.array(grid, dtype=float)
    z[~np.isfinite(z)] = fill
    return gaussian_filter(z, sigma) if sigma else z


def _panel_contour(ax, fig, sil, noise, xvals, yvals, *, xlabel, ylabel,
                   op_xy, op_text, noise_levels, xscale="linear", yscale="linear",
                   xticks=None, yticks=None, op_xytext=(30, -42), op_rad=0.15,
                   smooth=0.7):
    X, Y = np.meshgrid(np.asarray(xvals, float), np.asarray(yvals, float))
    zsil = _smooth_fill(sil, np.nanmin(sil), smooth)
    znoise = _smooth_fill(noise * 100.0, 100.0, smooth)

    cf = ax.contourf(X, Y, zsil, levels=14, cmap="viridis")
    try:                                # crisp output, no seam lines (mpl-version safe)
        cf.set_edgecolor("face")
    except AttributeError:
        for c in getattr(cf, "collections", []):
            c.set_edgecolor("face")
    cl = ax.contour(X, Y, znoise, levels=noise_levels, colors="white",
                    linewidths=1.0, alpha=0.95)
    ax.clabel(cl, fmt="%d%%", fontsize=6.4, inline=True)

    ox, oy = op_xy
    ax.plot(ox, oy, marker="s", ms=9, mfc=C["ACCENT"], mec="#111111",
            mew=1.0, zorder=6, linestyle="none")
    ax.annotate(op_text, (ox, oy), textcoords="offset points", xytext=op_xytext,
                fontsize=6.6, va="center", ha="left", zorder=7,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="0.5", lw=0.7,
                          alpha=0.95),
                arrowprops=dict(arrowstyle="-", color="#111111", lw=0.9,
                                connectionstyle=f"arc3,rad={op_rad}"))

    ax.set_xscale(xscale); ax.set_yscale(yscale)
    if xticks is not None:
        ax.set_xticks(xticks); ax.set_xticklabels([f"{t:g}" for t in xticks])
    if yticks is not None:
        ax.set_yticks(yticks); ax.set_yticklabels([f"{t:g}" for t in yticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    thin_colorbar(fig, cf, ax, label=r"GMM silhouette $s$")
    ax.set_title(_CONTOUR_TITLE, pad=6)


def _panel_pareto(ax, fig, sil, noise, yvals, *, color_label, op_ij, op_text,
                  cmap="plasma", op_xytext=(14, -30), op_rad=-0.2):
    yvals = np.asarray(yvals, float)
    xs, ys, cs = [], [], []
    for i in range(sil.shape[0]):
        for j in range(sil.shape[1]):
            if not np.isfinite(sil[i, j]):
                continue
            xs.append(noise[i, j] * 100.0); ys.append(sil[i, j]); cs.append(yvals[i])
    xs, ys, cs = np.array(xs), np.array(ys), np.array(cs)

    order = np.argsort(xs)
    fx, fy, best = [], [], -np.inf
    for k in order:
        if ys[k] > best:
            best = ys[k]; fx.append(xs[k]); fy.append(ys[k])
    ax.plot(fx, fy, "-", color="0.45", lw=1.3, alpha=0.9, zorder=2,
            label="efficiency frontier")

    sc = ax.scatter(xs, ys, c=cs, cmap=cmap, s=30, ec="0.15", lw=0.4, zorder=3)

    oi, oj = op_ij
    ox, oy = noise[oi, oj] * 100.0, sil[oi, oj]
    ax.plot(ox, oy, marker="s", ms=9, mfc=C["ACCENT"], mec="#111111",
            mew=1.0, zorder=6, linestyle="none", label="operating point")

    ax.set_xlabel(_REMOVAL_LABEL)
    ax.set_ylabel("GMM silhouette (after noise removal)")
    ax.set_xlim(-3, 103)
    leg = ax.legend(loc="upper left", fontsize=7.6, frameon=True, framealpha=0.92,
                    edgecolor="0.7", handlelength=1.6, handletextpad=0.6,
                    borderpad=0.6, labelspacing=0.5)
    leg.get_frame().set_linewidth(0.6)
    thin_colorbar(fig, sc, ax, label=color_label)
    ax.set_title(_PARETO_TITLE, pad=6)


def _two_panel(sil, noise, *, contour_kw, pareto_kw, out_path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.5), layout="constrained")
    _panel_contour(axA, fig, sil, noise, **contour_kw)
    _panel_pareto(axB, fig, sil, noise, **pareto_kw)
    savefig(fig, out_path)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# cluster_number  —  BIC / AIC / silhouette vs number of GMM components
# ═════════════════════════════════════════════════════════════════════════════
def si_cluster_number(**_):
    set_nature_style()
    labels = os.path.join(CLUST, "tip4p2005_T-20_dbscan_gmm", "cluster_labels.csv")
    ks = [1, 2, 3, 4]
    out = os.path.join(_ROOT, "6_paper_writing", "paper_src", "images", "SI",
                       "cluster_number.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    df = pd.read_csv(labels)
    X = MinMaxScaler().fit_transform(df[FEATS].values)
    print(f"data {X.shape}")

    bic, aic, sil = [], [], []
    for k in ks:
        g = GaussianMixture(n_components=k, covariance_type="full",
                            random_state=42, n_init=5).fit(X)
        bic.append(g.bic(X)); aic.append(g.aic(X))
        if k >= 2:
            s = silhouette_score(X, g.predict(X), sample_size=8000, random_state=0)
            sil.append(s)
        print(f"  k={k}: BIC={g.bic(X):.0f}  AIC={g.aic(X):.0f}"
              + (f"  sil={sil[-1]:.4f}" if k >= 2 else ""))

    bic = np.array(bic) / 1e5
    aic = np.array(aic) / 1e5

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, y, ylab, col in (
            (axes[0], bic, r"BIC  ($\times 10^{5}$)", C["LFTS"]),
            (axes[1], aic, r"AIC  ($\times 10^{5}$)", "#1b7837")):
        ax.plot(ks, y, color=col, lw=1.6, marker="o", ms=5, mfc="white",
                mec=col, mew=1.3)
        ax.set_xlabel("number of components, $k$"); ax.set_ylabel(ylab)
        ax.set_xticks(ks)
        style_ax(ax)

    axes[2].plot(ks[1:], sil, color=C["DNLS"], lw=1.6, marker="o", ms=5,
                 mfc="white", mec=C["DNLS"], mew=1.3)
    axes[2].axvline(2, color="0.6", ls=(0, (4, 3)), lw=1.0)
    axes[2].set_xlabel("number of components, $k$")
    axes[2].set_ylabel("silhouette score"); axes[2].set_xticks(ks)
    style_ax(axes[2])

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.93, wspace=0.42)
    savefig(fig, out)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# dbscan_heatmap  —  DBSCAN (eps, MinPts) silhouette / noise tradeoff
# ═════════════════════════════════════════════════════════════════════════════
def si_dbscan_heatmap(recompute=False, **_):
    set_nature_style()
    labels = os.path.join(CLUST, "tip4p2005_T-20_dbscan_gmm", "cluster_labels.csv")
    eps_grid = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]
    ms_grid = [3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 22, 26, 30, 38, 50]
    op_eps, op_ms = 0.06, 15
    cache = os.path.join(HEATMAP_OUT, "grid_dbscan_gmm.npz")

    def compute_grid():
        df = pd.read_csv(labels)
        X = MinMaxScaler().fit_transform(df[FEATS].values)
        sil = np.full((len(ms_grid), len(eps_grid)), np.nan)
        noise = np.full((len(ms_grid), len(eps_grid)), np.nan)
        for i, ms in enumerate(ms_grid):
            for j, eps in enumerate(eps_grid):
                lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
                keep = lab != -1
                noise[i, j] = 1.0 - keep.mean()
                if keep.sum() < 50:
                    continue
                g = GaussianMixture(n_components=2, random_state=42,
                                    n_init=1).fit_predict(X[keep])
                if len(np.unique(g)) < 2:
                    continue
                sil[i, j] = silhouette_score(X[keep], g, sample_size=5000, random_state=0)
            print(f"  min_samples={ms:>2} done")
        np.savez(cache, sil=sil, noise=noise, eps=np.array(eps_grid), ms=np.array(ms_grid))
        return sil, noise

    if os.path.isfile(cache) and not recompute:
        d = np.load(cache); sil, noise = d["sil"], d["noise"]
        print("loaded cached grid")
    else:
        print("computing grid (DBSCAN -> GMM per cell) ...")
        sil, noise = compute_grid()
    print(f"silhouette range {np.nanmin(sil):.3f}..{np.nanmax(sil):.3f}; "
          f"valid cells {np.isfinite(sil).sum()}/{sil.size}")

    oi, oj = ms_grid.index(op_ms), eps_grid.index(op_eps)
    print(f"operating point eps={op_eps}, MinPts={op_ms}: "
          f"{noise[oi, oj]*100:.1f}% removed, silhouette {sil[oi, oj]:.3f}")

    contour_kw = dict(
        xvals=eps_grid, yvals=ms_grid,
        xlabel=r"DBSCAN $\varepsilon$  (scaled feature space)",
        ylabel="MinPts", xscale="linear", yscale="log",
        xticks=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
        yticks=[3, 5, 10, 20, 30, 50],
        op_xy=(op_eps, op_ms),
        op_text=("operating point\n"
                 r"$\varepsilon$=0.06, MinPts=15" "\n"
                 r"$\approx$23% removed, $s$=0.28"),
        noise_levels=[5, 10, 23, 40, 70],
        op_xytext=(34, 30), op_rad=0.2,
    )
    pareto_kw = dict(
        yvals=ms_grid, color_label="MinPts", op_ij=(oi, oj),
        op_text=("operating point\n" r"$\varepsilon$=0.06, MinPts=15"),
    )
    _two_panel(sil, noise, contour_kw=contour_kw, pareto_kw=pareto_kw,
               out_path=os.path.join(HEATMAP_OUT, "dbscan_hyperparam_2panel.png"))
    print(f"saved dbscan_hyperparam_2panel.png -> {HEATMAP_OUT}")


# ═════════════════════════════════════════════════════════════════════════════
# hdbscan_heatmap  —  HDBSCAN (min_cluster_size, min_samples) tradeoff
# ═════════════════════════════════════════════════════════════════════════════
def si_hdbscan_heatmap(recompute=False, **_):
    set_nature_style()
    labels = os.path.join(CLUST, "tip4p2005_T-20_hdbscan_gmm", "cluster_labels.csv")
    mcs_grid = [5, 8, 12, 18, 26, 40, 60, 100, 180, 300, 500]
    msamp_grid = [3, 5, 7, 8, 10, 13, 17, 22, 30, 45, 60]
    op_mcs, op_msamp = 8, 8
    cache = os.path.join(HEATMAP_OUT, "grid_hdbscan_gmm.npz")

    def compute_grid():
        df = pd.read_csv(labels)
        X = MinMaxScaler().fit_transform(df[FEATS].values)
        sil = np.full((len(msamp_grid), len(mcs_grid)), np.nan)
        noise = np.full((len(msamp_grid), len(mcs_grid)), np.nan)
        for i, ms in enumerate(msamp_grid):
            for j, mcs in enumerate(mcs_grid):
                lab = HDBSCAN(min_cluster_size=int(mcs), min_samples=int(ms),
                              cluster_selection_method="eom", n_jobs=-1).fit_predict(X)
                keep = lab != -1
                noise[i, j] = 1.0 - keep.mean()
                if keep.sum() < 50:
                    continue
                g = GaussianMixture(n_components=2, random_state=42,
                                    n_init=1).fit_predict(X[keep])
                if len(np.unique(g)) < 2:
                    continue
                sil[i, j] = silhouette_score(X[keep], g, sample_size=5000, random_state=0)
            print(f"  min_samples={ms:>2} done")
        np.savez(cache, sil=sil, noise=noise, mcs=np.array(mcs_grid),
                 msamp=np.array(msamp_grid))
        return sil, noise

    if os.path.isfile(cache) and not recompute:
        d = np.load(cache); sil, noise = d["sil"], d["noise"]
        print("loaded cached grid")
    else:
        print("computing grid (HDBSCAN -> GMM per cell) ...")
        sil, noise = compute_grid()
    print(f"silhouette range {np.nanmin(sil):.3f}..{np.nanmax(sil):.3f}; "
          f"valid cells {np.isfinite(sil).sum()}/{sil.size}")

    oi, oj = msamp_grid.index(op_msamp), mcs_grid.index(op_mcs)
    print(f"operating point mcs={op_mcs}, min_samples={op_msamp}: "
          f"{noise[oi, oj]*100:.1f}% removed, silhouette {sil[oi, oj]:.3f}")

    contour_kw = dict(
        xvals=mcs_grid, yvals=msamp_grid,
        xlabel="HDBSCAN min_cluster_size", ylabel="min_samples",
        xscale="log", yscale="log",
        xticks=[5, 10, 20, 50, 100, 200, 500],
        yticks=[3, 5, 10, 20, 40, 60],
        op_xy=(op_mcs, op_msamp),
        op_text=("operating point\n"
                 "mcs=8, min_samples=8\n"
                 r"$\approx$15% removed, $s$=0.26"),
        noise_levels=[10, 15, 30, 60, 90],
        op_xytext=(30, 34), op_rad=-0.2,
    )
    pareto_kw = dict(
        yvals=msamp_grid, color_label="min_samples", op_ij=(oi, oj),
        op_text=("operating point\nmin_cluster_size=8,\nmin_samples=8"),
    )
    _two_panel(sil, noise, contour_kw=contour_kw, pareto_kw=pareto_kw,
               out_path=os.path.join(HEATMAP_OUT, "hdbscan_hyperparam_2panel.png"))
    print(f"saved hdbscan_hyperparam_2panel.png -> {HEATMAP_OUT}")


# ═════════════════════════════════════════════════════════════════════════════
# method_comp  —  (zeta, q) overlay for K-means / GMM / HDBSCAN->GMM
# ═════════════════════════════════════════════════════════════════════════════
def si_method_comparison(**_):
    set_nature_style()
    methods = [("K-means", "kmeans"), ("GMM", "gmm"),
               ("HDBSCAN $\\rightarrow$ GMM", "hdbscan_gmm")]
    df0 = pd.read_csv(os.path.join(CLUST, "tip4p2005_T-20_dbscan_gmm", "cluster_labels.csv"))
    x_all, y_all = df0["zeta_all"].values, df0["q_all"].values
    xlo, xhi = np.percentile(x_all, 0.5), np.percentile(x_all, 99.5)
    ylo, yhi = np.percentile(y_all, 1.0), np.percentile(y_all, 99.8)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (title, meth), letter in zip(axes, methods, "abc"):
        df = pd.read_csv(os.path.join(CLUST, f"tip4p2005_T-20_{meth}", "cluster_labels.csv"))
        lab = df[f"label_{meth}"].values
        x, y = df["zeta_all"].values, df["q_all"].values
        ids = sorted(c for c in np.unique(lab) if c >= 0)
        zmeans = {c: np.nanmean(x[lab == c]) for c in ids}
        lfts = max(zmeans, key=zmeans.get)
        if (lab < 0).any():
            mn = lab < 0
            ax.scatter(x[mn], y[mn], s=1.2, c=C["NOISE"], alpha=0.25, linewidths=0, rasterized=True)
        for cid, name, col in ((lfts, "LFTS", C["LFTS"]),
                               ([c for c in ids if c != lfts][0], "DNLS", C["DNLS"])):
            m = lab == cid
            xs, ys = x[m], y[m]
            sub = rng.choice(len(xs), size=min(5000, len(xs)), replace=False)
            kde = gaussian_kde(np.vstack([xs[sub], ys[sub]]))
            gx, gy = np.mgrid[xlo:xhi:110j, ylo:yhi:110j]
            zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            lv = np.linspace(zz.max() * 0.1, zz.max(), 5)
            ax.contourf(gx, gy, zz, levels=lv, colors=col, alpha=0.16)
            ax.contour(gx, gy, zz, levels=lv, colors=col, linewidths=0.7, alpha=0.9)
            ax.plot([], [], color=col, lw=4, alpha=0.6, label=name)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel(r"$\zeta$  (translational order)")
        ax.set_title(title, fontsize=9)
        style_ax(ax)
        panel_label(ax, letter, x=(-0.12 if letter == "a" else -0.03), y=1.05)
    axes[0].set_ylabel(r"$q$  (tetrahedral order)")
    axes[0].legend(loc="lower right", fontsize=7.5, handlelength=1.1, borderpad=0.3)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.9, wspace=0.08)
    out = os.path.join(FIGDIR_REDESIGN, "fig_si_method_comparison.png")
    savefig(fig, out)
    plt.close(fig)
    print("wrote", out)


# ═════════════════════════════════════════════════════════════════════════════
# op_figures  —  per-method order-parameter distributions + scatter (via figlib.build)
# ═════════════════════════════════════════════════════════════════════════════
def si_op_figures(**_):
    NOISE = "#b9c0c8"  # light grey (matches nature_style C["NOISE"])
    pairs = {
        "kmeans":      ("#5e3c99", "#e66101"),  # purple / orange  (PuOr)
        "gmm":         ("#018571", "#a6611a"),  # teal   / brown   (BrBG)
        "hdbscan_gmm": ("#b386a0", "#7aa794"),  # faint greyish-pink / greyish-green
    }
    cond = {
        "kmeans":      r"TIP4P/2005, $-20\,^{\circ}$C  (K-means)",
        "gmm":         r"TIP4P/2005, $-20\,^{\circ}$C  (GMM)",
        "hdbscan_gmm": r"TIP4P/2005, $-20\,^{\circ}$C  (HDBSCAN$\rightarrow$GMM)",
    }
    for method, (lfts, dnls) in pairs.items():
        show_noise = (method == "hdbscan_gmm")  # only the hybrid pipeline has noise
        build(method, lfts, dnls, NOISE, show_noise, cond[method],
              f"fig_si_op_{method}.png",
              pt_alpha=0.45, noise_alpha=0.45, pt_size=4.0, noise_size=3.4)
        print("wrote", os.path.join(FIGDIR_REDESIGN, f"fig_si_op_{method}.png"))


_FIGS = {
    "cluster_number": si_cluster_number,
    "dbscan_heatmap": si_dbscan_heatmap,
    "hdbscan_heatmap": si_hdbscan_heatmap,
    "method_comp": si_method_comparison,
    "op_figures": si_op_figures,
}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="*", default=["all"],
                    help=" ".join(_FIGS) + " | all")
    ap.add_argument("--recompute", action="store_true",
                    help="rebuild the DBSCAN/HDBSCAN grids instead of using the cache")
    args = ap.parse_args()
    which = list(_FIGS) if args.which == ["all"] or "all" in args.which else args.which
    for name in which:
        print(f"--- {name} ---")
        _FIGS[name](recompute=args.recompute)
    print("done.")
