#!/usr/bin/env python3
"""
make_main_figures.py
====================
Build the main-text manuscript figures from the cached data.

  fig1  ->  figures_redesign/1_clustering_conf.png
            (a,b) two states in the (zeta, q) plane  [figlib.fig1_two_states]
            (c,d) order-parameter distributions + scatter  [figlib.build]
            stacked + panel-lettered by assemble_fig1().

  fig2  ->  figures_redesign/fig_validation_sk_conf.png   (row a)
            figures_redesign/fig_skzeta_restyled_conf.png (rows b/c, S(k,zeta))
            The published 2_validation.png was hand-assembled from row a plus
            the S(k,zeta) panels; assemble_fig2() stitches row a with the
            pre-exported 3D/contour strips (fig_skzeta_conf_rowb/rowc.png) when
            those are present.

  fig3  ->  figures_redesign/figR4_generality.png   (== 3_generality.png)
            S(k) vs T for each state + population(T) + model comparison.

Consolidated from the old validation_conf.py, redesign_skzeta.py,
redesign_paper_figures.py (figR4 only) and assemble_composites.py.

Usage:  python make_main_figures.py [fig1|fig2|fig3|all]   (default: all)
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from PIL import Image

os.environ.setdefault("OMP_NUM_THREADS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from nature_style import (set_nature_style, C, MODEL_COLOR, temperature_cmap,
                          panel_label, style_ax, savefig)
from figlib import (fig1_two_states, build, load_cache, k_norm, R_OO,
                    KN_RANGE, FSDP_KN, D1_KN, FIGDIR, FIGDIR_REDESIGN)

CACHE = os.path.join(_HERE, "sk_cache")
RC = os.path.join(_HERE, "redesign_cache")
CLUST = os.path.join(_ROOT, "results", "clustering")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1  —  clustering composite (a,b two states  +  c,d distributions)
# ═════════════════════════════════════════════════════════════════════════════
def _stack(paths, out_base, gap_px=24):
    """Resize images to a common width and stack vertically -> out_base PNG.
    Returns (width, height, y_tops, heights) with y positions in figure fraction
    (bottom-origin)."""
    ims = [Image.open(p).convert("RGB") for p in paths]
    W = max(im.width for im in ims)
    ims = [im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
           for im in ims]
    H = sum(im.height for im in ims) + gap_px * (len(ims) - 1)
    canvas = Image.new("RGB", (W, H), "white")
    y = 0
    tops_px = []
    for im in ims:
        tops_px.append(y)
        canvas.paste(im, (0, y))
        y += im.height + gap_px
    canvas.save(out_base)
    y_tops = [1.0 - (t / H) for t in tops_px]
    heights = [im.height / H for im in ims]
    return W, H, y_tops, heights


def _annotate(base_png, out_png, letters, boxes=None, dpi=300, fontsize=26):
    """Overlay panel letters and dashed boxes (all in figure-fraction coords).

    Rendered under matplotlib's default rcParams (not the nature style applied by
    figlib) so the panel letters match the original hand-assembly step exactly."""
    with mpl.rc_context(mpl.rcParamsDefault):
        arr = mpimg.imread(base_png)
        h, w = arr.shape[:2]
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr)
        ax.axis("off")
        for (x, y, txt) in letters:
            fig.text(x, y, txt, fontsize=fontsize, fontweight="bold",
                     ha="left", va="top", family="sans-serif")
        for (x, y, bw, bh) in (boxes or []):
            ax.add_patch(Rectangle((x, y), bw, bh, transform=fig.transFigure,
                                   fill=False, ls=(0, (6, 4)), lw=1.6,
                                   ec="0.25", clip_on=False))
        fig.savefig(out_png, dpi=dpi)
        plt.close(fig)
    print(f"  wrote {out_png}")


def assemble_fig1(out_png):
    """Stack the (zeta,q) panel over the distribution panel; add a,b,c,d + box."""
    top = os.path.join(FIGDIR, "fig1_two_states_conf.png")
    bot = os.path.join(FIGDIR_REDESIGN, "fig6_order_parameters_conf.png")
    base = os.path.join(FIGDIR_REDESIGN, "_fig1_base.png")
    W, H, y_tops, hts = _stack([top, bot], base)
    yt, yb = y_tops[0], y_tops[1]
    letters = [
        (0.010, yt - 0.005, "a"),
        (0.500, yt - 0.005, "b"),
        (0.010, yb - 0.005, "c"),
        (0.610, yb - 0.005, "d"),
    ]
    # dashed box around panel c (the 2x2 distribution grid, left ~60% of bottom img)
    boxes = [(0.006, 0.012, 0.600, hts[1] - 0.028)]
    _annotate(base, out_png, letters, boxes)


def fig1():
    """Panels + composite for the clustering figure (confidence-cleavage labels)."""
    fig1_two_states(method="dbscan_gmm_conf", outname="fig1_two_states_conf.png")
    build("dbscan_gmm_conf", C["LFTS"], C["DNLS"], C["NOISE"], True,
          r"TIP4P/2005, $-20\,^{\circ}$C  (confidence cleavage)",
          "fig6_order_parameters_conf.png",
          pt_alpha=0.45, noise_alpha=0.75, pt_size=4.0, noise_size=5.0)
    assemble_fig1(os.path.join(FIGDIR_REDESIGN, "1_clustering_conf.png"))


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 row a  —  per-cluster + total S(k) for the confidence-based labels,
#                    plus the novel transition-only S(k).
# ═════════════════════════════════════════════════════════════════════════════
_VAL_RC = {
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.weight": "bold", "axes.labelweight": "bold", "axes.titleweight": "bold",
    "mathtext.fontset": "dejavusans", "mathtext.default": "bf",
    "font.size": 10.5, "axes.labelsize": 13, "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8,
    "axes.linewidth": 1.0, "xtick.direction": "in", "ytick.direction": "in",
    "axes.unicode_minus": False, "savefig.dpi": 600,
}
_VAL_LEG = dict(loc="upper right", frameon=True, edgecolor="0.7", framealpha=0.92,
                fontsize=7, handlelength=1.1, handletextpad=0.4, labelspacing=0.3,
                borderpad=0.3, markerscale=0.85)
_VAL_C_LFTS = "#5b9bd5"      # cornflower blue
_VAL_C_DNLS = "#d9534f"      # salmon red
_VAL_C_TRANS = "#6f42c1"     # purple — transition / "noise"
_VAL_LFTS_BAND = (0.725, 0.775)
_VAL_DNLS_BAND = (0.975, 1.025)
_VAL_XR = (0.6, 2.0)


def _val_load_conf():
    """(kn, dict) with LFTS/DNLS/transition each {S,std,sem,pop}; LFTS/DNLS
    picked among the two states by the FSDP pre-peak enhancement."""
    cache = os.path.join(CACHE, "sk_4p_T-20_conf.npz")
    d = np.load(cache, allow_pickle=True)
    kn = k_norm(d["k_values"])
    cr = d["cluster_results"].item()
    pops = d["populations"].item()
    tkey = int(d["transition_key"])
    fsdp = (kn >= 0.72) & (kn <= 0.86)
    d1 = (kn >= 0.95) & (kn <= 1.10)
    states = [c for c in cr if c != tkey]
    score = {c: np.nanmean(np.asarray(cr[c]["S_k_avg"])[fsdp]) -
                np.nanmean(np.asarray(cr[c]["S_k_avg"])[d1]) for c in states}
    lfts, dnls = max(score, key=score.get), min(score, key=score.get)

    def pack(cid):
        nf = np.asarray(cr[cid]["S_k_frames"])
        return dict(S=np.asarray(cr[cid]["S_k_avg"]),
                    std=np.asarray(cr[cid]["S_k_std"]),
                    sem=np.asarray(cr[cid]["S_k_std"]) / np.sqrt(max(len(nf), 1)),
                    pop=float(np.asarray(pops[cid]).mean()))
    return kn, {"LFTS": pack(lfts), "DNLS": pack(dnls), "transition": pack(tkey)}


def _val_bands(ax):
    ax.axvspan(*_VAL_LFTS_BAND, color=_VAL_C_LFTS, alpha=0.22, lw=0, zorder=0,
               label="LFTS region")
    ax.axvspan(*_VAL_DNLS_BAND, color=_VAL_C_DNLS, alpha=0.22, lw=0, zorder=0,
               label="DNLS region")


def _val_grid(ax):
    ax.grid(True, alpha=0.25, ls="--", lw=0.6)
    ax.tick_params(direction="in", top=True, right=True)


def fig2_validation(make_noise=True):
    """Row a of 2_validation (per-cluster + total=superposition) and, optionally,
    the transition-only S(k) diagnostic (fig_noise_sk.png)."""
    with mpl.rc_context(_VAL_RC):
        kn, D = _val_load_conf()
        print(f"populations/frame: LFTS={D['LFTS']['pop']:.0f}  "
              f"DNLS={D['DNLS']['pop']:.0f}  transition={D['transition']['pop']:.0f}")
        total = os.path.join(RC, "total_sk_4p_T-20.npz")
        m = (kn >= _VAL_XR[0]) & (kn <= _VAL_XR[1])
        tot = np.load(total)
        knt = k_norm(tot["k_values"]); St = tot["S_total"]
        mt = (knt >= _VAL_XR[0]) & (knt <= _VAL_XR[1])

        # ---- row a ----
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
        for name, col, mk in (("LFTS", _VAL_C_LFTS, "o"), ("DNLS", _VAL_C_DNLS, "D")):
            S, sem = D[name]["S"], D[name]["sem"]
            ax1.plot(kn[m], S[m], color=col, lw=1.4, zorder=3)
            ev = slice(0, None, 8)
            ax1.errorbar(kn[m][ev], S[m][ev], yerr=sem[m][ev], fmt=mk, ms=4.0,
                         mfc="white", mec=col, mew=1.0, color=col, ecolor=col,
                         elinewidth=0.8, capsize=1.8, label=name, zorder=4)
        _val_bands(ax1)
        ax1.set_xlim(*_VAL_XR); ax1.set_ylim(0.5, 1.6)
        ax1.set_xlabel(r"$k \cdot r_{OO}/2\pi$"); ax1.set_ylabel(r"$S(k)$")
        ax1.legend(**_VAL_LEG); ax1.set_title("Per-cluster", fontsize=10.5)
        _val_grid(ax1)
        ax2.plot(kn[m], D["LFTS"]["S"][m], color=_VAL_C_LFTS, lw=1.1, alpha=0.40, zorder=2)
        ax2.plot(kn[m], D["DNLS"]["S"][m], color=_VAL_C_DNLS, lw=1.1, alpha=0.40, zorder=2)
        ax2.plot(knt[mt], St[mt], color="#333333", lw=1.7, zorder=3,
                 label="total (all molecules)")
        ev = slice(0, None, 8)
        ax2.plot(knt[mt][ev], St[mt][ev], "o", ms=4.0, mfc="white", mec="#333333",
                 mew=1.0, zorder=4)
        _val_bands(ax2)
        ax2.set_xlim(*_VAL_XR); ax2.set_xlabel(r"$k \cdot r_{OO}/2\pi$")
        ax2.legend(**_VAL_LEG); ax2.set_title("Total = superposition", fontsize=10.5)
        _val_grid(ax2)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.15, top=0.91, wspace=0.06)
        fig.savefig(os.path.join(FIGDIR_REDESIGN, "fig_validation_sk_conf.png"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)
        print("saved fig_validation_sk_conf.png")

        if not make_noise:
            return
        # ---- transition-only S(k) (novel; no analogue in the two-state picture)
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        ax.plot(kn[m], D["LFTS"]["S"][m], color=_VAL_C_LFTS, lw=1.2, alpha=0.55, label="LFTS")
        ax.plot(kn[m], D["DNLS"]["S"][m], color=_VAL_C_DNLS, lw=1.2, alpha=0.55, label="DNLS")
        ax.plot(knt[mt], St[mt], color="#333333", lw=1.2, alpha=0.45, ls="--", label="total")
        S, sem = D["transition"]["S"], D["transition"]["sem"]
        ax.plot(kn[m], S[m], color=_VAL_C_TRANS, lw=1.8, zorder=4)
        ev = slice(0, None, 8)
        ax.errorbar(kn[m][ev], S[m][ev], yerr=sem[m][ev], fmt="s", ms=4.2, mfc="white",
                    mec=_VAL_C_TRANS, mew=1.1, color=_VAL_C_TRANS, ecolor=_VAL_C_TRANS,
                    elinewidth=0.8, capsize=1.8,
                    label=f"transition ({D['transition']['pop']/1024*100:.1f}%)",
                    zorder=5)
        _val_bands(ax)
        ax.set_xlim(*_VAL_XR); ax.set_ylim(0.5, 1.6)
        ax.set_xlabel(r"$k \cdot r_{OO}/2\pi$"); ax.set_ylabel(r"$S(k)$")
        ax.set_title(r"Transition-state S(k)  (TIP4P/2005, $-20\,^{\circ}$C)", fontsize=10.5)
        ax.legend(**{**_VAL_LEG, "loc": "upper right"})
        _val_grid(ax)
        fig.subplots_adjust(left=0.14, right=0.97, bottom=0.14, top=0.91)
        fig.savefig(os.path.join(FIGDIR_REDESIGN, "fig_noise_sk.png"),
                    dpi=600, bbox_inches="tight")
        plt.close(fig)
        print("saved fig_noise_sk.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 rows b/c  —  per-cluster S(k, zeta) surfaces + contour maps
#   zeta kept in nm throughout (consistent with the clustering/SI figures).
# ═════════════════════════════════════════════════════════════════════════════
_SKZ_SIM = os.path.join(_ROOT, "data", "simulations", "tip4p2005")
_SKZ_DCD = os.path.join(_SKZ_SIM, "dcd_tip4p2005_T-20_N1024_Run01_0.dcd")
_SKZ_PDB = os.path.join(_SKZ_SIM, "inistate_tip4p2005_T-20_N1024_Run01.pdb")
_SKZ_FLAT = os.path.join(CLUST, "tip4p2005_T-20_dbscan_gmm", "cluster_labels.csv")
_SKZ_FLAT_CONF = os.path.join(CLUST, "tip4p2005_T-20_dbscan_gmm_conf", "cluster_labels.csv")
_SKZ_N_MOL = 1024
_SKZ_K_VALUES = np.linspace(0.1, 50.0, 300)
_SKZ_RC = 1.5
_SKZ_N_FRAMES = 20
_SKZ_K_RANGE = (0.4, 2.0)
_SKZ_S_RANGE = (0.0, 2.0)
_SKZ_CACHE = os.path.join(RC, "skzeta_percluster_4p_T-20.npz")
_SKZ_CACHE_CONF = os.path.join(RC, "skzeta_percluster_4p_T-20_conf.npz")


def _skz_knorm(k):
    return k * R_OO / (2 * np.pi)


def _skz_compute(flat, label_col, cache):
    import time
    sys.path.insert(0, os.path.join(_ROOT, "4_structure_factor"))
    from structure_factor_bycluster import load_trajectory
    from sk_zeta_3d import compute_sk_zeta_matrix
    t0 = time.time()
    traj = load_trajectory(_SKZ_DCD, _SKZ_PDB)[:_SKZ_N_FRAMES]
    df = pd.read_csv(flat)
    n = _SKZ_N_FRAMES * _SKZ_N_MOL
    lm = df[label_col].values[:n].astype(int).reshape(_SKZ_N_FRAMES, _SKZ_N_MOL)
    zeta = df["zeta_all"].values[:n].reshape(_SKZ_N_FRAMES, _SKZ_N_MOL)  # keep zeta in nm
    out = {"k_values": _SKZ_K_VALUES}
    for cid in (0, 1):
        zc = zeta[lm == cid]
        lo, hi = np.percentile(zc, 1), np.percentile(zc, 99)
        zbins = np.linspace(lo, hi, 34)
        print(f"  cluster {cid}: zeta-mean={zc.mean():.4f} nm, bins [{lo:.3f},{hi:.3f}]")
        Skz, zcent = compute_sk_zeta_matrix(traj, _SKZ_K_VALUES, lm, cid, zeta, zbins, _SKZ_RC)
        out[f"S_{cid}"] = Skz
        out[f"zeta_{cid}"] = zcent
        out[f"zmean_{cid}"] = float(zc.mean())
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    print(f"  cached -> {cache}  ({time.time()-t0:.1f}s)")


def _skz_prep(M, zc, kn):
    kk = (kn >= _SKZ_K_RANGE[0]) & (kn <= _SKZ_K_RANGE[1])
    M = np.ma.masked_invalid(M[:, kk])
    M = np.ma.masked_outside(M, _SKZ_S_RANGE[0], _SKZ_S_RANGE[1])
    filled = gaussian_filter(M.filled(np.nan), 0.6)
    return kn[kk], filled


def _skz_plot(cache, outname, title):
    set_nature_style()
    d = np.load(cache, allow_pickle=True)
    kn = _skz_knorm(d["k_values"])
    lfts = 0 if d["zmean_0"] > d["zmean_1"] else 1
    dnls = 1 - lfts
    order = [(lfts, "LFTS", C["LFTS"]), (dnls, "DNLS", C["DNLS"])]

    fig = plt.figure(figsize=(7.2, 6.2))
    cmap = "turbo"
    # ---- top row: 3D surfaces ----
    for j, (cid, name, col) in enumerate(order):
        ax = fig.add_subplot(2, 2, j + 1, projection="3d")
        knx, M = _skz_prep(d[f"S_{cid}"], d[f"zeta_{cid}"], kn)
        zc = d[f"zeta_{cid}"]
        K, Z = np.meshgrid(knx, zc)
        surf = ax.plot_surface(K, Z, np.nan_to_num(M, nan=_SKZ_S_RANGE[0]),
                               cmap=cmap, vmin=_SKZ_S_RANGE[0], vmax=_SKZ_S_RANGE[1],
                               rstride=1, cstride=2, linewidth=0, antialiased=True)
        ax.set_xlim(*_SKZ_K_RANGE); ax.set_zlim(*_SKZ_S_RANGE)
        ax.set_xlabel(r"$k\,r_{\mathrm{OO}}/2\pi$", labelpad=-2, fontsize=8)
        ax.set_ylabel("ζ  (nm)", labelpad=-2, fontsize=8)
        ax.set_zlabel(r"$S(k,\zeta)$", labelpad=-4, fontsize=8)
        ax.tick_params(labelsize=6.5, pad=-1)
        ax.set_title(name, fontsize=9, color=col, pad=-2)
        ax.view_init(elev=28, azim=-122)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.pane.set_facecolor("white"); pane.pane.set_edgecolor("#cccccc")
            pane.pane.set_alpha(1.0)
        ax.grid(True)

    # ---- bottom row: 2D contour maps ----
    for j, (cid, name, col) in enumerate(order):
        ax = fig.add_subplot(2, 2, j + 3)
        knx, M = _skz_prep(d[f"S_{cid}"], d[f"zeta_{cid}"], kn)
        zc = d[f"zeta_{cid}"]
        pm = ax.contourf(knx, zc, M, levels=np.linspace(_SKZ_S_RANGE[0], _SKZ_S_RANGE[1], 21),
                         cmap=cmap, extend="neither")
        ax.axvline(0.75, color="#0b3d91", ls=(0, (5, 3)), lw=1.1)
        ax.axvline(1.00, color="#7a0010", ls=(0, (5, 3)), lw=1.1)
        ax.text(0.75, zc.max(), r"$k_{T1}$", color="#0b3d91", fontsize=7.5,
                ha="center", va="bottom")
        ax.text(1.00, zc.max(), r"$k_{D1}$", color="#7a0010", fontsize=7.5,
                ha="center", va="bottom")
        ax.set_xlim(*_SKZ_K_RANGE); ax.set_ylim(zc.min(), zc.max())
        ax.set_xlabel(r"$k\,r_{\mathrm{OO}}/2\pi$")
        ax.set_ylabel("ζ  (nm)")
        ax.set_title(name, fontsize=9, color=col)
        style_ax(ax)
        cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.03,
                          ticks=[0, 0.5, 1.0, 1.5, 2.0])
        cb.set_label(r"$S(k,\zeta)$", fontsize=8); cb.ax.tick_params(labelsize=7)
        cb.outline.set_linewidth(0.8)
        panel_label(ax, "cd"[j], x=-0.16, y=1.04)

    fig.suptitle(title, fontsize=9.5, y=0.99)
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.08, top=0.93,
                        wspace=0.28, hspace=0.28)
    savefig(fig, os.path.join(FIGDIR_REDESIGN, outname))
    plt.close(fig)


def fig2_skzeta(conf=True, recompute=False):
    """Per-cluster S(k, zeta) 3D surfaces + 2D contour maps (fig2 rows b/c)."""
    if conf:
        flat, label, cache, outname = (
            _SKZ_FLAT_CONF, "label_dbscan_gmm_conf", _SKZ_CACHE_CONF,
            "fig_skzeta_restyled_conf.png")
        title = (r"$\zeta$-resolved structure factor  "
                 r"(TIP4P/2005, $-20\,^{\circ}$C, confidence)")
    else:
        flat, label, cache, outname = (
            _SKZ_FLAT, "label_dbscan_gmm", _SKZ_CACHE, "fig_skzeta_restyled.png")
        title = (r"$\zeta$-resolved structure factor  "
                 r"(TIP4P/2005, $-20\,^{\circ}$C)")
    if recompute or not os.path.isfile(cache):
        _skz_compute(flat, label, cache)
    _skz_plot(cache, outname, title)


def _split_skzeta_strips(combined="fig_skzeta_restyled_conf.png"):
    """Split the S(k,zeta) 2x2 figure into a top (3D surfaces) and bottom (2D
    contours) strip, so assemble_fig2 can stack them as rows b and c."""
    src = os.path.join(FIGDIR_REDESIGN, combined)
    if not os.path.isfile(src):
        return None, None
    im = Image.open(src).convert("RGB")
    W, H = im.size
    mid = int(round(H * 0.5))
    row_b = os.path.join(FIGDIR_REDESIGN, "fig_skzeta_conf_rowb.png")
    row_c = os.path.join(FIGDIR_REDESIGN, "fig_skzeta_conf_rowc.png")
    im.crop((0, 0, W, mid)).save(row_b)
    im.crop((0, mid, W, H)).save(row_c)
    print(f"  split {combined} -> rowb (3D) + rowc (2D)")
    return row_b, row_c


def assemble_fig2(out_png):
    """Stitch row a with the 3D-surface (row b) and 2D-contour (row c) strips.

    Prefers hand-cropped strips (fig_skzeta_conf_rowb/rowc.png); if those are
    absent it splits the fig2_skzeta 2x2 (fig_skzeta_restyled_conf.png) into its
    top (3D) and bottom (2D) halves automatically."""
    row_a = os.path.join(FIGDIR_REDESIGN, "fig_validation_sk_conf.png")
    row_b = os.path.join(FIGDIR_REDESIGN, "fig_skzeta_conf_rowb.png")
    row_c = os.path.join(FIGDIR_REDESIGN, "fig_skzeta_conf_rowc.png")
    if not (os.path.isfile(row_b) and os.path.isfile(row_c)):
        row_b, row_c = _split_skzeta_strips()
    missing = [p for p in (row_a, row_b, row_c) if not p or not os.path.isfile(p)]
    if missing:
        print("  [SKIP assemble_fig2] missing panel strips:")
        for p in missing:
            print(f"    {p}")
        print("  Build row a with fig2_validation() and the S(k,zeta) panels "
              "with fig2_skzeta(), then re-run.")
        return
    base = os.path.join(FIGDIR_REDESIGN, "_fig2_base.png")
    W, H, y_tops, hts = _stack([row_a, row_b, row_c], base)
    letters = [
        (0.008, y_tops[0] - 0.004, "a"),
        (0.008, y_tops[1] - 0.004, "b"),
        (0.008, y_tops[2] - 0.004, "c"),
    ]
    _annotate(base, out_png, letters, boxes=None)


def fig2():
    """Regenerable pieces of the validation figure (row a + S(k,zeta) panels)."""
    fig2_validation(make_noise=True)
    fig2_skzeta(conf=True)
    assemble_fig2(os.path.join(FIGDIR_REDESIGN, "2_validation_conf.png"))


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3  —  generality (temperature + model)   ->  figR4_generality.png
# ═════════════════════════════════════════════════════════════════════════════
_GEN_LFTS_BAND = (0.775, 0.825)
_GEN_DNLS_BAND = (0.975, 1.025)


def _gen_sk_markers(ax):
    ax.axvspan(*_GEN_LFTS_BAND, color="#5b9bd5", alpha=0.22, lw=0, zorder=0)
    ax.axvspan(*_GEN_DNLS_BAND, color="#d9534f", alpha=0.22, lw=0, zorder=0)


def _gen_panel_letter(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=14, fontweight="bold",
            va="bottom", ha="right", color="#111111")


def _gen_discrete_temperature(temps):
    """One distinct colour per temperature + BoundaryNorm for a discrete-block
    colourbar."""
    base = temperature_cmap()
    ts = sorted(temps)
    n = len(ts)
    colors = [base(i / (n - 1)) for i in range(n)]
    cmap = ListedColormap(colors)
    step = ts[1] - ts[0]
    bounds = [ts[0] - step / 2] + [(ts[i] + ts[i + 1]) / 2 for i in range(n - 1)] \
             + [ts[-1] + step / 2]
    norm = BoundaryNorm(bounds, cmap.N)
    tcolor = {t: colors[i] for i, t in enumerate(ts)}
    return cmap, norm, bounds, tcolor


def fig3():
    set_nature_style()
    temps = [-30, -20, -10, 0, 10, 20, 30]
    caches = {t: load_cache(f"4p_T{t}") for t in temps}
    caches = {t: c for t, c in caches.items() if c is not None}
    temps = sorted(caches)
    dcmap, dnorm, dbounds, tcolor = _gen_discrete_temperature(temps)

    fig = plt.figure(figsize=(8.0, 6.2))
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.30,
                           left=0.09, right=0.90, bottom=0.09, top=0.90)

    # (a,b) S(k) vs T for each state — DISCRETE colour per temperature
    for col_i, (state, letter) in enumerate((("LFTS", "a"), ("DNLS", "b"))):
        ax = fig.add_subplot(gs[0, col_i])
        for t in temps:
            dd = caches[t]; kn = dd["kn"]; mm = (kn >= KN_RANGE[0]) & (kn <= KN_RANGE[1])
            ax.plot(kn[mm], dd[state]["S"][mm], color=tcolor[t], lw=1.6)
        _gen_sk_markers(ax)
        ax.set_xlim(*KN_RANGE); ax.set_ylim(0.55, 1.35)
        ax.set_xlabel(r"$k\,r_{\mathrm{OO}}/2\pi$")
        if col_i == 0:
            ax.set_ylabel(r"$S(k)$")
        ax.set_title(f"{state}  vs. temperature", fontsize=9)
        style_ax(ax); _gen_panel_letter(ax, letter)
    sm = ScalarMappable(norm=dnorm, cmap=dcmap); sm.set_array([])
    cax = fig.add_axes([0.915, 0.55, 0.020, 0.34])
    cb = fig.colorbar(sm, cax=cax, boundaries=dbounds, ticks=temps,
                      spacing="uniform", drawedges=True)
    cb.set_label(r"$T$ ($^{\circ}$C)", fontsize=8.5); cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.8)
    cb.dividers.set_color("white"); cb.dividers.set_linewidth(1.3)

    # (c) LFTS fraction and noise/transition removal vs T (block-trim aggregate)
    ax = fig.add_subplot(gs[1, 0])
    rep_csv = os.path.join(CLUST, "multirun_dbscan_gmm_realreps", "fractions_blocktrim.csv")
    agg = pd.read_csv(rep_csv).sort_values("temp")
    xs = agg["temp"].values.astype(float)
    fr = agg["lfts_mean"].values;  fr_e = np.nan_to_num(agg["lfts_std"].values)
    nfr = agg["noise_mean"].values; nfr_e = np.nan_to_num(agg["noise_std"].values)
    ax.axvspan(xs.min() - 4, 0, color="#dbe7f3", alpha=0.45, zorder=0)
    ax.errorbar(xs, nfr, yerr=nfr_e, color="#6b7280", lw=1.5, marker="s", ms=4.5,
                mfc="white", mec="#6b7280", mew=1.3, capsize=2.5, capthick=1.0,
                elinewidth=1.0, label="noise removed", zorder=3)
    ax.errorbar(xs, fr, yerr=fr_e, color=C["LFTS"], lw=1.7, marker="o", ms=5,
                mfc="white", mec=C["LFTS"], mew=1.3, capsize=2.5, capthick=1.0,
                elinewidth=1.0, label=r"LFTS fraction $s$", zorder=4)
    ax.set_xlim(xs.min() - 4, xs.max() + 4); ax.set_ylim(0.10, 0.52); ax.set_xticks(xs)
    ax.set_xlabel(r"$T$  ($^{\circ}$C)"); ax.set_ylabel("fraction")
    ax.text((xs.min() - 4) / 2, 0.115, "supercooled", color="#3a567a", fontsize=7.5,
            rotation=0, ha="center", va="bottom")
    ax.legend(loc="upper right", fontsize=7, handlelength=1.6, borderpad=0.3,
              labelspacing=0.3, framealpha=0.9,
              title="error bar", title_fontsize=6.5)
    ax.set_title("Population vs. temperature", fontsize=9)
    style_ax(ax); _gen_panel_letter(ax, "c")

    # (d) model comparison of per-state S(k)
    ax = fig.add_subplot(gs[1, 1])
    models = [("TIP4P/2005", "4p_T-20"), ("TIP5P", "5p_T-20"), ("SWM4-NDP", "swm_T-20")]
    for name, tag in models:
        dd = load_cache(tag)
        if dd is None:
            continue
        col = MODEL_COLOR[name]; kn = dd["kn"]; mm = (kn >= KN_RANGE[0]) & (kn <= KN_RANGE[1])
        ax.plot(kn[mm], dd["LFTS"]["S"][mm], color=col, lw=1.6)
        ax.plot(kn[mm], dd["DNLS"]["S"][mm], color=col, lw=1.3, ls=(0, (4, 2)))
    _gen_sk_markers(ax)
    ax.set_xlim(*KN_RANGE); ax.set_ylim(0.55, 1.4)
    ax.set_xlabel(r"$k\,r_{\mathrm{OO}}/2\pi$"); ax.set_ylabel(r"$S(k)$")
    hm = [Line2D([0], [0], color=MODEL_COLOR[n], lw=1.8) for n, _ in models]
    hs = [Line2D([0], [0], color="0.3", lw=1.6),
          Line2D([0], [0], color="0.3", lw=1.3, ls=(0, (4, 2)))]
    leg1 = ax.legend(hm, [n for n, _ in models], loc="upper right", fontsize=7,
                     title=r"model ($-20\,^{\circ}$C)", title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(hs, ["LFTS", "DNLS"], loc="lower right", fontsize=7, handlelength=1.8)
    ax.set_title("Across water models", fontsize=9)
    style_ax(ax); _gen_panel_letter(ax, "d")

    fig.suptitle("Two-state signature is generic across temperature and model",
                 fontsize=9.5, y=0.975)
    savefig(fig, os.path.join(FIGDIR_REDESIGN, "figR4_generality.png"))
    plt.close(fig)


_FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="*", default=["all"],
                    help="fig1 | fig2 | fig3 | all")
    args = ap.parse_args()
    which = list(_FIGS) if args.which == ["all"] or "all" in args.which else args.which
    for name in which:
        print(f"--- {name} ---")
        _FIGS[name]()
    print("done.")
