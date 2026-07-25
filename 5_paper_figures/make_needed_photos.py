#!/usr/bin/env python3
"""
make_needed_photos.py
=====================
Regenerate the six paper figures collected in
``6_paper_writing/Paper_WaterMLClustering/images/needed_photo`` from the
UPDATED density-valley-cleavage pipeline (run_dbscan_gmm_density), applying the
final clean-up requested for the manuscript:

  * every narrative heading / suptitle / method-label removed (only concise
    colour-coded state identifiers kept where a panel would otherwise be
    ambiguous);
  * the S(k, zeta) heatmap AND its 3-D surface on a single uniform 0-1.8 scale;
  * the S(k) region bands fixed at 0.775-0.825 (LFTS) and 0.975-1.025 (DNLS);
  * one consistent Nature-style font / sizing / colour system across all six.

Figures (all reuse committed caches / label CSVs -- no trajectory recompute):

  1_clustering_b_density.png   LFTS/DNLS (zeta,q) KDE + transition + marginals
  1_clustering_c_density.png   order-parameter histograms (q, LSI, S_k, zeta)
  1_clustering_d_density.png   (q, zeta) classification scatter
  2_validation_density_max20.png  S(k,zeta) 3-D surfaces + 2-D heatmaps (0-1.8)
  2_validation_line_3state.png    per-state S(k) line plot + region bands
  3_generality.png             S(k) vs T, population(T), model comparison

Usage:  python make_needed_photos.py [name ...]   (default: all)
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, _HERE)

from nature_style import (set_nature_style, C, CMAP_DENSITY, MODEL_COLOR,      # noqa: E402
                          temperature_cmap, style_ax)
from figlib import load_flat, _joint_axes, load_cache, k_norm, KN_RANGE        # noqa: E402

set_nature_style()

OUT = os.environ.get("NEEDED_PHOTO_OUT") or os.path.join(
    _ROOT, "6_paper_writing", "Paper_WaterMLClustering", "images", "needed_photo")
RC = os.path.join(_HERE, "redesign_cache")
CLUST = os.path.join(_ROOT, "results", "clustering")
os.makedirs(OUT, exist_ok=True)

# ── shared conventions ───────────────────────────────────────────────────────
R_OO = 0.285
S_HEAT = (0.0, 1.8)                       # uniform S(k,zeta) surface + heatmap range
LFTS_BAND = (0.775, 0.825)                # requested S(k) region marks
DNLS_BAND = (0.975, 1.025)
C_LFTS_BAND, C_DNLS_BAND = "#5b9bd5", "#d9534f"   # light shades for region fills
STATE_FS = 10                             # font size for the one-word state labels
LETTER_FS = 14                            # panel-letter font size


def _state_label(ax, name, color, x=0.5, y=1.015, ha="center"):
    """Concise colour-coded state identifier (NOT a heading)."""
    ax.text(x, y, name, transform=ax.transAxes, color=color, fontsize=STATE_FS,
            fontweight="bold", ha=ha, va="bottom")


def _panel_letter(ax, letter, x=-0.16, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=LETTER_FS,
            fontweight="bold", va="bottom", ha="right", color="#111111")


def _save(fig, name, dpi=600):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — clustering panels (density-valley cleavage labels)
# ═════════════════════════════════════════════════════════════════════════════
_METHOD = "lr"          # likelihood-ratio labels (results/clustering/tip4p2005_T-20_lr)
OPS = ["q_all", "LSI_all", "Sk_all", "zeta_all"]
OP_LABEL = {"q_all": r"$q$  (tetrahedral order)", "LSI_all": r"LSI  (nm$^2$)",
            "Sk_all": r"$S_k$", "zeta_all": r"$\zeta$  (nm)"}
LO_PCT = {"Sk_all": 4.0}


def _fig1_load():
    df, lab, lfts = load_flat("tip4p2005_T-20", method=_METHOD)
    dnls = [c for c in np.unique(df[lab]) if c >= 0 and c != lfts][0]
    x = df["zeta_all"].values
    y = df["q_all"].values
    good = np.isfinite(x) & np.isfinite(y)
    xlo, xhi = np.percentile(x[good], 0.5), np.percentile(x[good], 99.5)
    ylo, yhi = np.percentile(y[good], 1.0), np.percentile(y[good], 99.8)
    return df, lab, lfts, dnls, good, (xlo, xhi, ylo, yhi)


def fig1_b():
    """Resolved LFTS / DNLS (KDE) + transition underlay + marginals. No heading."""
    df, lab, lfts, dnls, good, (xlo, xhi, ylo, yhi) = _fig1_load()
    x, y, L = df["zeta_all"].values[good], df["q_all"].values[good], df[lab].values[good]

    fig = plt.figure(figsize=(4.0, 3.6))
    gs = gridspec.GridSpec(1, 1, left=0.15, right=0.97, bottom=0.14, top=0.94)
    ax, axt, axr = _joint_axes(fig, gs[0, 0])
    mn = L < 0
    ax.scatter(x[mn], y[mn], s=1.5, c=C["NOISE"], alpha=0.30, lw=0,
               rasterized=True, zorder=0)
    axt.hist(x[mn], bins=110, range=(xlo, xhi), color=C["NOISE"], alpha=0.5,
             lw=0, density=True)
    axr.hist(y[mn], bins=110, range=(ylo, yhi), color=C["NOISE"], alpha=0.5,
             lw=0, density=True, orientation="horizontal")
    for cid, name, col in ((lfts, "LFTS", C["LFTS"]), (dnls, "DNLS", C["DNLS"])):
        m = L == cid
        xs, ys = x[m], y[m]
        sub = np.random.default_rng(0).choice(len(xs), size=min(6000, len(xs)),
                                               replace=False)
        kde = gaussian_kde(np.vstack([xs[sub], ys[sub]]))
        gx, gy = np.mgrid[xlo:xhi:120j, ylo:yhi:120j]
        zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        levels = np.linspace(zz.max() * 0.08, zz.max(), 6)
        ax.contourf(gx, gy, zz, levels=levels, colors=col, alpha=0.16)
        ax.contour(gx, gy, zz, levels=levels, colors=col, linewidths=0.7, alpha=0.9)
        axt.hist(xs, bins=110, range=(xlo, xhi), color=col, alpha=0.55, lw=0, density=True)
        axr.hist(ys, bins=110, range=(ylo, yhi), color=col, alpha=0.55, lw=0,
                 density=True, orientation="horizontal")
        ax.plot([], [], color=col, lw=4, alpha=0.6, label=name)
    ax.scatter([], [], s=10, c=C["NOISE"], label="transition")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$\zeta$  (translational order)")
    ax.set_ylabel(r"$q$  (tetrahedral order)")
    style_ax(ax)
    ax.legend(loc="lower right", fontsize=8, handlelength=1.1, borderpad=0.3)
    _save(fig, "1_clustering_b_density.png")


def fig1_c():
    """Order-parameter histograms (q, LSI, S_k, zeta). No heading."""
    df, lab, lfts, dnls, *_ = _fig1_load()
    groups = [(lfts, "LFTS", C["LFTS"], 0.32), (dnls, "DNLS", C["DNLS"], 0.32),
              (-1, "transition", C["NOISE"], 0.18)]
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.7))
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.11, top=0.96,
                        wspace=0.34, hspace=0.42)
    for ax, op in zip(axes.ravel(), OPS):
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
        style_ax(ax)
    leg_h = [Patch(facecolor=col, edgecolor=col, alpha=alpha, linewidth=1.4)
             for _, _, col, alpha in groups]
    leg_l = [name for _, name, _, _ in groups]
    axes.ravel()[0].legend(leg_h, leg_l, loc="upper left", fontsize=8,
                           handlelength=1.3, handleheight=1.0, labelspacing=0.3,
                           borderpad=0.35, framealpha=0.9)
    _save(fig, "1_clustering_c_density.png")


def fig1_d():
    """(q, zeta) classification scatter. No heading."""
    df, lab, lfts, dnls, *_ = _fig1_load()
    fig, axs = plt.subplots(figsize=(4.3, 4.8))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.11, top=0.96)
    x, y = df["q_all"].values, df["zeta_all"].values
    mn = df[lab].values < 0
    axs.scatter(x[mn], y[mn], s=3.4, c=C["NOISE"], alpha=0.6, lw=0,
                rasterized=True, label="transition")
    for cid, name, col in ((dnls, "DNLS", C["DNLS"]), (lfts, "LFTS", C["LFTS"])):
        m = df[lab].values == cid
        axs.scatter(x[m], y[m], s=4.0, c=col, alpha=0.45, lw=0,
                    rasterized=True, label=name)
    axs.set_xlabel(r"$q$  (tetrahedral order)"); axs.set_ylabel(r"$\zeta$  (nm)")
    axs.set_xlim(np.percentile(x, 0.3), np.percentile(x, 99.7))
    axs.set_ylim(np.percentile(y, 0.5), np.percentile(y, 99.9))
    leg = axs.legend(loc="upper left", fontsize=8, markerscale=3,
                     handletextpad=0.3, borderpad=0.3)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    style_ax(axs)
    _save(fig, "1_clustering_d_density.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2a — S(k, zeta) 2-D heatmaps (clean, uniform 0-1.8) built from the NEW
#             likelihood-ratio pipeline (make_ver2_skzeta3d), matching the
#             approved images/main/2_validation_ver2_heatmap.png look.
# ═════════════════════════════════════════════════════════════════════════════
_SKZ_CACHE = os.path.join(RC, "skzeta_ver2_needed.npz")
_K_T1, _K_D1 = 0.75, 1.00


def _skz_ver2_surfaces(recompute=False):
    """(kp, zc, S) per state from the ver2 likelihood-ratio S(k,zeta) pipeline.
    Cached because it loads the trajectory and reruns compute_sk_zeta_matrix."""
    if not recompute and os.path.isfile(_SKZ_CACHE):
        z = np.load(_SKZ_CACHE, allow_pickle=True)
        return {n: (z[f"kp_{n}"], z[f"zc_{n}"], z[f"S_{n}"]) for n in ("LFTS", "DNLS")}
    sys.path.insert(0, os.path.join(_ROOT, "4_structure_factor"))
    from make_ver2_skzeta3d import (new_labels, process, K, R_OO as _R,          # noqa: E402
                                     ZETA_BINS, RC as _RC, N_FRAMES, DCD, PDB, ZETA_MAT)
    from structure_factor_bycluster import load_trajectory                       # noqa: E402
    from sk_zeta_3d import compute_sk_zeta_matrix, _load_zeta                     # noqa: E402
    lm = new_labels()
    traj = load_trajectory(DCD, PDB)[:N_FRAMES]
    zeta_all = _load_zeta(ZETA_MAT)[:N_FRAMES]
    k_norm = K * _R / (2 * np.pi)
    out = {}
    for cid, name in ((0, "LFTS"), (1, "DNLS")):
        S_kz, zc = compute_sk_zeta_matrix(traj, K, lm, cid, zeta_all, ZETA_BINS, _RC)
        out[name] = process(S_kz, k_norm, zc)
    np.savez(_SKZ_CACHE,
             **{f"kp_{n}": out[n][0] for n in out},
             **{f"zc_{n}": out[n][1] for n in out},
             **{f"S_{n}": out[n][2] for n in out})
    return out


def fig2_skzeta():
    surf = _skz_ver2_surfaces()
    cmap = plt.get_cmap("jet"); norm = Normalize(*S_HEAT)
    fill_levels = np.linspace(*S_HEAT, 37)
    line_levels = np.linspace(*S_HEAT, 10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (name, col) in zip(axes, (("LFTS", C["LFTS"]), ("DNLS", C["DNLS"]))):
        kp, zc, S = surf[name]
        ax.contourf(kp, zc, S, levels=fill_levels, cmap=cmap, norm=norm, extend="both")
        ax.contour(kp, zc, S, levels=line_levels, colors="k", linewidths=0.4, alpha=0.35)
        ax.axvline(_K_T1, color="#12347a", ls="--", lw=1.8, label=r"$k_{T1}$ (FSDP)")
        ax.axvline(_K_D1, color="#7a1f17", ls="--", lw=1.8, label=r"$k_{D1}$ (DNLS)")
        ax.set_xlim(float(kp[0]), float(kp[-1])); ax.set_ylim(float(zc[0]), float(zc[-1]))
        ax.set_xlabel(r"$k\,r_{OO}/2\pi$"); ax.set_ylabel(r"$\zeta$ (Å)")
        ax.set_title(name, color=col, fontsize=STATE_FS, fontweight="bold")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        style_ax(ax)
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.tolist(), shrink=0.9, aspect=22,
                      label=r"$S(k,\zeta)$", pad=0.02,
                      ticks=np.arange(0.0, S_HEAT[1] + 1e-9, 0.3))
    cb.outline.set_linewidth(0.8); cb.ax.tick_params(labelsize=8)
    _save(fig, "2_validation_density_max20.png", dpi=200)


def fig2_skzeta_3d():
    """3-D S(k, zeta) surfaces (upper part of the old validation figure), from
    the same likelihood-ratio surfaces as the heatmaps, uniform 0-1.8 scale."""
    surf = _skz_ver2_surfaces()
    z_max = 2.0
    cmap = plt.get_cmap("jet"); norm = Normalize(*S_HEAT)
    fig = plt.figure(figsize=(13, 5.6))
    for i, (name, col) in enumerate((("LFTS", C["LFTS"]), ("DNLS", C["DNLS"]))):
        kp, zc, S = surf[name]
        Sc = np.clip(S, 0.0, z_max)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        K2, Z2 = np.meshgrid(kp, zc)
        ax.plot_surface(K2, Z2, Sc, facecolors=cmap(norm(Sc)), rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
        ax.set_xlabel(r"$k\,r_{OO}/2\pi$", labelpad=8)
        ax.set_ylabel(r"$\zeta$ (Å)", labelpad=8)
        ax.set_zlabel(r"$S(k,\zeta)$", labelpad=4)
        ax.set_xlim(float(kp[0]), float(kp[-1])); ax.set_ylim(float(zc[0]), float(zc[-1]))
        ax.set_zlim(0.0, z_max); ax.view_init(elev=28, azim=-115)
        ax.set_title(name, color=col, fontsize=STATE_FS, fontweight="bold")
        print(f"  {name}: max S(k,zeta) = {np.nanmax(S):.3f}")
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.55, aspect=16,
                      label=r"$S(k,\zeta)$", pad=0.02,
                      ticks=np.arange(0.0, S_HEAT[1] + 1e-9, 0.3))
    cb.outline.set_linewidth(0.8); cb.ax.tick_params(labelsize=8)
    # mplot3d axis labels fall outside the tight bbox; needs a wider pad than _save's
    path = os.path.join(OUT, "2_validation_3d.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"  wrote {path}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2b — per-state S(k) line plot (LFTS / DNLS / noise) + region bands
# ═════════════════════════════════════════════════════════════════════════════
_LINE_CACHE = os.path.join(RC, "validation_line_3state_lr.npz")
_LINE_XR = (0.6, 2.0)
_C_LINE = {"LFTS": "#5b9bd5", "DNLS": "#d9534f", "noise": "#7a7a7a"}


def fig2_line():
    z = np.load(_LINE_CACHE, allow_pickle=True)
    k = z["k"]; D = z["D"].item()
    m = (k >= _LINE_XR[0]) & (k <= _LINE_XR[1])
    fig, a = plt.subplots(figsize=(12.5, 4.8))
    series = [("LFTS", "LFTS", _C_LINE["LFTS"], "o"),
              ("DNLS", "DNLS", _C_LINE["DNLS"], "D"),
              ("noise only", "noise", _C_LINE["noise"], "s")]
    for label, key, col, mk in series:
        S, sem = D[key]["S"], D[key]["sem"]; ev = slice(0, None, 8)
        a.plot(k[m], S[m], color=col, lw=1.4, zorder=3)
        a.errorbar(k[m][ev], S[m][ev], yerr=sem[m][ev], fmt=mk, ms=4, mfc="white",
                   mec=col, mew=1, color=col, ecolor=col, elinewidth=0.8,
                   capsize=1.8, label=label, zorder=4)
    a.axvspan(*LFTS_BAND, color=C_LFTS_BAND, alpha=0.22, lw=0)
    a.axvspan(*DNLS_BAND, color=C_DNLS_BAND, alpha=0.22, lw=0)
    a.grid(True, alpha=0.25, ls="--", lw=0.6)
    a.tick_params(direction="in", top=True, right=True)
    a.set_xlim(*_LINE_XR); a.set_ylim(0.5, 1.6)
    a.set_box_aspect(1.0 / 3.0)
    a.set_xlabel(r"$k \cdot r_{OO}/2\pi$"); a.set_ylabel(r"$S(k)$")
    h, l = a.get_legend_handles_labels()
    h += [Patch(color=C_LFTS_BAND, alpha=0.22), Patch(color=C_DNLS_BAND, alpha=0.22)]
    l += ["LFTS region", "DNLS region"]
    a.legend(h, l, loc="upper right", fontsize=8, ncol=2)
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.14, top=0.98)
    _save(fig, "2_validation_line_3state.png")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — generality (temperature + model). Panel letters kept, headings cut.
# ═════════════════════════════════════════════════════════════════════════════
def _gen_bands(ax):
    ax.axvspan(*LFTS_BAND, color=C_LFTS_BAND, alpha=0.22, lw=0, zorder=0)
    ax.axvspan(*DNLS_BAND, color=C_DNLS_BAND, alpha=0.22, lw=0, zorder=0)


def _gen_discrete_temperature(temps):
    base = temperature_cmap()
    ts = sorted(temps); n = len(ts)
    colors = [base(i / (n - 1)) for i in range(n)]
    cmap = ListedColormap(colors)
    step = ts[1] - ts[0]
    bounds = [ts[0] - step / 2] + [(ts[i] + ts[i + 1]) / 2 for i in range(n - 1)] \
             + [ts[-1] + step / 2]
    return cmap, BoundaryNorm(bounds, cmap.N), bounds, {t: colors[i] for i, t in enumerate(ts)}


def fig3():
    temps = [-30, -20, -10, 0, 10, 20, 30]
    caches = {t: load_cache(f"4p_T{t}_lr") for t in temps}
    caches = {t: c for t, c in caches.items() if c is not None}
    temps = sorted(caches)
    dcmap, dnorm, dbounds, tcolor = _gen_discrete_temperature(temps)

    fig = plt.figure(figsize=(8.0, 6.2))
    gs = gridspec.GridSpec(2, 2, hspace=0.34, wspace=0.30,
                           left=0.09, right=0.90, bottom=0.09, top=0.94)

    # (a,b) S(k) vs T per state
    for col_i, (state, letter, col) in enumerate(
            (("LFTS", "a", C["LFTS"]), ("DNLS", "b", C["DNLS"]))):
        ax = fig.add_subplot(gs[0, col_i])
        for t in temps:
            dd = caches[t]; kn = dd["kn"]; mm = (kn >= KN_RANGE[0]) & (kn <= KN_RANGE[1])
            ax.plot(kn[mm], dd[state]["S"][mm], color=tcolor[t], lw=1.6)
        _gen_bands(ax)
        ax.set_xlim(*KN_RANGE); ax.set_ylim(0.55, 1.35)
        ax.set_xlabel(r"$k\,r_{OO}/2\pi$")
        if col_i == 0:
            ax.set_ylabel(r"$S(k)$")
        style_ax(ax); _panel_letter(ax, letter); _state_label(ax, state, col)
    sm = ScalarMappable(norm=dnorm, cmap=dcmap); sm.set_array([])
    cax = fig.add_axes([0.915, 0.56, 0.020, 0.33])
    cb = fig.colorbar(sm, cax=cax, boundaries=dbounds, ticks=temps,
                      spacing="uniform", drawedges=True)
    cb.set_label(r"$T$ ($^{\circ}$C)", fontsize=8.5); cb.ax.tick_params(labelsize=7)
    cb.outline.set_linewidth(0.8)
    cb.dividers.set_color("white"); cb.dividers.set_linewidth(1.3)

    # (c) constant clustering quality: noise removed rises as LFTS core falls vs T
    ax = fig.add_subplot(gs[1, 0])
    agg = pd.read_csv(os.path.join(CLUST, "multirun_constsil",
                                   "fractions_constsil_lr.csv")).sort_values("temp")
    xs = agg["temp"].values.astype(float)
    fr, fr_e = agg["lfts_mean"].values, np.nan_to_num(agg["lfts_std"].values)
    nfr, nfr_e = agg["noise_mean"].values, np.nan_to_num(agg["noise_std"].values)
    ax.axvspan(xs.min() - 4, 0, color="#dbe7f3", alpha=0.45, zorder=0)
    ax.errorbar(xs, nfr, yerr=nfr_e, color="#6b7280", lw=1.5, marker="s", ms=4.5,
                mfc="white", mec="#6b7280", mew=1.3, capsize=2.5, capthick=1.0,
                elinewidth=1.0, label="noise removed", zorder=3)
    ax.errorbar(xs, fr, yerr=fr_e, color=C["LFTS"], lw=1.7, marker="o", ms=5,
                mfc="white", mec=C["LFTS"], mew=1.3, capsize=2.5, capthick=1.0,
                elinewidth=1.0, label="LFTS fraction", zorder=4)
    ax.set_xlim(xs.min() - 4, xs.max() + 4); ax.set_ylim(0.0, 0.5); ax.set_xticks(xs)
    ax.set_xlabel(r"$T$  ($^{\circ}$C)"); ax.set_ylabel("fraction of molecules")
    ax.text((xs.min() - 4) / 2, 0.22, "supercooled", color="#3a567a", fontsize=7.5,
            ha="center", va="bottom")
    ax.legend(loc="upper right", fontsize=7, handlelength=1.6, borderpad=0.3,
              labelspacing=0.3, framealpha=0.9,
              title=r"constant silhouette $s^{*}$", title_fontsize=6.5)
    style_ax(ax); _panel_letter(ax, "c")

    # (d) model comparison
    ax = fig.add_subplot(gs[1, 1])
    models = [("TIP4P/2005", "4p_T-20_lr"), ("TIP5P", "5p_T-20_lr"), ("SWM4-NDP", "swm_T-20_lr")]
    for name, tag in models:
        dd = load_cache(tag)
        if dd is None:
            continue
        col = MODEL_COLOR[name]; kn = dd["kn"]; mm = (kn >= KN_RANGE[0]) & (kn <= KN_RANGE[1])
        ax.plot(kn[mm], dd["LFTS"]["S"][mm], color=col, lw=1.6)
        ax.plot(kn[mm], dd["DNLS"]["S"][mm], color=col, lw=1.3, ls=(0, (4, 2)))
    _gen_bands(ax)
    ax.set_xlim(*KN_RANGE); ax.set_ylim(0.55, 1.4)
    ax.set_xlabel(r"$k\,r_{OO}/2\pi$"); ax.set_ylabel(r"$S(k)$")
    hm = [Line2D([0], [0], color=MODEL_COLOR[n], lw=1.8) for n, _ in models]
    hs = [Line2D([0], [0], color="0.3", lw=1.6),
          Line2D([0], [0], color="0.3", lw=1.3, ls=(0, (4, 2)))]
    leg1 = ax.legend(hm, [n for n, _ in models], loc="upper right", fontsize=7,
                     title=r"model ($-20\,^{\circ}$C)", title_fontsize=7)
    ax.add_artist(leg1)
    ax.legend(hs, ["LFTS", "DNLS"], loc="lower right", fontsize=7, handlelength=1.8)
    style_ax(ax); _panel_letter(ax, "d")

    _save(fig, "3_generality.png")


_FIGS = {
    "1_clustering_b_density": fig1_b,
    "1_clustering_c_density": fig1_c,
    "1_clustering_d_density": fig1_d,
    "2_validation_density_max20": fig2_skzeta,
    "2_validation_3d": fig2_skzeta_3d,
    "2_validation_line_3state": fig2_line,
    "3_generality": fig3,
}


if __name__ == "__main__":
    which = sys.argv[1:] or list(_FIGS)
    for name in which:
        print(f"--- {name} ---")
        _FIGS[name]()
    print("done.")
