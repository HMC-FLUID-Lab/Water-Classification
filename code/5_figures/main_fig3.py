#!/usr/bin/env python3
"""Main-text Figure 3: structure factors of the classified groups, TIP4P/2005 at -20 C.

    main_fig3_a        S_g(k) of the LFTS, DNLS and transitional groups (+/- s.e.m. over frames)
    main_fig3_b_LFTS   S_H(k, zeta) surface, LFTS group
    main_fig3_b_DNLS   S_L(k, zeta) surface, DNLS group
    main_fig3_c_LFTS   S(k, zeta) contour map, LFTS group
    main_fig3_c_DNLS   S(k, zeta) contour map, DNLS group
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from style import C, DNLS_BAND, LFTS_BAND, save, set_style, structure_factor, style_ax
import config  # noqa: E402  (style puts code/ on the path)

LINE_RC = {"font.size": 14, "font.weight": "bold", "axes.labelsize": 15,
           "axes.labelweight": "bold", "xtick.labelsize": 13.5, "ytick.labelsize": 13.5,
           "legend.fontsize": 11.5, "axes.linewidth": 1.0}
SURFACE_RC = {k: 0.85 * v for k, v in {"font.size": 13.5, "axes.labelsize": 15,
                                      "axes.titlesize": 17, "xtick.labelsize": 12.5,
                                      "ytick.labelsize": 12.5, "legend.fontsize": 11.5}.items()}


def panel_a():
    sk = structure_factor(*config.REFERENCE[:2])
    k = sk["k_reduced"]
    m = (k >= 0.6) & (k <= 2.0)
    with plt.rc_context(LINE_RC):
        fig, a = plt.subplots(figsize=(12.5, 4.8))
        for label, group, col, mk in (("LFTS", "LFTS", C["LFTS"], "o"),
                                      ("DNLS", "DNLS", C["DNLS"], "D"),
                                      ("transition", "transitional", C["NOISE"], "s")):
            frames = sk[f"S_{group}"]
            S, sem = frames.mean(axis=0), frames.std(axis=0) / np.sqrt(len(frames))
            ev = slice(0, None, 8)
            a.plot(k[m], S[m], color=col, lw=2.6, zorder=3)
            a.errorbar(k[m][ev], S[m][ev], yerr=sem[m][ev], fmt=mk, ms=7, mfc="white",
                       mec=col, mew=1.7, color=col, ecolor=col, elinewidth=1.3,
                       capsize=2.8, capthick=1.3, label=label, zorder=4)
        a.axvspan(*LFTS_BAND, color=C["LFTS_l"], alpha=0.30, lw=0)
        a.axvspan(*DNLS_BAND, color=C["DNLS_l"], alpha=0.30, lw=0)
        a.grid(True, alpha=0.25, ls="--", lw=0.9)
        a.tick_params(direction="in", top=True, right=True, width=1.0, length=5.5, pad=5)
        a.set_xlim(0.6, 2.0); a.set_ylim(0.5, 1.6)
        a.set_box_aspect(1.0 / 3.0)
        a.set_xlabel(r"$k \cdot r_{OO}/2\pi$"); a.set_ylabel(r"$S_g(k)$")
        h, l = a.get_legend_handles_labels()
        h += [Patch(color=C["LFTS_l"], alpha=0.30), Patch(color=C["DNLS_l"], alpha=0.30)]
        l += ["LFTS region", "DNLS region"]
        a.legend(h, l, loc="upper right", ncol=2, handlelength=1.8, columnspacing=1.1,
                 labelspacing=0.35, borderpad=0.5, framealpha=0.92)
        fig.subplots_adjust(left=0.06, right=0.995, bottom=0.16, top=0.98)
        save(fig, "main_fig3_a")


def _export_3d(fig, name):
    extra = [lab for ax in fig.axes if hasattr(ax, "zaxis")
             for lab in (ax.xaxis.label, ax.yaxis.label, ax.zaxis.label)]
    save(fig, name, pad_inches=0.08, bbox_extra_artists=extra or None)


def panels_b_c():
    model, temp, run = config.REFERENCE
    path = os.path.join(config.RESULTS, "structure_factors",
                        config.tag(model, temp, run) + "_Sk_zeta.npz")
    data = dict(np.load(path))
    norm = Normalize(0, 1.8, clip=True)                 # common scale, saturates above 1.8
    cmap = matplotlib.colormaps["coolwarm"]
    tick_fs = SURFACE_RC["xtick.labelsize"]
    cb_ticks = np.arange(0, 1.81, 0.2)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    with plt.rc_context(SURFACE_RC):
        for state, symbol in (("LFTS", "H"), ("DNLS", "L")):
            k, z, s = data[f"kp_{state}"], data[f"zc_{state}"], data[f"S_{state}"]
            xlabel, ylabel = r"$k\,r_{\mathrm{OO}}/2\pi$", r"$\zeta$ ($\mathrm{\AA}$)"
            slabel = rf"$S_{symbol}(k,\zeta)$"

            fig = plt.figure(figsize=(6.2, 4.8))                     # panel b: surface
            ax = fig.add_subplot(111, projection="3d")
            kk, zz = np.meshgrid(k, z)
            ax.plot_surface(kk, zz, s, cmap=cmap, norm=norm, rstride=1, cstride=1,
                            linewidth=0, antialiased=False, shade=False)
            ax.view_init(elev=28, azim=-115)
            ax.set(xlim=(0.6, 2), ylim=(z.min(), z.max()), zlim=(0, 2),
                   xlabel=xlabel, ylabel=ylabel)
            ax.tick_params(labelsize=tick_fs)
            ax.zaxis.set_rotate_label(False)
            ax.set_zlabel(slabel, labelpad=6, rotation=90)
            ax.xaxis.labelpad = 4
            ax.yaxis.labelpad = 4
            y_ticks = np.arange(0.4, 1.61, 0.2) if state == "LFTS" else np.arange(-0.8, 1.21, 0.2)
            y_shown = (0.6, 1.0, 1.4) if state == "LFTS" else (-0.6, 0.2, 1.0)
            for ticks, shown, setter, label_setter in (
                    (np.arange(0.6, 2.01, 0.2), (0.8, 1.2, 1.6, 2.0), ax.set_xticks, ax.set_xticklabels),
                    (y_ticks, y_shown, ax.set_yticks, ax.set_yticklabels),
                    (np.arange(0, 2.01, 0.25), (0, 0.5, 1, 1.5, 2), ax.set_zticks, ax.set_zticklabels)):
                setter(ticks)
                label_setter([f"{t:.1f}" if np.any(np.isclose(t, shown)) else "" for t in ticks])
            ax.set(xlim=(0.6, 2), ylim=(z.min(), z.max()), zlim=(0, 2))
            fig.subplots_adjust(left=0.13, right=0.90, bottom=0.12, top=0.95)
            cb = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.13, ticks=cb_ticks)
            cb.ax.tick_params(labelsize=tick_fs)
            cb.set_label(r"$S(k,\zeta)$", labelpad=2)
            cb.outline.set_linewidth(0.8)
            _export_3d(fig, f"main_fig3_b_{state}")

            fig, ax = plt.subplots(figsize=(5.2, 4.3))                 # panel c: contours
            ax.contourf(k, z, s, levels=np.linspace(0, 1.8, 61), cmap=cmap, norm=norm,
                        extend="max")
            ax.contour(k, z, s, levels=np.linspace(0, 1.8, 19), colors="black",
                       linewidths=0.35, alpha=0.3)
            ax.set(xlim=(0.6, 2), ylim=(z.min(), z.max()), xlabel=xlabel, ylabel=ylabel)
            ax.xaxis.labelpad = 2
            ax.yaxis.labelpad = 2
            ax.set_xticks(np.arange(0.8, 1.81, 0.2))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            style_ax(ax)
            cb = fig.colorbar(mappable, ax=ax, ticks=cb_ticks)
            cb.ax.tick_params(labelsize=tick_fs)
            cb.set_label(r"$S(k,\zeta)$", labelpad=2)
            cb.outline.set_linewidth(0.8)
            _export_3d(fig, f"main_fig3_c_{state}")


if __name__ == "__main__":
    set_style()
    panel_a()
    panels_b_c()
