#!/usr/bin/env python3
"""Main-text Figure 4: temperature dependence, TIP4P/2005, frozen classifier (tau = 2).

    main_fig4_a   LFTS / DNLS / transitional fractions of all molecules vs T
                  (mean over Run01-Run03, first 100 frames each)
    main_fig4_b   S_H(k) of the LFTS group, -30 to +90 C
    main_fig4_c   S_L(k) of the DNLS group, -30 to +90 C
"""
import os

import pandas as pd
from matplotlib.lines import Line2D

from style import C, save, set_style, single_axes, structure_factor, style_ax, temperature_colors
import config  # noqa: E402

AX_W, AX_H = 2.74, 2.34          # inches, axes size of each panel


def bands(ax):
    ax.axvspan(0.775, 0.825, color="#5b9bd5", alpha=0.22, lw=0, zorder=0)
    ax.axvspan(0.975, 1.025, color="#d9534f", alpha=0.22, lw=0, zorder=0)


def temperature_legend(ax, temps, colors):
    handles = [Line2D([], [], color=colors[t], lw=2.3) for t in temps]
    ax.legend(handles, [f"{t:+d}" if t else "0" for t in temps], title=r"$T$ ($^{\circ}$C)",
              title_fontsize=8, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              handlelength=1.5, labelspacing=0.4)


def panel_a():
    temps = config.TEMPS
    fr = pd.read_csv(os.path.join(config.RESULTS, "temperature_series", "fractions_run_mean.csv"))
    assert list(fr["temp"]) == temps and (fr["n_runs"] == 3).all()
    fig, ax = single_axes(AX_W, AX_H)
    ax.axvspan(-34, 0, color="#dbe7f3", alpha=0.45, lw=0)
    for label, key, color, marker in (("DNLS", "dnls", C["DNLS"], "D"),
                                      ("LFTS", "lfts", C["LFTS"], "o"),
                                      ("transitional", "trans", "#6b7280", "s")):
        ax.plot(temps, fr[f"{key}_mean"], color=color, marker=marker, ms=4.5, mfc="white",
                mew=1.1, lw=1.5, label=label)
    ax.set_xlim(-34, 94)
    ax.set_ylim(0, 0.975)
    ax.set_xticks(temps)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel(r"$T$ ($^{\circ}$C)")
    ax.set_ylabel("fraction of molecules")
    ax.text(-17, 0.955, "supercooled", color="#3a567a", fontsize=8, ha="center", va="top")
    ax.legend(loc="center right", fontsize=8)
    style_ax(ax)
    save(fig, "main_fig4_a")


def panel_sk(group, ylabel, name, legend):
    temps = config.TEMPS
    colors = temperature_colors(temps)
    fig, ax = single_axes(AX_W, AX_H)
    for t in temps:
        sk = structure_factor("TIP4P-2005", t)
        ax.plot(sk["k_reduced"], sk[f"S_{group}"].mean(axis=0), color=colors[t], lw=1.5)
    bands(ax)
    ax.set_xlim(0.6, 2.2)
    ax.set_ylim(0.55, 1.35)
    ax.set_xlabel(r"$k\,r_{\mathrm{OO}}/2\pi$")
    ax.set_ylabel(ylabel)
    if legend:
        temperature_legend(ax, temps, colors)
    style_ax(ax)
    save(fig, name)


if __name__ == "__main__":
    set_style()
    panel_a()
    panel_sk("LFTS", r"$S_H(k)$", "main_fig4_b", legend=False)
    panel_sk("DNLS", r"$S_L(k)$", "main_fig4_c", legend=True)
