#!/usr/bin/env python3
"""SI Figure S6: group structure factors of three water models at -20 C, one frozen
classifier (fitted on TIP4P/2005 only). Shading: s.e.m. over the 20 frames.

    appendix_fig6_a   TIP4P/2005 (Run01)
    appendix_fig6_b   TIP5P (Run01)
    appendix_fig6_c   SWM4-NDP (Run02)
"""
import numpy as np

from style import C, save, set_style, single_axes, structure_factor, style_ax

if __name__ == "__main__":
    set_style()
    for letter, model in zip("abc", ("TIP4P-2005", "TIP5P", "SWM4-NDP")):
        sk = structure_factor(model, -20)
        fig, ax = single_axes(3.03, 2.28)
        for group in ("LFTS", "DNLS"):
            frames = sk[f"S_{group}"]
            mean = frames.mean(axis=0)
            sem = frames.std(axis=0, ddof=1) / np.sqrt(len(frames))
            ax.plot(sk["k_reduced"], mean, color=C[group], lw=1.7, label=group)
            ax.fill_between(sk["k_reduced"], mean - sem, mean + sem, color=C[group],
                            alpha=0.14, lw=0)
        for band in ((0.775, 0.825), (0.975, 1.025)):
            ax.axvspan(*band, color="#b9c0c8", alpha=0.3, lw=0, zorder=0)
        ax.set_xlim(0.6, 2.0)
        ax.set_ylim(0.5, 1.45)
        ax.set_xlabel(r"$k\,r_{\mathrm{OO}}/2\pi$")
        ax.set_ylabel(r"$S_g(k)$")
        if letter == "a":
            ax.legend(loc="upper left", fontsize=8)
        style_ax(ax)
        save(fig, f"appendix_fig6_{letter}")
