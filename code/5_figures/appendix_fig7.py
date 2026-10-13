#!/usr/bin/env python3
"""SI Figure S7: temperature-dependent group structure factors of SWM4-NDP and TIP5P,
-30 to +90 C, frozen TIP4P/2005 classifier (first 20 frames of Run02 / Run01).

    appendix_fig7_a   SWM4-NDP, LFTS group
    appendix_fig7_b   SWM4-NDP, DNLS group
    appendix_fig7_c   TIP5P, LFTS group
    appendix_fig7_d   TIP5P, DNLS group
"""
from main_fig4 import AX_H, AX_W, bands, temperature_legend
from style import save, set_style, single_axes, structure_factor, style_ax, temperature_colors
import config  # noqa: E402

PANELS = (("a", "SWM4-NDP", "LFTS"), ("b", "SWM4-NDP", "DNLS"),
          ("c", "TIP5P", "LFTS"), ("d", "TIP5P", "DNLS"))

if __name__ == "__main__":
    set_style()
    colors = temperature_colors(config.TEMPS)
    for letter, model, group in PANELS:
        fig, ax = single_axes(AX_W, AX_H)
        for t in config.TEMPS:
            sk = structure_factor(model, t)
            ax.plot(sk["k_reduced"], sk[f"S_{group}"].mean(axis=0), color=colors[t], lw=1.5)
        bands(ax)
        ax.set(xlim=(0.6, 2.0), ylim=(0.2, 1.5))
        ax.set_xlabel(r"$k\,r_{\mathrm{OO}}/2\pi$")
        ax.set_ylabel(r"$S_H(k)$" if group == "LFTS" else r"$S_L(k)$")
        if group == "DNLS":
            temperature_legend(ax, config.TEMPS, colors)
        style_ax(ax)
        save(fig, f"appendix_fig7_{letter}")
