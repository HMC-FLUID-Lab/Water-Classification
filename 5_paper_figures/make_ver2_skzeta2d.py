#!/usr/bin/env python3
"""
make_ver2_skzeta2d.py
=====================
2-D S(k, zeta) heatmaps (filled-contour, "panel c" style) for the NEW
likelihood-ratio method's LFTS / DNLS clusters. These use the *same* S(k, zeta)
matrices that feed the 3-D surfaces in make_ver2_skzeta3d.py — just rendered
top-down as contourf maps with zeta on the y-axis, k*r_OO/2pi on the x-axis,
and the FSDP (k_T1) / DNLS (k_D1) first-peak positions marked with dashed lines.

Output: 6_paper_writing/Paper_WaterMLClustering/images/main/2_validation_ver2_heatmap.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "3_clustering"))
sys.path.insert(0, os.path.join(_ROOT, "4_structure_factor"))

# reuse the exact data pipeline that produces the 3-D surfaces
from make_ver2_skzeta3d import (new_labels, process, K, R_OO, ZETA_BINS, RC,   # noqa: E402
                                N_FRAMES, S_RANGE, DCD, PDB, ZETA_MAT)
from structure_factor_bycluster import load_trajectory                         # noqa: E402
from sk_zeta_3d import compute_sk_zeta_matrix, _load_zeta                       # noqa: E402

K_T1 = 0.75          # FSDP first-diffraction pre-peak (Tanaka), k*r_OO/2pi
K_D1 = 1.00          # DNLS main first peak, k*r_OO/2pi
C_RANGE = (0.0, 1.8)  # colour-scale range (data clip stays at S_RANGE); capped
                      # below the true max so the peaks get more colour contrast
OUT = os.path.join(_ROOT, "6_paper_writing/Paper_WaterMLClustering/images/main/2_validation_ver2_heatmap.png")


def main():
    lm = new_labels()
    print(f"LFTS mol/frame ~ {(lm==0).sum()/N_FRAMES:.0f}   DNLS ~ {(lm==1).sum()/N_FRAMES:.0f}   removed ~ {(lm==-1).sum()/N_FRAMES:.0f}")
    traj = load_trajectory(DCD, PDB)[:N_FRAMES]
    zeta_all = _load_zeta(ZETA_MAT)[:N_FRAMES]
    k_norm = K * R_OO / (2 * np.pi)

    surfaces = {}
    for cid, name in ((0, "LFTS"), (1, "DNLS")):
        S_kz, zc = compute_sk_zeta_matrix(traj, K, lm, cid, zeta_all, ZETA_BINS, RC)
        surfaces[name] = process(S_kz, k_norm, zc)

    plt.rcParams.update({"font.size": 14, "axes.linewidth": 1.0, "font.weight": "bold",
                         "axes.labelweight": "bold"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    norm = Normalize(*C_RANGE)
    cmap = plt.get_cmap("jet")
    fill_levels = np.linspace(C_RANGE[0], C_RANGE[1], 37)
    line_levels = np.linspace(C_RANGE[0], C_RANGE[1], 10)

    for ax, (name, col) in zip(axes, (("LFTS", "#2c5fa8"), ("DNLS", "#c0392b"))):
        kp, zc, S = surfaces[name]
        ax.contourf(kp, zc, S, levels=fill_levels, cmap=cmap, norm=norm, extend="both")
        ax.contour(kp, zc, S, levels=line_levels, colors="k", linewidths=0.4, alpha=0.35)
        ax.axvline(K_T1, color="#12347a", ls="--", lw=1.8, label=r"$k_{T1}$ (FSDP)")
        ax.axvline(K_D1, color="#7a1f17", ls="--", lw=1.8, label=r"$k_{D1}$ (DNLS)")
        ax.set_xlim(float(kp[0]), float(kp[-1]))
        ax.set_ylim(float(zc[0]), float(zc[-1]))
        ax.set_xlabel(r"$k\,r_{OO}/2\pi$")
        ax.set_ylabel(r"$\zeta$ (Å)")
        ax.set_title(f"{name} — new method", color=col, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.9, aspect=22, label=r"$S(k,\zeta)$", pad=0.02)
    fig.suptitle("S(k, ζ) 2-D heatmaps — first-diffraction ridge, new method", fontweight="bold")
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)
    for name in ("LFTS", "DNLS"):
        _, _, S = surfaces[name]
        print(f"  {name}: max S(k,zeta) = {np.nanmax(S):.3f}")


if __name__ == "__main__":
    main()
