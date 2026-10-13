#!/usr/bin/env python3
"""SI Figure S4: the three benchmark partitions in the (zeta, q) plane, TIP4P/2005 at -20 C.

    appendix_fig4_a   K-means
    appendix_fig4_b   two-component GMM
    appendix_fig4_c   HDBSCAN -> GMM (removed molecules in grey)
"""
import numpy as np
from scipy.stats import gaussian_kde

from style import C, benchmark_data, save, set_style, single_axes, style_ax

if __name__ == "__main__":
    set_style()
    df = benchmark_data()
    x, y = df["zeta"].values, df["q"].values
    xlo, xhi = np.percentile(x, 0.5), np.percentile(x, 99.5)
    ylo, yhi = np.percentile(y, 1.0), np.percentile(y, 99.8)
    rng = np.random.default_rng(0)                        # one stream across a, b, c
    for letter, method in zip("abc", ("kmeans", "gmm", "hdbscan_gmm")):
        lab = df[method].values
        fig, ax = single_axes(2.05, 2.04)
        if (lab < 0).any():
            ax.scatter(x[lab < 0], y[lab < 0], s=1.2, c=C["NOISE"], alpha=0.25, linewidths=0,
                       rasterized=True)
        for cid, name in ((0, "LFTS"), (1, "DNLS")):
            xs, ys = x[lab == cid], y[lab == cid]
            sub = rng.choice(len(xs), size=min(5000, len(xs)), replace=False)
            kde = gaussian_kde(np.vstack([xs[sub], ys[sub]]))
            gx, gy = np.mgrid[xlo:xhi:110j, ylo:yhi:110j]
            zz = kde(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
            lv = np.linspace(zz.max() * 0.1, zz.max(), 5)
            ax.contourf(gx, gy, zz, levels=lv, colors=C[name], alpha=0.16)
            ax.contour(gx, gy, zz, levels=lv, colors=C[name], linewidths=0.7, alpha=0.9)
            ax.plot([], [], color=C[name], lw=4, alpha=0.6, label=name)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel(r"$\zeta$  (translational order)")
        ax.set_ylabel(r"$q$  (tetrahedral order)")
        if letter == "a":
            ax.legend(loc="lower right", fontsize=7.5, handlelength=1.1, borderpad=0.3)
        style_ax(ax)
        save(fig, f"appendix_fig4_{letter}")
