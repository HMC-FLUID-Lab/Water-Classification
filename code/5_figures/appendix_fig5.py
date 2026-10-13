#!/usr/bin/env python3
"""SI Figure S5: model selection for the Gaussian mixture, TIP4P/2005 at -20 C.
Five MinMax-scaled features (q, Q6, LSI, s_k, zeta); GMM full covariance,
n_init = 5, random_state = 42; silhouette on an 8,000-molecule subsample.

    appendix_fig5_a   BIC vs number of components K = 1-4
    appendix_fig5_b   AIC vs K
    appendix_fig5_c   silhouette score vs K = 2-4
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from style import C, reference_data, save, set_style, single_axes, style_ax
import config  # noqa: E402

KS = [1, 2, 3, 4]
AX_W, AX_H = 1.67, 1.95


def compute():
    X = MinMaxScaler().fit_transform(reference_data()[config.FEATURES_5].to_numpy(float))
    rows = []
    for k in KS:
        g = GaussianMixture(n_components=k, covariance_type="full", random_state=config.SEED,
                            n_init=5).fit(X)
        sil = (silhouette_score(X, g.predict(X), sample_size=8000, random_state=0)
               if k >= 2 else np.nan)
        rows.append(dict(K=k, BIC=g.bic(X), AIC=g.aic(X), silhouette=sil))
    out = pd.DataFrame(rows)
    out.to_csv(config.results("model_selection", "gmm_components.csv"), index=False)
    print(out.to_string(index=False))
    return out


def line(ax, x, y, color):
    ax.plot(x, y, color=color, lw=1.6, marker="o", ms=5, mfc="white", mec=color, mew=1.3)


if __name__ == "__main__":
    set_style()
    d = compute()
    for letter, col, ylabel, color in (("a", "BIC", r"BIC  ($\times 10^{5}$)", C["LFTS"]),
                                       ("b", "AIC", r"AIC  ($\times 10^{5}$)", "#1b7837")):
        fig, ax = single_axes(AX_W, AX_H)
        line(ax, KS, d[col] / 1e5, color)
        ax.set_xlabel("number of components, $k$"); ax.set_ylabel(ylabel)
        ax.set_xticks(KS)
        style_ax(ax)
        save(fig, f"appendix_fig5_{letter}")
    fig, ax = single_axes(AX_W, AX_H)
    line(ax, KS[1:], d["silhouette"][1:], C["DNLS"])
    ax.axvline(2, color="0.6", ls=(0, (4, 3)), lw=1.0)
    ax.set_xlabel("number of components, $k$"); ax.set_ylabel("silhouette score")
    ax.set_xticks(KS)
    style_ax(ax)
    save(fig, "appendix_fig5_c")
