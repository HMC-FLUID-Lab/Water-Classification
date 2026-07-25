#!/usr/bin/env python3
"""
make_ver2_skzeta3d.py
=====================
3-D S(k, zeta) surfaces (Tanaka Fig 2E style) for the NEW likelihood-ratio
method's LFTS / DNLS clusters, so the FSDP first-diffraction pre-peak is
actually visible as a ridge. This is the ALL-NEIGHBOUR zeta-resolved
representation where the two-state doublet is strong (unlike the per-cluster
same-label S(k) line plot).

Settings follow the project's verified recipe: k=linspace(0.1,50,500) (300
under-resolves the FSDP), fixed zeta bins linspace(-1,1.5,41) A, S range (0,2),
jet, Tanaka view angle, gaussian sigma=1.2.

Output: 6_paper_writing/Paper_WaterMLClustering/images/main/2_validation_ver2.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "3_clustering"))
sys.path.insert(0, os.path.join(_ROOT, "4_structure_factor"))
from water_clustering import scale_features, run_dbscan                 # noqa: E402
from structure_factor_bycluster import load_trajectory                 # noqa: E402
from sk_zeta_3d import compute_sk_zeta_matrix, _load_zeta               # noqa: E402

FEATURES = ["zeta_all", "q_all", "LSI_all", "Sk_all"]
EPS, MIN_SAMPLES, RS = 0.06, 20, 42
N_FRAMES, N_MOL = 20, 1024
RC, R_OO = 1.5, 0.285
TAU = 2.0
K = np.linspace(0.1, 50.0, 500)                     # 500 resolves the FSDP
ZETA_BINS = np.linspace(-1.0, 1.5, 41)              # A, fixed (Tanaka)
S_RANGE = (0.0, 2.0)
K_NORM_RANGE = (0.6, 2.0)

DCD = os.path.join(_ROOT, "data/simulations/tip4p2005/dcd_tip4p2005_T-20_N1024_Run01_0.dcd")
PDB = os.path.join(_ROOT, "data/simulations/tip4p2005/inistate_tip4p2005_T-20_N1024_Run01.pdb")
CONF_CSV = os.path.join(_ROOT, "results/clustering/tip4p2005_T-20_dbscan_gmm_conf/cluster_labels.csv")
ZETA_MAT = os.path.join(_ROOT, "data/order_params/OrderParamZeta_tip4p2005_T-20_Run01.mat")
OUT = os.path.join(_ROOT, "6_paper_writing/Paper_WaterMLClustering/images/main/2_validation_ver2.png")


def new_labels():
    df = pd.read_csv(CONF_CSV)
    good = np.isfinite(df[FEATURES].to_numpy()).all(1)
    df = df[good].reset_index(drop=True)
    X = scale_features(df[FEATURES].copy()).to_numpy(float)
    db = run_dbscan(pd.DataFrame(X, columns=FEATURES), eps=EPS, min_samples=MIN_SAMPLES)
    gm = GaussianMixture(2, covariance_type="full", n_init=5, random_state=RS).fit(X[db != -1])
    zidx = FEATURES.index("zeta_all")
    lfts = int(np.argmax(gm.means_[:, zidx])); dnls = 1 - lfts
    mvnL = multivariate_normal(gm.means_[lfts], gm.covariances_[lfts], allow_singular=True)
    mvnD = multivariate_normal(gm.means_[dnls], gm.covariances_[dnls], allow_singular=True)
    lam = mvnL.logpdf(X) - mvnD.logpdf(X)
    lab = np.where(lam >= TAU, 0, np.where(lam <= -TAU, 1, -1))          # 0=LFTS,1=DNLS,-1=removed
    return lab[:N_FRAMES * N_MOL].reshape(N_FRAMES, N_MOL)


def process(S_k_zeta, k_norm, zeta_centers):
    """Replicate sk_zeta_3d._plot_matplotlib_3d data conditioning."""
    S = np.copy(S_k_zeta)
    kmask = (k_norm >= K_NORM_RANGE[0]) & (k_norm <= K_NORM_RANGE[1])
    kp = k_norm[kmask]; S = S[:, kmask]
    row = ~np.all(np.isnan(S), axis=1)
    if row.any():
        f = int(np.argmax(row)); l = int(len(row) - np.argmax(row[::-1]))
        S = S[f:l]; zc = zeta_centers[f:l]
    else:
        zc = zeta_centers
    for ki in range(S.shape[1]):
        col = S[:, ki]; nan = np.isnan(col)
        if nan.any() and not nan.all():
            idx = np.arange(len(col)); col[nan] = np.interp(idx[nan], idx[~nan], col[~nan]); S[:, ki] = col
    S = np.clip(S, *S_RANGE)
    S = gaussian_filter(S, sigma=1.2)
    S = np.clip(S, *S_RANGE)
    return kp, zc, S


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

    fig = plt.figure(figsize=(15, 6.2))
    norm = Normalize(*S_RANGE); cmap = plt.get_cmap("jet")
    for i, (name, col) in enumerate((("LFTS", "#2c5fa8"), ("DNLS", "#c0392b"))):
        kp, zc, S = surfaces[name]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        K2, Z2 = np.meshgrid(kp, zc)
        ax.plot_surface(K2, Z2, S, facecolors=cmap(norm(S)), rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False)
        ax.set_xlabel(r"$k\,r_{OO}/2\pi$", labelpad=8); ax.set_ylabel(r"$\zeta$ (Å)", labelpad=8)
        ax.set_zlabel(r"$S(k,\zeta)$", labelpad=4)
        ax.set_xlim(float(kp[0]), float(kp[-1])); ax.set_ylim(float(zc[0]), float(zc[-1]))
        ax.set_zlim(*S_RANGE); ax.view_init(elev=28, azim=-115)
        peak = np.nanmax(S)
        ax.set_title(f"{name} — new method\nmax S(k,$\\zeta$) = {peak:.2f}", color=col, fontweight="bold")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    fig.colorbar(sm, ax=fig.axes, shrink=0.5, aspect=14, label=r"$S(k,\zeta)$", pad=0.02)
    fig.suptitle("S(k, ζ) 3-D surfaces — first-diffraction ridge (Tanaka Fig 2E style), new method",
                 fontweight="bold")
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)
    for name in ("LFTS", "DNLS"):
        _, _, S = surfaces[name]
        print(f"  {name}: max S(k,zeta) = {np.nanmax(S):.3f}")


if __name__ == "__main__":
    main()
