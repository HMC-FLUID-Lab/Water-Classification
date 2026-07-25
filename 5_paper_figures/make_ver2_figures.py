#!/usr/bin/env python3
"""
make_ver2_figures.py
====================
Generate ver2 of the two main figures using the NEW likelihood-ratio ambiguity
rejection (full-4-D Lambda = log p_LFTS - log p_DNLS, symmetric reject band),
so we can SEE its effect on the per-cluster S(k) FSDP peak vs the current method.

Outputs (standalone, minimal formatting):
  6_paper_writing/Paper_WaterMLClustering/images/main/1_clustering_ver2.png
  6_paper_writing/Paper_WaterMLClustering/images/main/2_validation_ver2.png

Figure 2 (validation) is the important one: per-cluster S(k) for the new method,
with the current confidence method's LFTS curve overlaid for a direct FSDP
comparison (Tanaka's first-diffraction pre-peak at k*r_OO/2pi ~ 0.75).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "3_clustering"))
sys.path.insert(0, os.path.join(_ROOT, "4_structure_factor"))
sys.path.insert(0, _HERE)
from water_clustering import scale_features, run_dbscan            # noqa: E402
from structure_factor_bycluster import (load_trajectory,           # noqa: E402
                                        compute_per_cluster_structure_factor,
                                        compute_partial_structure_factor_OO)

FEATURES = ["zeta_all", "q_all", "LSI_all", "Sk_all"]
EPS, MIN_SAMPLES, RS = 0.06, 20, 42
N_FRAMES, N_MOL = 20, 1024
RC, R_OO = 1.5, 0.285
K = np.linspace(0.1, 50.0, 300)
DCD = os.path.join(_ROOT, "data/simulations/tip4p2005/dcd_tip4p2005_T-20_N1024_Run01_0.dcd")
PDB = os.path.join(_ROOT, "data/simulations/tip4p2005/inistate_tip4p2005_T-20_N1024_Run01.pdb")
CONF_CSV = os.path.join(_ROOT, "results/clustering/tip4p2005_T-20_dbscan_gmm_conf/cluster_labels.csv")
OUTDIR = os.path.join(_ROOT, "6_paper_writing/Paper_WaterMLClustering/images/main")
CACHE = os.path.join(_HERE, "redesign_cache", "ver2_sk.npz")

C_LFTS, C_DNLS, C_NOISE = "#2c5fa8", "#c0392b", "#b9c0c8"
kn = lambda k: k * R_OO / (2 * np.pi)


def new_method_labels(X, tau):
    """Likelihood-ratio symmetric reject: 0=LFTS (Lambda>=tau), 1=DNLS (<=-tau), -1=removed."""
    db = run_dbscan(pd.DataFrame(X, columns=FEATURES), eps=EPS, min_samples=MIN_SAMPLES)
    gm = GaussianMixture(2, covariance_type="full", n_init=5, random_state=RS).fit(X[db != -1])
    zidx = FEATURES.index("zeta_all")
    lfts = int(np.argmax(gm.means_[:, zidx])); dnls = 1 - lfts
    mvnL = multivariate_normal(gm.means_[lfts], gm.covariances_[lfts], allow_singular=True)
    mvnD = multivariate_normal(gm.means_[dnls], gm.covariances_[dnls], allow_singular=True)
    lam = mvnL.logpdf(X) - mvnD.logpdf(X)
    lab = np.full(len(lam), -1, int)
    lab[lam >= tau] = 0
    lab[lam <= -tau] = 1
    return lab, lam


def matrix(lab):
    """flat (frame-major) -> (N_FRAMES,N_MOL) with -1 remapped to cluster 2 (transition)."""
    m = lab[:N_FRAMES * N_MOL].reshape(N_FRAMES, N_MOL).copy()
    m[m == -1] = 2
    return m


def lfts_dnls_ids(lm, zeta_flat):
    """which of cluster 0/1 is LFTS (higher mean zeta)."""
    zf = zeta_flat[:N_FRAMES * N_MOL].reshape(N_FRAMES, N_MOL)
    means = {c: zf[lm == c].mean() for c in (0, 1)}
    lfts = 0 if means[0] >= means[1] else 1
    return lfts, 1 - lfts


def sk_curves(traj, lm, zeta_flat):
    cr = compute_per_cluster_structure_factor(traj, RC, K, lm)
    lfts, dnls = lfts_dnls_ids(lm, zeta_flat)
    out = {"LFTS": np.asarray(cr[lfts]["S_k_avg"]), "DNLS": np.asarray(cr[dnls]["S_k_avg"])}
    if 2 in cr:
        out["transition"] = np.asarray(cr[2]["S_k_avg"])
    return out


def compute():
    df = pd.read_csv(CONF_CSV)
    good = np.isfinite(df[FEATURES].to_numpy()).all(1)
    df = df[good].reset_index(drop=True)
    X = scale_features(df[FEATURES].copy()).to_numpy(float)
    zeta_flat = df["zeta_all"].to_numpy()

    conf_lab = df["label_dbscan_gmm_conf"].to_numpy()
    # new method at two operating points
    _, lam = new_method_labels(X, tau=2.0)
    tau2 = 2.0
    conf_rm_frac = float((conf_lab == -1).mean())           # match the confidence method's removal budget
    tau_matched = float(np.quantile(np.abs(lam), conf_rm_frac))
    lab2 = np.where(lam >= tau2, 0, np.where(lam <= -tau2, 1, -1))
    labm = np.where(lam >= tau_matched, 0, np.where(lam <= -tau_matched, 1, -1))
    print(f"new tau=2.0 removed {100*(lab2==-1).mean():.1f}%   matched tau={tau_matched:.2f} removed {100*(labm==-1).mean():.1f}%   conf removed {100*(conf_lab==-1).mean():.1f}%")

    traj = load_trajectory(DCD, PDB)[:N_FRAMES]
    St, _, _ = compute_partial_structure_factor_OO(traj, RC, K)

    res = dict(
        conf=sk_curves(traj, matrix(conf_lab), zeta_flat),
        new2=sk_curves(traj, matrix(lab2), zeta_flat),
        newm=sk_curves(traj, matrix(labm), zeta_flat),
        total=np.asarray(St),
        tau2=tau2, tau_matched=tau_matched,
        conf_rm=100 * (conf_lab == -1).mean(), new2_rm=100 * (lab2 == -1).mean(),
        newm_rm=100 * (labm == -1).mean(),
        # for fig1
        zeta=df["zeta_all"].to_numpy()[:N_FRAMES * N_MOL],
        q=df["q_all"].to_numpy()[:N_FRAMES * N_MOL],
        lsi=df["LSI_all"].to_numpy()[:N_FRAMES * N_MOL],
        sk=df["Sk_all"].to_numpy()[:N_FRAMES * N_MOL],
        lab2=lab2[:N_FRAMES * N_MOL],
    )
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **{k: v for k, v in res.items() if not isinstance(v, dict)},
             **{f"{grp}_{c}": res[grp][c] for grp in ("conf", "new2", "newm") for c in res[grp]})
    return res


def fig_validation(res):
    plt.rcParams.update({"font.size": 15, "axes.linewidth": 1.0, "font.weight": "bold",
                         "axes.labelweight": "bold"})
    x = kn(K)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))

    # left: new method per-cluster S(k)
    a = ax[0]
    a.axvspan(0.725, 0.775, color=C_LFTS, alpha=0.10)
    a.axvspan(0.975, 1.025, color=C_DNLS, alpha=0.10)
    a.plot(x, res["new2"]["LFTS"], "-", color=C_LFTS, lw=2.6, label="LFTS")
    a.plot(x, res["new2"]["DNLS"], "-", color=C_DNLS, lw=2.6, label="DNLS")
    if "transition" in res["new2"]:
        a.plot(x, res["new2"]["transition"], "-", color="#7f8c8d", lw=1.8, label="transition")
    a.plot(x, res["total"], "--", color="#444444", lw=1.6, label="total (all mol.)")
    a.axvline(0.75, color=C_LFTS, ls=":", lw=1.0); a.axvline(1.0, color=C_DNLS, ls=":", lw=1.0)
    a.set_xlim(0.6, 2.0); a.set_ylim(0.5, 1.7)
    a.set_xlabel(r"$k\,r_{OO}/2\pi$"); a.set_ylabel(r"$S(k)$")
    a.set_title(f"(a) new method per-cluster S(k)\n(tau=2.0, {res['new2_rm']:.1f}% removed)")
    a.legend(fontsize=11)

    # right: LFTS FSDP comparison across methods
    b = ax[1]
    b.axvspan(0.725, 0.775, color=C_LFTS, alpha=0.10)
    b.plot(x, res["conf"]["LFTS"], "-", color="#8e44ad", lw=2.4, label=f"current confidence ({res['conf_rm']:.0f}% rm)")
    b.plot(x, res["new2"]["LFTS"], "-", color=C_LFTS, lw=2.4, label=f"new tau=2.0 ({res['new2_rm']:.0f}% rm)")
    b.plot(x, res["newm"]["LFTS"], "-", color="#16a085", lw=2.4, label=f"new matched ({res['newm_rm']:.0f}% rm)")
    b.axvline(0.75, color="k", ls=":", lw=1.0)
    b.set_xlim(0.6, 2.0); b.set_ylim(0.5, 1.7)
    b.set_xlabel(r"$k\,r_{OO}/2\pi$"); b.set_ylabel(r"$S_{\mathrm{LFTS}}(k)$")
    b.set_title("(b) LFTS FSDP peak: new vs current\n(higher at k~0.75 = stronger Tanaka pre-peak)")
    b.legend(fontsize=11)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "2_validation_ver2.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print("wrote", out)
    # FSDP peak heights for the record
    win = (x >= 0.70) & (x <= 0.80)
    print(f"  FSDP S_LFTS peak (k~0.75): conf={res['conf']['LFTS'][win].max():.3f}  "
          f"new2={res['new2']['LFTS'][win].max():.3f}  newm={res['newm']['LFTS'][win].max():.3f}")


def fig_clustering(res):
    plt.rcParams.update({"font.size": 14, "axes.linewidth": 1.0, "font.weight": "bold",
                         "axes.labelweight": "bold"})
    lab = res["lab2"]
    rng = np.random.default_rng(0)

    def sub(m, n=5000):
        idx = np.where(m)[0]
        return idx if len(idx) <= n else rng.choice(idx, n, replace=False)

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(2, 4)
    # big scatter (zeta,q)
    axs = fig.add_subplot(gs[:, :2])
    axs.scatter(res["zeta"][sub(lab == -1)], res["q"][sub(lab == -1)], s=4, c=C_NOISE, alpha=0.4, label="transition")
    axs.scatter(res["zeta"][sub(lab == 1)], res["q"][sub(lab == 1)], s=4, c=C_DNLS, alpha=0.5, label="DNLS")
    axs.scatter(res["zeta"][sub(lab == 0)], res["q"][sub(lab == 0)], s=4, c=C_LFTS, alpha=0.5, label="LFTS")
    axs.set_xlabel(r"$\zeta$ (nm)"); axs.set_ylabel(r"$q$")
    axs.set_title(f"new method assignment (tau=2.0, {res['new2_rm']:.1f}% removed)")
    axs.legend(fontsize=11, markerscale=3)

    # 2x2 per-feature histograms
    feats = [("zeta", r"$\zeta$ (nm)"), ("q", r"$q$"), ("lsi", "LSI"), ("sk", r"$S_k$")]
    for i, (key, xl) in enumerate(feats):
        ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
        v = res[key]
        lo, hi = np.percentile(v, 0.3), np.percentile(v, 99.7)
        bins = np.linspace(lo, hi, 55)
        for cid, col, name in ((0, C_LFTS, "LFTS"), (1, C_DNLS, "DNLS"), (-1, "#7f8c8d", "transition")):
            m = lab == cid
            if m.sum():
                ax.hist(v[m], bins=bins, density=True, histtype="step", color=col, lw=1.8, label=name)
        ax.set_xlabel(xl); ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTDIR, "1_clustering_ver2.png")
    fig.savefig(out, dpi=300); plt.close(fig)
    print("wrote", out)


def main():
    res = compute()
    fig_validation(res)
    fig_clustering(res)


if __name__ == "__main__":
    main()
