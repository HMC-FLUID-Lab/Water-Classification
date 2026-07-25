#!/usr/bin/env python3
"""
run_generality_lr.py
====================
Re-run the whole T-20 clustering + generality experiment under the LIKELIHOOD-
RATIO labelling (the "new method": DBSCAN core-detection -> 2-component GMM ->
symmetric likelihood-ratio reject band |Lambda|<tau, tau=2.0; 4 features, Q6
dropped), so every needed_photo figure is built from ONE consistent method.

CONTROL: the classifier is fit ONCE at the 4p_T-20 reference (fit_frozen_lr) and
then FROZEN -- scaler.transform + frozen GMM + tau, no refit -- and applied to
every other condition. Fig-3 panels a/b/c vary only temperature and panel d
varies only the water model, so each isolates its single variable. build_labels
and build_line are -20-only (fit==apply), so freezing leaves them unchanged.

Writes (all consumed by make_needed_photos.py):
  results/clustering/tip4p2005_T-20_lr/cluster_labels.csv   flat LR labels @ -20  (figs 1b/c/d)
  redesign_cache/validation_line_3state_lr.npz              per-state S(k) @ -20   (fig 2 line)
  sk_cache/sk_{4p_T*,5p_T-20,swm_T-20}_lr.npz               per-state S(k)         (fig 3 a/b/d)
  results/clustering/multirun_lr/fractions_blocktrim_lr.csv LFTS/transition frac(T) (fig 3c)

Usage:  python run_generality_lr.py [labels|skvsT|line|blocktrim|all]  (default all)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")                       # set before numpy/BLAS import (>64-thread segfault)

import sys
import io
import time
import contextlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "3_clustering"))
sys.path.insert(0, os.path.join(_ROOT, "4_structure_factor"))

from water_clustering import run_dbscan                                       # noqa: E402
from structure_factor_bycluster import (load_trajectory,                      # noqa: E402
                                        compute_per_cluster_structure_factor)

SIM = os.path.join(_ROOT, "data", "simulations")
OD = os.path.join(_ROOT, "data", "order_params")
CLUST = os.path.join(_ROOT, "results", "clustering")
SK_CACHE = os.path.join(_HERE, "sk_cache")
RC_CACHE = os.path.join(_HERE, "redesign_cache")
CONF_CSV = os.path.join(CLUST, "tip4p2005_T-20_dbscan_gmm_conf", "cluster_labels.csv")

FEATURES = ["zeta_all", "q_all", "LSI_all", "Sk_all"]     # 4 paper features (Q6 dropped)
EPS, MIN_SAMPLES, RS, TAU = 0.06, 20, 42, 2.0             # ver2 likelihood-ratio settings
N_FRAMES, N_MOL = 20, 1024
RC = 1.5
K_VALUES = np.linspace(0.1, 50.0, 300)
R_OO = 0.285
TEMPS = [-30, -20, -10, 0, 10, 20, 30]

# constant-clustering-quality panel c (fig 3c): tune DBSCAN eps at fixed MinPts to
# hold the 2-state silhouette at a target s* reachable at every T (4 features).
CS_EPS = np.round(np.arange(0.02, 0.1601, 0.0075), 4)     # eps sweep (tight -> loose)
CS_MS = 15                                                # DBSCAN MinPts (paper op)
CS_NF = 40                                                # frames/run (Run02/03 ship 100)
CS_KEEP_FLOOR = 0.03                                      # ignore cells keeping <3% (sil-inflation guard)
CS_MIN_COMP = 30                                          # each GMM component needs >= this many pts
CS_MARGIN = 0.005                                         # s* headroom below the worst-case ceiling


# ─────────────────────────────────────────────────────────────────────────────
# likelihood-ratio labelling
# ─────────────────────────────────────────────────────────────────────────────
def fit_frozen_lr(ref_df):
    """Fit the likelihood-ratio classifier ONCE on a reference frame and freeze
    it: MinMax scaler (fit on the finite reference rows), DBSCAN core, 2-comp GMM,
    and the LFTS/DNLS identity (component with the larger zeta mean = LFTS).
    Returns a dict {scaler, mvnL, mvnD, means, lfts, dnls} reused across
    conditions so temperature is the ONLY thing that varies downstream."""
    raw = ref_df[FEATURES].to_numpy(float)
    finite = np.isfinite(raw).all(axis=1)
    scaler = MinMaxScaler().fit(raw[finite])                 # [0,1] on reference ranges
    Xs = scaler.transform(raw[finite])
    with contextlib.redirect_stdout(io.StringIO()):
        db = run_dbscan(pd.DataFrame(Xs, columns=FEATURES), eps=EPS, min_samples=MIN_SAMPLES)
    gm = GaussianMixture(2, covariance_type="full", n_init=5, random_state=RS).fit(Xs[db != -1])
    zidx = FEATURES.index("zeta_all")
    lfts = int(np.argmax(gm.means_[:, zidx])); dnls = 1 - lfts
    return dict(
        scaler=scaler,
        mvnL=multivariate_normal(gm.means_[lfts], gm.covariances_[lfts], allow_singular=True),
        mvnD=multivariate_normal(gm.means_[dnls], gm.covariances_[dnls], allow_singular=True),
        means=gm.means_, lfts=lfts, dnls=dnls,
    )


def apply_lr(df_feats, model):
    """Flat LR labels for a feature frame using a FROZEN model (no refit):
    0=LFTS, 1=DNLS, -1=transition. Non-finite rows -> -1. Uses scaler.transform
    (NOT fit_transform) so test conditions are mapped into the reference space."""
    raw = df_feats[FEATURES].to_numpy(float)
    finite = np.isfinite(raw).all(axis=1)
    Xs = model["scaler"].transform(raw[finite])
    lam = model["mvnL"].logpdf(Xs) - model["mvnD"].logpdf(Xs)
    lab_fin = np.where(lam >= TAU, 0, np.where(lam <= -TAU, 1, -1))
    lab = np.full(len(df_feats), -1, int)
    lab[finite] = lab_fin
    return lab


def lr_labels(df_feats):
    """Per-frame LR labels: fit + apply on the SAME frame. Behaviour-identical to
    the original refit-every-call routine; kept for the -20-only steps (build_labels,
    build_line) where fit==apply so freezing makes no difference."""
    return apply_lr(df_feats, fit_frozen_lr(df_feats))


_FROZEN = {}


def _frozen_ref():
    """The single LR classifier frozen at the 4p_T-20 control, shared by EVERY
    generality panel so each varies ONLY the quantity under study: temperature
    (a/b S(k)-vs-T, c fraction-vs-T) or water model (d). Fit once, memoized."""
    if "m" not in _FROZEN:
        _FROZEN["m"] = fit_frozen_lr(feats_for("4p_T-20"))
    return _FROZEN["m"]


# ─────────────────────────────────────────────────────────────────────────────
# per-condition feature / trajectory access
# ─────────────────────────────────────────────────────────────────────────────
def _mat_feats(model_dir, prefix, t, run="Run01", n_frames=N_FRAMES):
    op = loadmat(os.path.join(OD, f"OrderParam_{prefix}_T{t}_{run}.mat"))
    zt = loadmat(os.path.join(OD, f"OrderParamZeta_{prefix}_T{t}_{run}.mat"))["zeta_all"]
    df = pd.DataFrame({c: op[c][:n_frames].flatten() for c in ("q_all", "Q6_all", "LSI_all", "Sk_all")})
    df["zeta_all"] = zt[:n_frames].flatten()
    return df


def feats_for(tag):
    """Feature frame (frame-major, N_FRAMES*N_MOL rows) for a condition tag."""
    if tag == "4p_T-20":
        df = pd.read_csv(CONF_CSV)                       # same source the heatmap uses
        return df.iloc[:N_FRAMES * N_MOL].reset_index(drop=True)
    if tag.startswith("4p_T"):
        return _mat_feats("tip4p2005", "tip4p2005", tag[4:])
    if tag == "5p_T-20":
        return _mat_feats("tip5p", "tip5p", "-20")
    if tag == "swm_T-20":
        return _mat_feats("swm4ndp", "swm4ndp", "-20")
    raise ValueError(tag)


def traj_for(tag):
    if tag.startswith("4p_T"):
        t = tag[4:]; d = os.path.join(SIM, "tip4p2005")
        return (os.path.join(d, f"dcd_tip4p2005_T{t}_N1024_Run01_0.dcd"),
                os.path.join(d, f"inistate_tip4p2005_T{t}_N1024_Run01.pdb"))
    if tag == "5p_T-20":
        d = os.path.join(SIM, "tip5p")
        return (os.path.join(d, "dcd_tip5p_T-20_N1024_Run01_0.dcd"),
                os.path.join(d, "inistate_tip5p_T-20_N1024_Run01.pdb"))
    if tag == "swm_T-20":
        d = os.path.join(SIM, "swm4ndp")
        return (os.path.join(d, "dcd_swm4ndp_T-20_N1024_Run01_0.dcd"),
                os.path.join(d, "inistate_swm4ndp_T-20_N1024_Run01.pdb"))
    raise ValueError(tag)


# ─────────────────────────────────────────────────────────────────────────────
# 1. flat LR labels @ -20  (figs 1b/c/d)
# ─────────────────────────────────────────────────────────────────────────────
def build_labels():
    df = feats_for("4p_T-20")
    lab = lr_labels(df)
    out_dir = os.path.join(CLUST, "tip4p2005_T-20_lr")
    os.makedirs(out_dir, exist_ok=True)
    cols = ["q_all", "Q6_all", "LSI_all", "Sk_all", "zeta_all"]
    o = df[cols].copy()
    o["label_lr"] = lab
    o.to_csv(os.path.join(out_dir, "cluster_labels.csv"), index=False)
    n = {c: int((lab == c).sum()) for c in (-1, 0, 1)}
    print(f"  labels @ -20: LFTS={n[0]:,}  DNLS={n[1]:,}  transition={n[-1]:,} "
          f"({100*n[-1]/len(lab):.1f}% removed)  -> tip4p2005_T-20_lr/cluster_labels.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2. per-state S(k) vs T + model  (figs 3 a/b/d)
# ─────────────────────────────────────────────────────────────────────────────
def _sk_one(tag, frozen):
    dcd, pdb = traj_for(tag)
    lab = apply_lr(feats_for(tag), frozen)              # frozen -20 model everywhere:
    #                                                     T sweep (a/b) AND cross-model (d)
    nf = len(lab) // N_MOL                              # some models ship < N_FRAMES frames
    lm = lab[:nf * N_MOL].reshape(nf, N_MOL)
    traj = load_trajectory(dcd, pdb)[:nf]
    with contextlib.redirect_stdout(io.StringIO()):
        cr = compute_per_cluster_structure_factor(traj, RC, K_VALUES, lm)   # ids >=0 only
    pops = {int(c): (lm == c).sum(axis=1) for c in np.unique(lm) if c >= 0}
    out = os.path.join(SK_CACHE, f"sk_{tag}_lr.npz")
    np.savez(out, k_values=K_VALUES, cluster_results=cr, populations=pops, n_frames=nf)
    kn = K_VALUES * R_OO / (2 * np.pi)                  # FSDP lives near kn~1.38
    win = (kn >= 1.2) & (kn <= 1.55)
    lfts_pk = float(np.nanmax(np.asarray(cr[0]["S_k_avg"])[win])) if 0 in cr else float("nan")
    print(f"  {tag} ({nf} frames): clusters={list(cr.keys())} "
          f"pops~{[int(np.mean(pops[c])) for c in sorted(pops)]} "
          f"LFTS_FSDP={lfts_pk:.3f}  -> {os.path.basename(out)}")


def build_skvsT():
    tags = [f"4p_T{t}" for t in TEMPS] + ["5p_T-20", "swm_T-20"]
    frozen = _frozen_ref()                              # single classifier frozen at the -20 control
    zidx = FEATURES.index("zeta_all")
    ref_lab = apply_lr(feats_for("4p_T-20"), frozen)
    nL, nD, nT = (int((ref_lab == c).sum()) for c in (0, 1, -1))
    print(f"  frozen model @ 4p_T-20: zeta means "
          f"L={frozen['means'][frozen['lfts'], zidx]:.3f} "
          f"D={frozen['means'][frozen['dnls'], zidx]:.3f}  "
          f"LFTS={nL:,} DNLS={nD:,} transition={nT:,}")
    t0 = time.time()
    for tag in tags:
        _sk_one(tag, frozen)
    print(f"  sk-vs-T done ({time.time()-t0:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. per-state S(k) @ -20 incl. transition  (fig 2 line)
# ─────────────────────────────────────────────────────────────────────────────
def _kn(k):
    return k * R_OO / (2 * np.pi)


def _fsdp_score(S, k):
    f = (k >= 0.72) & (k <= 0.86); d = (k >= 0.95) & (k <= 1.10)
    return np.nanmean(S[f]) - np.nanmean(S[d])


def build_line():
    lab = lr_labels(feats_for("4p_T-20"))
    lm = lab[:N_FRAMES * N_MOL].reshape(N_FRAMES, N_MOL).copy()
    lm[lm == -1] = 2                                     # transition -> its own id
    dcd, pdb = traj_for("4p_T-20")
    traj = load_trajectory(dcd, pdb)[:N_FRAMES]
    with contextlib.redirect_stdout(io.StringIO()):
        cr = compute_per_cluster_structure_factor(traj, RC, K_VALUES, lm)
    k = _kn(K_VALUES)
    states = [c for c in cr if c != 2]
    score = {c: _fsdp_score(np.asarray(cr[c]["S_k_avg"]), k) for c in states}
    lfts, dnls = max(score, key=score.get), min(score, key=score.get)

    def pack(cid):
        nf = np.asarray(cr[cid]["S_k_frames"])
        return dict(S=np.asarray(cr[cid]["S_k_avg"]),
                    sem=np.asarray(cr[cid]["S_k_std"]) / np.sqrt(max(len(nf), 1)))
    D = {"LFTS": pack(lfts), "DNLS": pack(dnls), "noise": pack(2)}
    out = os.path.join(RC_CACHE, "validation_line_3state_lr.npz")
    os.makedirs(RC_CACHE, exist_ok=True)
    np.savez(out, k=k, D=D)
    print(f"  line @ -20: LFTS=cid{lfts} DNLS=cid{dnls} + transition -> {os.path.basename(out)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. LFTS / transition fraction vs T, block-trimmed over 3 runs  (fig 3c)
# ─────────────────────────────────────────────────────────────────────────────
def build_blocktrim():
    op_rep = os.path.join(OD, "errbar_runs")
    out = os.path.join(CLUST, "multirun_lr")
    os.makedirs(out, exist_ok=True)
    runs = ["Run01", "Run02", "Run03"]
    block, n_blocks = 20, 5
    frozen = _frozen_ref()                              # same -20 classifier as a/b/d; blocks only sample data
    zidx = FEATURES.index("zeta_all")
    print(f"  frozen model @ 4p_T-20: zeta means "
          f"L={frozen['means'][frozen['lfts'], zidx]:.3f} "
          f"D={frozen['means'][frozen['dnls'], zidx]:.3f}")

    def frames_df(t, run, frames):
        cand = os.path.join(op_rep, f"OrderParam_tip4p2005_T{t}_{run}.mat")
        d = op_rep if os.path.exists(cand) else OD
        m = loadmat(os.path.join(d, f"OrderParam_tip4p2005_T{t}_{run}.mat"))
        z = loadmat(os.path.join(d, f"OrderParamZeta_tip4p2005_T{t}_{run}.mat"))
        cols = {c: np.concatenate([m[c][i] for i in frames]) for c in ("q_all", "LSI_all", "Sk_all")}
        cols["zeta_all"] = np.concatenate([z["zeta_all"][i] for i in frames])
        return pd.DataFrame(cols)

    def fracs(df):
        lab = apply_lr(df, frozen)                      # frozen -20 boundary, no per-block refit
        n_l, n_d, n_t = int((lab == 0).sum()), int((lab == 1).sum()), int((lab == -1).sum())
        tot = n_l + n_d + n_t
        return n_l / (n_l + n_d), n_t / tot

    def trim(vals):
        v = np.sort(np.asarray(vals, float)); core = v[1:-1]
        return core.mean(), core.std(ddof=1), len(core), len(v)

    per_rows, agg_rows = [], []
    t0 = time.time()
    for t in TEMPS:
        s_vals, n_vals = [], []
        for run in runs:
            for b in range(n_blocks):
                fr = list(range(b * block, b * block + block))
                s, nz = fracs(frames_df(t, run, fr))
                s_vals.append(s); n_vals.append(nz)
                per_rows.append(dict(temp=t, run=run, block=b, lfts_frac=s, noise_frac=nz))
        s_m, s_sd, kept, tot = trim(s_vals)
        n_m, n_sd, _, _ = trim(n_vals)
        agg_rows.append(dict(temp=t, lfts_mean=s_m, lfts_std=s_sd,
                             noise_mean=n_m, noise_std=n_sd, n_used=kept, n_blocks=tot))
        print(f"  T{t:+3d}: s={s_m:.3f}±{s_sd:.3f}  transition={n_m:.3f}±{n_sd:.3f}  ({kept}/{tot} blocks)")
    pd.DataFrame(per_rows).to_csv(os.path.join(out, "fractions_per_block_lr.csv"), index=False)
    pd.DataFrame(agg_rows).to_csv(os.path.join(out, "fractions_blocktrim_lr.csv"), index=False)
    print(f"  blocktrim done ({time.time()-t0:.1f}s) -> multirun_lr/fractions_blocktrim_lr.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 5. constant-clustering-quality panel c: noise-removed + LFTS vs T  (fig 3c alt)
# ─────────────────────────────────────────────────────────────────────────────
def build_constsil():
    """Panel c under CONSTANT clustering quality (4 features), the control the
    fixed-tau band cannot express. Per (T, run): MinMax-scale the 4 features, sweep
    DBSCAN eps at MinPts=CS_MS, and at each eps fit GMM(2) on the retained core and
    score its silhouette. Hold the silhouette at a target s* (the highest quality
    reachable at EVERY T, so the curve is defined throughout) by picking the loosest
    eps meeting s* (linear-interpolated to sil=s*). Report DBSCAN noise-removed
    fraction and the LFTS core as a fraction of ALL molecules. As T rises the two
    states overlap, so holding quality fixed forces discarding more ambiguous
    molecules -> noise up, LFTS down. Mean +/- std over Run01/02/03."""
    op_rep = os.path.join(OD, "errbar_runs")
    out = os.path.join(CLUST, "multirun_constsil")
    os.makedirs(out, exist_ok=True)
    runs = ["Run01", "Run02", "Run03"]
    zidx = FEATURES.index("zeta_all")

    def load(t, run):
        cand = os.path.join(op_rep, f"OrderParam_tip4p2005_T{t}_{run}.mat")
        d = op_rep if os.path.exists(cand) else OD
        m = loadmat(os.path.join(d, f"OrderParam_tip4p2005_T{t}_{run}.mat"))
        z = loadmat(os.path.join(d, f"OrderParamZeta_tip4p2005_T{t}_{run}.mat"))["zeta_all"]
        cols = {c: m[c][:CS_NF].flatten() for c in ("q_all", "LSI_all", "Sk_all")}
        cols["zeta_all"] = z[:CS_NF].flatten()
        X = np.column_stack([cols[c] for c in FEATURES]).astype(float)
        return X[np.isfinite(X).all(axis=1)]

    def sweep(t, run):
        """(eps, noise, sil, lfts_of_all) rows, eps ascending, valid cells only."""
        X = MinMaxScaler().fit_transform(load(t, run))
        n = len(X); rows = []
        for eps in CS_EPS:
            with contextlib.redirect_stdout(io.StringIO()):
                lab = DBSCAN(eps=float(eps), min_samples=CS_MS).fit_predict(X)
            keep = lab != -1; kf = keep.mean()
            if kf < CS_KEEP_FLOOR:
                continue
            gm = GaussianMixture(2, covariance_type="full", random_state=RS, n_init=1).fit(X[keep])
            gp = gm.predict(X[keep])
            if len(np.unique(gp)) < 2 or int(np.bincount(gp).min()) < CS_MIN_COMP:
                continue
            sil = silhouette_score(X[keep], gp, sample_size=min(4000, int(keep.sum())), random_state=0)
            lfts = (gp == int(np.argmax(gm.means_[:, zidx]))).sum() / n
            rows.append((float(eps), 1.0 - kf, float(sil), float(lfts)))
        return sorted(rows, key=lambda r: r[0])          # eps ascending (sil falls as eps grows)

    sweeps = {(t, r): sweep(t, r) for t in TEMPS for r in runs}
    s_star = min(max(s[2] for s in sweeps[(t, r)]) for t in TEMPS for r in runs) - CS_MARGIN
    print(f"  s* = {s_star:.3f}  (min over T,run of max achievable silhouette - {CS_MARGIN})")

    def op_point(rows, star):
        """noise, lfts at sil == star. Loosest eps meeting star, interpolated in eps
        to the exact crossing; fall back to the max-silhouette cell if unreachable."""
        reach = [r for r in rows if r[2] >= star]
        if not reach:                                    # even the tightest eps misses star
            r = max(rows, key=lambda z: z[2])
            return r[1], r[3], False
        r_lo = max(reach, key=lambda z: z[0])            # loosest eps with sil >= star
        after = [r for r in rows if r[0] > r_lo[0]]      # first cell that dips below star
        if not after:                                    # loosest cell already meets it
            return r_lo[1], r_lo[3], True
        r_hi = min(after, key=lambda z: z[0])
        f = (r_lo[2] - star) / (r_lo[2] - r_hi[2]) if r_lo[2] != r_hi[2] else 0.0
        return (r_lo[1] + f * (r_hi[1] - r_lo[1]),        # noise at sil = star
                r_lo[3] + f * (r_hi[3] - r_lo[3]), True)  # lfts  at sil = star

    per_rows, agg_rows = [], []
    t0 = time.time()
    for t in TEMPS:
        ns, ls, reached = [], [], []
        for r in runs:
            noise, lfts, ok = op_point(sweeps[(t, r)], s_star)
            ns.append(noise); ls.append(lfts); reached.append(ok)
            per_rows.append(dict(temp=t, run=r, noise=noise, lfts=lfts, reached=ok))
        agg_rows.append(dict(temp=t,
                             noise_mean=float(np.mean(ns)), noise_std=float(np.std(ns, ddof=1)),
                             lfts_mean=float(np.mean(ls)), lfts_std=float(np.std(ls, ddof=1)),
                             s_star=float(s_star), n_reached=int(sum(reached))))
        tail = "" if all(reached) else f"  [{len(runs)-sum(reached)} run(s) at max-sil]"
        print(f"  T{t:+3d}: noise={np.mean(ns)*100:5.1f}±{np.std(ns)*100:4.1f}%  "
              f"LFTS={np.mean(ls)*100:5.1f}±{np.std(ls)*100:4.1f}%{tail}")
    pd.DataFrame(per_rows).to_csv(os.path.join(out, "constsil_per_run.csv"), index=False)
    pd.DataFrame(agg_rows).to_csv(os.path.join(out, "fractions_constsil_lr.csv"), index=False)
    print(f"  constsil done ({time.time()-t0:.1f}s) -> multirun_constsil/fractions_constsil_lr.csv")


_STEPS = {"labels": build_labels, "skvsT": build_skvsT, "line": build_line,
          "blocktrim": build_blocktrim, "constsil": build_constsil}


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    which = list(_STEPS) if which == ["all"] or "all" in which else which
    for name in which:
        print(f"--- {name} ---")
        _STEPS[name]()
    print("done.")
