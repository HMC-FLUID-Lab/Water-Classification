#!/usr/bin/env python3
"""
build_data.py
=============
Build the cached inputs the figure scripts read. Run these once (they need the
MD trajectories + the order-parameter .mat files); afterwards make_main_figures
and make_si_figures read only the caches.

  conf_sk     per-cluster S(k) for the confidence-based labels, INCLUDING the
              transition (middle-band) population
              -> sk_cache/sk_4p_T-20_conf.npz
  redesign    total O-O S(k), the S(k,zeta) landscape, and a real-space snapshot
              -> redesign_cache/{total_sk,skzeta,snapshot}_4p_T-20.npz
  blocktrim   LFTS-fraction / noise-fraction vs T aggregated over 15 twenty-frame
              blocks (5 x Run01/02/03) with the two extremes trimmed
              -> results/clustering/multirun_dbscan_gmm_realreps/fractions_blocktrim.csv

Consolidated from build_conf_sk.py, redesign_data.py and
multirun_clustering_blocktrim.py.

Usage:  python build_data.py [conf_sk|redesign|blocktrim|all]   (default: all)
"""
from __future__ import annotations

import os
import sys
import io
import time
import contextlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

os.environ.setdefault("OMP_NUM_THREADS", "8")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "4_structure_factor"))
sys.path.insert(0, os.path.join(_ROOT, "3_clustering"))

SIM = os.path.join(_ROOT, "data", "simulations", "tip4p2005")
CLUST = os.path.join(_ROOT, "results", "clustering")
CSM = os.path.join(CLUST, "cluster_labels_matrices")
SK_CACHE = os.path.join(_HERE, "sk_cache")
RC_CACHE = os.path.join(_HERE, "redesign_cache")

DCD = os.path.join(SIM, "dcd_tip4p2005_T-20_N1024_Run01_0.dcd")
PDB = os.path.join(SIM, "inistate_tip4p2005_T-20_N1024_Run01.pdb")

K_VALUES = np.linspace(0.1, 50.0, 300)
RC = 1.5
N_FRAMES = 20
N_MOL = 1024


# ═════════════════════════════════════════════════════════════════════════════
# conf_sk  —  per-cluster S(k) for the confidence-based labels (+ transition)
# ═════════════════════════════════════════════════════════════════════════════
def build_conf_sk():
    from structure_factor_bycluster import (load_trajectory,
                                            compute_per_cluster_structure_factor)
    flat = os.path.join(CLUST, "tip4p2005_T-20_dbscan_gmm_conf", "cluster_labels.csv")
    out = os.path.join(SK_CACHE, "sk_4p_T-20_conf.npz")
    transition_key = 2   # -1 -> 2 so it is computed alongside the two states

    t0 = time.time()
    traj = load_trajectory(DCD, PDB)[:N_FRAMES]
    df = pd.read_csv(flat)
    n = N_FRAMES * N_MOL
    lm = df["label_dbscan_gmm_conf"].values[:n].astype(int).reshape(N_FRAMES, N_MOL)
    lm_remap = lm.copy()
    lm_remap[lm == -1] = transition_key

    cr = compute_per_cluster_structure_factor(traj, RC, K_VALUES, lm_remap)
    pops = {int(c): (lm_remap == c).sum(axis=1) for c in np.unique(lm_remap)}

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, k_values=K_VALUES, cluster_results=cr, populations=pops,
             n_frames=N_FRAMES, transition_key=transition_key)
    print(f"saved {out}  clusters={list(cr.keys())}  ({time.time()-t0:.1f}s)")


# ═════════════════════════════════════════════════════════════════════════════
# redesign  —  total S(k), S(k,zeta) landscape, real-space snapshot
# ═════════════════════════════════════════════════════════════════════════════
def build_redesign_caches():
    from structure_factor_bycluster import load_trajectory, compute_structure_factor
    from sk_zeta_3d import compute_sk_zeta_matrix
    mat = os.path.join(CSM, "cluster_labels_matrix_tip4p2005_T-20_dbscan_gmm.csv")
    flat = os.path.join(CLUST, "tip4p2005_T-20_dbscan_gmm", "cluster_labels.csv")
    os.makedirs(RC_CACHE, exist_ok=True)

    t0 = time.time()
    traj = load_trajectory(DCD, PDB)[:N_FRAMES]
    lm = pd.read_csv(mat, header=None).values.astype(int)[:N_FRAMES]
    flat_df = pd.read_csv(flat)
    # zeta aligned with the label matrix (flat rows = frame-major molecule order)
    zeta = flat_df["zeta_all"].values[:N_FRAMES * N_MOL].reshape(N_FRAMES, N_MOL)

    zmean = {c: np.nanmean(zeta[lm == c]) for c in (0, 1)}
    lfts_id = max(zmean, key=zmean.get)
    print(f"  zeta means per matrix cluster: {zmean}  -> LFTS id = {lfts_id}")

    # 1. total O-O S(k) over all molecules
    oxy = [a.index for a in traj.topology.atoms if a.name == "O"]
    print(f"  total S(k): {len(oxy)} oxygens, {traj.n_frames} frames ...")
    S_avg, S_std, _ = compute_structure_factor(traj, np.array(oxy), K_VALUES, RC)
    np.savez(os.path.join(RC_CACHE, "total_sk_4p_T-20.npz"),
             k_values=K_VALUES, S_total=S_avg, S_total_std=S_std)
    print("  -> total_sk_4p_T-20.npz")

    # 2. S(k, zeta) landscape over all molecules
    zlo, zhi = np.percentile(zeta, 1), np.percentile(zeta, 99)
    zeta_bins = np.linspace(zlo, zhi, 36)
    all_one = np.zeros_like(lm)
    print(f"  S(k,zeta): zeta in [{zlo:.3f},{zhi:.3f}] nm, {len(zeta_bins)-1} bins ...")
    Skz, zcent = compute_sk_zeta_matrix(traj, K_VALUES, all_one, 0, zeta, zeta_bins, RC)
    np.savez(os.path.join(RC_CACHE, "skzeta_4p_T-20.npz"),
             k_values=K_VALUES, zeta_centers=zcent, S_k_zeta=Skz)
    print("  -> skzeta_4p_T-20.npz")

    # 3. real-space snapshot (one frame)
    res_oxy = {a.residue.index: a.index for a in traj.topology.atoms if a.name == "O"}
    mol_ids = np.array(sorted(res_oxy))
    atom_ids = np.array([res_oxy[m] for m in mol_ids])
    pos = traj.xyz[0, atom_ids, :]                   # (1024, 3) nm
    box = traj.unitcell_lengths[0]
    labels0 = lm[0]
    np.savez(os.path.join(RC_CACHE, "snapshot_4p_T-20.npz"),
             positions=pos, box=box, labels=labels0, lfts_id=lfts_id)
    print("  -> snapshot_4p_T-20.npz")
    print(f"\nDone in {time.time()-t0:.1f}s -> {RC_CACHE}")


# ═════════════════════════════════════════════════════════════════════════════
# blocktrim  —  LFTS/noise fraction vs T, block-trimmed over 3 runs
# ═════════════════════════════════════════════════════════════════════════════
def build_blocktrim():
    from water_clustering import scale_features, run_dbscan, run_gmm  # noqa
    op = os.path.join(_ROOT, "data", "order_params")
    op_rep = os.path.join(op, "errbar_runs")
    out = os.path.join(CLUST, "multirun_dbscan_gmm_realreps")
    os.makedirs(out, exist_ok=True)

    feat = ["q_all", "Q6_all", "LSI_all", "Sk_all", "zeta_all"]
    temps = [-30, -20, -10, 0, 10, 20, 30]
    runs = ["Run01", "Run02", "Run03"]
    block = 20            # frames per block (keep = published unit)
    n_blocks = 5          # blocks per run
    eps, min_samples, nc, seed = 0.06, 15, 2, 42

    def frames_df(t, run, frames):
        cand = os.path.join(op_rep, f"OrderParam_tip4p2005_T{t}_{run}.mat")
        d = op_rep if os.path.exists(cand) else op
        m = loadmat(os.path.join(d, f"OrderParam_tip4p2005_T{t}_{run}.mat"))
        z = loadmat(os.path.join(d, f"OrderParamZeta_tip4p2005_T{t}_{run}.mat"))
        cols = {k: np.concatenate([m[k][i] for i in frames]) for k in feat[:-1]}
        cols["zeta_all"] = np.concatenate([z["zeta_all"][i] for i in frames])
        return (pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)
                .dropna().reset_index(drop=True))

    def cluster(df):
        ds = scale_features(df[feat])
        with contextlib.redirect_stdout(io.StringIO()):
            db = run_dbscan(ds, eps=eps, min_samples=min_samples)
            clean = db != -1
            gl, _ = run_gmm(ds[clean], n_components=nc, random_state=seed)
        lab = np.full(len(ds), -1); lab[clean] = gl
        zeta = df["zeta_all"].values
        cids = [c for c in np.unique(lab) if c >= 0]
        lfts = max(cids, key=lambda c: zeta[lab == c].mean())
        n_l = int((lab == lfts).sum()); n_d = int(((lab >= 0) & (lab != lfts)).sum())
        n_n = int((lab == -1).sum()); tot = n_l + n_d + n_n
        return n_l / (n_l + n_d), n_n / tot

    def trim_stats(vals):
        v = np.sort(np.asarray(vals, float))
        core = v[1:-1]
        return core.mean(), core.std(ddof=1), len(core), len(v)

    per_rows, agg_rows = [], []
    for t in temps:
        s_vals, n_vals = [], []
        for run in runs:
            for b in range(n_blocks):
                fr = list(range(b * block, b * block + block))
                s, nz = cluster(frames_df(t, run, fr))
                s_vals.append(s); n_vals.append(nz)
                per_rows.append(dict(temp=t, run=run, block=b,
                                     frame_lo=fr[0], frame_hi=fr[-1],
                                     lfts_frac=s, noise_frac=nz))
        s_m, s_sd, kept, tot = trim_stats(s_vals)
        n_m, n_sd, _, _ = trim_stats(n_vals)
        agg_rows.append(dict(temp=t, lfts_mean=s_m, lfts_std=s_sd,
                             noise_mean=n_m, noise_std=n_sd,
                             n_used=kept, n_blocks=tot))
        print(f"T{t:+3d}: s(trim)={s_m:.4f}±{s_sd:.4f}  noise(trim)={n_m:.4f}±{n_sd:.4f} "
              f"| {kept}/{tot} blocks (s range {min(s_vals):.3f}-{max(s_vals):.3f})")
    pd.DataFrame(per_rows).to_csv(os.path.join(out, "fractions_per_block.csv"), index=False)
    pd.DataFrame(agg_rows).to_csv(os.path.join(out, "fractions_blocktrim.csv"), index=False)
    print("\nwrote -> fractions_blocktrim.csv, fractions_per_block.csv")


_STEPS = {
    "conf_sk": build_conf_sk,
    "redesign": build_redesign_caches,
    "blocktrim": build_blocktrim,
}


if __name__ == "__main__":
    which = sys.argv[1:] or ["all"]
    which = list(_STEPS) if which == ["all"] or "all" in which else which
    for name in which:
        print(f"--- {name} ---")
        _STEPS[name]()
    print("done.")
