#!/usr/bin/env python3
"""
Stage 3b: TIP4P/2005 two-state populations versus temperature.

    python temperature_series.py

Writes to results/temperature_series/
  fractions_per_run.csv      LFTS / DNLS / transitional fractions of all molecules,
                             frozen classifier, first 100 frames of each run
  fractions_run_mean.csv     mean and SD over Run01-Run03            -> Fig. 4a
  fractions_per_block.csv    LFTS share of classified molecules and transitional
                             fraction per 20-frame block (5 blocks x 3 runs)
  fractions_block_trim.csv   mean +/- SD over the 13 blocks left after dropping the
                             highest and lowest                      -> Table S6, column 2
  constant_quality_per_run.csv, constant_quality.csv
                             constant-quality control, -30..+30 C   -> Table S6, columns 3-4
"""
import contextlib
import io
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import config  # noqa: E402
import classifier as C  # noqa: E402

MODEL = "TIP4P-2005"
RUNS = config.MODELS[MODEL]
BLOCK, N_BLOCKS = 20, 5
CODE = {"LFTS": 0, "DNLS": 1, "transitional": -1}

# constant-quality control: sweep the density-trim radius at fixed min_samples,
# fit GMM(2) on what remains, and hold the silhouette at a common target s*
CQ_TEMPS = [-30, -20, -10, 0, 10, 20, 30]
CQ_EPS = np.round(np.arange(0.02, 0.1601, 0.0075), 4)
CQ_MIN_SAMPLES = 15
CQ_FRAMES = 40
CQ_KEEP_FLOOR = 0.03          # ignore settings that keep < 3% of molecules
CQ_MIN_COMPONENT = 30         # each GMM component needs >= this many molecules
CQ_MARGIN = 0.005             # s* = (lowest per-condition maximum silhouette) - margin


def labels(temp, run, n_frames=config.N_FRAMES_RUN):
    lab = pd.read_csv(config.label_file(MODEL, temp, run))["label"].map(CODE).to_numpy()
    return lab[:n_frames * config.N_MOL]


def run_fractions():
    rows = []
    for t in config.TEMPS:
        for run in RUNS:
            lab = labels(t, run)
            rows.append(dict(temp=t, run=run, lfts=np.mean(lab == 0), dnls=np.mean(lab == 1),
                             trans=np.mean(lab == -1), n_mol=len(lab)))
    per = pd.DataFrame(rows)
    agg = per.groupby("temp").agg(
        lfts_mean=("lfts", "mean"), lfts_std=("lfts", "std"),
        dnls_mean=("dnls", "mean"), dnls_std=("dnls", "std"),
        trans_mean=("trans", "mean"), trans_std=("trans", "std"),
        n_runs=("run", "count")).reset_index()
    per.to_csv(config.results("temperature_series", "fractions_per_run.csv"), index=False)
    agg.to_csv(config.results("temperature_series", "fractions_run_mean.csv"), index=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def block_fractions():
    per, agg = [], []
    for t in config.TEMPS:
        share, trans = [], []
        for run in RUNS:
            lab = labels(t, run).reshape(-1, config.N_MOL)
            for b in range(N_BLOCKS):
                blk = lab[b * BLOCK:(b + 1) * BLOCK]
                n_h, n_l = (blk == 0).sum(), (blk == 1).sum()
                share.append(n_h / (n_h + n_l))
                trans.append(np.mean(blk == -1))
                per.append(dict(temp=t, run=run, block=b, lfts_share=share[-1], trans=trans[-1]))
        s, n = np.sort(share)[1:-1], np.sort(trans)[1:-1]          # drop the two extremes
        agg.append(dict(temp=t, lfts_share_mean=s.mean(), lfts_share_std=s.std(ddof=1),
                        trans_mean=n.mean(), trans_std=n.std(ddof=1),
                        n_used=len(s), n_blocks=len(share)))
    pd.DataFrame(per).to_csv(config.results("temperature_series", "fractions_per_block.csv"),
                             index=False)
    agg = pd.DataFrame(agg)
    agg.to_csv(config.results("temperature_series", "fractions_block_trim.csv"), index=False)
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _cq_sweep(temp, run):
    """(eps, set-aside, silhouette, LFTS of all) for each trim radius, eps ascending."""
    df = C.load_order_parameters(MODEL, temp, run, CQ_FRAMES)
    X = MinMaxScaler().fit_transform(df[config.FEATURES].to_numpy(float))
    zi = config.FEATURES.index("zeta")
    rows = []
    for eps in CQ_EPS:
        keep = DBSCAN(eps=float(eps), min_samples=CQ_MIN_SAMPLES).fit_predict(X) != -1
        if keep.mean() < CQ_KEEP_FLOOR:
            continue
        gm = GaussianMixture(2, covariance_type="full", random_state=config.SEED,
                             n_init=1).fit(X[keep])
        g = gm.predict(X[keep])
        if len(np.unique(g)) < 2 or np.bincount(g).min() < CQ_MIN_COMPONENT:
            continue
        sil = silhouette_score(X[keep], g, sample_size=min(4000, int(keep.sum())),
                               random_state=0)
        lfts = (g == int(np.argmax(gm.means_[:, zi]))).sum() / len(X)
        rows.append((float(eps), 1.0 - keep.mean(), float(sil), float(lfts)))
    return sorted(rows)


def _cq_operating_point(rows, s_star):
    """Set-aside and LFTS fractions where the silhouette crosses s* (linear interpolation
    in eps between the loosest radius meeting s* and the next one)."""
    reach = [r for r in rows if r[2] >= s_star]
    if not reach:
        r = max(rows, key=lambda z: z[2])
        return r[1], r[3], False
    lo = max(reach, key=lambda z: z[0])
    after = [r for r in rows if r[0] > lo[0]]
    if not after:
        return lo[1], lo[3], True
    hi = min(after, key=lambda z: z[0])
    f = (lo[2] - s_star) / (lo[2] - hi[2]) if lo[2] != hi[2] else 0.0
    return lo[1] + f * (hi[1] - lo[1]), lo[3] + f * (hi[3] - lo[3]), True


def constant_quality():
    keys = [(t, r) for t in CQ_TEMPS for r in RUNS]
    with contextlib.redirect_stdout(io.StringIO()):
        sweeps = dict(zip(keys, Parallel(n_jobs=len(keys))(delayed(_cq_sweep)(*k) for k in keys)))
    s_star = min(max(s[2] for s in rows) for rows in sweeps.values()) - CQ_MARGIN
    per, agg = [], []
    for t in CQ_TEMPS:
        pts = [_cq_operating_point(sweeps[t, r], s_star) for r in RUNS]
        for r, (noise, lfts, ok) in zip(RUNS, pts):
            per.append(dict(temp=t, run=r, set_aside=noise, lfts=lfts, reached=ok))
        noise, lfts = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        agg.append(dict(temp=t, set_aside_mean=noise.mean(), set_aside_std=noise.std(ddof=1),
                        lfts_mean=lfts.mean(), lfts_std=lfts.std(ddof=1), s_star=s_star))
    pd.DataFrame(per).to_csv(config.results("temperature_series", "constant_quality_per_run.csv"),
                             index=False)
    agg = pd.DataFrame(agg)
    agg.to_csv(config.results("temperature_series", "constant_quality.csv"), index=False)
    print(f"s* = {s_star:.3f}")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    print("--- per-run fractions (Fig. 4a) ---"); run_fractions()
    print("--- 20-frame blocks (Table S6) ---"); block_fractions()
    print("--- constant quality (Table S6) ---"); constant_quality()
