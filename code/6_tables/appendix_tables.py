#!/usr/bin/env python3
"""
Stage 6: the numbers in the Supporting-Information tables, as CSV files in tables/.

    python appendix_tables.py

  appendix_table2_order_parameters.csv  ranges and Cohen's d of q, LSI, s_k, zeta (H vs L)
  appendix_table3_mixture_means.csv     frozen Gaussian components: means (scaled, raw), weights
  appendix_table3_mixture_covariances.csv  covariances (scaled) x 1e3, order (q, LSI, s_k, zeta)
  appendix_table3_scaler_ranges.csv     MinMax reference ranges
  appendix_table4_pipelines.csv         set-aside fraction and silhouette of each pipeline
  appendix_table5_density.csv           local 4-D density of the set-aside molecules
  appendix_table6_temperature.csv       frozen-classifier LFTS share and constant-quality series
  appendix_table7_model_transfer.csv    three water models at -20 C
  appendix_table8_model_temperature.csv TIP5P and SWM4-NDP fractions vs temperature
(Table S1 lists simulation settings only; see the main README.)

Requires stages 3a (make_labels.py) and 3b (temperature_series.py).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, _CODE)
sys.path.insert(0, os.path.join(_CODE, "3_classification"))
import config  # noqa: E402
import classifier as C  # noqa: E402

ORDER = ["q", "LSI", "s_k", "zeta"]
CODE = {"LFTS": 0, "DNLS": 1, "transitional": -1}
BENCH = {"H": 0, "L": 1, "set_aside": -1}


def labels(model, temp, run, n_frames=None):
    lab = pd.read_csv(config.label_file(model, temp, run))["label"].map(CODE).to_numpy()
    return lab if n_frames is None else lab[:n_frames * config.N_MOL]


def reference():
    df = C.reference_features()
    df["label"] = labels(*config.REFERENCE, config.N_FRAMES)
    b = pd.read_csv(config.label_file(*config.REFERENCE).replace("_labels.csv",
                                                                  "_benchmark_labels.csv"))
    for col in b.columns[2:]:
        df[col] = b[col].map(BENCH).to_numpy()
    return df


def table2(df):
    rows = []
    for c in ["zeta", "LSI", "s_k", "q"]:
        v = df[c].to_numpy(float)
        h, l = v[df["label"] == 0], v[df["label"] == 1]
        d = (h.mean() - l.mean()) / np.sqrt((h.var(ddof=1) + l.var(ddof=1)) / 2)
        p1, p99 = np.percentile(v, [1, 99])
        rows.append(dict(descriptor=c, p1=p1, p99=p99, min=v.min(), max=v.max(), cohens_d=d))
    return pd.DataFrame(rows)


def table3():
    clf = C.load_classifier()
    idx = [clf["features"].index(c) for c in ORDER]
    lo, hi = np.asarray(clf["scaler_min"]), np.asarray(clf["scaler_max"])
    mH, mL = np.asarray(clf["mean_H"]), np.asarray(clf["mean_L"])
    means = pd.DataFrame({"descriptor": ORDER, "H_scaled": mH[idx], "L_scaled": mL[idx],
                          "H_raw": (lo + mH * (hi - lo))[idx], "L_raw": (lo + mL * (hi - lo))[idx]})
    means.loc[len(means)] = ["weight", clf["weight_H"], clf["weight_L"], np.nan, np.nan]
    cov = []
    for name in ("H", "L"):
        M = np.asarray(clf[f"cov_{name}"])[np.ix_(idx, idx)] * 1e3
        for i, c in enumerate(ORDER):
            cov.append(dict(component=name, row=c, **{ORDER[j]: M[i, j] for j in range(4)}))
    ranges = pd.DataFrame({"descriptor": ORDER, "min": lo[idx], "max": hi[idx]})
    return means, pd.DataFrame(cov), ranges


def table4(df):
    rows = []
    for name, col, feats in (("K-means", "kmeans", config.FEATURES_5),
                             ("GMM", "gmm", config.FEATURES_5),
                             ("HDBSCAN->GMM", "hdbscan_gmm", config.FEATURES_5),
                             ("likelihood ratio", "label", config.FEATURES)):
        X = MinMaxScaler().fit_transform(df[feats].to_numpy(float))
        lab = df[col].to_numpy()
        k = lab >= 0
        rows.append(dict(pipeline=name, n_features=len(feats), set_aside=1 - k.mean(),
                         silhouette=silhouette_score(X[k], lab[k])))
    return pd.DataFrame(rows)


def table5(df):
    """Neighbour count within r = 0.06 (scaled 4-D space) for the molecules set aside by the
    Fisher density-valley cut and the posterior-confidence cut, against the retained ones."""
    X = MinMaxScaler().fit_transform(df[config.FEATURES].to_numpy(float))
    nn = NearestNeighbors(radius=config.TRIM_EPS).fit(X)
    dens = np.array([len(a) - 1 for a in nn.radius_neighbors(X, return_distance=False)], float)
    rows = []
    for name, col in (("Fisher density-valley cut", "fisher_valley_cut"),
                      ("posterior-confidence cut", "posterior_cut")):
        out = df[col].to_numpy() == -1
        d_out, d_in = dens[out], dens[~out]
        n = int(out.sum())
        lowest = np.zeros(len(dens), bool)
        lowest[np.argsort(dens)[:n]] = True
        rows.append(dict(
            method=name, n_set_aside=n, pct_set_aside=100 * out.mean(),
            median_density_set_aside=np.median(d_out), median_density_retained=np.median(d_in),
            ratio=np.median(d_out) / np.median(d_in),
            pct_denser_than_retained_median=100 * np.mean(d_out > np.median(d_in)),
            pct_overlap_with_lowest_density_set=100 * (out & lowest).sum() / max(n, 1),
            pct_in_lowest_density_decile=100 * np.mean(d_out <= np.quantile(dens, 0.10))))
    return pd.DataFrame(rows)


def table6():
    ts = os.path.join(config.RESULTS, "temperature_series")
    fz = pd.read_csv(os.path.join(ts, "fractions_block_trim.csv"))
    cq = pd.read_csv(os.path.join(ts, "constant_quality.csv"))
    fz = fz[["temp", "lfts_share_mean", "lfts_share_std", "n_used", "n_blocks"]].rename(
        columns={"lfts_share_mean": "frozen_lfts_share_mean", "lfts_share_std": "frozen_lfts_share_std"})
    cq = cq.rename(columns={"set_aside_mean": "cq_set_aside_mean", "set_aside_std": "cq_set_aside_std",
                            "lfts_mean": "cq_lfts_mean", "lfts_std": "cq_lfts_std"})
    return fz.merge(cq, on="temp", how="left")


def fractions(model, temp):
    lab = labels(model, temp, config.MAIN_RUN[model], config.N_FRAMES)
    return {n: np.mean(lab == c) for n, c in CODE.items()}


def table7():
    return pd.DataFrame([dict(model=m, run=config.MAIN_RUN[m], frames=config.N_FRAMES,
                              **fractions(m, -20)) for m in config.MODELS])


def table8():
    rows = []
    for t in config.TEMPS:
        row = {"temp": t}
        for m in ("TIP5P", "SWM4-NDP"):
            row.update({f"{m}_{k}": v for k, v in fractions(m, t).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    os.makedirs(config.TABLES, exist_ok=True)
    df = reference()
    means, cov, ranges = table3()
    tables = {
        "appendix_table2_order_parameters": table2(df),
        "appendix_table3_mixture_means": means,
        "appendix_table3_mixture_covariances": cov,
        "appendix_table3_scaler_ranges": ranges,
        "appendix_table4_pipelines": table4(df),
        "appendix_table5_density": table5(df),
        "appendix_table6_temperature": table6(),
        "appendix_table7_model_transfer": table7(),
        "appendix_table8_model_temperature": table8(),
    }
    pd.set_option("display.width", 220)
    for name, t in tables.items():
        t.to_csv(os.path.join(config.TABLES, name + ".csv"), index=False)
        print(f"\n=== {name} ===")
        print(t.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print(f"\nwrote tables -> {config.TABLES}")


if __name__ == "__main__":
    main()
