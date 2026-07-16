#!/usr/bin/env python3
"""
make_confidence_labels.py
=========================
Regenerate the tip4p2005, -20 C cluster labels with the *confidence-based*
two-state assignment (`run_dbscan_gmm_confidence`), clustering on the FOUR
paper features (zeta, q, LSI, S_k) — Q6 dropped.

Reads the existing 5-feature CSV (features already present, so no .mat needed),
writes a new CSV in a separate directory so the figure loaders pick up exactly
one label column:

  results/clustering/tip4p2005_T-20_dbscan_gmm_conf/cluster_labels.csv
      columns: q_all, Q6_all, LSI_all, Sk_all, zeta_all,
               label_dbscan_gmm_conf, pmax_dbscan_gmm_conf

Also prints a sensitivity sweep of the transition fraction vs the confidence
threshold alpha.
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from water_clustering import scale_features, run_dbscan, run_dbscan_gmm_confidence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# four paper features (Q6 deliberately excluded)
FEATURES = ["zeta_all", "q_all", "LSI_all", "Sk_all"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default=os.path.join(
        _ROOT, "results/clustering/tip4p2005_T-20_dbscan_gmm/cluster_labels.csv"))
    ap.add_argument("--out_dir", default=os.path.join(
        _ROOT, "results/clustering/tip4p2005_T-20_dbscan_gmm_conf"))
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--min_samples", type=int, default=20)
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    df_raw = pd.read_csv(args.in_csv)
    print(f"Loaded {len(df_raw):,} molecules from {args.in_csv}")
    print(f"Clustering features (Q6 dropped): {FEATURES}")

    df_feat = df_raw[FEATURES].copy()
    good = np.isfinite(df_feat.to_numpy()).all(axis=1)
    df_feat = df_feat[good].reset_index(drop=True)
    df_raw = df_raw[good].reset_index(drop=True)
    df_scaled = scale_features(df_feat)

    print("─" * 60)
    labels, pmax = run_dbscan_gmm_confidence(
        df_scaled, eps=args.eps, min_samples=args.min_samples,
        n_components=2, alpha=args.alpha, random_state=args.random_state)

    # ── sensitivity sweep: reuse the same DBSCAN-seeded GMM posteriors ──
    print("─" * 60)
    print("Sensitivity of the transition band to alpha")
    db = run_dbscan(df_scaled, eps=args.eps, min_samples=args.min_samples)
    gm = GaussianMixture(n_components=2, covariance_type="full", n_init=5,
                         random_state=args.random_state)
    gm.fit(df_scaled[db != -1])
    probs = gm.predict_proba(df_scaled)
    argmax = probs.argmax(axis=1)
    pm = probs.max(axis=1)
    zmean = [df_scaled["zeta_all"][argmax == c].mean() for c in (0, 1)]
    lfts = int(np.argmax(zmean))
    n = len(pm)
    print(f"  {'alpha':>6} {'transition%':>12} {'LFTS%':>8} {'DNLS%':>8} {'LFTS:DNLS':>10}")
    for a in (0.55, 0.60, 0.65, 0.70, 0.75):
        trans = pm < a
        n_l = int(((argmax == lfts) & ~trans).sum())
        n_d = int(((argmax != lfts) & ~trans).sum())
        ratio = n_l / n_d if n_d else float("nan")
        print(f"  {a:>6.2f} {100*trans.sum()/n:>11.1f}% {100*n_l/n:>7.1f}% "
              f"{100*n_d/n:>7.1f}% {ratio:>10.2f}")

    # ── write the CSV the figures consume ──
    os.makedirs(args.out_dir, exist_ok=True)
    out = df_raw.copy()
    if "label_dbscan_gmm" in out.columns:
        out = out.drop(columns=["label_dbscan_gmm"])
    out["label_dbscan_gmm_conf"] = labels
    out["pmax_dbscan_gmm_conf"] = pmax
    out_csv = os.path.join(args.out_dir, "cluster_labels.csv")
    out.to_csv(out_csv, index=False)
    print("─" * 60)
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
