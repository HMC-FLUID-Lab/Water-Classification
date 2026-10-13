#!/usr/bin/env python3
"""
Stage 3a: fit the likelihood-ratio classifier once at the reference condition,
freeze it, and label every molecule of every analyzed condition.

    python make_labels.py

Writes
  results/classifier/frozen_classifier.json
      scaler ranges, Gaussian means/covariances/weights (SI Table S3), tau
  data/<FORCE FIELD>/classification_labels/<FORCE FIELD>_T<T>C_<run>_labels.csv
      frame, molecule, Lambda, label     (label: LFTS, DNLS or transitional)
      one row per molecule and frame of the order-parameter file, same order
  data/TIP4P-2005/classification_labels/TIP4P-2005_T-20C_Run01_benchmark_labels.csv
      frame, molecule and the SI benchmark partitions at the reference
      (kmeans, gmm, hdbscan_gmm, posterior_cut, fisher_valley_cut; H/L/set_aside)
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import config  # noqa: E402
import classifier as C  # noqa: E402


def frame_molecule(n_rows):
    idx = np.arange(n_rows)
    return idx // config.N_MOL, idx % config.N_MOL


def main():
    ref = C.reference_features()
    clf = C.fit_classifier(ref)
    C.save_classifier(clf)
    print(f"classifier fitted at {config.REFERENCE}: trim set aside "
          f"{100 * clf['trimmed_fraction']:.1f}%, weights H={clf['weight_H']:.3f} "
          f"L={clf['weight_L']:.3f} -> {C.CLASSIFIER_FILE}")
    clf = C.load_classifier()                               # use the stored (frozen) copy

    names = np.array(["transitional", "LFTS", "DNLS"])      # index = code + 1
    for model, runs in config.MODELS.items():
        for temp in config.TEMPS:
            for run in runs:
                df = C.load_order_parameters(model, temp, run)
                lab, lam = C.classify(df, clf)
                frame, mol = frame_molecule(len(df))
                out = config.label_file(model, temp, run)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                pd.DataFrame({"frame": frame, "molecule": mol, "Lambda": lam,
                              "label": names[lab + 1]}).to_csv(out, index=False,
                                                               float_format="%.6f")
                f = {n: np.mean(lab == c) for c, n in config.LABEL_NAMES.items()}
                print(f"  {config.tag(model, temp, run):<24s} {len(df) // config.N_MOL:4d} frames  "
                      f"LFTS {f['LFTS']:.3f}  DNLS {f['DNLS']:.3f}  "
                      f"transitional {f['transitional']:.3f}")

    bench = C.benchmark_labels(ref)
    frame, mol = frame_molecule(len(ref))
    named = np.array(["set_aside", "H", "L"])
    out = config.label_file(*config.REFERENCE).replace("_labels.csv", "_benchmark_labels.csv")
    pd.DataFrame({"frame": frame, "molecule": mol,
                  **{k: named[v + 1] for k, v in bench.items()}}).to_csv(out, index=False)
    for k, v in bench.items():
        print(f"  benchmark {k:<18s} set aside {100 * np.mean(v == -1):5.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
