#!/usr/bin/env python3
"""
Stage 4: structure factors of the classified groups.

    python compute_structure_factors.py [--jobs N]

For every force field and temperature, the first 20 frames of the main run
(TIP4P/2005 Run01, TIP5P Run01, SWM4-NDP Run02) with their frozen-classifier
labels give S_g(k) for the LFTS, DNLS and transitional groups:
    results/structure_factors/<FORCE FIELD>_T<T>C_<run>_Sk.npz
        k (nm^-1), k_reduced (= k r_OO / 2 pi), S_<group> [n_frames, n_k], n_<group> [n_frames]
At the reference (TIP4P/2005, -20 C) also the zeta-resolved S(k, zeta):
    results/structure_factors/TIP4P-2005_T-20C_Run01_Sk_zeta.npz
        raw: S_raw_<group>, zeta_centers_raw; conditioned for plotting: kp_, zc_, S_<group>
"""
import argparse
import os
import sys

import mdtraj as md
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import config  # noqa: E402
from structure_factor import group_structure_factor, sk_zeta, condition_surface  # noqa: E402

GROUPS = {"LFTS": 0, "DNLS": 1, "transitional": 2}       # transitional (-1) -> its own id
CODE = {"LFTS": 0, "DNLS": 1, "transitional": 2}


def load(model, temp, run, n=config.N_FRAMES):
    dcd, pdb = config.trajectory(model, temp, run)
    traj = md.load(dcd, top=md.load(pdb).topology)[:n]
    lab = pd.read_csv(config.label_file(model, temp, run))["label"].map(CODE).to_numpy()
    return traj, lab[:n * config.N_MOL].reshape(n, config.N_MOL)


def group_sk(model, temp, run):
    out = config.results("structure_factors", config.tag(model, temp, run) + "_Sk.npz")
    if os.path.isfile(out):
        return out
    traj, lab = load(model, temp, run)
    k = np.linspace(*config.K_VALUES_NM)
    S = group_structure_factor(traj, config.RC, k, lab)
    np.savez(out, k=k, k_reduced=k * config.R_OO / (2 * np.pi),
             **{f"S_{g}": S[i] for g, i in GROUPS.items()},
             **{f"n_{g}": (lab == i).sum(axis=1) for g, i in GROUPS.items()})
    print(f"  wrote {out}", flush=True)
    return out


def reference_sk_zeta():
    model, temp, run = config.REFERENCE
    out = config.results("structure_factors", config.tag(model, temp, run) + "_Sk_zeta.npz")
    if os.path.isfile(out):
        return
    traj, lab = load(model, temp, run)
    with np.load(config.order_parameter_file(model, temp, run)) as d:
        zeta_A = d["zeta"][:config.N_FRAMES] * 10.0               # nm -> angstrom
    k = np.linspace(*config.K_VALUES_ZETA_NM)
    kr = k * config.R_OO / (2 * np.pi)
    bins = np.linspace(*config.ZETA_BINS_A)

    def one(name):
        S, zc = sk_zeta(traj, k, lab, GROUPS[name], zeta_A, bins, config.RC)
        return name, S, zc
    res = Parallel(n_jobs=2)(delayed(one)(n) for n in ("LFTS", "DNLS"))
    data = dict(k=k, k_reduced=kr, zeta_centers_raw=res[0][2])
    for name, S, zc in res:
        kp, zcp, Sp = condition_surface(S, kr, zc)
        data.update({f"S_raw_{name}": S, f"kp_{name}": kp, f"zc_{name}": zcp, f"S_{name}": Sp})
    np.savez(out, **data)
    print(f"  wrote {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()
    todo = [(m, t, config.MAIN_RUN[m]) for m in config.MODELS for t in config.TEMPS]
    Parallel(n_jobs=args.jobs)(delayed(group_sk)(*c) for c in todo)
    reference_sk_zeta()


if __name__ == "__main__":
    main()
