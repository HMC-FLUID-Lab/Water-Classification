#!/usr/bin/env python3
"""
Stage 2 (vectorised): q, LSI, s_k and zeta for the first frames of a trajectory.

    python fast_order_parameters.py <force field> <temperature in C> <run> [n_frames=20]
    python fast_order_parameters.py TIP5P 40 Run01

Same definitions as compute_order_parameters.py (batched over molecules; no Q6).
Used for TIP5P and SWM4-NDP at every temperature except -20 C; checked against
the reference implementation at -20 C for both models (max. deviation < 1e-6,
identical classification labels). Writes the same .npz format.
"""
import os
import sys
from itertools import combinations

import numpy as np
import mdtraj as md

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

NMOL = config.N_MOL


def descriptors(traj):
    """Stage-2 q, s_k, LSI and nine-neighbor hydrogen-bond zeta, batched.

    Keep mdtraj's periodic float32 distances/angles, including the original
    double-precision accumulations for q and s_k. No descriptor is redefined.
    """
    NF = traj.n_frames
    waters = [list(res.atoms) for res in traj.topology.residues]
    assert len(waters) == NMOL
    assert all([atom.name for atom in atoms[:3]] == ["O", "H1", "H2"]
               for atoms in waters)
    oxygen = np.array([atoms[0].index for atoms in waters])
    oh_pairs = np.array([(atoms[0].index, atoms[h].index)
                         for atoms in waters for h in (1, 2)])
    ii, jj = np.triu_indices(NMOL, 1)
    pairs = np.column_stack((oxygen[ii], oxygen[jj]))
    out = {key: np.empty((NF, NMOL)) for key in ("zeta_all", "q_all", "LSI_all", "Sk_all")}
    angle_combinations = list(combinations(range(4), 2))
    for frame in range(NF):
        water = traj[frame]
        distances = np.full((NMOL, NMOL), np.inf, dtype=np.float32)
        upper = md.compute_distances(water, pairs, periodic=True)[0]
        distances[ii, jj] = upper
        distances[jj, ii] = upper
        neighbor_order = distances.argsort(axis=1)
        ordered = np.take_along_axis(distances, neighbor_order, axis=1)[:, :-1]
        nearest = neighbor_order[:, :9]
        near_dist = ordered[:, :9]

        count = np.sum(ordered < 0.37, axis=1)
        assert (count > 0).all()
        delta = np.diff(ordered, axis=1)
        mask = ordered[:, :-1] < 0.37
        delta_mean = np.sum(delta * mask, axis=1) / count
        out["LSI_all"][frame] = np.sum(
            ((delta - delta_mean[:, None]) * mask)**2, axis=1) / count

        cosines = np.empty((NMOL, 6), dtype=np.float64)
        for column, (j, k) in enumerate(angle_combinations):
            neighbor_pairs = np.column_stack((oxygen[nearest[:, j]], oxygen[nearest[:, k]]))
            a = md.compute_distances(water, neighbor_pairs, periodic=True)[0]
            b, c = near_dist[:, j], near_dist[:, k]
            cosines[:, column] = (b**2 + c**2 - a**2) / (2 * b * c)
        out["q_all"][frame] = 1 - 3 / 8 * np.sum((cosines + 1 / 3)**2, axis=1)

        four_dist = near_dist[:, :4].astype(np.float64)
        r_mean = four_dist.mean(axis=1, keepdims=True)
        out["Sk_all"][frame] = 1 - np.sum(
            (four_dist - r_mean)**2 / (4 * r_mean**2), axis=1) / 3

        oo_pairs = np.column_stack((np.repeat(oxygen, 9), oxygen[nearest.ravel()]))
        oo = md.compute_displacements(water, oo_pairs, periodic=True)[0].reshape(NMOL, 9, 3)
        oh = md.compute_displacements(water, oh_pairs, periodic=True)[0].reshape(NMOL, 2, 3)
        oh_distance = md.compute_distances(water, oh_pairs, periodic=True)[0].reshape(NMOL, 2)
        donor_cos = np.einsum("nmc,nhc->nmh", oo, oh) / (
            near_dist[:, :, None] * oh_distance[:, None, :])
        acceptor_cos = -np.einsum("nmc,nmhc->nmh", oo, oh[nearest]) / (
            near_dist[:, :, None] * oh_distance[nearest])
        hydrogen_bond = (near_dist < 0.35) & (
            np.any(donor_cos > np.cos(np.pi / 6), axis=2)
            | np.any(acceptor_cos > np.cos(np.pi / 6), axis=2))
        assert np.any(~hydrogen_bond, axis=1).all()
        nearest_nonbond = np.min(np.where(~hydrogen_bond, near_dist, np.inf), axis=1)
        farthest_bond = np.max(np.where(hydrogen_bond, near_dist, 0), axis=1)
        out["zeta_all"][frame] = nearest_nonbond - farthest_bond
    return out


def main():
    if len(sys.argv) not in (4, 5):
        print(__doc__)
        sys.exit(1)
    model, temp, run = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    n_frames = int(sys.argv[4]) if len(sys.argv) == 5 else config.N_FRAMES
    dcd, pdb = config.trajectory(model, temp, run)
    traj = next(md.iterload(dcd, top=pdb, chunk=n_frames))
    out = descriptors(traj)
    path = config.order_parameter_file(model, temp, run)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, q=out["q_all"], LSI=out["LSI_all"], s_k=out["Sk_all"],
                        zeta=out["zeta_all"])
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
