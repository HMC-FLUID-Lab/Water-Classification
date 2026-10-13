"""
Oxygen-oxygen structure factors of classified groups (main text eqs 3, 4, 9, 10).

group_structure_factor   S_g(k): Debye sum over O-O pairs WITHIN group g in each
                         frame, with the sinc window W(r) = sin(pi r/rc)/(pi r/rc),
                         rc = 1.5 nm, normalised per molecule of the group.
sk_zeta                  S(k, zeta): per-molecule S_i(k) = 1 + sum_j sinc(k r_ij) W(r_ij)
                         over all neighbours j within rc, averaged in zeta bins over
                         the molecules of one group.
"""
import numpy as np
import mdtraj as md
from scipy.ndimage import gaussian_filter


def group_structure_factor(traj, rc, k_values, labels):
    """S_g(k) per frame for every group id >= 0 in `labels` [n_frames, n_molecules].
    Returns {group_id: array [n_frames, n_k]}."""
    oxygen = {a.residue.index: a.index for a in traj.topology.atoms if a.name == "O"}
    n_frames = min(traj.n_frames, labels.shape[0])
    out = {}
    for gid in sorted(int(c) for c in np.unique(labels) if c >= 0):
        S = np.zeros((n_frames, len(k_values)))
        for f in range(n_frames):
            idx = np.array([oxygen[m] for m in np.where(labels[f] == gid)[0] if m in oxygen])
            n = len(idx)
            if n < 2:
                S[f] = 1.0
                continue
            pairs = np.array([[idx[i], idx[j]] for i in range(n) for j in range(i + 1, n)])
            r = md.compute_distances(traj[f], pairs, periodic=True, opt=True)[0]
            for ki, k in enumerate(k_values):
                if k == 0:
                    S[f, ki] = 1.0
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    w = np.nan_to_num(np.sin(np.pi * r / rc) / (np.pi * r / rc), nan=0.0)
                    s = np.nan_to_num(np.sin(k * r) / (k * r), nan=1.0)
                S[f, ki] = 1.0 + (2.0 / n) * np.sum(s * w)
        out[gid] = S
    return out


def sk_zeta(traj, k_values, labels, group, zeta_A, zeta_bins, rc=1.5):
    """S(k, zeta) [n_zeta_bins, n_k] for one group (NaN where a bin is empty).
    zeta_A: [n_frames, n_molecules] in angstrom."""
    oxygen = {a.residue.index: a.index for a in traj.topology.atoms if a.name == "O"}
    centers = 0.5 * (zeta_bins[:-1] + zeta_bins[1:])
    S_sum = np.zeros((len(centers), len(k_values)))
    counts = np.zeros(len(centers))
    mols = np.array(sorted(oxygen))
    atoms = np.array([oxygen[m] for m in mols])
    n_frames = min(traj.n_frames, labels.shape[0], zeta_A.shape[0])
    for f in range(n_frames):
        frame = traj[f]
        box = frame.unitcell_lengths[0]
        pos = frame.xyz[0, atoms, :]
        members = np.where(labels[f] == group)[0]
        if len(members) < 2:
            continue
        for m in members:
            b = int(np.searchsorted(zeta_bins, float(zeta_A[f, m]), side="right")) - 1
            if not 0 <= b < len(centers):
                continue
            d = pos - pos[m]
            d -= box * np.round(d / box)
            r = np.linalg.norm(d, axis=1)
            r = r[(r > 1e-6) & (r < rc)]
            if len(r) == 0:
                continue
            x = np.pi * r / rc
            w = np.where(x > 1e-9, np.sin(x) / x, 1.0)
            kr = k_values[:, None] * r[None, :]
            s = np.where(kr > 1e-9, np.sin(kr) / kr, 1.0)
            S_sum[b] += 1.0 + np.sum(s * w[None, :], axis=1)
            counts[b] += 1
    S = np.full(S_sum.shape, np.nan)
    ok = counts > 0
    S[ok] = S_sum[ok] / counts[ok, None]
    return S, centers


def condition_surface(S, k_reduced, zeta_centers, k_range=(0.6, 2.0), s_range=(0.0, 2.0),
                      sigma=1.2):
    """Plotting conditioning of S(k, zeta): crop to k_range, drop empty zeta rows,
    fill isolated empty bins along zeta, clip, Gaussian-smooth (sigma in bins), clip."""
    S = np.copy(S)
    km = (k_reduced >= k_range[0]) & (k_reduced <= k_range[1])
    kp, S = k_reduced[km], S[:, km]
    rows = ~np.all(np.isnan(S), axis=1)
    if rows.any():
        a, b = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        S, zc = S[a:b], zeta_centers[a:b]
    else:
        zc = zeta_centers
    for j in range(S.shape[1]):
        col = S[:, j]
        nan = np.isnan(col)
        if nan.any() and not nan.all():
            i = np.arange(len(col))
            col[nan] = np.interp(i[nan], i[~nan], col[~nan])
            S[:, j] = col
    S = np.clip(gaussian_filter(np.clip(S, *s_range), sigma=sigma), *s_range)
    return kp, zc, S
