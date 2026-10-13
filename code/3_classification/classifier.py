"""
The unsupervised likelihood-ratio classifier and the benchmark pipelines.

Likelihood-ratio classifier (main text, eqs 1-2; Fig. 2)
  1. Scale the four order parameters (zeta, q, LSI, s_k) to [0, 1] with a
     MinMax scaler fitted on the reference condition (TIP4P/2005, -20 C, Run01,
     first 20 frames = 20,480 molecules).
  2. Density trim: molecules that are not core points of
     sklearn.cluster.DBSCAN(eps=0.06, min_samples=20) in the scaled space (about
     5% of the reference) are left out of the mixture fit only.
  3. Fit a two-component full-covariance Gaussian mixture (n_init=5,
     random_state=42). The component with the larger mean zeta is H (LFTS),
     the other is L (DNLS).
  4. Lambda(x) = log p_H(x) - log p_L(x) (component densities; mixture weights
     not included). LFTS if Lambda >= tau, DNLS if Lambda <= -tau, transitional
     otherwise, with tau = 2.
The scaler, both Gaussian components and tau are fitted once and then FROZEN:
every other temperature and force field is classified with the same parameters.

Label codes: 0 = LFTS (H), 1 = DNLS (L), -1 = transitional / set aside.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, multivariate_normal
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# data access
# ─────────────────────────────────────────────────────────────────────────────
def load_order_parameters(model, temp, run, n_frames=None):
    """Per-molecule order parameters as a frame-major DataFrame
    (row = frame * 1024 + molecule)."""
    with np.load(config.order_parameter_file(model, temp, run)) as d:
        n = d["q"].shape[0] if n_frames is None else n_frames
        assert d["q"].shape[0] >= n, f"{model} {temp} {run}: only {d['q'].shape[0]} frames"
        return pd.DataFrame({k: d[k][:n].ravel() for k in d.files})


def reference_features():
    model, temp, run = config.REFERENCE
    return load_order_parameters(model, temp, run, config.N_FRAMES)


# ─────────────────────────────────────────────────────────────────────────────
# likelihood-ratio classifier
# ─────────────────────────────────────────────────────────────────────────────
def fit_classifier(ref):
    """Fit the classifier on the reference DataFrame and return its frozen parameters."""
    raw = ref[config.FEATURES].to_numpy(float)
    scaler = MinMaxScaler().fit(raw)
    X = scaler.transform(raw)
    core = DBSCAN(eps=config.TRIM_EPS, min_samples=config.TRIM_MIN_SAMPLES).fit_predict(X) != -1
    gmm = GaussianMixture(2, covariance_type="full", n_init=5,
                          random_state=config.SEED).fit(X[core])
    h = int(np.argmax(gmm.means_[:, config.FEATURES.index("zeta")]))
    l = 1 - h
    return dict(features=config.FEATURES, tau=config.TAU,
                scaler_min=scaler.data_min_.tolist(), scaler_max=scaler.data_max_.tolist(),
                mean_H=gmm.means_[h].tolist(), mean_L=gmm.means_[l].tolist(),
                cov_H=gmm.covariances_[h].tolist(), cov_L=gmm.covariances_[l].tolist(),
                weight_H=float(gmm.weights_[h]), weight_L=float(gmm.weights_[l]),
                trimmed_fraction=float(1 - core.mean()))


def scale(df, clf):
    """MinMax transform with the frozen reference ranges (same arithmetic as
    sklearn.preprocessing.MinMaxScaler.transform)."""
    lo, hi = np.asarray(clf["scaler_min"]), np.asarray(clf["scaler_max"])
    s = 1.0 / (hi - lo)
    X = df[clf["features"]].to_numpy(float) * s
    X += -lo * s
    return X


def log_likelihood_ratio(df, clf):
    X = scale(df, clf)
    pH = multivariate_normal(clf["mean_H"], clf["cov_H"], allow_singular=True)
    pL = multivariate_normal(clf["mean_L"], clf["cov_L"], allow_singular=True)
    return pH.logpdf(X) - pL.logpdf(X)


def classify(df, clf):
    """(labels, Lambda) with labels 0 = LFTS, 1 = DNLS, -1 = transitional."""
    lam = log_likelihood_ratio(df, clf)
    tau = clf["tau"]
    return np.where(lam >= tau, 0, np.where(lam <= -tau, 1, -1)), lam


CLASSIFIER_FILE = os.path.join(config.RESULTS, "classifier", "frozen_classifier.json")


def save_classifier(clf, path=CLASSIFIER_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(clf, f, indent=2)


def load_classifier(path=CLASSIFIER_FILE):
    if not os.path.isfile(path):                      # fit on demand (deterministic)
        save_classifier(fit_classifier(reference_features()), path)
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# benchmark pipelines at the reference condition (SI Table S4, Figs. S1-S5)
# Five features (q, Q6, LSI, s_k, zeta), each MinMax-scaled on the reference.
# ─────────────────────────────────────────────────────────────────────────────
def _h_first(labels, zeta):
    """Relabel a two-group partition so that 0 = higher mean zeta (H), 1 = L."""
    ids = [c for c in np.unique(labels) if c >= 0]
    h = max(ids, key=lambda c: zeta[labels == c].mean())
    out = np.full(len(labels), -1)
    out[labels == h] = 0
    out[(labels >= 0) & (labels != h)] = 1
    return out


def benchmark_kmeans(X5):
    return KMeans(n_clusters=2, random_state=config.SEED, init="k-means++",
                  n_init=10).fit_predict(X5)


def benchmark_gmm(X5):
    return GaussianMixture(2, random_state=config.SEED, covariance_type="full",
                           n_init=5).fit(X5).predict(X5)


def benchmark_hdbscan_gmm(X5, min_cluster_size=8, min_samples=8):
    """HDBSCAN removes a density-defined subset, then GMM(2) on the remainder."""
    keep = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                   cluster_selection_method="eom", n_jobs=-1).fit_predict(X5) != -1
    out = np.full(len(X5), -1)
    out[keep] = benchmark_gmm(X5[keep])
    return out


def _trimmed_gmm(X4):
    core = DBSCAN(eps=config.TRIM_EPS, min_samples=config.TRIM_MIN_SAMPLES).fit_predict(X4) != -1
    return GaussianMixture(2, covariance_type="full", n_init=5,
                           random_state=config.SEED).fit(X4[core])


def posterior_cut(X4, alpha=0.95):
    """Set aside molecules whose largest GMM posterior is below alpha (SI Table S5)."""
    p = _trimmed_gmm(X4).predict_proba(X4)
    lab = p.argmax(axis=1)
    lab[p.max(axis=1) < alpha] = -1
    return lab


def fisher_valley_cut(X4, valley_frac=0.95):
    """Set aside molecules in the density valley along the Fisher axis (SI Table S5)."""
    gm = _trimmed_gmm(X4)
    mu, cov = gm.means_, gm.covariances_
    w = np.linalg.solve(0.5 * (cov[0] + cov[1]), mu[1] - mu[0])
    w /= np.linalg.norm(w)
    t = X4 @ w
    t0, t1 = float(mu[0] @ w), float(mu[1] @ w)
    lo, hi = min(t0, t1), max(t0, t1)
    kde = gaussian_kde(t)
    grid = np.linspace(lo, hi, 240)
    tstar = float(grid[int(np.argmin(kde(grid)))])
    peak = float(min(kde(np.array([t0]))[0], kde(np.array([t1]))[0]))
    lab = (t > tstar).astype(int)
    lab[(t >= lo) & (t <= hi) & (kde(t) < valley_frac * peak)] = -1
    return lab


def benchmark_labels(ref):
    """All benchmark partitions at the reference, relabelled 0 = H, 1 = L, -1 = set aside."""
    X5 = MinMaxScaler().fit_transform(ref[config.FEATURES_5].to_numpy(float))
    X4 = MinMaxScaler().fit_transform(ref[config.FEATURES].to_numpy(float))
    zeta = ref["zeta"].to_numpy()
    return {
        "kmeans": _h_first(benchmark_kmeans(X5), zeta),
        "gmm": _h_first(benchmark_gmm(X5), zeta),
        "hdbscan_gmm": _h_first(benchmark_hdbscan_gmm(X5), zeta),
        "posterior_cut": _h_first(posterior_cut(X4), zeta),
        "fisher_valley_cut": _h_first(fisher_valley_cut(X4), zeta),
    }
