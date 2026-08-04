"""Phase 2: intrinsic-dimension estimators.

Three independent estimators, each implemented natively on top of
numpy/scipy/scikit-learn primitives (no black-box dimension-estimation
packages): the Grassberger-Procaccia correlation dimension, a PCA-based
eigenvalue-spectrum estimator, and the Levina-Bickel k-NN MLE estimator.
"""

from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import linregress
from sklearn.neighbors import NearestNeighbors

from src import config


def correlation_integral(X: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Compute the Grassberger-Procaccia correlation integral C(r).

    C(r) = (2 / (N(N-1))) * sum_{i<j} I(||x_i - x_j|| < r)

    Implementation notes
    ---------------------
    The condensed pairwise-distance vector from `scipy.spatial.distance.pdist`
    already enumerates exactly the N(N-1)/2 unordered pairs i<j, so C(r) is
    simply the fraction of that vector below each radius. Distances are
    sorted once and each radius is located via binary search
    (`np.searchsorted`), which is far cheaper than re-thresholding the full
    distance vector once per radius. At N=801 the condensed vector has
    320,400 entries (~2.6 MB in float64), which is memory-trivial.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    radii : np.ndarray of shape (n_radii,)

    Returns
    -------
    np.ndarray of shape (n_radii,)
        C(r) for each radius in `radii`.
    """
    distances = np.sort(pdist(X, metric="euclidean"))
    n_pairs = distances.shape[0]
    counts = np.searchsorted(distances, radii, side="left")
    return counts / n_pairs


def default_radii(X: np.ndarray, seed: int) -> np.ndarray:
    """Log-spaced radii spanning empirical pairwise-distance quantiles.

    Quantiles are estimated from a random subsample of
    `config.CORR_DIM_RADIUS_SAMPLE_SIZE` points (rather than the full
    pairwise-distance vector) purely to keep radius-selection cheap; the
    actual correlation integral is still evaluated on the full dataset.
    """
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(X.shape[0], size=min(config.CORR_DIM_RADIUS_SAMPLE_SIZE, X.shape[0]), replace=False)
    sample_distances = pdist(X[sample_idx], metric="euclidean")
    r_min = np.quantile(sample_distances, config.CORR_DIM_MIN_QUANTILE)
    r_max = np.quantile(sample_distances, config.CORR_DIM_MAX_QUANTILE)
    return np.logspace(np.log10(r_min), np.log10(r_max), config.CORR_DIM_N_RADII)


def _auto_detect_scaling_region(log_r: np.ndarray, log_C: np.ndarray, min_points: int) -> tuple:
    """Locate the plateau of the local slope d(log C)/d(log r).

    Slides a fixed-width window of `min_points - 1` consecutive local-slope
    segments across the curve and returns the window with the lowest slope
    variance -- i.e. the most linear stretch of the log-log curve, which is
    the theoretically expected signature of the scaling region for a
    self-similar point cloud.
    """
    n = len(log_r)
    window = min_points - 1
    if n <= min_points:
        return 0, n - 1

    local_slopes = np.diff(log_C) / np.diff(log_r)
    best_std, best_start = np.inf, 0
    for start in range(len(local_slopes) - window + 1):
        segment_std = np.std(local_slopes[start:start + window])
        if segment_std < best_std:
            best_std, best_start = segment_std, start
    return best_start, best_start + window


def estimate_correlation_dimension(
    X: np.ndarray, radii: np.ndarray, scaling_region: Optional[tuple] = None
) -> dict:
    """Fit log C(r) vs log r over a scaling region to estimate D2.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    radii : np.ndarray of shape (n_radii,)
    scaling_region : tuple of (r_min, r_max), optional
        Radius bounds defining the fit region. If None, the region is
        auto-detected via `_auto_detect_scaling_region` using
        `config.CORR_DIM_MIN_SCALING_POINTS`.

    Returns
    -------
    dict with keys 'D2', 'r_squared', 'scaling_region', 'radii', 'C_r'.
    """
    C_r = correlation_integral(X, radii)
    valid = C_r > 0  # log(0) is undefined; C(r)=0 occurs at microscopic radii
    log_r, log_C = np.log(radii[valid]), np.log(C_r[valid])

    if scaling_region is None:
        start_idx, end_idx = _auto_detect_scaling_region(log_r, log_C, config.CORR_DIM_MIN_SCALING_POINTS)
    else:
        r_min, r_max = scaling_region
        in_region = np.where((np.exp(log_r) >= r_min) & (np.exp(log_r) <= r_max))[0]
        start_idx, end_idx = int(in_region[0]), int(in_region[-1])

    fit_log_r = log_r[start_idx:end_idx + 1]
    fit_log_C = log_C[start_idx:end_idx + 1]
    slope, intercept, r_value, _, _ = linregress(fit_log_r, fit_log_C)

    return {
        "D2": float(slope),
        "intercept": float(intercept),
        "r_squared": float(r_value ** 2),
        "scaling_region": (float(np.exp(fit_log_r[0])), float(np.exp(fit_log_r[-1]))),
        "radii": radii,
        "C_r": C_r,
    }


def _kaiser_elbow(eigenvalues: np.ndarray) -> int:
    """Kaiser criterion: number of eigenvalues exceeding the mean eigenvalue."""
    return int(np.sum(eigenvalues > eigenvalues.mean()))


def _curvature_elbow(explained_variance_ratio: np.ndarray) -> int:
    """Second-derivative elbow: index of maximum curvature in the scree curve.

    A scree curve is decreasing and convex, so its discrete second difference
    is non-negative and peaks where the steep initial drop flattens out --
    the elbow. Taking the maximum of the *negated* second difference instead
    would locate the point of maximum concavity, which on such a curve is not
    an elbow at all (on the 5-D manifold baseline it returns 4 rather than 5,
    and on isotropic noise it returns the final index).
    """
    second_derivative = np.diff(explained_variance_ratio, n=2)
    return int(np.argmax(second_derivative)) + 1


def pca_based_dimension(X: np.ndarray, variance_thresholds: list) -> dict:
    """Estimate intrinsic dimension from the covariance eigenvalue spectrum.

    Numerical note
    ---------------
    With d=20,264 >> n=801, the (d x d) sample covariance matrix is never
    formed explicitly (it would require ~3.3 GB and is rank-deficient with
    rank <= n-1). Instead the economy SVD of the centered (n x d) data
    matrix is used: eigenvalues of the covariance equal S^2 / (n-1) for
    singular values S, giving all min(n, d) non-trivial eigenvalues exactly.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    variance_thresholds : list of float
        Cumulative explained-variance cutoffs, e.g. [0.90, 0.95, 0.99].

    Returns
    -------
    dict with keys 'eigenvalues', 'explained_variance_ratio',
    'cumulative_variance_ratio', 'n_components_at_threshold',
    'elbow_kaiser', 'elbow_curvature'.
    """
    X_centered = X - X.mean(axis=0)
    _, singular_values, _ = np.linalg.svd(X_centered, full_matrices=False)
    eigenvalues = (singular_values ** 2) / (X.shape[0] - 1)

    explained_variance_ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained_variance_ratio)

    n_components_at_threshold = {
        t: int(np.searchsorted(cumulative, t) + 1) for t in variance_thresholds
    }

    return {
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": cumulative,
        "n_components_at_threshold": n_components_at_threshold,
        "elbow_kaiser": _kaiser_elbow(eigenvalues),
        "elbow_curvature": _curvature_elbow(explained_variance_ratio),
    }


def knn_mle_dimension(X: np.ndarray, k_values: list) -> dict:
    """Levina-Bickel k-NN maximum-likelihood intrinsic dimension estimator.

    m_k(x_i) = [ (1 / (k - 2)) * sum_{j=1}^{k-1} ln(T_k(x_i) / T_j(x_i)) ]^(-1)

    where T_j(x_i) is the Euclidean distance from x_i to its j-th nearest
    neighbor. The global estimate at each k is the mean of m_k(x_i) over all
    samples. `config.KNN_MLE_EPSILON` is added to every neighbor distance to
    guard against ties (T_j == T_k) causing log(1) = 0 in the denominator
    sum, which would otherwise make individual local estimates blow up.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    k_values : list of int
        Neighborhood sizes to evaluate; each must be >= 3.

    Returns
    -------
    dict with keys 'per_k' (per-k mean/std/local estimates), 'k_values',
    'mean_across_k', 'std_across_k'.
    """
    per_k = {}
    for k in k_values:
        assert k >= 3, "Levina-Bickel requires k >= 3 (division by k - 2)"

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
        distances, _ = nbrs.kneighbors(X)
        distances = distances[:, 1:] + config.KNN_MLE_EPSILON  # drop self (distance 0)

        T_j = distances[:, : k - 1]      # j = 1, ..., k-1
        T_k = distances[:, k - 1: k]     # k-th neighbor distance
        sum_log_ratios = np.log(T_k / T_j).sum(axis=1)

        with np.errstate(divide="ignore"):
            local_m = (k - 2) / sum_log_ratios

        valid = np.isfinite(local_m) & (local_m > 0)
        per_k[k] = {
            "mean_dimension": float(local_m[valid].mean()),
            "std_dimension": float(local_m[valid].std()),
            "n_valid_samples": int(valid.sum()),
            "local_estimates": local_m,
        }

    means = [v["mean_dimension"] for v in per_k.values()]
    return {
        "per_k": per_k,
        "k_values": k_values,
        "mean_across_k": float(np.mean(means)),
        "std_across_k": float(np.std(means)),
    }
