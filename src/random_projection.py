"""Phase 3: Johnson-Lindenstrauss random projection experiments."""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import GaussianRandomProjection

from src import config, intrinsic_dimension


def jl_target_dimension(n_samples: int, epsilon: float) -> int:
    """Johnson-Lindenstrauss target dimension for `n_samples` points.

    Uses the Dasgupta-Gupta (1999/2003) bound, tighter than the original JL
    bound, which guarantees that a random projection into k dimensions
    preserves all pairwise distances up to a factor (1 +/- epsilon) with high
    probability whenever

        k >= 4 * ln(n_samples) / (epsilon^2 / 2 - epsilon^3 / 3).

    Parameters
    ----------
    n_samples : int
        Number of points whose pairwise distances must be preserved.
    epsilon : float
        Allowed distortion, in (0, 1).

    Returns
    -------
    int
        The smallest integer target dimension satisfying the bound.
    """
    denom = (epsilon ** 2) / 2 - (epsilon ** 3) / 3
    k = 4 * np.log(n_samples) / denom
    return int(np.ceil(k))


def random_gaussian_projection(X: np.ndarray, target_dim: int, seed: int) -> np.ndarray:
    """Project X onto `target_dim` dimensions via a random Gaussian matrix.

    Entries are drawn i.i.d. N(0, 1/target_dim), the standard JL scaling that
    makes the projection approximately norm- (and hence distance-) preserving
    in expectation: E[||Rx||^2] = ||x||^2 for R with this entrywise variance.
    """
    rng = np.random.default_rng(seed)
    R = rng.normal(loc=0.0, scale=1.0 / np.sqrt(target_dim), size=(X.shape[1], target_dim))
    return X @ R


def evaluate_distance_preservation(X: np.ndarray, X_proj: np.ndarray) -> dict:
    """Compare all-pairs distances in X vs X_proj.

    Returns the empirical distortion ratio (projected / original distance)
    for every pair, plus summary statistics comparable to the theoretical
    (1 +/- epsilon) JL bound.

    Returns
    -------
    dict with keys 'original_distances', 'projected_distances',
    'distortion_ratios', 'mean_distortion', 'max_abs_deviation'.
    """
    original_distances = pdist(X, metric="euclidean")
    projected_distances = pdist(X_proj, metric="euclidean")

    nonzero = original_distances > 0
    distortion_ratios = projected_distances[nonzero] / original_distances[nonzero]

    return {
        "original_distances": original_distances,
        "projected_distances": projected_distances,
        "distortion_ratios": distortion_ratios,
        "mean_distortion": float(distortion_ratios.mean()),
        "max_abs_deviation": float(np.max(np.abs(distortion_ratios - 1.0))),
        "frac_within_epsilon": float(
            np.mean(np.abs(distortion_ratios - 1.0) <= config.JL_EPSILON)
        ),
    }


def jl_theoretical_max_deviation(n_pairs: int, target_dim: int) -> float:
    """High-probability bound on the largest |distortion ratio - 1| over
    `n_pairs` pairs under a Gaussian projection to `target_dim` dimensions,
    sqrt(2 ln n_pairs) / sqrt(2 target_dim) -- a union bound over the
    per-pair sub-exponential tail of the squared-norm ratio. Data-independent:
    holds identically for any point set, which is what makes it the right
    null model for the metric-distortion curve (see jl_dimension_sweep).
    """
    return float(np.sqrt(2 * np.log(n_pairs)) / np.sqrt(2 * target_dim))


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's k nearest neighbors in X, self excluded."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, indices = nbrs.kneighbors(X)
    return indices[:, 1:]


def _mean_neighborhood_overlap(knn_a: np.ndarray, knn_b: np.ndarray) -> float:
    """Mean, over rows, of |knn_a[i] intersect knn_b[i]| / k for same-shape k-NN index arrays."""
    k = knn_a.shape[1]
    overlaps = [len(set(knn_a[i]) & set(knn_b[i])) / k for i in range(knn_a.shape[0])]
    return float(np.mean(overlaps))


def jl_dimension_sweep(X: np.ndarray, y: np.ndarray, k_values: list, seeds: list, k_nn: int = 10) -> dict:
    """Sweep target dimension k, evaluating a Gaussian random projection at
    every (k, seed) pair on two axes: data-independent metric distortion, and
    data-dependent structure preservation.

    A single (epsilon, k) JL experiment is uninformative once k approaches or
    exceeds n-1 (the exact-distortion rank bound for n centred points): the
    guarantee stops being binding. Sweeping k separates two effects that a
    single point conflates -- oblivious metric distortion, which theory
    predicts regardless of the data, from neighborhood/cluster structure,
    which survives compression only as far as the data's actual
    (lower-dimensional) structure allows.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Same standardized matrix used elsewhere in the report.
    y : array-like of shape (n_samples,)
        Class labels, for the silhouette score.
    k_values : list of int
        Target dimensions to sweep.
    seeds : list of int
        Random seeds; the projection at each k is repeated once per seed.
    k_nn : int
        Neighborhood size for both the 10-NN overlap metric and the
        Levina-Bickel estimator run on the projected data.

    Returns
    -------
    dict keyed by k. Each value is a dict mapping stat name -> (mean, std)
    across seeds, for 'mean_rho', 'max_abs_dev', 'p999_abs_dev', 'overlap',
    'lb_id', 'silhouette'; plus 'theoretical_max_dev' (seed-independent).
    """
    original_distances = pdist(X, metric="euclidean")
    n_pairs = original_distances.shape[0]
    nonzero = original_distances > 0
    original_knn = _knn_indices(X, k_nn)

    per_k = {}
    for k in k_values:
        stats = {name: [] for name in ("mean_rho", "max_abs_dev", "p999_abs_dev", "overlap", "lb_id", "silhouette")}
        for seed in seeds:
            X_proj = GaussianRandomProjection(n_components=k, random_state=seed).fit_transform(X)

            rho = pdist(X_proj, metric="euclidean")[nonzero] / original_distances[nonzero]
            abs_dev = np.abs(rho - 1.0)
            stats["mean_rho"].append(float(rho.mean()))
            stats["max_abs_dev"].append(float(abs_dev.max()))
            stats["p999_abs_dev"].append(float(np.quantile(abs_dev, 0.999)))

            projected_knn = _knn_indices(X_proj, k_nn)
            stats["overlap"].append(_mean_neighborhood_overlap(original_knn, projected_knn))

            lb_result = intrinsic_dimension.knn_mle_dimension(X_proj, [k_nn])
            stats["lb_id"].append(lb_result["per_k"][k_nn]["mean_dimension"])

            stats["silhouette"].append(float(silhouette_score(X_proj, y)))

        per_k[k] = {name: (float(np.mean(values)), float(np.std(values))) for name, values in stats.items()}
        per_k[k]["theoretical_max_dev"] = jl_theoretical_max_deviation(n_pairs, k)

    return per_k
