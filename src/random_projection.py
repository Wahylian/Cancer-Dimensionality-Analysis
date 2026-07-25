"""Phase 3: Johnson-Lindenstrauss random projection experiments."""

import numpy as np
from scipy.spatial.distance import pdist

from src import config


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
