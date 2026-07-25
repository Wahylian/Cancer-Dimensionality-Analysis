"""Phase 3: distance concentration / curse-of-dimensionality diagnostics."""

import numpy as np
from scipy.spatial.distance import pdist


def pairwise_distance_concentration(X: np.ndarray, feature_counts: list, seed: int) -> dict:
    """Distance-concentration diagnostic across increasing feature-subsample sizes.

    For each value in `feature_counts`, a random subset of that many columns
    (genes) is drawn without replacement and all pairwise Euclidean distances
    are computed. The concentration-of-measure phenomenon predicts the
    relative spread ratio (max - min) / min shrinks toward 0 as the feature
    count grows, since pairwise distances converge (relative to their mean)
    in high dimensions.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    feature_counts : list of int
        Numbers of features to subsample; values >= X.shape[1] use all
        features (no subsampling).
    seed : int

    Returns
    -------
    dict keyed by the (possibly clipped) feature count, each holding
    {'distances', 'ratio', 'mean', 'std'}.
    """
    rng = np.random.default_rng(seed)
    results = {}
    for count in feature_counts:
        count = min(count, X.shape[1])
        cols = rng.choice(X.shape[1], size=count, replace=False)
        distances = pdist(X[:, cols], metric="euclidean")
        results[count] = {
            "distances": distances,
            "ratio": float((distances.max() - distances.min()) / distances.min()),
            "mean": float(distances.mean()),
            "std": float(distances.std()),
        }
    return results


def theoretical_concentration_bound(feature_count: int, epsilon: float) -> float:
    """Chernoff-type (Laurent-Massart) tail bound for chi-squared concentration.

    Treats the squared Euclidean distance between two independent
    standardized feature vectors as approximately a chi-squared variable
    with `feature_count` degrees of freedom (exact under an i.i.d. Gaussian
    approximation of the standardized coordinates; see report limitations
    for the correlated-features caveat). The Laurent-Massart (2000) tail
    bounds for X ~ chi^2_d are

        P(X - d >= 2*sqrt(d*t) + 2*t) <= exp(-t)
        P(d - X >= 2*sqrt(d*t))       <= exp(-t).

    Solving 2*sqrt(d*t) + 2*t = epsilon*d for t (the deviation level at
    which the upper-tail bound matches a relative distortion of `epsilon`)
    and combining both tails by a union bound gives the two-sided bound
    returned here: P(|X/d - 1| >= epsilon) <= 2*exp(-t).

    Parameters
    ----------
    feature_count : int
        Ambient dimension `d` of the (sub)space distances are computed in.
    epsilon : float
        Relative deviation threshold.

    Returns
    -------
    float
        Upper bound on P(|X/d - 1| >= epsilon), capped at 1.0.
    """
    d = feature_count
    s = np.sqrt(d) * (-1 + np.sqrt(1 + 2 * epsilon)) / 2
    t = s ** 2
    return float(min(1.0, 2 * np.exp(-t)))
