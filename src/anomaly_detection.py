"""Phase 3: shrinkage covariance estimation and Mahalanobis outlier detection.

Operates on the top-k most variable genes (config.SPECTRAL_N_TOP_GENES),
the same tractable subset used in spectral_analysis.py: even Ledoit-Wolf
shrinkage cannot make a (20,264 x 20,264) covariance matrix invertible in
practice (it can be computed, but at ~3.3 GB and an O(d^3) inversion cost
that is not a reasonable per-sample diagnostic), and d >> n makes the raw
sample covariance on any gene subset with d > n exactly singular, which is
precisely why shrinkage is required here rather than optional.
"""

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf


def shrinkage_covariance(X: np.ndarray) -> dict:
    """Ledoit-Wolf shrinkage covariance estimate.

    Required because the raw sample covariance of X (n_samples x n_features
    with n_features possibly >= n_samples) is singular or ill-conditioned:
    Ledoit-Wolf shrinks the sample covariance toward a scaled identity
    target, guaranteeing a well-conditioned, invertible estimate.

    Returns
    -------
    dict with keys 'covariance' (n_features x n_features), 'shrinkage'
    (the estimated shrinkage intensity in [0, 1]), 'location' (mean vector).
    """
    lw = LedoitWolf().fit(X)
    return {
        "covariance": lw.covariance_,
        "shrinkage": float(lw.shrinkage_),
        "location": lw.location_,
    }


def chi_squared_threshold(dof: int, alpha: float) -> float:
    """Upper-tail chi-squared quantile threshold at significance `alpha`.

    Under the null model that samples are multivariate Gaussian with the
    estimated covariance, squared Mahalanobis distances are approximately
    chi-squared distributed with `dof` degrees of freedom (one per feature).
    """
    return float(chi2.ppf(1 - alpha, df=dof))


def mahalanobis_outliers(X: np.ndarray, cov_estimate: dict, threshold: float) -> dict:
    """Per-sample Mahalanobis distance and outlier flagging.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    cov_estimate : dict
        Output of `shrinkage_covariance` (uses 'covariance' and 'location').
    threshold : float
        Squared-distance threshold above which a sample is flagged (e.g.
        from `chi_squared_threshold`).

    Returns
    -------
    dict with keys 'distances_squared' (n_samples,), 'flagged_mask',
    'flagged_indices', 'threshold'.
    """
    centered = X - cov_estimate["location"]
    cov_inv = np.linalg.inv(cov_estimate["covariance"])
    distances_squared = np.einsum("ij,jk,ik->i", centered, cov_inv, centered)

    flagged_mask = distances_squared > threshold
    return {
        "distances_squared": distances_squared,
        "flagged_mask": flagged_mask,
        "flagged_indices": np.where(flagged_mask)[0],
        "threshold": threshold,
    }
