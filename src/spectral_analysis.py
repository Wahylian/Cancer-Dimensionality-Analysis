"""Phase 3: empirical spectral distribution vs. the Marchenko-Pastur law.

Operates on the top-k most variable genes (config.SPECTRAL_N_TOP_GENES),
matching the tractable subset already cached by Phase 1 preprocessing: the
full 20,264 x 20,264 covariance matrix is never formed (~3.3 GB), whereas a
2000 x 2000 correlation matrix (~32 MB) is trivial to form and diagonalize
exactly.
"""

import numpy as np


def empirical_spectral_distribution(X_scaled: np.ndarray) -> np.ndarray:
    """Eigenvalues of the sample correlation matrix of X_scaled.

    X_scaled is assumed already zero-mean, unit-variance per column (as
    produced by `preprocessing.standardize`), so (X^T X) / n is directly the
    sample correlation matrix. Formed explicitly and diagonalized with
    `np.linalg.eigh` (X_scaled has p = config.SPECTRAL_N_TOP_GENES <= 2000
    columns, so the p x p matrix is small).

    Returns
    -------
    np.ndarray of shape (p,), eigenvalues sorted ascending.
    """
    n_samples = X_scaled.shape[0]
    corr = (X_scaled.T @ X_scaled) / n_samples
    eigenvalues = np.linalg.eigvalsh(corr)
    return np.clip(eigenvalues, 0.0, None)  # guard tiny negative numerical noise


def marchenko_pastur_density(aspect_ratio: float, eigenvalue_grid: np.ndarray) -> np.ndarray:
    """Theoretical Marchenko-Pastur density for aspect ratio p/n = `aspect_ratio`.

    For a p x n data matrix with i.i.d. entries of variance sigma^2 = 1
    (standardized features) and p/n -> aspect_ratio, the limiting spectral
    density of the sample correlation matrix is

        f(x) = 1 / (2 * pi * aspect_ratio * x) * sqrt((b - x) * (x - a)),
        a <= x <= b,

    with support edges a = (1 - sqrt(aspect_ratio))^2,
    b = (1 + sqrt(aspect_ratio))^2. When aspect_ratio > 1 the matrix is
    rank-deficient and the full distribution additionally carries a point
    mass of (1 - 1/aspect_ratio) at x = 0, which is not part of this
    continuous density and must be reported separately.

    Returns
    -------
    np.ndarray, same shape as `eigenvalue_grid`, zero outside [a, b].
    """
    a = (1 - np.sqrt(aspect_ratio)) ** 2
    b = (1 + np.sqrt(aspect_ratio)) ** 2

    density = np.zeros_like(eigenvalue_grid, dtype=float)
    in_support = (eigenvalue_grid >= a) & (eigenvalue_grid <= b)
    x = eigenvalue_grid[in_support]
    density[in_support] = np.sqrt(np.clip((b - x) * (x - a), 0.0, None)) / (
        2 * np.pi * aspect_ratio * x
    )
    return density


def marchenko_pastur_support(aspect_ratio: float) -> tuple:
    """Return (a, b), the lower/upper edges of the MP continuous support."""
    a = (1 - np.sqrt(aspect_ratio)) ** 2
    b = (1 + np.sqrt(aspect_ratio)) ** 2
    return float(a), float(b)


def identify_spectral_outliers(eigenvalues: np.ndarray, aspect_ratio: float) -> dict:
    """Flag eigenvalues exceeding the theoretical MP upper support edge.

    Eigenvalues above `b` cannot be explained by a null (unstructured)
    random matrix of matching aspect ratio and are interpreted as evidence
    of real, non-random covariance structure (biological signal).

    Returns
    -------
    dict with keys 'support_lower', 'support_upper', 'n_outliers',
    'outlier_eigenvalues', 'point_mass_at_zero_theoretical',
    'frac_near_zero_empirical'.
    """
    a, b = marchenko_pastur_support(aspect_ratio)
    outliers = eigenvalues[eigenvalues > b]
    near_zero = eigenvalues[eigenvalues < 1e-8]
    return {
        "support_lower": a,
        "support_upper": b,
        "n_outliers": int(outliers.size),
        "outlier_eigenvalues": np.sort(outliers)[::-1],
        "point_mass_at_zero_theoretical": max(0.0, 1 - 1 / aspect_ratio),
        "frac_near_zero_empirical": float(near_zero.size / eigenvalues.size),
    }
