"""Phase 2: synthetic baseline generators for intrinsic-dimension calibration.

Both generators produce arrays matched in shape to the real preprocessed
data so that the three estimators in `intrinsic_dimension.py` can be run
identically on real and synthetic inputs and compared side by side.
"""

import numpy as np

from src import config


def generate_gaussian_noise_baseline(
    n_samples: int,
    n_features: int,
    seed: int,
    std: float = config.SYNTHETIC_NOISE_STD,
) -> np.ndarray:
    """Sample pure high-dimensional isotropic Gaussian noise.

    Every coordinate is drawn i.i.d. from N(0, std^2), so the data cloud has
    no linear or nonlinear low-dimensional structure: its correlation,
    PCA-based, and k-NN intrinsic dimension estimates should all saturate
    near the ambient dimension `n_features`, providing an upper-bound
    calibration reference for the real dataset's estimates.

    Parameters
    ----------
    n_samples : int
        Number of samples (rows) to generate.
    n_features : int
        Ambient dimensionality (columns) to generate.
    seed : int
        Random seed for reproducibility.
    std : float, default config.SYNTHETIC_NOISE_STD
        Per-coordinate standard deviation.

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=std, size=(n_samples, n_features))


def generate_low_dim_manifold_baseline(
    n_samples: int,
    ambient_dim: int,
    intrinsic_dim: int,
    noise_std: float,
    seed: int,
) -> np.ndarray:
    """Sample points on a known linear subspace embedded in a high-dim space.

    Latent coordinates Z (n_samples x intrinsic_dim) are drawn i.i.d. N(0, 1)
    and mapped into the ambient space through a fixed random orthonormal
    basis B (ambient_dim x intrinsic_dim), obtained via QR decomposition of
    a random Gaussian matrix. Isotropic Gaussian noise is then added in the
    full ambient space, so the resulting cloud has a known ground-truth
    intrinsic dimension `intrinsic_dim`, perturbed by noise of controlled
    magnitude `noise_std` -- a calibration target the three estimators must
    recover approximately.

    Parameters
    ----------
    n_samples : int
        Number of samples (rows) to generate.
    ambient_dim : int
        Ambient embedding dimensionality.
    intrinsic_dim : int
        True dimensionality of the linear subspace the data lies on.
    noise_std : float
        Standard deviation of the additive isotropic Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples, ambient_dim)
    """
    rng = np.random.default_rng(seed)

    random_matrix = rng.normal(size=(ambient_dim, intrinsic_dim))
    basis, _ = np.linalg.qr(random_matrix)  # orthonormal columns spanning the subspace

    latent = rng.normal(loc=0.0, scale=1.0, size=(n_samples, intrinsic_dim))
    embedded = latent @ basis.T

    noise = rng.normal(loc=0.0, scale=noise_std, size=(n_samples, ambient_dim))
    return embedded + noise
