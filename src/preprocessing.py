"""Phase 1: log transform, standardization, and variance-based feature filtering."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def log_transform(X: np.ndarray) -> np.ndarray:
    """Apply log1p to raw expression counts. Justify choice in report re: skewness."""
    return np.log1p(X)


def drop_zero_variance_genes(X: np.ndarray) -> tuple:
    """Remove genes with zero variance.

    Required before standardization: a zero-variance column has std = 0, and
    z-scoring it would divide by zero. Returns (X_filtered, keep_mask).
    """
    keep_mask = X.var(axis=0) > 0
    return X[:, keep_mask], keep_mask


def standardize(X: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance scale each gene (column). Must run after log_transform."""
    return StandardScaler().fit_transform(X)


def filter_top_variable_genes(X: np.ndarray, k: int) -> tuple:
    """Return (X_filtered, selected_gene_indices) keeping the k highest-variance genes.

    Variance is computed on `X` as passed in. Callers must pass log-scale (not
    standardized) data: after z-scoring, every gene has unit variance, which
    would make variance-based ranking meaningless.
    """
    variances = X.var(axis=0)
    selected = np.sort(np.argsort(variances)[::-1][:k])
    return X[:, selected], selected
