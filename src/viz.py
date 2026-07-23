"""Shared plotting utilities.

Phase 1 plotting functions (class balance, per-gene mean/variance) are
implemented below. Phase 2/3 functions are left as TODO stubs per the
project's phased scope and will be implemented alongside their respective
analysis modules.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_balance(y: pd.Series, save_path: str) -> None:
    """Bar chart of cancer-type sample counts, saved to `save_path`."""
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values, color="steelblue")
    ax.set_xlabel("Cancer type")
    ax.set_ylabel("Number of samples")
    ax.set_title("Class balance across cancer types")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_gene_mean_variance(X: np.ndarray, save_path: str, title_suffix: str = "") -> None:
    """Two-panel histogram of per-gene mean and per-gene variance, saved to `save_path`."""
    means = X.mean(axis=0)
    variances = X.var(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(means, bins=60, color="steelblue")
    axes[0].set_title(f"Per-gene mean {title_suffix}".strip())
    axes[0].set_xlabel("Mean (log1p scale)")
    axes[0].set_ylabel("Number of genes")

    axes[1].hist(variances, bins=60, color="indianred")
    axes[1].set_title(f"Per-gene variance {title_suffix}".strip())
    axes[1].set_xlabel("Variance (log1p scale)")
    axes[1].set_ylabel("Number of genes")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def scree_plot(eigenvalues: "np.ndarray", save_path: str) -> None:
    # TODO: Phase 2
    raise NotImplementedError("Phase 2 not yet implemented")


def scatter_embedding(embedding: "np.ndarray", labels: "np.ndarray", title: str, save_path: str) -> None:
    # TODO: Phase 3
    raise NotImplementedError("Phase 3 not yet implemented")


def log_log_correlation_plot(radii: "np.ndarray", C_r: "np.ndarray", scaling_region: tuple, save_path: str) -> None:
    # TODO: Phase 2
    raise NotImplementedError("Phase 2 not yet implemented")


def spectral_overlay_plot(empirical_eigs: "np.ndarray", mp_density_x: "np.ndarray", mp_density_y: "np.ndarray", save_path: str) -> None:
    # TODO: Phase 3
    raise NotImplementedError("Phase 3 not yet implemented")
