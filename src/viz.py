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


def scree_plot(
    eigenvalues: "np.ndarray",
    save_path: str,
    elbow_kaiser: int = None,
    elbow_curvature: int = None,
    n_show: int = 50,
) -> None:
    """Scree plot (per-component and cumulative explained variance).

    Marks the Kaiser-criterion and curvature-based elbow indices, if given,
    on the per-component panel.
    """
    explained = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained)
    n_show = min(n_show, len(eigenvalues))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, n_show + 1), explained[:n_show], "o-", color="steelblue", markersize=3)
    if elbow_kaiser is not None and elbow_kaiser <= n_show:
        axes[0].axvline(elbow_kaiser, color="indianred", linestyle="--", label=f"Kaiser elbow (k={elbow_kaiser})")
    if elbow_curvature is not None and elbow_curvature <= n_show:
        axes[0].axvline(elbow_curvature, color="darkorange", linestyle=":", label=f"Curvature elbow (k={elbow_curvature})")
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(f"Scree plot (first {n_show} PCs)")
    axes[0].legend()

    axes[1].plot(range(1, len(cumulative) + 1), cumulative, color="seagreen")
    axes[1].set_xlabel("Number of components")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title("Cumulative explained variance")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def scatter_embedding(embedding: "np.ndarray", labels: "np.ndarray", title: str, save_path: str) -> None:
    # TODO: Phase 3
    raise NotImplementedError("Phase 3 not yet implemented")


def log_log_correlation_plot(radii: "np.ndarray", C_r: "np.ndarray", scaling_region: tuple, save_path: str) -> None:
    """Log-log plot of the correlation integral with the fitted scaling region highlighted."""
    valid = C_r > 0
    log_r, log_C = np.log(radii[valid]), np.log(C_r[valid])

    r_min, r_max = scaling_region
    in_region = (radii[valid] >= r_min) & (radii[valid] <= r_max)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(log_r, log_C, "o", markersize=3, color="steelblue", label="log C(r)")
    ax.plot(log_r[in_region], log_C[in_region], "o", markersize=4, color="indianred", label="scaling region")

    if in_region.sum() >= 2:
        slope, intercept = np.polyfit(log_r[in_region], log_C[in_region], 1)
        fit_line = slope * log_r[in_region] + intercept
        ax.plot(log_r[in_region], fit_line, "-", color="black", linewidth=1.5, label=f"fit slope $D_2$={slope:.2f}")

    ax.set_xlabel("log r")
    ax.set_ylabel("log C(r)")
    ax.set_title("Grassberger-Procaccia correlation integral")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def spectral_overlay_plot(empirical_eigs: "np.ndarray", mp_density_x: "np.ndarray", mp_density_y: "np.ndarray", save_path: str) -> None:
    # TODO: Phase 3
    raise NotImplementedError("Phase 3 not yet implemented")
