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
    """2D or 3D scatter plot (based on embedding.shape[1]) colored by class label."""
    classes = pd.Series(labels).unique()
    palette = plt.get_cmap("tab10")
    is_3d = embedding.shape[1] >= 3

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d" if is_3d else None)
    for i, cls in enumerate(classes):
        mask = np.asarray(labels) == cls
        coords = [embedding[mask, d] for d in range(3 if is_3d else 2)]
        ax.scatter(*coords, s=12, alpha=0.7, color=palette(i % 10), label=str(cls))

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    if is_3d:
        ax.set_zlabel("Component 3")
    ax.set_title(title)
    ax.legend(markerscale=1.5, fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def embedding_hyperparam_comparison_plot(
    embeddings: dict, labels: "np.ndarray", param_name: str, title: str, save_path: str
) -> None:
    """Side-by-side 2D scatter panels, one per hyperparameter value in `embeddings`."""
    classes = pd.Series(labels).unique()
    palette = plt.get_cmap("tab10")
    n_panels = len(embeddings)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), squeeze=False)
    for ax, (param_value, embedding) in zip(axes[0], embeddings.items()):
        for i, cls in enumerate(classes):
            mask = np.asarray(labels) == cls
            ax.scatter(embedding[mask, 0], embedding[mask, 1], s=10, alpha=0.7, color=palette(i % 10), label=str(cls))
        ax.set_title(f"{param_name} = {param_value}")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    axes[0][-1].legend(markerscale=1.5, fontsize=8, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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


def spectral_overlay_plot(
    bulk_eigs: "np.ndarray", all_eigs: "np.ndarray",
    mp_density_x: "np.ndarray", mp_density_y: "np.ndarray",
    support_upper: float, save_path: str,
) -> None:
    """Two-panel spectral overlay: zoomed MP-bulk fit, and full-range outlier tail.

    Left panel restricts to `bulk_eigs` (eigenvalues within the theoretical MP
    support) so the density comparison is legible; right panel shows the full
    spectrum `all_eigs` on a log y-axis with the MP upper edge marked, so
    outlier eigenvalues -- which can be orders of magnitude larger than the
    bulk -- remain visible without compressing the bulk fit.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(bulk_eigs, bins=60, density=True, color="steelblue", alpha=0.6, label="Empirical (bulk)")
    axes[0].plot(mp_density_x, mp_density_y, color="indianred", linewidth=2, label="Marchenko-Pastur density")
    axes[0].set_xlabel("Eigenvalue")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Bulk spectrum vs. Marchenko-Pastur law")
    axes[0].legend()

    axes[1].hist(all_eigs, bins=80, color="steelblue", alpha=0.7)
    axes[1].axvline(support_upper, color="indianred", linestyle="--", label=f"MP upper edge = {support_upper:.2f}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Eigenvalue")
    axes[1].set_ylabel("Number of eigenvalues (log scale)")
    axes[1].set_title("Full spectrum: outliers beyond the MP edge")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def jl_distortion_plot(
    original_distances: "np.ndarray", projected_distances: "np.ndarray",
    distortion_ratios: "np.ndarray", epsilon: float, save_path: str,
) -> None:
    """Two-panel figure: original vs. projected pairwise distances, and the distortion-ratio histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(original_distances, projected_distances, s=2, alpha=0.15, color="steelblue")
    lims = [0, max(original_distances.max(), projected_distances.max())]
    axes[0].plot(lims, lims, color="black", linewidth=1, label="y = x")
    axes[0].set_xlabel("Original pairwise distance")
    axes[0].set_ylabel("Projected pairwise distance")
    axes[0].set_title("Distance preservation under JL projection")
    axes[0].legend()

    axes[1].hist(distortion_ratios, bins=80, color="seagreen", alpha=0.7)
    axes[1].axvline(1 - epsilon, color="indianred", linestyle="--", label=f"1 - eps = {1 - epsilon:.2f}")
    axes[1].axvline(1 + epsilon, color="indianred", linestyle="--", label=f"1 + eps = {1 + epsilon:.2f}")
    axes[1].set_xlabel("Distortion ratio (projected / original)")
    axes[1].set_ylabel("Number of pairs")
    axes[1].set_title("Empirical distortion vs. theoretical JL bound")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def distance_concentration_plot(results: dict, theoretical_bounds: dict, save_path: str) -> None:
    """Concentration ratio (max-min)/min vs. feature count, log-x, with the theoretical bound curve."""
    feature_counts = sorted(results.keys())
    ratios = [results[c]["ratio"] for c in feature_counts]
    bounds = [theoretical_bounds[c] for c in feature_counts]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(feature_counts, ratios, "o-", color="steelblue", label="Empirical (max-min)/min")
    ax2 = ax.twinx()
    ax2.plot(feature_counts, bounds, "s--", color="indianred", label="Theoretical tail bound")
    ax.set_xscale("log")
    ax.set_xlabel("Number of features (log scale)")
    ax.set_ylabel("(max - min) / min distance", color="steelblue")
    ax2.set_ylabel("Theoretical bound on P(|X/d - 1| >= eps)", color="indianred")
    ax.set_title("Pairwise distance concentration with increasing dimension")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def mahalanobis_distribution_plot(
    distances_squared: "np.ndarray", threshold: float, labels: "np.ndarray", save_path: str
) -> None:
    """Histogram of squared Mahalanobis distances with the chi-squared outlier threshold marked."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(distances_squared, bins=60, color="steelblue", alpha=0.7)
    ax.axvline(threshold, color="indianred", linestyle="--", label=f"chi-squared threshold = {threshold:.1f}")
    ax.set_xlabel("Squared Mahalanobis distance")
    ax.set_ylabel("Number of samples")
    ax.set_title("Mahalanobis distance distribution and outlier threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
