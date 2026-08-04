"""Top-level orchestration script.

Implements Phase 1 (preprocessing & cleaning) and Phase 2 (intrinsic
dimension estimation). Later phases will be added here as their `src/`
modules are implemented, per project_framework/SKELETON.md. Contains no
analysis logic of its own -- only orchestration, logging, and figure/table
export.
"""

import json
from pathlib import Path

import numpy as np

from src import (
    anomaly_detection,
    config,
    data_loading,
    embeddings,
    evaluation,
    geometry_diagnostics,
    intrinsic_dimension,
    preprocessing,
    random_projection,
    spectral_analysis,
    synthetic,
    viz,
)


def _estimate_all(X: np.ndarray, seed: int) -> dict:
    """Run the correlation, PCA, and k-NN MLE estimators on one dataset."""
    radii = intrinsic_dimension.default_radii(X, seed)
    return {
        "correlation": intrinsic_dimension.estimate_correlation_dimension(X, radii),
        "pca": intrinsic_dimension.pca_based_dimension(X, config.PCA_VARIANCE_THRESHOLDS),
        "knn": intrinsic_dimension.knn_mle_dimension(X, config.KNN_MLE_K_VALUES),
    }


def _manifold_noise_sweep(n_samples: int, n_features: int, noise_levels: list, seed: int) -> None:
    """Print estimator behaviour on the manifold baseline across noise levels.

    Motivates config.MANIFOLD_NOISE_STD: the additive noise contributes
    n_features * sigma^2 of total variance against only MANIFOLD_INTRINSIC_DIM
    from signal, so "small" sigma must be judged relative to the ambient
    dimension. Preprocessing matches the main run so the sigma chosen there is
    directly comparable to the summary table.
    """
    print("\nPhase 2 manifold-baseline noise sensitivity (true dimension "
          f"{config.MANIFOLD_INTRINSIC_DIM}):")
    for sigma in noise_levels:
        X = preprocessing.standardize(
            synthetic.generate_low_dim_manifold_baseline(
                n_samples, n_features, config.MANIFOLD_INTRINSIC_DIM, sigma, seed
            )
        )
        res = _estimate_all(X, seed)
        print(
            f"  sigma={sigma:<7} D2={res['correlation']['D2']:>7.2f}  "
            f"kaiser={res['pca']['elbow_kaiser']:>4}  "
            f"PCA@90={res['pca']['n_components_at_threshold'][0.90]:>4}  "
            f"kNN-MLE(k=10)={res['knn']['per_k'][10]['mean_dimension']:>8.2f}"
        )


def main():
    Path(config.PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    # --- Phase 1: load & validate ---
    X_df = data_loading.load_expression_matrix(config.RAW_DATA_PATH)
    y = data_loading.load_labels(config.RAW_LABELS_PATH)
    validation = data_loading.validate_dataset(X_df, y)
    print("Validation summary:")
    print(json.dumps(validation, indent=2))

    X_raw = X_df.values.astype(np.float64)

    # --- Phase 1: log transform -> drop zero-variance genes -> standardize (full) ---
    X_log = preprocessing.log_transform(X_raw)
    X_log_clean, keep_mask = preprocessing.drop_zero_variance_genes(X_log)
    X_scaled_full = preprocessing.standardize(X_log_clean)

    # --- Phase 1: reduced top-k variant (ranked by variance on log-scale data) ---
    X_log_topk, topk_idx = preprocessing.filter_top_variable_genes(
        X_log_clean, config.N_TOP_VARIABLE_GENES
    )
    X_scaled_topk = preprocessing.standardize(X_log_topk)

    # --- Save processed artifacts ---
    np.savez_compressed(
        Path(config.PROCESSED_DIR) / "phase1_full.npz",
        X_scaled=X_scaled_full.astype(np.float32),
        keep_mask=keep_mask,
    )
    np.savez_compressed(
        Path(config.PROCESSED_DIR) / "phase1_topk.npz",
        X_scaled=X_scaled_topk.astype(np.float32),
        topk_idx=topk_idx,
    )
    y.to_frame(name="Class").to_csv(Path(config.PROCESSED_DIR) / "labels.csv")
    with open(Path(config.PROCESSED_DIR) / "validation_summary.json", "w") as f:
        json.dump(validation, f, indent=2)

    # --- Figures ---
    viz.plot_class_balance(y, str(Path(config.FIGURES_DIR) / "class_balance.png"))
    y.value_counts().rename("count").to_csv(Path(config.FIGURES_DIR) / "class_balance_table.csv")

    viz.plot_gene_mean_variance(
        X_log_clean,
        str(Path(config.FIGURES_DIR) / "gene_mean_variance_full.png"),
        title_suffix=f"(full, n={X_log_clean.shape[1]} genes)",
    )
    viz.plot_gene_mean_variance(
        X_log_topk,
        str(Path(config.FIGURES_DIR) / "gene_mean_variance_topk.png"),
        title_suffix=f"(top-{config.N_TOP_VARIABLE_GENES} variable genes)",
    )

    print(
        f"Phase 1 complete. Kept {int(keep_mask.sum())}/{len(keep_mask)} genes "
        f"after dropping {int((~keep_mask).sum())} zero-variance genes."
    )

    # --- Phase 2: synthetic baselines, matched in shape to the real full matrix ---
    # Both baselines are standardized like the real data, so the estimators see
    # one identical preprocessing path across all three datasets. Without this the
    # manifold baseline would be calibrated on unscaled coordinates while the real
    # data is z-scored, making the comparison an unequal one.
    n_samples, n_features = X_scaled_full.shape
    noise_baseline = preprocessing.standardize(
        synthetic.generate_gaussian_noise_baseline(n_samples, n_features, config.RANDOM_SEED)
    )
    manifold_baseline = preprocessing.standardize(
        synthetic.generate_low_dim_manifold_baseline(
            n_samples, n_features, config.MANIFOLD_INTRINSIC_DIM, config.MANIFOLD_NOISE_STD, config.RANDOM_SEED
        )
    )

    # --- Phase 2: run all three estimators on real data and both baselines ---
    real_results = _estimate_all(X_scaled_full, config.RANDOM_SEED)
    synthetic_results = {
        "Gaussian Noise": _estimate_all(noise_baseline, config.RANDOM_SEED),
        "Linear Manifold": _estimate_all(manifold_baseline, config.RANDOM_SEED),
    }

    # --- Phase 2: figures ---
    viz.scree_plot(
        real_results["pca"]["eigenvalues"],
        str(Path(config.FIGURES_DIR) / "scree_plot.png"),
        elbow_kaiser=real_results["pca"]["elbow_kaiser"],
        elbow_curvature=real_results["pca"]["elbow_curvature"],
    )
    viz.log_log_correlation_plot(
        real_results["correlation"]["radii"],
        real_results["correlation"]["C_r"],
        real_results["correlation"]["scaling_region"],
        str(Path(config.FIGURES_DIR) / "log_log_correlation.png"),
    )

    # --- Phase 2: summary table and written reconciliation ---
    summary_table = evaluation.build_intrinsic_dimension_summary_table(real_results, synthetic_results)
    reconciliation = evaluation.reconcile_estimators(summary_table)

    summary_table.to_csv(Path(config.FIGURES_DIR) / "phase2_summary_table.csv")
    with open(Path(config.PROCESSED_DIR) / "phase2_reconciliation.txt", "w") as f:
        f.write(reconciliation)

    print("\nPhase 2 intrinsic dimension summary:")
    print(summary_table.to_string())
    print("\nPhase 2 PCA elbow criteria (component index):")
    for name, res in {"Real (RNA-Seq)": real_results, **synthetic_results}.items():
        print(f"  {name}: kaiser={res['pca']['elbow_kaiser']}, curvature={res['pca']['elbow_curvature']}")
    print(
        f"Phase 2 correlation-dimension scaling region (real): "
        f"[{real_results['correlation']['scaling_region'][0]:.1f}, "
        f"{real_results['correlation']['scaling_region'][1]:.1f}]"
    )
    _manifold_noise_sweep(n_samples, n_features, [0.1, 0.01, 0.001], config.RANDOM_SEED)
    print("\nReconciliation:\n" + reconciliation)
    print(
        f"\nPhase 2 complete. Real data: D2={real_results['correlation']['D2']:.2f}, "
        f"PCA@90%={real_results['pca']['n_components_at_threshold'][0.90]}, "
        f"kNN-MLE mean={real_results['knn']['mean_across_k']:.2f}."
    )

    # --- Phase 3: PCA 2D/3D visualization ---
    y_values = y.values
    pca_viz = embeddings.run_pca(X_scaled_full, config.PCA_VIZ_N_COMPONENTS)
    viz.scatter_embedding(
        pca_viz["embedding"][:, :2], y_values, "PCA (2D) colored by cancer type",
        str(Path(config.FIGURES_DIR) / "pca_2d.png"),
    )
    viz.scatter_embedding(
        pca_viz["embedding"], y_values, "PCA (3D) colored by cancer type",
        str(Path(config.FIGURES_DIR) / "pca_3d.png"),
    )
    pca_viz_cumvar = float(np.sum(pca_viz["explained_variance_ratio"]))
    print(f"\nPhase 3 PCA viz: cumulative variance in top {config.PCA_VIZ_N_COMPONENTS} PCs = {pca_viz_cumvar:.4f}")

    # --- Phase 3: t-SNE / UMAP hyperparameter comparisons ---
    tsne_embeddings = embeddings.run_tsne(X_scaled_full, config.TSNE_PERPLEXITIES, config.RANDOM_SEED)
    viz.embedding_hyperparam_comparison_plot(
        tsne_embeddings, y_values, "perplexity", "t-SNE embeddings across perplexity values",
        str(Path(config.FIGURES_DIR) / "tsne_comparison.png"),
    )
    umap_embeddings = embeddings.run_umap(X_scaled_full, config.UMAP_N_NEIGHBORS_VALUES, config.RANDOM_SEED)
    viz.embedding_hyperparam_comparison_plot(
        umap_embeddings, y_values, "n_neighbors", "UMAP embeddings across n_neighbors values",
        str(Path(config.FIGURES_DIR) / "umap_comparison.png"),
    )
    print("Phase 3 t-SNE/UMAP comparison figures saved.")

    # --- Phase 3: Johnson-Lindenstrauss random projection ---
    jl_target_dim = random_projection.jl_target_dimension(X_scaled_full.shape[0], config.JL_EPSILON)
    X_proj = random_projection.random_gaussian_projection(X_scaled_full, jl_target_dim, config.RANDOM_SEED)
    jl_eval = random_projection.evaluate_distance_preservation(X_scaled_full, X_proj)
    viz.jl_distortion_plot(
        jl_eval["original_distances"], jl_eval["projected_distances"], jl_eval["distortion_ratios"],
        config.JL_EPSILON, str(Path(config.FIGURES_DIR) / "jl_distortion.png"),
    )
    print(
        f"Phase 3 JL projection: target_dim={jl_target_dim} (from d={X_scaled_full.shape[1]}), "
        f"mean distortion={jl_eval['mean_distortion']:.4f}, max |deviation|={jl_eval['max_abs_deviation']:.4f}, "
        f"fraction within eps={config.JL_EPSILON}: {jl_eval['frac_within_epsilon']:.4f}"
    )

    # --- Phase 3: spectral analysis (top-k genes) vs. Marchenko-Pastur ---
    spectral_eigs = spectral_analysis.empirical_spectral_distribution(X_scaled_topk)
    # Columns are mean-centred, so the correlation matrix carries n-1 effective
    # degrees of freedom, not n. Using n would put the MP point mass at 59.95%
    # against an empirical 60.00% that is in fact exact.
    spectral_aspect_ratio = X_scaled_topk.shape[1] / (X_scaled_topk.shape[0] - 1)
    spectral_outliers = spectral_analysis.identify_spectral_outliers(spectral_eigs, spectral_aspect_ratio)
    support_upper = spectral_outliers["support_upper"]
    bulk_eigs = spectral_eigs[(spectral_eigs > 1e-8) & (spectral_eigs <= support_upper)]
    grid = np.linspace(max(spectral_outliers["support_lower"] * 0.8, 1e-6), support_upper * 1.05, 1000)
    mp_density_conditional = spectral_analysis.marchenko_pastur_density(spectral_aspect_ratio, grid) * spectral_aspect_ratio
    viz.spectral_overlay_plot(
        bulk_eigs, spectral_eigs, grid, mp_density_conditional, support_upper,
        str(Path(config.FIGURES_DIR) / "spectral_overlay.png"),
    )
    print(
        f"Phase 3 spectral analysis (top-{config.SPECTRAL_N_TOP_GENES} genes, aspect ratio="
        f"{spectral_aspect_ratio:.3f}): MP support=[{spectral_outliers['support_lower']:.3f}, "
        f"{spectral_outliers['support_upper']:.3f}], n_outliers={spectral_outliers['n_outliers']}, "
        f"empirical near-zero fraction={spectral_outliers['frac_near_zero_empirical']:.4f} vs. "
        f"theoretical point mass={spectral_outliers['point_mass_at_zero_theoretical']:.4f}"
    )
    top10_eigs = spectral_outliers["outlier_eigenvalues"][:10]
    print("Phase 3 ten largest spectral outliers:", " ".join(f"{e:.2f}" for e in top10_eigs))
    print(f"Phase 3 outlier eigenvalue mass: {top10_eigs.sum():.2f} of trace {spectral_eigs.sum():.0f}")

    # --- Phase 3: distance concentration / curse of dimensionality ---
    geometry_feature_counts = config.GEOMETRY_FEATURE_COUNTS + [X_scaled_full.shape[1]]
    concentration_results = geometry_diagnostics.pairwise_distance_concentration(
        X_scaled_full, geometry_feature_counts, config.RANDOM_SEED
    )
    theoretical_bounds = {
        count: geometry_diagnostics.theoretical_concentration_bound(count, config.GEOMETRY_CONCENTRATION_EPSILON)
        for count in concentration_results
    }
    viz.distance_concentration_plot(
        concentration_results, theoretical_bounds, str(Path(config.FIGURES_DIR) / "distance_concentration.png"),
    )
    print("Phase 3 distance concentration ratios:", {c: round(v["ratio"], 4) for c, v in concentration_results.items()})
    print("Phase 3 theoretical concentration bounds:", {c: round(b, 6) for c, b in theoretical_bounds.items()})

    # --- Phase 3: Ledoit-Wolf shrinkage covariance + Mahalanobis anomaly detection ---
    cov_estimate = anomaly_detection.shrinkage_covariance(X_scaled_topk)
    # Degrees of freedom are the rank of the centred sample covariance (n-1), not the
    # feature count: with p=2000 > n=801 the squared distances live in an 800-dim
    # subspace and their mean is bounded by (n-1)/(1-shrinkage) regardless of the data,
    # so a chi^2_p threshold sits above the statistic's mathematical ceiling.
    mahalanobis_dof = X_scaled_topk.shape[0] - 1
    mahalanobis_threshold = anomaly_detection.chi_squared_threshold(mahalanobis_dof, config.MAHALANOBIS_ALPHA)
    mahalanobis_result = anomaly_detection.mahalanobis_outliers(X_scaled_topk, cov_estimate, mahalanobis_threshold)
    viz.mahalanobis_distribution_plot(
        mahalanobis_result["distances_squared"], mahalanobis_threshold, y_values,
        str(Path(config.FIGURES_DIR) / "mahalanobis_distribution.png"),
    )
    flagged_labels = y.iloc[mahalanobis_result["flagged_indices"]]
    outlier_crosstab = flagged_labels.value_counts().rename("n_flagged").to_frame()
    outlier_crosstab.to_csv(Path(config.FIGURES_DIR) / "mahalanobis_outliers_by_class.csv")
    d2 = mahalanobis_result["distances_squared"]
    print(
        f"Phase 3 anomaly detection: shrinkage={cov_estimate['shrinkage']:.4f}, "
        f"threshold(chi2, dof={mahalanobis_dof}, alpha={config.MAHALANOBIS_ALPHA})={mahalanobis_threshold:.2f}, "
        f"n_flagged={len(mahalanobis_result['flagged_indices'])}/{X_scaled_topk.shape[0]}"
    )
    print(
        f"Phase 3 squared Mahalanobis distances: min={d2.min():.2f}, max={d2.max():.2f}, "
        f"mean={d2.mean():.2f}, std={d2.std():.2f}"
    )
    print("Phase 3 ten largest squared Mahalanobis distances:")
    for rank in np.argsort(d2)[::-1][:10]:
        print(f"  {y.index[rank]}  {y.iloc[rank]}  {d2[rank]:.2f}")
    print("Flagged outliers by class:\n", outlier_crosstab.to_string())

    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
