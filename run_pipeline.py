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

from src import config, data_loading, evaluation, intrinsic_dimension, preprocessing, synthetic, viz


def _estimate_all(X: np.ndarray, seed: int) -> dict:
    """Run the correlation, PCA, and k-NN MLE estimators on one dataset."""
    radii = intrinsic_dimension.default_radii(X, seed)
    return {
        "correlation": intrinsic_dimension.estimate_correlation_dimension(X, radii),
        "pca": intrinsic_dimension.pca_based_dimension(X, config.PCA_VARIANCE_THRESHOLDS),
        "knn": intrinsic_dimension.knn_mle_dimension(X, config.KNN_MLE_K_VALUES),
    }


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
    n_samples, n_features = X_scaled_full.shape
    noise_baseline = synthetic.generate_gaussian_noise_baseline(n_samples, n_features, config.RANDOM_SEED)
    manifold_baseline = synthetic.generate_low_dim_manifold_baseline(
        n_samples, n_features, config.MANIFOLD_INTRINSIC_DIM, config.MANIFOLD_NOISE_STD, config.RANDOM_SEED
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
    print("\nReconciliation:\n" + reconciliation)
    print(
        f"\nPhase 2 complete. Real data: D2={real_results['correlation']['D2']:.2f}, "
        f"PCA@90%={real_results['pca']['n_components_at_threshold'][0.90]}, "
        f"kNN-MLE mean={real_results['knn']['mean_across_k']:.2f}."
    )


if __name__ == "__main__":
    main()
