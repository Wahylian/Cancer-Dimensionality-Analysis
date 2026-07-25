"""Phase 2: cross-estimator summary tables and reconciliation.

Consolidates the three intrinsic-dimension estimators (correlation, PCA,
k-NN MLE) across the real dataset and both synthetic baselines into one
comparison table, per the mandatory side-by-side calibration requirement.
"""

import pandas as pd

from src import config


def build_intrinsic_dimension_summary_table(real_results: dict, synthetic_results: dict) -> pd.DataFrame:
    """Consolidate per-dataset estimator results into one comparison table.

    Parameters
    ----------
    real_results : dict
        Keys 'correlation', 'pca', 'knn' holding the return values of
        `estimate_correlation_dimension`, `pca_based_dimension`, and
        `knn_mle_dimension` respectively, computed on the real data.
    synthetic_results : dict
        Maps baseline name (e.g. "Gaussian Noise", "Linear Manifold") to a
        dict with the same 'correlation'/'pca'/'knn' structure.

    Returns
    -------
    pd.DataFrame indexed by dataset name, one column per estimator/setting.
    """
    all_datasets = {"Real (RNA-Seq)": real_results, **synthetic_results}

    records = []
    for name, res in all_datasets.items():
        record = {
            "Dataset": name,
            "Correlation dim (D2)": round(res["correlation"]["D2"], 2),
            "Corr. R2": round(res["correlation"]["r_squared"], 3),
        }
        for t in config.PCA_VARIANCE_THRESHOLDS:
            record[f"PCA @ {int(t * 100)}%"] = res["pca"]["n_components_at_threshold"][t]
        for k in config.KNN_MLE_K_VALUES:
            record[f"kNN-MLE (k={k})"] = round(res["knn"]["per_k"][k]["mean_dimension"], 2)
        record["kNN-MLE mean"] = round(res["knn"]["mean_across_k"], 2)
        records.append(record)

    return pd.DataFrame(records).set_index("Dataset")


def reconcile_estimators(summary_table: pd.DataFrame) -> str:
    """Produce a written reconciliation of disagreement between estimators.

    Compares the real-data row's correlation dimension, k-NN MLE mean, and
    PCA @ 90%-variance component count, and explains the gap between them
    with reference to known estimator biases.
    """
    real = summary_table.loc["Real (RNA-Seq)"]
    pca_col = f"PCA @ {int(config.PCA_VARIANCE_THRESHOLDS[0] * 100)}%"

    return (
        f"On the real RNA-Seq data, the correlation dimension estimator gives "
        f"D2 = {real['Correlation dim (D2)']:.2f}, the k-NN MLE estimator gives a "
        f"mean of {real['kNN-MLE mean']:.2f} across k = {config.KNN_MLE_K_VALUES}, while "
        f"the PCA-based linear estimator requires {real[pca_col]} components to reach "
        f"{int(config.PCA_VARIANCE_THRESHOLDS[0] * 100)}% variance. These estimators are "
        "not expected to agree: the correlation dimension and k-NN MLE both probe local "
        "neighborhood geometry and are known to be negatively biased when the number of "
        "samples is small relative to the ambient dimension, since pairwise distances "
        "become sparse and concentrate, starving the small-radius scaling region of "
        "pairs. PCA, in contrast, is a purely linear, global estimator: it counts every "
        "direction of non-negligible variance, including nonlinear structure that gets "
        "spread across many linear components, and non-biological technical variance, so "
        "it systematically reports a much larger 'dimension' than the local nonlinear "
        "estimators. The synthetic Gaussian-noise baseline, which has no low-dimensional "
        "structure by construction, calibrates the upper end of this range, while the "
        "linear-manifold baseline (true dimension "
        f"{config.MANIFOLD_INTRINSIC_DIM}) calibrates how closely each estimator can "
        "recover a known ground truth under additive noise."
    )
