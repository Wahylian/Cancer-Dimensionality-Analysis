"""Phase 0/1: dataset ingestion and integrity validation."""

import pandas as pd


def load_expression_matrix(data_path: str) -> pd.DataFrame:
    """Load the (801 x 20531) RNA-Seq expression matrix.

    Returns
    -------
    DataFrame indexed by sample id, columns gene_0..gene_20530.
    """
    X = pd.read_csv(data_path, index_col=0)
    if X.shape != (801, 20531):
        raise ValueError(f"Expected shape (801, 20531), got {X.shape}")
    return X


def load_labels(labels_path: str) -> pd.Series:
    """Load per-sample cancer-type labels aligned to the expression matrix index."""
    y = pd.read_csv(labels_path, index_col=0)["Class"]
    if y.shape[0] != 801:
        raise ValueError(f"Expected 801 labels, got {y.shape[0]}")
    return y


def validate_dataset(X: pd.DataFrame, y: pd.Series) -> dict:
    """Run integrity checks: shape assertions, NaN count, zero-variance gene count,
    class balance. Returns a dict summary suitable for logging into the report.
    """
    assert X.shape[0] == y.shape[0], "X and y sample counts must match"
    assert list(X.index) == list(y.index), "X and y sample ids must be aligned"

    n_missing = int(X.isna().sum().sum())
    zero_variance_mask = X.var(axis=0) == 0
    n_zero_variance = int(zero_variance_mask.sum())
    class_counts = y.value_counts()

    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_missing_values": n_missing,
        "n_zero_variance_genes": n_zero_variance,
        "zero_variance_gene_names": X.columns[zero_variance_mask].tolist(),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "class_proportions": {str(k): round(float(v), 4) for k, v in (class_counts / len(y)).items()},
    }
