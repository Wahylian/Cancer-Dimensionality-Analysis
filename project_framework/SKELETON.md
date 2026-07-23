# SKELETON.md — Technical Blueprint

**Project:** Intrinsic Dimensionality and Geometric Structure in Pan-Cancer RNA-Seq Data
**Dataset:** Gene Expression Cancer RNA-Seq — https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq

This is an architectural blueprint, not a solution. Function signatures, docstrings, and TODOs are provided to structure your implementation; the actual algorithmic logic (marked `# TODO`) is left for you (and your AI tooling, per course policy) to implement and defend.

---

## 1. Project Architecture

```
hdp-rnaseq-project/
├── README.md
├── requirements.txt
├── .gitignore                      # excludes /data/raw
├── data/
│   ├── raw/                        # downloaded, gitignored
│   └── processed/                 # cached intermediate artifacts (npz/parquet)
├── src/
│   ├── __init__.py
│   ├── config.py                   # seeds, paths, hyperparameter defaults
│   ├── data_loading.py             # Phase 0/1: ingestion + validation
│   ├── preprocessing.py            # Phase 1: transform, scale, filter
│   ├── synthetic.py                # Phase 2: synthetic baseline generators
│   ├── intrinsic_dimension.py      # Phase 2: correlation dim, PCA-based, kNN-based
│   ├── embeddings.py               # Phase 3: PCA / t-SNE / UMAP wrappers
│   ├── random_projection.py        # Phase 3: JL lemma experiments
│   ├── spectral_analysis.py        # Phase 3: covariance spectrum + Marchenko-Pastur
│   ├── geometry_diagnostics.py     # Phase 3: distance concentration, concentration bounds
│   ├── anomaly_detection.py        # Phase 3: shrinkage covariance + Mahalanobis distance
│   ├── evaluation.py               # Phase 4: summary tables, cross-method reconciliation
│   └── viz.py                      # shared plotting utilities
├── notebooks/
│   └── 00_exploration.ipynb        # optional scratch exploration only
├── figures/                        # all saved plots referenced by the report
├── reports/
│   └── project_report.pdf
├── ai_usage_log.md                 # mandatory AI documentation (mirrors report Appendix A)
└── run_pipeline.py                 # top-level orchestration script
```

**Module responsibility principle:** each `src/*.py` module corresponds to exactly one phase (or sub-phase) from `INSTRUCTIONS.md`. `run_pipeline.py` should import and call these modules in sequence and should contain no analysis logic itself — only orchestration, logging, and figure/table export.

---

## 2. Pipeline Workflow

```
                ┌─────────────────────┐
                │  data_loading.py    │  → raw X (801x20531), y (801,)
                └──────────┬──────────┘
                           ▼
                ┌─────────────────────┐
                │  preprocessing.py   │  → X_log, X_scaled, gene_filter_mask
                └──────────┬──────────┘
                           ▼
        ┌──────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────┐                ┌────────────────────────┐
│   synthetic.py     │                │ intrinsic_dimension.py │
│ (Gaussian noise,    │───baselines──▶│ correlation_dim()       │
│  low-dim manifold)  │                │ pca_dimension()         │
└─────────────────────┘                │ knn_dimension()         │
                                        └────────────┬────────────┘
                                                      ▼
                                        ┌────────────────────────┐
                                        │  embeddings.py           │
                                        │  random_projection.py    │
                                        │  spectral_analysis.py     │
                                        │  geometry_diagnostics.py  │
                                        │  anomaly_detection.py     │
                                        └────────────┬────────────┘
                                                      ▼
                                        ┌────────────────────────┐
                                        │    evaluation.py         │
                                        │  (summary tables, cross- │
                                        │   phase reconciliation)  │
                                        └────────────┬────────────┘
                                                      ▼
                                        ┌────────────────────────┐
                                        │   figures/ + reports/    │
                                        └────────────────────────┘
```

Execution order (`run_pipeline.py`, high level):
1. Load & validate raw data → assert shape `(801, 20531)`, check NaNs/zero-variance columns.
2. Preprocess (log transform → standardize → optional variance filter), cache to `data/processed/`.
3. Generate synthetic baselines (noise + manifold) matched in shape to the real data.
4. Run all three intrinsic dimension estimators on real data **and** both synthetic baselines.
5. Run embeddings (PCA/t-SNE/UMAP), random projection/JL experiment, spectral analysis, distance-concentration diagnostics, and Mahalanobis anomaly detection — each independent and cacheable.
6. Aggregate everything into summary tables/figures for the report.
7. Log every run's parameters (seed, hyperparameters) to a run manifest for reproducibility.

---

## 3. Key Functions & Classes

### `src/config.py`
```python
RANDOM_SEED: int = 42
DATA_DIR = "data/"
FIGURES_DIR = "figures/"
N_TOP_VARIABLE_GENES = 2000     # used only where full 20531-dim computation is intractable
CLASS_LABELS = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]
```

### `src/data_loading.py`
```python
def load_expression_matrix(data_path: str) -> "pd.DataFrame":
    """Load the (801 x 20531) RNA-Seq expression matrix.

    Returns
    -------
    DataFrame indexed by sample id, columns gene_0..gene_20530.
    """
    # TODO: read CSV/parquet, assert expected shape, return DataFrame

def load_labels(labels_path: str) -> "pd.Series":
    """Load per-sample cancer-type labels aligned to the expression matrix index."""
    # TODO

def validate_dataset(X: "pd.DataFrame", y: "pd.Series") -> dict:
    """Run integrity checks: shape assertions, NaN count, zero-variance gene count,
    class balance. Returns a dict summary suitable for logging into the report.
    """
    # TODO
```

### `src/preprocessing.py`
```python
def log_transform(X: "np.ndarray") -> "np.ndarray":
    """Apply log1p to raw expression counts. Justify choice in report re: skewness."""
    # TODO

def standardize(X: "np.ndarray") -> "np.ndarray":
    """Zero-mean, unit-variance scale each gene (column). Must run after log_transform."""
    # TODO

def filter_top_variable_genes(X: "np.ndarray", k: int) -> tuple:
    """Return (X_filtered, selected_gene_indices) keeping the k highest-variance genes.
    Used only for computationally expensive sub-analyses (e.g., spectral analysis);
    full-dimensional results must also be reported where feasible per INSTRUCTIONS.md.
    """
    # TODO
```

### `src/synthetic.py`
```python
def generate_gaussian_noise_baseline(n_samples: int, n_features: int, seed: int) -> "np.ndarray":
    """High-dimensional isotropic Gaussian noise; expected correlation dimension ~ n_features."""
    # TODO

def generate_low_dim_manifold_baseline(
    n_samples: int, ambient_dim: int, intrinsic_dim: int, noise_std: float, seed: int
) -> "np.ndarray":
    """Sample points on a known `intrinsic_dim`-dimensional linear subspace embedded in
    `ambient_dim` dimensions, plus small Gaussian noise, to calibrate estimators against
    a known ground-truth intrinsic dimension.
    """
    # TODO
```

### `src/intrinsic_dimension.py`
```python
def correlation_integral(X: "np.ndarray", radii: "np.ndarray") -> "np.ndarray":
    """Compute C(r) for each r in `radii`: fraction of point-pairs with pairwise
    distance < r. Returns array of C(r) values same length as `radii`.

    Implementation notes:
    - Consider subsampling pairs if N is large enough that all-pairs is infeasible
      (not a concern at N=801, but document your pairwise-distance memory budget).
    """
    # TODO

def estimate_correlation_dimension(
    X: "np.ndarray", radii: "np.ndarray", scaling_region: tuple
) -> dict:
    """Fit a line to log C(r) vs log r restricted to `scaling_region` (indices or
    radius bounds into `radii`). Returns {'D2': slope, 'r_squared': fit quality,
    'scaling_region': scaling_region}. The choice of scaling_region MUST be justified
    in the report (e.g., via a plateau in the local slope estimate).
    """
    # TODO

def pca_based_dimension(X: "np.ndarray", variance_thresholds: list) -> dict:
    """Return the number of principal components required to reach each variance
    threshold (e.g., [0.90, 0.95, 0.99]), plus the full eigenvalue spectrum for
    scree-plot generation.
    """
    # TODO

def knn_mle_dimension(X: "np.ndarray", k_values: list) -> dict:
    """Levina-Bickel (or equivalent) k-NN maximum-likelihood intrinsic dimension
    estimator, evaluated at each k in `k_values`. Return per-k estimates and their
    mean/variance to characterize sensitivity to k.
    """
    # TODO
```

### `src/embeddings.py`
```python
def run_pca(X: "np.ndarray", n_components: int) -> dict:
    """Return {'embedding': ..., 'explained_variance_ratio': ...}."""
    # TODO

def run_tsne(X: "np.ndarray", perplexities: list, seed: int) -> dict:
    """Return {perplexity: 2D embedding} for each value in `perplexities`."""
    # TODO

def run_umap(X: "np.ndarray", n_neighbors_values: list, seed: int) -> dict:
    """Return {n_neighbors: 2D embedding} for each value in `n_neighbors_values`."""
    # TODO
```

### `src/random_projection.py`
```python
def jl_target_dimension(n_samples: int, epsilon: float) -> int:
    """Compute the Johnson-Lindenstrauss target dimension for `n_samples` points
    and allowed distortion `epsilon`, per the standard JL bound.
    """
    # TODO

def random_gaussian_projection(X: "np.ndarray", target_dim: int, seed: int) -> "np.ndarray":
    """Project X onto `target_dim` dimensions using a random Gaussian matrix
    (scaled appropriately to approximately preserve norms).
    """
    # TODO

def evaluate_distance_preservation(X: "np.ndarray", X_proj: "np.ndarray") -> dict:
    """Compare all-pairs distances in X vs X_proj; return empirical distortion
    distribution (ratios of projected/original distances) and summary stats
    (mean, max deviation from 1) to compare against the theoretical (1±epsilon) bound.
    """
    # TODO
```

### `src/spectral_analysis.py`
```python
def empirical_spectral_distribution(X_scaled: "np.ndarray") -> "np.ndarray":
    """Return eigenvalues of the sample covariance/correlation matrix of X_scaled
    (expected to be computed on a feature subset for tractability — document choice).
    """
    # TODO

def marchenko_pastur_density(aspect_ratio: float, eigenvalue_grid: "np.ndarray") -> "np.ndarray":
    """Evaluate the theoretical Marcenko-Pastur density over `eigenvalue_grid` for
    the given aspect ratio (p/n), for overlay comparison against the empirical
    spectrum. Identify spectral outliers beyond the theoretical support edge.
    """
    # TODO
```

### `src/geometry_diagnostics.py`
```python
def pairwise_distance_concentration(X: "np.ndarray", feature_counts: list, seed: int) -> dict:
    """For each value in `feature_counts`, subsample that many features (columns),
    compute all pairwise distances, and return the ratio
    (max_dist - min_dist) / min_dist as a concentration diagnostic.
    Expect this ratio to shrink as feature_counts grows if concentration-of-measure holds.
    """
    # TODO

def theoretical_concentration_bound(feature_count: int, epsilon: float) -> float:
    """Return a Hoeffding- or Chernoff-type theoretical bound on deviation probability
    for comparison against the empirical concentration observed above.
    """
    # TODO
```

### `src/anomaly_detection.py`
```python
def shrinkage_covariance(X: "np.ndarray") -> "np.ndarray":
    """Ledoit-Wolf (or equivalent) shrinkage covariance estimate; required since the
    raw sample covariance is singular when d >> n.
    """
    # TODO

def mahalanobis_outliers(X: "np.ndarray", cov_estimate: "np.ndarray", threshold: float) -> dict:
    """Compute Mahalanobis distance per sample using `cov_estimate`; flag samples
    above `threshold` (e.g., a chi-squared quantile at chosen alpha). Return
    distances, flagged indices, and a cross-tab against class labels for report Table X.
    """
    # TODO
```

### `src/evaluation.py`
```python
def build_intrinsic_dimension_summary_table(real_results: dict, synthetic_results: dict) -> "pd.DataFrame":
    """Consolidate correlation/PCA/kNN intrinsic dimension estimates for the real
    dataset alongside both synthetic baselines into one comparison table
    (rows = dataset, columns = estimator).
    """
    # TODO

def reconcile_estimators(summary_table: "pd.DataFrame") -> str:
    """Produce a short written reconciliation (returned as text, to be pasted/adapted
    into the report) explaining any disagreement between estimators, referencing
    known estimator biases (e.g., correlation dimension underestimates in sparse
    high-dimensional samples; kNN-MLE sensitivity to k; PCA capturing only linear
    structure).
    """
    # TODO
```

### `src/viz.py`
```python
def scatter_embedding(embedding: "np.ndarray", labels: "np.ndarray", title: str, save_path: str) -> None:
    """Standardized 2D/3D scatter plot colored by class label, saved to `save_path`."""
    # TODO

def scree_plot(eigenvalues: "np.ndarray", save_path: str) -> None:
    # TODO

def log_log_correlation_plot(radii: "np.ndarray", C_r: "np.ndarray", scaling_region: tuple, save_path: str) -> None:
    # TODO

def spectral_overlay_plot(empirical_eigs: "np.ndarray", mp_density_x: "np.ndarray", mp_density_y: "np.ndarray", save_path: str) -> None:
    # TODO
```

---

## 4. Orchestration Skeleton (`run_pipeline.py`)

```python
from src import (
    config, data_loading, preprocessing, synthetic,
    intrinsic_dimension, embeddings, random_projection,
    spectral_analysis, geometry_diagnostics, anomaly_detection,
    evaluation, viz,
)

def main():
    # Phase 0/1
    X_raw = data_loading.load_expression_matrix(...)
    y = data_loading.load_labels(...)
    data_loading.validate_dataset(X_raw, y)
    X_log = preprocessing.log_transform(X_raw.values)
    X_scaled = preprocessing.standardize(X_log)

    # Phase 2
    noise_baseline = synthetic.generate_gaussian_noise_baseline(...)
    manifold_baseline = synthetic.generate_low_dim_manifold_baseline(...)
    # TODO: run all three estimators on X_scaled, noise_baseline, manifold_baseline
    # TODO: evaluation.build_intrinsic_dimension_summary_table(...)

    # Phase 3
    # TODO: embeddings.run_pca / run_tsne / run_umap on X_scaled
    # TODO: random_projection.* experiment
    # TODO: spectral_analysis.* on a documented feature subset
    # TODO: geometry_diagnostics.* concentration experiment
    # TODO: anomaly_detection.* Mahalanobis pipeline

    # Phase 4
    # TODO: assemble all summary tables/figures into reports/ and figures/

if __name__ == "__main__":
    main()
```

**Note:** this skeleton intentionally leaves all algorithmic bodies as `# TODO`. Filling them in — correctly, efficiently, and with justified hyperparameters — constitutes the graded technical work of the project.
