# PLAN.md

## Architecture & Workflow Overview

```
Kaggle CSV (801 × 20531)
        │
   [Phase 1] Ingestion & Validation
        │
   [Phase 2] Preprocessing & Normalization
        │
        ├──[Phase 3] Global Spectral Structure (PCA full spectrum)
        │        │
        │        └──[Phase 4] Random Matrix Theory / Marchenko–Pastur comparison
        │
        ├──[Phase 5] Covariance Regularization (Ledoit–Wolf shrinkage)
        │        │
        │        └──[Phase 13] Mahalanobis Anomaly Detection
        │
        ├──[Phase 6] Intrinsic Dim: Correlation Dimension (Grassberger–Procaccia)
        ├──[Phase 7] Intrinsic Dim: MLE (Levina–Bickel) kNN estimator
        ├──[Phase 8] Intrinsic Dim: Fisher Separability
        │        │
        │        └──[Phase 9] Cross-Estimator Synthesis Table (uses 3,6,7,8)
        │
        ├──[Phase 10] Curse-of-Dimensionality Empirics (distance concentration, volume ratios)
        ├──[Phase 11] Concentration of Measure / Johnson–Lindenstrauss random projections
        ├──[Phase 12] Nonlinear Visualization (PCA vs t-SNE vs UMAP)
        │
        └──[Phase 14] AI-Interaction Documentation Log
                 │
        [Phase 15] Report Assembly & Figure Compilation (PDF)
                 │
        [Phase 16] GitHub Packaging & Final Repo Validation
```

Synthetic control matrices (matched n×d Gaussian noise, and a matched-shape low-rank manifold embedded in ambient noise) are generated once in Phase 2 and reused as null baselines throughout Phases 4, 6–10.

---

### Phase 1 — Data Ingestion & Validation
**Phase Objective:** Load the raw Kaggle CSV files (`data.csv`, `labels.csv`) into a validated, reproducible in-memory representation.
**Prerequisites & Inputs:** Downloaded `TCGA-PANCAN-HiSeq-801x20531/data.csv` and `labels.csv` from the Kaggle dataset page.
**Technical Tasks:**
- Load `data.csv` into a DataFrame; drop the sample-ID index column if present.
- Load `labels.csv`, align sample order with `data.csv` via the shared sample-ID column.
- Verify no missing values; verify all feature columns are numeric.
- Persist as `X_raw.npy` (float32, shape (801, 20531)) and `y_labels.npy` (string array of 5 tumor-type codes).
**Target Python Libraries:** `pandas`, `numpy`
**Outputs & Deliverables:** `data/X_raw.npy`, `data/y_labels.npy`, `logs/phase1_validation_report.txt`
**Agent Verification Step:**
```python
assert X_raw.shape == (801, 20531)
assert y_labels.shape == (801,)
assert not np.isnan(X_raw).any()
assert set(np.unique(y_labels)) == {"BRCA","KIRC","COAD","LUAD","PRAD"}
```

---

### Phase 2 — Preprocessing, Normalization & Synthetic Controls
**Phase Objective:** Produce an analysis-ready normalized matrix plus matched-shape synthetic null datasets used as baselines in later phases.
**Prerequisites & Inputs:** `X_raw.npy` from Phase 1.
**Technical Tasks:**
- Apply `log2(x + 1)` transform to stabilize variance across gene expression magnitudes.
- Standardize each gene (column) to zero mean, unit variance (`StandardScaler`).
- Generate `X_gauss_null` — i.i.d. N(0,1) matrix of identical shape (801, 20531), fixed random seed.
- Generate `X_manifold_null` — points sampled from a 10-dimensional linear subspace embedded in ambient dimension 20531 plus small isotropic noise, same sample count.
**Target Python Libraries:** `numpy`, `scikit-learn` (`StandardScaler`)
**Outputs & Deliverables:** `data/X_processed.npy`, `data/X_gauss_null.npy`, `data/X_manifold_null.npy`
**Agent Verification Step:**
```python
assert np.allclose(X_processed.mean(axis=0), 0, atol=1e-6)
assert np.allclose(X_processed.std(axis=0), 1, atol=1e-3)
assert X_gauss_null.shape == X_manifold_null.shape == X_processed.shape
```

---

### Phase 3 — Global Spectral Structure via Full PCA
**Phase Objective:** Compute the complete eigenvalue spectrum of the processed data to establish baseline variance-explained structure.
**Prerequisites & Inputs:** `X_processed.npy`.
**Technical Tasks:**
- Fit `PCA(n_components=min(n,d)-1)` on `X_processed`.
- Record explained variance ratio per component and cumulative variance curve.
- Compute the PCA-based "elbow" dimension: smallest k such that cumulative variance ≥ 90%.
- Compute participation ratio: PR = (Σλᵢ)² / Σλᵢ².
**Target Python Libraries:** `scikit-learn` (`sklearn.decomposition.PCA`), `numpy`, `matplotlib`
**Outputs & Deliverables:** `results/pca_eigenvalues.npy`, `figures/scree_plot.png`, `figures/cumulative_variance.png`, `results/pca_summary.json` (elbow_dim, participation_ratio)
**Agent Verification Step:**
```python
assert len(pca_eigenvalues) == min(800, 20531)
assert pca_summary["elbow_dim"] < 801   # non-trivial dimension reduction
assert os.path.exists("figures/scree_plot.png")
```

---

### Phase 4 — Random Matrix Theory: Marchenko–Pastur Comparison
**Phase Objective:** Determine which eigenvalues of the empirical covariance spectrum exceed pure-noise predictions.
**Prerequisites & Inputs:** `pca_eigenvalues.npy` (Phase 3), `X_gauss_null.npy` (Phase 2).
**Technical Tasks:**
- Compute empirical spectral density of `X_processed` covariance eigenvalues (normalized by trace).
- Compute Marchenko–Pastur theoretical density for aspect ratio γ = d/n using closed-form MP support bounds `[λ₋, λ₊]`.
- Compute empirical eigenvalue histogram of `X_gauss_null` as an empirical sanity check against the theoretical MP curve.
- Count "signal" eigenvalues of the real data exceeding λ₊ (the MP upper edge).
**Target Python Libraries:** `numpy`, `scipy`, `matplotlib`
**Outputs & Deliverables:** `figures/mp_spectrum_overlay.png`, `results/rmt_summary.json` (mp_upper_edge, n_signal_eigenvalues)
**Agent Verification Step:**
```python
assert rmt_summary["n_signal_eigenvalues"] > 0
assert rmt_summary["n_signal_eigenvalues"] < 801
# sanity: null Gaussian data should show near-zero signal eigenvalues
assert rmt_summary["n_signal_eigenvalues_null_check"] <= 5
```

---

### Phase 5 — Covariance Regularization via Ledoit–Wolf Shrinkage
**Phase Objective:** Produce a well-conditioned, invertible covariance estimate usable for downstream Mahalanobis analysis, since the raw sample covariance is rank-deficient (rank ≤ 800 ≪ 20531).
**Prerequisites & Inputs:** `X_processed.npy`.
**Technical Tasks:**
- Fit `sklearn.covariance.LedoitWolf` on `X_processed`.
- Compute and report the shrinkage intensity coefficient.
- Compare condition number of raw sample covariance vs. shrinkage covariance.
**Target Python Libraries:** `scikit-learn` (`sklearn.covariance.LedoitWolf`), `numpy`
**Outputs & Deliverables:** `models/shrinkage_covariance.npy`, `results/shrinkage_summary.json` (shrinkage_coef, cond_number_raw, cond_number_shrunk)
**Agent Verification Step:**
```python
assert 0.0 < shrinkage_summary["shrinkage_coef"] < 1.0
assert shrinkage_summary["cond_number_shrunk"] < shrinkage_summary["cond_number_raw"]
assert np.linalg.matrix_rank(shrinkage_covariance) == 20531  # full rank after shrinkage
```

---

### Phase 6 — Intrinsic Dimension: Correlation Dimension (Grassberger–Procaccia)
**Phase Objective:** Estimate intrinsic dimension D₂ via the correlation integral, on real data and both synthetic controls.
**Prerequisites & Inputs:** `X_processed.npy`, `X_gauss_null.npy`, `X_manifold_null.npy`.
**Technical Tasks:**
- Compute pairwise Euclidean distances (`scipy.spatial.distance.pdist`).
- For a log-spaced grid of radii r, compute correlation integral C(r) = fraction of pairs with distance < r.
- Fit slope of log C(r) vs. log r over the linear-scaling region to obtain D₂.
- Repeat identically for `X_gauss_null` (expect D₂ ≈ ambient d, bounded by n−1) and `X_manifold_null` (expect D₂ ≈ 10).
**Target Python Libraries:** `numpy`, `scipy`, `matplotlib`
**Outputs & Deliverables:** `figures/correlation_dim_loglog.png`, `results/correlation_dimension.json` (D2_real, D2_gauss_null, D2_manifold_null)
**Agent Verification Step:**
```python
assert correlation_dimension["D2_manifold_null"] < 15   # recovers known ~10-dim ground truth
assert correlation_dimension["D2_real"] < correlation_dimension["D2_gauss_null"]
```

---

### Phase 7 — Intrinsic Dimension: Levina–Bickel MLE kNN Estimator
**Phase Objective:** Provide a second, methodologically independent intrinsic-dimension estimate.
**Prerequisites & Inputs:** `X_processed.npy`, `X_gauss_null.npy`, `X_manifold_null.npy`.
**Technical Tasks:**
- Implement Levina–Bickel MLE estimator: for each point, use k nearest-neighbor distances to compute local dimension estimate; average across points and across a range of k (e.g., k = 5..20).
- Report mean and standard deviation of dimension estimate across k values.
- Apply identically to both synthetic controls.
**Target Python Libraries:** `scikit-learn` (`NearestNeighbors`), `numpy`
**Outputs & Deliverables:** `results/mle_dimension.json` (mle_dim_real, mle_dim_gauss_null, mle_dim_manifold_null, per-k breakdown), `figures/mle_dim_vs_k.png`
**Agent Verification Step:**
```python
assert mle_dimension["mle_dim_manifold_null"] < 15
assert mle_dimension["mle_dim_real"] < 801
```

---

### Phase 8 — Intrinsic Dimension: Fisher Separability Analysis
**Phase Objective:** Provide a third intrinsic-dimension estimate based on the fraction of "separable" points under random linear discriminants.
**Prerequisites & Inputs:** `X_processed.npy`, `X_gauss_null.npy`, `X_manifold_null.npy`.
**Technical Tasks:**
- For each point, generate random hyperplanes and measure the fraction of other points linearly separable from it in a local neighborhood.
- Convert the empirical separability probability to an estimated Fisher-separability dimension using the known asymptotic formula for high-dimensional random point configurations.
- Apply identically to both synthetic controls.
**Target Python Libraries:** `numpy`, `scipy`
**Outputs & Deliverables:** `results/fisher_separability.json` (dim_real, dim_gauss_null, dim_manifold_null)
**Agent Verification Step:**
```python
assert fisher_separability["dim_manifold_null"] < 15
assert 0 < fisher_separability["dim_real"] < 20531
```

---

### Phase 9 — Cross-Estimator Synthesis Table
**Phase Objective:** Consolidate all four independent intrinsic-dimension estimates (PCA elbow, correlation dimension, MLE, Fisher separability) into a single comparison table to assess convergence.
**Prerequisites & Inputs:** Outputs of Phases 3, 6, 7, 8.
**Technical Tasks:**
- Build a table: rows = {PCA elbow, Correlation Dim, MLE kNN, Fisher Separability}; columns = {Real data, Gaussian null, Manifold null}.
- Compute pairwise relative disagreement (%) between estimators on real data.
- Flag whether all four estimators agree the real data's intrinsic dimension is substantially below 20531 (the qualitative claim the whole project is testing).
**Target Python Libraries:** `pandas`
**Outputs & Deliverables:** `results/intrinsic_dim_comparison_table.csv`, `figures/intrinsic_dim_comparison_bar.png`
**Agent Verification Step:**
```python
assert os.path.exists("results/intrinsic_dim_comparison_table.csv")
table = pd.read_csv("results/intrinsic_dim_comparison_table.csv")
assert table.shape == (4, 3)
assert (table["Real data"] < 20531).all()
```

---

### Phase 10 — Curse-of-Dimensionality Empirical Verification
**Phase Objective:** Empirically test distance-concentration and volume-ratio phenomena on real vs. synthetic data.
**Prerequisites & Inputs:** `X_processed.npy`, `X_gauss_null.npy`, `X_manifold_null.npy`.
**Technical Tasks:**
- Compute coefficient of variation (std/mean) of pairwise distances for each dataset — lower CV indicates stronger concentration.
- Compute ratio of nearest-neighbor to farthest-neighbor distance per point, averaged.
- Compute the ratio of volume of a unit d-ball to its bounding cube analytically for d = ambient dimension and for the estimated intrinsic dimension from Phase 9, using the Gamma-function volume formula.
**Target Python Libraries:** `numpy`, `scipy.spatial.distance`, `scipy.special` (`gamma`)
**Outputs & Deliverables:** `results/curse_of_dimensionality.json` (cv_real, cv_gauss_null, cv_manifold_null, nn_fn_ratio_real, ball_cube_ratio_ambient, ball_cube_ratio_intrinsic)
**Agent Verification Step:**
```python
assert curse_of_dimensionality["cv_gauss_null"] < curse_of_dimensionality["cv_manifold_null"]
assert curse_of_dimensionality["ball_cube_ratio_ambient"] < curse_of_dimensionality["ball_cube_ratio_intrinsic"]
```

---

### Phase 11 — Concentration of Measure: Johnson–Lindenstrauss Random Projections
**Phase Objective:** Empirically measure how well random projections preserve pairwise distances at decreasing target dimensions, versus the JL theoretical distortion bound.
**Prerequisites & Inputs:** `X_processed.npy`.
**Technical Tasks:**
- Project `X_processed` via `sklearn.random_projection.GaussianRandomProjection` into target dims k ∈ {50, 100, 200, 500, 1000, johnson_lindenstrauss_min_dim(n=801, eps=0.1)}.
- For each k, compute the empirical distribution of pairwise-distance distortion ratios (projected distance / original distance).
- Overlay empirical distortion range against the theoretical (1±ε) JL bound for each k.
**Target Python Libraries:** `scikit-learn` (`sklearn.random_projection`), `numpy`, `matplotlib`
**Outputs & Deliverables:** `figures/jl_distortion_vs_k.png`, `results/jl_summary.json` (per-k empirical distortion min/max/mean)
**Agent Verification Step:**
```python
k_star = jl_summary["johnson_lindenstrauss_min_dim"]
assert jl_summary["distortion"][str(k_star)]["max"] <= 1.15   # within tolerance of eps=0.1 bound
```

---

### Phase 12 — Nonlinear Visualization: PCA vs t-SNE vs UMAP
**Phase Objective:** Produce comparative 2D embeddings to visually assess how well each method preserves known tumor-type structure (exploratory only, not classification).
**Prerequisites & Inputs:** `X_processed.npy`, `y_labels.npy`.
**Technical Tasks:**
- Compute 2D PCA projection, 2D t-SNE (`perplexity=30`), 2D UMAP (`n_neighbors=15, min_dist=0.1`).
- Color each scatter plot by tumor type label.
- Compute a simple quantitative structure-preservation proxy: silhouette score of the true tumor-type labels in each 2D embedding.
**Target Python Libraries:** `scikit-learn` (`PCA`, `TSNE`), `umap-learn`, `matplotlib`
**Outputs & Deliverables:** `figures/pca_2d.png`, `figures/tsne_2d.png`, `figures/umap_2d.png`, `results/embedding_silhouette_scores.json`
**Agent Verification Step:**
```python
assert all(k in embedding_silhouette_scores for k in ["pca","tsne","umap"])
assert all(-1.0 <= v <= 1.0 for v in embedding_silhouette_scores.values())
```

---

### Phase 13 — Mahalanobis-Distance Anomaly Detection
**Phase Objective:** Use the Phase 5 shrinkage covariance (the raw covariance is singular and unusable here) to flag atypical samples within each tumor type.
**Prerequisites & Inputs:** `X_processed.npy`, `y_labels.npy`, `shrinkage_covariance.npy` (Phase 5).
**Technical Tasks:**
- For each tumor type separately, compute per-class mean and reuse the global shrinkage covariance (or per-class shrinkage covariance if sample size per class permits).
- Compute Mahalanobis distance of each sample to its own class mean.
- Flag samples with Mahalanobis distance exceeding the 99th percentile within their class as statistical outliers.
**Target Python Libraries:** `scikit-learn` (`sklearn.covariance.LedoitWolf`), `scipy.spatial.distance` (`mahalanobis`), `numpy`
**Outputs & Deliverables:** `results/mahalanobis_outliers.csv` (sample_id, tumor_type, mahalanobis_distance, is_outlier_flag), `figures/mahalanobis_distance_hist_by_class.png`
**Agent Verification Step:**
```python
assert (mahalanobis_outliers["mahalanobis_distance"] >= 0).all()
assert mahalanobis_outliers["is_outlier_flag"].sum() > 0
assert mahalanobis_outliers["is_outlier_flag"].sum() < 0.05 * len(mahalanobis_outliers)
```

---

### Phase 14 — AI-Interaction Documentation Log
**Phase Objective:** Satisfy the course's mandatory requirement to heavily use and transparently document LLM/Cursor AI usage in code generation.
**Prerequisites & Inputs:** All prompts and AI-generated code diffs used across Phases 1–13.
**Technical Tasks:**
- For every phase, log: (a) the exact prompt(s) issued to the AI tool, (b) a summary of what the AI generated, (c) any manual corrections made and why.
- Store as a structured, chronological log rather than free-form notes.
**Target Python Libraries:** none (markdown/JSON logging only)
**Outputs & Deliverables:** `docs/AI_INTERACTION_LOG.md` (one dated entry per phase, minimum 13 entries)
**Agent Verification Step:**
```python
with open("docs/AI_INTERACTION_LOG.md") as f:
    content = f.read()
assert content.count("## Phase") >= 13
```

---

### Phase 15 — Report Assembly & Figure Compilation
**Phase Objective:** Compile all JSON results and PNG figures from Phases 1–14 into the final PDF report structure required by the course.
**Prerequisites & Inputs:** All `results/*.json`, `results/*.csv`, and `figures/*.png` from Phases 1–14; `docs/AI_INTERACTION_LOG.md`.
**Technical Tasks:**
- Assemble report sections in order: Dataset justification → Global spectral structure (PCA+RMT) → Covariance regularization → Intrinsic dimension synthesis (4-estimator table) → Curse-of-dimensionality findings → JL concentration results → Visualization comparison → Mahalanobis anomaly findings → AI-tool usage disclosure.
- Embed each phase's key figure(s) and summary JSON values as inline tables/numbers in the corresponding section.
- Render to PDF.
**Target Python Libraries:** `matplotlib`, `pandas`, report rendering via `markdown`-to-PDF toolchain (e.g., `weasyprint` or LaTeX via `pandoc`)
**Outputs & Deliverables:** `report/final_report.pdf`
**Agent Verification Step:**
```python
assert os.path.exists("report/final_report.pdf")
assert os.path.getsize("report/final_report.pdf") > 500_000  # non-trivial, figure-rich report
```

---

### Phase 16 — GitHub Packaging & Final Repository Validation
**Phase Objective:** Produce the exact deliverable structure required by the course's submission guidelines.
**Prerequisites & Inputs:** Entire project directory from Phases 1–15.
**Technical Tasks:**
- Organize repository into `data/`, `results/`, `figures/`, `docs/`, `report/`, `src/` (all phase scripts), `README.md`.
- `README.md` must state dataset source, project goal, how to reproduce each phase, and link to `docs/AI_INTERACTION_LOG.md`.
- Verify no large raw data files exceed GitHub's file-size limits (use Git LFS or a data-download script instead of committing the raw CSV if >100MB).
**Target Python Libraries:** none (repository/file-structure task)
**Outputs & Deliverables:** Public GitHub repository containing `README.md`, `src/`, `report/final_report.pdf`, `docs/AI_INTERACTION_LOG.md`
**Agent Verification Step:**
```python
required = ["README.md","src","report/final_report.pdf","docs/AI_INTERACTION_LOG.md"]
assert all(os.path.exists(p) for p in required)
assert not any(os.path.getsize(f) > 100_000_000 for f in glob.glob("data/**", recursive=True) if os.path.isfile(f))
```