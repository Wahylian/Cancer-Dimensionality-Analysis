# AI Usage Log

Per `project_framework/INSTRUCTIONS.md` Section 4.4 (mandatory AI Usage Log,
mirrored as report Appendix A). This log is updated incrementally as each
phase is completed.

## Tool

**Claude Code** (Anthropic), model Claude Sonnet 5, CLI agent with direct
read/write/execute access to this repository. Used for the majority of code
generation, the Phase 1 LaTeX report, and this log itself.

## Phase 0 — Environment & Reproducibility

Completed in a prior session (git commit `8b80b37`, "done phase 0 of
instructions"). Produced `requirements.txt`, `.gitignore`, `download_data.py`,
`src/config.py`, and `README.md`. Prompt transcript from that session was not
retained in this log; the resulting files are reviewed and taken as given for
Phase 1.

## Phase 1 — Preprocessing & Cleaning

**Files/functions generated with AI assistance:**
- `src/data_loading.py` — `load_expression_matrix()`, `load_labels()`,
  `validate_dataset()`
- `src/preprocessing.py` — `log_transform()`, `drop_zero_variance_genes()`,
  `standardize()`, `filter_top_variable_genes()`
- `src/viz.py` — `plot_class_balance()`, `plot_gene_mean_variance()` (Phase 2/3
  functions left as `NotImplementedError` stubs, out of scope for this phase)
- `run_pipeline.py` — Phase 1 orchestration
- `reports/phase_1.tex` — full Phase 1 methodology/results report
- `src/config.py` — added `RAW_DATA_PATH`, `RAW_LABELS_PATH`, `PROCESSED_DIR`
- `.gitignore` — added `data/processed/`

**Representative prompts (paraphrased from the driving instruction set):**
1. "Ingest and inspect `IDEA.md`, `INSTRUCTIONS.md`, and `SKELETON.md`; map
   Phase 1 requirements in `INSTRUCTIONS.md` to the module definitions in
   `SKELETON.md`."
2. "Implement `src/data_loading.py`: load the (801 x 20,531) expression matrix
   and label vector, assert shapes, check for missing values, validate class
   balance."
3. "Implement `src/preprocessing.py`: log1p transform, standardization
   (zero mean, unit variance) applied *after* the log transform, and
   variance-based feature filtering retaining the top-k most variable genes."
4. "Update `run_pipeline.py` to orchestrate loading, validation, and
   preprocessing; save intermediate processed artifacts to `data/processed/`;
   verify the codebase runs cleanly."
5. "Generate Phase 1 exploratory artifacts to `figures/`: a class-balance
   table/chart and a per-gene mean/variance histogram, computed on both the
   full 20,531-feature matrix and the reduced top-k subset."
6. "Author a standalone, publication-quality LaTeX report
   (`reports/phase_1.tex`) with dedicated justification paragraphs for the
   log transform, standardization order, and the zero-variance/variance-
   filtering decisions."

**What was changed, corrected, or rejected from the AI-drafted output, and why:**
- *Zero-variance gene handling was added as a new function not present in
  `SKELETON.md`* (`drop_zero_variance_genes()`). The skeleton's
  `standardize()` signature has no provision for this, but standardizing raw
  log-transformed data with zero-variance columns divides by zero. This was
  identified as a correctness requirement (267 of 20,531 genes are constant
  across all 801 samples in this dataset) and added as a minimal, explicit
  preprocessing step rather than silently guarding inside `standardize()`.
- *`filter_top_variable_genes()` was deliberately applied to log-scale data,
  not standardized data.* An initial implementation draft ranked variance
  after standardization; this was rejected because every standardized column
  has unit variance by construction, making post-standardization variance
  ranking meaningless. The function is now documented to require log-scale
  input.
- *Phase 2/3 stub functions in `src/viz.py`* (`scree_plot`,
  `scatter_embedding`, `log_log_correlation_plot`, `spectral_overlay_plot`)
  were kept as `NotImplementedError` stubs matching the skeleton's
  signatures, rather than implemented, to respect the Phase 1 scope boundary
  explicitly set for this task.
- *`run_pipeline.py` imports only the modules that exist* (`config`,
  `data_loading`, `preprocessing`, `viz`), diverging from the skeleton's
  illustrative top-level import list, which also names Phase 2--4 modules
  (`synthetic`, `intrinsic_dimension`, `embeddings`, etc.) that are not yet
  implemented. Importing non-existent modules would break the script; those
  imports will be added phase-by-phase as the corresponding modules are
  written.
- *Data artifact storage format*: chose `np.savez_compressed` with
  `float32` arrays (rather than CSV/parquet) for `data/processed/*.npz` to
  keep the ~801 x 20,264 standardized matrix compact (~53 MB) while
  remaining trivially loadable by later-phase modules via NumPy.
- Verified end-to-end by running `run_pipeline.py` against the real
  downloaded dataset and inspecting the printed validation summary and all
  three generated figures before accepting the output.

## Phase 2 — Intrinsic Dimension Estimation

**Files/functions generated with AI assistance:**
- `src/synthetic.py` — `generate_gaussian_noise_baseline()`,
  `generate_low_dim_manifold_baseline()`
- `src/intrinsic_dimension.py` — `correlation_integral()`, `default_radii()`,
  `_auto_detect_scaling_region()`, `estimate_correlation_dimension()`,
  `_kaiser_elbow()`, `_curvature_elbow()`, `pca_based_dimension()`,
  `knn_mle_dimension()`
- `src/evaluation.py` — `build_intrinsic_dimension_summary_table()`,
  `reconcile_estimators()`
- `src/viz.py` — `scree_plot()`, `log_log_correlation_plot()` (the two
  remaining Phase 2 stubs; `scatter_embedding()` and
  `spectral_overlay_plot()` are left as Phase 3 `NotImplementedError` stubs)
- `run_pipeline.py` — Phase 2 orchestration (`_estimate_all()` helper plus
  synthetic-baseline generation, figure export, summary table/reconciliation
  export)
- `src/config.py` — added `CORR_DIM_*`, `PCA_VARIANCE_THRESHOLDS`,
  `KNN_MLE_*`, `SYNTHETIC_NOISE_STD`, `MANIFOLD_INTRINSIC_DIM`,
  `MANIFOLD_NOISE_STD`
- `reports/phase_2.tex` — full Phase 2 methodology/results report

**Representative prompts (paraphrased from the driving instruction set):**
1. "Implement the Grassberger-Procaccia correlation integral natively via
   `scipy.spatial.distance.pdist`, computing exact distances once and
   vectorizing across a log-spaced radii array; auto-detect the linear
   scaling region via the plateau in the local log-log slope and justify
   the choice in the report."
2. "Implement the PCA-based estimator via the full eigenvalue spectrum,
   avoiding ever forming the (20531 x 20531) covariance matrix; report
   components needed for 90/95/99% variance and two elbow criteria (Kaiser,
   second-derivative curvature)."
3. "Implement the Levina-Bickel k-NN MLE estimator exactly per the given
   formula with a (k-2) divisor, swept over k=[5,10,20], with an epsilon
   guard against ln(1)=0 from distance ties under high-dimensional
   concentration."
4. "Implement synthetic baseline generators: isotropic Gaussian noise and a
   known-dimension linear subspace embedded in ambient space with additive
   noise, matched in shape to the real preprocessed data; run all three
   estimators on both baselines side by side with the real data in one
   summary table (`evaluation.py`)."
5. "Run the full pipeline, verify `figures/scree_plot.png` and
   `figures/log_log_correlation.png` are produced, and debug any shape or
   numerical-stability issues before reporting results."
6. "Author `reports/phase_2.tex` to the same standard as Phase 1, with a
   dedicated methodological-justification paragraph for every hyperparameter
   choice and a written reconciliation of the three estimators' disagreement
   on the real data."

**What was changed, corrected, or rejected from the AI-drafted output, and why:**
- *The manifold baseline's noise standard deviation was corrected after an
  empirical sensitivity check, not assumed.* An initial draft used
  `MANIFOLD_NOISE_STD = 0.1`, chosen only because it "looked small." Running
  the pipeline showed this made the linear-manifold baseline
  indistinguishable from pure Gaussian noise across all three estimators
  (e.g. PCA@90% = 686 vs. 688 for noise, k-NN MLE mean ~522-619 for both).
  Diagnosis: with an orthonormal embedding basis, isotropic ambient noise of
  variance $\sigma^2$ adds $d\sigma^2$ of *aggregate* variance across all
  20,264 coordinates, which swamps the ~5 units of true signal variance for
  any $\sigma$ that isn't tiny relative to $d$, not just relative to the
  signal's own scale. A manual sweep (`0.1 -> 0.01 -> 0.001`) was run and
  recorded in the report (Table "Manifold-baseline sensitivity") before
  settling on `MANIFOLD_NOISE_STD = 0.001`, at which all three estimators
  recover the true dimension of 5 closely. This is flagged explicitly in
  `reports/phase_2.tex` as a finding in its own right, not hidden as a
  silent parameter tweak.
- *`intrinsic_dimension.py` exceeds the project's soft ~200-line-per-file
  guideline (222 lines) after adding `default_radii()`.* This was a
  deliberate rejection of splitting the file: `SKELETON.md` explicitly
  designs this module to hold all three estimators (correlation, PCA,
  k-NN) as one cohesive Phase-2 unit, and `default_radii()` is intrinsically
  part of the correlation-dimension methodology (radius range selection),
  not a separate concern. Splitting it out would have fragmented one
  estimator's logic across files for a ~10% line-count overshoot.
- *Radius quantiles for the correlation integral are estimated from a
  300-point random subsample, not the full 801-point pairwise-distance
  vector.* An initial draft computed the full `pdist` twice per dataset
  (once for quantile estimation, once inside `correlation_integral`),
  roughly doubling a ~29s-per-dataset cost for no accuracy benefit, since
  only approximate quantile bounds are needed to pick a radius range. This
  was reduced to a cheap subsample specifically to keep the full
  three-dataset x three-estimator pipeline under ~2 minutes.
- *Levina-Bickel used the `(k-2)` divisor exactly as specified*, not the
  more commonly cited `(k-1)` normalization seen in some derivations of
  the estimator, per the explicit worked formula in the driving
  instructions; `assert k >= 3` was added since the sweep values
  (5, 10, 20) all satisfy this but the function would otherwise divide by
  zero or a negative number for smaller k.
- Verified end-to-end by running `run_pipeline.py` against the real
  preprocessed data, inspecting the printed Phase 2 summary table and
  reconciliation text, visually reviewing both generated figures
  (`scree_plot.png`, `log_log_correlation.png`), and independently
  sanity-checking the correlation-dimension implementation against a pure
  5-D Gaussian control (no ambient embedding) to confirm the estimator
  itself was correct before attributing the manifold-baseline anomaly to
  the noise-level parameter rather than a bug.

## Phase 3 — Dimensionality Reduction, Visualization & High-Dimensional Geometry

**Files/functions generated with AI assistance:**
- `src/embeddings.py` — `run_pca()`, `_pca_preprocess()`, `run_tsne()`, `run_umap()`
- `src/random_projection.py` — `jl_target_dimension()`,
  `random_gaussian_projection()`, `evaluate_distance_preservation()`
- `src/spectral_analysis.py` — `empirical_spectral_distribution()`,
  `marchenko_pastur_density()`, `marchenko_pastur_support()`,
  `identify_spectral_outliers()`
- `src/geometry_diagnostics.py` — `pairwise_distance_concentration()`,
  `theoretical_concentration_bound()`
- `src/anomaly_detection.py` — `shrinkage_covariance()`,
  `chi_squared_threshold()`, `mahalanobis_outliers()`
- `src/viz.py` — `scatter_embedding()`, `embedding_hyperparam_comparison_plot()`,
  `spectral_overlay_plot()` (rewritten from a single-panel stub, see below),
  `jl_distortion_plot()`, `distance_concentration_plot()`,
  `mahalanobis_distribution_plot()`
- `run_pipeline.py` — Phase 3 orchestration (PCA/t-SNE/UMAP visualization,
  JL projection experiment, spectral analysis, distance concentration,
  Mahalanobis anomaly detection; figure and summary-artifact export)
- `src/config.py` — added `PCA_VIZ_N_COMPONENTS`, `EMBEDDING_PCA_PREPROCESS_DIM`,
  `TSNE_PERPLEXITIES`, `UMAP_N_NEIGHBORS_VALUES`, `JL_EPSILON`,
  `SPECTRAL_N_TOP_GENES`, `GEOMETRY_FEATURE_COUNTS`,
  `GEOMETRY_CONCENTRATION_EPSILON`, `MAHALANOBIS_ALPHA`
- `reports/phase_3.tex` — full Phase 3 methodology/results report

**Representative prompts (paraphrased from the driving instruction set):**
1. "Implement `src/embeddings.py`: PCA via exact SVD for 2D/3D visualization,
   and t-SNE/UMAP wrappers, each swept over at least two hyperparameter
   values, all seeded from `config.RANDOM_SEED`."
2. "Implement the Johnson-Lindenstrauss target-dimension formula and a
   random Gaussian projection, then empirically evaluate pairwise-distance
   distortion against the theoretical `(1 +/- epsilon)` bound."
3. "Implement the empirical spectral distribution of a tractable gene
   subset's correlation matrix and the theoretical Marchenko-Pastur density
   for overlay comparison, including the point-mass-at-zero case for
   aspect ratio > 1, and flag eigenvalues beyond the theoretical support
   edge as outliers."
4. "Implement pairwise distance concentration across feature-subsample
   sizes and a theoretical Hoeffding- or Chernoff-type bound for
   comparison, justified explicitly since Hoeffding itself requires
   bounded variables that standardized RNA-Seq features do not satisfy."
5. "Implement Ledoit-Wolf shrinkage covariance and chi-squared-thresholded
   Mahalanobis outlier detection; the raw sample covariance must never be
   used directly given d >> n."
6. "Run the full pipeline, inspect every generated Phase 3 figure visually
   before writing the report, and cite only the real printed/computed
   numbers -- no fabricated metrics."
7. "Author `reports/phase_3.tex` to the same standard as Phases 1-2, with a
   dedicated methodological-justification paragraph for every hyperparameter
   choice (t-SNE/UMAP settings, JL epsilon, spectral feature-subset size,
   Mahalanobis threshold) and a discussion section reconciling Phase 3
   findings with Phase 2's intrinsic-dimension estimates."

**What was changed, corrected, or rejected from the AI-drafted output, and why:**
- *`JL_EPSILON` was corrected from an initial draft value of `0.1` to
  `0.2`, based on computing the actual target dimension before committing
  to a default.* At `n=801`, the Dasgupta-Gupta bound
  `k >= 4 ln(n) / (eps^2/2 - eps^3/3)` scales as `1/eps^2`; `eps=0.1`
  yields `k=5,732`, which barely compresses `d=20,264` (a `3.5x` reduction)
  and would make a weak demonstration figure. `eps=0.2` yields `k=1,543`
  (`13.1x` reduction) while remaining a standard, defensible distortion
  tolerance. This is documented directly in `config.py` as a comment, not
  hidden as an arbitrary constant.
- *`viz.spectral_overlay_plot()` was rewritten from single-panel to
  two-panel after visually reviewing the first pipeline run's output.* The
  initial single-panel version plotted all nonzero eigenvalues on one
  linear x-axis; because 27 outlier eigenvalues (up to 318.3) coexist with
  a Marchenko-Pastur bulk edge at 6.66, the single-axis histogram forced an
  x-range of ~330, compressing the entire bulk-vs-theory comparison into an
  unreadable sliver near the origin. This was corrected to a two-panel
  design: a zoomed bulk-only panel (eigenvalues within the theoretical
  support) overlaid with the (renormalized-conditional) Marchenko-Pastur
  density, and a full-range log-count panel showing the outlier tail
  against a marked support edge. The run_pipeline.py call site and the
  eigenvalue filtering (`bulk_eigs` computed strictly within
  `[0, support_upper]`, excluding both the zero point-mass and the 27
  outliers) were updated to match.
- *Anomaly detection and spectral analysis both operate on the cached
  top-2,000-gene subset, not the full 20,264-gene matrix.* This mirrors
  Phase 2's own justification for never forming the full covariance matrix
  (~3.3 GB, and here additionally requiring an inversion for Mahalanobis
  distance, which is not tractable as a per-sample diagnostic at that
  size); documented explicitly in both modules' docstrings rather than
  left implicit.
- *The distance-concentration theoretical bound was deliberately not
  implemented as a literal Hoeffding bound*, despite `SKELETON.md` naming
  "Hoeffding- or Chernoff-type." Hoeffding's inequality requires bounded
  random variables; standardized, log-transformed RNA-Seq features are
  effectively unbounded (approximately Gaussian). A chi-squared tail bound
  (Laurent & Massart, 2000) was used instead -- the correct Chernoff-type
  analogue for sums of squared sub-Gaussian coordinates -- and this
  substitution is stated explicitly in both the module docstring and the
  report's methodology section rather than silently mislabeled as
  "Hoeffding."
- *Zero Mahalanobis outliers were flagged at the chosen threshold, and
  this null result was reported as-is rather than adjusting the threshold
  to force a non-empty flagged set.* The observed mean squared Mahalanobis
  distance (778.1) is only 38.9% of its nominal `chi^2_2000` expectation
  (2,000), diagnosed as a direct consequence of Ledoit-Wolf's isotropic
  shrinkage target compressing distances relative to the classical
  asymptotic calibration, given the real spectrum's extreme heavy-tailedness
  (eigenvalues from ~0 to 318.3). This is reported and explained in the
  report's Discussion section as a methodological finding about shrinkage
  estimators under `d >> n`, per the project's "do not fabricate metrics"
  constraint, with the ten most-extreme samples reported descriptively
  (by rank, not by formal significance) as a partial answer to the
  cross-reference-against-class-labels requirement.
- Verified end-to-end by running `run_pipeline.py` twice against the real
  preprocessed data (once before, once after the `spectral_overlay_plot`
  fix), visually inspecting all eight new Phase 3 figures before citing
  them in the report, and cross-checking the empirical near-zero
  eigenvalue fraction (60.00%) against the independently-derived
  theoretical point-mass prediction (`1 - 1/gamma = 59.95%`) as a
  correctness check on the spectral analysis implementation before
  reporting it as a finding.
- *`src/viz.py` (257 lines) and `run_pipeline.py` (243 lines) both exceed
  the project's soft ~200-line-per-file guideline*, the same situation
  Phase 2's `intrinsic_dimension.py` (222 lines) was in. This was again a
  deliberate rejection of splitting: `SKELETON.md` fixes `viz.py` as the
  single "shared plotting utilities" module for all phases and
  `run_pipeline.py` as the single top-level orchestrator containing "no
  analysis logic of its own," so fragmenting either along an arbitrary
  line-count boundary would break the project's own specified
  architecture for a ~20-25% overshoot, rather than a natural
  module-responsibility split.
