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
