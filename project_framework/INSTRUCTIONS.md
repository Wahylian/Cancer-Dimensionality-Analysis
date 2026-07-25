# INSTRUCTIONS.md — High-Dimensional Data Analysis Course Project

**Project:** Intrinsic Dimensionality and Geometric Structure in Pan-Cancer RNA-Seq Data
**Course reference:** HDP Research Project specification (aigner-horev.wixsite.com/eigen/hdppyproj)
**Dataset:** Gene Expression Cancer RNA-Seq — https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq

This document converts the course's open-ended project brief into an unambiguous, gradable specification. Every requirement below is a **definitive technical requirement**, not a suggestion. Where the course brief says "you are free to design your own analysis," this document fixes that freedom into one concrete, defensible design so there is no room for scope ambiguity.

---

## 1. Objective & Scope

### 1.1 Learning outcomes
By completing this project you must be able to:
1. Explain, with your own derivations and evidence, the distinction between **ambient dimension** and **intrinsic dimension**, and estimate the latter using at least three independent methods.
2. Apply and critically compare linear (PCA) and nonlinear (t-SNE, UMAP) dimensionality reduction on a real, small-n/large-d dataset.
3. Empirically validate or refute at least two theoretical high-dimensional phenomena (e.g., Johnson–Lindenstrauss preservation, Marčenko–Pastur spectral behavior, curse-of-dimensionality distance concentration) using this dataset.
4. Justify every methodological choice (preprocessing, regularization, hyperparameters) in writing, and defend it against at least one plausible alternative.
5. Document, transparently and specifically, how AI tools (LLMs / Cursor) were used to produce the code and analysis.

### 1.2 Scope boundaries (what you MUST NOT do)
- This is **not** a predictive-modeling project. Building a high-accuracy classifier is **out of scope**. If you train any classifier, it must be used only as a *diagnostic probe* of geometric structure (e.g., "do the 5 tumor types separate linearly after PCA?"), never as the deliverable's centerpiece.
- Do not simply run `sklearn.decomposition.PCA` and `sklearn.manifold.TSNE` with no further analysis and call it complete. Every plot must be interpreted, quantified, and connected back to an intrinsic-dimensionality or high-dimensional-geometry claim.
- Do not treat missing preprocessing steps casually — this dataset has no NaNs, but it does have scale, skew, and near-zero-variance features that must be explicitly handled and justified (see Phase 1).

---

## 2. Grading Rubric Alignment

The course's six graded highlights are mapped to explicit, checkable requirements as follows:

| Course grading highlight | Concrete requirement in this project |
|---|---|
| **(1) Impress with dataset + techniques chosen** | Dataset is justified in Section 4 of `IDEA.md` (d/n ≈ 25.6, real clinical relevance, no missing data, 5-class ground truth for validating unsupervised structure). Techniques span intrinsic dimension estimation, linear + nonlinear embeddings, and random matrix / concentration theory — matching and exceeding the "Intrinsic Dimension Estimation" example plus one "Additional Project Example" category. |
| **(2) Higher technicality / deeper insight → higher grade** | Minimum of **3 independent intrinsic-dimension estimators** required (Section 3, Phase 2) with a written reconciliation of any disagreement between them — not just reporting three numbers. |
| **(3) Defend every choice** | Every phase below ends with a **"Justify in report"** requirement. Your PDF report must contain a dedicated subsection per phase explaining *why* (not just *what*) for every non-trivial choice (e.g., why log-transform before PCA, why Levina–Bickel over MLE-only, why shrinkage covariance for Mahalanobis distance). |
| **(4) Exhaustive, complete report** | Section 5 (Deliverables) specifies mandatory report sections; omitting any is an automatic deduction. |
| **(5) Professional, error-free English/formatting** | Report must be proofread; Section 5.3 specifies formatting requirements (headings, captions, numbered equations/figures, consistent citation style). |
| **(6)+(7) Mandatory, documented AI-tool usage** | Section 5.4 mandates an **AI Usage Log** as a required report appendix. Absence of this log, or evidence that AI tools were not substantively used to generate code, results in the lowest grade band per the course's explicit policy. |

---

## 3. Step-by-Step Requirements

### Phase 0 — Environment & Reproducibility (prerequisite) - DONE
1. Set a global random seed (e.g., `42`) and pass it explicitly to every stochastic routine (t-SNE, UMAP, k-means, train/test splits).
2. Pin package versions in a `requirements.txt` (minimum: `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `umap-learn`).
3. Download the dataset programmatically or document the manual download step; **do not** commit the raw ~70 MB tar/CSV data to GitHub — add it to `.gitignore` and provide a `download_data.py` / documented Kaggle CLI command instead.

### Phase 1 — Preprocessing & Cleaning - DONE
1. Load the expression matrix `X` (801 × 20,531) and label vector `y` (5 classes); assert shapes and check for and report any missing values (expected: none) and any zero-variance genes (report the count; you must decide whether to drop them and justify it).
2. Apply a `log2(1 + x)` (or `log1p`) transform to the raw expression counts and justify this choice with reference to the right-skewed nature of RNA-Seq count data.
3. Standardize features (zero mean, unit variance) **after** the log transform, and explicitly justify why standardization matters for PCA/covariance-based methods but must be applied consistently before distance-based methods (t-SNE/UMAP/k-NN) too.
4. Report basic exploratory statistics: class balance table, per-gene mean/variance distribution (histogram), and a justification of any variance-based feature filtering you choose to apply (e.g., keeping the top-k most variable genes for computational tractability in specific sub-analyses) — you must show results **both** with the full 20,531-feature matrix (where computationally feasible) and any reduced-feature variant, never only the reduced one.
5. **Justify in report:** log-transform choice, standardization order, any feature filtering threshold.

### Phase 2 — Intrinsic Dimension Estimation (core requirement) - DONE
1. **Correlation dimension:** Implement the Grassberger–Procaccia correlation integral C(r) = (2 / (N(N-1))) · #{pairs with distance < r}. Compute it across a logarithmically spaced range of radii, plot log C(r) vs. log r, and estimate the slope (= correlation dimension D₂) via linear regression over the scaling region you identify and justify.
2. **PCA-based estimator:** Compute the full eigenvalue spectrum of the (regularized) covariance matrix; report the number of components needed to explain 90%, 95%, and 99% of variance, and produce a scree plot with an explicit "elbow" identification method (e.g., Kaiser criterion or explained-variance second-derivative).
3. **k-NN–based estimator:** Implement at least one neighborhood-based intrinsic dimension estimator (e.g., Levina–Bickel maximum-likelihood estimator) using `k` values from a small sweep (e.g., k = 5, 10, 20) and report sensitivity to k.
4. **Calibration against synthetic baselines (mandatory):** Generate (a) pure high-dimensional Gaussian noise matched in shape to the real data and (b) points sampled from a known low-dimensional manifold (e.g., a 2D or 5D linear subspace) embedded in a 20,531-dimensional ambient space with added noise. Run all three estimators from steps 1–3 on both synthetic datasets and report results **side by side** with the real data's results in one comparison table.
5. **Justify in report:** scaling-region choice for correlation dimension, elbow criterion, k sweep, and a written reconciliation of any disagreement between the three estimators (they will not agree exactly — you must explain why, referencing estimator bias/assumptions).

### Phase 3 — Dimensionality Reduction, Visualization & High-Dimensional Geometry
1. **PCA:** Produce a 2D and 3D PCA scatter plot colored by cancer type; report cumulative explained variance.
2. **t-SNE and UMAP:** Produce 2D embeddings colored by cancer type; run each with at least two different hyperparameter settings (e.g., two perplexity values for t-SNE, two `n_neighbors` values for UMAP) and discuss how the visual structure changes — you must not present only one hyperparameter setting per method.
3. **Johnson–Lindenstrauss / random projections:** Project the data into a lower-dimensional space (target dimension computed from the JL lemma for a chosen distortion ε) using a random Gaussian projection matrix, and empirically verify pairwise-distance preservation by plotting original vs. projected pairwise distances and reporting the empirical distortion distribution against the theoretical JL bound.
4. **Random matrix theory / covariance spectrum:** Compute the empirical spectral distribution of the (standardized) gene covariance matrix restricted to a computationally tractable feature subset (e.g., top 1,000–2,000 most variable genes, explicitly justified); overlay the theoretical Marčenko–Pastur density for a null random matrix of matching aspect ratio, and identify and interpret spectral outliers as evidence of real biological signal.
5. **Curse of dimensionality / concentration of measure:** Compute the distribution of pairwise Euclidean distances and the ratio (max distance − min distance) / min distance for increasing numbers of features (e.g., subsampling 10, 100, 1,000, all genes) to empirically demonstrate distance concentration; relate the observed behavior to theoretical concentration inequalities (Hoeffding or Chernoff bounds), stating the bound and comparing it to the empirical result.
6. **Anomaly detection (supporting analysis):** Using a shrinkage covariance estimator (e.g., Ledoit–Wolf) — required because the raw sample covariance is singular for d ≫ n — compute Mahalanobis distances for all samples and flag outliers using an explicit, justified threshold (e.g., chi-squared quantile). Cross-reference flagged outliers against cancer-type labels and discuss whether they correspond to any known biological subtype ambiguity or are likely technical artifacts.
7. **Justify in report:** hyperparameter choices for t-SNE/UMAP, JL target-dimension derivation, feature-subset choice for the spectral analysis, and the anomaly-detection threshold.

### Phase 4 — Evaluation, Synthesis & Reporting
1. Synthesize Phases 2–3 into a single coherent narrative: does the estimated intrinsic dimension (Phase 2) roughly match the number of components needed for the visual/geometric structure observed in Phase 3? Reconcile any apparent contradictions explicitly.
2. Provide a summary table consolidating: ambient dimension, all intrinsic dimension estimates (real vs. synthetic baselines), variance explained at key PCA cutoffs, JL empirical distortion vs. theoretical bound, and number of Mahalanobis-flagged outliers.
3. Discuss limitations explicitly: small sample size per class (esp. COAD, n≈78), potential batch effects in TCGA data, and the fact that RNA-Seq correlation structure may violate i.i.d. assumptions underlying some theoretical bounds.
4. State what you would do differently with more time/compute (e.g., stability selection across bootstrap resamples of the correlation-dimension estimate).

---

## 4. Deliverables & Submission Guidelines

### 4.1 Repository (GitHub, public)
- `README.md` with a project summary, environment setup, and instructions to reproduce every figure in the report.
- `/src` or equivalent: modular, documented Python code (see `SKELETON.md` for required structure).
- `/notebooks` (optional but recommended): exploratory notebooks are allowed in addition to, not instead of, modular source code.
- `/figures`: all generated plots, named consistently with the report's figure numbers.
- `requirements.txt` / `environment.yml`.
- No large raw data files committed; document the download procedure instead.

### 4.2 Report (PDF, English, Cambridge-level proofreading standard)
Mandatory sections, in order:
1. Title, author, date.
2. Abstract (≤ 250 words).
3. Introduction & motivation (why high-dimensional analysis is necessary for this dataset).
4. Dataset description (provenance, shape, class balance, license).
5. Methodology (one subsection per phase in Section 3 above, each with an explicit "Justification" paragraph).
6. Results (all required figures/tables from Section 3, each with a caption and in-text interpretation — a figure with no accompanying interpretation is incomplete).
7. Discussion (synthesis per Phase 4).
8. Limitations & future work.
9. References (dataset citation, any papers/theorems referenced — Johnson–Lindenstrauss, Marčenko–Pastur, Grassberger–Procaccia, Levina–Bickel, Ledoit–Wolf, etc.).
10. **Appendix A: AI Usage Log** (mandatory — see 4.4).

### 4.3 Formatting requirements
- Numbered figures and tables, each with a descriptive caption.
- Equations numbered and referenced in text where derivations are shown (e.g., correlation integral, JL lemma statement, Marčenko–Pastur density).
- Consistent citation style (any standard style is acceptable if used consistently).
- No raw code dumps inside the PDF report — code lives in the GitHub repository; the report may show short (≤ 15 line) illustrative snippets only where essential to explain a non-obvious step.

### 4.4 AI Usage Log (mandatory, per course grading policy)
You **must** use LLMs/Cursor AI to generate the majority of your code, and you must document this usage explicitly. The log must include, at minimum:
- Which tool(s) were used (e.g., Claude, ChatGPT, Cursor) and for which specific files/functions.
- A representative sample of prompts used (not necessarily every prompt, but enough to demonstrate substantive reliance on AI tooling).
- What you changed, corrected, or rejected from the AI's output, and why — demonstrating that you understand and can defend every line of code, per grading criterion (3).
- Failure to include this log, or evidence that AI tools were used only trivially, will be graded in the lowest band per the course's explicit policy.

### 4.5 Submission checklist
- [ ] Public GitHub repository link submitted.
- [ ] PDF report included in the repository.
- [ ] Report written in English at a professional standard.
- [ ] All required visualizations (Phases 2–3) present and captioned.
- [ ] AI Usage Log included as Appendix A.
- [ ] Code runs end-to-end from a fresh clone using only `requirements.txt` and the documented data-download step.
