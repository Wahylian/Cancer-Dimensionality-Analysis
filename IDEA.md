# IDEA.md

## Project Title
**The Geometry of the Transcriptome: Intrinsic Dimensionality, Spectral Structure, and High-Dimensional Statistical Phenomena in Pan-Cancer RNA-Seq Data**

### Executive Summary
This project treats a real 801-patient × 20,531-gene pan-cancer RNA-Seq matrix purely as a *geometric and statistical object* — no classifier is trained as an end goal. We ask: what is the data's true ("intrinsic") dimensionality versus its ambient dimensionality of 20,531? How does its empirical covariance spectrum compare to pure-noise random matrix predictions? Do textbook high-dimensional phenomena (distance concentration, Johnson–Lindenstrauss distortion bounds, Mahalanobis breakdown under d ≫ n) actually manifest in real biological data, or are they artifacts of idealized Gaussian assumptions? Every technique is applied to the *same* matrix so that findings from one method (e.g., estimated intrinsic dimension) are cross-checked against another (e.g., the rank of the "signal" eigenvalues surviving Marchenko–Pastur denoising), producing a single coherent, defensible narrative rather than a checklist of disconnected demos.

### Dataset Specification
- **Name:** Gene Expression Cancer RNA-Seq (TCGA Pan-Cancer subset)
- **Kaggle link:** https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq/data
- **Dimensionality:** n = 801 samples, d = 20,531 features (genes) → d/n ≈ 25.6, a severe n ≪ d regime
- **Domain:** RNA-Seq gene expression levels (Illumina HiSeq platform) for five tumor types — BRCA (breast), KIRC (kidney), COAD (colon), LUAD (lung), PRAD (prostate) — drawn from The Cancer Genome Atlas Pan-Cancer Analysis Project
- **Why "high-dimensional":** With d ≈ 25× n, the sample covariance matrix is rank-deficient (rank ≤ n−1 ≪ d) and provably inconsistent as an estimator of the true 20,531×20,531 population covariance — the textbook regime where classical multivariate statistics breaks down and specialized high-dimensional tools (shrinkage estimators, random matrix theory, intrinsic-dimension estimators) become necessary rather than optional.

### Problem Statement & Objectives
Real transcriptomic data is measured in tens of thousands of gene-expression coordinates, yet biological processes are governed by a comparatively small number of coordinated regulatory programs. The central question is whether this expected "low intrinsic dimension inside a high ambient dimension" structure can be (a) detected and (b) quantified with multiple independent, mathematically distinct estimators that should agree with each other if the underlying phenomenon is real rather than a single method's artifact. A secondary objective is to empirically test, on this specific dataset rather than in simulation, whether classical high-dimensional pathologies (distance concentration, covariance singularity, curse-of-dimensionality volume effects) are present and how much they are mitigated by dimension-reduction and shrinkage techniques.

### Methodological Approach
1. **Spectral / Random Matrix Theory analysis** — compare the empirical eigenvalue distribution of the (shrinkage-regularized) sample covariance matrix to the Marchenko–Pastur law for pure noise, isolating the "signal" eigenvalues that exceed the noise bulk edge.
2. **Covariance regularization** — Ledoit–Wolf shrinkage estimator, contrasted against the singular raw sample covariance, to obtain a well-conditioned covariance usable downstream.
3. **Intrinsic dimension estimation (triangulated across four independent estimators)** — Grassberger–Procaccia correlation dimension, global PCA-based dimension (variance-explained elbow / participation ratio), Levina–Bickel Maximum Likelihood kNN estimator, and Fisher separability analysis — compared against each other and against synthetic Gaussian-noise and low-dimensional-manifold controls of matched shape.
4. **Curse-of-dimensionality empirics** — pairwise distance concentration (coefficient of variation of distances), nearest/farthest-neighbor distance ratio, and sphere-to-cube volume ratio, computed on the real data and on matched-shape synthetic controls.
5. **Concentration of measure / Johnson–Lindenstrauss** — random projections into progressively lower dimensions, empirically measuring pairwise-distance distortion against the JL theoretical bound.
6. **Nonlinear visualization** — PCA, t-SNE, and UMAP embeddings compared for how well they preserve the tumor-type cluster structure, purely as an exploratory/visual diagnostic (not classification).
7. **Practical payoff: Mahalanobis-distance anomaly detection** — enabled specifically by the shrinkage covariance (impossible with the raw singular sample covariance), used to flag statistically atypical samples within each tumor type.

### Grading Alignment
| Grading Highlight | How This Project Addresses It |
|---|---|
| Impress via dataset & technique choice | Extreme, genuine n≪d biomedical dataset (d/n≈25.6) analyzed with 4 independent intrinsic-dimension estimators + RMT, not a single off-the-shelf method |
| Deeper technicality/insight → higher grade | Every technique is cross-validated against another (RMT signal-rank vs. intrinsic-dimension estimates vs. PCA elbow) rather than presented in isolation |
| Must defend every choice | Each phase includes explicit synthetic-control baselines (matched-shape Gaussian noise, matched-shape low-rank manifolds) so every real-data finding is benchmarked against a null |
| Exhaustive, professional report | PLAN.md specifies a fixed, complete set of figures/tables per phase feeding directly into report sections |
| Heavy, documented LLM/Cursor usage | PLAN.md dedicates an explicit phase to structured AI-interaction logging (prompts, outputs, edits) required by the rubric |
| Analysis of data itself, not ML models | No classifier is trained as an end product; PCA/t-SNE/UMAP and Mahalanobis distance are used purely as descriptive/diagnostic tools |