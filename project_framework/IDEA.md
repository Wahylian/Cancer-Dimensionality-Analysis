# Project Idea: Mapping the Hidden Geometry of Pan-Cancer Transcriptomes

## 1. Project Title & Theme

**"When 20,531 Dimensions Hide a Handful of Truths: Intrinsic Dimensionality and Geometric Structure in Pan-Cancer RNA-Seq Data"**

Domain: computational biology / cancer genomics. The project treats a real clinical gene-expression dataset as a laboratory for the core questions of the High-Dimensional Data Analysis (HDP) course: *how many degrees of freedom does this data actually have, and how does its geometry behave as dimensionality explodes?*

## 2. Problem Statement

Modern tumor profiling technologies (RNA-Seq / Illumina HiSeq) routinely measure the expression of **more than 20,000 genes per patient**, while the number of patients in any given study is typically in the hundreds. This is the canonical small-n, large-d regime: the ambient feature space is enormous, but biology strongly suggests that tumor identity is governed by a much smaller number of underlying regulatory programs (pathways, co-expression modules, cell-of-origin signals).

This project asks:

- What is the **ambient** dimensionality of this dataset versus its **intrinsic** dimensionality — the number of degrees of freedom actually needed to describe the variation between tumor samples?
- Do classical high-dimensional phenomena predicted by theory (concentration of measure, distance concentration, curse-of-dimensionality effects on nearest neighbors, empirical spectral behavior of covariance matrices) actually manifest in this real dataset, or does biological structure make it behave differently from random high-dimensional noise?
- Can we recover clinically meaningful structure (five distinct cancer types) using only unsupervised, geometry-driven analysis of the raw feature space?

High-dimensional analysis is *necessary* here, not optional: with d = 20,531 ≫ n = 801, standard statistical intuitions (density estimation, Euclidean nearest neighbors, ordinary covariance estimation) break down, and dedicated techniques — dimensionality reduction, intrinsic dimension estimators, regularized covariance estimation, and random-projection theory — are required to extract reliable insight.

## 3. Dataset Overview

- **Name:** Gene Expression Cancer RNA-Seq (a.k.a. the TCGA Pan-Cancer HiSeq / PANCAN gene-expression dataset)
- **Kaggle URL:** https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq
  *(This is a Kaggle mirror of the original UCI Machine Learning Repository release: Fiorini, S. (2016), "gene expression cancer RNA-Seq," UCI ML Repository, DOI 10.24432/C5R88H — sourced from the TCGA Pan-Cancer Atlas / Illumina HiSeq platform.)*
- **Sample size (n):** 801 tumor samples (patients)
- **Feature dimensionality (d):** 20,531 gene-expression features (columns named `gene_0` … `gene_20530`)
- **Aspect ratio:** d / n ≈ 25.6 — a textbook high-dimensional, small-sample regime
- **Data types:** all features are continuous, non-negative RNA-Seq expression intensities (real-valued); the target is a categorical label with 5 classes: **BRCA** (breast, n≈300), **KIRC** (kidney, n≈146), **LUAD** (lung, n≈141), **PRAD** (prostate, n≈136), **COAD** (colon, n≈78)
- **Missing values:** none reported
- **License:** CC BY 4.0

## 4. Proposed Methodology

The analysis is organized around three complementary high-dimensional lenses, all applied to the *same* dataset so the findings reinforce one another:

1. **Intrinsic Dimension Estimation** (the course's flagship example)
   - Correlation dimension via the Grassberger–Procaccia correlation integral, estimated from log–log slope fitting.
   - PCA-based estimators (variance-explained thresholds, eigenvalue "elbow"/scree analysis, participation ratio).
   - k-NN–based intrinsic dimension estimators (e.g., the Levina–Bickel MLE estimator).
   - Comparison against synthetic controls: pure high-dimensional Gaussian noise and a known low-dimensional manifold embedded in high-dimensional space, so the real dataset's intrinsic dimension can be judged against calibrated baselines.

2. **Dimensionality Reduction & Visualization**
   - Linear projection via PCA (with scree/variance-explained plots).
   - Nonlinear embeddings via t-SNE and UMAP for exploratory visualization of cancer-type separation.
   - Random 2D projections and the Johnson–Lindenstrauss lemma, empirically testing how well pairwise distances are preserved under random projection versus PCA.

3. **High-Dimensional Statistical Geometry**
   - Empirical spectral distribution of the gene covariance/correlation matrix, compared against Marčenko–Pastur predictions for a null (unstructured) high-dimensional regime, to show where real biological signal departs from random-matrix expectations.
   - Concentration-of-measure / curse-of-dimensionality diagnostics: pairwise distance distributions, nearest-neighbor distance ratios, and (optionally) the Gaussian annulus phenomenon on standardized features.
   - Mahalanobis-distance-based outlier/anomaly detection (using a regularized/shrinkage covariance estimate, since the raw sample covariance is singular when d ≫ n).

## 5. Key Objectives

- Quantify the gap between ambient dimensionality (20,531) and intrinsic dimensionality (estimated single-digit-to-low-double-digit range) and explain that gap biologically.
- Demonstrate, empirically and visually, that this real dataset deviates from pure high-dimensional noise in specific, measurable ways (spectral outliers, faster-than-random distance concentration collapse, clean low-dimensional cluster structure).
- Produce a professional, defensible report — with full methodological justification and complete AI-tool usage documentation — of publication-adjacent quality, per the course's grading expectations.
