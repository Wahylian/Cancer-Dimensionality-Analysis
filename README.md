# Intrinsic Dimensionality and Geometric Structure in Pan-Cancer RNA-Seq Data

A High-Dimensional Data Analysis (HDP) course project estimating the intrinsic
dimensionality of a small-n/large-d cancer gene-expression dataset. Three
independent estimators (Grassberger–Procaccia correlation dimension,
PCA-based, Levina–Bickel k-NN MLE) are calibrated against synthetic
Gaussian-noise and low-dimensional-manifold baselines, then cross-checked
against empirical tests of Johnson–Lindenstrauss projection,
Marchenko–Pastur spectral theory, curse-of-dimensionality/concentration of
measure, and Mahalanobis/Ledoit–Wolf anomaly detection, visualized via
PCA, t-SNE, and UMAP.

## Dataset

TCGA Pan-Cancer HiSeq RNA-Seq gene expression data, 801 tumor samples ×
20,531 genes, across 5 cancer classes (BRCA, KIRC, LUAD, PRAD, COAD).
Sourced from Kaggle
([`debatreyadas/gene-expression-cancer-rna-seq`](https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq)),
originally from the UCI ML Repository. Not committed to this repo.

## Environment setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Dataset download

Requires the Kaggle CLI authenticated with a token at `~/.kaggle/kaggle.json`
(see https://www.kaggle.com/docs/api).

```bash
python download_data.py
```

Downloads and extracts the dataset into `data/raw/` (gitignored; not committed).

## Running the pipeline

```bash
python run_pipeline.py
```

Runs Phases 1–3 end-to-end: preprocessing (log transform, standardization,
variance filtering), intrinsic-dimension estimation (real data vs. synthetic
baselines), and dimensionality-reduction/geometry analysis (PCA/t-SNE/UMAP
embeddings, Johnson–Lindenstrauss projection, spectral analysis, distance
concentration, Mahalanobis anomaly detection). Cached intermediates are
written to `data/processed/`, and every figure/table is regenerated into
`figures/`.

## Repository structure

```
Cancer-Dimensionality-Analysis/
├── src/                     # one module per phase
│   ├── config.py               # seeds, paths, hyperparameter defaults
│   ├── data_loading.py         # ingestion + validation
│   ├── preprocessing.py        # log transform, scaling, variance filtering
│   ├── synthetic.py            # synthetic baseline generators
│   ├── intrinsic_dimension.py  # correlation, PCA-based, k-NN MLE estimators
│   ├── embeddings.py           # PCA / t-SNE / UMAP wrappers
│   ├── random_projection.py    # Johnson–Lindenstrauss experiments
│   ├── spectral_analysis.py    # covariance spectrum + Marchenko–Pastur
│   ├── geometry_diagnostics.py # distance concentration bounds
│   ├── anomaly_detection.py    # shrinkage covariance + Mahalanobis distance
│   ├── evaluation.py           # summary tables, cross-method reconciliation
│   └── viz.py                  # shared plotting utilities
├── figures/                 # all generated plots and summary tables
├── reports/                 # per-phase LaTeX writeups (phase_1.tex-phase_4.tex)
├── data/                    # raw/ and processed/ (gitignored)
├── project_framework/       # course specification and design docs
├── ai_usage_log.md          # AI-tool usage documentation
├── download_data.py         # Kaggle CLI download script
├── run_pipeline.py          # top-level orchestration script
└── requirements.txt
```
