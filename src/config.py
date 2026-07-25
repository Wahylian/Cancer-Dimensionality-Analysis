# Global seed, paths, and hyperparameter defaults. Import RANDOM_SEED and pass it
# explicitly to every stochastic routine (t-SNE, UMAP, k-means, train/test splits).

RANDOM_SEED: int = 42
DATA_DIR = "data/"
RAW_DATA_PATH = "data/raw/TCGA-PANCAN-HiSeq-801x20531/TCGA-PANCAN-HiSeq-801x20531/data.csv"
RAW_LABELS_PATH = "data/raw/TCGA-PANCAN-HiSeq-801x20531/TCGA-PANCAN-HiSeq-801x20531/labels.csv"
PROCESSED_DIR = "data/processed/"
FIGURES_DIR = "figures/"
N_TOP_VARIABLE_GENES = 2000     # used only where full 20531-dim computation is intractable
CLASS_LABELS = ["BRCA", "KIRC", "LUAD", "PRAD", "COAD"]

# --- Phase 2: Intrinsic Dimension Estimation ---

# Grassberger-Procaccia correlation integral: radii are log-spaced between the
# given quantiles of the empirical pairwise-distance distribution, which keeps
# r_min above microscopic quantization noise and r_max below the ambient
# cloud's finite diameter (see reports/phase_2.tex, Sec. Methodology).
CORR_DIM_N_RADII = 60
CORR_DIM_MIN_QUANTILE = 0.01
CORR_DIM_MAX_QUANTILE = 0.90
CORR_DIM_MIN_SCALING_POINTS = 5     # minimum width of the auto-detected scaling-region plateau
CORR_DIM_RADIUS_SAMPLE_SIZE = 300   # subsample size used only to estimate distance quantiles for radii bounds

# PCA-based estimator: cumulative explained-variance cutoffs to report.
PCA_VARIANCE_THRESHOLDS = [0.90, 0.95, 0.99]

# Levina-Bickel k-NN MLE: neighborhood-size sweep for sensitivity analysis.
KNN_MLE_K_VALUES = [5, 10, 20]
KNN_MLE_EPSILON = 1e-10             # guards against ln(1)=0 / div-by-zero under distance ties

# Synthetic baseline generators.
SYNTHETIC_NOISE_STD = 1.0           # isotropic Gaussian baseline, matched to standardized real data scale
MANIFOLD_INTRINSIC_DIM = 5          # known ground-truth subspace dimension for calibration
MANIFOLD_NOISE_STD = 0.001          # additive noise std; see reports/phase_2.tex for the noise-level
                                     # sensitivity analysis that motivates this choice
