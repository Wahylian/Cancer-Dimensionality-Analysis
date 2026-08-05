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

# --- Phase 3: Dimensionality Reduction, Visualization & High-Dimensional Geometry ---

# PCA visualization: number of components to retain for the 2D/3D scatter plots.
PCA_VIZ_N_COMPONENTS = 3

# t-SNE / UMAP: both are preprocessed with a linear PCA to this many dimensions
# first (standard practice: denoises and speeds up the nonlinear neighbor search
# without materially changing the recovered structure, since Phase 2 showed
# >90% of variance is linear-PCA-recoverable well below this many components).
EMBEDDING_PCA_PREPROCESS_DIM = 50
TSNE_PERPLEXITIES = [30, 50]
UMAP_N_NEIGHBORS_VALUES = [15, 50]

# Johnson-Lindenstrauss random projection: target distortion epsilon. 0.2 is
# chosen over the more conservative textbook default of 0.1 because, at
# n=801, the Dasgupta-Gupta bound k >= 4 ln(n) / (eps^2/2 - eps^3/3) scales
# as 1/eps^2 and eps=0.1 yields a target dimension (5731) that barely
# compresses the d=20,264 ambient space, obscuring the demonstration; 0.2
# yields a still-tight, standard distortion tolerance with a much more
# illustrative ~13x reduction.
JL_EPSILON = 0.2

# JL dimension sweep: replaces the single-point (epsilon, k) experiment above,
# which is uninformative here since k=1543 > n-1=800 (the exact-distortion
# rank bound for 801 centred points) makes the JL guarantee non-binding.
# k=800 is that rank bound; k=1543 is JL_EPSILON's target dimension; the rest
# span two orders of magnitude around both landmarks.
JL_SWEEP_K_VALUES = [20, 30, 50, 100, 200, 321, 500, 800, 1543, 3000]
JL_SWEEP_SEEDS = list(range(10))
JL_SWEEP_KNN = 10

# Spectral analysis (Marchenko-Pastur): reuse the top-k variable gene subset
# for tractability (same subset already cached in data/processed/phase1_topk.npz).
SPECTRAL_N_TOP_GENES = N_TOP_VARIABLE_GENES

# Distance concentration diagnostic: feature-subsample sizes; the full gene
# count is appended at runtime from the actual (post zero-variance-filter) shape.
GEOMETRY_FEATURE_COUNTS = [10, 100, 1000]
GEOMETRY_CONCENTRATION_EPSILON = 0.1   # relative deviation threshold for the theoretical bound

# Mahalanobis anomaly detection: Ledoit-Wolf shrinkage on the top-k gene subset
# (the full 20,264-d covariance matrix is ~3.3 GB and never formed, matching the
# Phase 2 PCA estimator's approach to the same d >> n singularity problem).
MAHALANOBIS_ALPHA = 0.025           # upper-tail chi-squared quantile (97.5%) for the outlier threshold
