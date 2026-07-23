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
