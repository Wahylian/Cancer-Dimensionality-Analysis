# Intrinsic Dimensionality and Geometric Structure in Pan-Cancer RNA-Seq Data

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
