"""Download the Gene Expression Cancer RNA-Seq dataset from Kaggle into data/raw/.

Requires the Kaggle CLI to be installed and authenticated: place your API token
at ~/.kaggle/kaggle.json (see https://www.kaggle.com/docs/api).
"""
import subprocess
import zipfile
from pathlib import Path

DATASET = "debatreyadas/gene-expression-cancer-rna-seq"
RAW_DIR = Path("data/raw")


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR)],
        check=True,
    )
    zip_path = next(RAW_DIR.glob("*.zip"))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR)
    zip_path.unlink()


if __name__ == "__main__":
    main()
