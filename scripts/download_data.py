"""Download raw datasets for MultiDiseaseAI from UCI / public mirrors.

Run once before preprocessing:

    python scripts/download_data.py

Datasets fetched:
    - Heart Disease (UCI ML Repository, id=45, Cleveland subset)
    - Pima Indians Diabetes (public-domain mirror, NIDDK original)
    - Chronic Kidney Disease (UCI ML Repository, id=336)
    - ILPD - Indian Liver Patient Dataset (UCI ML Repository, id=225)

All datasets are redistributed under the UCI Machine Learning Repository's
open terms for academic/research use (see data/README.md for full citation
and license notes per dataset).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

PIMA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
PIMA_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree_function",
    "age",
    "outcome",
]


def download_heart_disease() -> pd.DataFrame:
    """Fetch the UCI Heart Disease (Cleveland) dataset via ucimlrepo (id=45)."""
    logger.info("Fetching Heart Disease dataset (UCI id=45)...")
    ds = fetch_ucirepo(id=45)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    out_path = RAW_DIR / "heart_disease.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s rows x %s cols to %s", *df.shape, out_path)
    return df


def download_diabetes() -> pd.DataFrame:
    """Fetch the Pima Indians Diabetes dataset from a public-domain CSV mirror."""
    logger.info("Fetching Pima Indians Diabetes dataset...")
    resp = requests.get(PIMA_URL, timeout=30)
    resp.raise_for_status()
    from io import StringIO

    df = pd.read_csv(StringIO(resp.text), header=None, names=PIMA_COLUMNS)
    out_path = RAW_DIR / "diabetes.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s rows x %s cols to %s", *df.shape, out_path)
    return df


def download_ckd() -> pd.DataFrame:
    """Fetch the UCI Chronic Kidney Disease dataset via ucimlrepo (id=336)."""
    logger.info("Fetching Chronic Kidney Disease dataset (UCI id=336)...")
    ds = fetch_ucirepo(id=336)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    out_path = RAW_DIR / "ckd.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s rows x %s cols to %s", *df.shape, out_path)
    return df


def download_ilpd() -> pd.DataFrame:
    """Fetch the UCI ILPD (Indian Liver Patient Dataset) via ucimlrepo (id=225)."""
    logger.info("Fetching ILPD (Indian Liver Patient Dataset) (UCI id=225)...")
    ds = fetch_ucirepo(id=225)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    out_path = RAW_DIR / "liver_disease.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved %s rows x %s cols to %s", *df.shape, out_path)
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_heart_disease()
    download_diabetes()
    download_ckd()
    download_ilpd()
    logger.info("All datasets downloaded to %s", RAW_DIR)


if __name__ == "__main__":
    main()
