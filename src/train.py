"""Train one TabPFN classifier per disease and persist it to `models/`.

Why TabPFN and not a gradient-boosted tree / logistic regression baseline:
TabPFN is a *foundation model for tabular data* - a transformer pretrained
once on millions of synthetic classification tasks that performs Bayesian
in-context learning at inference time. Instead of iteratively fitting
weights to one dataset, "fitting" a TabPFN model means caching the training
table itself; prediction is a single forward pass that attends over the
cached training rows to produce a posterior-predictive class distribution
for the query row. For the small (few hundred row), tabular, mixed
numeric/categorical clinical datasets used across all four diseases in this
project, TabPFN has been shown in the original paper (Hollmann et al., 2023,
*Nature*) to match or beat carefully tuned gradient-boosted trees while
requiring no hyperparameter search - which matters here since each disease
would otherwise need its own tuning pass.

Usage:
    python -m src.train              # trains all four diseases
    python -m src.train --disease heart
"""

from __future__ import annotations

import argparse
import logging
import time

import joblib
import pandas as pd
from tabpfn import TabPFNClassifier

from src.artifacts import PreprocessArtifact
from src.config import DISEASES, RANDOM_STATE, get_disease

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_disease(disease_key: str, device: str = "cpu") -> TabPFNClassifier:
    """Fit a TabPFNClassifier for `disease_key` and save it to `models/`.

    Assumes `src.preprocessing.DiseasePreprocessor(...).run()` has already
    produced `data/processed/{key}_train.csv` and the matching
    `models/{key}_preprocessor.joblib` artifact (for categorical feature
    indices).
    """
    disease = get_disease(disease_key)
    artifact: PreprocessArtifact = PreprocessArtifact.load(disease.preprocessor_path)

    train_df = pd.read_csv(disease.processed_train_path)
    X_train = train_df[artifact.feature_order]
    y_train = train_df["target"]

    logger.info(
        "[%s] training TabPFNClassifier on %d rows x %d features (%d categorical)",
        disease_key,
        len(X_train),
        X_train.shape[1],
        len(artifact.categorical_feature_indices),
    )

    clf = TabPFNClassifier(
        categorical_features_indices=artifact.categorical_feature_indices or None,
        random_state=RANDOM_STATE,
        device=device,
        n_estimators=8,
    )

    t0 = time.time()
    clf.fit(X_train.to_numpy(), y_train.to_numpy())
    elapsed = time.time() - t0
    logger.info("[%s] fit complete in %.2fs", disease_key, elapsed)

    joblib.dump(
        {
            "model": clf,
            "feature_order": artifact.feature_order,
            "disease_key": disease_key,
            "n_train": len(X_train),
            "train_time_seconds": elapsed,
        },
        disease.model_path,
    )
    logger.info("[%s] saved model to %s", disease_key, disease.model_path)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=list(DISEASES), default=None, help="Train a single disease only")
    parser.add_argument("--device", default="cpu", help="TabPFN device: 'cpu', 'cuda', 'mps', or 'auto'")
    args = parser.parse_args()

    keys = [args.disease] if args.disease else list(DISEASES)
    for key in keys:
        train_disease(key, device=args.device)


if __name__ == "__main__":
    main()
