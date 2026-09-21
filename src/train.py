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
project, TabPFN has been shown in the original paper (Hollmann et al., 2025,
*Nature*) to match or beat carefully tuned gradient-boosted trees while
requiring no hyperparameter search - which matters here since each disease
would otherwise need its own tuning pass.

Alongside the fitted model, each bundle also stores three cheap, precomputed
add-ons (see `src/uncertainty.py`) so that serving stays a single forward
pass:

* ``conformal_residuals`` - the out-of-fold residuals behind a nominal 90%
  probability band (a K-fold jackknife-style interval; coverage is approximate).
* ``decision_threshold`` - the probability cutoff maximising Youden's J on
  out-of-fold predictions (replaces a blind 0.5).
* ``ood`` - a robust-Mahalanobis novelty detector over the encoded training
  features, to flag live inputs unlike the training cohort.

Usage:
    python -m src.train                  # trains all four diseases
    python -m src.train --disease heart
    python -m src.train --no-calibrate   # skip the out-of-fold add-ons (faster)
"""

from __future__ import annotations

import argparse
import logging
import time

import joblib
import pandas as pd
from tabpfn import TabPFNClassifier

from src.artifacts import PreprocessArtifact
from src.config import CONFORMAL_ALPHA, CV_FOLDS, DISEASES, OOD_QUANTILE, RANDOM_STATE, get_disease
from src.uncertainty import (
    OODDetector,
    conformal_residuals,
    oof_probabilities,
    youden_threshold,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _make_tabpfn(artifact: PreprocessArtifact, device: str, n_estimators: int) -> TabPFNClassifier:
    return TabPFNClassifier(
        categorical_features_indices=artifact.categorical_feature_indices or None,
        random_state=RANDOM_STATE,
        device=device,
        n_estimators=n_estimators,
    )


def train_disease(
    disease_key: str,
    device: str = "cpu",
    n_estimators: int = 8,
    calibrate: bool = True,
) -> TabPFNClassifier:
    """Fit a TabPFNClassifier for `disease_key` and save it to `models/`.

    Assumes `src.preprocessing.DiseasePreprocessor(...).run()` has already
    produced `data/processed/{key}_train.csv` and the matching
    `models/{key}_preprocessor.joblib` artifact (for categorical feature
    indices).

    When ``calibrate`` is true (default) a k-fold out-of-fold pass is run to
    precompute the conformal calibration set, the learned decision threshold,
    and the novelty detector. Pass ``calibrate=False`` to skip that pass.
    """
    disease = get_disease(disease_key)
    artifact: PreprocessArtifact = PreprocessArtifact.load(disease.preprocessor_path)

    train_df = pd.read_csv(disease.processed_train_path)
    X_train = train_df[artifact.feature_order]
    y_train = train_df["target"]
    X_np = X_train.to_numpy()
    y_np = y_train.to_numpy()

    logger.info(
        "[%s] training TabPFNClassifier on %d rows x %d features (%d categorical)",
        disease_key,
        len(X_train),
        X_train.shape[1],
        len(artifact.categorical_feature_indices),
    )

    clf = _make_tabpfn(artifact, device, n_estimators)

    t0 = time.time()
    clf.fit(X_np, y_np)
    elapsed = time.time() - t0
    logger.info("[%s] fit complete in %.2fs", disease_key, elapsed)

    bundle = {
        "model": clf,
        "feature_order": artifact.feature_order,
        "disease_key": disease_key,
        "n_train": len(X_train),
        "train_time_seconds": elapsed,
        "decision_threshold": 0.5,
        "conformal_residuals": None,
        "conformal_alpha": CONFORMAL_ALPHA,
        "ood": None,
    }

    if calibrate:
        t0 = time.time()
        oof = oof_probabilities(
            lambda: _make_tabpfn(artifact, device, n_estimators),
            X_np,
            y_np,
            n_splits=CV_FOLDS,
            random_state=RANDOM_STATE,
        )
        bundle["conformal_residuals"] = conformal_residuals(oof, y_np)
        bundle["decision_threshold"] = youden_threshold(oof, y_np)
        bundle["oof_probabilities"] = oof
        bundle["ood"] = OODDetector.fit(X_np, quantile=OOD_QUANTILE)
        logger.info(
            "[%s] calibration pass in %.2fs -> threshold=%.3f, conformal n=%d, ood thr=%.2f",
            disease_key,
            time.time() - t0,
            bundle["decision_threshold"],
            len(bundle["conformal_residuals"]),
            bundle["ood"].threshold,
        )

    joblib.dump(bundle, disease.model_path)
    logger.info("[%s] saved model bundle to %s", disease_key, disease.model_path)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=list(DISEASES), default=None, help="Train a single disease only")
    parser.add_argument("--device", default="cpu", help="TabPFN device: 'cpu', 'cuda', 'mps', or 'auto'")
    parser.add_argument("--n-estimators", type=int, default=8, help="TabPFN ensemble size")
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip the out-of-fold conformal / threshold / novelty-detector pass",
    )
    args = parser.parse_args()

    keys = [args.disease] if args.disease else list(DISEASES)
    for key in keys:
        train_disease(
            key,
            device=args.device,
            n_estimators=args.n_estimators,
            calibrate=not args.no_calibrate,
        )


if __name__ == "__main__":
    main()
