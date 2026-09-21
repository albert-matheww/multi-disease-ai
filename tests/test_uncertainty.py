"""Tests for `src/uncertainty.py` - the precomputed conformal band, the
learned decision threshold, and the robust-Mahalanobis novelty detector.

Everything here is model-agnostic, so a `LogisticRegression` factory stands
in for TabPFN (no license / weight download needed), exactly as in
`tests/test_prediction.py`.
"""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.uncertainty import (
    OODDetector,
    conformal_halfwidth,
    conformal_residuals,
    oof_probabilities,
    probability_band,
    youden_threshold,
)


@pytest.fixture(scope="module")
def toy_data():
    X, y = make_classification(
        n_samples=240,
        n_features=8,
        n_informative=5,
        weights=[0.65, 0.35],
        random_state=42,
    )
    return X.astype(float), y.astype(int)


@pytest.fixture(scope="module")
def oof(toy_data):
    X, y = toy_data
    return oof_probabilities(lambda: LogisticRegression(max_iter=1000), X, y, n_splits=5, random_state=42)


def test_oof_probabilities_shape_and_range(toy_data, oof):
    _, y = toy_data
    assert oof.shape == y.shape
    assert oof.min() >= 0.0 and oof.max() <= 1.0
    # OOF preds should be better than a coin flip on separable toy data.
    assert ((oof >= 0.5).astype(int) == y).mean() > 0.7


def test_oof_probabilities_is_deterministic(toy_data):
    X, y = toy_data
    a = oof_probabilities(lambda: LogisticRegression(max_iter=1000), X, y, random_state=7)
    b = oof_probabilities(lambda: LogisticRegression(max_iter=1000), X, y, random_state=7)
    assert np.allclose(a, b)


def test_conformal_residuals_sorted_and_bounded(toy_data, oof):
    _, y = toy_data
    res = conformal_residuals(oof, y)
    assert np.all(np.diff(res) >= 0)
    assert res.min() >= 0.0 and res.max() <= 1.0
    assert len(res) == len(y)


def test_conformal_halfwidth_shrinks_as_alpha_grows(toy_data, oof):
    _, y = toy_data
    res = conformal_residuals(oof, y)
    wide = conformal_halfwidth(res, alpha=0.05)
    narrow = conformal_halfwidth(res, alpha=0.20)
    assert wide >= narrow


def test_probability_band_contains_point_and_clips(toy_data, oof):
    _, y = toy_data
    res = conformal_residuals(oof, y)
    lo, hi = probability_band(0.98, res, alpha=0.1)
    assert 0.0 <= lo <= 0.98 <= hi <= 1.0
    lo2, hi2 = probability_band(0.0, res, alpha=0.1)
    assert lo2 == 0.0 and hi2 >= 0.0


def test_probability_band_empty_residuals_is_degenerate():
    lo, hi = probability_band(0.4, np.array([]), alpha=0.1)
    assert lo == pytest.approx(0.4) and hi == pytest.approx(0.4)


def test_youden_threshold_in_open_unit_interval(toy_data, oof):
    _, y = toy_data
    t = youden_threshold(oof, y)
    assert 0.0 < t < 1.0


def test_youden_threshold_single_class_falls_back():
    assert youden_threshold(np.array([0.2, 0.8, 0.5]), np.array([1, 1, 1])) == 0.5


def test_ood_detector_flags_extreme_points(toy_data):
    X, _ = toy_data
    det = OODDetector.fit(X, quantile=0.975)

    # The cohort centroid is maximally typical.
    centroid = X.mean(axis=0)
    assert not det.is_ood(centroid)
    assert det.novelty_score(centroid) < 0.10

    extreme = X.mean(axis=0) + 25.0 * X.std(axis=0)
    assert det.is_ood(extreme)
    assert det.novelty_score(extreme) == pytest.approx(1.0)

    # By construction ~2.5% of the training rows sit above the 97.5% quantile.
    flagged = det.distances(X) > det.threshold
    assert 0.0 < flagged.mean() < 0.10


def test_ood_detector_batch_matches_scalar(toy_data):
    X, _ = toy_data
    det = OODDetector.fit(X)
    batch = det.distances(X[:10])
    scalar = np.array([det.distance(row) for row in X[:10]])
    assert np.allclose(batch, scalar)


def test_ood_detector_is_picklable(tmp_path, toy_data):
    X, _ = toy_data
    det = OODDetector.fit(X)
    path = tmp_path / "ood.joblib"
    joblib.dump(det, path)
    reloaded = joblib.load(path)
    assert reloaded.distance(X[3]) == pytest.approx(det.distance(X[3]))
    assert reloaded.threshold == pytest.approx(det.threshold)
