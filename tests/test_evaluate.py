"""`evaluate_disease` must score the same decision rule the app serves with."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import src.evaluate as evaluate
from src.config import DiseaseConfig, get_disease
from src.preprocessing import DiseasePreprocessor


@pytest.fixture
def stub_setup(tmp_path, monkeypatch):
    disease = get_disease("heart")
    DiseasePreprocessor(disease).run()

    bundle_path, metrics_path = tmp_path / "bundle.joblib", tmp_path / "metrics.json"
    monkeypatch.setattr(DiseaseConfig, "model_path", property(lambda self: bundle_path))
    monkeypatch.setattr(DiseaseConfig, "metrics_path", property(lambda self: metrics_path))
    monkeypatch.setattr(evaluate, "FIGURES_DIR", tmp_path)

    train = pd.read_csv(disease.processed_train_path)
    cols = [c for c in train.columns if c != "target"]
    clf = LogisticRegression(max_iter=1000).fit(train[cols].to_numpy(), train["target"])
    return bundle_path, metrics_path, clf, cols


def _run(stub_setup, threshold):
    bundle_path, metrics_path, clf, cols = stub_setup
    joblib.dump({"model": clf, "feature_order": cols, "decision_threshold": threshold}, bundle_path)
    metrics = evaluate.evaluate_disease("heart")
    return metrics, json.loads(metrics_path.read_text())


def test_served_threshold_metrics_are_reported_and_persisted(stub_setup):
    metrics, on_disk = _run(stub_setup, 0.3)
    assert metrics["decision_threshold"] == pytest.approx(0.3)
    assert set(on_disk["at_decision_threshold"]) >= {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    }


def test_lower_threshold_cannot_reduce_recall(stub_setup):
    metrics, _ = _run(stub_setup, 0.3)
    assert metrics["at_decision_threshold"]["recall"] >= metrics["recall"]


def test_bundle_without_a_threshold_falls_back_to_half(stub_setup):
    bundle_path, _, clf, cols = stub_setup
    joblib.dump({"model": clf, "feature_order": cols}, bundle_path)
    metrics = evaluate.evaluate_disease("heart")
    assert metrics["decision_threshold"] == 0.5
    assert metrics["at_decision_threshold"]["accuracy"] == pytest.approx(metrics["accuracy"])
