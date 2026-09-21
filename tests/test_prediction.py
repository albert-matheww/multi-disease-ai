"""Tests for `src/prediction.py`.

`DiseasePredictor` is model-agnostic (it only relies on `predict_proba`, the
same sklearn-style interface TabPFNClassifier implements), so these tests
swap in a lightweight `LogisticRegression` as the saved "model" to verify
the wiring - preprocessing -> model -> SHAP -> risk level -> PredictionResult
- without requiring network access to download TabPFN's weights. The
regression-quality of the *classifier itself* is exercised separately by
`src/evaluate.py` once a real TabPFN model is trained.
"""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import DiseaseConfig, get_disease
from src.prediction import DiseasePredictor, risk_level_from_probability
from src.preprocessing import DiseasePreprocessor
from src.uncertainty import OODDetector, conformal_residuals, oof_probabilities, youden_threshold


@pytest.mark.parametrize(
    "probability,expected",
    [
        (0.0, "Low"),
        (0.1, "Low"),
        (0.32, "Low"),
        (0.33, "Moderate"),
        (0.5, "Moderate"),
        (0.65, "Moderate"),
        (0.66, "High"),
        (0.9, "High"),
        (1.0, "High"),
    ],
)
def test_risk_level_thresholds(probability, expected):
    assert risk_level_from_probability(probability) == expected


@pytest.fixture
def heart_predictor(tmp_path, monkeypatch) -> DiseasePredictor:
    """Build a real DiseasePredictor for 'heart', backed by a fitted
    LogisticRegression stand-in saved to a temp path (so no real training
    run or trained TabPFN artifact is required for this test)."""
    disease = get_disease("heart")

    # Ensure processed train/test CSVs + preprocessor artifact exist.
    pre = DiseasePreprocessor(disease)
    pre.run()

    stub_model_path = tmp_path / "heart_stub_model.joblib"
    monkeypatch.setattr(DiseaseConfig, "model_path", property(lambda self: stub_model_path))

    train_df = pd.read_csv(disease.processed_train_path)
    feature_order = [c for c in train_df.columns if c != "target"]
    clf = LogisticRegression(max_iter=1000).fit(train_df[feature_order], train_df["target"])
    joblib.dump({"model": clf, "feature_order": feature_order, "disease_key": "heart"}, stub_model_path)

    return DiseasePredictor("heart")


SAMPLE_HEART_PATIENT = {
    "age": 63,
    "sex": 1,
    "cp": 4,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": 6,
}


def test_predict_returns_consistent_result(heart_predictor):
    result = heart_predictor.predict(SAMPLE_HEART_PATIENT)

    assert result.disease_key == "heart"
    assert 0.0 <= result.probability <= 1.0
    assert result.confidence == max(result.probability, 1 - result.probability)
    assert result.risk_level == risk_level_from_probability(result.probability)
    assert result.predicted_label in (
        heart_predictor.disease.positive_label,
        heart_predictor.disease.negative_label,
    )
    assert result.patient_input == SAMPLE_HEART_PATIENT


def test_predict_explanation_present_and_sorted_by_magnitude(heart_predictor):
    result = heart_predictor.predict(SAMPLE_HEART_PATIENT, top_n=5)
    assert len(result.top_contributors) == 5

    magnitudes = [abs(c["shap_value"]) for c in result.top_contributors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for c in result.top_contributors:
        assert c["direction"] in ("increases_risk", "decreases_risk")


def test_predict_without_explanation_is_faster_and_empty(heart_predictor):
    result = heart_predictor.predict(SAMPLE_HEART_PATIENT, explain=False)
    assert result.top_contributors == []


def test_predict_batch_matches_single_predictions(heart_predictor):
    df = pd.DataFrame([SAMPLE_HEART_PATIENT, SAMPLE_HEART_PATIENT])
    batch_result = heart_predictor.predict_batch(df)

    single_result = heart_predictor.predict(SAMPLE_HEART_PATIENT, explain=False)
    assert batch_result["probability"].iloc[0] == pytest.approx(single_result.probability)
    assert batch_result["probability"].iloc[1] == pytest.approx(single_result.probability)
    assert set(batch_result["risk_level"]) == {single_result.risk_level}


def test_predict_batch_requires_all_raw_fields(heart_predictor):
    incomplete_df = pd.DataFrame([{"age": 50}])
    with pytest.raises(ValueError, match="Missing required columns"):
        heart_predictor.predict_batch(incomplete_df)


def test_feature_order_raw_excludes_engineered_features(heart_predictor):
    raw_fields = heart_predictor.feature_order_raw()
    assert "rate_pressure_product" not in raw_fields
    assert "heart_rate_reserve" not in raw_fields
    assert set(raw_fields) == set(SAMPLE_HEART_PATIENT)


@pytest.fixture
def heart_predictor_calibrated(tmp_path, monkeypatch) -> DiseasePredictor:
    """Like `heart_predictor` but the saved bundle also carries the precomputed
    conformal residuals, learned threshold, and novelty detector - so the
    uncertainty wiring in `DiseasePredictor.predict` is exercised end to end."""
    disease = get_disease("heart")
    DiseasePreprocessor(disease).run()

    stub_model_path = tmp_path / "heart_calibrated_model.joblib"
    monkeypatch.setattr(DiseaseConfig, "model_path", property(lambda self: stub_model_path))

    train_df = pd.read_csv(disease.processed_train_path)
    feature_order = [c for c in train_df.columns if c != "target"]
    X = train_df[feature_order].to_numpy()
    y = train_df["target"].to_numpy()
    clf = LogisticRegression(max_iter=1000).fit(X, y)

    oof = oof_probabilities(lambda: LogisticRegression(max_iter=1000), X, y, n_splits=4, random_state=42)
    joblib.dump(
        {
            "model": clf,
            "feature_order": feature_order,
            "disease_key": "heart",
            "decision_threshold": youden_threshold(oof, y),
            "conformal_residuals": conformal_residuals(oof, y),
            "conformal_alpha": 0.10,
            "ood": OODDetector.fit(X),
        },
        stub_model_path,
    )
    return DiseasePredictor("heart")


def test_predict_populates_conformal_band(heart_predictor_calibrated):
    result = heart_predictor_calibrated.predict(SAMPLE_HEART_PATIENT, explain=False)
    assert 0.0 <= result.probability_low <= result.probability <= result.probability_high <= 1.0
    assert result.probability_high > result.probability_low  # a real interval, not degenerate
    assert 0.0 < result.decision_threshold < 1.0
    assert 0.0 <= result.novelty_score <= 1.0
    assert isinstance(result.out_of_distribution, bool)


def test_predicted_label_respects_learned_threshold(heart_predictor_calibrated):
    p = heart_predictor_calibrated
    result = p.predict(SAMPLE_HEART_PATIENT, explain=False)
    expected_positive = result.probability >= p.decision_threshold
    assert (result.predicted_label == p.disease.positive_label) == expected_positive


def test_batch_emits_band_and_ood_columns(heart_predictor_calibrated):
    df = pd.DataFrame([SAMPLE_HEART_PATIENT, SAMPLE_HEART_PATIENT])
    out = heart_predictor_calibrated.predict_batch(df)
    for col in ("probability_low", "probability_high", "out_of_distribution"):
        assert col in out.columns
    assert (out["probability_low"] <= out["probability"]).all()
    assert (out["probability"] <= out["probability_high"]).all()


def test_explanation_is_cached(heart_predictor_calibrated):
    p = heart_predictor_calibrated
    p.predict(SAMPLE_HEART_PATIENT, top_n=5)
    assert len(p._explain_cache) == 1
    p.predict(SAMPLE_HEART_PATIENT, top_n=5)  # identical input -> cache hit, no new entry
    assert len(p._explain_cache) == 1
