"""Unit tests for the clinically-grounded engineered features.

Each function is pure (DataFrame in, DataFrame out), so these are plain
input/output checks against hand-computed expected values.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.feature_engineering import (
    engineer_ckd_features,
    engineer_diabetes_features,
    engineer_heart_features,
    engineer_liver_features,
)


def test_heart_features():
    df = pd.DataFrame({"age": [40, 50], "thalach": [150, 100], "trestbps": [120, 140], "chol": [200, 250]})
    out = engineer_heart_features(df)

    assert out["rate_pressure_product"].tolist() == pytest.approx([150 * 120 / 100, 100 * 140 / 100])
    assert out["heart_rate_reserve"].tolist() == pytest.approx([(220 - 40) - 150, (220 - 50) - 100])
    assert out["chol_age_ratio"].tolist() == pytest.approx([200 / 40, 250 / 50])


def test_diabetes_features():
    df = pd.DataFrame({"glucose": [100, 200], "bmi": [25.0, 30.0], "insulin": [50.0, 0.0]})
    out = engineer_diabetes_features(df)

    assert out["glucose_bmi_interaction"].tolist() == pytest.approx([100 * 25.0 / 100, 200 * 30.0 / 100])
    assert out["insulin_glucose_ratio"].iloc[0] == pytest.approx(50.0 / 100.001, rel=1e-3)
    assert out["insulin_glucose_ratio"].iloc[1] >= 0  # 0 insulin -> ~0, never negative/NaN


def test_ckd_features_sums_comorbidities():
    df = pd.DataFrame({"htn": [0.0, 1.0, 1.0], "dm": [0.0, 1.0, 0.0], "cad": [0.0, 1.0, 1.0]})
    out = engineer_ckd_features(df)
    assert out["comorbidity_count"].tolist() == [0.0, 3.0, 2.0]


def test_liver_features():
    df = pd.DataFrame({"Sgot": [40.0, 80.0], "Sgpt": [40.0, 20.0], "DB": [0.2, 5.0], "TB": [1.0, 10.0]})
    out = engineer_liver_features(df)

    assert out["ast_alt_ratio"].iloc[0] == pytest.approx(1.0, rel=1e-2)
    assert out["ast_alt_ratio"].iloc[1] == pytest.approx(4.0, rel=1e-2)
    assert out["bilirubin_ratio"].iloc[0] == pytest.approx(0.2, rel=1e-2)
    assert out["bilirubin_ratio"].iloc[1] == pytest.approx(0.5, rel=1e-2)


def test_engineered_functions_do_not_mutate_input():
    df = pd.DataFrame({"htn": [0.0], "dm": [0.0], "cad": [0.0]})
    original = df.copy()
    engineer_ckd_features(df)
    pd.testing.assert_frame_equal(df, original)
