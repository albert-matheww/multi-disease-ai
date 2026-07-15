"""Integrity checks for the declarative disease registry in `src/config.py`.

These catch the class of bug that isn't a Python exception but a silent
data-consistency mistake - e.g. a `FieldSpec` whose category maps to a raw
value never seen in the source CSV, or a numeric_features list missing an
engineered feature name that `feature_engineering.py` actually produces.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DISEASES, RISK_THRESHOLDS, get_disease
from src.feature_engineering import FEATURE_ENGINEERS


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_raw_data_file_exists(disease_key):
    disease = get_disease(disease_key)
    assert disease.raw_path.exists(), f"Missing raw data for {disease_key}: {disease.raw_path}"


# Engineered feature names actually produced by feature_engineering.py, kept
# here as an independent ground truth (not derived from config.py) so this
# test can catch a real regression: batch prediction once demanded these as
# required raw CSV/form columns because field_names() wrongly returned
# numeric_features + categorical_features instead of the FieldSpec list.
_KNOWN_ENGINEERED_FEATURES = {
    "heart": {"rate_pressure_product", "heart_rate_reserve", "chol_age_ratio"},
    "diabetes": {"glucose_bmi_interaction", "insulin_glucose_ratio"},
    "ckd": {"comorbidity_count"},
    "liver": {"ast_alt_ratio", "bilirubin_ratio"},
}


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_field_names_are_raw_inputs_not_engineered(disease_key):
    """field_names() must return only genuine raw inputs, never engineered
    feature names - regression test for a real bug where batch prediction
    demanded engineered columns (e.g. rate_pressure_product) from the user.
    """
    disease = get_disease(disease_key)
    raw_fields = set(disease.field_names())
    engineered = _KNOWN_ENGINEERED_FEATURES[disease_key]

    assert raw_fields.isdisjoint(
        engineered
    ), f"{disease_key}: field_names() must not include engineered features {engineered & raw_fields}"
    # The engineered features must still feed the model (just not the user).
    assert engineered.issubset(set(disease.numeric_features))


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_every_field_is_numeric_or_categorical(disease_key):
    disease = get_disease(disease_key)
    declared = set(disease.numeric_features) | set(disease.categorical_features)
    for field in disease.fields:
        assert (
            field.name in declared
        ), f"{disease_key}: field '{field.name}' not declared in numeric/categorical lists"
        assert field.kind in ("numeric", "categorical")
        if field.kind == "categorical":
            assert field.categories, f"{disease_key}: categorical field '{field.name}' has no categories"


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_categorical_raw_values_appear_in_source_data(disease_key):
    """Every FieldSpec category's raw value must be a value that actually
    occurs in the source CSV for that column, otherwise the UI could submit
    a value the fitted OrdinalEncoder has never seen."""
    disease = get_disease(disease_key)
    df = pd.read_csv(disease.raw_path)

    for field in disease.fields:
        if field.kind != "categorical" or field.name not in df.columns:
            continue
        observed_raw = set(df[field.name].dropna().unique())
        # Numeric columns may be read back as float (e.g. thal: 3 vs 3.0) even
        # though FieldSpec declares the plain int/str form, so compare via a
        # numeric cast when possible before falling back to string equality.
        observed_numeric = set()
        for v in observed_raw:
            try:
                observed_numeric.add(float(v))
            except (TypeError, ValueError):
                pass
        observed_str = {str(v).strip() for v in observed_raw}

        for raw_value in field.categories.values():
            matches_numeric = False
            try:
                matches_numeric = float(raw_value) in observed_numeric
            except (TypeError, ValueError):
                pass
            assert matches_numeric or str(raw_value) in observed_str, (
                f"{disease_key}.{field.name}: category raw value {raw_value!r} "
                f"not found among observed values {observed_raw}"
            )


def test_every_disease_has_a_registered_feature_engineer():
    assert set(DISEASES) == set(FEATURE_ENGINEERS)


def test_risk_thresholds_ordered():
    assert 0.0 < RISK_THRESHOLDS["low"] < RISK_THRESHOLDS["moderate"] < 1.0


def test_disease_keys_match_dict_keys():
    for key, disease in DISEASES.items():
        assert disease.key == key
