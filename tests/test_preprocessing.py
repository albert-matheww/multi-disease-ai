"""Tests for the cleaning/imputation/encoding pipeline in `src/preprocessing.py`.

Runs against the real downloaded raw CSVs (small, already in `data/raw/`) so
these double as an integration check that the whole pipeline still works
end-to-end after any refactor - no network access or trained model needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DISEASES, get_disease
from src.preprocessing import DiseasePreprocessor


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_clean_raw_has_no_whitespace_artifacts(disease_key):
    disease = get_disease(disease_key)
    pre = DiseasePreprocessor(disease)
    cleaned = pre.clean_raw(pre.load_raw())

    str_cols = cleaned.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        values = cleaned[col].dropna()
        assert (values == values.str.strip()).all(), f"{disease_key}.{col} has un-stripped whitespace"


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_clean_raw_produces_binary_target(disease_key):
    disease = get_disease(disease_key)
    pre = DiseasePreprocessor(disease)
    cleaned = pre.clean_raw(pre.load_raw())
    assert set(cleaned["target"].unique()) <= {0, 1}
    assert cleaned["target"].isna().sum() == 0


def test_diabetes_zero_sentinel_becomes_missing():
    disease = get_disease("diabetes")
    pre = DiseasePreprocessor(disease)
    raw = pre.load_raw()
    n_zero_glucose = int((raw["glucose"] == 0).sum())
    assert n_zero_glucose > 0  # sanity: the raw data really has this quirk

    cleaned = pre.clean_raw(raw)
    assert (cleaned["glucose"] == 0).sum() == 0
    assert cleaned["glucose"].isna().sum() == n_zero_glucose


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_fit_transform_has_no_missing_values(disease_key):
    disease = get_disease(disease_key)
    pre = DiseasePreprocessor(disease)
    cleaned = pre.clean_raw(pre.load_raw())
    train_df, test_df = pre.split(cleaned)
    train_proc, test_proc, artifact = pre.fit_transform(train_df, test_df)

    assert train_proc.isna().sum().sum() == 0
    assert test_proc.isna().sum().sum() == 0
    assert set(artifact.feature_order) == set(train_proc.columns) - {"target"}


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_fit_transform_is_leakage_safe(disease_key):
    """Outlier bounds must be computed from the (imputed) training split
    only, never see the test split's values. Imputation happens before
    outlier-bound computation in `fit_transform`, so the expected quantiles
    here are computed on the *imputed* training column to match."""
    disease = get_disease(disease_key)
    pre = DiseasePreprocessor(disease)
    cleaned = pre.clean_raw(pre.load_raw())
    train_df, test_df = pre.split(cleaned)
    _, _, artifact = pre.fit_transform(train_df.copy(), test_df.copy())

    imputed_train = pd.DataFrame(
        artifact.numeric_imputer.transform(train_df[artifact.base_numeric_features]),
        columns=artifact.base_numeric_features,
    )

    for col, (lower, upper) in artifact.outlier_bounds.items():
        train_q1, train_q3 = imputed_train[col].quantile([0.25, 0.75])
        train_iqr = train_q3 - train_q1
        assert lower == pytest.approx(train_q1 - 1.5 * train_iqr)
        assert upper == pytest.approx(train_q3 + 1.5 * train_iqr)


@pytest.mark.parametrize("disease_key", list(DISEASES))
def test_transform_new_matches_training_pipeline_shape(disease_key):
    """A single raw record run through transform_new must produce the exact
    feature columns (same names/order) the model was trained on."""
    disease = get_disease(disease_key)
    pre = DiseasePreprocessor(disease)
    cleaned = pre.clean_raw(pre.load_raw())
    train_df, test_df = pre.split(cleaned)
    _, _, artifact = pre.fit_transform(train_df, test_df)

    raw_row = pd.DataFrame([{f.name: f.default for f in disease.fields}])
    encoded = pre.transform_new(artifact, raw_row)

    assert list(encoded.columns) == artifact.feature_order
    assert encoded.isna().sum().sum() == 0
    assert encoded.shape[0] == 1


def test_ckd_whitespace_target_label_is_cleaned():
    """CKD's raw target has a "ckd\\t" variant (trailing tab) - must map to
    the same binary value as the clean "ckd" label."""
    disease = get_disease("ckd")
    pre = DiseasePreprocessor(disease)
    raw = pre.load_raw()
    assert raw["class"].astype(str).str.contains("\t").any()  # sanity check on the quirk

    cleaned = pre.clean_raw(raw)
    assert cleaned["target"].isna().sum() == 0
    assert set(cleaned["target"].unique()) == {0, 1}
