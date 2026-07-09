"""Domain-driven feature engineering, one function per disease.

Every engineered feature is grounded in an established clinical heuristic
(cited in each docstring) rather than an arbitrary combination of columns.
Each function is pure: it takes the cleaned DataFrame produced by
`preprocessing.py` and returns a new DataFrame with additional columns,
so it can be unit-tested independently of the rest of the pipeline.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-3


def engineer_heart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cardiology-informed engineered features to the heart disease frame.

    - ``rate_pressure_product``: ``thalach * trestbps / 100`` — the
      Rate-Pressure Product, a standard clinical index of myocardial oxygen
      demand used in exercise cardiology.
    - ``heart_rate_reserve``: ``(220 - age) - thalach`` — the gap between the
      age-predicted maximum heart rate and the achieved heart rate; a large
      unused reserve (chronotropic incompetence) is an established marker of
      cardiovascular risk.
    - ``chol_age_ratio``: ``chol / age`` — normalizes cholesterol by age so
      that elevated lipid levels in younger patients are weighted more
      heavily, consistent with age-adjusted risk scoring (e.g. Framingham).
    """
    df = df.copy()
    df["rate_pressure_product"] = df["thalach"] * df["trestbps"] / 100.0
    df["heart_rate_reserve"] = (220 - df["age"]) - df["thalach"]
    df["chol_age_ratio"] = df["chol"] / df["age"].replace(0, _EPS)
    return df


def engineer_diabetes_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add metabolic engineered features to the diabetes frame.

    - ``glucose_bmi_interaction``: ``glucose * bmi / 100`` — captures the
      compounding effect of hyperglycemia and adiposity, two independent
      pillars of metabolic syndrome.
    - ``insulin_glucose_ratio``: ``insulin / (glucose + eps)`` — a simplified
      insulin-resistance proxy in the spirit of HOMA-IR (note: HOMA-IR is
      formally defined on *fasting* glucose/insulin, while this dataset's
      glucose is a 2-hour OGTT reading, so this ratio is a descriptive proxy
      rather than a clinically validated HOMA-IR score).
    """
    df = df.copy()
    df["glucose_bmi_interaction"] = df["glucose"] * df["bmi"] / 100.0
    df["insulin_glucose_ratio"] = df["insulin"] / (df["glucose"] + _EPS)
    return df


def engineer_ckd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a comorbidity burden feature to the CKD frame.

    - ``comorbidity_count``: sum of hypertension, diabetes mellitus, and
      coronary artery disease indicators (each 0/1). CKD risk compounds with
      the number of coexisting cardiometabolic conditions, so a simple count
      summarizes overall comorbidity burden in one feature.

    Expects ``htn``, ``dm``, ``cad`` to already be encoded as 0/1 (done in
    `preprocessing.py` before this function is called).
    """
    df = df.copy()
    df["comorbidity_count"] = df["htn"] + df["dm"] + df["cad"]
    return df


def engineer_liver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add liver-function-panel ratios to the ILPD frame.

    - ``ast_alt_ratio`` (De Ritis ratio): ``Sgot / (Sgpt + eps)`` — a
      well-established clinical ratio; values > 2 suggest alcoholic liver
      disease while values < 1 are more typical of viral hepatitis.
    - ``bilirubin_ratio``: ``DB / (TB + eps)`` — the direct-to-total
      bilirubin ratio, used clinically to distinguish conjugated
      (post-hepatic/obstructive) from unconjugated (pre-hepatic) causes of
      hyperbilirubinemia.
    """
    df = df.copy()
    df["ast_alt_ratio"] = df["Sgot"] / (df["Sgpt"] + _EPS)
    df["bilirubin_ratio"] = df["DB"] / (df["TB"] + _EPS)
    return df


FEATURE_ENGINEERS = {
    "heart": engineer_heart_features,
    "diabetes": engineer_diabetes_features,
    "ckd": engineer_ckd_features,
    "liver": engineer_liver_features,
}


def engineer_features(disease_key: str, df: pd.DataFrame) -> pd.DataFrame:
    """Dispatch to the correct engineering function for `disease_key`."""
    try:
        fn = FEATURE_ENGINEERS[disease_key]
    except KeyError as exc:
        raise KeyError(f"No feature engineering registered for '{disease_key}'") from exc
    logger.info("Engineering features for '%s'", disease_key)
    return fn(df)
