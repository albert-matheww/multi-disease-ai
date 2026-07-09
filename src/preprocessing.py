"""Data cleaning, imputation, outlier handling, encoding, and splitting.

`DiseasePreprocessor` implements one consistent pipeline applied to all four
diseases, driven entirely by the declarative `DiseaseConfig` objects in
`config.py`. All statistics used for imputation, outlier clipping, and
categorical encoding are fit on the training split only, then reused
(via a saved `PreprocessArtifact`) to transform the test split and, later,
live patient input at inference time — this avoids train/test leakage and
guarantees the Streamlit app encodes new patients exactly as training data
was encoded.

Pipeline order (see module docstring in `feature_engineering.py` for why
engineered features are computed *after* cleaning):

1. ``clean_raw``      - whitespace/typo fixes, sentinel-missing -> NaN, target
                         mapping, duplicate removal (no fitting; deterministic).
2. stratified split   - train/test split on the cleaned base columns.
3. impute             - median (numeric) / most-frequent (categorical), fit on train.
4. clip outliers      - 1.5x IQR winsorization on numeric columns, fit on train.
5. encode categorical - ordinal encoding, fit on train.
6. engineer features  - domain ratios/interactions computed from the already
                         clean numeric/categorical columns (see `feature_engineering.py`).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from src.artifacts import PreprocessArtifact
from src.config import RANDOM_STATE, TEST_SIZE, DiseaseConfig, get_disease
from src.feature_engineering import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Sentinel-missing columns: a raw value of 0 is not biologically plausible
# and actually encodes a missing measurement (documented in data/README.md).
_ZERO_AS_MISSING = {
    "diabetes": ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"],
}


class DiseasePreprocessor:
    """Cleans, splits, imputes, clips, encodes, and engineers features for one disease."""

    def __init__(self, disease: DiseaseConfig):
        self.disease = disease

    # ------------------------------------------------------------------ #
    # Step 1: structural cleaning (deterministic, no fitting)
    # ------------------------------------------------------------------ #
    def load_raw(self) -> pd.DataFrame:
        df = pd.read_csv(self.disease.raw_path)
        logger.info("[%s] loaded raw data: %s rows x %s cols", self.disease.key, *df.shape)
        return df

    def clean_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Strip stray whitespace/tabs from every string-like column (e.g. CKD's
        # "ckd\t" target label and "\tno" category value). Converted back to
        # plain object dtype with np.nan (not pd.NA) so downstream sklearn
        # transformers - which test missingness via `X != X` - work correctly;
        # pandas' nullable StringDtype uses pd.NA, whose `__bool__` raises.
        str_cols = df.select_dtypes(include=["object", "string"]).columns
        for col in str_cols:
            stripped = df[col].astype("string").str.strip()
            df[col] = stripped.to_numpy(dtype=object, na_value=np.nan)

        # Dataset-specific sentinel-missing correction.
        for col in _ZERO_AS_MISSING.get(self.disease.key, []):
            n_zero = int((df[col] == 0).sum())
            if n_zero:
                logger.info("[%s] treating %d zero-values in '%s' as missing", self.disease.key, n_zero, col)
                df.loc[df[col] == 0, col] = np.nan

        # Map raw target to binary 0/1 "target" column.
        df["target"] = df[self.disease.target_column].map(self.disease.target_mapper)
        drop_cols = set(self.disease.drop_columns) | {self.disease.target_column}
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        n_dupes = before - len(df)
        if n_dupes:
            logger.info("[%s] dropped %d duplicate rows", self.disease.key, n_dupes)

        return df

    # ------------------------------------------------------------------ #
    # Step 2: split
    # ------------------------------------------------------------------ #
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=df["target"],
        )
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Steps 3-6: fit on train, transform train & test
    # ------------------------------------------------------------------ #
    def fit_transform(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessArtifact]:
        base_num = [c for c in self.disease.numeric_features if c in train_df.columns]
        base_cat = [c for c in self.disease.categorical_features if c in train_df.columns]

        train_df = train_df.copy()
        test_df = test_df.copy()

        # --- Impute ---------------------------------------------------
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")

        if base_num:
            train_df[base_num] = num_imputer.fit_transform(train_df[base_num])
            test_df[base_num] = num_imputer.transform(test_df[base_num])
        if base_cat:
            train_df[base_cat] = cat_imputer.fit_transform(train_df[base_cat])
            test_df[base_cat] = cat_imputer.transform(test_df[base_cat])

        # --- Outlier clipping (1.5x IQR winsorization), numeric only ---
        outlier_bounds: dict[str, tuple[float, float]] = {}
        for col in base_num:
            q1, q3 = train_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_bounds[col] = (float(lower), float(upper))
            train_df[col] = train_df[col].clip(lower, upper)
            test_df[col] = test_df[col].clip(lower, upper)

        # --- Categorical ordinal encoding -------------------------------
        ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        if base_cat:
            train_df[base_cat] = ordinal_encoder.fit_transform(train_df[base_cat])
            test_df[base_cat] = ordinal_encoder.transform(test_df[base_cat])
            train_df[base_cat] = train_df[base_cat].astype(float)
            test_df[base_cat] = test_df[base_cat].astype(float)

        # --- Domain feature engineering ---------------------------------
        train_df = engineer_features(self.disease.key, train_df)
        test_df = engineer_features(self.disease.key, test_df)

        feature_order = self.disease.numeric_features + self.disease.categorical_features
        feature_order = [c for c in feature_order if c in train_df.columns]
        categorical_feature_indices = [
            feature_order.index(c) for c in self.disease.categorical_features if c in feature_order
        ]

        artifact = PreprocessArtifact(
            disease_key=self.disease.key,
            numeric_features=self.disease.numeric_features,
            categorical_features=self.disease.categorical_features,
            base_numeric_features=base_num,
            base_categorical_features=base_cat,
            numeric_imputer=num_imputer,
            categorical_imputer=cat_imputer,
            ordinal_encoder=ordinal_encoder,
            outlier_bounds=outlier_bounds,
            feature_order=feature_order,
            categorical_feature_indices=categorical_feature_indices,
        )

        cols = feature_order + ["target"]
        return train_df[cols], test_df[cols], artifact

    def transform_new(self, artifact: PreprocessArtifact, raw_record: pd.DataFrame) -> pd.DataFrame:
        """Apply a fitted `PreprocessArtifact` to new, raw patient record(s).

        Used by `prediction.py` so a single patient submitted through the
        Streamlit form is encoded identically to the training data.
        """
        df = raw_record.copy()

        if artifact.base_numeric_features:
            df[artifact.base_numeric_features] = artifact.numeric_imputer.transform(
                df[artifact.base_numeric_features]
            )
            for col in artifact.base_numeric_features:
                lower, upper = artifact.outlier_bounds[col]
                df[col] = df[col].clip(lower, upper)

        if artifact.base_categorical_features:
            df[artifact.base_categorical_features] = artifact.categorical_imputer.transform(
                df[artifact.base_categorical_features]
            )
            df[artifact.base_categorical_features] = artifact.ordinal_encoder.transform(
                df[artifact.base_categorical_features]
            ).astype(float)

        df = engineer_features(artifact.disease_key, df)
        return df[artifact.feature_order]

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, PreprocessArtifact]:
        raw = self.load_raw()
        cleaned = self.clean_raw(raw)
        train_df, test_df = self.split(cleaned)
        train_proc, test_proc, artifact = self.fit_transform(train_df, test_df)

        train_proc.to_csv(self.disease.processed_train_path, index=False)
        test_proc.to_csv(self.disease.processed_test_path, index=False)
        artifact.save(self.disease.preprocessor_path)

        summary = {
            "disease": self.disease.key,
            "n_train": len(train_proc),
            "n_test": len(test_proc),
            "n_features": len(artifact.feature_order),
            "train_positive_rate": float(train_proc["target"].mean()),
            "test_positive_rate": float(test_proc["target"].mean()),
        }
        logger.info("[%s] preprocessing complete: %s", self.disease.key, summary)
        return train_proc, test_proc, artifact


def run_all() -> None:
    from src.config import DISEASES

    for key in DISEASES:
        DiseasePreprocessor(get_disease(key)).run()


if __name__ == "__main__":
    run_all()
