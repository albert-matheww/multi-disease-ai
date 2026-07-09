"""Shared, picklable artifact types.

Kept in their own module (rather than defined in `preprocessing.py`) so that
`joblib`/`pickle` always records their defining module as `src.artifacts`,
regardless of which script is run as `__main__`. A class pickled while
defined in `__main__` (e.g. `python -m src.preprocessing`) cannot be
unpickled from a different entry point (`python -m src.train`, the
Streamlit app, etc.) - that mismatch is exactly what this module avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class PreprocessArtifact:
    """Everything needed to transform a *raw* patient record into model-ready
    features identically to how the training data was transformed."""

    disease_key: str
    numeric_features: list[str]
    categorical_features: list[str]
    base_numeric_features: list[str]
    base_categorical_features: list[str]
    numeric_imputer: SimpleImputer
    categorical_imputer: SimpleImputer
    ordinal_encoder: OrdinalEncoder
    outlier_bounds: dict[str, tuple[float, float]]
    feature_order: list[str]
    categorical_feature_indices: list[int]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "PreprocessArtifact":
        return joblib.load(path)
