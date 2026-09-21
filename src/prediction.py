"""End-to-end single-patient and batch inference for a trained disease model.

`DiseasePredictor` is the one object the Streamlit app and the report
generator both depend on: given a raw patient record (the same field names
and human units used in the UI form / config.FieldSpec), it applies the
saved preprocessing artifact, runs the TabPFN model, and returns a
structured `PredictionResult` with probability, a distribution-free
probability band, a risk level, an out-of-distribution flag, and a
SHAP-based explanation - everything `report.py` needs to render a report and
everything the dashboard needs to render a result card.

The conformal band, learned decision threshold and novelty detector are all
read straight out of the model bundle (precomputed in `src/train.py`), so a
prediction is still one TabPFN forward pass plus the SHAP explanation, with
no extra model calls.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.artifacts import PreprocessArtifact
from src.config import CONFORMAL_ALPHA, RISK_THRESHOLDS, DiseaseConfig, get_disease
from src.explainability import DiseaseExplainer
from src.preprocessing import DiseasePreprocessor
from src.uncertainty import OODDetector, probability_band

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# A smaller SHAP background than the global-plot default keeps the interactive
# per-patient explanation snappy without materially changing the ranking.
_INTERACTIVE_SHAP_BACKGROUND = 10


@dataclass
class PredictionResult:
    disease_key: str
    disease_display_name: str
    probability: float
    predicted_label: str
    risk_level: str
    confidence: float
    top_contributors: list[dict] = field(default_factory=list)
    patient_input: dict = field(default_factory=dict)
    timestamp: str = ""
    # --- uncertainty / novelty (precomputed add-ons; safe defaults if absent) ---
    probability_low: float = 0.0
    probability_high: float = 1.0
    band_alpha: float = CONFORMAL_ALPHA
    decision_threshold: float = 0.5
    out_of_distribution: bool = False
    novelty_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def risk_level_from_probability(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    if probability < RISK_THRESHOLDS["moderate"]:
        return "Moderate"
    return "High"


class DiseasePredictor:
    """Loads a trained model + preprocessing artifact + SHAP explainer for one disease."""

    def __init__(self, disease_key: str):
        self.disease: DiseaseConfig = get_disease(disease_key)

        if not self.disease.model_path.exists():
            raise FileNotFoundError(
                f"No trained model for '{disease_key}' at {self.disease.model_path}. "
                "Run `python -m src.train --disease "
                f"{disease_key}` first."
            )

        bundle = joblib.load(self.disease.model_path)
        self.model = bundle["model"]
        self.feature_order: list[str] = bundle["feature_order"]
        self.artifact: PreprocessArtifact = PreprocessArtifact.load(self.disease.preprocessor_path)
        self.preprocessor = DiseasePreprocessor(self.disease)
        self._explainer: DiseaseExplainer | None = None
        self._explain_cache: dict[tuple, list[dict]] = {}

        # Precomputed uncertainty add-ons - `.get` so older/stub bundles (and the
        # LogisticRegression test stand-in) that lack them still load fine.
        self.decision_threshold: float = float(bundle.get("decision_threshold", 0.5) or 0.5)
        self._conformal_residuals = bundle.get("conformal_residuals")
        self._conformal_alpha: float = float(bundle.get("conformal_alpha", CONFORMAL_ALPHA))
        self._ood: OODDetector | None = bundle.get("ood")

    @property
    def explainer(self) -> DiseaseExplainer:
        if self._explainer is None:
            train_df = pd.read_csv(self.disease.processed_train_path)
            X_train = train_df[self.feature_order].to_numpy()
            self._explainer = DiseaseExplainer(
                self.disease.key,
                self.model,
                self.feature_order,
                X_train,
                max_background=_INTERACTIVE_SHAP_BACKGROUND,
            )
        return self._explainer

    def _encode(self, raw_record: dict) -> np.ndarray:
        raw_df = pd.DataFrame([raw_record])
        encoded_df = self.preprocessor.transform_new(self.artifact, raw_df)
        return encoded_df.to_numpy()[0]

    def _explain_cached(self, x_encoded: np.ndarray, top_n: int) -> list[dict]:
        key = (np.round(x_encoded, 4).tobytes(), top_n)
        hit = self._explain_cache.get(key)
        if hit is None:
            hit = self.explainer.explain_instance(x_encoded, top_n=top_n)["top_contributors"]
            self._explain_cache[key] = hit
        return hit

    def predict(self, raw_record: dict, explain: bool = True, top_n: int = 5) -> PredictionResult:
        """Predict disease risk for one raw patient record.

        `raw_record` keys must match `disease.field_names()` (the field
        names declared in `config.py`), using raw category strings/values
        for categorical fields (see `FieldSpec.categories`), not pre-encoded
        integers.
        """
        x_encoded = self._encode(raw_record)
        proba = float(self.model.predict_proba(x_encoded.reshape(1, -1))[0, 1])
        predicted_class = int(proba >= self.decision_threshold)
        label = self.disease.positive_label if predicted_class == 1 else self.disease.negative_label

        low, high = proba, proba
        if self._conformal_residuals is not None:
            low, high = probability_band(proba, self._conformal_residuals, self._conformal_alpha)

        is_ood, novelty = False, 0.0
        if self._ood is not None:
            is_ood = bool(self._ood.is_ood(x_encoded))
            novelty = float(self._ood.novelty_score(x_encoded))

        top_contributors: list[dict] = []
        if explain:
            top_contributors = self._explain_cached(x_encoded, top_n)

        return PredictionResult(
            disease_key=self.disease.key,
            disease_display_name=self.disease.display_name,
            probability=proba,
            predicted_label=label,
            risk_level=risk_level_from_probability(proba),
            confidence=max(proba, 1 - proba),
            top_contributors=top_contributors,
            patient_input=raw_record,
            timestamp=datetime.now(timezone.utc).isoformat(),
            probability_low=low,
            probability_high=high,
            band_alpha=self._conformal_alpha,
            decision_threshold=self.decision_threshold,
            out_of_distribution=is_ood,
            novelty_score=novelty,
        )

    def predict_batch(self, records_df: pd.DataFrame) -> pd.DataFrame:
        """Predict for many patients at once (CSV batch-upload feature).

        `records_df` must have one column per `disease.field_names()`, raw
        (unencoded) values. Returns the input frame with `probability`,
        `probability_low`/`probability_high` (conformal band),
        `predicted_label`, `risk_level`, and `out_of_distribution` columns
        appended. Explanations are omitted for batch scoring (too slow to
        compute SHAP per row for large uploads); use `.predict()` for a
        single-record explanation.
        """
        missing = set(self.feature_order_raw()) - set(records_df.columns)
        if missing:
            raise ValueError(f"Missing required columns for batch prediction: {sorted(missing)}")

        encoded_df = self.preprocessor.transform_new(self.artifact, records_df.copy())
        X = encoded_df.to_numpy()
        proba = self.model.predict_proba(X)[:, 1]
        pred_class = (proba >= self.decision_threshold).astype(int)

        out = records_df.copy()
        out["probability"] = proba
        if self._conformal_residuals is not None:
            bands = [probability_band(p, self._conformal_residuals, self._conformal_alpha) for p in proba]
            out["probability_low"] = [b[0] for b in bands]
            out["probability_high"] = [b[1] for b in bands]
        out["predicted_label"] = np.where(
            pred_class == 1, self.disease.positive_label, self.disease.negative_label
        )
        out["risk_level"] = [risk_level_from_probability(p) for p in proba]
        if self._ood is not None:
            out["out_of_distribution"] = self._ood.distances(X) > self._ood.threshold
        return out

    def feature_order_raw(self) -> list[str]:
        """The raw input field names expected in a patient record (pre-engineering)."""
        return self.disease.field_names()


@lru_cache(maxsize=None)
def get_predictor(disease_key: str) -> DiseasePredictor:
    """Process-wide cached predictor loader (safe to call repeatedly, e.g. from Streamlit)."""
    return DiseasePredictor(disease_key)
