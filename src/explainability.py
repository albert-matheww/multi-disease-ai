"""SHAP-based explainability for TabPFN predictions.

TabPFN is a transformer, not a tree ensemble or a linear model, so SHAP's
fast closed-form explainers (`TreeExplainer`, `LinearExplainer`) do not apply.
Instead this module uses SHAP's **model-agnostic** `PermutationExplainer`
(via `shap.Explainer`) against `model.predict_proba`, which works with any
callable and is the officially recommended SHAP approach for black-box
models. To keep runtime reasonable given TabPFN's in-context-learning cost
per call, the background reference set is summarized down to a small
k-means prototype set rather than using the full training data.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt

from src.config import FIGURES_DIR, get_disease

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MAX_BACKGROUND = 25


class DiseaseExplainer:
    """Wraps a fitted disease model with a SHAP explainer for that model."""

    def __init__(
        self,
        disease_key: str,
        model,
        feature_order: list[str],
        background_data: np.ndarray,
        max_background: int = _MAX_BACKGROUND,
    ):
        self.disease_key = disease_key
        self.model = model
        self.feature_order = feature_order

        n_background = min(max_background, len(background_data))
        if len(background_data) > n_background:
            self.background = shap.kmeans(background_data, n_background).data
        else:
            self.background = background_data

        self._predict_fn = lambda x: self.model.predict_proba(np.asarray(x))[:, 1]
        self.explainer = shap.Explainer(self._predict_fn, self.background, feature_names=feature_order)

    @classmethod
    def from_disease_key(cls, disease_key: str) -> "DiseaseExplainer":
        disease = get_disease(disease_key)
        bundle = joblib.load(disease.model_path)
        train_df = pd.read_csv(disease.processed_train_path)
        X_train = train_df[bundle["feature_order"]].to_numpy()
        return cls(disease_key, bundle["model"], bundle["feature_order"], X_train)

    def explain(self, X: np.ndarray) -> shap.Explanation:
        """Compute SHAP values for one or more rows of already-encoded features."""
        return self.explainer(np.asarray(X))

    def explain_instance(self, x_row: np.ndarray, top_n: int = 5) -> dict:
        """Explain a single patient record; returns a JSON-serializable summary.

        Used by `prediction.py` / the Streamlit app to render a local
        explanation ("why did the model predict this?") alongside the
        probability score.
        """
        explanation = self.explain(x_row.reshape(1, -1))
        values = explanation.values[0]
        data = explanation.data[0]
        base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])

        contributions = [
            {
                "feature": name,
                "value": float(val),
                "shap_value": float(sv),
                "direction": "increases_risk" if sv > 0 else "decreases_risk",
            }
            for name, val, sv in zip(self.feature_order, data, values)
        ]
        contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

        return {
            "base_value": base_value,
            "predicted_value": float(base_value + values.sum()),
            "top_contributors": contributions[:top_n],
            "all_contributions": contributions,
        }

    def plot_summary(self, X_sample: np.ndarray, max_display: int = 12) -> plt.Figure:
        """Global feature-importance summary (mean |SHAP value|) over a sample of rows."""
        explanation = self.explain(X_sample)
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(
            explanation.values,
            X_sample,
            feature_names=self.feature_order,
            max_display=max_display,
            show=False,
            plot_type="bar",
        )
        plt.title(f"Global Feature Importance (mean |SHAP value|) - {self.disease_key.title()}")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"{self.disease_key}_shap_summary.png", bbox_inches="tight")
        return fig

    def plot_beeswarm(self, X_sample: np.ndarray, max_display: int = 12) -> plt.Figure:
        explanation = self.explain(X_sample)
        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(
            explanation.values,
            X_sample,
            feature_names=self.feature_order,
            max_display=max_display,
            show=False,
        )
        plt.title(f"SHAP Value Distribution - {self.disease_key.title()}")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f"{self.disease_key}_shap_beeswarm.png", bbox_inches="tight")
        return fig

    def plot_waterfall(self, x_row: np.ndarray, max_display: int = 10) -> plt.Figure:
        """Local explanation plot for a single patient record."""
        explanation = self.explain(x_row.reshape(1, -1))
        fig = plt.figure(figsize=(8, 6))
        shap.plots.waterfall(explanation[0], max_display=max_display, show=False)
        plt.title(f"Local Explanation - {self.disease_key.title()}")
        plt.tight_layout()
        return fig


def generate_global_explanations(disease_key: str, sample_size: int = 30) -> None:
    """CLI/orchestrator entry point: saves global SHAP plots for one disease."""
    disease = get_disease(disease_key)
    explainer = DiseaseExplainer.from_disease_key(disease_key)

    test_df = pd.read_csv(disease.processed_test_path)
    X_sample = test_df[explainer.feature_order].to_numpy()[:sample_size]

    logger.info("[%s] computing SHAP values for %d test rows...", disease_key, len(X_sample))
    explainer.plot_summary(X_sample)
    explainer.plot_beeswarm(X_sample)
    plt.close("all")
    logger.info("[%s] saved SHAP summary/beeswarm plots to %s", disease_key, FIGURES_DIR)


def main() -> None:
    import argparse

    from src.config import DISEASES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=list(DISEASES), default=None)
    parser.add_argument("--sample-size", type=int, default=30)
    args = parser.parse_args()

    keys = [args.disease] if args.disease else list(DISEASES)
    for key in keys:
        generate_global_explanations(key, args.sample_size)


if __name__ == "__main__":
    main()
