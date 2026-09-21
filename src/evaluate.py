"""Evaluate a trained disease model: metrics, confusion matrix, ROC & PR curves.

Usage:
    python -m src.evaluate                 # evaluates all four diseases
    python -m src.evaluate --disease heart
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import DISEASES, FIGURES_DIR, get_disease

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150


def _load_model_bundle(disease_key: str) -> dict:
    disease = get_disease(disease_key)
    if not disease.model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {disease.model_path}. Run `python -m src.train` first."
        )
    return joblib.load(disease.model_path)


def plot_confusion_matrix(y_true, y_pred, disease_key: str, labels: list[str]) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Confusion Matrix - {disease_key.title()}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{disease_key}_confusion_matrix.png", bbox_inches="tight")
    return fig


def plot_roc_curve(y_true, y_proba, disease_key: str) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"TabPFN (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {disease_key.title()}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{disease_key}_roc_curve.png", bbox_inches="tight")
    return fig


def plot_precision_recall_curve(y_true, y_proba, disease_key: str) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    baseline = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision, linewidth=2, label="TabPFN")
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Baseline (prevalence={baseline:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {disease_key.title()}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{disease_key}_pr_curve.png", bbox_inches="tight")
    return fig


def evaluate_disease(disease_key: str) -> dict:
    disease = get_disease(disease_key)
    bundle = _load_model_bundle(disease_key)
    model = bundle["model"]
    feature_order = bundle["feature_order"]

    test_df = pd.read_csv(disease.processed_test_path)
    X_test = test_df[feature_order].to_numpy()
    y_test = test_df["target"].to_numpy()

    y_pred = model.predict(X_test)  # default rule (argmax, i.e. a 0.5 cutoff on P(y=1))
    y_proba = model.predict_proba(X_test)[:, 1]
    # The app labels patients with the learned Youden threshold stored in the bundle, so
    # score that rule too; older bundles without one fall back to 0.5.
    threshold = float(bundle.get("decision_threshold", 0.5) or 0.5)
    y_pred_served = (y_proba >= threshold).astype(int)

    metrics = {
        "disease": disease_key,
        "n_test": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "decision_threshold": threshold,
        "at_decision_threshold": {
            "accuracy": float(accuracy_score(y_test, y_pred_served)),
            "precision": float(precision_score(y_test, y_pred_served, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_served, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred_served, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred_served).tolist(),
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=[disease.negative_label, disease.positive_label], output_dict=True
        ),
    }

    plot_confusion_matrix(y_test, y_pred, disease_key, [disease.negative_label, disease.positive_label])
    plot_roc_curve(y_test, y_proba, disease_key)
    plot_precision_recall_curve(y_test, y_proba, disease_key)
    plt.close("all")

    with open(disease.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "[%s] acc=%.3f prec=%.3f rec=%.3f f1=%.3f auc=%.3f",
        disease_key,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics


def evaluate_all() -> pd.DataFrame:
    rows = []
    for key in DISEASES:
        m = evaluate_disease(key)
        rows.append(
            {
                "disease": key,
                "n_test": m["n_test"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"],
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = FIGURES_DIR.parent / "model_comparison.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved model comparison summary to %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease", choices=list(DISEASES), default=None)
    args = parser.parse_args()

    if args.disease:
        evaluate_disease(args.disease)
    else:
        print(evaluate_all().to_string(index=False))


if __name__ == "__main__":
    main()
