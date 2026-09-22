#!/usr/bin/env python3
"""Worked examples from REAL held-out test rows, scored by the served predictor.

For each disease this picks test rows by a fixed rule (no cherry-picking) and prints what the
app would show: raw inputs, probability, band, learned-threshold label, novelty flag, and the
top SHAP contributors. It needs the trained TabPFN bundle (`models/{key}_tabpfn.joblib`),
i.e. a run with a valid TABPFN_TOKEN; it refuses to run without one.

Selection rule (fixed in advance): in `data/processed/{key}_test.csv`, take
  (1) the first row whose true label is 1,
  (2) the first row whose true label is 0,
  (3) the test row with the largest Mahalanobis novelty score.
Raw values are read back from the cleaned raw data by matching the test row's position in the
project's own stratified split, so what is shown is the patient's real record.

    python paper-draft/experiments/worked_example.py > paper-draft/results/worked_examples.md
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import run_experiments as rx
from src.config import DISEASES, get_disease
from src.prediction import DiseasePredictor


def main() -> None:
    for key in DISEASES:
        disease = get_disease(key)
        if not disease.model_path.exists():
            sys.exit(f"No trained bundle at {disease.model_path}: run the pipeline with a TabPFN token first. Nothing was invented.")
    print("# Worked examples (real held-out test rows; selection rule fixed in advance)\n")
    for key in DISEASES:
        disease = get_disease(key)
        predictor = DiseasePredictor(key)
        pre, cleaned = rx.cleaned_frame(key)
        _, raw_test = pre.split(cleaned)  # same stratified split as the shipped processed files
        test = pd.read_csv(disease.processed_test_path)
        X = test[predictor.feature_order].to_numpy(float)
        novelty = predictor._ood.distances(X) if predictor._ood is not None else np.zeros(len(X))
        picks = {"first positive": int(np.argmax(test["target"].to_numpy() == 1)),
                 "first negative": int(np.argmax(test["target"].to_numpy() == 0)),
                 "most novel": int(np.argmax(novelty))}
        print(f"## {disease.display_name}\n")
        for label, i in picks.items():
            row = raw_test.iloc[i]
            record = {f.name: (row[f.name].item() if hasattr(row[f.name], "item") else row[f.name]) for f in disease.fields}
            result = predictor.predict({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in record.items()}, explain=True, top_n=3)
            print(f"### {label} (test row {i}, true label {int(test['target'].iloc[i])})\n")
            print("Inputs: " + ", ".join(f"{k}={v}" for k, v in record.items()) + "\n")
            print(f"- probability {result.probability:.3f}, band [{result.probability_low:.3f}, {result.probability_high:.3f}], "
                  f"threshold {result.decision_threshold:.3f}, label **{result.predicted_label}**, out-of-distribution: {result.out_of_distribution}")
            print("- top contributors: " + "; ".join(f"{c['feature']} ({c['shap_value']:+.3f})" for c in result.top_contributors) + "\n")


if __name__ == "__main__":
    main()
