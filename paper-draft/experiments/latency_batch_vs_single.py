#!/usr/bin/env python3
"""Single-row vs batched `predict_proba` latency, same fitted model, real test rows.

Fills a gap the main harness leaves: `stage_shap`'s `predict_proba_ms_per_row` times single-row
calls in a loop; nothing in the harness times a batched call on the same already-fitted model for
comparison. This script does, and saves a normal provenance-stamped result file.

    python paper-draft/experiments/latency_batch_vs_single.py
"""
from __future__ import annotations

import time

import numpy as np

import run_experiments as rx
from src.config import DISEASES

OUT = rx.OUT if hasattr(rx, "OUT") else None


def main() -> None:
    import json
    from pathlib import Path

    out_dir = Path(__file__).resolve().parents[1] / "results"
    for key in DISEASES:
        X_tr, y_tr, X_te, y_te, art = rx.shipped_split(key)
        model = rx.tabpfn_factory(art, False)().fit(X_tr, y_tr)
        rows = X_te[: min(20, len(X_te))]

        single = []
        for x in rows:
            t0 = time.perf_counter()
            model.predict_proba(x.reshape(1, -1))
            single.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        model.predict_proba(rows)
        batched_total = time.perf_counter() - t0

        result = {
            "disease": key, "n_rows": len(rows),
            "single_call_ms": {"mean": 1000 * float(np.mean(single)), "median": 1000 * float(np.median(single))},
            "batched_call_ms_per_row": 1000 * batched_total / len(rows),
            "batched_call_total_ms": 1000 * batched_total,
            "provenance": rx.provenance(),
        }
        rx.save(out_dir, f"latency_{key}", result)
        rx.log(f"latency | {key} | single(mean)={result['single_call_ms']['mean']:.1f}ms "
               f"| batched={result['batched_call_ms_per_row']:.1f}ms/row (n={len(rows)})")


if __name__ == "__main__":
    main()
