#!/usr/bin/env python3
"""Experiments for the MultiDiseaseAI paper.

Every number the paper reports is read from a JSON written by this script (see
`results/`). It reuses the project's own preprocessing, out-of-fold, Youden and
novelty code so the results describe the real pipeline, not a re-implementation.
It does not modify anything under `src/`.

Run from the project root with the project's venv:

    python paper-draft/experiments/run_experiments.py --stage main     --models baselines
    python paper-draft/experiments/run_experiments.py --stage novelty
    python paper-draft/experiments/run_experiments.py --stage main     --models tabpfn   # needs TABPFN_TOKEN
    python paper-draft/experiments/run_experiments.py --stage analyze

Stages
  main      fixed 80/20 split the project ships (data/processed): test probabilities per model
  cv        repeated stratified 5-fold CV over the whole cleaned dataset; preprocessing is
            re-fit on each fold's training part with the project's DiseasePreprocessor
  coverage  empirical label-coverage of the 90% band: 5-fold outer CV, inner OOF residuals
  novelty   flag rate of the Mahalanobis detector (held-out rows; an age-shift experiment)
  shap      SHAP timing and rank agreement for background sizes 10 vs 25   (TabPFN only)
  analyze   bootstrap CIs, paired comparisons, tables -> results/summary.json, tables.md

`--dry-run` swaps a LogisticRegression in for TabPFN and writes to `results_dryrun/`.
It exists only to check the plumbing before a real run; its output is never a result.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.artifacts import PreprocessArtifact  # noqa: E402
from src.config import CONFORMAL_ALPHA, CV_FOLDS, DISEASES, OOD_QUANTILE, RANDOM_STATE, get_disease  # noqa: E402
from src.preprocessing import DiseasePreprocessor  # noqa: E402
from src.uncertainty import (  # noqa: E402
    OODDetector,
    conformal_halfwidth,
    conformal_residuals,
    oof_probabilities,
    youden_threshold,
)

AGE_COLUMN = {"heart": "age", "diabetes": "age", "ckd": "age", "liver": "Age"}
BASELINES = ("logreg", "hgb", "rf", "svm")
N_BOOT = 2000


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def provenance() -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except Exception:  # noqa: BLE001 - provenance is best-effort
            return None

    def version(pkg: str) -> str | None:
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "versions": {p: version(p) for p in ("tabpfn", "torch", "scikit-learn", "numpy", "pandas", "scipy", "shap")},
        "random_state": RANDOM_STATE,
    }


# --------------------------------------------------------------------------- #
# Data access (the project's own preprocessing)
# --------------------------------------------------------------------------- #
def cleaned_frame(key: str) -> tuple[DiseasePreprocessor, pd.DataFrame]:
    pre = DiseasePreprocessor(get_disease(key))
    return pre, pre.clean_raw(pre.load_raw())


def shipped_split(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, PreprocessArtifact]:
    disease = get_disease(key)
    art = PreprocessArtifact.load(disease.preprocessor_path)
    train, test = pd.read_csv(disease.processed_train_path), pd.read_csv(disease.processed_test_path)
    cols = art.feature_order
    return (
        train[cols].to_numpy(float),
        train["target"].to_numpy(int),
        test[cols].to_numpy(float),
        test["target"].to_numpy(int),
        art,
    )


def encode(pre: DiseasePreprocessor, train_df: pd.DataFrame, test_df: pd.DataFrame):
    tr, te, art = pre.fit_transform(train_df, test_df)
    cols = art.feature_order
    return tr[cols].to_numpy(float), tr["target"].to_numpy(int), te[cols].to_numpy(float), te["target"].to_numpy(int), art


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def tuned_baseline(name: str) -> GridSearchCV:
    """Baselines get a small inner-CV grid on the training split only (scoring: ROC-AUC)."""
    if name == "logreg":
        return GridSearchCV(
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
            {"logisticregression__C": [0.01, 0.1, 1, 10, 100]},
            scoring="roc_auc",
            cv=_cv(),
        )
    if name == "hgb":
        return GridSearchCV(
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {"learning_rate": [0.03, 0.1], "max_depth": [2, 3, None], "max_iter": [100, 300]},
            scoring="roc_auc",
            cv=_cv(),
        )
    if name == "rf":
        return GridSearchCV(
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
            {"max_depth": [None, 5], "min_samples_leaf": [1, 3]},
            scoring="roc_auc",
            cv=_cv(),
        )
    if name == "svm":
        return GridSearchCV(
            make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            {"svc__C": [0.1, 1, 10]},
            scoring="roc_auc",
            cv=_cv(),
        )
    raise ValueError(name)


def tabpfn_factory(art: PreprocessArtifact, dry_run: bool):
    if dry_run:
        return lambda: LogisticRegression(max_iter=2000)  # plumbing check only
    from src.train import _make_tabpfn  # the project's own constructor (n_estimators=8, cpu)

    return lambda: _make_tabpfn(art, "cpu", 8)


def require_token(dry_run: bool) -> None:
    if dry_run:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.environ.get("TABPFN_TOKEN"):
        sys.exit("TABPFN_TOKEN is not set (put it in the project's .env). Aborting; nothing was run.")


def fit_baseline(name: str, X_tr, y_tr, X_te):
    grid = tuned_baseline(name).fit(X_tr, y_tr)
    best = grid.best_estimator_
    return best.predict_proba(X_te)[:, 1], grid.best_params_, (lambda: clone(best))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def cls_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def nadeau_bengio(diffs: np.ndarray, n_train: float, n_test: float) -> dict:
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for paired per-fold differences."""
    k = len(diffs)
    mean, var = float(np.mean(diffs)), float(np.var(diffs, ddof=1))
    if var == 0 or k < 2:
        return {"mean_diff": mean, "t": None, "p": None, "k": k}
    t = mean / np.sqrt((1 / k + n_test / n_train) * var)
    return {"mean_diff": mean, "t": float(t), "p": float(2 * stats.t.sf(abs(t), k - 1)), "k": k}


# --------------------------------------------------------------------------- #
# Stage: main (shipped 80/20 split)
# --------------------------------------------------------------------------- #
def stage_main(diseases, models, dry_run, out):
    for key in diseases:
        X_tr, y_tr, X_te, y_te, art = shipped_split(key)
        for name in models:
            log(f"main | {key} | {name}")
            record = {"disease": key, "model": name, "n_train": len(y_tr), "n_test": len(y_te), "y_test": y_te.tolist()}
            if name in BASELINES:
                p, params, factory = fit_baseline(name, X_tr, y_tr, X_te)
                oof = oof_probabilities(factory, X_tr, y_tr, n_splits=CV_FOLDS, random_state=RANDOM_STATE)
                record.update(p_test=p.tolist(), best_params=params)
            else:
                factory = tabpfn_factory(art, dry_run)
                if dry_run:
                    model = factory().fit(X_tr, y_tr)
                    oof = oof_probabilities(factory, X_tr, y_tr, n_splits=CV_FOLDS, random_state=RANDOM_STATE)
                    record["bundle_decision_threshold"] = youden_threshold(oof, y_tr)
                else:  # the project's own training path -> the bundle that would be served
                    import joblib

                    from src.train import train_disease

                    train_disease(key)
                    bundle = joblib.load(get_disease(key).model_path)
                    model, oof = bundle["model"], np.asarray(bundle["oof_probabilities"])
                    record["bundle_decision_threshold"] = float(bundle["decision_threshold"])
                    record["bundle_train_time_seconds"] = bundle.get("train_time_seconds")
                record["p_test"] = model.predict_proba(X_te)[:, 1].tolist()
            record["oof_train"] = oof.tolist()
            record["threshold_youden"] = youden_threshold(oof, y_tr)
            record["provenance"] = provenance()
            save(out, f"main_{key}_{name}", record)


# --------------------------------------------------------------------------- #
# Stage: cv (repeated stratified 5-fold over the cleaned data)
# --------------------------------------------------------------------------- #
def stage_cv(diseases, models, dry_run, out, repeats):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=RANDOM_STATE)
    for key in diseases:
        pre, df = cleaned_frame(key)
        for name in models:
            folds = []
            for i, (tr, te) in enumerate(splitter.split(df, df["target"])):
                X_tr, y_tr, X_te, y_te, art = encode(pre, df.iloc[tr], df.iloc[te])
                t0 = time.perf_counter()
                if name in BASELINES:
                    p, _, _ = fit_baseline(name, X_tr, y_tr, X_te)
                else:
                    p = tabpfn_factory(art, dry_run)().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
                folds.append(
                    {"fold": i, "n_train": len(y_tr), "n_test": len(y_te), "auc": roc_auc_score(y_te, p),
                     **cls_metrics(y_te, p, 0.5), "seconds": time.perf_counter() - t0}
                )
                log(f"cv | {key} | {name} | fold {i + 1}/{5 * repeats} | auc={folds[-1]['auc']:.3f}")
            save(out, f"cv_{key}_{name}", {"disease": key, "model": name, "repeats": repeats, "folds": folds,
                                            "provenance": provenance()})


# --------------------------------------------------------------------------- #
# Stage: coverage (does the 90% band cover the label?)
# --------------------------------------------------------------------------- #
def stage_coverage(diseases, models, dry_run, out):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for key in diseases:
        pre, df = cleaned_frame(key)
        for name in models:
            folds = []
            for i, (tr, te) in enumerate(outer.split(df, df["target"])):
                X_tr, y_tr, X_te, y_te, art = encode(pre, df.iloc[tr], df.iloc[te])
                if name in BASELINES:
                    p, _, factory = fit_baseline(name, X_tr, y_tr, X_te)
                else:
                    factory = tabpfn_factory(art, dry_run)
                    p = factory().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
                oof = oof_probabilities(factory, X_tr, y_tr, n_splits=CV_FOLDS, random_state=RANDOM_STATE)
                h = conformal_halfwidth(conformal_residuals(oof, y_tr), CONFORMAL_ALPHA)
                lo, hi = np.clip(p - h, 0, 1), np.clip(p + h, 0, 1)
                covered = (y_te >= lo) & (y_te <= hi)  # is the binary label inside [lo, hi]?
                folds.append({"fold": i, "h": h, "n_test": len(y_te), "covered": int(covered.sum()),
                              "band_contains_half": int(((lo <= 0.5) & (0.5 <= hi)).sum()),
                              "mean_width": float(np.mean(hi - lo))})
                log(f"coverage | {key} | {name} | fold {i + 1}/5 | h={h:.3f} cov={covered.mean():.3f}")
            save(out, f"coverage_{key}_{name}", {"disease": key, "model": name, "alpha": CONFORMAL_ALPHA,
                                                  "folds": folds, "provenance": provenance()})


# --------------------------------------------------------------------------- #
# Stage: novelty (model-free)
# --------------------------------------------------------------------------- #
def flag_rate(det: OODDetector, X: np.ndarray) -> float:
    return float(np.mean(det.distances(X) > det.threshold))


def stage_novelty(diseases, out):
    for key in diseases:
        pre, df = cleaned_frame(key)
        result = {"disease": key, "quantile": OOD_QUANTILE, "nominal_flag_rate": round(1 - OOD_QUANTILE, 4)}

        X_tr, _, X_te, _, _ = shipped_split(key)
        det = OODDetector.fit(X_tr, quantile=OOD_QUANTILE)
        result["shipped_split"] = {"train_flag_rate": flag_rate(det, X_tr), "test_flag_rate": flag_rate(det, X_te),
                                   "n_test": len(X_te), "distance_threshold": det.threshold}

        flagged = total = 0
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(df, df["target"]):
            Xa, _, Xb, _, _ = encode(pre, df.iloc[tr], df.iloc[te])
            d = OODDetector.fit(Xa, quantile=OOD_QUANTILE)
            flagged += int((d.distances(Xb) > d.threshold).sum())
            total += len(Xb)
        result["cv_pooled"] = {"flag_rate": flagged / total, "n": total}

        # Age shift: fit on the younger half (80/20), compare held-out younger vs older rows.
        age = df[AGE_COLUMN[key]]
        median = float(age.median())
        young, old = df[age <= median], df[age > median]
        young_tr, young_te = train_test_split(young, test_size=0.2, random_state=RANDOM_STATE, stratify=young["target"])
        Xa, _, Xb, _, _ = encode(pre, young_tr, pd.concat([young_te, old]))
        d = OODDetector.fit(Xa, quantile=OOD_QUANTILE)
        n_in = len(young_te)
        result["age_shift"] = {
            "median_age": median, "n_fit": len(young_tr), "n_heldout_young": n_in, "n_old": len(old),
            "flag_rate_heldout_young": flag_rate(d, Xb[:n_in]), "flag_rate_old": flag_rate(d, Xb[n_in:]),
            "n_rows_without_age": int(age.isna().sum()),
        }
        result["provenance"] = provenance()
        log(f"novelty | {key} | test={result['shipped_split']['test_flag_rate']:.3f} "
            f"cv={result['cv_pooled']['flag_rate']:.3f} young={result['age_shift']['flag_rate_heldout_young']:.3f} "
            f"old={result['age_shift']['flag_rate_old']:.3f}")
        save(out, f"novelty_{key}", result)


# --------------------------------------------------------------------------- #
# Stage: shap (TabPFN only)
# --------------------------------------------------------------------------- #
def stage_shap(diseases, dry_run, out, n_rows=20):
    from src.explainability import DiseaseExplainer

    for key in diseases:
        X_tr, y_tr, X_te, _, art = shipped_split(key)
        if dry_run:
            model = tabpfn_factory(art, True)().fit(X_tr, y_tr)
        else:
            import joblib

            model = joblib.load(get_disease(key).model_path)["model"]
        rows = X_te[:n_rows]
        t0 = time.perf_counter()
        for x in rows:
            model.predict_proba(x.reshape(1, -1))
        predict_ms = 1000 * (time.perf_counter() - t0) / len(rows)

        per_bg: dict[int, dict] = {}
        for bg in (10, 25):
            expl = DiseaseExplainer(key, model, art.feature_order, X_tr, max_background=bg)
            times, vecs = [], []
            for x in rows:
                np.random.seed(0)
                t0 = time.perf_counter()
                r = expl.explain_instance(x, top_n=len(art.feature_order))
                times.append(time.perf_counter() - t0)
                by_name = {c["feature"]: abs(c["shap_value"]) for c in r["all_contributions"]}
                vecs.append(np.array([by_name[f] for f in art.feature_order]))
            per_bg[bg] = {"times": times, "vecs": np.array(vecs)}
        import shap

        kernel_times, kernel_vecs = [], []
        kx = shap.KernelExplainer(lambda z: model.predict_proba(np.asarray(z))[:, 1], shap.kmeans(X_tr, 10).data)
        for x in rows:
            np.random.seed(0)
            t0 = time.perf_counter()
            vals = kx.shap_values(x.reshape(1, -1), nsamples="auto", silent=True)
            kernel_times.append(time.perf_counter() - t0)
            kernel_vecs.append(np.abs(np.asarray(vals).reshape(-1)))
        rho_kernel = [stats.spearmanr(a, b).statistic for a, b in zip(per_bg[25]["vecs"], kernel_vecs)]
        rho = [stats.spearmanr(a, b).statistic for a, b in zip(per_bg[10]["vecs"], per_bg[25]["vecs"])]
        top3 = [len(set(np.argsort(-a)[:3]) & set(np.argsort(-b)[:3])) for a, b in zip(per_bg[10]["vecs"], per_bg[25]["vecs"])]
        summary = {
            "disease": key, "n_rows": len(rows), "predict_proba_ms_per_row": predict_ms,
            "seconds_per_explanation": {str(bg): {"median": float(np.median(v["times"])), "mean": float(np.mean(v["times"])),
                                                  "max": float(np.max(v["times"]))} for bg, v in per_bg.items()},
            "spearman_10_vs_25": {"mean": float(np.nanmean(rho)), "min": float(np.nanmin(rho))},
            "kernel_bg10": {"median_seconds": float(np.median(kernel_times)), "spearman_vs_permutation25": float(np.nanmean(rho_kernel))},
            "top3_overlap_10_vs_25": {"mean": float(np.mean(top3)), "min": int(np.min(top3))},
            "provenance": provenance(),
        }
        log(f"shap | {key} | bg10 {summary['seconds_per_explanation']['10']['median']:.2f}s "
            f"bg25 {summary['seconds_per_explanation']['25']['median']:.2f}s rho={summary['spearman_10_vs_25']['mean']:.2f}")
        save(out, f"shap_{key}", summary)


# --------------------------------------------------------------------------- #
# Stage: analyze
# --------------------------------------------------------------------------- #
def load_all(out: Path, prefix: str) -> dict[tuple[str, str], dict]:
    found = {}
    for path in sorted((out / "raw").glob(f"{prefix}_*.json")):
        rec = json.loads(path.read_text())
        found[(rec["disease"], rec.get("model", ""))] = rec
    return found


def bootstrap_main(rec_by_model: dict[str, dict], seed: int = 0) -> dict:
    models = list(rec_by_model)
    y = np.array(next(iter(rec_by_model.values()))["y_test"])
    probs = {m: np.array(r["p_test"]) for m, r in rec_by_model.items()}
    thr = {m: r["threshold_youden"] for m, r in rec_by_model.items()}
    rng = np.random.default_rng(seed)
    draws = {m: {k: [] for k in ("auc", "acc05", "f105", "acc_y", "f1_y", "rec_y", "prec_y")} for m in models}
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        for m in models:
            yb, pb = y[idx], probs[m][idx]
            a, b = cls_metrics(yb, pb, 0.5), cls_metrics(yb, pb, thr[m])
            draws[m]["auc"].append(roc_auc_score(yb, pb))
            draws[m]["acc05"].append(a["accuracy"]); draws[m]["f105"].append(a["f1"])
            draws[m]["acc_y"].append(b["accuracy"]); draws[m]["f1_y"].append(b["f1"])
            draws[m]["rec_y"].append(b["recall"]); draws[m]["prec_y"].append(b["precision"])

    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    out = {}
    for m in models:
        p = probs[m]
        out[m] = {
            "n_test": len(y), "threshold_youden": thr[m],
            "auc": roc_auc_score(y, p), "auc_ci": ci(draws[m]["auc"]),
            "at_0.5": {**cls_metrics(y, p, 0.5), "f1_ci": ci(draws[m]["f105"])},
            "at_youden": {**cls_metrics(y, p, thr[m]), "f1_ci": ci(draws[m]["f1_y"])},
        }
    if "tabpfn" in models:
        for m in models:
            if m != "tabpfn":
                d = np.array(draws["tabpfn"]["auc"]) - np.array(draws[m]["auc"])
                out[m]["auc_diff_tabpfn_minus_this"] = {"mean": float(d.mean()), "ci": ci(d),
                                                         "ci_excludes_zero": bool(ci(d)[0] > 0 or ci(d)[1] < 0)}
    return out


def stage_analyze(out: Path):
    summary: dict = {"main": {}, "cv": {}, "coverage": {}, "novelty": {}, "shap": {}}
    lines = ["# Results tables (generated by run_experiments.py --stage analyze)", ""]

    main = load_all(out, "main")
    for key in DISEASES:
        recs = {m: r for (d, m), r in main.items() if d == key}
        if recs:
            summary["main"][key] = bootstrap_main(recs)
    if summary["main"]:
        lines += [f"## Fixed shipped split (bootstrap 95% CI, {N_BOOT} resamples)", "",
                  "| Disease | Model | n_test | AUC [CI] | Acc@0.5 | F1@0.5 [CI] | Youden thr | Acc@Y | Rec@Y | Prec@Y | F1@Y [CI] |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, models in summary["main"].items():
            for m, r in models.items():
                a, b = r["at_0.5"], r["at_youden"]
                lines.append(f"| {key} | {m} | {r['n_test']} | {r['auc']:.3f} [{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}] | "
                             f"{a['accuracy']:.3f} | {a['f1']:.3f} [{a['f1_ci'][0]:.3f}, {a['f1_ci'][1]:.3f}] | "
                             f"{r['threshold_youden']:.3f} | {b['accuracy']:.3f} | {b['recall']:.3f} | {b['precision']:.3f} | "
                             f"{b['f1']:.3f} [{b['f1_ci'][0]:.3f}, {b['f1_ci'][1]:.3f}] |")
        lines.append("")

    cv = load_all(out, "cv")
    if cv:
        lines += ["## Repeated stratified 5-fold CV (mean +/- SD over folds; folds overlap so SD is optimistic)", "",
                  "| Disease | Model | folds | AUC | Acc@0.5 | Prec@0.5 | Rec@0.5 | F1@0.5 |", "|---|---|---|---|---|---|---|---|"]
        for key in DISEASES:
            per_model = {m: r for (d, m), r in cv.items() if d == key}
            for m, r in per_model.items():
                f = pd.DataFrame(r["folds"])
                summary["cv"].setdefault(key, {})[m] = {c: [float(f[c].mean()), float(f[c].std(ddof=1))]
                                                          for c in ("auc", "accuracy", "precision", "recall", "f1")}
                s = summary["cv"][key][m]
                lines.append(f"| {key} | {m} | {len(f)} | " + " | ".join(f"{s[c][0]:.3f} +/- {s[c][1]:.3f}"
                             for c in ("auc", "accuracy", "precision", "recall", "f1")) + " |")
            if "tabpfn" in per_model:
                base = pd.DataFrame(per_model["tabpfn"]["folds"])
                for m, r in per_model.items():
                    if m == "tabpfn":
                        continue
                    other = pd.DataFrame(r["folds"])
                    test = nadeau_bengio((base["auc"] - other["auc"]).to_numpy(), base["n_train"].mean(), base["n_test"].mean())
                    summary["cv"][key][m]["auc_diff_tabpfn_minus_this"] = test
        lines.append("")
        rows = [(k, m, v["auc_diff_tabpfn_minus_this"]) for k, ms in summary["cv"].items() for m, v in ms.items()
                if "auc_diff_tabpfn_minus_this" in v]
        if rows:
            lines += ["### AUC difference, TabPFN minus baseline (Nadeau-Bengio corrected resampled t-test, per-fold pairs)", "",
                      "| Disease | Baseline | mean diff | t | p | folds |", "|---|---|---|---|---|---|"]
            for k, m, t in rows:
                tt = "n/a" if t["t"] is None else f"{t['t']:.2f}"
                pp = "n/a" if t["p"] is None else f"{t['p']:.3f}"
                lines.append(f"| {k} | {m} | {t['mean_diff']:+.4f} | {tt} | {pp} | {t['k']} |")
            lines.append("")

    cov = load_all(out, "coverage")
    if cov:
        lines += ["## Empirical label-coverage of the nominal 90% band (5-fold outer CV, inner OOF residuals)", "",
                  "| Disease | Model | n | coverage [Wilson 95%] | mean h | mean band width |", "|---|---|---|---|---|---|"]
        for (key, m), r in cov.items():
            n = sum(f["n_test"] for f in r["folds"]); k = sum(f["covered"] for f in r["folds"])
            lo, hi = wilson(k, n)
            summary["coverage"].setdefault(key, {})[m] = {"n": n, "coverage": k / n, "wilson95": [lo, hi],
                                                           "mean_h": float(np.mean([f["h"] for f in r["folds"]])),
                                                           "mean_width": float(np.mean([f["mean_width"] for f in r["folds"]]))}
            s = summary["coverage"][key][m]
            lines.append(f"| {key} | {m} | {n} | {s['coverage']:.3f} [{lo:.3f}, {hi:.3f}] | {s['mean_h']:.3f} | "
                         f"{s['mean_width']:.3f} |")
        lines.append("")

    for path in sorted((out / "raw").glob("novelty_*.json")):
        rec = json.loads(path.read_text())
        summary["novelty"][rec["disease"]] = rec
    if summary["novelty"]:
        lines += ["## Novelty flag (Mahalanobis, 97.5% training-distance quantile; nominal flag rate 2.5%)", "",
                  "| Disease | train | shipped test | CV pooled | age <= median, held-out | age > median |", "|---|---|---|---|---|---|"]
        for key, r in summary["novelty"].items():
            lines.append(f"| {key} | {r['shipped_split']['train_flag_rate']:.3f} | {r['shipped_split']['test_flag_rate']:.3f} "
                         f"| {r['cv_pooled']['flag_rate']:.3f} | {r['age_shift']['flag_rate_heldout_young']:.3f} "
                         f"(n={r['age_shift']['n_heldout_young']}) | {r['age_shift']['flag_rate_old']:.3f} (n={r['age_shift']['n_old']}) |")
        lines.append("")

    for path in sorted((out / "raw").glob("shap_*.json")):
        rec = json.loads(path.read_text())
        summary["shap"][rec["disease"]] = rec
    if summary["shap"]:
        lines += ["## SHAP: background 10 vs 25 (first 20 test rows; CPU)", "",
                  "| Disease | predict_proba ms/row | s/explanation bg=10 (median) | bg=25 (median) | Spearman 10 vs 25 (mean / min) | top-3 overlap (mean / min) |",
                  "|---|---|---|---|---|---|"]
        for key, r in summary["shap"].items():
            lines.append(f"| {key} | {r['predict_proba_ms_per_row']:.1f} | {r['seconds_per_explanation']['10']['median']:.2f} | "
                         f"{r['seconds_per_explanation']['25']['median']:.2f} | {r['spearman_10_vs_25']['mean']:.2f} / "
                         f"{r['spearman_10_vs_25']['min']:.2f} | {r['top3_overlap_10_vs_25']['mean']:.2f} / {r['top3_overlap_10_vs_25']['min']} |")
        lines.append("")

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "tables.md").write_text("\n".join(lines))
    log(f"wrote {out / 'summary.json'} and {out / 'tables.md'}")


def save(out: Path, name: str, obj: dict) -> None:
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "raw" / f"{name}.json").write_text(json.dumps(obj, default=float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=["main", "cv", "coverage", "novelty", "shap", "analyze"])
    parser.add_argument("--models", default="all", choices=["baselines", "core", "extra", "tabpfn", "all"],
                        help="core = logreg+hgb; extra = rf+svm; baselines = all four; all = TabPFN + baselines")
    parser.add_argument("--diseases", nargs="+", default=list(DISEASES), choices=list(DISEASES))
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="LogisticRegression in place of TabPFN; plumbing check only")
    args = parser.parse_args()

    out = HERE.parent / ("results_dryrun" if args.dry_run else "results")
    models = {"baselines": list(BASELINES), "core": ["logreg", "hgb"], "extra": ["rf", "svm"],
              "tabpfn": ["tabpfn"], "all": ["tabpfn", *BASELINES]}[args.models]
    if args.stage in ("main", "cv", "coverage", "shap") and "tabpfn" in models or args.stage == "shap":
        require_token(args.dry_run)

    if args.stage == "main":
        stage_main(args.diseases, models, args.dry_run, out)
    elif args.stage == "cv":
        stage_cv(args.diseases, models, args.dry_run, out, args.cv_repeats)
    elif args.stage == "coverage":
        stage_coverage(args.diseases, models, args.dry_run, out)
    elif args.stage == "novelty":
        stage_novelty(args.diseases, out)
    elif args.stage == "shap":
        stage_shap(args.diseases, args.dry_run, out)
    else:
        stage_analyze(out)


if __name__ == "__main__":
    main()
