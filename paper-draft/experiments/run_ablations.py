#!/usr/bin/env python3
"""Design-choice comparisons for the MultiDiseaseAI paper.

Every choice the project made is run head-to-head against named alternatives on the same
folds, so the paper can say what the evidence shows - including where a choice loses.
Results go to `results/raw/ablate_*.json`; `--stage analyze` writes `results/ablation_tables.md`.

    python paper-draft/experiments/run_ablations.py --stage selfcheck
    python paper-draft/experiments/run_ablations.py --stage preproc   --models core      # no token
    python paper-draft/experiments/run_ablations.py --stage threshold --models core
    python paper-draft/experiments/run_ablations.py --stage conformal --models core
    python paper-draft/experiments/run_ablations.py --stage novelty
    python paper-draft/experiments/run_ablations.py --stage preproc   --models tabpfn --repeats 1   # needs token
    python paper-draft/experiments/run_ablations.py --stage analyze

Ablation runs use FIXED, untuned hyperparameters (below) so that only the design choice varies;
the tuned model comparison is in run_experiments.py (`--stage cv`).
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import OneClassSVM

import run_experiments as rx  # noqa: E402  (also puts the project root on sys.path)
from src.config import CONFORMAL_ALPHA, DISEASES, OOD_QUANTILE, RANDOM_STATE  # noqa: E402
from src.feature_engineering import engineer_features  # noqa: E402
from src.uncertainty import OODDetector, conformal_halfwidth, conformal_residuals, probability_band, youden_threshold  # noqa: E402

OUT = rx.HERE.parent / "results"
FIXED = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    "hgb": lambda: HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, max_iter=150, random_state=RANDOM_STATE),
}


def factory_for(name: str, cols: list[str], pre, dry_run: bool):
    if name != "tabpfn":
        return FIXED[name]
    idx = [cols.index(c) for c in pre.disease.categorical_features if c in cols]
    return rx.tabpfn_factory(SimpleNamespace(categorical_feature_indices=idx), dry_run)


# --------------------------------------------------------------------------- #
# A. Preprocessing variants (same steps as the project, one switch at a time)
# --------------------------------------------------------------------------- #
def prep(pre, train_df, test_df, *, impute="median", winsorize=True, engineer=True, fit_on="train"):
    d = pre.disease
    num = [c for c in d.numeric_features if c in train_df.columns]
    cat = [c for c in d.categorical_features if c in train_df.columns]
    tr, te = train_df.copy(), test_df.copy()
    raw_fit = pd.concat([tr, te]) if fit_on == "all" else tr
    if impute != "none":
        if num:
            imp = KNNImputer(n_neighbors=5) if impute == "knn" else SimpleImputer(strategy=impute)
            imp.fit(raw_fit[num])
            tr[num], te[num] = imp.transform(tr[num]), imp.transform(te[num])
        if cat:
            imp_c = SimpleImputer(strategy="most_frequent").fit(raw_fit[cat])
            tr[cat], te[cat] = imp_c.transform(tr[cat]), imp_c.transform(te[cat])
    fit = pd.concat([tr, te]) if fit_on == "all" else tr
    if winsorize:
        for c in num:
            q1, q3 = fit[c].quantile([0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            tr[c], te[c] = tr[c].clip(lo, hi), te[c].clip(lo, hi)
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(fit[cat])
        tr[cat], te[cat] = enc.transform(tr[cat]).astype(float), enc.transform(te[cat]).astype(float)
    if engineer:
        tr, te = engineer_features(d.key, tr), engineer_features(d.key, te)
    cols = [c for c in d.numeric_features + d.categorical_features if c in tr.columns]
    return tr[cols].to_numpy(float), tr["target"].to_numpy(int), te[cols].to_numpy(float), te["target"].to_numpy(int), cols


VARIANTS = {
    "project (median, winsorize, engineered)": {},
    "no winsorization": {"winsorize": False},
    "no engineered features": {"engineer": False},
    "mean imputation": {"impute": "mean"},
    "kNN imputation (k=5)": {"impute": "knn"},
    "native missing values (no imputation)": {"impute": "none"},
    "LEAKY: preprocessing fit on train+test": {"fit_on": "all"},
}
NATIVE_MISSING = "native missing values (no imputation)"


def stage_selfcheck(diseases):
    """`prep` with default switches must reproduce the project's own fit_transform exactly."""
    for key in diseases:
        pre, df = rx.cleaned_frame(key)
        train, test = pre.split(df)
        tr, te, art = pre.fit_transform(train, test)
        Xtr, _, Xte, _, cols = prep(pre, train, test)
        assert cols == art.feature_order, (key, cols, art.feature_order)
        assert np.allclose(Xtr, tr[cols].to_numpy(float), equal_nan=True) and np.allclose(Xte, te[cols].to_numpy(float), equal_nan=True), key
        rx.log(f"selfcheck | {key} | prep() reproduces DiseasePreprocessor.fit_transform exactly")


def stage_preproc(diseases, models, dry_run, repeats):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=RANDOM_STATE)
    for key in diseases:
        pre, df = rx.cleaned_frame(key)
        for name in models:
            variants = {v: kw for v, kw in VARIANTS.items()
                        if not (v == NATIVE_MISSING and (name == "logreg" or (dry_run and name == "tabpfn")))}
            aucs = {v: [] for v in variants}
            sizes = []
            for i, (tr, te) in enumerate(splitter.split(df, df["target"])):
                for v, kw in variants.items():
                    Xtr, ytr, Xte, yte, cols = prep(pre, df.iloc[tr], df.iloc[te], **kw)
                    p = factory_for(name, cols, pre, dry_run)().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                    aucs[v].append(float(roc_auc_score(yte, p)))
                sizes.append((len(tr), len(te)))
                rx.log(f"preproc | {key} | {name} | fold {i + 1}/{5 * repeats}")
            rx.save(OUT, f"ablate_preproc_{key}_{name}", {"disease": key, "model": name, "aucs": aucs,
                    "n_train": float(np.mean([s[0] for s in sizes])), "n_test": float(np.mean([s[1] for s in sizes])),
                    "provenance": rx.provenance()})


# --------------------------------------------------------------------------- #
# B. Decision-threshold rules
# --------------------------------------------------------------------------- #
def f1_optimal_threshold(oof, y):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([f1_score(y, (oof >= t).astype(int), zero_division=0) for t in grid]))])


def rule_metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {"balanced_accuracy": balanced_accuracy_score(y, pred), "sensitivity": recall_score(y, pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else float("nan"), "precision": precision_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0), "threshold": float(t)}


def stage_threshold(diseases, models, dry_run, repeats):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=RANDOM_STATE)
    for key in diseases:
        pre, df = rx.cleaned_frame(key)
        for name in models:
            rules = {"fixed 0.5": [], "Youden J (project)": [], "F1-optimal": [], "training prevalence": []}
            for i, (tr, te) in enumerate(splitter.split(df, df["target"])):
                Xtr, ytr, Xte, yte, cols = prep(pre, df.iloc[tr], df.iloc[te])
                fac = factory_for(name, cols, pre, dry_run)
                oof = rx.oof_probabilities(fac, Xtr, ytr, n_splits=rx.CV_FOLDS, random_state=RANDOM_STATE)
                p = fac().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                thr = {"fixed 0.5": 0.5, "Youden J (project)": youden_threshold(oof, ytr),
                       "F1-optimal": f1_optimal_threshold(oof, ytr), "training prevalence": float(ytr.mean())}
                for r, t in thr.items():
                    rules[r].append(rule_metrics(yte, p, t))
                rx.log(f"threshold | {key} | {name} | fold {i + 1}/{5 * repeats}")
            rx.save(OUT, f"ablate_threshold_{key}_{name}", {"disease": key, "model": name, "rules": rules, "provenance": rx.provenance()})


# --------------------------------------------------------------------------- #
# C. Uncertainty: the project's band vs CV+ vs label-set conformal
# --------------------------------------------------------------------------- #
def inner_models(factory, X, y, n_splits=5, seed=RANDOM_STATE):
    """Same folds and seeds as src.uncertainty.oof_probabilities, but keeps the fold models."""
    y = y.astype(int)
    n_splits = max(2, min(n_splits, int(np.bincount(y).min())))
    oof, fold_of, models = np.zeros(len(y)), np.zeros(len(y), int), []
    for k, (a, b) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)):
        m = factory().fit(X[a], y[a])
        oof[b], fold_of[b] = m.predict_proba(X[b])[:, 1], k
        models.append(m)
    return oof, fold_of, models


def set_stats(y, in0, in1):
    size = in0.astype(int) + in1.astype(int)
    covered = np.where(y == 1, in1, in0)
    return {"coverage": float(covered.mean()), "mean_set_size": float(size.mean()), "singleton_rate": float((size == 1).mean()),
            "both_labels_rate": float((size == 2).mean()), "empty_rate": float((size == 0).mean())}


def stage_conformal(diseases, models, dry_run, repeats):
    a = CONFORMAL_ALPHA
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=RANDOM_STATE)
    for key in diseases:
        pre, df = rx.cleaned_frame(key)
        for name in models:
            res = {"project band (global width)": [], "CV+ (Barber et al.)": [], "label-set conformal (1 - p_true)": []}
            for i, (tr, te) in enumerate(splitter.split(df, df["target"])):
                Xtr, ytr, Xte, yte, cols = prep(pre, df.iloc[tr], df.iloc[te])
                fac = factory_for(name, cols, pre, dry_run)
                oof, fold_of, fold_models = inner_models(fac, Xtr, ytr)
                p = fac().fit(Xtr, ytr).predict_proba(Xte)[:, 1]
                n = len(ytr)

                h = conformal_halfwidth(conformal_residuals(oof, ytr), a)
                res["project band (global width)"].append({**set_stats(yte, p - h <= 0.0, p + h >= 1.0), "mean_width": float(np.mean(np.clip(p + h, 0, 1) - np.clip(p - h, 0, 1)))})

                mu = np.column_stack([m.predict_proba(Xte)[:, 1] for m in fold_models])[:, fold_of]  # (n_test, n)
                r = np.abs(ytr - oof)
                lo_k, hi_k = int(np.floor(a * (n + 1))), int(np.ceil((1 - a) * (n + 1)))
                lower = np.sort(mu - r, axis=1)[:, lo_k - 1] if lo_k >= 1 else np.full(len(p), -np.inf)
                upper = np.sort(mu + r, axis=1)[:, hi_k - 1] if hi_k <= n else np.full(len(p), np.inf)
                res["CV+ (Barber et al.)"].append({**set_stats(yte, (lower <= 0) & (0 <= upper), (lower <= 1) & (1 <= upper)),
                                                   "mean_width": float(np.mean(np.clip(upper, 0, 1) - np.clip(lower, 0, 1)))})

                score = np.where(ytr == 1, 1 - oof, oof)
                q = np.sort(score)[min(max(int(np.ceil((n + 1) * (1 - a))), 1), n) - 1]
                res["label-set conformal (1 - p_true)"].append(set_stats(yte, p <= q, (1 - p) <= q))
                rx.log(f"conformal | {key} | {name} | fold {i + 1}/{5 * repeats}")
            rx.save(OUT, f"ablate_conformal_{key}_{name}", {"disease": key, "model": name, "alpha": a, "methods": res, "provenance": rx.provenance()})


# --------------------------------------------------------------------------- #
# D. Novelty detectors (model-free)
# --------------------------------------------------------------------------- #
DETECTORS = ["Mahalanobis + Ledoit-Wolf (project)", "Mahalanobis, empirical covariance", "Isolation Forest",
             "kNN distance (k=5)", "Local Outlier Factor", "One-class SVM"]


def make_detector(name, X):
    t0 = time.perf_counter()
    if name.startswith("Mahalanobis + "):
        det = OODDetector.fit(X, quantile=OOD_QUANTILE)
        score, train = det.distances, det.distances(X)
    elif name.startswith("Mahalanobis, "):
        mu, prec = X.mean(0), np.linalg.pinv(np.cov(X, rowvar=False))
        score = lambda Z: np.sqrt(np.clip(np.einsum("ij,jk,ik->i", Z - mu, prec, Z - mu), 0, None))  # noqa: E731
        train = score(X)
    else:
        sc = StandardScaler().fit(X)
        Xs = sc.transform(X)
        if name == "Isolation Forest":
            m = IsolationForest(random_state=RANDOM_STATE).fit(Xs)
            score = lambda Z: -m.score_samples(sc.transform(Z))  # noqa: E731
            train = score(X)
        elif name.startswith("kNN"):
            nn = NearestNeighbors(n_neighbors=5).fit(Xs)
            train = nn.kneighbors(Xs, n_neighbors=6)[0][:, 1:].mean(1)
            score = lambda Z: nn.kneighbors(sc.transform(Z), n_neighbors=5)[0].mean(1)  # noqa: E731
        elif name == "Local Outlier Factor":
            m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xs)
            train = -m.negative_outlier_factor_
            score = lambda Z: -m.score_samples(sc.transform(Z))  # noqa: E731
        else:
            m = OneClassSVM(nu=1 - OOD_QUANTILE, gamma="scale").fit(Xs)
            score = lambda Z: -m.decision_function(sc.transform(Z))  # noqa: E731
            train = score(X)
    return score, float(np.quantile(train, OOD_QUANTILE)), time.perf_counter() - t0


def stage_novelty(diseases):
    for key in diseases:
        pre, df = rx.cleaned_frame(key)
        age = df[rx.AGE_COLUMN[key]]
        median = float(age.median())
        young, old = df[age <= median], df[age > median]
        young_tr, young_te = train_test_split(young, test_size=0.2, random_state=RANDOM_STATE, stratify=young["target"])
        Xa, _, Xb, _, _ = rx.encode(pre, young_tr, pd.concat([young_te, old]))
        n_in = len(young_te)
        X_ship_tr, _, X_ship_te, _, _ = rx.shipped_split(key)
        out = {}
        for name in DETECTORS:
            flagged = total = 0
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE).split(df, df["target"]):
                A, _, B, _, _ = rx.encode(pre, df.iloc[tr], df.iloc[te])
                sc, thr, _ = make_detector(name, A)
                flagged += int((sc(B) > thr).sum())
                total += len(B)
            sc, thr, _ = make_detector(name, Xa)
            s_in, s_old = sc(Xb[:n_in]), sc(Xb[n_in:])
            labels = np.r_[np.zeros(len(s_in)), np.ones(len(s_old))]
            sc2, _, fit_s = make_detector(name, X_ship_tr)
            t0 = time.perf_counter()
            for row in X_ship_te[:100]:
                sc2(row.reshape(1, -1))
            out[name] = {"cv_flag_rate": flagged / total, "flag_young": float((s_in > thr).mean()), "flag_old": float((s_old > thr).mean()),
                         "shift_auroc": float(roc_auc_score(labels, np.r_[s_in, s_old])), "fit_ms": 1000 * fit_s,
                         "query_us": 1e6 * (time.perf_counter() - t0) / min(100, len(X_ship_te))}
            rx.log(f"novelty | {key} | {name} | cv={out[name]['cv_flag_rate']:.3f} old={out[name]['flag_old']:.3f} auroc={out[name]['shift_auroc']:.3f}")
        rx.save(OUT, f"ablate_novelty_{key}", {"disease": key, "detectors": out, "n_young_heldout": n_in, "n_old": len(old), "provenance": rx.provenance()})


def stage_overhead():
    """Per-call cost of the band lookup (the project's other add-on cost is in the novelty stage)."""
    resid = np.sort(np.random.default_rng(0).random(600))
    t0 = time.perf_counter()
    for _ in range(20000):
        probability_band(0.62, resid, 0.10)
    rx.save(OUT, "ablate_overhead", {"band_lookup_us": 1e6 * (time.perf_counter() - t0) / 20000, "provenance": rx.provenance()})


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def stage_analyze():
    raw = OUT / "raw"
    L = ["# Design-choice comparison tables (generated by run_ablations.py --stage analyze)", ""]

    def files(prefix):
        return sorted(raw.glob(f"{prefix}_*.json"))

    if files("ablate_preproc"):
        L += ["## A. Preprocessing variants: mean ROC-AUC over folds, and paired difference vs the project's pipeline",
              "(difference = variant minus project; p from the Nadeau-Bengio corrected resampled t-test; fixed hyperparameters)", "",
              "| Disease | Model | Variant | AUC | diff vs project | p |", "|---|---|---|---|---|---|"]
        for f in files("ablate_preproc"):
            r = json.loads(f.read_text())
            base = np.array(r["aucs"]["project (median, winsorize, engineered)"])
            for v, a in r["aucs"].items():
                a = np.array(a)
                if v.startswith("project"):
                    L.append(f"| {r['disease']} | {r['model']} | {v} | {a.mean():.4f} | - | - |")
                else:
                    t = rx.nadeau_bengio(a - base, r["n_train"], r["n_test"])
                    pv = "n/a" if t["p"] is None else f"{t['p']:.3f}"
                    L.append(f"| {r['disease']} | {r['model']} | {v} | {a.mean():.4f} | {t['mean_diff']:+.4f} | {pv} |")
        L.append("")

    if files("ablate_threshold"):
        L += ["## B. Decision-threshold rules: mean over outer folds (threshold chosen on inner out-of-fold predictions)", "",
              "| Disease | Model | Rule | mean threshold | balanced acc | sensitivity | specificity | precision | F1 |", "|---|---|---|---|---|---|---|---|---|"]
        for f in files("ablate_threshold"):
            r = json.loads(f.read_text())
            for rule, folds in r["rules"].items():
                m = pd.DataFrame(folds).mean()
                L.append(f"| {r['disease']} | {r['model']} | {rule} | {m['threshold']:.3f} | {m['balanced_accuracy']:.3f} | {m['sensitivity']:.3f} | "
                         f"{m['specificity']:.3f} | {m['precision']:.3f} | {m['f1']:.3f} |")
        L.append("")

    if files("ablate_threshold"):
        L += ["## B2. Balanced accuracy of the project's Youden rule minus each alternative (paired over outer folds)",
              "(Nadeau-Bengio corrected resampled t-test; outer folds are 5-fold so n_test/n_train = 0.25)", "",
              "| Disease | Model | vs rule | mean diff | p |", "|---|---|---|---|---|"]
        for f in files("ablate_threshold"):
            r = json.loads(f.read_text())
            ref = np.array([x["balanced_accuracy"] for x in r["rules"]["Youden J (project)"]])
            for rule, folds in r["rules"].items():
                if rule.startswith("Youden"):
                    continue
                t = rx.nadeau_bengio(ref - np.array([x["balanced_accuracy"] for x in folds]), 4.0, 1.0)
                pv = "n/a" if t["p"] is None else f"{t['p']:.3f}"
                L.append(f"| {r['disease']} | {r['model']} | {rule} | {t['mean_diff']:+.3f} | {pv} |")
        L.append("")

    if files("ablate_conformal"):
        L += ["## C. Uncertainty representation at alpha = 0.10 (label-set view: a band/interval is read as the set of labels {0,1} it contains)", "",
              "| Disease | Model | Method | coverage | mean set size | singleton rate | both labels | empty | mean band width |", "|---|---|---|---|---|---|---|---|---|"]
        for f in files("ablate_conformal"):
            r = json.loads(f.read_text())
            for meth, folds in r["methods"].items():
                m = pd.DataFrame(folds).mean()
                w = f"{m['mean_width']:.3f}" if "mean_width" in m else "-"
                L.append(f"| {r['disease']} | {r['model']} | {meth} | {m['coverage']:.3f} | {m['mean_set_size']:.2f} | {m['singleton_rate']:.3f} | "
                         f"{m['both_labels_rate']:.3f} | {m['empty_rate']:.3f} | {w} |")
        L.append("")

    if files("ablate_novelty"):
        L += [f"## D. Novelty detectors at a nominal {100 * (1 - OOD_QUANTILE):.1f}% training flag rate",
              "(flag rates: held-out CV pooled; age shift = fit on the younger half, then flag held-out younger vs older rows; AUROC = score separation old vs young; timing on the shipped split)", "",
              "| Disease | Detector | CV flag rate | flag young | flag old | shift AUROC | fit ms | query us |", "|---|---|---|---|---|---|---|---|"]
        for f in files("ablate_novelty"):
            r = json.loads(f.read_text())
            for n, m in r["detectors"].items():
                L.append(f"| {r['disease']} | {n} | {m['cv_flag_rate']:.3f} | {m['flag_young']:.3f} | {m['flag_old']:.3f} | {m['shift_auroc']:.3f} | {m['fit_ms']:.1f} | {m['query_us']:.0f} |")
        L.append("")

    zoo = {}
    for f in sorted(raw.glob("cv_*.json")):
        r = json.loads(f.read_text())
        d = pd.DataFrame(r["folds"])
        zoo.setdefault(r["disease"], {})[r["model"]] = (d["auc"].mean(), d["auc"].std(ddof=1), d["seconds"].mean(), len(d))
    if zoo:
        L += ["## E. Model comparison, repeated 5-fold CV: AUC and wall-clock seconds per fold to a usable predictor",
              "(baselines: includes their inner-CV hyperparameter search; TabPFN: fit + predict only, no tuning; CPU)", "",
              "| Disease | Model | folds | AUC | SD | seconds/fold |", "|---|---|---|---|---|---|"]
        for key in DISEASES:
            for m, (auc, sd, sec, n) in sorted(zoo.get(key, {}).items(), key=lambda kv: -kv[1][0]):
                L.append(f"| {key} | {m} | {n} | {auc:.3f} | {sd:.3f} | {sec:.2f} |")
        L.append("")

    ov = raw / "ablate_overhead.json"
    if ov.exists():
        L += [f"Band lookup cost: {json.loads(ov.read_text())['band_lookup_us']:.2f} microseconds per call.", ""]
    (OUT / "ablation_tables.md").write_text("\n".join(L))
    rx.log(f"wrote {OUT / 'ablation_tables.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["selfcheck", "preproc", "threshold", "conformal", "novelty", "overhead", "analyze"])
    ap.add_argument("--models", default="core", choices=["core", "tabpfn"])
    ap.add_argument("--diseases", nargs="+", default=list(DISEASES), choices=list(DISEASES))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    models = ["logreg", "hgb"] if a.models == "core" else ["tabpfn"]
    if a.models == "tabpfn" and a.stage in ("preproc", "threshold", "conformal"):
        rx.require_token(a.dry_run)
    if a.stage == "selfcheck":
        stage_selfcheck(a.diseases)
    elif a.stage == "preproc":
        stage_preproc(a.diseases, models, a.dry_run, a.repeats)
    elif a.stage == "threshold":
        stage_threshold(a.diseases, models, a.dry_run, a.repeats)
    elif a.stage == "conformal":
        stage_conformal(a.diseases, models, a.dry_run, a.repeats)
    elif a.stage == "novelty":
        stage_novelty(a.diseases)
    elif a.stage == "overhead":
        stage_overhead()
    else:
        stage_analyze()


if __name__ == "__main__":
    main()
