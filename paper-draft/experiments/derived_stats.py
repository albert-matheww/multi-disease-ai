#!/usr/bin/env python3
"""Aggregate numbers quoted in the paper text, computed from results/raw (never typed by hand).

    python paper-draft/experiments/derived_stats.py > paper-draft/results/derived_stats.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_experiments as rx

RAW = Path(__file__).resolve().parents[1] / "results" / "raw"
out: dict = {}

# --- A. preprocessing variants (reference models only: logreg, hgb) --------------------------
def _ref_only(paths):
    return [f for f in paths if not f.name.endswith("_tabpfn.json")]


pp = {"n_comparisons": 0, "n_with_p": 0, "n_p_lt_0.05": 0, "min_p": 1.0, "max_abs_diff": 0.0, "leaky_max_abs_diff": 0.0, "leaky_signed": []}
for f in _ref_only(sorted(RAW.glob("ablate_preproc_*.json"))):
    r = json.loads(f.read_text())
    base = np.array(r["aucs"]["project (median, winsorize, engineered)"])
    for v, a in r["aucs"].items():
        if v.startswith("project"):
            continue
        a = np.array(a)
        t = rx.nadeau_bengio(a - base, r["n_train"], r["n_test"])
        pp["n_comparisons"] += 1
        pp["max_abs_diff"] = max(pp["max_abs_diff"], abs(t["mean_diff"]))
        if v.startswith("LEAKY"):
            pp["leaky_max_abs_diff"] = max(pp["leaky_max_abs_diff"], abs(t["mean_diff"]))
            pp["leaky_signed"].append(round(t["mean_diff"], 4))
        if t["p"] is not None:
            pp["n_with_p"] += 1
            pp["min_p"] = min(pp["min_p"], t["p"])
            pp["n_p_lt_0.05"] += int(t["p"] < 0.05)
out["preprocessing"] = pp
per_variant: dict = {}
for f in _ref_only(sorted(RAW.glob("ablate_preproc_*.json"))):
    r = json.loads(f.read_text())
    base = np.array(r["aucs"]["project (median, winsorize, engineered)"])
    for v, a in r["aucs"].items():
        if v.startswith("project"):
            continue
        t = rx.nadeau_bengio(np.array(a) - base, r["n_train"], r["n_test"])
        e = per_variant.setdefault(v, {"diffs": [], "ps": []})
        e["diffs"].append(t["mean_diff"])
        if t["p"] is not None:
            e["ps"].append(t["p"])
pp["per_variant"] = {v: {"n_pairs": len(e["diffs"]), "min_diff": round(min(e["diffs"]), 4), "max_diff": round(max(e["diffs"]), 4),
                         "min_p": round(min(e["ps"]), 3) if e["ps"] else None} for v, e in per_variant.items()}

# --- B. thresholds: Youden vs each rule, balanced accuracy (reference models only) -------------
th = []
for f in _ref_only(sorted(RAW.glob("ablate_threshold_*.json"))):
    r = json.loads(f.read_text())
    ba = {k: np.array([x["balanced_accuracy"] for x in v]) for k, v in r["rules"].items()}
    for k in ba:
        if k.startswith("Youden"):
            continue
        t = rx.nadeau_bengio(ba["Youden J (project)"] - ba[k], 4.0, 1.0)
        th.append({"disease": r["disease"], "model": r["model"], "vs": k, "diff": t["mean_diff"], "p": t["p"]})
th = pd.DataFrame(th)
out["threshold"] = {
    "youden_vs_fixed_p<0.05": th[(th.vs == "fixed 0.5") & (th.p < 0.05)][["disease", "model", "diff", "p"]].round(4).to_dict("records"),
    "youden_vs_prevalence_min_p": float(th[th.vs == "training prevalence"].p.min()),
    "youden_vs_prevalence_max_abs_diff": float(th[th.vs == "training prevalence"]["diff"].abs().max()),
    "youden_vs_F1opt_p<0.05": th[(th.vs == "F1-optimal") & (th.p < 0.05)][["disease", "model", "diff", "p"]].round(4).to_dict("records"),
}
f1opt_spec = {}
for f in sorted(RAW.glob("ablate_threshold_liver_*.json")):
    r = json.loads(f.read_text())
    f1opt_spec[r["model"]] = float(np.mean([x["specificity"] for x in r["rules"]["F1-optimal"]]))
out["threshold"]["liver_F1optimal_mean_specificity"] = f1opt_spec

# --- C. uncertainty representation (reference models only) ------------------------------------------
cf = []
for f in _ref_only(sorted(RAW.glob("ablate_conformal_*.json"))):
    r = json.loads(f.read_text())
    m = {k: pd.DataFrame(v).mean() for k, v in r["methods"].items()}
    cf.append({"disease": r["disease"], "model": r["model"],
               "band_cov": m["project band (global width)"]["coverage"], "cvplus_cov": m["CV+ (Barber et al.)"]["coverage"],
               "lac_cov": m["label-set conformal (1 - p_true)"]["coverage"],
               "band_eq_lac": bool(np.allclose(*[pd.DataFrame(r["methods"][k])[["coverage", "mean_set_size", "singleton_rate", "both_labels_rate", "empty_rate"]].to_numpy()
                                                 for k in ("project band (global width)", "label-set conformal (1 - p_true)")])),
               "band_singleton": m["project band (global width)"]["singleton_rate"], "band_both": m["project band (global width)"]["both_labels_rate"],
               "band_empty": m["project band (global width)"]["empty_rate"]})
cf = pd.DataFrame(cf)
out["uncertainty"] = {
    "band_coverage_range": [float(cf.band_cov.min()), float(cf.band_cov.max())],
    "cvplus_coverage_range": [float(cf.cvplus_cov.min()), float(cf.cvplus_cov.max())],
    "max_abs_cov_diff_cvplus_minus_band": float((cf.cvplus_cov - cf.band_cov).abs().max()),
    "band_identical_to_label_set_in_all_pairs": bool(cf.band_eq_lac.all()),
    "n_pairs": int(len(cf)),
    "singleton_range": [float(cf.band_singleton.min()), float(cf.band_singleton.max())],
    "both_labels_range": [float(cf.band_both.min()), float(cf.band_both.max())],
    "empty_max": float(cf.band_empty.max()),
}
tuned = json.loads((RAW.parent / "summary.json").read_text()).get("coverage", {})
out["uncertainty"]["tuned_baseline_coverage_note"] = "see tables.md coverage table"

# --- D. novelty ----------------------------------------------------------------------------------
nv = []
for f in sorted(RAW.glob("ablate_novelty_*.json")):
    r = json.loads(f.read_text())
    for n, m in r["detectors"].items():
        nv.append({"disease": r["disease"], "det": n, **m})
nv = pd.DataFrame(nv)
g = nv.groupby("det")
out["novelty"] = {
    "cv_flag_rate_range_by_detector": {k: [round(float(v.cv_flag_rate.min()), 3), round(float(v.cv_flag_rate.max()), 3)] for k, v in g},
    "query_us_range_by_detector": {k: [round(float(v.query_us.min()), 1), round(float(v.query_us.max()), 1)] for k, v in g},
    "shift_auroc_by_detector": {k: {d: round(float(x), 3) for d, x in zip(v.disease, v.shift_auroc)} for k, v in g},
    "shift_auroc_rank_of_project": {d: int((1 + (v.shift_auroc > v[v.det.str.startswith("Mahalanobis + Ledoit")].shift_auroc.iloc[0]).sum())) for d, v in nv.groupby("disease")},
}

# --- E. baselines --------------------------------------------------------------------------------
zoo = []
for f in sorted(RAW.glob("cv_*.json")):
    r = json.loads(f.read_text())
    d = pd.DataFrame(r["folds"])
    zoo.append({"disease": r["disease"], "model": r["model"], "auc": float(d.auc.mean()), "sec": float(d.seconds.mean()), "n": len(d)})
zoo = pd.DataFrame(zoo)
zoo_ref = zoo[zoo.model != "tabpfn"]
out["baselines"] = {
    "best_auc_model_by_disease": {d: v.sort_values("auc").iloc[-1]["model"] for d, v in zoo_ref.groupby("disease")},
    "logreg_minus_hgb_auc": {d: round(float(v[v.model == "logreg"].auc.iloc[0] - v[v.model == "hgb"].auc.iloc[0]), 4) for d, v in zoo.groupby("disease")},
    "seconds_per_fold_range_by_model": {m: [round(float(v.sec.min()), 2), round(float(v.sec.max()), 2)] for m, v in zoo_ref.groupby("model")},
}

# --- F. TabPFN itself --------------------------------------------------------------------------
tp: dict = {}
zoo_t = zoo[zoo.model == "tabpfn"].set_index("disease")
best_ref_auc = zoo_ref.groupby("disease").auc.max()
tp["cv_auc"] = {d: round(float(zoo_t.loc[d, "auc"]), 4) for d in rx.DISEASES}
tp["cv_best_of_all"] = [d for d in rx.DISEASES if zoo_t.loc[d, "auc"] >= zoo[zoo.disease == d].auc.max()]
tp["seconds_per_fold_range"] = [round(float(zoo_t.sec.min()), 2), round(float(zoo_t.sec.max()), 2)]

summ = json.loads((RAW.parent / "summary.json").read_text())
rows = []
for d, ms in summ.get("cv", {}).items():
    for m, v in ms.items():
        if "auc_diff_tabpfn_minus_this" in v:
            t = v["auc_diff_tabpfn_minus_this"]
            rows.append({"disease": d, "vs": m, "diff": round(t["mean_diff"], 4), "p": t.get("p")})
tp["pairwise_vs_baselines_raw"] = rows

# preprocessing ablation for tabpfn
pv_t: dict = {}
for f in sorted(RAW.glob("ablate_preproc_*_tabpfn.json")):
    r = json.loads(f.read_text())
    base = np.array(r["aucs"]["project (median, winsorize, engineered)"])
    for v, a in r["aucs"].items():
        if v.startswith("project"):
            continue
        t = rx.nadeau_bengio(np.array(a) - base, r["n_train"], r["n_test"])
        e = pv_t.setdefault(v, {"diffs": [], "ps": []})
        e["diffs"].append(t["mean_diff"])
        if t["p"] is not None:
            e["ps"].append(t["p"])
tp["preproc_per_variant"] = {v: {"n_pairs": len(e["diffs"]), "min_diff": round(min(e["diffs"]), 4), "max_diff": round(max(e["diffs"]), 4),
                                 "min_p": round(min(e["ps"]), 3) if e["ps"] else None} for v, e in pv_t.items()}
tp["preproc_max_abs_diff"] = round(max(abs(d) for e in pv_t.values() for d in e["diffs"]), 4)
tp["preproc_min_p"] = round(min(p for e in pv_t.values() for p in e["ps"]), 3)

# threshold for tabpfn
th_t = []
for f in sorted(RAW.glob("ablate_threshold_*_tabpfn.json")):
    r = json.loads(f.read_text())
    ba = {k: np.array([x["balanced_accuracy"] for x in v]) for k, v in r["rules"].items()}
    for k in ba:
        if k.startswith("Youden"):
            continue
        t = rx.nadeau_bengio(ba["Youden J (project)"] - ba[k], 4.0, 1.0)
        th_t.append({"disease": r["disease"], "vs": k, "diff": round(t["mean_diff"], 4), "p": t["p"]})
tp["threshold_vs_fixed"] = [x for x in th_t if x["vs"] == "fixed 0.5"]
tp["threshold_vs_F1opt"] = [x for x in th_t if x["vs"] == "F1-optimal"]
tp["threshold_vs_prevalence"] = [x for x in th_t if x["vs"] == "training prevalence"]

# conformal for tabpfn
cf_t = []
for f in sorted(RAW.glob("ablate_conformal_*_tabpfn.json")):
    r = json.loads(f.read_text())
    m = {k: pd.DataFrame(v).mean() for k, v in r["methods"].items()}
    cols = ["coverage", "mean_set_size", "singleton_rate", "both_labels_rate", "empty_rate"]
    cf_t.append({
        "disease": r["disease"], "band_cov": m["project band (global width)"]["coverage"],
        "cvplus_cov": m["CV+ (Barber et al.)"]["coverage"], "singleton": m["project band (global width)"]["singleton_rate"],
        "width": m["project band (global width)"]["mean_width"],
        "band_eq_lac": bool(np.allclose(pd.DataFrame(r["methods"]["project band (global width)"])[cols].to_numpy(),
                                         pd.DataFrame(r["methods"]["label-set conformal (1 - p_true)"])[cols].to_numpy())),
    })
cf_t = pd.DataFrame(cf_t)
tp["band_coverage_range"] = [float(cf_t.band_cov.min()), float(cf_t.band_cov.max())]
tp["band_identical_to_label_set_in_all_pairs"] = bool(cf_t.band_eq_lac.all())
tp["max_abs_cov_diff_cvplus_minus_band"] = float((cf_t.cvplus_cov - cf_t.band_cov).abs().max())
tp["ckd_band_width"] = float(cf_t[cf_t.disease == "ckd"].width.iloc[0])
tp["ckd_band_coverage"] = float(cf_t[cf_t.disease == "ckd"].band_cov.iloc[0])
tp["singleton_range"] = [float(cf_t.singleton.min()), float(cf_t.singleton.max())]

# SHAP
shap_t: dict = {}
for f in sorted(RAW.glob("shap_*.json")):
    r = json.loads(f.read_text())
    shap_t[r["disease"]] = {
        "predict_ms": round(r["predict_proba_ms_per_row"], 1),
        "bg10_median": round(r["seconds_per_explanation"]["10"]["median"], 2),
        "bg25_median": round(r["seconds_per_explanation"]["25"]["median"], 2),
        "spearman_mean": round(r["spearman_10_vs_25"]["mean"], 3), "spearman_min": round(r["spearman_10_vs_25"]["min"], 3),
        "top3_mean": round(r["top3_overlap_10_vs_25"]["mean"], 2), "top3_min": r["top3_overlap_10_vs_25"]["min"],
        "kernel_median": round(r["kernel_bg10"]["median_seconds"], 2), "kernel_spearman": round(r["kernel_bg10"]["spearman_vs_permutation25"], 3),
    }
tp["shap"] = shap_t
tp["shap_speedup_range"] = [round(shap_t[d]["bg25_median"] / shap_t[d]["bg10_median"], 2) for d in rx.DISEASES]
tp["shap_predict_ms_range"] = [round(min(v["predict_ms"] for v in shap_t.values()), 1), round(max(v["predict_ms"] for v in shap_t.values()), 1)]
tp["shap_kernel_slower_than_bg25_everywhere"] = all(shap_t[d]["kernel_median"] > shap_t[d]["bg25_median"] for d in rx.DISEASES)

lat: dict = {}
for f in sorted(RAW.glob("latency_*.json")):
    r = json.loads(f.read_text())
    lat[r["disease"]] = {"single_ms": round(r["single_call_ms"]["mean"], 1), "batched_ms_per_row": round(r["batched_call_ms_per_row"], 1),
                          "speedup": round(r["single_call_ms"]["mean"] / r["batched_call_ms_per_row"], 1)}
tp["latency_single_vs_batched"] = lat
tp["latency_speedup_range"] = [min(v["speedup"] for v in lat.values()), max(v["speedup"] for v in lat.values())] if lat else None
tp["latency_single_ms_range"] = [min(v["single_ms"] for v in lat.values()), max(v["single_ms"] for v in lat.values())] if lat else None

out["tabpfn"] = tp
print(json.dumps(out, indent=2, default=str))
