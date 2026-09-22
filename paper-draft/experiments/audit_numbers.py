#!/usr/bin/env python3
"""Audit 1 helper: recompute every results-table cell of paper.md from results/raw and diff it.

    python paper-draft/experiments/audit_numbers.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
D = json.loads((ROOT / "results" / "derived_stats.json").read_text())
md = (ROOT / "paper.md").read_text()
DIS = ["heart", "diabetes", "ckd", "liver"]
bad, checked = [], 0


def table(label: str) -> list[list[str]]:
    m = re.search(r"Table:[^\n]*\{#" + label + r"\}\n\n((?:\|.*\n?)+)", md)
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in m.group(1).strip().splitlines()]
    return [r for r in rows if not re.match(r"^[-\s:]+$", r[0])][1:]


def unbold(s: str) -> str:
    return s.replace("**", "")


def chk(name, got, want):
    global checked
    checked += 1
    if got != want:
        bad.append(f"{name}: paper has {got!r}, artifacts give {want!r}")


f3 = lambda x: f"{x:.3f}"  # noqa: E731
nz = lambda x: f"{x:.3f}".lstrip("0")  # noqa: E731

# ---- shared: cv fold data (all models incl. tabpfn) -------------------------------------------
cv = {}
for f in RAW.glob("cv_*.json"):
    r = json.loads(f.read_text())
    d = pd.DataFrame(r["folds"])
    cv[(r["disease"], r["model"])] = (d.auc.mean(), d.seconds.mean(), len(d))

# Table II (tab:models) --------------------------------------------------------------------------
names = {"TabPFN": "tabpfn", "Logistic regression": "logreg", "Random forest": "rf", "Gradient boosting": "hgb", "RBF SVM": "svm"}
for row in table("tab:models"):
    if row[0] not in names:
        continue
    m = names[row[0]]
    for d, cell in zip(DIS, row[1:5]):
        chk(f"TabII {m} {d}", unbold(cell), f3(cv[(d, m)][0]))
        chk(f"TabII {m} {d} folds", cv[(d, m)][2], 15)
    secs = [cv[(d, m)][1] for d in DIS]
    lo, hi = min(secs), max(secs)
    if m == "tabpfn":
        want = f"{lo:.1f}–{hi:.1f}"
    else:
        want = f"{lo:.2f}" if f"{lo:.2f}" == f"{hi:.2f}" else (f"{lo:.1f}–{hi:.1f}" if m in ("rf", "hgb") else f"{lo:.2f}–{hi:.2f}")
    chk(f"TabII {m} seconds", row[5], want)

# Table III (tab:tabpfn-vs-ref) -------------------------------------------------------------------
pw = {(r["disease"], r["vs"]): r for r in D["tabpfn"]["pairwise_vs_baselines_raw"]}
cols = {"vs HGB": "hgb", "vs LogReg": "logreg", "vs RF": "rf", "vs SVM": "svm"}
disease_row = {"Heart": "heart", "Diabetes": "diabetes", "Kidney": "ckd", "Liver": "liver"}
for row in table("tab:tabpfn-vs-ref"):
    d = disease_row[row[0]]
    for colname, m in zip(table("tab:tabpfn-vs-ref")[0] if False else ["vs HGB", "vs LogReg", "vs RF", "vs SVM"], cols.values()):
        pass
hdr = re.search(r"Table:[^\n]*\{#tab:tabpfn-vs-ref\}\n\n(\|.*\n)", md).group(1)
colnames = [c.strip() for c in hdr.strip().strip("|").split("|")][1:]
for row in table("tab:tabpfn-vs-ref"):
    d = disease_row[row[0]]
    for colname, cell in zip(colnames, row[1:]):
        m = cols[colname]
        r = pw.get((d, m))
        cell_clean = unbold(cell)
        if r is None or r["p"] is None:
            want = "+0.000 (n/a)"
        else:
            p_str = "p<.001" if r["p"] < 0.0005 else f"p={r['p']:.3f}".replace("p=0.", "p=.")
            want = f"{r['diff']:+.3f} (mode)".replace("mode", p_str).replace("+-", "−").replace("-", "−" if r["diff"] < 0 else "-")
            sign = "+" if r["diff"] >= 0 else "−"
            want = f"{sign}{abs(r['diff']):.3f} ({p_str})"
        chk(f"TabIII {d} vs {m}", cell_clean, want)

# Table IV (tab:preproc) --------------------------------------------------------------------------
def _ref_only(paths):
    return [f for f in paths if not f.name.endswith("_tabpfn.json")]


pv = D["preprocessing"]["per_variant"]
pv_t = D["tabpfn"]["preproc_per_variant"]
key = {"No winsorization": "no winsorization", "No engineered features": "no engineered features", "Mean imputation": "mean imputation",
       "kNN imputation (k = 5)": "kNN imputation (k=5)", "Native missing values": "native missing values (no imputation)",
       "Leaky: fit on train + test": "LEAKY: preprocessing fit on train+test"}
sg = lambda x: f"{x:+.4f}".replace("-", "−")  # noqa: E731
for row in table("tab:preproc"):
    e, et = pv[key[row[0]]], pv_t[key[row[0]]]
    chk(f"TabIV {row[0]} ref pairs", int(row[1]), e["n_pairs"])
    chk(f"TabIV {row[0]} ref range", row[2], f"{sg(e['min_diff'])} to {sg(e['max_diff'])}")
    chk(f"TabIV {row[0]} ref p", row[3], f"{e['min_p']:.3f}")
    chk(f"TabIV {row[0]} tabpfn pairs", int(row[4]), et["n_pairs"])
    chk(f"TabIV {row[0]} tabpfn range", row[5], f"{sg(et['min_diff'])} to {sg(et['max_diff'])}")
    chk(f"TabIV {row[0]} tabpfn p", row[6], f"{et['min_p']:.3f}")

# Table V (tab:threshold) --------------------------------------------------------------------------
ba = {}
for f in RAW.glob("ablate_threshold_*.json"):
    r = json.loads(f.read_text())
    for rule, folds in r["rules"].items():
        ba[(rule, r["disease"], r["model"])] = np.mean([x["balanced_accuracy"] for x in folds])
rules = {"Fixed 0.5": "fixed 0.5", "Youden J (project)": "Youden J (project)", "F1-optimal": "F1-optimal", "Training prevalence": "training prevalence"}
for row in table("tab:threshold"):
    for d, cell in zip(DIS, row[1:5]):
        want = f"{nz(ba[(rules[row[0]], d, 'logreg')])} / {nz(ba[(rules[row[0]], d, 'hgb')])} / {nz(ba[(rules[row[0]], d, 'tabpfn')])}"
        chk(f"TabV {row[0]} {d}", cell, want)

# Table VI (tab:band) --------------------------------------------------------------------------
cf = {}
for f in RAW.glob("ablate_conformal_*.json"):
    r = json.loads(f.read_text())
    for meth, folds in r["methods"].items():
        m = pd.DataFrame(folds).mean()
        cf[(meth, r["disease"], r["model"])] = (m["coverage"], m["singleton_rate"])
spec = {"Coverage, band": ("project band (global width)", 0), "Coverage, CV+": ("CV+ (Barber et al.)", 0),
        "Singleton, band": ("project band (global width)", 1)}
for row in table("tab:band"):
    meth, i = spec[row[0]]
    for d, cell in zip(DIS, row[1:5]):
        want = f"{nz(cf[(meth, d, 'logreg')][i])}/{nz(cf[(meth, d, 'hgb')][i])}/{nz(cf[(meth, d, 'tabpfn')][i])}"
        chk(f"TabVI {row[0]} {d}", unbold(cell), want)

# Table VII (tab:novelty) --------------------------------------------------------------------------
nv = {}
for f in RAW.glob("ablate_novelty_*.json"):
    r = json.loads(f.read_text())
    for n, m in r["detectors"].items():
        nv[(n, r["disease"])] = m
det = {"Mahalanobis + Ledoit–Wolf (project)": "Mahalanobis + Ledoit-Wolf (project)", "Mahalanobis, empirical covariance": "Mahalanobis, empirical covariance",
       "Isolation Forest": "Isolation Forest", "kNN distance (k = 5)": "kNN distance (k=5)", "Local Outlier Factor": "Local Outlier Factor", "One-class SVM": "One-class SVM"}
for row in table("tab:novelty"):
    n = det[row[0]]
    fr = [nv[(n, d)]["cv_flag_rate"] for d in DIS]
    chk(f"TabVII {n} flag", row[1], f"{min(fr):.3f}–{max(fr):.3f}")
    au = [nv[(n, d)]["shift_auroc"] for d in ["ckd", "diabetes", "heart", "liver"]]
    chk(f"TabVII {n} auroc", row[2], " / ".join(nz(a) for a in au))
    q = [nv[(n, d)]["query_us"] for d in DIS]
    chk(f"TabVII {n} us", row[3], f"{round(min(q))}–{round(max(q))}")

# Table VIII (tab:shap) --------------------------------------------------------------------------
shap = D["tabpfn"]["shap"]
disease_shap_row = {"Kidney (25 features)": "ckd", "Diabetes (10 features)": "diabetes", "Heart (16 features)": "heart", "Liver (12 features)": "liver"}
for row in table("tab:shap"):
    d = disease_shap_row[row[0]]
    s = shap[d]
    chk(f"TabVIII {d} predict_ms", int(row[1]), round(s["predict_ms"]))
    chk(f"TabVIII {d} bg10", float(row[2]), round(s["bg10_median"], 1))
    chk(f"TabVIII {d} bg25", float(row[3]), round(s["bg25_median"], 1))
    speedup = round(s["bg25_median"] / s["bg10_median"], 2)
    chk(f"TabVIII {d} speedup", row[4], f"{speedup:.2f}×")
    chk(f"TabVIII {d} spearman", row[5], f"{s['spearman_mean']:.2f} / {s['spearman_min']:.2f}")
    chk(f"TabVIII {d} top3", row[6], f"{s['top3_mean']:.2f} / {s['top3_min']}")

# predict_proba single-row latency prose numbers (0.70 to 1.17 s) ---------------------------------
lo = min(s["predict_ms"] for s in shap.values()) / 1000
hi = max(s["predict_ms"] for s in shap.values()) / 1000
chk("predict_proba single-row range (s)", "0.70 to 1.17", f"{lo:.2f} to {hi:.2f}")
lat = D["tabpfn"]["latency_single_vs_batched"]
sp_lo, sp_hi = D["tabpfn"]["latency_speedup_range"]
chk("latency speedup range", "18 to 20", f"{int(sp_lo)} to {int(np.ceil(sp_hi))}")
ms_lo, ms_hi = D["tabpfn"]["latency_single_ms_range"]
chk("latency single ms range", "581 to 1,144", f"{ms_lo:.0f} to {ms_hi:,.0f}".replace(",", ",").replace("1144", "1,144"))
batched_lo = min(v["batched_ms_per_row"] for v in lat.values())
batched_hi = max(v["batched_ms_per_row"] for v in lat.values())
chk("latency batched ms/row range", "32 to 59", f"{batched_lo:.0f} to {batched_hi:.0f}")

# Table IX (tab:worked) -- against results/worked_examples.md ------------------------------------
we_text = (ROOT / "results" / "worked_examples.md").read_text()
we_sections = re.split(r"^## ", we_text, flags=re.M)[1:]
we = {}
for sec in we_sections:
    disease_name = sec.splitlines()[0].strip()
    for m in re.finditer(
        r"### (?P<kind>[\w ]+) \(test row \d+, true label (?P<true>\d)\)\n\nInputs:.*?\n\n"
        r"- probability (?P<p>[\d.]+), band \[(?P<lo>[\d.]+), (?P<hi>[\d.]+)\], threshold [\d.]+, "
        r"label \*\*(?P<label>[^*]+)\*\*, out-of-distribution: (?P<ood>True|False)\n"
        r"- top contributors: (?P<top>[^;]+) \(([+-][\d.]+)\)",
        sec,
    ):
        we[(disease_name, m.group("kind").strip())] = m.groupdict()

worked_map = {"Heart": ("Heart Disease", "most novel"), "Kidney": ("Chronic Kidney Disease", "first positive"), "Liver": ("Liver Disease", "most novel")}
for row in table("tab:worked"):
    disease_name, kind = worked_map[row[0]]
    ex = we[(disease_name, kind)]
    chk(f"TabIX {row[0]} true", int(row[2]), int(ex["true"]))
    chk(f"TabIX {row[0]} p", float(row[3]), float(ex["p"]))
    chk(f"TabIX {row[0]} band", row[4], f"[{ex['lo']}, {ex['hi']}]")
    chk(f"TabIX {row[0]} ood", row[6] == "**True**", ex["ood"] == "True")

# Table I (tab:data) --------------------------------------------------------------------------
sizes = {"Heart": "heart", "Diabetes": "diabetes", "Kidney": "ckd", "Liver": "liver"}
for row in table("tab:data"):
    k = sizes[row[0]]
    tr, te = pd.read_csv(ROOT.parent / f"data/processed/{k}_train.csv"), pd.read_csv(ROOT.parent / f"data/processed/{k}_test.csv")
    chk(f"TabI {k} rows", int(row[1]), len(tr) + len(te))
    chk(f"TabI {k} split", row[3], f"{len(tr)} / {len(te)}")
    chk(f"TabI {k} test pos", int(row[5]), int(te["target"].sum()))
    chk(f"TabI {k} features", int(row[2]), tr.shape[1] - 1)
    chk(f"TabI {k} pos rate", row[4], f"{100 * tr['target'].mean():.1f}%")

print(f"checked {checked} table cells; mismatches: {len(bad)}")
for b in bad:
    print(" -", b)
