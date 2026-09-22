#!/usr/bin/env python3
"""Figures for the paper, drawn from results/raw only (no hand-typed values)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RAW, FIG = ROOT / "results" / "raw", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

# --- Fig 1: pipeline (drawn to match src/train.py, src/prediction.py) -----------------------------
fig, ax = plt.subplots(figsize=(3.5, 2.55))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")


def box(x, y, w, h, text, fc="#eef3fb"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec="#33507a", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.3)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=0.8, color="#33507a"))


box(0.1, 6.9, 2.6, 2.4, "UCI / NIDDK\nraw table")
box(3.2, 6.9, 3.1, 2.4, "clean -> split 80/20\nimpute, winsorize,\nencode, engineer\n(fit on train only)")
box(6.8, 6.9, 3.1, 2.4, "TabPFN\n(fit = cache\ntraining table)")
arrow(2.7, 8.1, 3.2, 8.1); arrow(6.3, 8.1, 6.8, 8.1)
box(0.1, 3.3, 4.8, 2.5, "5-fold out-of-fold pass:\nresidual quantile h,\nYouden threshold,\nLedoit-Wolf Mahalanobis", fc="#fdf3e3")
box(5.4, 3.3, 4.5, 2.5, "one model bundle\n(model, h, threshold,\ndetector, feature order)", fc="#e9f5ec")
arrow(8.3, 6.9, 8.3, 5.8); arrow(2.4, 6.9, 2.4, 5.8); arrow(4.9, 4.55, 5.4, 4.55)
box(0.1, 0.2, 9.8, 2.2, "serve: one forward pass gives p; band [p-h, p+h];\nlabel (p >= threshold); novelty flag; SHAP on request", fc="#f3ecf8")
arrow(7.6, 3.3, 7.6, 2.4)
fig.savefig(FIG / "fig1_pipeline.pdf", bbox_inches="tight"); fig.savefig(FIG / "fig1_pipeline.png", dpi=200, bbox_inches="tight")

# --- Fig 2: novelty detectors, calibration vs sensitivity ---------------------------------------
import pandas as pd

rows = []
for f in sorted(RAW.glob("ablate_novelty_*.json")):
    r = json.loads(f.read_text())
    for n, m in r["detectors"].items():
        rows.append({"disease": r["disease"], "det": n, **m})
df = pd.DataFrame(rows)
style = {"Mahalanobis + Ledoit-Wolf (project)": ("#c0392b", "o"), "Mahalanobis, empirical covariance": ("#e59866", "s"), "Isolation Forest": ("#7f8c8d", "^"),
         "kNN distance (k=5)": ("#2e86c1", "v"), "Local Outlier Factor": ("#1abc9c", "D"), "One-class SVM": ("#8e44ad", "P")}
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for det, (c, mk) in style.items():
    d = df[df.det == det]
    ax.scatter(100 * d.cv_flag_rate, d.shift_auroc, c=c, marker=mk, s=22, label=det.replace(" (project)", "*"), edgecolors="k", linewidths=0.3)
ax.axvline(2.5, color="k", lw=0.6, ls="--")
ax.text(2.8, 0.985, "nominal 2.5%", fontsize=6, va="top")
ax.set_ylim(0.58, 0.99)
ax.set_xlabel("flag rate on held-out normal rows (%)"); ax.set_ylabel("age-shift AUROC")
ax.legend(fontsize=5.3, loc="lower right", frameon=False)
fig.savefig(FIG / "fig2_novelty.pdf", bbox_inches="tight"); fig.savefig(FIG / "fig2_novelty.png", dpi=200, bbox_inches="tight")
print("figures written to", FIG)
