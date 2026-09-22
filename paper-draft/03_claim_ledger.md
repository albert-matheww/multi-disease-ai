# 03 - Claim ledger

Types: PROJECT-FACT (code/config), RESULT (run artifact), BACKGROUND (source + read depth),
COMPARISON, INTERPRETATION (authors' reasoning, flagged). Status: `ok` = backed and audited,
`PENDING` = needs a TabPFN run (no token on this machine), `dropped` = removed as unsupported.
Result cells are produced by `paper-draft/experiments/*.py` -> `paper-draft/results/`; numbers in
prose come from `results/derived_stats.json` (script `derived_stats.py`), never typed by hand.

## Project facts

| id | section | claim | type | evidence | status |
|---|---|---|---|---|---|
| P1 | III | Four binary tasks (heart, diabetes, CKD, liver), one TabPFN model each; TabPFN is the only classifier in the shipped system | PROJECT-FACT | `src/config.py`, `src/train.py` | ok |
| P2 | IV | Sizes / splits: heart 303 (242/61), diabetes 768 (614/154), CKD 400 (320/80), liver 583 -> 570 after 13 duplicates (456/114); model features 16/10/25/12; test positives 28/54/50/81 | PROJECT-FACT | `data/raw/*.csv`, `data/processed/*.csv`, recounted 2026-09-21 | ok |
| P3 | III | Order: clean -> one stratified 80/20 split (seed 42) -> median/most-frequent imputation -> 1.5xIQR winsorization -> ordinal encoding -> engineered features; every fitted statistic from the training split; a saved artifact replays the transform | PROJECT-FACT | `src/preprocessing.py`, `src/artifacts.py`, `tests/test_preprocessing.py` | ok |
| P4 | III | Engineered features: heart rate-pressure product, heart-rate reserve, cholesterol/age; diabetes glucose x BMI, insulin/glucose; CKD comorbidity count; liver AST/ALT, direct/total bilirubin | PROJECT-FACT | `src/feature_engineering.py` | ok |
| P5 | III | `TabPFNClassifier(n_estimators=8, random_state=42, device="cpu", categorical_features_indices=...)`; no hyperparameter search; pins tabpfn 8.2.0, torch 2.13.0, scikit-learn 1.9.0, shap 0.51.0 | PROJECT-FACT | `src/train.py`, `requirements.txt` | ok |
| P6 | III | Calibration pass: stratified K=5 out-of-fold predictions (K capped at the smaller class) on the training split give residuals, a Youden-J threshold and a Ledoit-Wolf Mahalanobis detector (flag above the 97.5% training-distance quantile); all stored in one bundle | PROJECT-FACT | `src/train.py`, `src/uncertainty.py`, `src/config.py` | ok |
| P7 | III | Band `[p-h, p+h]` clipped to [0,1], `h` = ceil((n+1)(1-alpha))-th sorted OOF residual `|y-p|`, alpha = 0.10; **h is one global value per disease** | PROJECT-FACT | `src/uncertainty.py` (`conformal_halfwidth`, `probability_band`) | ok |
| P8 | III | SHAP `PermutationExplainer` on `predict_proba`, k-means background 25 (global plots) / 10 (interactive) | PROJECT-FACT | `src/explainability.py` | ok |
| P9 | III | Serving: label = `p >= decision_threshold`; risk display bands 0.33/0.66; batch scoring, PDF report, SQLite history; Streamlit; Docker and Hugging Face Space files | PROJECT-FACT | `src/prediction.py`, `src/report.py`, `src/history.py`, `app/`, `Dockerfile`, `deploy/` | ok (deployment not verified live) |
| P10 | III/VII | 81 unit/integration tests pass; prediction wiring is tested on a logistic-regression stand-in, not on TabPFN | PROJECT-FACT | `pytest tests/` 2026-09-21; `tests/test_prediction.py` | ok |
| P11 | IV | Data quirks: Pima zeros treated as missing (glucose 5, blood pressure 35, skin thickness 227, insulin 374, BMI 11); CKD missingness up to 152/400 (`rbc`); heart `ca` 4 and `thal` 2 missing; ILPD 4 missing A/G ratio, 13 duplicate rows | PROJECT-FACT | recounted from `data/raw/` 2026-09-21; `src/preprocessing.py:78` | ok |
| P12 | VII | `src/evaluate.py` now reports metrics at 0.5 and at the served (Youden) threshold; earlier it reported only 0.5 while serving used Youden | PROJECT-FACT | commit history, `src/evaluate.py`, `tests/test_evaluate.py` | ok |
| P13 | VII | Deleted `reports/threshold_sweep_results.md` had no run behind it; **excluded** | PROJECT-FACT | git history (`5f79eab`, `d009d4d`) | ok (excluded) |

## Results actually measured (reference models; no TabPFN)

Reference models: L2 logistic regression (standardised), histogram gradient boosting, random forest, RBF SVM. Baselines get an inner 5-fold grid on the training split (fit counts 26 / 61 / 21 / 16 from the grids in `run_experiments.py`); ablation studies use fixed hyperparameters (LR C=1, HGB lr 0.05 / depth 3 / 150 iterations) so that only the studied factor varies. Outer protocol: repeated stratified 5-fold, 3 repeats = 15 folds; paired tests are Nadeau-Bengio corrected resampled t-tests.

| id | section | claim | type | evidence | status |
|---|---|---|---|---|---|
| R1 | V-A | 5-fold x3 CV AUC of four reference models per disease (Table II) | RESULT | `results/tables.md` (CV), `results/raw/cv_*.json` | ok |
| R2 | V-A | Best reference model by mean CV AUC: CKD svm (all four ~1.000), diabetes rf, heart logreg, liver logreg; logreg minus HGB = +0.0003 / +0.0084 / +0.0078 / +0.0218 (ckd / diabetes / heart / liver) | RESULT | `derived_stats.json` baselines | ok |
| R3 | V-A | On the shipped liver split logreg AUC 0.816 [0.741, 0.887] vs HGB 0.729 [0.632, 0.818]; in CV 0.744 vs 0.722 | RESULT | `tables.md` (fixed split, CV) | ok |
| R4 | V-A | Wall-clock per outer fold incl. tuning: logreg 0.05 s, svm 0.05-0.20 s, rf 1.4-1.8 s, hgb 6.5-10.3 s (CPU) | RESULT | `derived_stats.json` baselines; `ablation_tables.md` E | ok |
| R5 | V-B | 44 variant-vs-project preprocessing comparisons (5 variants x 8 model-disease pairs, plus native-missing values for the 4 HGB pairs, since logistic regression cannot take NaN); 40 have a defined p; none has p < 0.05; smallest p 0.205; largest |dAUC| 0.0056 | RESULT | `derived_stats.json` preprocessing; `ablation_tables.md` A | ok |
| R6 | V-B | Leaky variant (statistics fit on train+test) changes AUC by -0.0003 .. +0.0015 (max abs 0.0015, all p >= 0.512) | RESULT | same | ok |
| R7 | V-B | Per-variant dAUC ranges and min p (Table III) | RESULT | `derived_stats.json` per_variant | ok |
| R8 | V-C | Youden vs fixed 0.5 (balanced accuracy): significant only on liver (+0.086 hgb p=0.002; +0.096 logreg p=0.001); diabetes +0.027 / +0.024 (p 0.133 / 0.253); heart and CKD differences within -0.007 .. -0.002 (p >= 0.397) | RESULT | `ablation_tables.md` B2 | ok |
| R9 | V-C | Youden vs training-prevalence rule: no significant difference in any of 8 pairs (min p 0.335, max abs diff 0.009) | RESULT | `derived_stats.json` threshold | ok |
| R10 | V-C | F1-optimal rule is degenerate on liver: mean specificity 0.029 (hgb) / 0.018 (logreg); Youden better by +0.161 / +0.182 balanced accuracy (p < 0.001) | RESULT | same | ok |
| R11 | V-D | Project band coincides with the label-set conformal method (identical coverage, set size, singleton/both/empty rates) in all 8 pairs; coverage 0.900-0.920 at nominal 0.90 | RESULT | `derived_stats.json` uncertainty; `ablation_tables.md` C | ok |
| R12 | V-D | CV+ coverage 0.898-0.929; max |CV+ - band| = 0.012 | RESULT | same | ok |
| R13 | V-D | Band informativeness: singleton-set rate 0.526 (liver hgb) .. 0.92 (CKD); both-label rate up to 0.474; empty-set rate up to 0.082 (CKD, where h is tiny) | RESULT | same | ok |
| R14 | V-D | Band lookup costs 0.53 microseconds per call | RESULT | `results/raw/ablate_overhead.json` | ok |
| R15 | V-D | On tuned reference models the band's outer-CV label coverage is 0.897-0.920 with all Wilson 95% intervals containing 0.90 | RESULT | `tables.md` (coverage) | ok |
| R16 | V-E | CV flag rate at nominal 2.5%: Mahalanobis+LW 0.030-0.036; empirical-covariance Mahalanobis 0.030-0.087; IF 0.030-0.048; kNN 0.020-0.040; LOF 0.020-0.028; OCSVM 0.108-0.237 | RESULT | `derived_stats.json` novelty | ok |
| R17 | V-E | Age-shift AUROC (project vs best alternative): CKD 0.638 (rank 5 of 6; all detectors 0.64-0.69); diabetes 0.946 (rank 3; OCSVM 0.968, LOF 0.951); heart 0.888 (rank 3); liver 0.909 (rank 2; OCSVM 0.951) | RESULT | same | ok |
| R18 | V-E | Query time per row: Mahalanobis+LW 9-14 us; OCSVM 92-102 us; LOF 185-512 us; kNN 155-458 us; IF 1544-1552 us; Ledoit-Wolf beats empirical covariance on liver (AUROC 0.909 vs 0.798) and CKD flag rate (0.030 vs 0.087) | RESULT | same | ok |
| R19 | V-E | Heart, age <= median vs > median (held-out): project flag rate 0.061 vs 0.574 | RESULT | `tables.md` novelty | ok |

## Live implementation (Sec. V-G, added 2026-09-23)

| id | section | claim | evidence | status |
|---|---|---|---|---|
| L1 | V-G | Fig. 3: the heart-disease form filled with Table IX's "most novel" row, and the app's live result panel for that submission (probability 97.1%, band 31-100%, label, novelty warning, SHAP chart), captured from the Streamlit app running locally against the real trained bundle | `paper-draft/experiments/capture_screenshots.py` (Playwright, headless Chromium) -> `figures/screenshots/02_filled_form.png`, `04_shap_contributions.png`; composited into `figures/fig3_ui_prediction.png` | ok |
| L2 | V-G | The live SHAP values (ca +0.136, cp +0.095, thalach +0.054) differ from Table IX's harness values (ca +0.139, cp +0.086, thalach +0.055) because the interactive serving path (`src/prediction.py`, `_INTERACTIVE_SHAP_BACKGROUND=10`) does not seed its k-means background sample, unlike the timing harness (`run_experiments.py stage_shap`, seeds every call) | re-read `src/prediction.py`, `src/explainability.py`; both runs' raw values compared by hand | ok (real, disclosed as a shipped-code gap, not hidden) |
| L3 | V-G | Fig. 4: the app's "Model Performance" page for heart disease reproduces Table II's fixed-split numbers exactly (accuracy 88.5%, F1 88.1%, AUC 0.959; at t=0.56: accuracy 88.5%, precision 86.2%, recall 89.3%, F1 87.7%; confusion matrix 28/5/2/26, n=61) | `reports/heart_metrics.json` (same file the app reads); cross-checked against `results/tables.md` heart-tabpfn row | ok |

## Results that needed TabPFN (unblocked 2026-09-22: user obtained a `TABPFN_TOKEN` from ux.priorlabs.ai and accepted the TabPFN-3 non-commercial licence themselves in the browser; token stored only in the untracked local `.env`)

| id | section | claim | evidence path | status |
|---|---|---|---|---|
| T1 | V-A | TabPFN CV AUC (0.918/0.843/1.000/0.741), fixed-split metrics with CIs, seconds/fold (1.0-1.8s) | `results/raw/cv_*_tabpfn.json`, `results/raw/main_*_tabpfn.json`, `results/tables.md` | ok |
| T2 | V-A | Paired TabPFN-minus-baseline AUC differences with Nadeau-Bengio p (Table III): significant only vs SVM (3/4) and HGB (1/4) | `results/summary.json` (`auc_diff_tabpfn_minus_this`), `results/derived_stats.json` -> `tabpfn.pairwise_vs_baselines_raw` | ok |
| T3 | V-D | Band coverage/informativeness for TabPFN: 0.901-0.930, identical to the label-set method, near-zero width on kidney (h=0.001) | `results/raw/coverage_*_tabpfn.json`, `results/raw/ablate_conformal_*_tabpfn.json` | ok |
| T4 | V-B/C | Preprocessing (24 pairs, incl. native-missing) and threshold (12 comparisons) ablations with TabPFN: same pattern as the reference models (no significant preprocessing effect; Youden beats 0.5 only on liver) | `results/raw/ablate_preproc_*_tabpfn.json`, `results/raw/ablate_threshold_*_tabpfn.json` | ok |
| T5 | V-F | SHAP timing: single-row predict 0.70-1.17s; background 10 vs 25 saves 1.3-2.4x (Spearman 0.92-0.96 mean, as low as 0.67); KernelSHAP faster than bg=25 on 2/4 datasets, slower on 2/4; separate batched-vs-single latency check: 18-20x slower unbatched | `results/raw/shap_*.json`, `results/raw/latency_*.json` (new script `latency_batch_vs_single.py`) | ok |
| T6 | V-F | Worked examples on real held-out test rows (selection rule fixed in advance): 12 rows across 4 diseases, incl. one genuine misclassification (liver, most-novel row) | `results/worked_examples.md` | ok |

## Background (source, read depth) and interpretation

| id | section | claim | type | evidence | status |
|---|---|---|---|---|---|
| B1 | II | TabPFN: transformer trained on synthetic tasks, predicts in one forward pass; strongest reported advantage up to 10,000 samples / 500 features; supports categoricals and missing values | BACKGROUND | hollmann2025, full | ok |
| B2 | II | Original TabPFN limits (1,000 rows, 100 numeric features, 10 classes) | BACKGROUND | hollmann2023, full | ok |
| B3 | II | PFN idea: train on tasks drawn from a prior, approximate posterior predictive in one pass | BACKGROUND | muller2022, full | ok |
| B4 | II | Tree ensembles ahead of neural networks on ~10K-sample tabular data (pre-TabPFN-generation) | BACKGROUND | grinsztajn2022, full | ok |
| B5 | II | Clinical benchmark preprint: 12 binary tasks, TabPFN generally competitive, best in 16.7% of tasks, about 5.5x longer runtime | BACKGROUND | clinbench2026, **abstract only, preprint** | ok (hedged) |
| B6 | II | A fine-tuned TabPFN was reported to consistently outperform conventional methods on clinical prediction tasks (no named baselines or dataset count are claimed: the saved abstract does not give them) | BACKGROUND | ehrsmall2026, abstract only (corrected in audit 2) | ok (hedged) |
| B7 | II | SHAP definition; TabPFN-specific Shapley/LOCO via in-context learning; one-pass explanation model; caveats on Shapley importance | BACKGROUND | lundberg2017, rundel2024, realtime2026 (workshop), kumar2020, full | ok |
| B8 | II | Jackknife interval form and its possibly very poor coverage; CV+ carries a worst-case guarantee of about 1-2alpha; cross-conformal validity studied empirically; coverage marginal, usefulness set by the score | BACKGROUND | barber2021, vovk2015, angelopoulos2021, full | ok |
| B9 | II | Mahalanobis OOD scoring on deep features; shrinkage covariance; calibration of modern networks; Youden index | BACKGROUND | lee2018, ledoit2004 (**metadata + sklearn docs**), guo2017, shan2015 | ok (hedged) |
| B10 | II/IV | Leakage survey (17 fields, 329 papers, some overoptimistic conclusions; taxonomy item L1.2 = pre-processing on training and test set); small-sample CV error bars; corrected resampled t-test; Wilson interval | BACKGROUND | kapoor2023, varoquaux2018, nadeau2003 (full); brown2001 (abstract) | ok |
| B11 | IV | Dataset provenance and citations (UCI 1989 / 2015 / 2022 suggested years); Pima population female, >= 21, Pima heritage; ILPD subgroup disparity | BACKGROUND | uci_*, detrano1989 (abstract), pima_docs, straw2022 | ok |
| I1 | III | The band is a K-fold analogue of the jackknife interval and, for binary labels, equals a label-set conformal predictor (`|y-p| = 1 - p_true`) | INTERPRETATION + RESULT R11 | derivation in Sec. III; empirical identity R11 | ok |
| I2 | VI | Choices are "not distinguishable from alternatives" when p >= 0.2 across a paired test with 15 overlapping folds; that is absence of evidence of a difference, not evidence of equivalence | INTERPRETATION | R5-R9 | ok |
| I3 | VI | Efficiency is measured as wall-clock and per-query cost on CPU, not as accuracy | INTERPRETATION | R4, R14, R18 | ok |
| I4 | VII | CV+ needs the K fold models' predictions at query time, so serving would need K forward passes; the shipped band needs one | INTERPRETATION (by construction, not measured) | barber2021 (full); `src/train.py` stores one model | ok |

## Dropped

- "The band contains 0.5 for every patient, hence no class information" - **false** (labels are 0/1; singleton rate 0.53-0.92); removed from docs and tables.
- "A wide band means poor OOF accuracy on similar rows" (old README) - false (global h); corrected.
- Any performance figure for TabPFN, any timeline claim, the threshold-sweep table (P13).
