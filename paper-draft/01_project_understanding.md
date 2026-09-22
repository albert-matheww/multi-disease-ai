# 01 - Project understanding: MultiDiseaseAI

> **Later update (2026-09-21):** this file is the Stage-1 snapshot. Since then the uncertainty layer and deployment files were committed
> (HEAD `4029225`, 81 tests pass), the doc discrepancies D1-D5, D7 were fixed in the repo, and evidence E5-E6 were extended by the ablation study
> (`results/ablation_tables.md`). The authoritative evidence map is `03_claim_ledger.md`; audit results are in `audit_report.md`.

Basis: working tree at `/Users/albertmathew/Desktop/projects/MultiDiseaseAI`, read on
2026-09-21. `HEAD` = `5667851`; **the working tree has substantial uncommitted work**
(see section 8). Test suite run today: `pytest tests/` -> **78 passed**.

## 1. What the project is

A config-driven pipeline and serving application that predicts the probability of four
diseases (heart disease, diabetes, chronic kidney disease, liver disease) from routine
clinical measurements, using **TabPFN as the only classifier**, one model per disease,
each trained on a small public UCI/NIDDK dataset. Around the model it provides
leakage-safe preprocessing, clinically-motivated engineered features, SHAP
explanations, three precomputed "uncertainty add-ons" (a conformal-style probability
band, a learned decision threshold, a Mahalanobis novelty flag), a Streamlit dashboard
with batch scoring and PDF reports, and Docker / Hugging Face Space deployment.
Stated origin: a Foundations of Data Science capstone.

## 2. Components -> files

| Component | Files |
|---|---|
| Disease registry (single source of truth for columns, targets, UI fields) | `src/config.py` |
| Data acquisition | `scripts/download_data.py` |
| Cleaning, split, impute, winsorize, encode; replayable transform | `src/preprocessing.py`, `src/artifacts.py` |
| Engineered features | `src/feature_engineering.py` |
| EDA | `src/eda.py`, `scripts/generate_eda_notebooks.py`, `notebooks/*_eda.ipynb` |
| Training (TabPFN + calibration pass) | `src/train.py` |
| Uncertainty add-ons | `src/uncertainty.py` (uncommitted) |
| Evaluation | `src/evaluate.py` |
| SHAP explanations | `src/explainability.py` |
| Serving (single + batch) | `src/prediction.py` |
| PDF report; SQLite history | `src/report.py`, `src/history.py` |
| Dashboard | `app/streamlit_app.py`, `app/components.py`, `app/theme.py` |
| Orchestration, tests, CI | `main.py`, `tests/`, `.github/workflows/ci.yml` |
| Deployment | `Dockerfile`, `docker-compose.yml`, `deploy/huggingface/`, `.github/workflows/deploy-hf-space.yml` (last two uncommitted) |

## 3. Methodology as implemented

- **Pipeline order** (`src/preprocessing.py`): deterministic cleaning -> one stratified
  80/20 split (`RANDOM_STATE=42`) -> median/most-frequent imputation -> 1.5xIQR
  winsorization -> ordinal encoding (`unknown_value=-1`) -> feature engineering. All
  fitted statistics come from the training split only (`tests/test_preprocessing.py`
  checks the IQR bounds against the imputed train split). A saved `PreprocessArtifact`
  replays the same transform on a live record.
- **Engineered features** (`src/feature_engineering.py`): heart - rate-pressure product,
  heart-rate reserve, cholesterol/age; diabetes - glucose x BMI, insulin/glucose;
  CKD - comorbidity count (htn+dm+cad); liver - AST/ALT ratio, direct/total bilirubin.
- **Model** (`src/train.py`): `TabPFNClassifier(categorical_features_indices=...,
  n_estimators=8, random_state=42, device="cpu")`. Pins: `tabpfn==8.2.0`,
  `torch==2.13.0`, `scikit-learn==1.9.0`, `shap==0.51.0`. No hyperparameter search.
- **Calibration pass** (`train.py` + `uncertainty.py`), k=5 stratified out-of-fold
  (capped at the smaller class size) on the training split, giving: sorted residuals
  `|y - p_oof|`; a Youden-J threshold on OOF probabilities; a Ledoit-Wolf
  Mahalanobis detector fit on the encoded training matrix (flag above the 97.5%
  training-distance quantile). All stored in one bundle `models/{key}_tabpfn.joblib`.
- **Conformal-style band** (`probability_band`): `[p - h, p + h]` clipped to [0,1],
  where `h` is the ceil((n+1)(1-alpha))-th sorted OOF residual, alpha = 0.10.
  **`h` is one global constant per disease** - it does not depend on the patient.
- **Explanations** (`src/explainability.py`): SHAP `PermutationExplainer` on
  `predict_proba`, k-means background (25 for global plots, 10 for interactive use).
- **Serving** (`src/prediction.py`): label = `probability >= decision_threshold`;
  risk band display thresholds 0.33 / 0.66; SHAP memoised per input.

## 4. Data (reproducible from `scripts/download_data.py` + `python -m src.preprocessing`)

Counts below were computed today from `data/raw/` and `data/processed/`.

| Disease | Source | Raw rows | Train / test | Model features | Test pos / neg |
|---|---|---|---|---|---|
| Heart | UCI id 45 (Cleveland) | 303 | 242 / 61 | 16 | 28 / 33 |
| Diabetes | Pima Indians (NIDDK mirror) | 768 | 614 / 154 | 10 | 54 / 100 |
| CKD | UCI id 336 | 400 | 320 / 80 | 25 | 50 / 30 |
| Liver | UCI ILPD id 225 | 583 (570 after dropping 13 duplicates) | 456 / 114 | 12 | 81 / 33 |

Positive rate: heart 45.9%, diabetes 34.9%, CKD 62.5%, liver 71.3% (train) / 71.1% (test).
Known quirks handled in code: Pima zeros as missing (glucose 5, blood pressure 35, skin
thickness 227, insulin 374, BMI 11 - taken from an earlier preprocessing log; re-derive in the accuracy audit before use), CKD whitespace labels,
CKD missingness up to ~38% (`rbc`). The test sets are **small** (61-154 rows).

## 5. Evidence inventory

### Exists (traceable to an artifact)

| id | Evidence | How to reproduce |
|---|---|---|
| E1 | 78 unit/integration tests pass (config consistency, feature math, preprocessing leakage-safety, uncertainty functions, prediction wiring on a `LogisticRegression` stand-in, PDF content) | `pytest tests/ -q` (run 2026-09-21) |
| E2 | Dataset sizes, splits, class balance, feature counts (section 4) | `python -m src.preprocessing`; CSVs in `data/` |
| E3 | Executed EDA notebooks (missingness, prevalence, distributions, correlations) | `notebooks/*_eda.ipynb`, `reports/figures/` |
| E4 | Software design and behaviour of each module | source + tests |
| E5 | **Baseline experiments** (tuned logistic regression and gradient boosting): fixed-split test probabilities with bootstrap CIs, repeated 5-fold CV (15 folds), empirical coverage of the nominal 90% band | `paper-draft/experiments/run_experiments.py`; outputs in `paper-draft/results/` with provenance (git head 5667851, dirty tree, package versions), run 2026-09-21 |
| E6 | **Novelty-flag experiments** (model-free): held-out flag rate on the shipped split and in pooled 5-fold CV; an age-shift experiment | same harness, `--stage novelty` |

### Does NOT exist

(as of 2026-09-21, before any TabPFN run)

- **No trained TabPFN model of any kind**: no `models/*_tabpfn.joblib`, no
  `reports/*_metrics.json`, no ROC / PR / confusion-matrix / SHAP figures, no OOF
  probabilities. `TABPFN_TOKEN` is not set on this machine and `~/.tabpfn` does not exist.
  (`docs/HOW_IT_WORKS.md` s6 and the README both say the repo ships no metric numbers.)
- **No performance result of any kind** - no accuracy, AUC, F1, calibration.
- No measured coverage of the probability band **for TabPFN** (measured for the baselines, E5); no TabPFN
  novelty-independent validation beyond E6; no OOF-vs-test comparison of the Youden threshold.
- No baseline model (by design: "TabPFN only"), no confidence intervals, no
  repeated-split or external validation, no user study, no clinical validation.
- No benchmark behind the latency claims ("materially faster", "ranking essentially
  unchanged" for SHAP background 25 -> 10).
- No evidence that the Hugging Face Space is live.

### Untraceable results found in git history - **do not use**

Commit `5f79eab` added `reports/threshold_sweep_results.md` with a table of "best t /
F1 / acc" per disease (e.g. liver F1 0.803, heart acc 0.928), described as produced by
running `scripts/sweep_thresholds.py` "against the tabpfn test predictions". The script
reads `data/processed/{name}_test_scores.npy` and `_test_labels.npy`, **which no code
in the repository ever wrote** (`git log -S"test_scores"` finds only the script and its
deletion), and no trained model exists. Commit `d009d4d` later deleted the file. These
numbers have no run behind them and are excluded from the paper.

## 6. Discrepancies between docs and code (seeds for the accuracy audit)

| id | Claim | What the code does | Consequence for the paper |
|---|---|---|---|
| D1 | README l.310 and `HOW_IT_WORKS.md` s5a: "a wide band means the model's own out-of-fold accuracy on similar rows was poor" | `h` is a single global residual quantile; every patient gets the same band width (except clipping at 0/1). It says nothing about *similar rows* | Do not repeat. Describe the band as a global, non-adaptive width |
| D2 | "90% band with distribution-free marginal coverage" of the probability | The conformal guarantee concerns whether the binary label `y in {0,1}` lies in `[p-h, p+h]`, not whether the true `P(y=1|x)` does; residuals are pooled OOF residuals applied to the full-data model's `p` (cross-conformal heuristic, approximate) and coverage was never measured | Describe as conformal-inspired, approximate, unvalidated |
| D3 | Dashboard shows metrics and a label | `src/evaluate.py` scores `model.predict(X_test)` (default 0.5 rule) but `DiseasePredictor` labels with the Youden threshold | Evaluation and serving use different decision rules; report both or align them |
| D4 | `src/train.py` docstring: "Hollmann et al., 2023, *Nature*" | The *Nature* paper is 2025 (Crossref); the 2023 paper is the ICLR one | Cite correctly in the paper; docstring should be fixed |
| D5 | Sidebar: "TabPFN v2" | Pinned package is `tabpfn==8.2.0`; the weights version is unknown until a model is loaded | Say "package version 8.2.0"; do not say "v2" |
| D7 | `data/README.md` and the README cite Heart Disease as (1988) and ILPD as (2012) | UCI's own suggested citations read 1989 (heart), 2015 (CKD), 2022 (ILPD); 1988 and 2012 are *donation* dates | Paper follows the UCI suggested citation; the project docs should be corrected (this was an error introduced when the docs were first written) |
| D8 | UCI ILPD page: "Has Missing Values? No" | The project's raw ILPD file has 4 missing `A/G Ratio` values (found in EDA) | Paper reports the count found in the data and does not repeat the UCI statement |
| D5b | (evidence for D5) | An empty lock file `~/Library/Caches/tabpfn/.tabpfn-v3-classifier-v3_default.ckpt.lock` shows the installed package tried to fetch a "v3" default checkpoint | Indirect only; the loaded model's metadata settles it at run time |
| D6 | Commit dates in `git log` are 2026-07-08 .. 2026-07-17 | File modification times are 2026-07-30 onward; `HEAD` is dated 2026-07-17 but files changed later | Commit dates are not reliable evidence of chronology; **the paper makes no development-timeline claims** |

## 7. Contribution analysis

| Candidate | Type | Evidence today | Verdict |
|---|---|---|---|
| C1. Config-driven, leakage-safe, replayable preprocessing + clinically-motivated features feeding TabPFN across four datasets | System / engineering | Code + tests (E1) | Defensible as a system description |
| C2. Precomputed uncertainty add-ons (band, threshold, novelty flag) bundled with a TabPFN model so serving stays one forward pass | System / engineering | Code + property tests only | Defensible as *design*; **effectiveness unevidenced** |
| C3. SHAP for a transformer classifier via permutation explainer with k-means background, plus caching | System | Code | Design only; no fidelity or timing data |
| C4. Deployment path (Docker, HF Space) | Engineering | Files only. An earlier version of the Dockerfile was build-tested in a prior session; the *current* one (non-root, HF-oriented) has not been build-tested by me | Describe as provided, not as verified |
| C5. Empirical evaluation of TabPFN on four small clinical datasets | Empirical / application | **None** | **Not claimable** |

One-sentence test (best honest version): *relative to notebook-style analyses that report
a single classifier's score on these datasets, this work packages TabPFN into a
reproducible serving system with a leakage-safe replayable transform and cached
uncertainty add-ons, shown by the code and a 78-test suite.* It passes as a **system
paper**. It does not pass as a paper claiming that the model or the uncertainty layer
*works*, because no run supports that. Realistic framing: student system / application
paper (student symposium, workshop, technical report), not a methods paper.

## 8. Reproducibility caveat

The uncertainty layer (`src/uncertainty.py`, `tests/test_uncertainty.py`), the HF
deployment files, `docs/HOW_IT_WORKS.md`, and edits to eleven tracked files are
**uncommitted**. A paper describing them needs a commit hash that contains them.

## 9. Experiments that would create real evidence (each needs the user's approval)

| id | Experiment | Needs | Supports |
|---|---|---|---|
| X1 | Train the four bundles; evaluate on the held-out test split at 0.5 and at the Youden threshold; bootstrap CIs (test n = 61-154) | `TABPFN_TOKEN` (user's account and licence) | Any performance claim |
| X2 | Empirical label-coverage and mean width of the 90% band on test data, ideally over repeated splits | X1 | Whether D2's heuristic holds; whether the band is informative or trivially wide |
| X3 | Novelty-flag rate on held-out rows vs the nominal 2.5%; optionally a controlled shift | X1 | Whether the flag is calibrated |
| X4 | SHAP timing and top-feature rank agreement, background 10 vs 25 | X1 | The latency claims |
| X5 | Reference baselines (e.g. logistic regression, gradient boosting) used **only inside the paper's experiments** | User decision: the project's rule is "TabPFN only" | Any claim that TabPFN is *good relative to* something |

Without X5 the paper can report absolute numbers only and cannot claim TabPFN is
suitable or better; it can compare with published numbers only where dataset, split and
metric match (they will not).

## 10. Length feasibility

- Without X1-X4: an honest system paper is roughly **3,000-4,000 words**. Reaching 6,000
  would require padding, which this skill does not do.
- With X1-X4 (and X5 if approved): **5,000-6,000 words** is realistic.
