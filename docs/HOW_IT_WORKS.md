# How MultiDiseaseAI works, start to end

This document walks the whole system in the order data actually flows through
it: **raw CSVs → cleaning → feature engineering → TabPFN training → precomputed
uncertainty add-ons → evaluation → explainability → serving (single + batch) →
PDF report → history/monitoring → dashboard → deployment**. Every stage maps to
one module under `src/`; the Streamlit app and the test suite are thin callers of
those same modules, so there is exactly one implementation of each piece of
logic.

```
download_data.py ─► data/raw/*.csv
        │
        ▼
preprocessing.py ──► clean ─► split ─► impute ─► winsorize ─► ordinal-encode
        │                                                        │
        │                                          feature_engineering.py
        ▼                                                        ▼
data/processed/{key}_{train,test}.csv        models/{key}_preprocessor.joblib
        │
        ▼
train.py ──► TabPFNClassifier.fit  +  uncertainty.py (k-fold OOF pass)
        │        │                         │
        │        │            conformal residuals · Youden threshold · OOD detector
        ▼        ▼                         ▼
        └────────────► models/{key}_tabpfn.joblib  (one bundle: model + add-ons)
                                  │
             ┌────────────────────┼─────────────────────┐
             ▼                    ▼                     ▼
        evaluate.py         explainability.py       prediction.py ──► DiseasePredictor
        metrics.json        SHAP global plots        │        │
        ROC/PR/CM figs                               │        ├─► report.py  (PDF)
                                                     │        └─► history.py (SQLite)
                                                     ▼
                                          app/streamlit_app.py (5 pages)
                                                     │
                                          deploy/huggingface/  ──►  HF Docker Space
```

---

## 0. The single source of truth: `src/config.py`

Nothing in the pipeline hard-codes a column name. `config.py` defines a
`DiseaseConfig` dataclass per disease (`HEART`, `DIABETES`, `CKD`, `LIVER`,
collected in the `DISEASES` dict) that declares:

- `raw_filename`, `target_column`, and a `target_mapper` lambda that binarises
  the raw label (e.g. heart's `num > 0 → 1`, CKD's `"ckd" → 1`).
- `numeric_features` / `categorical_features` — the **model input columns**,
  *including* the engineered ones (so training and serving agree on order).
- `fields: list[FieldSpec]` — the **raw user inputs** (pre-engineering): label,
  unit, min/max, default, and for categoricals a `{human label → raw value}`
  map. This one list drives the Streamlit form, input validation, the CSV batch
  template, and the human-readable rows in the PDF.
- `field_names()` returns only the raw inputs — deliberately *not*
  `numeric_features + categorical_features`, because engineered features must
  never be asked of the user. `tests/test_config.py` guards this distinction.

Also here: `RANDOM_STATE`, `TEST_SIZE`, the `RISK_THRESHOLDS` display bands
(<33 % Low, <66 % Moderate, else High), and the uncertainty knobs
`CONFORMAL_ALPHA = 0.10`, `OOD_QUANTILE = 0.975`, `CV_FOLDS = 5`. The helper
`humanize_feature_name(disease_key, name)` maps any raw *or* engineered feature
to a display label and is shared by the report and the dashboard so they never
disagree.

---

## 1. Data acquisition — `scripts/download_data.py`

Fetches all four datasets into `data/raw/` (idempotent, safe to re-run):

| Disease | Source | Rows | Positive rate |
|---|---|---|---|
| Heart | UCI id=45 (Cleveland) via `ucimlrepo` | 303 | 45.9 % |
| Diabetes | Pima Indians (NIDDK) public-domain CSV mirror | 768 | 34.9 % |
| CKD | UCI id=336 via `ucimlrepo` | 400 | 62.5 % |
| Liver | UCI ILPD id=225 via `ucimlrepo` | 583 (570 de-duped) | 71.2 % |

Per-dataset quirks, licences and citations live in [`data/README.md`](../data/README.md).

---

## 2. Preprocessing — `src/preprocessing.py`

`DiseasePreprocessor(disease).run()` executes a fixed, leakage-safe sequence and
writes `data/processed/{key}_train.csv`, `{key}_test.csv`, and the fitted
`models/{key}_preprocessor.joblib` artifact.

1. **`clean_raw`** (deterministic, no fitting): strip stray whitespace/tabs from
   string cells (CKD's `"ckd\t"`), convert dataset-specific sentinel zeros to
   `NaN` (Pima's `glucose/blood_pressure/skin_thickness/insulin/bmi == 0`), map
   the target to a binary `target` column, drop the raw target + duplicates.
2. **`split`**: one stratified 80/20 `train_test_split` on the cleaned columns.
3. **Impute** (fit on train only): median for numerics, most-frequent for
   categoricals.
4. **Winsorize** (fit on train only): clip numerics to `[Q1 − 1.5·IQR,
   Q3 + 1.5·IQR]`.
5. **Ordinal-encode** categoricals (fit on train only), `unknown_value = -1`.
6. **Engineer features** (see §3) on the now-clean columns.

Every fitted statistic (imputers, IQR bounds, encoder, feature order,
categorical-column indices) is stored in a `PreprocessArtifact`
(`src/artifacts.py` — its own module so `joblib` always records the class path
as `src.artifacts`, never `__main__`). `transform_new(artifact, raw_df)` replays
the exact same transforms on a live patient record, so a form submission is
encoded identically to the training data. `tests/test_preprocessing.py` checks
leakage-safety (bounds derived only from the imputed *train* split) and that
`transform_new` reproduces the training feature columns exactly.

---

## 3. Feature engineering — `src/feature_engineering.py`

One pure `DataFrame → DataFrame` function per disease, each feature grounded in a
clinical heuristic (not an arbitrary column combo):

- **Heart** — `rate_pressure_product` (myocardial O₂ demand),
  `heart_rate_reserve` (`(220 − age) − thalach`, chronotropic incompetence),
  `chol_age_ratio` (age-adjusted lipids).
- **Diabetes** — `glucose_bmi_interaction`, `insulin_glucose_ratio` (a
  descriptive HOMA-IR-style proxy).
- **CKD** — `comorbidity_count` = `htn + dm + cad` (expects those already 0/1).
- **Liver** — `ast_alt_ratio` (De Ritis ratio), `bilirubin_ratio` (`DB/TB`).

Pure functions ⇒ unit-tested against hand-computed values in
`tests/test_feature_engineering.py`.

---

## 4. Training — `src/train.py`

`train_disease(key, device="cpu", n_estimators=8, calibrate=True)`:

1. Load the processed train split and the preprocessor artifact.
2. **Fit TabPFN.** `TabPFNClassifier(categorical_features_indices=…,
   n_estimators=8, random_state=42).fit(X_train, y_train)`. TabPFN is a
   transformer pretrained once on millions of synthetic tasks; "fitting" just
   caches the training table, and prediction is a single forward pass of
   in-context Bayesian inference. No per-dataset hyperparameter search — the
   reason the project can run four independent clinical models without four
   tuning campaigns.
3. **Calibration pass** (`calibrate=True`, skip with `--no-calibrate`): a
   `CV_FOLDS`-way stratified out-of-fold loop (`src/uncertainty.py`,
   `oof_probabilities`) produces an honest held-out probability for every
   training row. From those:
   - `conformal_residuals` — sorted `|y − p_oof|` nonconformity scores.
   - `decision_threshold` — the probability cutoff maximising Youden's J
     (`youden_threshold`), replacing a blind 0.5.
   - `ood` — an `OODDetector` (§5) fit on the encoded training matrix.
4. **Save one bundle**: `models/{key}_tabpfn.joblib` =
   `{model, feature_order, decision_threshold, conformal_residuals,
   conformal_alpha, ood, oof_probabilities, …}`.

`main.py` orchestrates `download → preprocess → train → evaluate → explain` for
one or all diseases (`python main.py --disease heart`,
`python main.py --stages train,evaluate`, `--no-calibrate`, `--n-estimators`).

---

## 5. The uncertainty layer — `src/uncertainty.py`

Everything here is **computed once at training time and read from the bundle at
inference**, so serving stays one forward pass plus cheap arithmetic. It is
model-agnostic — it only needs `fit` / `predict_proba`, which is why the tests
can exercise it with `LogisticRegression`.

### 5a. Cross-conformal probability band

`conformal_halfwidth(residuals, alpha)` takes the `⌈(n+1)(1−α)⌉`-th sorted
residual; `probability_band(p̂, residuals, α)` returns
`[p̂ − h, p̂ + h]` clipped to `[0, 1]`, where `h` is a **single global
constant per disease** (the same for every patient). With `α = 0.10` the target is
90 % marginal coverage of the *binary label* by the band. The construction is the
K-fold analogue of the jackknife interval (Barber et al., 2021, Eq. 7), not the
CV+ variant that carries a proven guarantee, so coverage is approximate. Because
`h` does not depend on the patient, the band says nothing about how hard a
particular patient is; when the model is only moderately accurate `h` is large and
the band spans the midpoint, which carries no class information. It replaces the
old "confidence = |p − 0.5|".

### 5b. Learned decision threshold

`youden_threshold(oof_proba, y)` picks the ROC point maximising
`sensitivity + specificity − 1`, clamped to `[0.01, 0.99]`. `predict()` and
`predict_batch()` both use this bundle value for the binary label, so the
positive/negative call reflects each disease's class balance instead of a
one-size 0.5.

### 5c. Out-of-distribution / novelty detection

`OODDetector.fit(X)` shrinks a Gaussian with `LedoitWolf` (always invertible),
stores the mean, the precision matrix, and the sorted training Mahalanobis
distances. Per query: `distance(x)` is one quadratic form; `is_ood(x)` compares
it to the 97.5 % training-distance quantile; `novelty_score(x)` is the fraction
of training rows nearer the centroid (0 = typical, 1 = extreme). This matters
because the source cohorts are small and narrow (Pima = one sex/ethnicity;
Cleveland = one 1980s hospital) — most real inputs sit at the edge of or outside
what the model ever saw, and the app now says so instead of returning a
confident-looking number.

Covered by `tests/test_uncertainty.py` (13 cases: band monotonicity in α,
threshold sanity, extreme-point flagging, batch/scalar agreement, picklability).

---

## 6. Evaluation — `src/evaluate.py`

`evaluate_disease(key)` scores the held-out test split: accuracy, precision,
recall, F1, ROC-AUC, confusion matrix, full `classification_report` → written to
`reports/{key}_metrics.json`, plus PNGs for the confusion matrix, ROC curve
(with AUC) and precision-recall curve (with the prevalence baseline) into
`reports/figures/`. `evaluate_all()` also emits `reports/model_comparison.csv`.
The repo intentionally ships **no committed metric numbers** — they depend on
each user's own TabPFN weights, so you look at numbers from a run you triggered.

---

## 7. Explainability — `src/explainability.py`

TabPFN is a transformer, so SHAP's closed-form `TreeExplainer` / `LinearExplainer`
don't apply. `DiseaseExplainer` wraps SHAP's model-agnostic
`PermutationExplainer` around `model.predict_proba`, with the background set
summarised by k-means. `max_background` is a constructor arg:

- **Global** plots (`generate_global_explanations`, the Model Performance page)
  use the full default (25) → `{key}_shap_summary.png` / `_beeswarm.png`.
- **Interactive** per-patient explanations use `_INTERACTIVE_SHAP_BACKGROUND = 10`
  (set in `prediction.py`) — materially faster, ranking essentially unchanged.

`explain_instance(x)` returns the top-N signed contributors as a
JSON-serialisable list (feature, patient value, SHAP value,
`increases_risk` / `decreases_risk`).

---

## 8. Serving — `src/prediction.py`

`DiseasePredictor(disease_key)` loads the bundle + preprocessor + (lazily) the
explainer, and reads the uncertainty add-ons with `.get(...)` fallbacks so an
older bundle or the `LogisticRegression` test stand-in still works
(`decision_threshold → 0.5`, no band, no OOD flag).

**`predict(raw_record, explain=True, top_n=5)`** →

1. `transform_new` encodes the one raw row.
2. `model.predict_proba` → `probability` (one forward pass).
3. `probability_band(...)` → `probability_low` / `probability_high`.
4. `ood.is_ood` / `ood.novelty_score` → `out_of_distribution` / `novelty_score`.
5. binary label via `probability ≥ decision_threshold`; `risk_level` via the
   display bands.
6. SHAP explanation, **memoised** on `round(x, 4).tobytes()` in
   `self._explain_cache` (identical inputs = no recompute).

Returns a `PredictionResult` dataclass carrying all of the above (`confidence`
is kept as `max(p, 1−p)` for backward compatibility but is no longer surfaced as
the headline).

**`predict_batch(df)`** vectorises the same path over an uploaded CSV, appending
`probability`, `probability_low/high`, `predicted_label`, `risk_level`, and
`out_of_distribution` columns; SHAP is skipped for throughput.
`get_predictor(key)` is `lru_cache`d process-wide.

`tests/test_prediction.py` builds a real `DiseasePredictor` on a
`LogisticRegression` stand-in (both the plain bundle and a fully-calibrated one)
and checks the band ordering, threshold-respecting labels, batch columns, and
the explanation cache.

---

## 9. PDF report — `src/report.py`

`build_report_pdf(result) → bytes` (ReportLab) renders: title + timestamp;
**Prediction Summary** (outcome, probability, *N %* conformal band, decision
threshold, colour-coded risk level); a **novelty warning** paragraph when
`out_of_distribution`; a **Patient Input** table with `FieldSpec` labels + units
+ decoded category names; a **top contributing factors** table (humanised
feature names, patient value, direction, SHAP value); and the fixed academic
disclaimer. `save_report()` also drops a copy in `reports/generated/`.
`tests/test_report.py` asserts valid PDF bytes and that labels are humanised
(`"Chest Pain Type: Asymptomatic"`, not `"cp: 4"`).

---

## 10. History & monitoring — `src/history.py`

A local SQLite file (`reports/prediction_history.db`). `_connect()` creates the
table and idempotently `ALTER TABLE ... ADD COLUMN`s the newer fields
(`probability_low/high`, `decision_threshold`, `out_of_distribution`,
`novelty_score`) — no migration tool needed. `save_prediction`, `load_history`
(optionally filtered by disease, returns a DataFrame), `clear_history`. Backs
the dashboard's History page, including the out-of-distribution rate over time.

---

## 11. Dashboard — `app/streamlit_app.py` + `app/components.py` + `app/theme.py`

Five pages routed from the sidebar (disease selector + dark-mode toggle):

- **Predict** — `render_patient_form` builds one widget per `FieldSpec` inside an
  `st.form`; on submit, `predictor.predict()` runs inside a spinner, the result
  is stashed in `st.session_state` and `save_prediction`d. The result panel
  (`_render_result_panel`) is an **`st.fragment`**: a probability gauge with the
  conformal band shaded on the arc, a horizontal SHAP bar chart with
  **humanised** labels, the threshold/band caption, an all-contributions
  expander, and a **PDF download** whose bytes come from
  `_cached_report_pdf(result)` (keyed on `result.timestamp`). Because it's a
  fragment, toggling the expander or clicking download does a local rerun — it
  does **not** re-run TabPFN.
- **Batch Prediction** — CSV template download, upload, `predict_batch`, an
  out-of-distribution count warning, results table + risk histogram + export.
- **Model Performance** — metric cards + evaluation figures + global SHAP plot
  (metrics JSON read through an mtime-keyed `st.cache_data`).
- **Prediction History** — totals, avg probability, high-risk count,
  out-of-distribution count, risk distribution, probability-over-time line,
  full table, CSV export, clear.
- **About** — datasets, the uncertainty summary, limitations, disclaimer.

### Latency choices

| Cost | Before | Now |
|---|---|---|
| SHAP per prediction | recomputed every rerun | memoised per input in `DiseasePredictor`; interactive background 25 → 10 |
| PDF build | every rerun of the page | once per prediction (`st.cache_data` on timestamp) |
| Predictor load | `lru_cache` | `lru_cache` + `st.cache_resource` |
| Metrics JSON | read every rerun | `st.cache_data` keyed on file mtime |
| Result-panel widgets | full-app rerun | isolated `st.fragment` |
| Conformal band / OOD | — | precomputed at train time, O(d²) at inference |

`app/components.py` holds the reusable form / gauge / chart / summary widgets;
`app/theme.py` injects the CSS and the runtime light/dark override.

---

## 12. Tests & CI

`pytest tests/` — **81 tests** covering the config registry's internal
consistency, the feature-engineering math, the preprocessing pipeline
(leakage-safety, sentinel quirks), the uncertainty module, the prediction
wiring (plain + calibrated stand-in bundles), and PDF content. Model-dependent
tests swap in `LogisticRegression`, so **no TabPFN weights / network are needed
in CI**. `.github/workflows/ci.yml` runs `black --check` + `ruff` then the suite
on Python 3.11.

---

## 13. Deployment — Hugging Face Docker Space

Chosen flow: **pre-train locally, bundle the models, mirror to a Space.**

1. **Train once, with your token** (`TABPFN_TOKEN` in `.env`):
   `python main.py --stages preprocess,train,evaluate,explain`. This produces
   `models/*_preprocessor.joblib` (plain git blobs) and
   `models/*_tabpfn.joblib` (**Git LFS**, see `.gitattributes`).
2. **Commit** the model bundles + `reports/*_metrics.json` + `reports/figures/`.
   `.gitignore` was updated to allow `models/*.joblib`.
3. **`.github/workflows/deploy-hf-space.yml`** runs after CI is green on `main`
   (or on manual dispatch) and calls **`deploy/huggingface/sync_space.py`**,
   which `HfApi.upload_folder`s the repo (LFS models included, local cruft
   ignored) into the Space and overwrites its `README.md` with
   **`deploy/huggingface/README.md`** (the Space card: `sdk: docker`,
   `app_port: 8501`). Needs GitHub secret `HF_TOKEN` and variable `HF_SPACE`.
4. The Space builds the repo **`Dockerfile`** — now non-root (uid 1000, matching
   HF), `WORKDIR /home/user/app`, `TABPFN_MODEL_CACHE_DIR` set, an optional
   `--build-arg TABPFN_TOKEN` step to bake TabPFN's gated weights into the image
   for fully offline runtime. The container serves the same Streamlit app on
   8501. `docker-compose.yml` mounts were updated to the new WORKDIR so local
   Docker still works.
5. **Runtime**: the app only *loads* models. If a bundled model still triggers a
   one-time weight download on first predict, add `TABPFN_TOKEN` as a Space
   secret. The history SQLite resets on Space restart (fine for a public demo).

Full instructions: [`deploy/huggingface/README.md`](../deploy/huggingface/README.md).

---

## One-line mental model

> Config declares every column once → preprocessing makes a leakage-safe,
> reusable transform → TabPFN fits in one pass → a k-fold pass bakes a conformal
> band, a learned threshold and a novelty detector into the same bundle →
> `DiseasePredictor` is the one serving object (single + batch) → the Streamlit
> app and the PDF report are thin, cached views over it → CI runs it all without
> a TabPFN licence → a GitHub Action mirrors the trained bundle to a Hugging Face
> Docker Space.
