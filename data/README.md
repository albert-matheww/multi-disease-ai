# Datasets

All raw data is fetched programmatically by [`scripts/download_data.py`](../scripts/download_data.py)
into `data/raw/`. Nothing under `data/raw/` or `data/processed/` is hand-edited — regenerate
at any time with:

```bash
python scripts/download_data.py
python -m src.preprocessing
```

## 1. Heart Disease (UCI ML Repository, id=45)

- **Source**: Cleveland Clinic Foundation subset of the UCI Heart Disease dataset,
  accessed via [`ucimlrepo`](https://pypi.org/project/ucimlrepo/) (`fetch_ucirepo(id=45)`).
- **Citation**: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988).
  Heart Disease [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C52P4X
- **License**: Available for academic/research use under the UCI Machine Learning
  Repository's open redistribution terms.
- **Rows / Columns**: 303 rows x 13 features + 1 target (`num`).
- **Target**: `num` (0 = no disease, 1-4 = increasing severity). We binarize to
  `0` = no heart disease, `1` = heart disease present (`num > 0`), which is the
  standard practice for this dataset in the literature.
- **Missing values**: `ca` (4 rows), `thal` (2 rows) — encoded as NaN.
- **Key features**: age, sex, chest pain type (`cp`), resting blood pressure
  (`trestbps`), cholesterol (`chol`), fasting blood sugar (`fbs`), resting ECG
  (`restecg`), max heart rate (`thalach`), exercise-induced angina (`exang`),
  ST depression (`oldpeak`), slope of peak exercise ST segment (`slope`),
  number of major vessels colored by fluoroscopy (`ca`), thalassemia (`thal`).

## 2. Pima Indians Diabetes

- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases
  (NIDDK), redistributed as a public-domain CSV mirror
  (`jbrownlee/Datasets` on GitHub — a commonly used, unmodified copy of the
  original UCI/NIDDK data).
- **License**: Public domain (US government-funded research data).
- **Rows / Columns**: 768 rows x 8 features + 1 target (`outcome`).
- **Population**: All patients are female, at least 21 years old, of Pima
  Indian heritage, near Phoenix, Arizona.
- **Target**: `outcome` (0 = no diabetes, 1 = diabetes, per WHO criteria of
  2-hour post-load plasma glucose >= 200 mg/dl).
- **Data quality note**: `glucose`, `blood_pressure`, `skin_thickness`,
  `insulin`, and `bmi` use `0` as a placeholder for missing measurements
  (a value of 0 is not biologically plausible for these fields). These are
  converted to `NaN` during preprocessing before imputation.
- **Key features**: number of pregnancies, plasma glucose concentration,
  diastolic blood pressure, triceps skin-fold thickness, 2-hour serum
  insulin, BMI, diabetes pedigree function (genetic risk score), age.

## 3. Chronic Kidney Disease (UCI ML Repository, id=336)

- **Source**: UCI Machine Learning Repository, accessed via `ucimlrepo`
  (`fetch_ucirepo(id=336)`).
- **Citation**: Rubini, L., Soundarapandian, P., & Eswaran, P. (2015).
  Chronic Kidney Disease [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5G020
- **License**: Available for academic/research use under UCI's open
  redistribution terms.
- **Rows / Columns**: 400 rows x 24 features + 1 target (`class`).
- **Target**: `class` (`ckd` vs `notckd`). One raw label has a trailing tab
  character (`"ckd\t"`), stripped during preprocessing.
- **Missing values**: substantial missingness across many columns (e.g. `rbcc`
  131 missing, `wbcc` 106 missing, `pot`/`sod` ~88 missing) — this dataset was
  collected from real clinical records with incomplete lab panels, so robust
  imputation is essential.
- **Key features**: a mix of numeric lab values (blood pressure, blood glucose,
  blood urea, serum creatinine, sodium, potassium, hemoglobin, packed cell
  volume, white/red blood cell counts) and categorical/binary clinical signs
  (red blood cells, pus cells, hypertension, diabetes mellitus, coronary
  artery disease, appetite, pedal edema, anemia).

## 4. ILPD - Indian Liver Patient Dataset (UCI ML Repository, id=225)

- **Source**: UCI Machine Learning Repository, accessed via `ucimlrepo`
  (`fetch_ucirepo(id=225)`).
- **Citation**: Ramana, B., & Venkateswarlu, N. (2012). ILPD (Indian Liver
  Patient Dataset) [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5D02C
- **License**: Available for academic/research use under UCI's open
  redistribution terms.
- **Rows / Columns**: 583 rows x 10 features + 1 target (`Selector`).
- **Target**: `Selector` (1 = liver patient, 2 = non-liver patient in the raw
  data). Remapped to `1` = liver disease, `0` = no liver disease.
- **Missing values**: `A/G Ratio` (4 rows).
- **Key features**: age, gender, total/direct bilirubin, alkaline phosphotase,
  alamine/aspartate aminotransferase (liver enzymes), total proteins, albumin,
  albumin/globulin ratio.

## Ethical & Usage Notes

These datasets are small, decades-old, single-institution or single-population
samples (e.g. Pima dataset is restricted to one ethnic group and sex; Cleveland
heart data is from one US hospital in the 1980s). They are **suitable for
demonstrating a modeling methodology and foundation-model workflow**, but the
resulting models are **not validated for real clinical deployment** — see
`reports/` and the main `README.md` "Limitations" section for a full discussion
of dataset bias, generalizability, and the gap between this project and a
clinically-safe product.
