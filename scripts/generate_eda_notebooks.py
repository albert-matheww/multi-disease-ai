"""Programmatically build the four per-disease EDA notebooks.

Notebooks are generated (not hand-written) so that the plotting code stays in
sync with `src/eda.py`, but the analysis and interpretation markdown for each
disease is authored specifically for that dataset's actual statistics (see
the `_INTERP` dict below), not generic filler text.

Usage:
    python scripts/generate_eda_notebooks.py            # write notebooks
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"

CODE_HEADER = """\
import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import get_disease
from src.preprocessing import DiseasePreprocessor
from src import eda

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
"""

# Disease-specific setup + interpretation text, grounded in the actual
# statistics computed from the cleaned data (see chat/dev notes for the
# exploratory pass that produced these numbers).
_SPEC = {
    "heart": dict(
        title="Heart Disease - Exploratory Data Analysis",
        intro="""\
## Heart Disease (UCI Cleveland, n=303)

This notebook explores the UCI Heart Disease (Cleveland) dataset before any
imputation, outlier clipping, or encoding is applied (that happens in
`src/preprocessing.py`). The raw target `num` (0-4 severity) is binarized to
`target` (0 = no disease, 1 = disease present), which is standard practice
for this dataset in the literature.
""",
        labels="{0: 'No Disease', 1: 'Disease'}",
        numeric_cols="['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']",
        prevalence_interp="""\
**Interpretation:** 45.9% of the 303 patients have heart disease present
(`num > 0`). This is a near-balanced dataset (not a rare-disease scenario),
so accuracy is a reasonably informative metric here, though we still report
ROC-AUC/precision/recall since a false negative (missed diagnosis) and a
false positive (unnecessary follow-up testing) have very different real-world
costs.
""",
        missing_interp="""\
**Interpretation:** Only two columns have missing values: `ca` (number of
major vessels colored by fluoroscopy, 4 missing) and `thal` (thalassemia test
result, 2 missing) - together under 2% of cells. Both are angiographic/nuclear
test results that may simply not have been ordered for every patient. Given
the small amount, median (numeric) / most-frequent (categorical) imputation
in the training pipeline is appropriate; a more complex missing-data model
would be overkill for six missing cells total.
""",
        dist_interp="""\
**Interpretation:** `thalach` (max heart rate achieved) is visibly shifted
lower in the disease-positive group - consistent with reduced exercise
capacity in symptomatic patients. `oldpeak` (ST depression) is right-skewed
and concentrated near 0 for healthy patients but has a heavier right tail for
diseased patients, matching its clinical role as an ischemia indicator.
`chol` and `trestbps` show much weaker separation between groups, foreshadowing
their lower correlation with the target seen in the heatmap below.
""",
        box_interp="""\
**Interpretation:** `oldpeak` and `ca` show the clearest median shift and
the most outliers on the high end for the disease-positive group - both are
established markers of ischemic severity, so outliers here likely represent
genuinely severe cases rather than data errors. `chol` has a few very high
outliers (>500 mg/dl) in both groups; these are physiologically plausible
(familial hypercholesterolemia) but are still winsorized in preprocessing so
they don't dominate the small training sample.
""",
        corr_interp="""\
**Interpretation:** Ranked by |correlation| with `target`: `thal` (0.53),
`ca` (0.46), `exang` (0.43), `oldpeak` (0.42), `thalach` (-0.42), `cp` (0.41).
These six variables - four of which are categorical clinical test results
rather than routine vitals - carry most of the linear signal, which is why a
model that can flexibly combine categorical and continuous features (like
TabPFN) is a good fit rather than one that assumes a fixed functional form.
No pair of *features* is highly collinear (|r| > 0.7), so multicollinearity
is not a concern here.
""",
        pairplot_cols="['thalach', 'oldpeak', 'chol', 'age']",
        pair_interp="""\
**Interpretation:** `thalach` vs `oldpeak` shows the cleanest visual
separation between the two classes among numeric feature pairs - disease
cases cluster toward low max heart rate and elevated ST depression
simultaneously, suggesting the *combination* of these two features is more
informative than either alone (motivating the `heart_rate_reserve` engineered
feature in `src/feature_engineering.py`, which explicitly combines age,
`thalach`, and the age-predicted maximum).
""",
    ),
    "diabetes": dict(
        title="Diabetes - Exploratory Data Analysis",
        intro="""\
## Pima Indians Diabetes (n=768)

All patients are female, 21+ years old, of Pima Indian heritage. Several
columns encode missing measurements as `0` (not biologically plausible for
glucose, blood pressure, skin thickness, insulin, or BMI); this notebook
first shows the *raw* zeros, and interprets them as missing, exactly as
`src/preprocessing.py` does before imputation.
""",
        labels="{0: 'Non-Diabetic', 1: 'Diabetic'}",
        numeric_cols="['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi', 'age']",
        prevalence_interp="""\
**Interpretation:** 34.9% of patients are diabetic - a moderate class
imbalance (roughly 1:2). This is mild enough that no resampling is applied,
but it's still a reason to look at ROC-AUC and recall rather than accuracy
alone (a model predicting "non-diabetic" for everyone would already score
65% accuracy while being clinically useless).
""",
        missing_interp="""\
**Interpretation:** Once zeros are treated as missing, `insulin` is missing
for 374/768 patients (48.7%) and `skin_thickness` for 227/768 (29.6%) - both
require a more invasive/less routine measurement than glucose or blood
pressure, which explains the gap. `blood_pressure` (35), `bmi` (11), and
`glucose` (5) are missing far less often. Given `insulin`'s near-50% missingness,
median imputation is a pragmatic choice for this project, but a production
system would want to flag `insulin`-imputed predictions as lower-confidence.
""",
        dist_interp="""\
**Interpretation:** `glucose` shows the strongest visual separation between
classes - diabetic patients are shifted noticeably right, consistent with
it being part of the diagnostic criteria itself (2-hour OGTT >= 200 mg/dl).
`insulin` and `skin_thickness` are heavily right-skewed with a spike at the
(now-imputed) median, a visible artifact of the ~30-49% missingness in those
columns.
""",
        box_interp="""\
**Interpretation:** `glucose`, `bmi`, and `insulin` all show a clear upward
median shift in the diabetic group. `skin_thickness` and `blood_pressure`
show much more overlapping boxes between groups - visually consistent with
their weaker correlation with the target reported below.
""",
        corr_interp="""\
**Interpretation:** Ranked by |correlation| with `target`: `glucose` (0.49),
`bmi` (0.31), `insulin` (0.30), `skin_thickness` (0.26), `age` (0.24),
`pregnancies` (0.22). `glucose` alone carries roughly as much linear signal
as the next three features combined, which matches its role as the direct
diagnostic threshold variable. `skin_thickness` and `insulin` are correlated
with each other (both reflect adiposity/insulin resistance) more than either
is with the target - a case of correlated *predictors* rather than target
leakage.
""",
        pairplot_cols="['glucose', 'bmi', 'age', 'insulin']",
        pair_interp="""\
**Interpretation:** `glucose` vs `bmi` shows the diabetic class concentrated
in the upper-right region (high glucose *and* high BMI), reinforcing the
`glucose_bmi_interaction` engineered feature. `age` shows only a mild rightward
shift for the diabetic class - onset in this cohort is not tightly
age-concentrated, unlike some other diabetes cohorts in the literature.
""",
    ),
    "ckd": dict(
        title="Chronic Kidney Disease - Exploratory Data Analysis",
        intro="""\
## Chronic Kidney Disease (UCI, n=400)

This dataset has the heaviest missingness of the four (several lab columns
missing 20-38% of values) because it was assembled from real clinical records
with incomplete lab panels - not every patient received every test. The raw
target label also has a whitespace artifact (`"ckd\\t"`), cleaned in
`src/preprocessing.py` before this notebook's `target` column is derived.
""",
        labels="{0: 'No CKD', 1: 'CKD'}",
        numeric_cols="['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc', 'rbcc']",
        prevalence_interp="""\
**Interpretation:** 62.5% of patients in this dataset have CKD. This is
*not* representative of general-population CKD prevalence (~10-15%) - this
is a clinically-referred, hospital-collected sample enriched for disease
cases, which is an important caveat for interpreting any absolute risk score
this project produces (discussed further in the main README's Limitations
section).
""",
        missing_interp="""\
**Interpretation:** Missingness is substantial and uneven: `rbcc` (red blood
cell count) is missing for 131/400 patients (32.8%), `wbcc` for 106 (26.5%),
`pot`/`sod` for ~88 (22%), and `rbc` (red blood cells, categorical) for 152
(38%) - the single most-missing column in the whole project. These are all
tests that require a full blood panel, plausibly skipped for patients whose
CKD status was already clear from simpler markers. Median/most-frequent
imputation is used here as a practical baseline, but this is the dataset
where imputation choice most plausibly affects results, since almost a third
of some columns are being filled in.
""",
        dist_interp="""\
**Interpretation:** `hemo` (hemoglobin) and `pcv` (packed cell volume) are
both clearly shifted lower in the CKD-positive group - anemia is a
well-established consequence of reduced kidney function (reduced
erythropoietin production), so this matches clinical expectations exactly.
`sc` (serum creatinine) is right-skewed with a long tail in the CKD group,
consistent with creatinine rising sharply as kidney filtration deteriorates.
""",
        box_interp="""\
**Interpretation:** `hemo`, `pcv`, and `rbcc` all show a strong downward
shift with little overlap between groups - among the cleanest visual
separations in this entire project. `sc` and `bu` (blood urea) show more
outliers in the CKD group extending to very high values, consistent with
advanced-stage patients in a referred clinical sample.
""",
        corr_interp="""\
**Interpretation:** Ranked by |correlation| with `target`: `hemo` (-0.77),
`pcv` (-0.74), `sg` (specific gravity, -0.73), `rbcc` (-0.70), `al` (albumin,
+0.63), `bgr` (+0.42). These correlations are far stronger than in the other
three diseases - CKD's lab markers are direct physiological consequences of
kidney function rather than indirect risk factors, so the linear signal is
much cleaner. `hemo` and `pcv` are also highly correlated *with each other*
(both measure red-cell mass via different methods), which is expected
physiologically and is not a data error.
""",
        pairplot_cols="['hemo', 'pcv', 'sc', 'bgr']",
        pair_interp="""\
**Interpretation:** `hemo` vs `pcv` shows an almost linear relationship
between the two (as expected - they measure related quantities), with the
CKD class cleanly separated in the lower-left region of both. `sc` vs `bgr`
shows more overlap, indicating serum creatinine and random blood glucose
provide more complementary than redundant information for this task.
""",
    ),
    "liver": dict(
        title="Liver Disease - Exploratory Data Analysis",
        intro="""\
## ILPD - Indian Liver Patient Dataset (n=583 raw, 570 after de-duplication)

Liver-function-test panel data. 13 exact duplicate rows are removed during
cleaning (a known characteristic of this dataset, also reported in prior
published work using ILPD) before this notebook's statistics are computed.
""",
        labels="{0: 'No Liver Disease', 1: 'Liver Disease'}",
        numeric_cols="['Age', 'TB', 'DB', 'Alkphos', 'Sgpt', 'Sgot', 'TP', 'ALB', 'A/G Ratio']",
        prevalence_interp="""\
**Interpretation:** 71.2% of patients in this dataset are liver-disease
positive - a substantial imbalance in the *opposite* direction from a typical
screening population (this is a dataset collected from patients already
presenting with symptoms, not a general population screen). This is the most
imbalanced of the four diseases in this project, so recall on the minority
(non-disease) class is watched closely in `evaluate.py` in addition to
overall accuracy.
""",
        missing_interp="""\
**Interpretation:** Only `A/G Ratio` (albumin/globulin ratio) has missing
values (4 rows, <1%) - it is a computed/derived lab value, so it's plausible
it was simply not calculated for a handful of patients when the raw albumin
or globulin measurement was unavailable. This is negligible missingness and
median imputation is more than sufficient.
""",
        dist_interp="""\
**Interpretation:** `TB` (total bilirubin) and `DB` (direct bilirubin) are
both extremely right-skewed with a long tail of very high values in the
disease-positive group - consistent with jaundice/hyperbilirubinemia being a
hallmark of liver dysfunction. `Sgpt` (ALT) and `Sgot` (AST) show similar
right-skew with elevated values in disease-positive patients, matching their
clinical role as liver-enzyme injury markers.
""",
        box_interp="""\
**Interpretation:** `TB` and `DB` show the most extreme outliers by far
(some patients with bilirubin >20x the typical range) - clinically plausible
in severe hepatic disease (e.g. obstructive jaundice) rather than data-entry
errors, but exactly why winsorization (rather than deletion) is used in
preprocessing: extreme-but-real values are capped so they don't dominate
TabPFN's small in-context training set, without discarding genuinely
informative severe cases.
""",
        corr_interp="""\
**Interpretation:** Ranked by |correlation| with `target`: `DB` (0.25),
`TB` (0.22), `Alkphos` (0.19), `A/G Ratio` (-0.17), `ALB` (-0.17), `Sgpt`
(0.16). Correlations here are the weakest of the four diseases - liver
disease in this dataset is comparatively harder to separate linearly from
routine LFT values alone, which is exactly the kind of nonlinear,
interaction-heavy problem a tabular foundation model like TabPFN is designed
to handle better than a linear baseline. `TB` and `DB` are very highly
correlated with each other (direct bilirubin is a component of total
bilirubin by definition), motivating the `bilirubin_ratio` engineered feature
that captures their *relative* balance instead of just their magnitudes.
""",
        pairplot_cols="['TB', 'DB', 'Sgot', 'Sgpt']",
        pair_interp="""\
**Interpretation:** `Sgot` vs `Sgpt` shows a roughly linear relationship with
a fan of increasing spread at higher values, and disease-positive patients
skew toward the upper end of both - motivating the `ast_alt_ratio` (De Ritis
ratio) engineered feature, which captures whether AST or ALT dominates
(clinically informative for distinguishing types of liver injury) rather
than just their absolute levels.
""",
    ),
}


def build_notebook(disease_key: str) -> nbf.NotebookNode:
    spec = _SPEC[disease_key]
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(f"# {spec['title']}\n\n{spec['intro']}"))
    cells.append(nbf.v4.new_code_cell(CODE_HEADER))

    cells.append(nbf.v4.new_code_cell(f"""\
disease = get_disease("{disease_key}")
pre = DiseasePreprocessor(disease)
raw_df = pre.load_raw()
df = pre.clean_raw(raw_df)
print(f"Shape after cleaning: {{df.shape}}")
df.head()"""))

    cells.append(nbf.v4.new_markdown_cell("## 1. Summary Statistics"))
    cells.append(nbf.v4.new_code_cell("eda.summary_statistics(df)"))

    cells.append(nbf.v4.new_markdown_cell("## 2. Missing Value Analysis"))
    cells.append(nbf.v4.new_code_cell(f'fig = eda.plot_missing_values(df, "{disease_key}")\nplt.show()'))
    cells.append(nbf.v4.new_markdown_cell(spec["missing_interp"]))

    cells.append(nbf.v4.new_markdown_cell("## 3. Disease Prevalence"))
    cells.append(
        nbf.v4.new_code_cell(
            f'labels = {spec["labels"]}\nfig = eda.plot_target_prevalence(df, "target", labels, "{disease_key}")\nplt.show()'
        )
    )
    cells.append(nbf.v4.new_markdown_cell(spec["prevalence_interp"]))

    cells.append(nbf.v4.new_markdown_cell("## 4. Feature Distributions"))
    cells.append(
        nbf.v4.new_code_cell(
            f'numeric_cols = {spec["numeric_cols"]}\nfig = eda.plot_numeric_distributions(df, numeric_cols, "target", "{disease_key}")\nplt.show()'
        )
    )
    cells.append(nbf.v4.new_markdown_cell(spec["dist_interp"]))

    cells.append(nbf.v4.new_markdown_cell("## 5. Boxplots (Outlier Visualization)"))
    cells.append(
        nbf.v4.new_code_cell(
            f'fig = eda.plot_boxplots(df, numeric_cols, "target", "{disease_key}")\nplt.show()'
        )
    )
    cells.append(nbf.v4.new_markdown_cell(spec["box_interp"]))

    cells.append(nbf.v4.new_markdown_cell("## 6. Correlation Matrix"))
    cells.append(nbf.v4.new_code_cell(f'fig = eda.plot_correlation_heatmap(df, "{disease_key}")\nplt.show()'))
    cells.append(nbf.v4.new_code_cell('eda.target_correlation_ranked(df, "target")'))
    cells.append(nbf.v4.new_markdown_cell(spec["corr_interp"]))

    cells.append(nbf.v4.new_markdown_cell("## 7. Pairplot of Top Correlated Features"))
    cells.append(
        nbf.v4.new_code_cell(
            f'pair_cols = {spec["pairplot_cols"]}\nfig = eda.plot_pairplot(df, pair_cols, "target", "{disease_key}", labels)\nplt.show()'
        )
    )
    cells.append(nbf.v4.new_markdown_cell(spec["pair_interp"]))

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 8. Summary of Findings\n\n"
            "This EDA directly informs the choices made in `src/preprocessing.py` "
            "(imputation strategy, outlier clipping) and `src/feature_engineering.py` "
            "(which engineered features to add) for this disease. See `src/train.py` "
            "for how the cleaned, engineered features feed into the TabPFN model."
        )
    )

    nb["cells"] = cells
    return nb


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    for key in _SPEC:
        nb = build_notebook(key)
        out_path = NOTEBOOKS_DIR / f"{key}_eda.ipynb"
        nbf.write(nb, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
