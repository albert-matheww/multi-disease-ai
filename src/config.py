"""Central configuration for MultiDiseaseAI.

Defines filesystem paths and a declarative registry (`DISEASES`) describing
every disease pipeline: where its raw data lives, how its target is encoded,
which columns are numeric vs. categorical, and the input specification used
to render Streamlit forms and validate patient records. Keeping this in one
place means `preprocessing.py`, `train.py`, `evaluate.py`, `prediction.py`
and the Streamlit app all agree on column names and encodings without
duplicating logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
GENERATED_REPORTS_DIR = REPORTS_DIR / "generated"

for _d in (PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, GENERATED_REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Risk-level thresholds applied to the predicted probability of disease.
RISK_THRESHOLDS = {"low": 0.33, "moderate": 0.66}


@dataclass(frozen=True)
class FieldSpec:
    """Describes one model input feature for validation and UI form rendering."""

    name: str
    label: str
    kind: str  # "numeric" | "categorical"
    dtype: str = "float"  # "float" | "int" (numeric only)
    min_value: float | None = None
    max_value: float | None = None
    default: float | str | None = None
    step: float | None = None
    unit: str = ""
    help_text: str = ""
    categories: dict[str, int] | None = None  # label -> encoded value (categorical only)


@dataclass(frozen=True)
class DiseaseConfig:
    key: str
    display_name: str
    icon: str
    description: str
    raw_filename: str
    target_column: str
    positive_label: str
    negative_label: str
    target_mapper: Callable[[object], int]
    numeric_features: list[str]
    categorical_features: list[str]
    drop_columns: list[str] = field(default_factory=list)
    fields: list[FieldSpec] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    @property
    def raw_path(self) -> Path:
        return RAW_DATA_DIR / self.raw_filename

    @property
    def processed_train_path(self) -> Path:
        return PROCESSED_DATA_DIR / f"{self.key}_train.csv"

    @property
    def processed_test_path(self) -> Path:
        return PROCESSED_DATA_DIR / f"{self.key}_test.csv"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / f"{self.key}_tabpfn.joblib"

    @property
    def preprocessor_path(self) -> Path:
        return MODELS_DIR / f"{self.key}_preprocessor.joblib"

    @property
    def metrics_path(self) -> Path:
        return REPORTS_DIR / f"{self.key}_metrics.json"

    def field_names(self) -> list[str]:
        """Raw input field names (pre-engineering) - what a user/CSV must supply.

        NOT the same as `numeric_features + categorical_features`, which also
        includes engineered feature names computed automatically by
        `feature_engineering.py` and must not be requested from the user.
        """
        return [f.name for f in self.fields]


# --------------------------------------------------------------------------- #
# Heart Disease
# --------------------------------------------------------------------------- #
HEART = DiseaseConfig(
    key="heart",
    display_name="Heart Disease",
    icon="\U0001fac0",
    description=(
        "Predicts the presence of heart disease from clinical and exercise-test "
        "attributes (UCI Cleveland dataset)."
    ),
    raw_filename="heart_disease.csv",
    target_column="num",
    positive_label="Heart Disease Present",
    negative_label="No Heart Disease",
    target_mapper=lambda v: 1 if float(v) > 0 else 0,
    numeric_features=[
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak",
        "ca",
        "rate_pressure_product",
        "heart_rate_reserve",
        "chol_age_ratio",
    ],
    categorical_features=["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"],
    fields=[
        FieldSpec("age", "Age", "numeric", "int", 1, 120, 55, 1, "years"),
        FieldSpec("sex", "Sex", "categorical", categories={"Female": 0, "Male": 1}),
        FieldSpec(
            "cp",
            "Chest Pain Type",
            "categorical",
            categories={
                "Typical Angina": 1,
                "Atypical Angina": 2,
                "Non-anginal Pain": 3,
                "Asymptomatic": 4,
            },
        ),
        FieldSpec("trestbps", "Resting Blood Pressure", "numeric", "int", 60, 250, 130, 1, "mm Hg"),
        FieldSpec("chol", "Serum Cholesterol", "numeric", "int", 80, 700, 240, 1, "mg/dl"),
        FieldSpec("fbs", "Fasting Blood Sugar > 120 mg/dl", "categorical", categories={"No": 0, "Yes": 1}),
        FieldSpec(
            "restecg",
            "Resting ECG",
            "categorical",
            categories={"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2},
        ),
        FieldSpec("thalach", "Max Heart Rate Achieved", "numeric", "int", 60, 220, 150, 1, "bpm"),
        FieldSpec("exang", "Exercise-Induced Angina", "categorical", categories={"No": 0, "Yes": 1}),
        FieldSpec("oldpeak", "ST Depression (Exercise vs Rest)", "numeric", "float", 0.0, 7.0, 1.0, 0.1),
        FieldSpec(
            "slope",
            "Slope of Peak Exercise ST Segment",
            "categorical",
            categories={"Upsloping": 1, "Flat": 2, "Downsloping": 3},
        ),
        FieldSpec("ca", "Major Vessels Colored by Fluoroscopy", "numeric", "int", 0, 3, 0, 1),
        FieldSpec(
            "thal",
            "Thalassemia",
            "categorical",
            categories={"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7},
        ),
    ],
    references=[
        "Janosi, Steinbrunn, Pfisterer & Detrano (1988). Heart Disease. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X"
    ],
)

# --------------------------------------------------------------------------- #
# Diabetes (Pima Indians)
# --------------------------------------------------------------------------- #
DIABETES = DiseaseConfig(
    key="diabetes",
    display_name="Diabetes",
    icon="\U0001f489",
    description=(
        "Predicts the presence of diabetes from metabolic and anthropometric "
        "measurements (Pima Indians Diabetes dataset)."
    ),
    raw_filename="diabetes.csv",
    target_column="outcome",
    positive_label="Diabetic",
    negative_label="Non-Diabetic",
    target_mapper=lambda v: int(v),
    numeric_features=[
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree_function",
        "age",
        "glucose_bmi_interaction",
        "insulin_glucose_ratio",
    ],
    categorical_features=[],
    fields=[
        FieldSpec("pregnancies", "Number of Pregnancies", "numeric", "int", 0, 20, 1, 1),
        FieldSpec("glucose", "Plasma Glucose Concentration", "numeric", "int", 40, 300, 120, 1, "mg/dl"),
        FieldSpec("blood_pressure", "Diastolic Blood Pressure", "numeric", "int", 30, 140, 70, 1, "mm Hg"),
        FieldSpec("skin_thickness", "Triceps Skin-Fold Thickness", "numeric", "int", 0, 100, 20, 1, "mm"),
        FieldSpec("insulin", "2-Hour Serum Insulin", "numeric", "int", 0, 900, 80, 1, "mu U/ml"),
        FieldSpec("bmi", "Body Mass Index", "numeric", "float", 10.0, 70.0, 28.0, 0.1, "kg/m^2"),
        FieldSpec(
            "diabetes_pedigree_function",
            "Diabetes Pedigree Function",
            "numeric",
            "float",
            0.0,
            3.0,
            0.5,
            0.01,
            help_text="Genetic risk score based on family history",
        ),
        FieldSpec("age", "Age", "numeric", "int", 1, 120, 33, 1, "years"),
    ],
    references=[
        "Smith, Everhart, Dickson, Knowler & Johannes (1988). Using the ADAP "
        "learning algorithm to forecast the onset of diabetes mellitus. "
        "Proc. Symposium on Computer Applications and Medical Care."
    ],
)

# --------------------------------------------------------------------------- #
# Chronic Kidney Disease
# --------------------------------------------------------------------------- #
CKD = DiseaseConfig(
    key="ckd",
    display_name="Chronic Kidney Disease",
    icon="\U0001fa78",
    description=(
        "Predicts chronic kidney disease from urinalysis, blood chemistry, "
        "and clinical signs (UCI CKD dataset)."
    ),
    raw_filename="ckd.csv",
    target_column="class",
    positive_label="CKD Present",
    negative_label="No CKD",
    target_mapper=lambda v: 1 if str(v).strip() == "ckd" else 0,
    numeric_features=[
        "age",
        "bp",
        "bgr",
        "bu",
        "sc",
        "sod",
        "pot",
        "hemo",
        "pcv",
        "wbcc",
        "rbcc",
        "comorbidity_count",
    ],
    categorical_features=[
        "sg",
        "al",
        "su",
        "rbc",
        "pc",
        "pcc",
        "ba",
        "htn",
        "dm",
        "cad",
        "appet",
        "pe",
        "ane",
    ],
    fields=[
        FieldSpec("age", "Age", "numeric", "int", 1, 120, 50, 1, "years"),
        FieldSpec("bp", "Blood Pressure", "numeric", "int", 40, 200, 80, 1, "mm/Hg"),
        FieldSpec(
            "sg",
            "Specific Gravity",
            "categorical",
            categories={"1.005": 1.005, "1.010": 1.010, "1.015": 1.015, "1.020": 1.020, "1.025": 1.025},
        ),
        FieldSpec("al", "Albumin", "categorical", categories={str(i): float(i) for i in range(6)}),
        FieldSpec("su", "Sugar", "categorical", categories={str(i): float(i) for i in range(6)}),
        FieldSpec(
            "rbc", "Red Blood Cells", "categorical", categories={"Normal": "normal", "Abnormal": "abnormal"}
        ),
        FieldSpec("pc", "Pus Cell", "categorical", categories={"Normal": "normal", "Abnormal": "abnormal"}),
        FieldSpec(
            "pcc",
            "Pus Cell Clumps",
            "categorical",
            categories={"Not Present": "notpresent", "Present": "present"},
        ),
        FieldSpec(
            "ba", "Bacteria", "categorical", categories={"Not Present": "notpresent", "Present": "present"}
        ),
        FieldSpec("bgr", "Blood Glucose Random", "numeric", "int", 20, 500, 120, 1, "mgs/dl"),
        FieldSpec("bu", "Blood Urea", "numeric", "int", 1, 400, 40, 1, "mgs/dl"),
        FieldSpec("sc", "Serum Creatinine", "numeric", "float", 0.1, 20.0, 1.2, 0.1, "mgs/dl"),
        FieldSpec("sod", "Sodium", "numeric", "float", 100, 160, 138, 1, "mEq/L"),
        FieldSpec("pot", "Potassium", "numeric", "float", 2.0, 10.0, 4.5, 0.1, "mEq/L"),
        FieldSpec("hemo", "Hemoglobin", "numeric", "float", 3.0, 18.0, 13.0, 0.1, "gms"),
        FieldSpec("pcv", "Packed Cell Volume", "numeric", "int", 10, 60, 40, 1),
        FieldSpec("wbcc", "White Blood Cell Count", "numeric", "int", 2000, 25000, 8000, 100, "cells/cmm"),
        FieldSpec("rbcc", "Red Blood Cell Count", "numeric", "float", 2.0, 7.0, 4.7, 0.1, "millions/cmm"),
        FieldSpec("htn", "Hypertension", "categorical", categories={"No": "no", "Yes": "yes"}),
        FieldSpec("dm", "Diabetes Mellitus", "categorical", categories={"No": "no", "Yes": "yes"}),
        FieldSpec("cad", "Coronary Artery Disease", "categorical", categories={"No": "no", "Yes": "yes"}),
        FieldSpec("appet", "Appetite", "categorical", categories={"Good": "good", "Poor": "poor"}),
        FieldSpec("pe", "Pedal Edema", "categorical", categories={"No": "no", "Yes": "yes"}),
        FieldSpec("ane", "Anemia", "categorical", categories={"No": "no", "Yes": "yes"}),
    ],
    references=[
        "Rubini, Soundarapandian & Eswaran (2015). Chronic Kidney Disease. "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C5G020"
    ],
)

# --------------------------------------------------------------------------- #
# Liver Disease (ILPD)
# --------------------------------------------------------------------------- #
LIVER = DiseaseConfig(
    key="liver",
    display_name="Liver Disease",
    icon="\U0001fac1",
    description=(
        "Predicts liver disease from liver-function-test panel results "
        "(Indian Liver Patient Dataset, ILPD)."
    ),
    raw_filename="liver_disease.csv",
    target_column="Selector",
    positive_label="Liver Disease Present",
    negative_label="No Liver Disease",
    target_mapper=lambda v: 1 if int(v) == 1 else 0,
    numeric_features=[
        "Age",
        "TB",
        "DB",
        "Alkphos",
        "Sgpt",
        "Sgot",
        "TP",
        "ALB",
        "A/G Ratio",
        "ast_alt_ratio",
        "bilirubin_ratio",
    ],
    categorical_features=["Gender"],
    fields=[
        FieldSpec("Age", "Age", "numeric", "int", 1, 120, 45, 1, "years"),
        FieldSpec("Gender", "Gender", "categorical", categories={"Female": "Female", "Male": "Male"}),
        FieldSpec("TB", "Total Bilirubin", "numeric", "float", 0.1, 75.0, 1.0, 0.1, "mg/dl"),
        FieldSpec("DB", "Direct Bilirubin", "numeric", "float", 0.0, 20.0, 0.3, 0.1, "mg/dl"),
        FieldSpec("Alkphos", "Alkaline Phosphotase", "numeric", "int", 60, 2200, 290, 1, "IU/L"),
        FieldSpec("Sgpt", "Alamine Aminotransferase (ALT)", "numeric", "int", 5, 2000, 40, 1, "IU/L"),
        FieldSpec("Sgot", "Aspartate Aminotransferase (AST)", "numeric", "int", 5, 5000, 45, 1, "IU/L"),
        FieldSpec("TP", "Total Proteins", "numeric", "float", 2.0, 10.0, 6.8, 0.1, "g/dl"),
        FieldSpec("ALB", "Albumin", "numeric", "float", 0.5, 6.0, 3.2, 0.1, "g/dl"),
        FieldSpec("A/G Ratio", "Albumin/Globulin Ratio", "numeric", "float", 0.1, 3.0, 0.9, 0.01),
    ],
    references=[
        "Ramana & Venkateswarlu (2012). ILPD (Indian Liver Patient Dataset). "
        "UCI Machine Learning Repository. https://doi.org/10.24432/C5D02C"
    ],
)

DISEASES: dict[str, DiseaseConfig] = {
    HEART.key: HEART,
    DIABETES.key: DIABETES,
    CKD.key: CKD,
    LIVER.key: LIVER,
}


def get_disease(key: str) -> DiseaseConfig:
    try:
        return DISEASES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown disease key '{key}'. Available: {list(DISEASES)}") from exc
