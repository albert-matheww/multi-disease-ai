"""Tests for PDF report generation in `src/report.py`."""

from __future__ import annotations

import io
from datetime import datetime, timezone

from pypdf import PdfReader

from src.prediction import PredictionResult
from src.report import build_report_pdf


def _sample_result() -> PredictionResult:
    return PredictionResult(
        disease_key="heart",
        disease_display_name="Heart Disease",
        probability=0.69,
        predicted_label="Heart Disease Present",
        risk_level="High",
        confidence=0.69,
        top_contributors=[
            {"feature": "thal", "value": 6.0, "shap_value": 0.14, "direction": "increases_risk"},
            {"feature": "ca", "value": 0.0, "shap_value": -0.13, "direction": "decreases_risk"},
        ],
        patient_input={"age": 63, "sex": 1, "cp": 4},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def test_build_report_pdf_produces_valid_pdf_bytes():
    pdf_bytes = build_report_pdf(_sample_result())
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 1000


def test_report_humanizes_patient_field_labels():
    """Regression check: the PDF should show 'Chest Pain Type: Asymptomatic',
    not the raw internal name/code 'cp: 4'."""
    pdf_bytes = build_report_pdf(_sample_result())
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = "\n".join(page.extract_text() for page in reader.pages)

    assert "Age" in text
    assert "Sex" in text
    assert "Male" in text  # sex=1 -> humanized category label, not the raw code
    assert "Chest Pain Type" in text
    assert "Asymptomatic" in text  # cp=4 -> humanized category label
    # Raw internal field/category codes must not leak into the report.
    assert "cp:" not in text.lower()
