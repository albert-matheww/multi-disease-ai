"""Generate a downloadable PDF prediction report for one patient.

Used by the Streamlit app's "Download Report" button and available
standalone for scripting/batch report generation. Produces a single PDF
containing: patient inputs, prediction, probability, risk category, top
explanation contributors, and a generation timestamp - everything required
by the assignment's "Report Generation" section.
"""

from __future__ import annotations

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.config import GENERATED_REPORTS_DIR, get_disease, humanize_feature_name
from src.prediction import PredictionResult

_RISK_COLORS = {
    "Low": colors.HexColor("#2e7d32"),
    "Moderate": colors.HexColor("#f9a825"),
    "High": colors.HexColor("#c62828"),
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontSize=20, spaceAfter=4),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontSize=11, textColor=colors.grey
        ),
        "heading": ParagraphStyle("SectionHeading", parent=base["Heading2"], spaceBefore=16, spaceAfter=8),
        "body": base["BodyText"],
        "disclaimer": ParagraphStyle(
            "Disclaimer", parent=base["Normal"], fontSize=8, textColor=colors.grey, spaceBefore=18
        ),
    }


def _humanize_patient_input(disease_key: str, patient_input: dict) -> list[tuple[str, str]]:
    """Map raw field names/values to their human-readable label/category text."""
    disease = get_disease(disease_key)
    field_by_name = {f.name: f for f in disease.fields}
    rows = []
    for raw_name, raw_value in patient_input.items():
        field = field_by_name.get(raw_name)
        if field is None:
            rows.append((raw_name, str(raw_value)))
            continue
        display_value = str(raw_value)
        if field.kind == "categorical" and field.categories:
            for cat_label, cat_raw in field.categories.items():
                if cat_raw == raw_value or str(cat_raw) == str(raw_value):
                    display_value = cat_label
                    break
        elif field.unit:
            display_value = f"{raw_value} {field.unit}"
        rows.append((field.label, display_value))
    return rows


def _humanize_feature_name(disease_key: str, feature_name: str) -> str:
    """Back-compat shim; canonical implementation lives in `src.config`."""
    return humanize_feature_name(disease_key, feature_name)


def build_report_pdf(result: PredictionResult) -> bytes:
    """Render `result` to a PDF and return the raw bytes (for `st.download_button`)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
    )
    styles = _styles()
    story = []

    story.append(Paragraph("MultiDiseaseAI - Prediction Report", styles["title"]))
    story.append(Paragraph(f"{result.disease_display_name} Risk Assessment", styles["subtitle"]))
    story.append(Spacer(1, 6))
    generated_at = datetime.fromisoformat(result.timestamp).strftime("%Y-%m-%d %H:%M:%S UTC")
    story.append(Paragraph(f"Generated: {generated_at}", styles["subtitle"]))

    # --- Prediction summary -----------------------------------------
    story.append(Paragraph("Prediction Summary", styles["heading"]))
    risk_color = _RISK_COLORS.get(result.risk_level, colors.black)
    band_pct = int(round((1 - result.band_alpha) * 100))
    summary_data = [
        ["Predicted Outcome", result.predicted_label],
        ["Probability of Disease", f"{result.probability:.1%}"],
        [
            f"{band_pct}% Conformal Band",
            f"{result.probability_low:.1%} – {result.probability_high:.1%}",
        ],
        ["Decision Threshold", f"{result.decision_threshold:.0%}"],
        ["Risk Level", result.risk_level],
    ]
    risk_row = len(summary_data) - 1
    summary_table = Table(summary_data, colWidths=[2.3 * inch, 3.5 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f5f5")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.white),
                ("TEXTCOLOR", (1, risk_row), (1, risk_row), risk_color),
                ("FONTNAME", (1, risk_row), (1, risk_row), "Helvetica-Bold"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_table)

    if result.out_of_distribution:
        story.append(Spacer(1, 8))
        story.append(
            Paragraph(
                f"&#9888; <b>Novelty warning.</b> This record sits outside "
                f"{result.novelty_score:.0%} of the training cohort on a "
                "robust-Mahalanobis distance, i.e. it is unlike the data this model "
                "learned from. Treat the probability above as low-confidence and "
                "defer to clinical judgement.",
                styles["body"],
            )
        )

    # --- Patient input ------------------------------------------------
    story.append(Paragraph("Patient Input", styles["heading"]))
    humanized = _humanize_patient_input(result.disease_key, result.patient_input)
    input_rows = [["Field", "Value"]] + [[label, value] for label, value in humanized]
    input_table = Table(input_rows, colWidths=[2.8 * inch, 3.0 * inch])
    input_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(input_table)

    # --- Explanation ----------------------------------------------------
    if result.top_contributors:
        story.append(Paragraph("Why This Prediction? (Top Contributing Factors)", styles["heading"]))
        exp_rows = [["Feature", "Patient Value", "Effect", "SHAP Value"]]
        for c in result.top_contributors:
            effect = "Increases Risk" if c["direction"] == "increases_risk" else "Decreases Risk"
            feature_label = _humanize_feature_name(result.disease_key, c["feature"])
            exp_rows.append([feature_label, f"{c['value']:.3g}", effect, f"{c['shap_value']:+.3f}"])
        exp_table = Table(exp_rows, colWidths=[1.9 * inch, 1.3 * inch, 1.5 * inch, 1.1 * inch])
        exp_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37474f")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(exp_table)
        story.append(
            Paragraph(
                "SHAP values estimate each feature's contribution to the predicted probability "
                "relative to the model's average prediction over the training population. "
                "Positive values push the prediction toward disease-present; negative values push "
                "it toward disease-absent.",
                styles["body"],
            )
        )

    story.append(
        Paragraph(
            "Disclaimer: This report is generated by an academic demonstration project (MultiDiseaseAI) "
            "using TabPFN trained on small, publicly available research datasets. It is NOT a validated "
            "clinical diagnostic tool and must not be used to make real medical decisions. Consult a "
            "qualified healthcare professional for any medical concerns.",
            styles["disclaimer"],
        )
    )

    doc.build(story)
    return buffer.getvalue()


def save_report(result: PredictionResult) -> str:
    """Build the PDF and also persist a copy under reports/generated/ (prediction history)."""
    pdf_bytes = build_report_pdf(result)
    safe_ts = result.timestamp.replace(":", "-")
    filename = f"{result.disease_key}_{safe_ts}.pdf"
    path = GENERATED_REPORTS_DIR / filename
    path.write_bytes(pdf_bytes)
    return str(path)
