"""Reusable Streamlit UI components shared across pages in `streamlit_app.py`.

Kept separate from `streamlit_app.py` so the same patient-input-form logic,
probability gauge, and SHAP contribution chart aren't duplicated across the
Predict / Batch / History pages.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.config import DiseaseConfig
from src.prediction import PredictionResult


def _field_widget_key(disease_key: str, field_name: str) -> str:
    return f"{disease_key}__{field_name}"


def render_patient_form(disease: DiseaseConfig) -> dict | None:
    """Render one input widget per `FieldSpec` inside a form; returns the raw
    patient record dict (categorical labels already mapped to raw values)
    when the user submits, otherwise `None`.
    """
    with st.form(key=f"form_{disease.key}"):
        cols = st.columns(2)
        raw_record: dict = {}
        for i, field in enumerate(disease.fields):
            col = cols[i % 2]
            widget_key = _field_widget_key(disease.key, field.name)
            if field.kind == "numeric":
                raw_record[field.name] = col.number_input(
                    f"{field.label}" + (f" ({field.unit})" if field.unit else ""),
                    min_value=float(field.min_value) if field.min_value is not None else None,
                    max_value=float(field.max_value) if field.max_value is not None else None,
                    value=float(field.default),
                    step=float(field.step) if field.step else 1.0,
                    help=field.help_text or None,
                    key=widget_key,
                )
            else:
                labels = list(field.categories.keys())
                default_index = 0
                selected_label = col.selectbox(field.label, labels, index=default_index, key=widget_key)
                raw_record[field.name] = field.categories[selected_label]

        submitted = st.form_submit_button("🔍 Predict Risk", use_container_width=True, type="primary")

    if submitted:
        return raw_record
    return None


def reset_patient_form(disease: DiseaseConfig) -> None:
    """Clear all widget state for `disease`'s form so fields revert to defaults."""
    for field in disease.fields:
        key = _field_widget_key(disease.key, field.name)
        if key in st.session_state:
            del st.session_state[key]


def render_probability_gauge(probability: float, risk_level: str) -> go.Figure:
    color = {"Low": "#2e7d32", "Moderate": "#e65100", "High": "#b71c1c"}.get(risk_level, "#455a64")
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            domain={"x": [0.08, 0.92], "y": [0, 1]},
            gauge={
                # Boundary ticks (0/100) sit exactly at the arc's edge and get
                # visually clipped by the SVG viewport, so only interior ticks
                # are shown; the exact value is already displayed prominently
                # in the center via mode="gauge+number".
                "axis": {"range": [0, 100], "tickvals": [20, 40, 60, 80]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 33], "color": "#e8f5e9"},
                    {"range": [33, 66], "color": "#fff3e0"},
                    {"range": [66, 100], "color": "#ffebee"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 4},
                    "thickness": 0.85,
                    "value": probability * 100,
                },
            },
            title={"text": "Predicted Probability of Disease"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=40, r=40, t=50, b=10))
    return fig


def render_contributor_chart(top_contributors: list[dict]) -> go.Figure:
    contributors = list(reversed(top_contributors))
    labels = [c["feature"] for c in contributors]
    values = [c["shap_value"] for c in contributors]
    colors = ["#c62828" if v > 0 else "#2e7d32" for v in values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top Contributing Factors (SHAP values)",
        xaxis_title="Impact on predicted probability",
        height=90 + 40 * len(contributors),
        margin=dict(l=10, r=30, t=50, b=30),
    )
    fig.add_vline(x=0, line_width=1, line_color="gray")
    return fig


def render_result_summary(result: PredictionResult) -> None:
    from app.theme import metric_card_html, risk_badge_html

    c1, c2, c3 = st.columns(3)
    c1.markdown(metric_card_html("Prediction", result.predicted_label), unsafe_allow_html=True)
    c2.markdown(metric_card_html("Confidence", f"{result.confidence:.1%}"), unsafe_allow_html=True)
    c3.markdown(
        f'<div class="mdai-card"><div class="mdai-metric-label">Risk Level</div>'
        f"{risk_badge_html(result.risk_level)}</div>",
        unsafe_allow_html=True,
    )
