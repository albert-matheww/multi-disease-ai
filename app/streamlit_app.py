"""MultiDiseaseAI - Streamlit dashboard.

A single-file Streamlit entry point that routes between five pages (Predict,
Batch Prediction, Model Performance, Prediction History, About), all backed
by the framework-agnostic modules in `src/` (`prediction.py`, `report.py`,
`history.py`, `evaluate.py`'s saved metrics). UI-only concerns (forms,
charts, theming) live here and in `app/components.py` / `app/theme.py`;
everything else is imported from `src` so the same logic can be reused from
`main.py`, tests, or a future non-Streamlit frontend.

Performance notes: the per-patient SHAP explanation is memoised inside
`DiseasePredictor`, the PDF report is built once per prediction (cached on
its timestamp), and the whole result panel is an `st.fragment` so toggling
the "view all contributions" expander or clicking download does not re-run
inference.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.components import (
    render_contributor_chart,
    render_patient_form,
    render_probability_gauge,
    render_result_summary,
    reset_patient_form,
)
from app.theme import apply_theme
from src.config import DISEASES, get_disease
from src.history import clear_history, init_db, load_history, save_prediction
from src.prediction import DiseasePredictor, PredictionResult, get_predictor
from src.report import build_report_pdf

st.set_page_config(
    page_title="MultiDiseaseAI",
    page_icon="\U0001fa7a",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

PAGES = ["Predict", "Batch Prediction", "Model Performance", "Prediction History", "About"]


@st.cache_resource(show_spinner="Loading model…")
def _cached_predictor(disease_key: str) -> DiseasePredictor:
    return get_predictor(disease_key)


@st.cache_data(show_spinner=False)
def _load_metrics(path_str: str, _mtime: float) -> dict:
    return json.loads(Path(path_str).read_text())


@st.cache_data(show_spinner=False, hash_funcs={PredictionResult: lambda r: r.timestamp})
def _cached_report_pdf(result: PredictionResult) -> bytes:
    """Build the PDF once per prediction (keyed on its timestamp), not per rerun."""
    return build_report_pdf(result)


def _load_predictor_safely(disease_key: str) -> DiseasePredictor | None:
    try:
        return _cached_predictor(disease_key)
    except FileNotFoundError:
        st.error(
            f"No trained model found for **{get_disease(disease_key).display_name}** yet.\n\n"
            f"Run the training pipeline first:\n\n"
            f"```bash\npython -m src.preprocessing\npython -m src.train --disease {disease_key}\npython -m src.evaluate --disease {disease_key}\n```"
        )
        return None


def render_sidebar() -> tuple[str, str, bool]:
    with st.sidebar:
        st.markdown("## \U0001fa7a MultiDiseaseAI")
        st.caption("Early multi-disease risk prediction using TabPFN")
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")

        st.divider()
        disease_labels = {key: f"{d.icon} {d.display_name}" for key, d in DISEASES.items()}
        disease_key = st.selectbox(
            "Disease",
            list(DISEASES.keys()),
            format_func=lambda k: disease_labels[k],
        )

        st.divider()
        dark_mode = st.toggle("\U0001f319 Dark Mode", value=st.session_state.get("dark_mode", False))
        st.session_state["dark_mode"] = dark_mode

        st.divider()
        st.caption("Model: TabPFN (foundation model for tabular data)")
        st.caption("Built for a Foundations of Data Science capstone project.")

    return page, disease_key, dark_mode


@st.fragment
def _render_result_panel(result: PredictionResult, disease_key: str) -> None:
    """Everything below the form. An `st.fragment` so expander/download clicks
    do a local rerun instead of re-running TabPFN inference."""
    st.divider()
    st.subheader("Prediction Result")

    if result.out_of_distribution:
        st.warning(
            f"**Novelty warning** — this record is unlike ~{result.novelty_score:.0%} of the "
            "training cohort (robust-Mahalanobis distance). The probability below is "
            "low-confidence; treat it as indicative only.",
            icon="⚠️",
        )

    render_result_summary(result)

    gauge_col, chart_col = st.columns(2)
    with gauge_col:
        st.plotly_chart(
            render_probability_gauge(
                result.probability,
                result.risk_level,
                band=(result.probability_low, result.probability_high),
            ),
            use_container_width=True,
        )
    with chart_col:
        st.plotly_chart(
            render_contributor_chart(result.top_contributors, disease_key), use_container_width=True
        )

    st.caption(
        f"Decision threshold {result.decision_threshold:.0%} (learned via Youden's J on "
        f"out-of-fold predictions) · {int(round((1 - result.band_alpha) * 100))}% conformal band "
        f"{result.probability_low:.0%}–{result.probability_high:.0%}"
    )

    with st.expander("View all feature contributions"):
        st.dataframe(pd.DataFrame(result.top_contributors), use_container_width=True, hide_index=True)

    st.download_button(
        "\U0001f4c4 Download Prediction Report (PDF)",
        data=_cached_report_pdf(result),
        file_name=f"{disease_key}_report_{result.timestamp.replace(':', '-')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    st.markdown(
        '<div class="mdai-disclaimer">This prediction is generated by an academic demonstration model '
        "trained on small public research datasets. It is not a validated diagnostic tool.</div>",
        unsafe_allow_html=True,
    )


def page_predict(disease_key: str) -> None:
    disease = get_disease(disease_key)
    st.title(f"{disease.icon} {disease.display_name} Risk Prediction")
    st.write(disease.description)

    predictor = _load_predictor_safely(disease_key)
    if predictor is None:
        return

    col_form, col_reset = st.columns([6, 1])
    with col_reset:
        st.write("")
        if st.button("↻ Reset", use_container_width=True):
            reset_patient_form(disease)
            st.session_state.pop(f"result_{disease_key}", None)
            st.rerun()

    with col_form:
        raw_record = render_patient_form(disease)

    if raw_record is not None:
        with st.spinner("Running TabPFN inference and computing SHAP explanation..."):
            result = predictor.predict(raw_record)
            save_prediction(result)
        st.session_state[f"result_{disease_key}"] = result

    result = st.session_state.get(f"result_{disease_key}")
    if result is None:
        return

    _render_result_panel(result, disease_key)


def page_batch(disease_key: str) -> None:
    disease = get_disease(disease_key)
    st.title(f"\U0001f4c1 Batch Prediction - {disease.display_name}")
    st.write(
        "Upload a CSV with one row per patient and one column per field below. "
        "Categorical columns must use the same raw values as the source dataset "
        "(see the template)."
    )

    predictor = _load_predictor_safely(disease_key)
    if predictor is None:
        return

    template_df = pd.DataFrame([{f: "" for f in predictor.feature_order_raw()}])
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False).encode(),
        file_name=f"{disease_key}_batch_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload patient CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        input_df = pd.read_csv(uploaded)
    except Exception as exc:  # noqa: BLE001 - surfaced directly to the user
        st.error(f"Could not read CSV: {exc}")
        return

    try:
        with st.spinner(f"Scoring {len(input_df)} patients..."):
            results_df = predictor.predict_batch(input_df)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.success(f"Scored {len(results_df)} patients.")
    if "out_of_distribution" in results_df.columns:
        n_ood = int(results_df["out_of_distribution"].sum())
        if n_ood:
            st.warning(
                f"{n_ood} of {len(results_df)} rows are outside the training distribution "
                "and flagged in the `out_of_distribution` column.",
                icon="⚠️",
            )
    st.dataframe(results_df, use_container_width=True)

    risk_counts = results_df["risk_level"].value_counts()
    st.bar_chart(risk_counts)

    st.download_button(
        "\U0001f4e5 Export Predictions (CSV)",
        data=results_df.to_csv(index=False).encode(),
        file_name=f"{disease_key}_batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def page_model_performance(disease_key: str) -> None:
    disease = get_disease(disease_key)
    st.title(f"\U0001f4ca Model Performance - {disease.display_name}")

    st.markdown(
        "**Why TabPFN?** TabPFN is a transformer-based *foundation model for "
        "tabular data*: pretrained once on millions of synthetic classification "
        "tasks, it performs Bayesian in-context learning at inference time "
        "instead of being separately optimized per dataset. For small clinical "
        "datasets like these (a few hundred rows), it matches or beats tuned "
        "gradient-boosted trees with zero hyperparameter search."
    )

    if not disease.metrics_path.exists():
        st.warning(
            f"No evaluation metrics found yet. Run:\n\n"
            f"```bash\npython -m src.evaluate --disease {disease_key}\n```"
        )
        return

    metrics = _load_metrics(str(disease.metrics_path), disease.metrics_path.stat().st_mtime)
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{metrics['accuracy']:.1%}")
    cols[1].metric("Precision", f"{metrics['precision']:.1%}")
    cols[2].metric("Recall", f"{metrics['recall']:.1%}")
    cols[3].metric("F1 Score", f"{metrics['f1_score']:.1%}")
    cols[4].metric("ROC AUC", f"{metrics['roc_auc']:.3f}")
    served = metrics.get("at_decision_threshold")
    if served:
        st.caption(
            f"Cards above use the default 0.5 rule. At the learned decision threshold the app applies "
            f"(t = {metrics['decision_threshold']:.2f}): accuracy {served['accuracy']:.1%}, "
            f"precision {served['precision']:.1%}, recall {served['recall']:.1%}, F1 {served['f1_score']:.1%}."
        )

    st.divider()
    fig_cols = st.columns(3)
    figure_specs = [
        ("Confusion Matrix", f"{disease_key}_confusion_matrix.png"),
        ("ROC Curve", f"{disease_key}_roc_curve.png"),
        ("Precision-Recall Curve", f"{disease_key}_pr_curve.png"),
    ]
    for col, (title, filename) in zip(fig_cols, figure_specs):
        path = disease.metrics_path.parent / "figures" / filename
        col.markdown(f"**{title}**")
        if path.exists():
            col.image(str(path), use_container_width=True)
        else:
            col.info("Not generated yet.")

    st.divider()
    st.markdown("**Global Feature Importance (SHAP)**")
    shap_path = disease.metrics_path.parent / "figures" / f"{disease_key}_shap_summary.png"
    if shap_path.exists():
        st.image(str(shap_path), use_container_width=True)
    else:
        st.info(f"Run `python -m src.explainability --disease {disease_key}` to generate this plot.")


def page_history() -> None:
    st.title("\U0001f553 Prediction History & Monitoring")

    history_df = load_history()
    if history_df.empty:
        st.info("No predictions logged yet. Make a prediction on the Predict page first.")
        return

    disease_filter = st.multiselect(
        "Filter by disease", options=sorted(history_df["disease_key"].unique()), default=None
    )
    filtered = (
        history_df if not disease_filter else history_df[history_df["disease_key"].isin(disease_filter)]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", len(filtered))
    c2.metric("Avg. Predicted Probability", f"{filtered['probability'].mean():.1%}")
    c3.metric("High-Risk Predictions", int((filtered["risk_level"] == "High").sum()))
    if "out_of_distribution" in filtered.columns:
        c4.metric("Out-of-Distribution", int(filtered["out_of_distribution"].fillna(0).sum()))

    st.markdown("**Risk Level Distribution**")
    st.bar_chart(filtered["risk_level"].value_counts())

    st.markdown("**Predictions Over Time**")
    trend = filtered.set_index("timestamp").sort_index()["probability"]
    st.line_chart(trend)

    st.markdown("**Full History**")
    display_df = filtered.drop(columns=["patient_input"]).copy()
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    dl_col, clear_col = st.columns(2)
    dl_col.download_button(
        "\U0001f4e5 Export History (CSV)",
        data=filtered.to_csv(index=False).encode(),
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if clear_col.button("\U0001f5d1 Clear History", use_container_width=True):
        clear_history()
        st.rerun()


def page_about() -> None:
    st.title("ℹ️ About MultiDiseaseAI")
    st.markdown("""
MultiDiseaseAI predicts the probability of four diseases - **Heart Disease**,
**Diabetes**, **Chronic Kidney Disease**, and **Liver Disease** - from
routine clinical measurements, using **TabPFN** as the sole modeling
algorithm.

Every prediction carries a **90% cross-conformal probability band**, a
**learned decision threshold** (Youden's J on out-of-fold predictions), and
a **robust-Mahalanobis novelty check** that flags inputs unlike the training
cohort. All three are precomputed at training time, so serving a prediction
is still a single TabPFN forward pass.

### Datasets
| Disease | Source | Rows |
|---|---|---|
| Heart Disease | UCI ML Repository (id=45, Cleveland) | 303 |
| Diabetes | Pima Indians Diabetes (NIDDK) | 768 |
| Chronic Kidney Disease | UCI ML Repository (id=336) | 400 |
| Liver Disease | UCI ILPD (id=225) | 583 |

See `data/README.md` in the repository for full citations and licenses.

### Limitations
These datasets are small, decades-old, and drawn from specific
populations/hospitals - they are **not representative samples** of the
general population, and the resulting models are **not validated for
clinical deployment**. This project demonstrates a modeling methodology
(a tabular foundation model with conformal uncertainty, novelty detection,
and SHAP explainability) rather than a production diagnostic system. See the
main `README.md` for the full limitations discussion.

### Disclaimer
This application is an academic demonstration and must not be used to make
real medical decisions. Always consult a qualified healthcare professional.
        """)


def main() -> None:
    page, disease_key, dark_mode = render_sidebar()
    apply_theme(dark_mode)

    if page == "Predict":
        page_predict(disease_key)
    elif page == "Batch Prediction":
        page_batch(disease_key)
    elif page == "Model Performance":
        page_model_performance(disease_key)
    elif page == "Prediction History":
        page_history()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
