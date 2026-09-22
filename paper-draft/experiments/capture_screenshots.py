#!/usr/bin/env python3
"""Capture real screenshots of the running Streamlit app (localhost:8532) for the paper.

Fills the heart-disease form with the EXACT patient from the paper's Table IX worked example
("most novel" row, results/worked_examples.md) so the screenshot corroborates a number already
in the paper, then submits and captures the result page (probability, band, label, novelty flag,
SHAP). Requires the app already running: .venv/bin/streamlit run app/streamlit_app.py --server.port 8532

    .venv/bin/python paper-draft/experiments/capture_screenshots.py
"""
from __future__ import annotations

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://localhost:8532"
OUT = Path(__file__).resolve().parents[1] / "figures" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

# Heart, "most novel" row from results/worked_examples.md (test row 15, true label 1).
FIELDS = [
    ("Age (years)", "number", "59"),
    ("Sex", "select", "Male"),
    ("Chest Pain Type", "select", "Asymptomatic"),
    ("Resting Blood Pressure (mm Hg)", "number", "164"),
    ("Serum Cholesterol (mg/dl)", "number", "176"),
    ("Fasting Blood Sugar > 120 mg/dl", "select", "Yes"),
    ("Resting ECG", "select", "LV Hypertrophy"),
    ("Max Heart Rate Achieved (bpm)", "number", "90"),
    ("Exercise-Induced Angina", "select", "No"),
    ("ST Depression (Exercise vs Rest)", "number", "1.00"),
    ("Slope of Peak Exercise ST Segment", "select", "Flat"),
    ("Major Vessels Colored by Fluoroscopy", "number", "2"),
    ("Thalassemia", "select", "Fixed Defect"),
]


def fill_select(page, label: str, option: str) -> None:
    combo = page.get_by_label(label)
    combo.click()
    page.get_by_text(option, exact=True).last.click()


def fill_number(page, label: str, value: str) -> None:
    box = page.get_by_label(label)
    box.click()
    box.fill(value)
    page.keyboard.press("Tab")


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("text=Heart Disease Risk Prediction", timeout=15000)
        time.sleep(1)

        page.screenshot(path=str(OUT / "01_empty_form.png"))

        for label, kind, value in FIELDS:
            if kind == "select":
                fill_select(page, label, value)
            else:
                fill_number(page, label, value)
            time.sleep(0.15)

        page.screenshot(path=str(OUT / "02_filled_form.png"))

        page.get_by_role("button", name="Predict Risk").click()
        # Real TabPFN inference + SHAP (background=10) took up to ~18-38s per disease when
        # timed for the paper (Table VIII); give this generous headroom.
        result_heading = page.get_by_text("Prediction Result", exact=True)
        result_heading.wait_for(state="visible", timeout=90000)
        result_heading.scroll_into_view_if_needed()
        time.sleep(1.5)
        page.screenshot(path=str(OUT / "03_prediction_result.png"), full_page=True)

        contrib_heading = page.get_by_text("View all feature contributions", exact=True)
        if contrib_heading.count():
            contrib_heading.click()
            time.sleep(1)
            page.screenshot(path=str(OUT / "04_shap_contributions.png"), full_page=True)

        # Sidebar: Model Performance page.
        page.get_by_text("Model Performance", exact=True).click()
        time.sleep(2)
        page.screenshot(path=str(OUT / "05_model_performance.png"), full_page=True)

        browser.close()
        print("saved screenshots to", OUT)


if __name__ == "__main__":
    main()
