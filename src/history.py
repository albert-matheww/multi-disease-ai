"""SQLite-backed storage for past predictions.

Backs two bonus features: the Streamlit app's "Prediction History" page and
its lightweight "Model Monitoring" view (predictions served over time, risk
distribution, out-of-distribution rate). A local SQLite file is more than
sufficient for a single-user academic demo app and requires no external
service.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd

from src.config import ROOT_DIR
from src.prediction import PredictionResult

DB_PATH = ROOT_DIR / "reports" / "prediction_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    disease_key TEXT NOT NULL,
    disease_display_name TEXT NOT NULL,
    probability REAL NOT NULL,
    predicted_label TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    confidence REAL NOT NULL,
    patient_input TEXT NOT NULL
);
"""

# Columns added after the original schema shipped. Applied idempotently on every
# connect via ``ALTER TABLE ... ADD COLUMN`` (SQLite raises if the column already
# exists, so each is guarded).
_MIGRATIONS = {
    "probability_low": "REAL",
    "probability_high": "REAL",
    "decision_threshold": "REAL",
    "out_of_distribution": "INTEGER",
    "novelty_score": "REAL",
}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    for column, decl in _MIGRATIONS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {column} {decl}")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(_SCHEMA)


def save_prediction(result: PredictionResult) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (timestamp, disease_key, disease_display_name, probability,
                 predicted_label, risk_level, confidence, patient_input,
                 probability_low, probability_high, decision_threshold,
                 out_of_distribution, novelty_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.timestamp,
                result.disease_key,
                result.disease_display_name,
                result.probability,
                result.predicted_label,
                result.risk_level,
                result.confidence,
                json.dumps(result.patient_input),
                result.probability_low,
                result.probability_high,
                result.decision_threshold,
                int(result.out_of_distribution),
                result.novelty_score,
            ),
        )


def load_history(disease_key: str | None = None) -> pd.DataFrame:
    with _connect() as conn:
        if disease_key:
            df = pd.read_sql_query(
                "SELECT * FROM predictions WHERE disease_key = ? ORDER BY timestamp DESC",
                conn,
                params=(disease_key,),
            )
        else:
            df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def clear_history(disease_key: str | None = None) -> None:
    with _connect() as conn:
        if disease_key:
            conn.execute("DELETE FROM predictions WHERE disease_key = ?", (disease_key,))
        else:
            conn.execute("DELETE FROM predictions")
