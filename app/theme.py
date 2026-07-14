"""Custom CSS theming for the Streamlit dashboard, including a runtime
light/dark toggle.

Streamlit's native theme is normally fixed at launch via `.streamlit/config.toml`
and cannot be swapped at runtime from inside the app script. To still offer a
genuine in-app "Dark Mode" toggle (a bonus feature), this module injects a CSS
override block that restyles the app chrome and our custom components
(cards, badges) directly, keyed off a session-state flag set by the sidebar
toggle in `streamlit_app.py`.
"""

from __future__ import annotations

import streamlit as st

_SHARED_CSS = """
<style>
.mdai-card {
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    border: 1px solid var(--mdai-border);
    background: var(--mdai-card-bg);
}
.mdai-metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    opacity: 0.7;
    margin-bottom: 0.2rem;
}
.mdai-metric-value {
    font-size: 1.8rem;
    font-weight: 700;
}
.mdai-badge {
    display: inline-block;
    padding: 0.25rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.95rem;
}
.mdai-badge-low { background: #1b5e20; color: #e8f5e9; }
.mdai-badge-moderate { background: #e65100; color: #fff3e0; }
.mdai-badge-high { background: #b71c1c; color: #ffebee; }
.mdai-disclaimer {
    font-size: 0.78rem;
    opacity: 0.65;
    border-top: 1px solid var(--mdai-border);
    padding-top: 0.6rem;
    margin-top: 1.2rem;
}
</style>
"""

_LIGHT_VARS = """
<style>
:root {
    --mdai-border: rgba(49, 51, 63, 0.15);
    --mdai-card-bg: #ffffff;
}
</style>
"""

_DARK_VARS = """
<style>
:root {
    --mdai-border: rgba(250, 250, 250, 0.15);
    --mdai-card-bg: #1e2129;
}
[data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stSidebar"] {
    background-color: #0e1117;
    color: #fafafa;
}
[data-testid="stSidebar"] { border-right: 1px solid rgba(250,250,250,0.1); }
.stMarkdown, .stText, label, p, span, div { color: inherit; }
</style>
"""


def apply_theme(dark_mode: bool) -> None:
    st.markdown(_SHARED_CSS, unsafe_allow_html=True)
    st.markdown(_DARK_VARS if dark_mode else _LIGHT_VARS, unsafe_allow_html=True)


def risk_badge_html(risk_level: str) -> str:
    css_class = {"Low": "mdai-badge-low", "Moderate": "mdai-badge-moderate", "High": "mdai-badge-high"}.get(
        risk_level, "mdai-badge-low"
    )
    return f'<span class="mdai-badge {css_class}">{risk_level} Risk</span>'


def metric_card_html(label: str, value: str) -> str:
    return (
        f'<div class="mdai-card">'
        f'<div class="mdai-metric-label">{label}</div>'
        f'<div class="mdai-metric-value">{value}</div>'
        f"</div>"
    )
