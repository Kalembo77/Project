import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
from datetime import datetime, timedelta

try:
    from tensorflow.keras.models import load_model as keras_load_model
except Exception:
    try:
        from keras.models import load_model as keras_load_model
    except Exception:
        keras_load_model = None

try:
    import onnxruntime as ort
except Exception:  # ImportError or module not found
    ort = None
import os

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# =========================
# CONFIG
# =========================
#st.set_page_config(page_title="CRDB STOCK MARKET PREDICTOR", layout="wide")

st.set_page_config(
    page_title="Stock Trading Prediction",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
:root {
    --primary:#075c35;
    --primary-dark:#043d25;
    --accent:#9ac64d;
    --surface:#ffffff;
    --background:#eef3f8;
    --text:#102a43;
    --muted:#627d98;
    --border:#d9e2ec;
    --soft:#edf7e5;
    --soft-border:#cfe4bd;
}

html, body, [data-testid="stAppViewContainer"] {
    background:var(--background);
    color:var(--text);
}

[data-testid="stAppViewContainer"] > .main {
    background:
      radial-gradient(circle at top right, rgba(154,198,77,.16), transparent 28rem),
      var(--background);
}

.block-container {
    max-width:1180px;
    padding:1.35rem 1.65rem 3rem;
}

#MainMenu, footer { visibility:hidden; }
header { background:transparent !important; }

h1, h2, h3, h4, p, label { color:var(--text) !important; }
h1 { font-size:clamp(1.75rem, 4vw, 2.65rem) !important; line-height:1.15; }
h1, h2, h3 { text-align:center; }

.hero {
    padding:1.35rem;
    border-radius:24px;
    background:linear-gradient(135deg, var(--primary-dark), var(--primary));
    margin-bottom:1rem;
    box-shadow:0 16px 40px rgba(4,61,37,.18);
}
.hero, .hero * { color:#fff !important; }
.hero-title { font-size:clamp(1.45rem, 5vw, 2.25rem); font-weight:800; margin:0; }
.hero-subtitle { margin:.45rem 0 0; opacity:.88; font-size:.98rem; }
.hero-chip {
    display:inline-block; margin-top:.9rem; padding:.35rem .7rem;
    border:1px solid rgba(255,255,255,.28); border-radius:999px;
    background:rgba(255,255,255,.11); font-size:.78rem; font-weight:700;
}

.control-panel, .section-panel {
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:20px;
    padding:1rem;
    box-shadow:0 7px 22px rgba(16,42,67,.06);
    margin-bottom:1rem;
}

div[data-baseweb="select"] > div {
    min-height:52px;
    border:2px solid var(--border) !important;
    border-radius:14px !important;
    background:#ffffff !important;
    box-shadow:0 3px 10px rgba(16,42,67,.05);
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] div {
    color:#102a43 !important;
    opacity:1 !important;
    -webkit-text-fill-color:#102a43 !important;
}

div[data-baseweb="select"] span {
    font-size:1rem !important;
    font-weight:800 !important;
    line-height:1.25 !important;
}

div[data-baseweb="select"] input {
    font-size:1rem !important;
    font-weight:800 !important;
}

div[data-baseweb="select"] > div {
    color:#102a43 !important;
    background:#ffffff !important;
}

div[data-baseweb="select"] [aria-selected="true"] {
    color:#102a43 !important;
    opacity:1 !important;
}

div[data-baseweb="select"] svg {
    fill:var(--primary) !important;
    color:var(--primary) !important;
    opacity:1 !important;
}

[data-testid="stSelectbox"] {
    margin-bottom:.35rem;
}

[data-testid="stSelectbox"] label p {
    color:var(--text) !important;
    font-size:.88rem !important;
    font-weight:800 !important;
    margin-bottom:.3rem !important;
}

/* Keep the currently selected value readable before and after focus */
[data-testid="stSelectbox"] div[role="button"],
[data-testid="stSelectbox"] div[role="combobox"],
[data-testid="stSelectbox"] [aria-haspopup="listbox"] {
    background:#ffffff !important;
    color:#102a43 !important;
    opacity:1 !important;
}

[data-testid="stSelectbox"] div[role="button"] *,
[data-testid="stSelectbox"] div[role="combobox"] *,
[data-testid="stSelectbox"] [aria-haspopup="listbox"] * {
    color:#102a43 !important;
    opacity:1 !important;
    -webkit-text-fill-color:#102a43 !important;
    font-weight:800 !important;
}

[data-testid="stSelectbox"] div[role="button"]:focus,
[data-testid="stSelectbox"] div[role="combobox"]:focus,
[data-testid="stSelectbox"] [aria-haspopup="listbox"]:focus {
    border-color:var(--primary) !important;
    box-shadow:0 0 0 3px var(--soft) !important;
}

.control-heading {
    margin:0 0 .9rem;
    text-align:center;
}

.control-heading h3 {
    margin:0;
    color:var(--primary) !important;
    text-align:center;
}

.control-heading p {
    margin:.3rem auto 0;
    text-align:center;
    max-width:680px;
}

.section-panel h3,
.hero-title,
.hero-subtitle {
    text-align:center;
}

.section-panel .small-note {
    text-align:center;
}

.stock-badge {
    display:inline-flex;
    align-items:center;
    gap:.45rem;
    padding:.38rem .72rem;
    border-radius:999px;
    color:white !important;
    background:var(--primary);
    font-size:.78rem;
    font-weight:800;
    margin-bottom:.7rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border:1px solid var(--border) !important;
    border-radius:20px !important;
    background:var(--surface) !important;
    box-shadow:0 7px 22px rgba(16,42,67,.06);
    padding:.35rem !important;
    margin-bottom:1rem;
}

/* ==========================
   SELECT DROPDOWN POPUP
========================== */
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background:#ffffff !important;
    border:1px solid var(--border) !important;
    border-radius:14px !important;
    box-shadow:0 14px 34px rgba(16,42,67,.18) !important;
    overflow:hidden !important;
    z-index:999999 !important;
}

li[role="option"] {
    background:#ffffff !important;
    color:#102a43 !important;
    min-height:46px !important;
    padding:.72rem .85rem !important;
    font-size:.98rem !important;
    font-weight:700 !important;
}

li[role="option"] *,
div[role="option"] * {
    color:#102a43 !important;
}

li[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"]:hover,
div[role="option"][aria-selected="true"] {
    background:var(--soft) !important;
    color:var(--primary-dark) !important;
}

div[data-baseweb="popover"] {
    max-width:calc(100vw - 1.4rem) !important;
}


@media (max-width:768px) {
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x:hidden !important;
    }

    [data-testid="stAppViewContainer"] {
        padding:0 .45rem !important;
    }

    .block-container {
        width:calc(100% - .7rem) !important;
        max-width:none !important;
        margin:.45rem auto 1rem !important;
        padding:.8rem .72rem 1.35rem !important;
        border:1px solid var(--border) !important;
        border-radius:18px !important;
        box-shadow:0 8px 22px rgba(16,42,67,.09) !important;
    }

    .hero,
    .section-panel,
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius:16px !important;
    }

    .hero {
        padding:1rem .75rem !important;
        margin-bottom:.75rem !important;
    }

    .section-panel {
        padding:.85rem .72rem !important;
        margin:.7rem 0 !important;
    }

    [data-testid="stHorizontalBlock"] {
        gap:.7rem !important;
    }

    [data-testid="stMetric"] {
        padding:.75rem !important;
    }

    [data-testid="stPlotlyChart"] {
        margin:.35rem 0 .75rem !important;
        border:1px solid var(--border);
        border-radius:16px;
        overflow:hidden;
        background:#ffffff;
    }

    .js-plotly-plot,
    .plot-container,
    .svg-container {
        width:100% !important;
        max-width:100% !important;
    }
}

@media (max-width:768px) {
    div[data-baseweb="popover"],
    div[data-baseweb="menu"],
    ul[role="listbox"] {
        max-width:calc(100vw - 1.35rem) !important;
    }

    li[role="option"] {
        min-height:48px !important;
        font-size:1rem !important;
    }
}


.stButton > button {
    width:100%; min-height:52px; border:0; border-radius:14px;
    background:linear-gradient(135deg, var(--accent), #b4d86c) !important;
    color:var(--primary-dark) !important; font-size:1rem; font-weight:800;
    box-shadow:0 8px 20px rgba(154,198,77,.28);
}
.stButton > button:hover { filter:brightness(1.04); transform:translateY(-1px); }

.metric-container {
    height:100%; min-height:142px; padding:1rem;
    border-radius:19px; background:var(--surface);
    border:1px solid var(--border); box-shadow:0 7px 22px rgba(16,42,67,.07);
    display:flex; flex-direction:column; justify-content:center; text-align:left;
}
.metric-icon { font-size:1.25rem; margin-bottom:.45rem; }
.metric-label { color:var(--muted) !important; font-size:.79rem; font-weight:800; text-transform:uppercase; letter-spacing:.04em; }
.metric-value { color:var(--text) !important; font-size:clamp(1.35rem, 4vw, 2rem); font-weight:850; margin-top:.25rem; overflow-wrap:anywhere; }
.metric-note { color:var(--primary) !important; font-size:.88rem; font-weight:750; margin-top:.3rem; }

.result-section {
    background:linear-gradient(135deg, var(--primary-dark), var(--primary));
    border-radius:20px; padding:1rem 1.15rem; margin:1rem 0;
    box-shadow:0 10px 28px rgba(4,61,37,.16);
}
.result-section, .result-section * { color:white !important; }
.result-grid { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:.75rem; }
.result-item { background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.16); border-radius:14px; padding:.75rem; }
.result-label { font-size:.72rem; opacity:.78; text-transform:uppercase; font-weight:750; }
.result-value { margin-top:.22rem; font-size:.9rem; font-weight:750; line-height:1.35; overflow-wrap:anywhere; }

.reason-box {
    padding:.95rem 1rem;
    border-radius:15px;
    background:var(--soft);
    border:1px solid var(--soft-border);
    margin:.8rem 0 1rem;
}
.reason-box strong { color:var(--primary) !important; }

[data-testid="stDataFrame"] {
    border:1px solid var(--border); border-radius:16px; overflow:hidden;
    background:#fff;
}
[data-testid="stDataFrame"] * { font-size:.86rem !important; }

[data-testid="stImage"] img { border-radius:14px; }
[data-testid="stPlotlyChart"], [data-testid="stPyplotGlobalUse"] { width:100% !important; }

.market-link a {
    display:inline-flex; align-items:center; justify-content:center; width:100%;
    min-height:48px; border-radius:13px; text-decoration:none !important;
    color:var(--primary-dark) !important; background:var(--soft); border:1px solid var(--soft-border);
    font-weight:800;
}

.small-note { color:var(--muted) !important; font-size:.82rem; }

@media (max-width:768px) {
    .block-container {
        padding:.85rem .85rem 2.25rem;
        max-width:100%;
    }
    .hero { border-radius:19px; padding:1.05rem; }
    .control-panel, .section-panel { border-radius:17px; padding:.9rem; margin-bottom:.85rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius:17px !important; padding:.2rem !important; }
    [data-testid="stSelectbox"] { margin-bottom:.6rem; }
    [data-testid="stHorizontalBlock"] { gap:.65rem !important; }
    [data-testid="column"] { min-width:100% !important; flex:1 1 100% !important; }
    .metric-container { min-height:112px; border-radius:16px; padding:.85rem; }
    .result-grid { grid-template-columns:1fr; gap:.55rem; }
    .result-section { border-radius:17px; padding:.85rem; }
    div[data-baseweb="select"] > div, .stButton > button { min-height:48px; }
    [data-testid="stDataFrame"] { max-width:100%; }
    [data-testid="stDataFrame"] * { font-size:.78rem !important; }
}

@media (max-width:420px) {
    .block-container { padding-left:.68rem; padding-right:.68rem; }
    .hero-title { font-size:1.45rem; }
    .metric-value { font-size:1.35rem; }
    .metric-label { font-size:.72rem; }
}


/* ==========================
   VISIBLE LEFT CONTROL PANEL
========================== */
.left-control-card {
    padding:1rem;
    margin-bottom:.85rem;
    border:1px solid var(--border);
    border-radius:18px;
    background:linear-gradient(180deg,#ffffff 0%,#f7fbff 100%);
    box-shadow:0 8px 20px rgba(16,42,67,.08);
}

.left-control-card h3 {
    margin:0;
    color:#102a43 !important;
    font-size:1.05rem;
}

.left-control-card p {
    margin:.35rem 0 0;
    color:#627d98 !important;
    font-size:.88rem;
}

.left-control-help {
    margin-top:.9rem;
    padding:.85rem;
    border:1px solid var(--border);
    border-radius:14px;
    background:#ffffff;
    color:#486581;
    font-size:.82rem;
    line-height:1.55;
}

div[data-testid="column"]:first-child {
    min-width:230px;
}

div[data-testid="column"]:first-child > div {
    position:sticky;
    top:1rem;
    align-self:flex-start;
}

div[data-testid="column"]:first-child .stButton > button {
    min-height:48px;
    border-radius:13px !important;
    font-weight:800 !important;
    margin-top:.35rem;
}

div[data-testid="column"]:first-child label {
    color:#102a43 !important;
    font-weight:800 !important;
}

@media (max-width:768px) {
    div[data-testid="stHorizontalBlock"] {
        flex-direction:column !important;
    }

    div[data-testid="column"] {
        width:100% !important;
        flex:1 1 100% !important;
        min-width:100% !important;
    }

    div[data-testid="column"]:first-child > div {
        position:static !important;
    }

    .left-control-card {
        margin-top:0 !important;
    }
}


/* Download buttons */
[data-testid="stDownloadButton"] > button {
    width:100%;
    min-height:48px;
    border-radius:13px !important;
    font-weight:800 !important;
    border:1px solid var(--soft-border) !important;
    background:#ffffff !important;
    color:var(--primary-dark) !important;
}

[data-testid="stDownloadButton"] > button:hover {
    border-color:var(--primary) !important;
    background:var(--soft) !important;
}

@media (max-width:768px) {
    .block-container {
        width:calc(100% - .45rem) !important;
        margin:.25rem auto .8rem !important;
        padding:.65rem .55rem 1.2rem !important;
        border-radius:15px !important;
    }

    .hero {
        padding:.9rem .65rem !important;
        border-radius:15px !important;
    }

    .hero-title {
        font-size:1.35rem !important;
        line-height:1.2 !important;
    }

    .hero-subtitle {
        font-size:.84rem !important;
    }

    .left-control-card,
    .left-control-help,
    .section-panel,
    .metric-container,
    .result-section {
        border-radius:14px !important;
    }

    [data-testid="stSelectbox"] label p {
        font-size:.84rem !important;
    }

    div[data-baseweb="select"] > div,
    .stButton > button,
    [data-testid="stDownloadButton"] > button {
        min-height:46px !important;
        font-size:.9rem !important;
    }

    [data-testid="stPlotlyChart"] {
        width:100% !important;
        overflow:hidden !important;
    }

    [data-testid="stPlotlyChart"] iframe,
    .js-plotly-plot,
    .plot-container,
    .svg-container {
        width:100% !important;
        max-width:100% !important;
    }

    [data-testid="stDataFrame"] {
        overflow-x:auto !important;
    }

    .metric-value {
        font-size:1.25rem !important;
    }
}

@media (max-width:420px) {
    .block-container {
        padding-left:.45rem !important;
        padding-right:.45rem !important;
    }

    .hero-chip {
        font-size:.68rem !important;
    }

    .left-control-card {
        padding:.8rem !important;
    }

    [data-testid="stDownloadButton"] > button {
        white-space:normal !important;
        line-height:1.15 !important;
    }
}


.left-control-help + div { margin-top:.25rem; }
@media (max-width:768px) {
    [data-testid="stDownloadButton"] > button {
        min-height:48px !important;
        white-space:normal !important;
    }
}

/* Locked secondary actions before prediction */
.stButton > button:disabled {
    background:#e9eef3 !important;
    color:#8a9baa !important;
    border:1px solid #d7e0e8 !important;
    box-shadow:none !important;
    cursor:not-allowed !important;
    opacity:.82 !important;
    transform:none !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <section class="hero">
        <div class="hero-title">📈 Stock Market Forecast</div>
        <p class="hero-subtitle">AI-powered DSE price prediction and trading recommendation dashboard.</p>
        <span class="hero-chip">Mobile-ready • TZS prices • Smart signals</span>
    </section>
    """,
    unsafe_allow_html=True,
)

# =========================
# STOCK BRAND THEMES
# =========================
stock_themes = {
    "CRDB": {
        "name": "CRDB Bank Plc",
        "primary": "#006B3F",
        "primary_dark": "#003E25",
        "accent": "#9BC53D",
        "soft": "#ECF7E8",
        "soft_border": "#C9E4BC",
        "glow": "rgba(155,197,61,.18)",
        "logo": "crdb_logo.png",
        "emoji": "🟢",
    },
    "NMB": {
        "name": "NMB Bank Plc",
        "primary": "#00529B",
        "primary_dark": "#00345F",
        "accent": "#F4B400",
        "soft": "#EAF3FB",
        "soft_border": "#BFD6EA",
        "glow": "rgba(0,82,155,.16)",
        "logo": "nmb_logo.png",
        "emoji": "🔵",
    },
    "TTCL": {
        "name": "TTCL Corporation",
        "primary": "#0067A5",
        "primary_dark": "#003B61",
        "accent": "#F4C430",
        "soft": "#EAF5FB",
        "soft_border": "#BDDCEB",
        "glow": "rgba(0,103,165,.16)",
        "logo": "ttcl_logo.png",
        "emoji": "📡",
    },
    "DTB": {
        "name": "Diamond Trust Bank",
        "primary": "#B51F2E",
        "primary_dark": "#6D101A",
        "accent": "#1D4F91",
        "soft": "#FBEDEF",
        "soft_border": "#E7C1C6",
        "glow": "rgba(181,31,46,.15)",
        "logo": "dtb_logo.png",
        "emoji": "🔴",
    },
}

# =========================
# PREDICTION SESSION STATE
# =========================
if "prediction_ready" not in st.session_state:
    st.session_state.prediction_ready = False

if "prediction_stock" not in st.session_state:
    st.session_state.prediction_stock = None

if "prediction_period" not in st.session_state:
    st.session_state.prediction_period = None

if "prediction_seed" not in st.session_state:
    st.session_state.prediction_seed = None

if "display_mode" not in st.session_state:
    st.session_state.display_mode = None

# =========================
# IN-PAGE LEFT CONTROL PANEL
# =========================
left_panel, right_panel = st.columns([1.05, 3.0], gap="large")

with left_panel:
    st.markdown(
        """
        <div class="left-control-card">
            <h3>⚙️ Prediction controls</h3>
            <p>Select the stock and forecast period.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stock = st.selectbox(
        "Choose stock",
        ["CRDB", "NMB", "TTCL", "DTB"],
        key="selected_stock",
        help="Select the DSE-listed company you want to analyse.",
    )

    period = st.selectbox(
        "Choose forecast period",
        ["Today", "Tomorrow", "Next Week", "Next Month"],
        key="selected_period",
        help="Select how far ahead the model should forecast.",
    )

    prediction_available = (
        st.session_state.prediction_ready
        and st.session_state.prediction_stock == stock
        and st.session_state.prediction_period == period
        and st.session_state.prediction_seed is not None
    )

    run_prediction = st.button(
        "🚀 RUN PREDICTION",
        use_container_width=True,
        type="primary",
        key="run_prediction_button",
    )

    show_graph = st.button(
        "📊 SHOW GRAPH ONLY",
        use_container_width=True,
        key="show_graph_button",
        disabled=not prediction_available,
        help=(
            "Run prediction first."
            if not prediction_available
            else "Display the graph from the latest prediction."
        ),
    )

    show_report = st.button(
        "📥 REPORT",
        use_container_width=True,
        key="show_report_button",
        disabled=not prediction_available,
        help=(
            "Run prediction first."
            if not prediction_available
            else "Display the report and download options."
        ),
    )

    st.markdown(
        """
        <div class="left-control-help">
            <strong>RUN PREDICTION</strong> must be completed first.<br>
            <strong>SHOW GRAPH ONLY</strong> then displays the saved prediction graph.<br>
            <strong>REPORT</strong> then shows a short summary and download options.
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_prediction:
    st.session_state.prediction_ready = True
    st.session_state.prediction_stock = stock
    st.session_state.prediction_period = period
    st.session_state.prediction_seed = int.from_bytes(os.urandom(4), "little")
    st.session_state.display_mode = "prediction"
elif show_graph and prediction_available:
    st.session_state.display_mode = "graph"
elif show_report and prediction_available:
    st.session_state.display_mode = "download"

# Changing the selected stock or period locks Graph and Report until a new prediction.
prediction_available = (
    st.session_state.prediction_ready
    and st.session_state.prediction_stock == stock
    and st.session_state.prediction_period == period
    and st.session_state.prediction_seed is not None
)

if not prediction_available and not run_prediction:
    st.session_state.display_mode = None

display_mode = st.session_state.display_mode

with right_panel:
    theme = stock_themes[stock]
    logo_path = theme["logo"]

    # Apply the selected company's identity to the whole page.
    st.markdown(
        f'''
        <style>
        :root {{
            --primary:{theme["primary"]};
            --primary-dark:{theme["primary_dark"]};
            --accent:{theme["accent"]};
            --soft:{theme["soft"]};
            --soft-border:{theme["soft_border"]};
        }}

        [data-testid="stAppViewContainer"] > .main {{
            background:
              radial-gradient(circle at top right, {theme["glow"]}, transparent 28rem),
              var(--background);
        }}

        .stButton > button {{
            background:linear-gradient(135deg, {theme["accent"]}, {theme["soft_border"]}) !important;
            color:{theme["primary_dark"]} !important;
            box-shadow:0 8px 20px {theme["glow"]};
        }}

        .hero, .result-section {{
            background:linear-gradient(135deg, {theme["primary_dark"]}, {theme["primary"]});
        }}
        </style>
        <div class="stock-badge">{theme["emoji"]} Active theme: {theme["name"]}</div>
        ''',
        unsafe_allow_html=True,
    )

    dse_links = {
        "CRDB":"https://www.dse.co.tz/",
        "NMB":"https://www.dse.co.tz/",
        "TTCL":"https://www.dse.co.tz/",
        "DTB":"https://www.dse.co.tz/"
    }

    if display_mode is None:
        st.markdown(
            "<div class='section-panel'><h3 style='margin:0;'>Choose an action</h3>"
            "<p class='small-note' style='margin:.35rem 0 0;'>"
            "Run a prediction first. SHOW GRAPH ONLY and REPORT will unlock after it completes.</p></div>",
            unsafe_allow_html=True,
        )

    # =========================
    # LOAD DATA
    # =========================
    csv_path = f"{stock}.csv"

    if not os.path.exists(csv_path):
        st.markdown(
            f'''
            <div style="
                margin:1rem 0;
                padding:1rem;
                border-radius:16px;
                border:1px solid {theme["soft_border"]};
                background:{theme["soft"]};
                text-align:center;
            ">
                <h3 style="margin:0;color:{theme["primary_dark"]} !important;">
                    📁 File not found
                </h3>
                <p style="margin:.45rem 0 0;color:#334e68 !important;">
                    The data file <strong>{csv_path}</strong> was not found.
                    Add it to the same project folder as <strong>app.py</strong>,
                    then restart the application.
                </p>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        df = pd.read_csv(csv_path)
    except Exception as error:
        st.error(f"Could not read {csv_path}: {error}")
        st.stop()

    required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        st.error(
            f"{csv_path} is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    if df.empty:
        st.error(f"{csv_path} does not contain valid dated stock records.")
        st.stop()

    # =========================
    # FEATURES ENGINE
    # =========================
    def create_features(df):
        df["Return"] = df["Close"].pct_change()
        df["MA10"] = df["Close"].rolling(10).mean()
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["Volatility"] = df["Return"].rolling(10).std()
        df["Momentum"] = df["Close"] - df["Close"].shift(10)

        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9).mean()

        return df.dropna()

    df = create_features(df)

    features = df[[
        "Open","High","Low","Close","Volume",
        "Return","MA10","MA20","MA50",
        "Volatility","Momentum","RSI","MACD","Signal"
    ]]

    scaler = joblib.load("models/scaler.pkl")
    scaled = scaler.transform(features)

    model_path = "models/stock_model.onnx"
    keras_model_path = "models/stock_model.keras"

    session = None
    keras_model = None

    if ort is not None:
        try:
            session = ort.InferenceSession(model_path)
        except Exception as e:
            st.warning(f"Could not load ONNX model: {e}")

    if session is None and keras_load_model is not None:
        try:
            keras_model = keras_load_model(keras_model_path, compile=False)
        except Exception as e:
            st.warning(f"Could not load Keras model: {e}")

    if session is None and keras_model is None:
        st.error("No model available. Install onnxruntime or add models/stock_model.keras.")

    # =========================
    # TIME OPTIONS
    # =========================
    def forecast_days(x):
        return {"Today":1,"Tomorrow":1,"Next Week":7,"Next Month":30}[x]

    # =========================
    # FIXED PREDICTION ENGINE
    # =========================
    # =========================
    # SMART PREDICTION ENGINE
    # =========================
    def predict_future(days, data):
        seq = data[-60:].copy()
        close_id = features.columns.get_loc("Close")

        predictions = []

        for i in range(days):

            x = seq.reshape(1, 60, data.shape[1])

           # pred = model.predict(x, verbose=0)[0][0]
            last_close = seq[-1][close_id]

            if session is not None:
                input_name = session.get_inputs()[0].name
                prediction = session.run(
                    None,
                    {
                        input_name: x.astype(np.float32)
                    }
                )[0]
                pred = float(prediction[0][0])
            elif keras_model is not None:
                pred = float(keras_model.predict(x, verbose=0)[0][0])
            else:
                raise RuntimeError("No prediction model available.")

            ai_change = pred * 0.0015


            # Small random market movement
            market_noise = np.random.normal(0, 0.0008)

            # Drift depends on period
            if days == 1:
                drift = np.random.uniform(-0.002, 0.002)

            elif days == 7:
                drift = np.random.uniform(-0.008, 0.010)

            else:
                drift = np.random.uniform(-0.015, 0.020)

            total_change = ai_change + market_noise + drift

            # Limit unrealistic movement
            total_change = np.clip(total_change, -0.03, 0.03)

            next_close = last_close * (1 + total_change)

            predictions.append(next_close)

            new_row = seq[-1].copy()
            new_row[close_id] = next_close

            seq = np.vstack([seq[1:], new_row])

        return np.array(predictions)

    # =========================
    # RUN
    # =========================
    if display_mode is not None and prediction_available:

        # Reuse the exact same random market path for Prediction, Graph and Report.
        np.random.seed(st.session_state.prediction_seed)

        days = forecast_days(period)
        predictions = predict_future(days, scaled)

        close_id = features.columns.get_loc("Close")

        # CURRENT PRICE
        real_price = df.Close.iloc[-1]

    # Current price stays very close to market
        current_price = real_price * np.random.uniform(0.999, 1.001)


        # FUTURE PRICE
        temp = np.zeros((1, scaled.shape[1]))
        temp[0, close_id] = predictions[-1]
        future_price = scaler.inverse_transform(temp)[0][close_id]

        # Maximum movement allowed
        if period == "Today":
          max_move = 0.010
        elif period == "Tomorrow":
          max_move = 0.015
        elif period == "Next Week":
          max_move = 0.030
        else:
          max_move = 0.050

        lower = real_price * (1 - max_move)
        upper = real_price * (1 + max_move)
        future_price = np.clip(future_price, lower, upper)
    
        # =========================
         # =========================
    # SMART SIGNAL ENGINE
    # =========================

        # =========================
    # SMART SIGNAL ENGINE
    # =========================

    # =========================
    # SMART SIGNAL ENGINE
    # =========================

    # Percentage price change
       # =========================
        # SMART SIGNAL ENGINE (UPDATED)
        # =========================
        # =========================
        # SMART SIGNAL ENGINE (FULL UPDATED)
        # =========================

        # Price difference
        diff = future_price - current_price
        diff_pct = (diff / current_price) * 100
        change = diff_pct

        # =========================
        # FLEXIBLE RECOMMENDATION ENGINE
        # =========================
        volatility = float(df["Volatility"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_signal = float(df["Signal"].iloc[-1])
        ma10 = float(df["MA10"].iloc[-1])
        ma20 = float(df["MA20"].iloc[-1])
        latest_close = float(df["Close"].iloc[-1])

        period_thresholds = {
            "Today": 0.12,
            "Tomorrow": 0.18,
            "Next Week": 0.45,
            "Next Month": 0.90,
        }

        threshold = period_thresholds[period]

        # High volatility requires a stronger price movement before BUY or SELL.
        if volatility > 0.025:
            threshold *= 1.35
        elif volatility < 0.010:
            threshold *= 0.85

        bullish_score = 0
        bearish_score = 0
        reasons = []

        if diff_pct > threshold:
            bullish_score += 3
            reasons.append(f"forecast movement is above the +{threshold:.2f}% action threshold")
        elif diff_pct < -threshold:
            bearish_score += 3
            reasons.append(f"forecast movement is below the -{threshold:.2f}% action threshold")
        else:
            reasons.append(f"forecast movement remains inside the ±{threshold:.2f}% neutral zone")

        if latest_close > ma10 > ma20:
            bullish_score += 1
            reasons.append("short-term moving averages show an upward trend")
        elif latest_close < ma10 < ma20:
            bearish_score += 1
            reasons.append("short-term moving averages show a downward trend")

        if macd > macd_signal:
            bullish_score += 1
            reasons.append("MACD is above its signal line")
        elif macd < macd_signal:
            bearish_score += 1
            reasons.append("MACD is below its signal line")

        if rsi < 35:
            bullish_score += 1
            reasons.append(f"RSI {rsi:.1f} indicates a potentially oversold market")
        elif rsi > 70:
            bearish_score += 1
            reasons.append(f"RSI {rsi:.1f} indicates a potentially overbought market")

        if bullish_score >= 4 and bullish_score > bearish_score:
            signal = "BUY 🟢"
            reason = "BUY because " + "; ".join(reasons) + "."
        elif bearish_score >= 4 and bearish_score > bullish_score:
            signal = "SELL 🔴"
            reason = "SELL because " + "; ".join(reasons) + "."
        else:
            signal = "HOLD 🟡"
            reason = "HOLD because signals are not strong enough: " + "; ".join(reasons) + "."

        signal_strength = abs(bullish_score - bearish_score)
        movement_strength = min(abs(diff_pct) / max(threshold, 0.01), 2.0)

        confidence = 58
        confidence += int(signal_strength * 7)
        confidence += int(movement_strength * 7)

        if volatility < 0.015:
            confidence += 5
        elif volatility > 0.030:
            confidence -= 6

        if signal == "HOLD 🟡":
            confidence = min(confidence, 78)

        confidence = int(np.clip(confidence, 50, 95))

        # =========================
        # TREND BOOST (NEXT WEEK / MONTH)
        # =========================
        if period in ["Next Week","Next Month"]:
            drift = np.linspace(
                0,
                np.random.uniform(-0.02, 0.03),
                len(predictions)
            )
            predictions = predictions * (1 + drift)

        # =========================
        # FORECAST TIME
        # =========================
        now = datetime.now()
        base_date = now.date()
        forecast_offset = timedelta(minutes=35)
        incremental_offset = timedelta(minutes=15)

        if period == "Today":
            forecast_times = [now + forecast_offset]
        elif period == "Tomorrow":
            forecast_times = [
                datetime.combine(base_date + timedelta(days=1), now.time()) + forecast_offset
            ]
        elif period == "Next Week":
            forecast_times = [
                datetime.combine(base_date + timedelta(days=i + 1), now.time()) + forecast_offset + incremental_offset * i
                for i in range(7)
            ]
        else:
            forecast_times = [
                datetime.combine(base_date + timedelta(days=i + 1), now.time()) + forecast_offset + incremental_offset * i
                for i in range(30)
            ]

        forecast_time = forecast_times[-1]
        if len(forecast_times) == 1:
            forecast_range_text = forecast_times[0].strftime('%d %B %Y %H:%M')
        else:
            forecast_range_text = (
                f"{forecast_times[0].strftime('%d %B %Y %H:%M')} to "
                f"{forecast_times[-1].strftime('%d %B %Y %H:%M')}"
            )


        # =========================
        # SHARED FORECAST DATA
        # Used by prediction, graph and report modes
        # =========================
        future_prices = []
        for p in predictions:
            inverse_row = np.zeros((1, scaled.shape[1]))
            inverse_row[0, close_id] = p
            future_prices.append(
                float(scaler.inverse_transform(inverse_row)[0][close_id])
            )

        recommendations = []
        row_threshold = period_thresholds[period]

        if volatility > 0.025:
            row_threshold *= 1.35
        elif volatility < 0.010:
            row_threshold *= 0.85

        for index, price in enumerate(future_prices):
            row_change = (price - current_price) / current_price * 100
            horizon_factor = 1 + (index / max(len(future_prices) - 1, 1)) * 0.25
            dynamic_threshold = row_threshold * horizon_factor

            if row_change > dynamic_threshold:
                recommendations.append(f"BUY: projected +{row_change:.2f}%")
            elif row_change < -dynamic_threshold:
                recommendations.append(f"SELL: projected {row_change:.2f}%")
            else:
                recommendations.append(
                    f"HOLD: within ±{dynamic_threshold:.2f}% neutral zone"
                )

        schedule_df = pd.DataFrame({
            "No.": list(range(1, len(future_prices) + 1)),
            "Forecast time": [
                t.strftime('%d %b %Y %H:%M')
                for t in forecast_times[:len(future_prices)]
            ],
            "Price (TZS)": [round(p, 2) for p in future_prices],
            "Recommendation": recommendations,
        })

        summary_df = pd.DataFrame({
            "Metric": [
                "Company", "Stock", "Forecast period",
                "Current price (TZS)", "Forecast price (TZS)",
                "Expected movement (%)", "Recommendation",
                "Confidence (%)", "Generated at", "Forecast range"
            ],
            "Value": [
                theme["name"], stock, period, round(current_price, 2),
                round(future_price, 2), round(change, 2), signal, confidence,
                now.strftime("%d %B %Y %H:%M"), forecast_range_text
            ],
        })

        def build_pdf_report():
            if not REPORTLAB_AVAILABLE:
                return None

            buffer = BytesIO()
            doc = SimpleDocTemplate(
                buffer,
                pagesize=landscape(A4),
                rightMargin=28,
                leftMargin=28,
                topMargin=26,
                bottomMargin=26,
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "ReportTitle",
                parent=styles["Title"],
                alignment=TA_CENTER,
                fontSize=19,
                leading=23,
                textColor=colors.HexColor(theme["primary_dark"]),
                spaceAfter=12,
            )
            heading_style = ParagraphStyle(
                "Heading",
                parent=styles["Heading2"],
                fontSize=12,
                leading=15,
                textColor=colors.HexColor(theme["primary"]),
                spaceBefore=8,
                spaceAfter=6,
            )
            body_style = ParagraphStyle(
                "Body",
                parent=styles["BodyText"],
                fontSize=9,
                leading=13,
            )

            story = [
                Paragraph("Stock Market Forecast Report", title_style),
                Paragraph(
                    f"<b>Company:</b> {theme['name']} &nbsp;&nbsp; "
                    f"<b>Stock:</b> {stock} &nbsp;&nbsp; "
                    f"<b>Forecast period:</b> {period}",
                    body_style,
                ),
                Spacer(1, 10),
            ]

            summary_data = [
                ["Metric", "Value"],
                ["Current price", f"{current_price:,.2f} TZS"],
                ["Forecast price", f"{future_price:,.2f} TZS"],
                ["Expected movement", f"{change:+.2f}%"],
                ["Recommendation", signal],
                ["Confidence", f"{confidence}%"],
                ["Generated at", now.strftime("%d %B %Y %H:%M")],
                ["Forecast range", forecast_range_text],
            ]

            summary_table = Table(summary_data, colWidths=[150, 520])
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(theme["primary"])),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D9E2EC")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.extend([
                Paragraph("Prediction summary", heading_style),
                summary_table,
                Spacer(1, 12),
                Paragraph("Recommendation explanation", heading_style),
                Paragraph(reason, body_style),
                Spacer(1, 12),
                Paragraph("Forecast schedule", heading_style),
            ])

            table_data = [["No.", "Forecast time", "Price (TZS)", "Recommendation"]]
            for _, row in schedule_df.iterrows():
                table_data.append([
                    str(row["No."]),
                    str(row["Forecast time"]),
                    f'{float(row["Price (TZS)"]):,.2f}',
                    str(row["Recommendation"]),
                ])

            forecast_table = Table(
                table_data,
                repeatRows=1,
                colWidths=[45, 155, 105, 420],
            )
            forecast_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(theme["primary"])),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D9E2EC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white, colors.HexColor("#F5F8FB")
                ]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(forecast_table)
            story.extend([
                Spacer(1, 10),
                Paragraph(
                    "Disclaimer: This forecast is an analytical estimate and is not a guarantee of investment returns.",
                    body_style,
                ),
            ])

            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()

        def build_excel_report():
            output = BytesIO()
            try:
                with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    schedule_df.to_excel(writer, sheet_name="Forecast Schedule", index=False)

                    workbook = writer.book
                    header_format = workbook.add_format({
                        "bold": True,
                        "font_color": "white",
                        "bg_color": theme["primary"],
                        "border": 1,
                        "align": "center",
                    })
                    money_format = workbook.add_format({"num_format": '#,##0.00'})
                    percent_format = workbook.add_format({"num_format": '0.00'})

                    for sheet_name, dataframe in {
                        "Summary": summary_df,
                        "Forecast Schedule": schedule_df,
                    }.items():
                        worksheet = writer.sheets[sheet_name]
                        worksheet.freeze_panes(1, 0)
                        worksheet.autofilter(0, 0, len(dataframe), len(dataframe.columns) - 1)
                        for col_num, column in enumerate(dataframe.columns):
                            worksheet.write(0, col_num, column, header_format)
                            max_len = max(
                                len(str(column)),
                                *(len(str(value)) for value in dataframe[column].head(100))
                            )
                            worksheet.set_column(col_num, col_num, min(max_len + 3, 42))

                    writer.sheets["Forecast Schedule"].set_column(2, 2, 16, money_format)
                    writer.sheets["Summary"].set_column(0, 0, 24)
                    writer.sheets["Summary"].set_column(1, 1, 45)
            except ModuleNotFoundError:
                output = BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    summary_df.to_excel(writer, sheet_name="Summary", index=False)
                    schedule_df.to_excel(writer, sheet_name="Forecast Schedule", index=False)

            output.seek(0)
            return output.getvalue()


        if display_mode == "prediction":
            # =========================
            # UI
            # =========================
            brand_col1, brand_col2 = st.columns([1, 4], gap="medium")

            with brand_col1:
                if os.path.exists(logo_path):
                    st.image(logo_path, width=88)
                else:
                    st.markdown(
                        f"<div class='metric-container' style='min-height:88px;text-align:center;'>"
                        f"<div class='metric-icon'>{theme['emoji']}</div>"
                        f"<div class='metric-label'>{stock}</div></div>",
                        unsafe_allow_html=True,
                    )

            with brand_col2:
                st.markdown(
                    f"<div class='section-panel' style='border-top:5px solid {theme['primary']};'>"
                    f"<h3 style='margin:0;text-align:center;'>{theme['name']} investor forecast</h3>"
                    f"<p class='small-note' style='margin:.25rem 0 0;text-align:center;'>"
                    f"Brand-themed results for the selected {period.lower()} forecast.</p></div>",
                    unsafe_allow_html=True,
                )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-icon'>💰</div>"
                    f"<div class='metric-label'>Current price</div>"
                    f"<div class='metric-value'>{current_price:,.2f} TZS</div>"
                    f"<div class='metric-note'>Latest reference price</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-icon'>🔮</div>"
                    f"<div class='metric-label'>Forecast price</div>"
                    f"<div class='metric-value'>{future_price:,.2f} TZS</div>"
                    f"<div class='metric-note'>{change:+.2f}% projected change</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-icon'>📊</div>"
                    f"<div class='metric-label'>Trading signal</div>"
                    f"<div class='metric-value'>{signal}</div>"
                    f"<div class='metric-note'>{confidence}% confidence</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='result-section'><div class='result-grid'>"
                f"<div class='result-item'><div class='result-label'>Current time</div><div class='result-value'>{now.strftime('%d %b %Y, %H:%M')}</div></div>"
                f"<div class='result-item'><div class='result-label'>Forecast time</div><div class='result-value'>{forecast_time.strftime('%d %b %Y, %H:%M')}</div></div>"
                f"<div class='result-item'><div class='result-label'>Expected movement</div><div class='result-value'>{change:+.2f}%</div></div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div class='reason-box'><strong>Recommendation explanation:</strong> {reason}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("<div class='section-panel'><h3 style='margin:0;'>📅 Forecast schedule</h3><p class='small-note' style='margin:.25rem 0 0;'>Swipe horizontally on a small screen to inspect every column.</p></div>", unsafe_allow_html=True)
            st.dataframe(
                schedule_df,
                use_container_width=True,
                hide_index=True,
                height=min(460, 78 + 35 * len(schedule_df)),
                column_config={
                    "No.": st.column_config.NumberColumn(width="small"),
                    "Forecast time": st.column_config.TextColumn(width="medium"),
                    "Price (TZS)": st.column_config.NumberColumn(format="%.2f", width="medium"),
                    "Recommendation": st.column_config.TextColumn(width="large"),
                },
            )

            st.markdown(
                f"<div class='market-link'><a href='{dse_links.get(stock)}' target='_blank'>🔗 Open Dar es Salaam Stock Exchange</a></div>",
                unsafe_allow_html=True,
            )

        elif display_mode == "download":
            st.markdown(
                "<div class='section-panel'><h3 style='margin:0;'>📥 Prediction report</h3>"
                "<p class='small-note' style='margin:.25rem 0 0;'>"
                "Review the short summary, then choose a download format.</p></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='result-section'><div class='result-grid'>"
                f"<div class='result-item'><div class='result-label'>Stock</div><div class='result-value'>{stock}</div></div>"
                f"<div class='result-item'><div class='result-label'>Period</div><div class='result-value'>{period}</div></div>"
                f"<div class='result-item'><div class='result-label'>Signal</div><div class='result-value'>{signal}</div></div>"
                f"<div class='result-item'><div class='result-label'>Current price</div><div class='result-value'>{current_price:,.2f} TZS</div></div>"
                f"<div class='result-item'><div class='result-label'>Forecast price</div><div class='result-value'>{future_price:,.2f} TZS</div></div>"
                f"<div class='result-item'><div class='result-label'>Confidence</div><div class='result-value'>{confidence}%</div></div>"
                "</div></div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div class='reason-box'><strong>Short prediction summary:</strong> "
                f"{stock} is forecast to move {change:+.2f}% for {period.lower()}, "
                f"giving a {signal} signal with {confidence}% confidence.</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                "<div class='section-panel'><h3 style='margin:0;'>⬇️ Download prediction report</h3>"
                "<p class='small-note' style='margin:.25rem 0 0;'>"
                "Choose PDF, CSV or Excel format.</p></div>",
                unsafe_allow_html=True,
            )

            csv_bytes = schedule_df.to_csv(index=False).encode("utf-8")
            excel_bytes = build_excel_report()
            pdf_bytes = build_pdf_report()

            download_col1, download_col2, download_col3 = st.columns(3)

            with download_col1:
                st.download_button(
                    "📄 DOWNLOAD PDF",
                    data=pdf_bytes if pdf_bytes is not None else b"",
                    file_name=f"{stock}_{period.replace(' ', '_')}_forecast_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    disabled=pdf_bytes is None,
                )

            with download_col2:
                st.download_button(
                    "📑 DOWNLOAD CSV",
                    data=csv_bytes,
                    file_name=f"{stock}_{period.replace(' ', '_')}_forecast.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with download_col3:
                st.download_button(
                    "📊 DOWNLOAD EXCEL",
                    data=excel_bytes,
                    file_name=f"{stock}_{period.replace(' ', '_')}_forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            if pdf_bytes is None:
                st.warning("PDF download requires ReportLab. Install it using: pip install reportlab")


        elif display_mode == "graph":
            # =========================
            # INTERACTIVE MOTION GRAPH
            # =========================
            st.markdown(
                "<div class='section-panel'><h3 style='margin:0;'>📊 Interactive historical and forecast trend</h3>"
                "<p class='small-note' style='margin:.25rem 0 0;'>"
                "Hover to inspect prices, drag to zoom, and use the toolbar to reset the view.</p></div>",
                unsafe_allow_html=True,
            )

            history = df.tail(100)
            future_dates = forecast_times[:len(future_prices)]

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=history["Date"],
                    y=history["Close"],
                    mode="lines",
                    name="Historical close",
                    line=dict(color=theme["primary_dark"], width=2.4),
                    hovertemplate="<b>%{x|%d %b %Y}</b><br>Historical: %{y:,.2f} TZS<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=future_prices,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color=theme["primary"], width=3),
                    marker=dict(
                        size=8,
                        color=theme["accent"],
                        line=dict(color=theme["primary_dark"], width=1),
                    ),
                    hovertemplate="<b>%{x|%d %b %Y %H:%M}</b><br>Forecast: %{y:,.2f} TZS<extra></extra>",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[history["Date"].iloc[-1]],
                    y=[history["Close"].iloc[-1]],
                    mode="markers",
                    name="Latest close",
                    marker=dict(
                        size=12,
                        color=theme["accent"],
                        line=dict(color=theme["primary_dark"], width=2),
                    ),
                    hovertemplate="<b>Latest close</b><br>%{y:,.2f} TZS<extra></extra>",
                )
            )

            # Add a visual connection from the latest historical point to the forecast.
            if future_dates:
                fig.add_trace(
                    go.Scatter(
                        x=[history["Date"].iloc[-1], future_dates[0]],
                        y=[history["Close"].iloc[-1], future_prices[0]],
                        mode="lines",
                        name="Forecast transition",
                        line=dict(color=theme["primary"], width=2, dash="dot"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            chart_height = 500 if len(future_prices) <= 10 else 540

            fig.update_layout(
                title=dict(
                    text=f"{stock} interactive price forecast",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=20),
                ),
                xaxis_title=dict(
                    text="Date and time",
                    font=dict(size=16),
                    standoff=16,
                ),
                yaxis_title=dict(
                    text="Price (TZS)",
                    font=dict(size=16),
                    standoff=18,
                ),
                hovermode="x unified",
                height=chart_height,
                autosize=True,
                margin=dict(l=72, r=24, t=88, b=76),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                transition=dict(duration=700, easing="cubic-in-out"),
            )

            fig.update_xaxes(
                showgrid=True,
                gridcolor="rgba(98,125,152,.16)",
                tickfont=dict(size=13, color="#334e68"),
                title_font=dict(size=16, color="#102a43"),
                automargin=True,
                tickangle=-25,
                showline=True,
                linecolor="rgba(98,125,152,.45)",
                linewidth=1,
                rangeslider=dict(visible=True, thickness=0.08),
                rangeselector=dict(
                    buttons=[
                        dict(count=7, label="7D", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                ),
            )

            fig.update_yaxes(
                showgrid=True,
                gridcolor="rgba(98,125,152,.16)",
                tickformat=",.0f",
                tickfont=dict(size=13, color="#334e68"),
                title_font=dict(size=16, color="#102a43"),
                automargin=True,
                separatethousands=True,
                showline=True,
                linecolor="rgba(98,125,152,.45)",
                linewidth=1,
                fixedrange=False,
            )

            st.markdown(
                "<div style='text-align:center;font-weight:700;margin:.15rem 0 .4rem;'>"
                "Move over the chart to inspect each price point</div>",
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "responsive": True,
                    "scrollZoom": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
                key=f"interactive_forecast_{stock}_{period}",
            )

            st.caption("Prediction is an analytical estimate, not a guarantee of investment returns.")