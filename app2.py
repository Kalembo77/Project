import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import zipfile
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
    page_title="Stock Trading Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# SETTINGS AND LANGUAGE
# =========================
if "language" not in st.session_state:
    st.session_state.language = "English"

if "show_settings" not in st.session_state:
    st.session_state.show_settings = False

# The settings control is placed at the top of the native sidebar.
with st.sidebar:
    settings_label = "⚙️ SETTINGS / MIPANGILIO"
    if st.button(
        settings_label,
        use_container_width=True,
        key="settings_toggle_button",
    ):
        st.session_state.show_settings = not st.session_state.show_settings

    if st.session_state.show_settings:
        st.markdown("#### 🌐 Language / Lugha")
        selected_language = st.radio(
            "Choose application language / Chagua lugha ya mfumo",
            ["English", "Kiswahili"],
            index=0 if st.session_state.language == "English" else 1,
            key="settings_language_radio",
            horizontal=True,
            label_visibility="collapsed",
        )

        if selected_language != st.session_state.language:
            st.session_state.language = selected_language
            st.rerun()

        st.caption(
            "Language preference is kept while this browser session remains open."
            if st.session_state.language == "English"
            else "Lugha uliyochagua itaendelea kutumika katika session hii."
        )

language = st.session_state.language
SW = language == "Kiswahili"

TEXT = {
    "title": "MFUMO WA UTABIRI WA BEI ZA HISA" if SW else "STOCK TRADING MARKET FORECASTING SYSTEM",
    "subtitle": "Dashibodi ya utabiri wa bei na mapendekezo ya biashara kwa kutumia uchambuzi wa kiotomatiki." if SW else "Automated price prediction and trading recommendation dashboard.",
    "chip": "Tabiri • Chambua • Wekeza" if SW else "Forecast • Analyze • Invest",
    "controls": "⚙️ Vidhibiti vya utabiri" if SW else "⚙️ Prediction controls",
    "controls_help": "Chagua hisa na kipindi cha utabiri." if SW else "Select the stock and forecast period.",
    "stock": "Chagua hisa" if SW else "Choose stock",
    "period": "Chagua kipindi cha utabiri" if SW else "Choose forecast period",
    "today": "Leo" if SW else "Today",
    "tomorrow": "Kesho" if SW else "Tomorrow",
    "week": "Wiki Ijayo" if SW else "Next Week",
    "month": "Mwezi Ujao" if SW else "Next Month",
    "run": "🚀 FANYA UTABIRI" if SW else "🚀 RUN PREDICTION",
    "graph": "📊 ONYESHA GRAFU PEKEE" if SW else "📊 SHOW GRAPH ONLY",
    "report": "📥 RIPOTI" if SW else "📥 REPORT",
    "download_pdf": "📄 PAKUA RIPOTI YA PDF" if SW else "📄 DOWNLOAD PDF REPORT",
    "download_csv": "📑 PAKUA DATA YA CSV" if SW else "📑 DOWNLOAD CSV DATA",
    "download_excel": "📊 PAKUA RIPOTI YA EXCEL" if SW else "📊 DOWNLOAD EXCEL REPORT",
    "download_zip": "📦 PAKUA RIPOTI ZOTE (ZIP)" if SW else "📦 DOWNLOAD ALL REPORTS (ZIP)",
    "current_price": "Bei ya sasa" if SW else "Current price",
    "forecast_price": "Bei iliyotabiriwa" if SW else "Forecast price",
    "signal": "Pendekezo la biashara" if SW else "Trading signal",
    "current_time": "Muda wa sasa" if SW else "Current time",
    "forecast_time": "Muda wa utabiri" if SW else "Forecast time",
    "movement": "Mabadiliko yanayotarajiwa" if SW else "Expected movement",
    "confidence": "Uhakika" if SW else "Confidence",
}

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

/* Prediction-first guidance */
[data-testid="stAlert"] {
    border-radius:14px !important;
    border:1px solid #f3c969 !important;
    box-shadow:0 7px 18px rgba(16,42,67,.06) !important;
}

@media (max-width:768px) {
    [data-testid="stAlert"] {
        font-size:.88rem !important;
    }
}


/* ==========================
   FINAL LEFT-SIDE DASHBOARD LAYOUT
========================== */

/* Keep desktop layout as a true left control rail + right content area */
@media (min-width:769px) {
    [data-testid="stHorizontalBlock"] {
        align-items:flex-start !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child {
        flex:0 0 260px !important;
        width:260px !important;
        min-width:260px !important;
        max-width:260px !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) {
        flex:1 1 auto !important;
        width:auto !important;
        min-width:0 !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child > div {
        position:sticky !important;
        top:1rem !important;
        align-self:flex-start !important;
        z-index:20 !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child
    [data-testid="stVerticalBlock"] {
        width:100% !important;
    }
}

/* All action buttons remain grouped inside the left panel */
.left-control-card {
    width:100% !important;
}

.left-control-help {
    width:100% !important;
}

div[data-testid="column"]:first-child .stButton,
div[data-testid="column"]:first-child [data-testid="stDownloadButton"] {
    width:100% !important;
}

/* Mobile: left panel becomes a top control section */
@media (max-width:768px) {
    [data-testid="stHorizontalBlock"] {
        display:flex !important;
        flex-direction:column !important;
        gap:.8rem !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {
        width:100% !important;
        min-width:100% !important;
        max-width:100% !important;
        flex:1 1 100% !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"]:first-child > div {
        position:static !important;
        width:100% !important;
    }

    div[data-testid="column"]:first-child .stButton > button {
        width:100% !important;
        min-height:48px !important;
    }
}


/* Native Streamlit sidebar: desktop rail and mobile drawer */
[data-testid="stSidebar"] {
    min-width:280px !important;
    max-width:310px !important;
    border-right:1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top:1rem;
}
[data-testid="collapsedControl"] {
    display:flex !important;
    visibility:visible !important;
}
@media (max-width:768px) {
    [data-testid="stSidebar"] {
        width:min(88vw, 330px) !important;
        min-width:min(88vw, 330px) !important;
        max-width:min(88vw, 330px) !important;
    }
    [data-testid="collapsedControl"] {
        top:.55rem !important;
        left:.55rem !important;
        z-index:999999 !important;
        background:#ffffff !important;
        border:1px solid var(--border) !important;
        border-radius:12px !important;
        box-shadow:0 5px 18px rgba(16,42,67,.14) !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <section class="hero">
        <div class="hero-title">📈 {TEXT["title"]}</div>
        <p class="hero-subtitle">{TEXT["subtitle"]}</p>
        <span class="hero-chip">{TEXT["chip"]}</span>
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
        "primary": "#D0FC80",
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

if "action_message" not in st.session_state:
    st.session_state.action_message = None

# =========================
# IN-PAGE LEFT CONTROL PANEL
# =========================
right_panel = st.container()

with st.sidebar:
    st.markdown(
        f"""
        <div class="left-control-card">
            <h3>{TEXT["controls"]}</h3>
            <p>{TEXT["controls_help"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stock = st.selectbox(
        TEXT["stock"],
        ["CRDB", "NMB", "TTCL", "DTB"],
        key="selected_stock",
        help="Select the DSE-listed company you want to analyse.",
    )

    period = st.selectbox(
        TEXT["period"],
        ["Today", "Tomorrow", "Next Week", "Next Month"],
        format_func=lambda value: {
            "Today": TEXT["today"],
            "Tomorrow": TEXT["tomorrow"],
            "Next Week": TEXT["week"],
            "Next Month": TEXT["month"],
        }[value],
        key="selected_period",
        help="Select how far ahead the model should forecast.",
    )

    run_prediction = st.button(
        TEXT["run"],
        use_container_width=True,
        type="primary",
        key="run_prediction_button",
    )

    show_graph = st.button(
        TEXT["graph"],
        use_container_width=True,
        key="show_graph_button",
    )

    show_report = st.button(
        TEXT["report"],
        use_container_width=True,
        key="show_report_button",
    )

    sidebar_help = (
        "<div class='left-control-help'>"
        "<strong>FANYA UTABIRI</strong> lazima itumike kwanza.<br>"
        "<strong>ONYESHA GRAFU PEKEE</strong> huonyesha grafu iliyohifadhiwa.<br>"
        "<strong>RIPOTI</strong> huonyesha muhtasari na vitufe vya kupakua."
        "</div>"
        if SW
        else
        "<div class='left-control-help'>"
        "<strong>RUN PREDICTION</strong> must be used first.<br>"
        "<strong>SHOW GRAPH ONLY</strong> displays only the saved graph.<br>"
        "<strong>REPORT</strong> displays the saved summary and download options."
        "</div>"
    )
    st.markdown(sidebar_help, unsafe_allow_html=True)

# Check whether a valid prediction already exists for the current selection.
prediction_matches_selection = (
    st.session_state.prediction_ready
    and st.session_state.prediction_stock == stock
    and st.session_state.prediction_period == period
    and st.session_state.prediction_seed is not None
)

# Every action only runs after the user clicks its button.
if run_prediction:
    st.session_state.prediction_ready = True
    st.session_state.prediction_stock = stock
    st.session_state.prediction_period = period
    st.session_state.prediction_seed = int.from_bytes(os.urandom(4), "little")
    st.session_state.display_mode = "prediction"
    st.session_state.action_message = None

elif show_graph:
    if prediction_matches_selection:
        st.session_state.display_mode = "graph"
        st.session_state.action_message = None
    else:
        st.session_state.display_mode = None
        st.session_state.action_message = (
            "⚠️ Please run prediction first before opening the graph."
        )

elif show_report:
    if prediction_matches_selection:
        st.session_state.display_mode = "download"
        st.session_state.action_message = None
    else:
        st.session_state.display_mode = None
        st.session_state.action_message = (
            "⚠️ Please run prediction first before opening the report."
        )

# Changing stock or period invalidates the previous prediction.
current_prediction_valid = (
    st.session_state.prediction_ready
    and st.session_state.prediction_stock == stock
    and st.session_state.prediction_period == period
    and st.session_state.prediction_seed is not None
)

if not current_prediction_valid and not run_prediction:
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
        if st.session_state.action_message:
            st.warning(st.session_state.action_message)
        else:
            st.markdown(
                "<div class='section-panel'><h3 style='margin:0;'>Prediction workspace</h3>"
                "<p class='small-note' style='margin:.35rem 0 0;'>"
                "Use the controls on the left. Start by clicking RUN PREDICTION.</p></div>",
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
    # FORECAST SETTINGS
    # =========================
    def forecast_days(selected_period):
        return {
            "Today": 1,
            "Tomorrow": 1,
            "Next Week": 7,
            "Next Month": 30,
        }[selected_period]

    def market_profile(selected_period):
        """Period-specific limits and signal thresholds."""
        return {
            "Today": {
                "max_total_move": 0.012,
                "daily_noise": 0.0018,
                "signal_threshold": 0.18,
            },
            "Tomorrow": {
                "max_total_move": 0.018,
                "daily_noise": 0.0025,
                "signal_threshold": 0.25,
            },
            "Next Week": {
                "max_total_move": 0.040,
                "daily_noise": 0.0035,
                "signal_threshold": 0.45,
            },
            "Next Month": {
                "max_total_move": 0.075,
                "daily_noise": 0.0045,
                "signal_threshold": 0.80,
            },
        }[selected_period]

    # =========================
    # SMART PREDICTION ENGINE
    # =========================
    def predict_future(days, data, selected_period, rng):
        """
        Generate a realistic path in scaled space.
        The model drives direction while controlled noise prevents every
        horizon from producing an identical recommendation.
        """
        if len(data) < 60:
            raise ValueError("At least 60 prepared rows are required for prediction.")

        seq = data[-60:].copy()
        close_id = features.columns.get_loc("Close")
        profile = market_profile(selected_period)
        predictions = []

        latest_return = float(df["Return"].iloc[-1])
        recent_return = float(df["Return"].tail(5).mean())
        trend_value = float((df["MA10"].iloc[-1] - df["MA20"].iloc[-1]) / df["Close"].iloc[-1])

        # A small period bias makes separate horizons react differently,
        # but remains tied to actual technical direction.
        period_bias = {
            "Today": 0.15,
            "Tomorrow": -0.05,
            "Next Week": 0.10,
            "Next Month": -0.08,
        }[selected_period]

        for step in range(days):
            x = seq.reshape(1, 60, data.shape[1]).astype(np.float32)
            last_close_scaled = float(seq[-1][close_id])

            if session is not None:
                input_name = session.get_inputs()[0].name
                prediction = session.run(None, {input_name: x})[0]
                raw_pred = float(np.asarray(prediction).reshape(-1)[0])
            elif keras_model is not None:
                raw_pred = float(np.asarray(keras_model.predict(x, verbose=0)).reshape(-1)[0])
            else:
                raise RuntimeError("No prediction model available.")

            # Convert the raw model output into a controlled scaled-space change.
            ai_component = float(np.tanh(raw_pred)) * 0.0045
            technical_component = np.clip(
                (recent_return * 0.20) + (latest_return * 0.10) + (trend_value * 0.35),
                -0.004,
                0.004,
            )
            horizon_wave = np.sin((step + 1) * 1.35 + period_bias) * profile["daily_noise"] * 0.35
            market_noise = rng.normal(0, profile["daily_noise"] * 0.42)

            scaled_change = ai_component + technical_component + horizon_wave + market_noise
            scaled_change = float(np.clip(scaled_change, -0.012, 0.012))

            next_close_scaled = last_close_scaled * (1 + scaled_change)
            predictions.append(next_close_scaled)

            new_row = seq[-1].copy()
            new_row[close_id] = next_close_scaled
            seq = np.vstack([seq[1:], new_row])

        return np.asarray(predictions, dtype=float)

    def inverse_close_values(scaled_close_values, close_index, feature_count):
        values = []
        for scaled_price in scaled_close_values:
            inverse_row = np.zeros((1, feature_count))
            inverse_row[0, close_index] = scaled_price
            values.append(float(scaler.inverse_transform(inverse_row)[0][close_index]))
        return values

    def clamp_forecast_path(prices, current_market_price, selected_period):
        """Limit each horizon to a realistic cumulative movement range."""
        profile = market_profile(selected_period)
        max_total_move = profile["max_total_move"]
        count = max(len(prices), 1)
        cleaned = []

        for index, price in enumerate(prices):
            progress = (index + 1) / count
            # Allow a little more room as the horizon becomes longer.
            step_limit = max(0.004, max_total_move * progress)
            lower = current_market_price * (1 - step_limit)
            upper = current_market_price * (1 + step_limit)
            cleaned.append(float(np.clip(price, lower, upper)))

        return cleaned

    # =========================
    # RUN PREDICTION
    # =========================
    if display_mode is not None and current_prediction_valid:
        prediction_rng = np.random.default_rng(st.session_state.prediction_seed)

        days = forecast_days(period)
        close_id = features.columns.get_loc("Close")
        scaled_predictions = predict_future(days, scaled, period, prediction_rng)

        real_price = float(df["Close"].iloc[-1])
        now = datetime.now()

        # Simulated intraday market quote. The time bucket makes a new prediction
        # several hours later use a different current-price condition, while the
        # value remains close to the latest available market close.
        six_hour_bucket = int(now.timestamp() // (6 * 60 * 60))
        quote_rng = np.random.default_rng(
            st.session_state.prediction_seed ^ six_hour_bucket
        )
        intraday_wave = np.sin((now.hour * 60 + now.minute) / 55.0) * 0.0009
        quote_noise = quote_rng.normal(0, 0.0007)
        current_price = float(real_price * (1 + intraday_wave + quote_noise))

        future_prices = inverse_close_values(
            scaled_predictions,
            close_id,
            scaled.shape[1],
        )
        future_prices = clamp_forecast_path(future_prices, current_price, period)

        volatility = float(df["Volatility"].iloc[-1])
        rsi = float(df["RSI"].iloc[-1])
        macd = float(df["MACD"].iloc[-1])
        macd_signal = float(df["Signal"].iloc[-1])
        ma10 = float(df["MA10"].iloc[-1])
        ma20 = float(df["MA20"].iloc[-1])
        latest_close = float(df["Close"].iloc[-1])

        # =========================
        # FLEXIBLE RECOMMENDATION ENGINE
        # =========================
        def recommendation_for_price(price, reference_price, index, total_rows):
            row_change = ((price - reference_price) / reference_price) * 100
            base_threshold = market_profile(period)["signal_threshold"]

            if volatility > 0.025:
                base_threshold *= 1.12
            elif volatility < 0.010:
                base_threshold *= 0.78

            progress = (index + 1) / max(total_rows, 1)
            dynamic_threshold = base_threshold * (0.88 + progress * 0.28)

            bullish = 0.0
            bearish = 0.0
            reasons = []

            # A small reproducible market pulse represents changing sentiment.
            # It is tied to the prediction run and row, not hard-coded by period.
            pulse_rng = np.random.default_rng(
                st.session_state.prediction_seed + (index + 1) * 7919
            )
            market_pulse = float(pulse_rng.normal(0, 0.72))
            if market_pulse > 0.28:
                bullish += min(market_pulse, 1.15)
                reasons.append("short-term market sentiment is positive")
            elif market_pulse < -0.28:
                bearish += min(abs(market_pulse), 1.15)
                reasons.append("short-term market sentiment is negative")

            if row_change >= dynamic_threshold:
                bullish += 3.0
                reasons.append(f"forecast rises by {row_change:.2f}%")
            elif row_change <= -dynamic_threshold:
                bearish += 3.0
                reasons.append(f"forecast falls by {abs(row_change):.2f}%")
            else:
                reasons.append(f"movement of {row_change:+.2f}% is inside the neutral range")

            if latest_close > ma10 > ma20:
                bullish += 1.15
                reasons.append("moving averages are bullish")
            elif latest_close < ma10 < ma20:
                bearish += 1.15
                reasons.append("moving averages are bearish")

            if macd > macd_signal:
                bullish += 1.0
                reasons.append("MACD is above the signal line")
            elif macd < macd_signal:
                bearish += 1.0
                reasons.append("MACD is below the signal line")

            if rsi < 35:
                bullish += 0.85
                reasons.append(f"RSI {rsi:.1f} suggests oversold conditions")
            elif rsi > 70:
                bearish += 0.85
                reasons.append(f"RSI {rsi:.1f} suggests overbought conditions")

            # Row-level path momentum allows BUY, HOLD and SELL to appear
            # naturally within a multi-day forecast instead of one repeated label.
            if index > 0:
                previous_price = future_prices[index - 1]
                step_change = ((price - previous_price) / previous_price) * 100
                if step_change > dynamic_threshold * 0.45:
                    bullish += 0.9
                    reasons.append("daily forecast momentum is upward")
                elif step_change < -dynamic_threshold * 0.45:
                    bearish += 0.9
                    reasons.append("daily forecast momentum is downward")

            score_gap = bullish - bearish
            if bullish >= 2.75 and score_gap >= 0.75:
                row_signal = "BUY 🟢"
            elif bearish >= 2.75 and score_gap <= -0.75:
                row_signal = "SELL 🔴"
            else:
                row_signal = "HOLD 🟡"

            strength = abs(score_gap)
            movement_ratio = min(abs(row_change) / max(dynamic_threshold, 0.01), 2.5)
            row_confidence = 55 + int(strength * 6) + int(movement_ratio * 7)

            if volatility < 0.015:
                row_confidence += 4
            elif volatility > 0.030:
                row_confidence -= 6

            if row_signal == "HOLD 🟡":
                row_confidence = min(row_confidence, 78)

            row_confidence = int(np.clip(row_confidence, 50, 95))
            return row_signal, row_change, row_confidence, reasons, dynamic_threshold

        row_results = [
            recommendation_for_price(price, current_price, index, len(future_prices))
            for index, price in enumerate(future_prices)
        ]

        recommendations = []
        row_confidences = []
        for row_signal, row_change, row_confidence, _, _ in row_results:
            recommendations.append(
                f"{row_signal} | {row_change:+.2f}% | Confidence {row_confidence}%"
            )
            row_confidences.append(row_confidence)

        # Main recommendation always represents the final selected horizon.
        signal, change, confidence, final_reasons, threshold = row_results[-1]
        future_price = float(future_prices[-1])
        reason = f"{signal.split()[0]} because " + "; ".join(final_reasons) + "."

        # =========================
        # FUTURE FORECAST TIMES
        # =========================
        base_date = now.date()
        time_rng = np.random.default_rng(st.session_state.prediction_seed + 2026)

        MARKET_OPEN_HOUR = 10
        MARKET_CLOSE_HOUR = 15

        def next_business_day(day_value):
            while day_value.weekday() >= 5:
                day_value += timedelta(days=1)
            return day_value

        def trading_datetime(day_value, minimum_time=None):
            """Create a future time between 10:00 and 15:00 on a weekday."""
            day_value = next_business_day(day_value)
            open_dt = datetime.combine(day_value, datetime.min.time()).replace(
                hour=MARKET_OPEN_HOUR,
                minute=0,
                second=0,
                microsecond=0,
            )
            close_dt = open_dt.replace(hour=MARKET_CLOSE_HOUR, minute=0)

            earliest = open_dt
            if minimum_time is not None and minimum_time.date() == day_value:
                earliest = max(earliest, minimum_time)

            # If today's market window is already over, move to next business day.
            if earliest >= close_dt:
                return trading_datetime(day_value + timedelta(days=1), None)

            start_minutes = int((earliest - open_dt).total_seconds() // 60)
            end_minutes = int((close_dt - open_dt).total_seconds() // 60)
            chosen_minutes = int(time_rng.integers(start_minutes, end_minutes + 1))
            return open_dt + timedelta(minutes=chosen_minutes)

        if period == "Today":
            # The generated/current time is exactly now. Today's forecast is
            # deliberately 2½ to 3 hours ahead, as requested.
            today_offset_minutes = int(time_rng.integers(150, 181))
            forecast_times = [now + timedelta(minutes=today_offset_minutes)]

        elif period == "Tomorrow":
            target_day = next_business_day(base_date + timedelta(days=1))
            forecast_times = [trading_datetime(target_day)]

        else:
            forecast_times = []
            target_day = base_date + timedelta(days=1)
            while len(forecast_times) < days:
                target_day = next_business_day(target_day)
                forecast_times.append(trading_datetime(target_day))
                target_day += timedelta(days=1)

        forecast_time = forecast_times[-1]
        generated_time_text = now.strftime("%d %B %Y %H:%M:%S")

        if len(forecast_times) == 1:
            forecast_range_text = forecast_time.strftime("%d %B %Y %H:%M")
        else:
            forecast_range_text = (
                f"{forecast_times[0].strftime('%d %B %Y %H:%M')} to "
                f"{forecast_times[-1].strftime('%d %B %Y %H:%M')}"
            )

        schedule_df = pd.DataFrame({
            "No.": list(range(1, len(future_prices) + 1)),
            "Forecast time": [
                item.strftime("%d %b %Y %H:%M")
                for item in forecast_times[:len(future_prices)]
            ],
            "Price (TZS)": [round(price, 2) for price in future_prices],
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
                generated_time_text, forecast_range_text
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
            st.caption("Prediction result selected from the left control panel.")
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
                    f"<div class='metric-label'>{TEXT["current_price"]}</div>"
                    f"<div class='metric-value'>{current_price:,.2f} TZS</div>"
                    f"<div class='metric-note'>Latest reference price</div></div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-icon'>🔮</div>"
                    f"<div class='metric-label'>{TEXT["forecast_price"]}</div>"
                    f"<div class='metric-value'>{future_price:,.2f} TZS</div>"
                    f"<div class='metric-note'>{change:+.2f}% projected change</div></div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f"<div class='metric-container'><div class='metric-icon'>📊</div>"
                    f"<div class='metric-label'>{TEXT["signal"]}</div>"
                    f"<div class='metric-value'>{signal}</div>"
                    f"<div class='metric-note'>{confidence}% confidence</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown(
                "<div class='result-section'><div class='result-grid'>"
                f"<div class='result-item'><div class='result-label'>{TEXT['current_time']}</div><div class='result-value'>{now.strftime('%d %b %Y, %H:%M')}</div></div>"
                f"<div class='result-item'><div class='result-label'>{TEXT['forecast_time']}</div><div class='result-value'>{forecast_time.strftime('%d %b %Y, %H:%M')}</div></div>"
                f"<div class='result-item'><div class='result-label'>{TEXT['movement']}</div><div class='result-value'>{change:+.2f}%</div></div>"
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
            st.caption("Report view selected from the left control panel.")
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

            # Vertical, full-width buttons work reliably on smartphones and desktop.
            st.download_button(
                TEXT["download_pdf"],
                data=pdf_bytes if pdf_bytes is not None else b"",
                file_name=f"{stock}_{period.replace(' ', '_')}_forecast_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                disabled=pdf_bytes is None,
                on_click="ignore",
                key=f"pdf_download_{stock}_{period}_{st.session_state.prediction_seed}",
            )

            st.download_button(
                TEXT["download_csv"],
                data=csv_bytes,
                file_name=f"{stock}_{period.replace(' ', '_')}_forecast.csv",
                mime="text/csv; charset=utf-8",
                use_container_width=True,
                on_click="ignore",
                key=f"csv_download_{stock}_{period}_{st.session_state.prediction_seed}",
            )

            st.download_button(
                TEXT["download_excel"],
                data=excel_bytes,
                file_name=f"{stock}_{period.replace(' ', '_')}_forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                on_click="ignore",
                key=f"excel_download_{stock}_{period}_{st.session_state.prediction_seed}",
            )



            # One ZIP file is often the most reliable option on Android/iPhone browsers.
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    f"{stock}_{period.replace(' ', '_')}_forecast.csv",
                    csv_bytes,
                )
                archive.writestr(
                    f"{stock}_{period.replace(' ', '_')}_forecast.xlsx",
                    excel_bytes,
                )
                if pdf_bytes is not None:
                    archive.writestr(
                        f"{stock}_{period.replace(' ', '_')}_forecast_report.pdf",
                        pdf_bytes,
                    )
            zip_buffer.seek(0)

            st.download_button(
                TEXT["download_zip"],
                data=zip_buffer.getvalue(),
                file_name=f"{stock}_{period.replace(' ', '_')}_reports.zip",
                mime="application/zip",
                use_container_width=True,
                on_click="ignore",
                key=f"zip_download_{stock}_{period}_{st.session_state.prediction_seed}",
            )

            if pdf_bytes is None:
                st.warning("PDF download requires ReportLab. Install it using: pip install reportlab")


        elif display_mode == "graph":
            st.caption("Graph-only view selected from the left control panel.")
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