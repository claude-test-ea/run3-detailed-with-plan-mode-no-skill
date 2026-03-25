"""Streamlit dashboard entry point for Loan Eligibility Prediction.

Gilded Observatory theme — a dark, warm luxury fintech interface
with art deco geometric touches and luminous data visualizations.

Launch with:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import pickle
import os

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Loan Eligibility Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Gilded Observatory: dark luxury fintech aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Google Fonts ---- */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Sora:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---- CSS Custom Properties ---- */
    :root {
        --bg-deep: #0B0B0F;
        --bg-surface: #16161D;
        --border: #2A2A35;
        --text-primary: #E8E4DD;
        --text-secondary: #8B8680;
        --accent-gold: #C9A96E;
        --accent-teal: #00B4D8;
        --color-approved: #00E5A0;
        --color-rejected: #FF6161;
        --sidebar-bg: #0D0D12;
        --font-heading: 'Cormorant Garamond', serif;
        --font-body: 'Sora', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    /* ---- Global ---- */
    html, body, [class*="css"] {
        font-family: var(--font-body);
        font-weight: 400;
        color: var(--text-primary);
    }

    .stApp {
        background: var(--bg-deep);
    }

    /* ---- Fade-in animation on page load ---- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(16px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .main .block-container {
        animation: fadeInUp 0.6s ease-out both;
    }

    /* ---- Headings ---- */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-heading) !important;
        color: var(--text-primary) !important;
        letter-spacing: 0.01em;
    }
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        border-bottom: none !important;
        padding-bottom: 0.2rem;
        margin-bottom: 0.4rem !important;
    }
    h2 {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-top: 1.6rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
    }

    /* ---- Paragraph / body text ---- */
    p, span, label, .stMarkdown {
        font-family: var(--font-body);
        color: var(--text-primary);
    }

    /* ---- Tabs — Art Deco inspired ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: var(--font-body);
        font-weight: 500;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
        padding: 0.7rem 1.4rem;
        border-radius: 0;
        color: var(--text-secondary);
        background: transparent;
        border-bottom: 2px solid transparent;
        transition: all 0.25s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(201, 169, 110, 0.04);
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent-gold) !important;
        border-bottom: 2px solid var(--accent-gold) !important;
        background: rgba(201, 169, 110, 0.06) !important;
    }

    /* ---- Metric Cards — Glassmorphic ---- */
    [data-testid="stMetric"] {
        background: rgba(22, 22, 29, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-top: 2px solid var(--accent-gold);
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-body);
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-secondary) !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono);
        font-weight: 500;
        font-size: 1.5rem;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: var(--font-mono);
    }

    /* ---- Sidebar — Dark with geometric accent ---- */
    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-gold), transparent 70%);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: var(--text-primary) !important;
    }
    section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div {
        background: var(--accent-gold) !important;
    }

    /* ---- Dataframes — Dark themed ---- */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border);
    }
    .stDataFrame [data-testid="stDataFrameResizable"] {
        background: var(--bg-surface);
    }

    /* ---- Text Input / Select / Slider — Dark themed ---- */
    .stTextInput input,
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"] {
        background-color: var(--bg-surface) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body);
        border-radius: 6px;
    }
    .stTextInput input:focus {
        border-color: var(--accent-gold) !important;
        box-shadow: 0 0 0 1px var(--accent-gold);
    }
    .stSelectbox [data-baseweb="select"]:focus-within {
        border-color: var(--accent-gold) !important;
    }

    /* ---- Slider styling ---- */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--accent-gold) !important;
        border-color: var(--accent-gold) !important;
    }

    /* ---- Download Button — Gold accent ---- */
    .stDownloadButton button {
        background: rgba(201, 169, 110, 0.12) !important;
        color: var(--accent-gold) !important;
        border: 1px solid var(--accent-gold) !important;
        border-radius: 6px;
        font-family: var(--font-body);
        font-weight: 500;
        font-size: 0.85rem;
        letter-spacing: 0.04em;
        padding: 0.55rem 1.6rem;
        transition: all 0.25s ease;
    }
    .stDownloadButton button:hover {
        background: var(--accent-gold) !important;
        color: var(--bg-deep) !important;
    }

    /* ---- Regular Buttons ---- */
    .stButton button {
        background: rgba(201, 169, 110, 0.08) !important;
        color: var(--accent-gold) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px;
        font-family: var(--font-body);
        transition: all 0.25s ease;
    }
    .stButton button:hover {
        border-color: var(--accent-gold) !important;
        background: rgba(201, 169, 110, 0.15) !important;
    }

    /* ---- Code blocks ---- */
    .stCodeBlock, code, pre {
        font-family: var(--font-mono) !important;
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border);
        border-radius: 6px;
    }

    /* ---- Expander ---- */
    .streamlit-expanderHeader {
        font-family: var(--font-body);
        font-weight: 500;
        color: var(--text-primary) !important;
        background: rgba(22, 22, 29, 0.5);
        border: 1px solid var(--border);
        border-radius: 6px;
    }
    .streamlit-expanderContent {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 6px 6px;
    }

    /* ---- Warning / Info / Success boxes ---- */
    .stAlert {
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--bg-surface);
    }

    /* ---- Custom Scrollbar ---- */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-deep);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }

    /* ---- Selectbox dropdown styling ---- */
    [data-baseweb="popover"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
    }
    [data-baseweb="menu"] {
        background: var(--bg-surface) !important;
    }
    [data-baseweb="menu"] li {
        color: var(--text-primary) !important;
    }
    [data-baseweb="menu"] li:hover {
        background: rgba(201, 169, 110, 0.1) !important;
    }

    /* ---- Plotly chart container ---- */
    .stPlotlyChart {
        border-radius: 8px;
        overflow: hidden;
    }

    /* ---- Divider ---- */
    hr {
        border-color: var(--border) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_full_data():
    """Load the full loan application dataset from data/input.csv."""
    return pd.read_csv(os.path.join(BASE_DIR, "data", "input.csv"))


@st.cache_data
def load_model_results():
    """Load model evaluation results from models/model_results.pkl."""
    path = os.path.join(BASE_DIR, "models", "model_results.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_best_model():
    """Load the best model artefact from models/best_model.pkl."""
    path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = load_full_data()
results = load_model_results()
best_model_data = load_best_model()

# ---------------------------------------------------------------------------
# Title — dramatic art deco header with gold accent line
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 1.6rem;">
        <h1 style="
            font-family: 'Cormorant Garamond', serif;
            font-weight: 700;
            font-size: 2.4rem;
            color: #E8E4DD;
            margin-bottom: 0.3rem;
            letter-spacing: 0.02em;
        ">Loan Eligibility Prediction</h1>
        <div style="
            width: 80px;
            height: 3px;
            background: linear-gradient(90deg, #C9A96E, transparent);
            margin-bottom: 0.6rem;
        "></div>
        <p style="
            font-family: 'Sora', sans-serif;
            font-size: 0.9rem;
            font-weight: 300;
            color: #8B8680;
            letter-spacing: 0.01em;
        ">Explore the dataset, review model performance, and analyse predictions on the hold-out set.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Data Explorer",
        "📈 EDA & Visualizations",
        "🤖 Model Performance",
        "🔍 Model Deep Dive",
        "🎯 Predict on Hold-Out",
    ]
)

from streamlit_app.tabs import (
    data_explorer,
    eda_visualizations,
    model_performance,
    model_deep_dive,
    predict_holdout,
)

with tab1:
    data_explorer.render(df)

with tab2:
    eda_visualizations.render(df)

with tab3:
    if results:
        model_performance.render(results)
    else:
        st.warning("Model results not found. Run the ML pipeline first.")

with tab4:
    if results and best_model_data:
        model_deep_dive.render(results, best_model_data)
    else:
        st.warning("Model data not found. Run the ML pipeline first.")

with tab5:
    predict_holdout.render(BASE_DIR)
