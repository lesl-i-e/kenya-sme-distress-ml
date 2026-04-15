import streamlit as st

st.set_page_config(
    page_title="SME Distress Predictor — East Africa",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0D1B3E; }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] hr { border-color: #1AA3B8 !important; }

    [data-testid="metric-container"] {
        background: #F4F6F8;
        border-radius: 10px;
        padding: 12px 16px;
        border-left: 4px solid #0F7B8C;
    }
    .section-header {
        font-size: 1.05rem; font-weight: 600; color: #0D1B3E;
        border-bottom: 2px solid #0F7B8C;
        padding-bottom: 5px; margin-bottom: 14px;
    }
    .verdict-stable {
        background:#E3F2FD; border-left:6px solid #1565C0;
        border-radius:10px; padding:18px; margin:14px 0;
    }
    .verdict-moderate {
        background:#FFF3E0; border-left:6px solid #E65100;
        border-radius:10px; padding:18px; margin:14px 0;
    }
    .verdict-highrisk {
        background:#FFEBEE; border-left:6px solid #B71C1C;
        border-radius:10px; padding:18px; margin:14px 0;
    }
    .signal-card {
        border-radius:8px; padding:14px; margin:6px 0;
        border:1px solid #E2E8F0;
    }
    #MainMenu { visibility:hidden; }
    footer     { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 SME Distress Predictor")
st.sidebar.markdown("**East African Business Risk Assessment**")
st.sidebar.markdown("---")

pages = {
    "🏠  Overview":                "pages/1_Overview.py",
    "🔍  Business Predictor":      "pages/2_Predictor.py",
    "📈  Model Performance":       "pages/3_Model_Performance.py",
    "🧠  Feature Importance":      "pages/4_SHAP.py",
    "🌍  Geographic Analysis":     "pages/5_Geography.py",
}

st.sidebar.markdown("### Navigate")
selection = st.sidebar.radio("", list(pages.keys()), label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project Details**
- Student: Gedion Leslie Kweya
- Reg: SCT213-C002-0062/2022
- Supervisor: Mr. Adhola Samuel
- Institution: JKUAT
""")
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Data Source**
World Bank Enterprise Surveys
8 countries · 14,688 firms
2007 – 2025
""")
st.sidebar.caption("BIT 2303 / SDS 2406 · 2026")

# ── Route to selected page ────────────────────────────────────────────────────
import importlib.util, os

page_file = pages[selection]
spec = importlib.util.spec_from_file_location("page", page_file)
mod  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)