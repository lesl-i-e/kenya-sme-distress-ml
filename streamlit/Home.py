import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util, os, sys

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
    .dataframe td { color: #1f2937 !important; }
    #MainMenu { visibility:hidden; }
    footer     { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📊 SME Distress Predictor")
st.sidebar.markdown("**East African Business Risk Assessment**")
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

# ── Overview content (was 1_Overview.py) ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from utils import load_data, load_results, CLASS_COLOURS, CLASS_NAMES

st.title("📊 SME Business Distress Predictor — East Africa")
st.markdown("""
> **What this tool does:** Using World Bank Enterprise Survey data from **14,688 businesses
> across 8 African countries**, this system predicts whether a business profile shows
> signs of **Stable**, **Moderate Risk**, or **High Risk** distress — based on observable
> financial, operational, and management characteristics.
>
> **Who it is for:** Investors, lenders, incubators, and policy makers who need
> evidence-based risk assessment before committing resources to a business.
""")
st.markdown("---")

# ── Dataset overview ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📌 Dataset at a Glance</div>',
            unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Firms",   "14,688", "8 countries")
c2.metric("Survey Rounds", "16",     "2007 – 2025")
c3.metric("Countries",     "8",      "East & West Africa")
c4.metric("Best ROC-AUC",  "0.8636", "XGBoost v2 (genuine)")
c5.metric("Features Used", "32",     "Leakage-free set")

st.markdown("---")

# ── Class distribution ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Business Distress Distribution in Training Data</div>',
            unsafe_allow_html=True)

col_chart, col_text = st.columns([1.3, 1])

with col_chart:
    try:
        df     = load_data()
        counts = df["distress_level"].value_counts().sort_index()
        labels = [CLASS_NAMES[i] for i in counts.index]
        colors = [CLASS_COLOURS[l] for l in labels]

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="none", width=0.55)
        axes[0].set_title("Count by Risk Class", fontsize=11)
        axes[0].set_ylabel("Number of Firms")
        for bar, v in zip(bars, counts.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + 50,
                         f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=8)
        for sp in ["top","right"]: axes[0].spines[sp].set_visible(False)

        axes[1].pie(counts.values, labels=labels, colors=colors,
                    autopct="%1.1f%%", startangle=140, pctdistance=0.78,
                    wedgeprops={"edgecolor":"white","linewidth":1.5})
        axes[1].set_title("Proportional Share", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.info(f"Chart unavailable: {e}")

with col_text:
    st.markdown("""
    **Three risk classes:**

    🔵 **Stable (44.1%)** — No significant distress signals.
    Healthy employment, adequate finance access, and good capacity.

    🟠 **Moderate Risk (46.9%)** — One distress signal present.
    Typically credit-constrained but still operating. Needs monitoring.

    🔴 **High Risk (9.1%)** — Two or more distress signals active
    simultaneously. Strong predictor of business decline. Immediate
    attention required before any funding.

    **The three distress signals (used for labelling only):**
    1. Credit constrained — needs financing but cannot access it
    2. Employment shrinking — workforce fell >10% over 3 years
    3. Low capacity utilisation — operating below 60% capacity
    """)

st.markdown("---")

# ── Country coverage ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌍 Country Coverage</div>',
            unsafe_allow_html=True)

try:
    df = load_data()
    country_stats = df.groupby("country").agg(
        Firms=("at_risk","count"),
        High_Risk_Pct=("at_risk", lambda x: f"{x.mean()*100:.1f}%"),
    ).reset_index()
    country_stats.columns = ["Country","Firms in Dataset","High Risk Rate"]
    st.dataframe(country_stats, use_container_width=True, hide_index=True)
except Exception as e:
    st.info(f"Country table unavailable: {e}")

st.markdown("---")

# ── Model performance summary ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Model Performance Summary</div>',
            unsafe_allow_html=True)

try:
    results = load_results()
    def highlight_best(row):
        model_col = row.get("model", row.get("Model", ""))
        if "xgboost" in str(model_col).lower() and "initial" in str(model_col).lower():
            return ["background-color:#DBEAFE;color:#1e3a8a;font-weight:bold"] * len(row)
        return [""] * len(row)
    st.dataframe(
        results.style.apply(highlight_best, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.caption("★ XGBoost (Initial) v2 — ROC-AUC 0.8636 — is deployed in the Business Predictor.")
except Exception as e:
    st.info(f"Results unavailable: {e}")

st.markdown("---")

# ── Methodology note ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Methodology Note — Leakage-Corrected Model (v2)</div>',
            unsafe_allow_html=True)
st.info("""
**Why ROC-AUC is 0.8636 and not 1.0000:**
An earlier model version achieved near-perfect scores because the three distress signal
features were used both to build the target labels *and* as model inputs — circular
reconstruction, not genuine prediction.

**This v2 model corrects that.** Signal features are excluded from all model inputs.
Nine independent features replace them: labour productivity, finance double stress,
infrastructure burden, obstacle count, firm age, size-age interaction, export
orientation, and management quality.

ROC-AUC **0.8636** across 14,688 firms and 8 African countries is the honest,
publishable number.
""")
st.caption("Data: World Bank Enterprise Surveys 2007–2025 · JKUAT Final Year Project 2026")
