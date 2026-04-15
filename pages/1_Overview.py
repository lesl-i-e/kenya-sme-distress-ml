"""Page 1 — Overview"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, load_results, CLASS_COLOURS, CLASS_NAMES, style_metric

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
c1.metric("Total Firms",     "14,688",  "8 countries")
c2.metric("Survey Rounds",   "16",      "2007 – 2025")
c3.metric("Countries",       "8",       "East & West Africa")
c4.metric("Best ROC-AUC",    "1.0000",  "Tuned XGBoost")
c5.metric("Features Used",   "31",      "Engineered from survey")

st.markdown("---")

# ── Class distribution ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Business Distress Distribution in Training Data</div>',
            unsafe_allow_html=True)

col_chart, col_text = st.columns([1.3, 1])

with col_chart:
    try:
        df = load_data()
        counts = df["distress_level"].value_counts().sort_index()
        labels = [CLASS_NAMES[i] for i in counts.index]
        colors = [CLASS_COLOURS[l] for l in labels]

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

        # Bar
        bars = axes[0].bar(labels, counts.values, color=colors,
                           edgecolor="none", width=0.55)
        axes[0].set_title("Count by Risk Class", fontsize=11)
        axes[0].set_ylabel("Number of Firms")
        for bar, v in zip(bars, counts.values):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + 50,
                         f"{v:,}\n({v/len(df)*100:.1f}%)",
                         ha="center", fontsize=8)
        for sp in ["top","right"]: axes[0].spines[sp].set_visible(False)

        # Pie
        axes[1].pie(counts.values, labels=labels, colors=colors,
                    autopct="%1.1f%%", startangle=140,
                    pctdistance=0.78,
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
    The business shows healthy employment, adequate access to
    finance, and good operational capacity.

    🟠 **Moderate Risk (46.9%)** — One distress signal present.
    Typically credit-constrained but still operating. Requires
    monitoring and possibly targeted support.

    🔴 **High Risk (9.1%)** — Two or more distress signals active
    simultaneously. Strong predictor of business decline or exit.
    Immediate attention required before funding.

    **The three distress signals:**
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

# ── Model performance teaser ──────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Model Performance Summary</div>',
            unsafe_allow_html=True)

try:
    results = load_results()
    def highlight(row):
        if "tuned" in str(row.get("Model","")).lower():
            return ["background-color:#E3F2FD"] * len(row)
        return [""] * len(row)
    st.dataframe(
        results.style.apply(highlight, axis=1),
        use_container_width=True, hide_index=True,
    )
    st.caption("★ Tuned XGBoost is used in the Business Predictor.")
except Exception as e:
    st.info(f"Results unavailable: {e}")

st.markdown("---")

# ── Important disclaimer ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚠️ Important Notice</div>',
            unsafe_allow_html=True)
st.warning("""
**On the near-perfect scores (ROC-AUC 1.0000):**
The extremely high model performance reflects that the three distress signals
(credit constraint, employment shrinkage, low capacity) are directly
computable from the same features the model uses as inputs. This means the
model is essentially re-detecting patterns it was trained to find —
which is expected and valid for a distress *assessment* tool.

The practical value of this tool is not prediction in the forecasting sense —
it is **structured, evidence-based risk scoring** of a business profile
against patterns from 14,688 real African businesses. Use predictions as one
input among many, not as a definitive verdict.
""")
st.caption("Data: World Bank Enterprise Surveys 2007–2025 · JKUAT Final Year Project 2026")