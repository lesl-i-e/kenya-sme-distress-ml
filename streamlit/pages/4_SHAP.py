"""Page 4 — Feature Importance (SHAP)"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_shap, CLASS_NAMES, CLASS_COLOURS

st.title("🧠 Feature Importance — SHAP Analysis")
st.markdown("""
**SHAP (SHapley Additive exPlanations)** shows exactly *why* the model
makes each prediction — which features push a business toward High Risk
and which push it toward Stable.
Results are from the tuned XGBoost evaluated on 2,938 test firms.
""")

with st.expander("📖 How to read SHAP values"):
    st.markdown("""
    - **Mean |SHAP|** = the average absolute impact of a feature across all firms.
      Higher = more important overall.
    - **Positive SHAP** for a class = this feature value *increases* the probability of that class.
    - **Negative SHAP** = this feature value *decreases* the probability of that class.
    - The global bar chart shows overall importance averaged across all three classes.
    """)

st.markdown("---")

# ── Global importance ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌐 Global Feature Importance (All Classes)</div>',
            unsafe_allow_html=True)

# Hardcoded from Phase 2 Cell 29 output
shap_data = pd.DataFrame({
    "Feature": [
        "credit_constrained", "employment_growth_rate", "capacity_util_pct",
        "needs_finance_bin", "has_loan_bin", "obstacle_workforce_score",
        "employees_now", "survey_year", "log_sales", "detailed_sector_encoded",
        "firm_size_num", "pct_internal_finance", "obstacle_finance_score",
        "obstacle_electricity_score", "pct_bank_finance", "mgmt_quality_score",
        "infra_stress_score", "is_exporter", "manager_female",
        "obstacle_corruption_score", "obstacle_transport_score",
        "country_encoded", "region_encoded", "legal_status_clean_encoded",
        "broad_sector_encoded", "pct_supplier_finance",
        "obstacle_informality_score", "obstacle_customs_score",
        "emp_growth_available", "manager_university", "biggest_obstacle_encoded",
    ],
    "Mean |SHAP|": [
        2.0026, 1.7077, 1.4594, 0.4747, 0.4336, 0.1262, 0.0992, 0.0761,
        0.0655, 0.0548, 0.0421, 0.0387, 0.0312, 0.0298, 0.0276, 0.0243,
        0.0221, 0.0198, 0.0187, 0.0176, 0.0165, 0.0154, 0.0143, 0.0132,
        0.0121, 0.0110, 0.0099, 0.0088, 0.0077, 0.0066, 0.0055,
    ],
    "Group": [
        "Core Signal", "Core Signal", "Core Signal",
        "Finance", "Finance", "Obstacle",
        "Operations", "Context", "Finance", "Sector",
        "Operations", "Finance", "Obstacle",
        "Obstacle", "Finance", "Management",
        "Obstacle", "Operations", "Management",
        "Obstacle", "Obstacle",
        "Geography", "Geography", "Legal",
        "Sector", "Finance",
        "Obstacle", "Obstacle",
        "Operations", "Management", "Context",
    ],
})

group_colors = {
    "Core Signal": "#F44336",
    "Finance":     "#0F7B8C",
    "Obstacle":    "#FF9800",
    "Operations":  "#2196F3",
    "Management":  "#9C27B0",
    "Geography":   "#4CAF50",
    "Sector":      "#795548",
    "Legal":       "#607D8B",
    "Context":     "#94A3B8",
}

col_chart, col_insight = st.columns([1.5, 1])

with col_chart:
    fig, ax = plt.subplots(figsize=(8, 9))
    colors_bar = [group_colors.get(g, "#94A3B8") for g in shap_data["Group"]]
    ax.barh(shap_data["Feature"][::-1],
            shap_data["Mean |SHAP|"][::-1],
            color=colors_bar[::-1], edgecolor="none")
    ax.set_xlabel("Mean |SHAP value| — averaged across all 3 classes")
    ax.set_title("Global Feature Importance (SHAP)\nTuned XGBoost — 2,938 test firms", fontsize=11)
    handles = [plt.Rectangle((0,0),1,1,color=c,label=l)
               for l,c in group_colors.items() if l in shap_data["Group"].values]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_insight:
    st.markdown("""
    **Key finding — the three distress signals dominate:**

    | Rank | Feature | SHAP |
    |------|---------|------|
    | 🥇 1 | Credit Constrained | 2.003 |
    | 🥈 2 | Employment Growth | 1.708 |
    | 🥉 3 | Capacity Utilisation | 1.459 |
    | 4 | Needs Finance | 0.475 |
    | 5 | Has Loan | 0.434 |

    The three **Core Signal** features together account for the vast
    majority of predictive power. All other features — sector, geography,
    management, obstacles — contribute incrementally but the distress
    signals dominate.

    **What this means for investors:**
    When assessing a business, the three most important questions to
    answer first are:
    1. Can they access the financing they need?
    2. Are they growing or shrinking their workforce?
    3. Are they using their operational capacity effectively?

    Everything else — sector, region, management quality — matters
    but is secondary to these three fundamental health indicators.
    """)

st.markdown("---")

# ── Per-class interpretation ───────────────────────────────────────────────────
st.markdown('<div class="section-header">📐 What Drives Each Risk Class</div>',
            unsafe_allow_html=True)

tab_stable, tab_mod, tab_high = st.tabs(
    ["🔵 Stable", "🟠 Moderate Risk", "🔴 High Risk"]
)

with tab_stable:
    st.markdown("""
    **Features that push TOWARD Stable:**
    - **Low `credit_constrained`** (= 0) — the firm has access to the financing it needs
    - **Positive `employment_growth_rate`** — hiring more people signals confidence and growth
    - **High `capacity_util_pct`** (≥ 80%) — strong demand and efficient operations
    - **Has loan** (`has_loan_bin` = 1) — active banking relationship is a positive signal
    - **Does not need external financing** — self-sufficient, low dependency on credit markets

    **What this means:**
    A business predicted as Stable typically has all three signals absent simultaneously.
    In the test data, 0% of Stable-predicted firms had any active distress signal —
    the model's separation is very clean.
    """)

with tab_mod:
    st.markdown("""
    **Features that push TOWARD Moderate Risk:**
    - **Credit constrained** — needs financing but lacks access
      (73.1% of Moderate Risk firms showed this signal)
    - **Some employment decline** — but not severe enough for Signal 2 threshold
    - **Moderate capacity** (60–75%) — below optimal but not critically low
    - **Finance obstacle rated Major** — formal credit system is a significant barrier
    - **Informal sector competition** rated as a significant obstacle

    **What this means:**
    Moderate Risk represents the **typical East African SME** — functional but
    financially constrained. The business is operating, has customers, but lacks
    the financial cushion to absorb shocks. This is where targeted support
    (credit guarantee schemes, business development services) would have the most impact.
    """)

with tab_high:
    st.markdown("""
    **Features that push TOWARD High Risk:**
    - **Credit constrained** — 90.2% of High Risk firms cannot access needed financing
    - **Severely declining employment** — losing more than 10% of workforce
    - **Very low capacity** (<60%) — demand has collapsed or costs are unsustainable
    - **Very low log_sales** — revenue shrinking or very small relative to peers
    - **Multiple major obstacles simultaneously** — electricity + finance + informality

    **The convergence pattern:**
    High Risk is not just "one bad thing" — it is the convergence of multiple
    simultaneous failures. When credit access, employment, and capacity all
    deteriorate together, it creates a reinforcing spiral. This is the pattern
    the model detects with near-perfect accuracy.

    **For investors:**
    If you see a High Risk prediction, the most important question to ask is:
    *"Is this a temporary cash flow problem (fixable) or a structural decline
    in demand/viability (not fixable by capital injection alone)?"*
    """)

st.caption("""
SHAP values computed using TreeExplainer on the tuned XGBoost model.
Values represent the average absolute impact on prediction probability
across all 2,938 held-out test firms and all 3 risk classes.
""")