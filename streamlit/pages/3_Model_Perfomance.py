"""Page 3 — Model Performance (v2 — leakage-corrected)"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results, CLASS_NAMES, CLASS_COLOURS

st.title("📈 Model Performance")
st.markdown("""
Full evaluation of all four models on the **held-out 20% test set** (2,938 firms).
These firms were never seen during training or SMOTE — they represent genuinely unseen data.
Model v2 corrects the target leakage present in the original version.
""")
st.markdown("---")

st.markdown('<div class="section-header">🏆 Final Results — 3-Class Classification (v2)</div>',
            unsafe_allow_html=True)

results_data = {
    "Model": [
        "Logistic Regression (baseline)",
        "Random Forest",
        "XGBoost (Initial) ★",
        "XGBoost (Tuned)",
    ],
    "ROC-AUC":      [0.7950, 0.8607, 0.8636, 0.8622],
    "F1 (Macro)":   [0.5708, 0.6251, 0.6402, 0.6367],
    "F1 Stable":    [0.7923, 0.8312, 0.8401, 0.8389],
    "F1 Moderate":  [0.6112, 0.6884, 0.7011, 0.6978],
    "F1 High Risk": [0.3089, 0.3558, 0.3794, 0.3734],
}
results_df = pd.DataFrame(results_data)

def highlight_best(row):
    if "★" in str(row["Model"]):
        return ["background-color:#E3F2FD;font-weight:bold"] * len(row)
    return [""] * len(row)

st.dataframe(
    results_df.style
              .apply(highlight_best, axis=1)
              .format({c: "{:.4f}" for c in results_df.columns if c != "Model"}),
    use_container_width=True, hide_index=True,
)
st.caption("★ XGBoost (Initial) achieves the best ROC-AUC (0.8636) and is deployed in the Business Predictor.")
st.markdown("---")

st.markdown('<div class="section-header">🔬 Why v2 Scores Look Different</div>',
            unsafe_allow_html=True)

col_old, col_new = st.columns(2)
with col_old:
    st.markdown("""
**v1 (original — leakage present)**
| Metric | Score |
|--------|-------|
| ROC-AUC | 1.0000 |
| F1-Macro | 0.9985 |
| F1-High Risk | 0.9962 |

The three distress signals (`credit_constrained`,
`employment_growth_rate`, `capacity_util_pct`) were
used to both **build the target** and **train the model**.
The model was simply reversing the labelling formula —
not learning genuine predictive relationships.
""")
with col_new:
    st.markdown("""
**v2 (corrected — leakage removed)**
| Metric | Score |
|--------|-------|
| ROC-AUC | 0.8636 |
| F1-Macro | 0.6402 |
| F1-High Risk | 0.3794 |

Signal features are excluded from model inputs. The model
learns from labour productivity, finance stress, infrastructure
burden, obstacle count, firm maturity, export orientation,
management quality, and sector / geography.

**ROC-AUC 0.8636 represents genuine cross-country prediction
across 14,688 African firms.** That is a strong, publishable result.
""")

st.markdown("---")
st.markdown('<div class="section-header">📊 Visual Comparison</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["LR", "RF", "XGB\n(Initial)", "XGB\n(Tuned)"]
    aucs   = [0.7950, 0.8607, 0.8636, 0.8622]
    colors = ["#94A3B8", "#2196F3", "#0F7B8C", "#64748B"]
    bars   = ax.bar(labels, aucs, color=colors, edgecolor="none", width=0.5)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.5,
               label="Random baseline (0.50)")
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("ROC-AUC (Macro OvR)")
    ax.set_title("ROC-AUC by Model (v2)")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
    bars[2].set_edgecolor("#0D1B3E")
    bars[2].set_linewidth(2.5)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    f1_vals  = [0.8401, 0.7011, 0.3794]
    colors_c = [CLASS_COLOURS[c] for c in CLASS_NAMES]
    bars = ax.bar(CLASS_NAMES, f1_vals, color=colors_c, edgecolor="none", width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 — XGBoost Initial (v2)")
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")
st.markdown('<div class="section-header">🔴 On High Risk F1 (0.3794)</div>',
            unsafe_allow_html=True)
st.info("""
The High Risk class F1 of 0.38 reflects two honest realities:

1. **High Risk is rare (9.1% of firms).** Even with SMOTE oversampling, the model
   sees far fewer genuine High Risk examples than Stable or Moderate Risk cases.

2. **High Risk requires all three signals simultaneously.** With signal features removed,
   the model must infer from indirect indicators — sales productivity, finance access,
   obstacle burden. This is harder, and honest models reflect that difficulty.

For practical use: the **ROC-AUC of 0.8636** means the model strongly ranks high-risk
firms above low-risk firms. The F1 score captures boundary precision — the AUC captures
overall discriminative power. Both are relevant; neither alone tells the full story.
""")

st.markdown("---")
st.markdown('<div class="section-header">⚙️ Training Setup</div>',
            unsafe_allow_html=True)

setup_df = pd.DataFrame([
    ("Total Firms",              "14,688"),
    ("Train Set",                "11,750 (80%)"),
    ("Test Set",                 "2,938 (20%)"),
    ("Train / Test Split",       "Stratified by distress_level"),
    ("Class Imbalance Handling", "SMOTE — training partition only"),
    ("Feature Scaling",          "StandardScaler — fit on train only"),
    ("Features",                 "32 (leakage-free v2 set)"),
    ("Target",                   "3-class: Stable / Moderate Risk / High Risk"),
    ("XGB Tuning",               "RandomizedSearchCV — 50 iterations × 5-fold CV"),
    ("CV Scoring",               "F1-macro"),
    ("Leakage Check",            "Automated — raises ValueError if signal features present"),
], columns=["Setting", "Value"])
st.dataframe(setup_df, use_container_width=True, hide_index=True)
st.caption("Source: World Bank Enterprise Surveys · 14,688 firms · 8 countries · JKUAT 2026")
