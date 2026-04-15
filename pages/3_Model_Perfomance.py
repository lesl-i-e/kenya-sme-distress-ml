"""Page 3 — Model Performance"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_results, CLASS_NAMES, CLASS_COLOURS

st.title("📈 Model Performance")
st.markdown("""
Full evaluation of all three models on the **held-out 20% test set** (2,938 firms).
These firms were never seen during training or SMOTE — they represent truly unseen data.
""")
st.markdown("---")

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Final Results — 3-Class Classification</div>',
            unsafe_allow_html=True)

results_data = {
    "Model": [
        "Logistic Regression (baseline)",
        "Random Forest",
        "XGBoost (initial)",
        "XGBoost (tuned) ★",
    ],
    "ROC-AUC": [0.9507, 1.0000, 1.0000, 1.0000],
    "F1 (macro)": [0.7824, 0.9970, 0.9985, 0.9985],
    "F1 Stable": [0.9115, 1.0000, 1.0000, 1.0000],
    "F1 Moderate": [0.8005, 0.9985, 0.9993, 0.9993],
    "F1 High Risk": [0.6352, 0.9924, 0.9962, 0.9962],
}
results_df = pd.DataFrame(results_data)

def highlight_best(row):
    if "tuned" in str(row["Model"]).lower():
        return ["background-color:#E3F2FD;font-weight:bold"] * len(row)
    return [""] * len(row)

st.dataframe(
    results_df.style.apply(highlight_best, axis=1)
              .format({c: "{:.4f}" for c in results_df.columns if c != "Model"}),
    use_container_width=True, hide_index=True,
)
st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Visual Comparison</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 3.5))
    labels = ["LR", "RF", "XGB", "XGB\n(tuned)"]
    aucs   = [0.9507, 1.0000, 1.0000, 1.0000]
    colors = ["#94A3B8", "#2196F3", "#0F7B8C", "#0D1B3E"]
    bars   = ax.bar(labels, aucs, color=colors, edgecolor="none", width=0.5)
    ax.axhline(0.95, color="#F44336", linestyle="--", linewidth=1,
               label="Baseline LR (0.9507)")
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("ROC-AUC (macro OVR)")
    ax.set_title("ROC-AUC by Model")
    ax.legend(fontsize=8)
    for bar, v in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col2:
    # Per-class F1 for tuned XGBoost
    f1_vals = [1.0000, 0.9993, 0.9962]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors_c = [CLASS_COLOURS[c] for c in CLASS_NAMES]
    bars = ax.bar(CLASS_NAMES, f1_vals, color=colors_c, edgecolor="none", width=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 — Tuned XGBoost")
    for bar, v in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ── Distress signal analysis ──────────────────────────────────────────────────
st.markdown('<div class="section-header">🚦 Signal Presence by Predicted Class</div>',
            unsafe_allow_html=True)
st.markdown("""
This chart shows the most important finding from the analysis:
**each risk class has a genuinely distinct distress profile.**
High Risk firms are overwhelmingly credit-constrained AND employment-shrinking AND
low-capacity simultaneously — confirming the three-class model captures real differences.
""")

fig, ax = plt.subplots(figsize=(10, 4))
classes = CLASS_NAMES
x       = np.arange(len(classes))
w       = 0.25

# Data from Phase 2 Cell 33 output
s1 = [0.0,  73.1, 90.2]   # credit constrained
s2 = [0.0,  12.9, 63.5]   # employment shrinking
s3 = [0.0,  14.1, 50.8]   # low capacity

ax.bar(x - w,  s1, w, label="Credit Constrained",    color="#F44336", edgecolor="none")
ax.bar(x,      s2, w, label="Employment Shrinking",   color="#FF9800", edgecolor="none")
ax.bar(x + w,  s3, w, label="Low Capacity (<60%)",    color="#0F7B8C", edgecolor="none")
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylabel("% of Firms Showing Signal")
ax.set_title("Active Distress Signals by Predicted Class")
ax.legend(fontsize=9)
ax.set_ylim(0, 100)
for i, (a, b, c) in enumerate(zip(s1, s2, s3)):
    if a > 0: ax.text(i-w,  a+1, f"{a:.0f}%", ha="center", fontsize=8)
    if b > 0: ax.text(i,    b+1, f"{b:.0f}%", ha="center", fontsize=8)
    if c > 0: ax.text(i+w,  c+1, f"{c:.0f}%", ha="center", fontsize=8)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Interpretation ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">💡 How to Read These Results</div>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("""
    **Why are ensemble scores so high?**

    The XGBoost and Random Forest scores of ROC-AUC 1.0000 reflect that
    the distress signals (credit constraint, employment shrinkage, low capacity)
    are directly related to the input features. The model is not predicting
    future closure — it is detecting whether a business currently shows
    the structural patterns associated with distress.

    This is intentional and valid for a **risk assessment tool**.
    The value lies in translating 31 features into a structured,
    interpretable risk class — not in forecasting unpredictable future events.

    Logistic Regression's lower score (0.9507) is more representative of
    what a genuinely predictive model would achieve on truly unseen scenarios.
    """)

with col_b:
    st.markdown("""
    **What the three classes mean in practice:**

    🔵 **Stable** — No active distress signals. 0% of stable firms show
    credit constraint, employment shrinkage, or low capacity simultaneously.

    🟠 **Moderate Risk** — One signal active. Typically 73% are credit
    constrained. This is the largest class (47% of all firms) and represents
    the typical East African SME — functioning but financially vulnerable.

    🔴 **High Risk** — All three signals converge. 90% are credit
    constrained, 64% are losing employees, and 51% are operating below
    60% capacity. This combination is the strongest available indicator
    of a business heading toward exit.

    **For investors:** Moderate Risk is not a no — it is a
    *conditional yes, with structured support*.
    """)

st.caption("Source: World Bank Enterprise Surveys · 14,688 firms · 8 countries · JKUAT 2026")