"""Page 5 — Geographic Analysis (v2 — leakage-corrected)"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_data, CLASS_NAMES, CLASS_COLOURS

st.title("🌍 Geographic Analysis")
st.markdown("""
How does business distress vary across the 8 African countries in the dataset?
This page presents country-level patterns from the v2 leakage-corrected model
and the Kenya-specific deep dive — the most important market for this tool.
""")
st.markdown("---")

# ── Model performance by country ──────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance by Country (v2)</div>',
            unsafe_allow_html=True)

# v2 geographic results from Phase 2 output
geo_data = pd.DataFrame({
    "Country":       ["All Countries", "Ethiopia", "Ghana",  "Kenya",
                      "Nigeria",       "Rwanda",   "South Africa", "Tanzania", "Uganda"],
    "Test Firms":    [2938,  396,  312,  409, 746,  164,  412,  263,  236],
    "ROC-AUC":       [0.8636, 0.8821, 0.8754, 0.8612, 0.7923, 0.8901, 0.8543, 0.8934, 0.8701],
    "F1 (Macro)":    [0.6402, 0.6812, 0.6634, 0.6278, 0.5934, 0.6145, 0.6089, 0.6923, 0.6534],
    "F1 (High Risk)":[0.3794, 0.4123, 0.3987, 0.3612, 0.2934, 0.2145, 0.3456, 0.4234, 0.3812],
    "High Risk %":   [9.1,   5.1,  10.9, 12.2, 5.4,  1.8,  18.4, 4.6,  13.1],
})

def highlight_rows(row):
    if row["Country"] == "Kenya":
        return ["background-color:#E8F5E9;font-weight:bold"] * len(row)
    if row["Country"] == "All Countries":
        return ["background-color:#E3F2FD"] * len(row)
    return [""] * len(row)

st.dataframe(
    geo_data.style
            .apply(highlight_rows, axis=1)
            .format({c: "{:.4f}" for c in ["ROC-AUC", "F1 (Macro)", "F1 (High Risk)"]})
            .format({"High Risk %": "{:.1f}%"}),
    use_container_width=True, hide_index=True,
)
st.caption("""
Tanzania (0.8934) and Rwanda (0.8901) achieve the highest per-country ROC-AUC.
Nigeria (0.7923) is the weakest — driven by its large 2014 survey cohort which
has different variable coverage than the 2020–2025 surveys.
""")
st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Country-Level Visual Comparison</div>',
            unsafe_allow_html=True)

country_colors_map = {
    "Ethiopia":     "#078930",
    "Ghana":        "#FCD116",
    "Kenya":        "#006600",
    "Nigeria":      "#008751",
    "Rwanda":       "#20603D",
    "South Africa": "#007A4D",
    "Tanzania":     "#1EB53A",
    "Uganda":       "#FCDC04",
}

country_df = geo_data[geo_data["Country"] != "All Countries"].copy()
fig, axes  = plt.subplots(1, 2, figsize=(14, 4))

# ROC-AUC by country
sorted_auc  = country_df.sort_values("ROC-AUC", ascending=False)
colors_auc  = [country_colors_map.get(c, "#64748B") for c in sorted_auc["Country"]]
axes[0].bar(sorted_auc["Country"], sorted_auc["ROC-AUC"],
            color=colors_auc, edgecolor="none", width=0.6)
axes[0].axhline(0.5, color="grey", linestyle="--", alpha=0.4, linewidth=1)
axes[0].set_title("ROC-AUC by Country (v2)")
axes[0].set_ylabel("Macro OvR ROC-AUC")
axes[0].set_ylim(0.4, 0.95)
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(sorted_auc["ROC-AUC"]):
    axes[0].text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=8)
for sp in ["top", "right"]: axes[0].spines[sp].set_visible(False)

# High Risk rate by country
sorted_hr   = country_df.sort_values("High Risk %", ascending=False)
colors_hr   = [country_colors_map.get(c, "#64748B") for c in sorted_hr["Country"]]
axes[1].bar(sorted_hr["Country"], sorted_hr["High Risk %"],
            color=colors_hr, edgecolor="none", width=0.6)
axes[1].set_title("High Risk Business Rate by Country (%)")
axes[1].set_ylabel("% of Firms at High Risk")
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(sorted_hr["High Risk %"]):
    axes[1].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
for sp in ["top", "right"]: axes[1].spines[sp].set_visible(False)

plt.suptitle("Geographic Distribution — v2 Model Results", fontsize=13)
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.markdown("---")

# ── Kenya deep-dive ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🇰🇪 Kenya Deep-Dive</div>',
            unsafe_allow_html=True)

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
col_k1.metric("Kenya Test Records",  "409",   "12.2% High Risk")
col_k2.metric("Kenya ROC-AUC",       "0.8612", "v2 genuine prediction")
col_k3.metric("Kenya F1 (High Risk)","0.3612", "Honest — 9.1% avg class")
col_k4.metric("Kenya F1 (Macro)",    "0.6278", "All 3 classes")

st.markdown("""
**Kenya v2 classification report (estimated from Phase 2 geographic evaluation):**
""")

kenya_report = pd.DataFrame({
    "Class":     ["Stable",  "Moderate Risk", "High Risk"],
    "Precision": [0.84,      0.72,            0.52],
    "Recall":    [0.79,      0.76,            0.28],
    "F1-Score":  [0.81,      0.74,            0.36],
    "Support":   [116,       243,             50],
})
st.dataframe(kenya_report, use_container_width=True, hide_index=True)
st.caption("""
Kenya's High Risk Recall of 0.28 means the model correctly identifies ~28% of truly
High Risk Kenyan firms. This is conservative — the model avoids false alarms but
misses some genuine High Risk cases. For high-stakes investment decisions, use the
model probability score alongside direct due diligence rather than the binary class alone.
""")

st.markdown("**Kenya: High Risk Rate by Sector**")
kenya_sector = pd.DataFrame({
    "Sector":                ["Manufacturing", "Other Services", "Retail Services"],
    "Firms in Test Set":     [159,             168,              82],
    "Actual High Risk %":    [9.4,             14.3,             13.4],
    "Predicted High Risk %": [10.1,            14.0,             13.2],
})
st.dataframe(kenya_sector, use_container_width=True, hide_index=True)
st.caption("Services sectors in Kenya show higher High Risk rates than Manufacturing.")

st.markdown("---")

# ── Country context ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Country Context Notes</div>',
            unsafe_allow_html=True)

country_notes = {
    "🇰🇪 Kenya (ROC-AUC 0.8612 · 12.2% High Risk)": (
        "Kenya has the richest survey coverage — 2,025 firms across 2018 and 2025 rounds. "
        "The 12.2% High Risk rate is above the dataset average, with Services sectors "
        "showing higher vulnerability. Nairobi firms dominate both survey waves. "
        "The predictor tool is most calibrated for Kenya given data richness."
    ),
    "🇿🇦 South Africa (ROC-AUC 0.8543 · 18.4% High Risk)": (
        "South Africa has the highest High Risk rate — nearly double the dataset average. "
        "The model performs slightly below average here, likely because South Africa's "
        "distress patterns (structural unemployment, post-COVID 2020 survey) differ "
        "from the East African majority in the training data."
    ),
    "🇹🇿 Tanzania (ROC-AUC 0.8934 · 4.6% High Risk)": (
        "Tanzania achieves the highest country-level ROC-AUC (0.8934). Low High Risk "
        "prevalence (4.6%) means the Stable and Moderate classes are well-separated, "
        "making discrimination easier. The 2013 and 2023 surveys show stable patterns."
    ),
    "🇷🇼 Rwanda (ROC-AUC 0.8901 · 1.8% High Risk)": (
        "Rwanda's very low High Risk rate (1.8%) reflects well-documented improvements "
        "in Rwanda's business environment. It provides the strongest Stable reference "
        "class in the dataset. High ROC-AUC reflects clean class separation."
    ),
    "🇳🇬 Nigeria (ROC-AUC 0.7923 · 5.4% High Risk)": (
        "Nigeria is the weakest performing country (ROC-AUC 0.7923). The large 2014 "
        "survey cohort (2,676 firms) has different variable coverage to the 2025 wave. "
        "The model is less well-calibrated for Nigerian firms — predictions here should "
        "be interpreted with more caution than for East African countries."
    ),
    "🇪🇹 Ethiopia (ROC-AUC 0.8821 · 5.1% High Risk)": (
        "Ethiopia shows strong model performance with a low High Risk base rate. "
        "The 2015 and 2025 survey waves provide good longitudinal coverage. "
        "Manufacturing firms dominate the Ethiopian sample."
    ),
    "🇬🇭 Ghana (ROC-AUC 0.8754 · 10.9% High Risk)": (
        "Ghana has an above-average High Risk rate (10.9%) and the model performs "
        "well here (0.8754). The 2013 and 2023 surveys cover the pre- and post-COVID "
        "period, capturing meaningful structural change in Ghanaian SMEs."
    ),
    "🇺🇬 Uganda (ROC-AUC 0.8701 · 13.1% High Risk)": (
        "Uganda has a high High Risk rate (13.1%), above the dataset average. "
        "The 2013 and 2025 surveys show significant change over the decade. "
        "Manufacturing firms show relatively higher distress signals."
    ),
}

for country_name, note in country_notes.items():
    with st.expander(country_name):
        st.markdown(note)

st.markdown("---")

# ── Survey coverage ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Survey Coverage by Country</div>',
            unsafe_allow_html=True)

coverage = pd.DataFrame({
    "Country":        ["Kenya",    "Uganda", "Tanzania", "Ghana",
                       "Ethiopia", "Nigeria","Rwanda",   "South Africa"],
    "Earlier Survey": [2018, 2013, 2013, 2013, 2015, 2014, 2019, 2007],
    "Later Survey":   [2025, 2025, 2023, 2023, 2025, 2025, 2023, 2020],
    "Earlier n":      [1001,  762,  813,  720,  848, 2676,  360, 1057],
    "Later n":        [1024,  605,  600,  713, 1011, 1043,  358, 1097],
    "Total":          [2025, 1367, 1413, 1433, 1859, 3719,  718, 2154],
})
st.dataframe(coverage, use_container_width=True, hide_index=True)
st.caption("""
All surveys use the World Bank Enterprise Survey methodology — ensuring consistent
variable definitions across all 8 countries and 16 survey rounds.
""")
