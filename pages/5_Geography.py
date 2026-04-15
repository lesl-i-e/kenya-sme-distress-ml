"""Page 5 — Geographic Analysis"""
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
This page presents country-level patterns and the Kenya-specific deep dive —
the most important market for this investor tool.
""")
st.markdown("---")

# ── Model performance by country ──────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance by Country</div>',
            unsafe_allow_html=True)

# From Phase 2 Cell 36 output
geo_data = pd.DataFrame({
    "Country": ["All Countries","Ethiopia","Ghana","Kenya",
                "Nigeria","Rwanda","South Africa","Tanzania","Uganda"],
    "ROC-AUC":       [1.0000, 1.0000, 1.0000, 1.0000,
                      1.0000, 1.0000, 0.9999, 1.0000, 1.0000],
    "F1 (macro)":    [0.9985, 1.0000, 1.0000, 0.9960,
                      1.0000, 1.0000, 0.9972, 1.0000, 1.0000],
    "F1 (High Risk)":[0.9962, 1.0000, 1.0000, 0.9901,
                      1.0000, 1.0000, 0.9934, 1.0000, 1.0000],
    "Test Firms":    [2938, 396, 312, 409, 746, 164, 412, 263, 236],
    "High Risk %":   [9.1, 5.1, 10.9, 12.2, 5.4, 1.8, 18.4, 4.6, 13.1],
})

def highlight_kenya(row):
    if row["Country"] == "Kenya":
        return ["background-color:#E8F5E9;font-weight:bold"] * len(row)
    if row["Country"] == "All Countries":
        return ["background-color:#E3F2FD"] * len(row)
    return [""] * len(row)

st.dataframe(
    geo_data.style.apply(highlight_kenya, axis=1)
                  .format({c: "{:.4f}" for c in ["ROC-AUC","F1 (macro)","F1 (High Risk)"]})
                  .format({"High Risk %": "{:.1f}%"}),
    use_container_width=True, hide_index=True,
)
st.markdown("---")

# ── High Risk rate by country chart ──────────────────────────────────────────
st.markdown('<div class="section-header">🔴 High Risk Rate by Country</div>',
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
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# High risk rate
sorted_hr = country_df.sort_values("High Risk %", ascending=False)
colors_bar = [country_colors_map.get(c, "#64748B") for c in sorted_hr["Country"]]
axes[0].bar(sorted_hr["Country"], sorted_hr["High Risk %"],
            color=colors_bar, edgecolor="none", width=0.6)
axes[0].set_title("High Risk Business Rate by Country (%)")
axes[0].set_ylabel("% of Firms at High Risk")
axes[0].tick_params(axis="x", rotation=30)
for i, v in enumerate(sorted_hr["High Risk %"]):
    axes[0].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=8)
for sp in ["top","right"]: axes[0].spines[sp].set_visible(False)

# Test firms count
axes[1].bar(country_df["Country"], country_df["Test Firms"],
            color=[country_colors_map.get(c,"#64748B") for c in country_df["Country"]],
            edgecolor="none", width=0.6)
axes[1].set_title("Test Firms per Country")
axes[1].set_ylabel("Number of Firms in Test Set")
axes[1].tick_params(axis="x", rotation=30)
for i, v in enumerate(country_df["Test Firms"]):
    axes[1].text(i, v + 3, f"{v:,}", ha="center", fontsize=8)
for sp in ["top","right"]: axes[1].spines[sp].set_visible(False)

plt.suptitle("Figure: Geographic Distribution of Distress", fontsize=13)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("---")

# ── Kenya deep-dive ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🇰🇪 Kenya Deep-Dive</div>',
            unsafe_allow_html=True)

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
col_k1.metric("Kenya Test Records", "409", "12.2% High Risk")
col_k2.metric("Kenya ROC-AUC", "1.0000", "Equal to global")
col_k3.metric("Kenya F1 (High Risk)", "0.9901", "Near perfect detection")
col_k4.metric("Kenya F1 (macro)", "0.9960", "All 3 classes")

st.markdown("""
**Kenya classification report (from Phase 2):**
""")

kenya_report = pd.DataFrame({
    "Class":     ["Stable", "Moderate Risk", "High Risk"],
    "Precision": [1.00, 1.00, 0.98],
    "Recall":    [1.00, 1.00, 1.00],
    "F1-score":  [1.00, 1.00, 0.99],
    "Support":   [116, 243, 50],
})
st.dataframe(kenya_report, use_container_width=True, hide_index=True)

# Kenya sector breakdown
st.markdown("**Kenya: High Risk Rate by Sector**")
kenya_sector = pd.DataFrame({
    "Sector":               ["Manufacturing", "Other Services", "Retail Services"],
    "Firms in Test Set":    [159, 168, 82],
    "True High Risk %":     [9.4, 14.3, 13.4],
    "Predicted High Risk %":[10.1, 14.3, 13.4],
    "Mean HR Probability":  ["10.0%", "14.1%", "13.4%"],
})
st.dataframe(kenya_sector, use_container_width=True, hide_index=True)
st.caption("""
Services sectors in Kenya show higher High Risk rates than Manufacturing.
Other Services (14.3%) and Retail (13.4%) are above the Kenya average of 12.2%.
""")

st.markdown("---")

# ── Country context ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Country Context Notes</div>',
            unsafe_allow_html=True)

country_notes = {
    "🇰🇪 Kenya (12.2% High Risk)":
        "Kenya has the highest survey representation of Stable + Moderate Risk firms. "
        "The 12.2% High Risk rate is above average, with Services particularly vulnerable. "
        "Nairobi firms are better represented in earlier surveys (2018). "
        "The predictor tool is most applicable here given data richness.",

    "🇿🇦 South Africa (18.4% High Risk)":
        "South Africa has the highest High Risk rate in the dataset — nearly double the average. "
        "This likely reflects structural unemployment challenges, infrastructure constraints, "
        "and the post-COVID economic recovery period captured in the 2020 survey.",

    "🇺🇬 Uganda (13.1% High Risk)":
        "Uganda's 13.1% High Risk rate is above average. The 2013 and 2025 survey rounds "
        "show significant change in the business environment over the intervening decade. "
        "Manufacturing firms show relatively higher distress signals.",

    "🇳🇬 Nigeria (5.4% High Risk)":
        "Nigeria's large dataset (2,676 firms from 2014) brings the High Risk rate down. "
        "The 2025 survey shows different patterns. Nigeria has the most firm records and "
        "has significant weight in training the model.",

    "🇷🇼 Rwanda (1.8% High Risk)":
        "Rwanda's very low High Risk rate (1.8%) reflects Rwanda's well-known improvements "
        "in business environment regulation and access to finance. The Rwandan data provides "
        "a strong Stable class reference for the model.",
}

for country_name, note in country_notes.items():
    with st.expander(country_name):
        st.markdown(note)

st.markdown("---")

# ── Dataset survey coverage ───────────────────────────────────────────────────
st.markdown('<div class="section-header">📅 Survey Coverage by Country</div>',
            unsafe_allow_html=True)

coverage = pd.DataFrame({
    "Country":         ["Kenya","Uganda","Tanzania","Ghana",
                        "Ethiopia","Nigeria","Rwanda","South Africa"],
    "Earlier Survey":  [2018, 2013, 2013, 2013, 2015, 2014, 2019, 2007],
    "Later Survey":    [2025, 2025, 2023, 2023, 2025, 2025, 2023, 2020],
    "Earlier n":       [1001, 762, 813, 720, 848, 2676, 360, 1057],
    "Later n":         [1024, 605, 600, 713, 1011, 1043, 358, 1097],
    "Total":           [2025, 1367, 1413, 1433, 1859, 3719, 718, 2154],
})
st.dataframe(coverage, use_container_width=True, hide_index=True)
st.caption("""
All surveys conducted using the World Bank Enterprise Survey methodology —
ensuring consistent variable definitions across all countries and years.
""")