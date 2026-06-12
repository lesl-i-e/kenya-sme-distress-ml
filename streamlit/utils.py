"""
utils.py — shared model loading, constants, and helper functions (v2 — leakage-corrected).
Cached once per session so the dashboard stays fast.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")

# ── Class definitions ─────────────────────────────────────────────────────────
CLASS_NAMES   = ["Stable", "Moderate Risk", "High Risk"]
CLASS_COLOURS = {
    "Stable":        "#2196F3",
    "Moderate Risk": "#FF9800",
    "High Risk":     "#F44336",
}
CLASS_ICONS = {
    "Stable":        "✅",
    "Moderate Risk": "⚠️",
    "High Risk":     "🚨",
}

# ── Feature list (v2 — must match Phase 1 v2 ALL_FEATURES_FINAL exactly) ─────
# Removed leakage features: credit_constrained, employment_growth_rate,
#   emp_growth_available, capacity_util_pct, low_capacity
# Added 9 new independent features engineered in Phase 1 v2
MODEL_FEATURES = [
    # Firm fundamentals
    "firm_size_num",
    "employees_now",
    "log_sales",
    "survey_year",
    # Finance (independent inputs — NOT the signal itself)
    "has_loan_bin",
    "needs_finance_bin",
    "pct_internal_finance",
    "pct_bank_finance",
    "pct_supplier_finance",
    # v2 engineered features
    "sales_per_employee",
    "finance_double_stress",
    "infra_burden_score",
    "obstacle_count",
    "log_firm_age",
    "size_age_interaction",
    "is_exporter",
    "mgmt_quality_score",
    # Manager characteristics
    "manager_female",
    "manager_university",
    # Obstacle scores (7)
    "obstacle_electricity_score",
    "obstacle_finance_score",
    "obstacle_informality_score",
    "obstacle_workforce_score",
    "obstacle_corruption_score",
    "obstacle_transport_score",
    "obstacle_customs_score",
    # Encoded categoricals (6)
    "broad_sector_encoded",
    "detailed_sector_encoded",
    "region_encoded",
    "legal_status_clean_encoded",
    "country_encoded",
    "biggest_obstacle_encoded",
]

# ── Survey input options ──────────────────────────────────────────────────────
COUNTRIES = [
    "Kenya", "Uganda", "Tanzania", "Ghana",
    "Ethiopia", "Nigeria", "Rwanda", "South Africa",
]
COUNTRY_REGIONS = {
    "Kenya": [
        "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
        "Rift Valley", "Central", "Coast", "Nyanza and Western",
        "North Eastern", "Eastern", "Other Kenya",
    ],
    "Uganda": [
        "Kampala", "Entebbe", "Jinja", "Gulu", "Mbarara",
        "Central Uganda", "Eastern Uganda", "Northern Uganda",
        "Western Uganda", "Other Uganda",
    ],
    "Tanzania": [
        "Dar es Salaam", "Arusha", "Mwanza", "Dodoma", "Zanzibar",
        "Northern Tanzania", "Southern Tanzania", "Other Tanzania",
    ],
    "Ghana": [
        "Accra", "Kumasi", "Tamale", "Cape Coast", "Tema",
        "Greater Accra", "Ashanti", "Northern Ghana", "Other Ghana",
    ],
    "Ethiopia": [
        "Addis Ababa", "Dire Dawa", "Bahir Dar", "Mekelle",
        "Hawassa", "Oromia", "Amhara", "Tigray", "Other Ethiopia",
    ],
    "Nigeria": [
        "Lagos", "Abuja", "Kano", "Ibadan", "Port Harcourt",
        "Enugu", "Kaduna", "South West", "South East",
        "South South", "North West", "North East", "Other Nigeria",
    ],
    "Rwanda": [
        "Kigali", "Musanze", "Huye", "Rubavu",
        "Eastern Province", "Western Province",
        "Northern Province", "Southern Province", "Other Rwanda",
    ],
    "South Africa": [
        "Johannesburg", "Cape Town", "Durban", "Pretoria",
        "Port Elizabeth", "Gauteng", "Western Cape",
        "KwaZulu-Natal", "Eastern Cape", "Other South Africa",
    ],
}

BROAD_SECTORS = ["Manufacturing", "Other services", "Retail services"]

DETAILED_SECTORS = {
    "Manufacturing": [
        "Food and Beverages", "Textiles and Garments", "Leather Products",
        "Wood and Furniture", "Paper and Printing", "Chemicals and Pharmaceuticals",
        "Rubber and Plastics", "Non-metallic Minerals", "Metal Products",
        "Electronics and Machinery", "Auto and Transport Equipment",
        "Other Manufacturing",
    ],
    "Other services": [
        "IT and Software", "Telecommunications", "Financial Services",
        "Education and Training", "Healthcare and Social Services",
        "Hotels and Restaurants", "Tourism and Travel", "Construction",
        "Real Estate", "Transport and Logistics", "Professional Services",
        "Media and Creative", "Cleaning and Facility Services",
        "Security Services", "Other Services",
    ],
    "Retail services": [
        "General Retail", "Grocery and Food Retail", "Clothing and Apparel",
        "Electronics Retail", "Hardware and Building Materials",
        "Pharmacy and Healthcare Retail", "Wholesale Trade",
        "Auto Parts and Vehicles", "Other Retail",
    ],
}

LEGAL_STATUSES = [
    "Sole Proprietorship",
    "Partnership",
    "Limited Partnership",
    "Private Limited Company",
    "Public Limited Company",
]

OBSTACLE_OPTIONS = [
    "Access to finance",
    "Tax rates",
    "Tax administration",
    "Electricity supply",
    "Transport infrastructure",
    "Access to land",
    "Business licensing and permits",
    "Corruption",
    "Political instability",
    "Crime, theft and disorder",
    "Practices of informal competitors",
    "Inadequately educated workforce",
    "Labour regulations",
    "Customs and trade regulations",
    "Courts and legal system",
    "None — no major obstacle",
]

OBSTACLE_SEVERITY_OPTIONS = [
    "No obstacle", "Minor obstacle",
    "Moderate obstacle", "Major obstacle", "Very severe obstacle",
]
OBSTACLE_SEVERITY_MAP = {
    "No obstacle": 0, "Minor obstacle": 1, "Moderate obstacle": 2,
    "Major obstacle": 3, "Very severe obstacle": 4,
}

MGMT_PRACTICES = [
    "Uses production targets to track performance",
    "Shares performance targets with employees",
    "Non-manager pay linked to performance",
]

# ── Cached model loaders ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    return {
        "LR":        joblib.load(os.path.join(MODELS_DIR, "model_logistic_regression_v2.pkl")),
        "RF":        joblib.load(os.path.join(MODELS_DIR, "model_random_forest_v2.pkl")),
        "XGB":       joblib.load(os.path.join(MODELS_DIR, "model_xgboost_v2.pkl")),
        "XGB_tuned": joblib.load(os.path.join(MODELS_DIR, "model_xgboost_tuned_v2.pkl")),
        "scaler":    joblib.load(os.path.join(MODELS_DIR, "scaler_v2.pkl")),
    }

@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    return pd.read_csv(os.path.join(DATA_DIR, "sme_model_ready_v2.csv"))

@st.cache_data(show_spinner="Loading results…")
def load_results():
    return pd.read_csv(os.path.join(DATA_DIR, "phase2_final_results_v2.csv"))

@st.cache_data(show_spinner="Loading SHAP data…")
def load_shap():
    return pd.read_csv(os.path.join(DATA_DIR, "shap_feature_importance_v2.csv"))

# ── Encoding helpers ──────────────────────────────────────────────────────────
BROAD_SECTOR_ENCODING = {
    "Manufacturing": 0, "Other services": 1, "Retail services": 2,
}
COUNTRY_ENCODING = {
    "Ethiopia": 0, "Ghana": 1, "Kenya": 2, "Nigeria": 3,
    "Rwanda": 4, "South Africa": 5, "Tanzania": 6, "Uganda": 7,
}
LEGAL_ENCODING = {
    "Limited Partnership": 0, "Other": 1, "Partnership": 2,
    "Private Limited": 3, "Public Company": 4, "Sole Proprietorship": 5,
}


def encode_input(inputs: dict) -> dict:
    """
    Map clean user inputs to encoded feature values for the v2 model.

    NOTE (v2): The three distress signals (credit_constrained,
    employment_growth_rate/emp_growth_available, capacity_util_pct/low_capacity)
    are NO LONGER model features. They are excluded here to match the
    leakage-corrected Phase 1 v2 feature set.
    The model instead uses: needs_finance_bin, has_loan_bin, sales_per_employee,
    finance_double_stress, infra_burden_score, obstacle_count, log_firm_age,
    size_age_interaction, is_exporter, mgmt_quality_score.
    """
    # ── Legal status normalisation ────────────────────────────────────────────
    legal_clean = inputs["legal_status"]
    if "Limited Partner" in legal_clean:                       legal_clean = "Limited Partnership"
    elif "Private Limited" in legal_clean:                     legal_clean = "Private Limited"
    elif "Public" in legal_clean:                              legal_clean = "Public Company"
    elif "Partnership" in legal_clean and "Limited" not in legal_clean: legal_clean = "Partnership"
    elif "Sole" in legal_clean:                                legal_clean = "Sole Proprietorship"
    else:                                                       legal_clean = "Other"

    # ── Obstacle severity helper ──────────────────────────────────────────────
    obs = inputs.get("obstacles", {})
    def sev(key): return OBSTACLE_SEVERITY_MAP.get(obs.get(key, "No obstacle"), 0)

    obs_scores = [
        sev("electricity"), sev("finance"), sev("informality"),
        sev("workforce"), sev("corruption"), sev("transport"), sev("customs"),
    ]
    infra_burden    = float(np.mean(obs_scores))
    obstacle_count  = float(sum(1 for s in obs_scores if s >= 3))

    # ── Finance features ──────────────────────────────────────────────────────
    has_loan_bin      = 1.0 if inputs["has_loan"]      else 0.0
    needs_finance_bin = 1.0 if inputs["needs_finance"]  else 0.0

    # finance_double_stress: needs finance AND rates finance as major obstacle
    finance_double_stress = 1.0 if (inputs["needs_finance"] and sev("finance") >= 3) else 0.0

    # ── Sales and productivity ────────────────────────────────────────────────
    sales      = float(inputs["annual_sales"])
    log_sales  = float(np.log1p(max(sales, 0)))

    emp_now    = float(inputs["employees_now"])
    sales_per_employee = (log_sales / emp_now) if emp_now > 0 else 0.0

    # ── Firm age (log) ────────────────────────────────────────────────────────
    # Phase 1 used survey_year proxy: log(survey_year - 2000)
    log_firm_age = float(np.log1p(2025 - 2000))   # constant = log(26) ≈ 3.296
    # If founding year is provided, use it instead
    year_founded = inputs.get("year_founded", None)
    if year_founded and int(year_founded) < 2025:
        firm_age     = max(0, 2025 - int(year_founded))
        log_firm_age = float(np.log1p(firm_age))

    # ── Size × age interaction ────────────────────────────────────────────────
    firm_size_num       = float(inputs["firm_size_num"])
    size_age_interaction = firm_size_num * log_firm_age

    # ── Export orientation ────────────────────────────────────────────────────
    export_pct  = float(inputs.get("export_pct", 0))
    is_exporter = 1.0 if export_pct > 5 else 0.0

    # ── Management quality ────────────────────────────────────────────────────
    mgmt_score = float(sum([
        1 if inputs.get("mgmt_targets",  False) else 0,
        1 if inputs.get("mgmt_shares",   False) else 0,
        1 if inputs.get("mgmt_pay_perf", False) else 0,
    ]))

    # ── Encoded categoricals ──────────────────────────────────────────────────
    broad        = inputs["broad_sector"]
    detailed_enc = hash(inputs.get("detailed_sector", "")) % 69
    region_enc   = hash(inputs.get("region", ""))           % 81
    obstacle_enc = hash(inputs.get("biggest_obstacle", "")) % 61

    return {
        # Firm fundamentals
        "firm_size_num":              firm_size_num,
        "employees_now":              emp_now,
        "log_sales":                  log_sales,
        "survey_year":                2025.0,
        # Finance
        "has_loan_bin":               has_loan_bin,
        "needs_finance_bin":          needs_finance_bin,
        "pct_internal_finance":       float(inputs.get("pct_internal", 70)),
        "pct_bank_finance":           float(inputs.get("pct_bank", 10)),
        "pct_supplier_finance":       float(inputs.get("pct_supplier", 10)),
        # v2 engineered features
        "sales_per_employee":         sales_per_employee,
        "finance_double_stress":      finance_double_stress,
        "infra_burden_score":         infra_burden,
        "obstacle_count":             obstacle_count,
        "log_firm_age":               log_firm_age,
        "size_age_interaction":       size_age_interaction,
        "is_exporter":                is_exporter,
        "mgmt_quality_score":         mgmt_score,
        # Manager
        "manager_female":             1.0 if inputs.get("manager_female",    False) else 0.0,
        "manager_university":         1.0 if inputs.get("manager_university", False) else 0.0,
        # Obstacle scores
        "obstacle_electricity_score": float(sev("electricity")),
        "obstacle_finance_score":     float(sev("finance")),
        "obstacle_informality_score": float(sev("informality")),
        "obstacle_workforce_score":   float(sev("workforce")),
        "obstacle_corruption_score":  float(sev("corruption")),
        "obstacle_transport_score":   float(sev("transport")),
        "obstacle_customs_score":     float(sev("customs")),
        # Encoded categoricals
        "broad_sector_encoded":          float(BROAD_SECTOR_ENCODING.get(broad, 0)),
        "detailed_sector_encoded":       float(detailed_enc),
        "region_encoded":                float(region_enc),
        "legal_status_clean_encoded":    float(LEGAL_ENCODING.get(legal_clean, 1)),
        "country_encoded":               float(COUNTRY_ENCODING.get(inputs["country"], 2)),
        "biggest_obstacle_encoded":      float(obstacle_enc),
    }


def predict(models, encoded_features: dict) -> dict:
    """Run the best model (XGBoost Initial v2) and return prediction + probabilities."""
    X     = pd.DataFrame([encoded_features])[MODEL_FEATURES]
    best  = models["XGB"]          # XGBoost Initial is best in v2 (ROC-AUC 0.8636)
    probs = best.predict_proba(X)[0]
    pred  = int(np.argmax(probs))
    return {
        "prediction": CLASS_NAMES[pred],
        "probs":      {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)},
        "pred_idx":   pred,
    }


def style_metric(label, value, colour="#0F7B8C"):
    return f"""
    <div style="background:#F4F6F8;border-left:4px solid {colour};
                border-radius:8px;padding:12px 16px;margin:4px 0;">
        <div style="font-size:0.78rem;color:#64748B;">{label}</div>
        <div style="font-size:1.5rem;font-weight:700;color:#0D1B3E;">{value}</div>
    </div>
    """
