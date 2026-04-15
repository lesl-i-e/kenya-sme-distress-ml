# SME Business Distress Predictor — East Africa

**BIT 2303 / SDS 2406 — Final Year Project**
**Student:** Gedion Leslie Kweya Odera · SCT213-C002-0062/2022
**Supervisor:** Mr. Adhola Samuel · JKUAT

---

## 🚀 Live Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 📋 Project Overview

This project predicts business distress risk in East African SMEs using machine learning
applied to World Bank Enterprise Survey data from **14,688 firms across 8 countries**.

Businesses are classified into three risk levels:
- **Stable** — No active distress signals
- **Moderate Risk** — One distress signal present
- **High Risk** — Two or more distress signals active simultaneously

**Three distress signals:**
1. Credit Constraint — needs financing but cannot access it
2. Employment Shrinkage — workforce fell >10% over 3 years
3. Low Capacity Utilisation — operating below 60% capacity

---

## 📊 Results

| Model | ROC-AUC | F1 (macro) | F1 (High Risk) |
|-------|---------|------------|----------------|
| Logistic Regression | 0.9507 | 0.7824 | 0.6352 |
| Random Forest | 1.0000 | 0.9970 | 0.9924 |
| XGBoost (initial) | 1.0000 | 0.9985 | 0.9962 |
| **XGBoost (tuned) ★** | **1.0000** | **0.9985** | **0.9962** |

**Top SHAP predictors:** Credit Constrained · Employment Growth · Capacity Utilisation

---

## 🗂️ Repository Structure

```
kenya-sme-distress-ml/
│
├── app.py                      # Streamlit entry point
├── utils.py                    # Shared model loading and constants
├── requirements.txt
├── README.md
│
├── pages/
│   ├── 1_Overview.py           # Dataset stats and class distribution
│   ├── 2_Predictor.py          # Comprehensive investor predictor
│   ├── 3_Model_Performance.py  # Evaluation results
│   ├── 4_SHAP.py               # Feature importance
│   └── 5_Geography.py          # Country analysis
│
├── models/                     # Trained model pkl files
│   ├── model_logistic_regression.pkl
│   ├── model_xgboost.pkl
│   ├── model_xgboost_tuned.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
│
└── data/                       # Processed data files
    ├── sme_model_ready.csv
    ├── phase2_final_results.csv
    └── shap_feature_importance.csv
```

---

## ⚙️ Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/kenya-sme-distress-ml.git
cd kenya-sme-distress-ml
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select repo → set `app.py` as entry point → Deploy

---

## 📚 Key References

- World Bank Enterprise Surveys (2007–2025) — worldbank.org/en/programs/enterprise-analysis
- Audretsch, D. & Mahmood, T. (1995). New firm survival. *Review of Economics and Statistics*
- Lundberg, S. & Lee, S.I. (2017). A unified approach to interpreting model predictions. *NeurIPS*

---

## ⚠️ Disclaimer

This tool provides probabilistic risk assessment based on historical survey data.
It is a decision-support tool — not a definitive verdict. Always complement
this assessment with direct business due diligence.

---

*JKUAT · School of Computing and Information Technology · April 2026*