"""Page 2 — Business Predictor (v2 — leakage-corrected)"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    load_models, predict, encode_input,
    CLASS_NAMES, CLASS_COLOURS, CLASS_ICONS,
    COUNTRIES, COUNTRY_REGIONS, BROAD_SECTORS, DETAILED_SECTORS,
    LEGAL_STATUSES, OBSTACLE_OPTIONS, OBSTACLE_SEVERITY_OPTIONS,
    OBSTACLE_SEVERITY_MAP, MGMT_PRACTICES,
)

st.title("🔍 Business Distress Predictor")
st.markdown("""
Complete the business profile below as thoroughly as possible.
Every section you fill in improves the accuracy of the risk assessment.
The model compares this profile against **14,688 real East African businesses**
surveyed by the World Bank and returns a structured distress risk score.
""")

models = load_models()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — BASIC BUSINESS INFORMATION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🏢 Section A — Basic Business Information")

colA1, colA2, colA3 = st.columns(3)

with colA1:
    business_name = st.text_input(
        "Business Name (optional — for your reference)",
        placeholder="e.g. Kamau General Supplies Ltd"
    )
    country = st.selectbox("Country of Operation", COUNTRIES, index=0)

with colA2:
    region_options = COUNTRY_REGIONS.get(country, ["Other"])
    region = st.selectbox("Region / City", region_options)

with colA3:
    year_founded = st.number_input(
        "Year Business Was Founded / Proposed Start Year",
        min_value=1980, max_value=2026, value=2018, step=1,
    )
    legal_status = st.selectbox("Legal Structure", LEGAL_STATUSES)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — SECTOR AND OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🏭 Section B — Sector and Operations")

colB1, colB2, colB3 = st.columns(3)

with colB1:
    broad_sector = st.selectbox("Broad Sector", BROAD_SECTORS)
    detailed_sector = st.selectbox(
        "Detailed Sector / Industry",
        DETAILED_SECTORS.get(broad_sector, ["Other"])
    )

with colB2:
    firm_size = st.selectbox(
        "Firm Size Classification",
        ["Small (5–19 employees)", "Medium (20–99 employees)", "Large (100+ employees)"]
    )
    firm_size_map = {
        "Small (5–19 employees)":   1,
        "Medium (20–99 employees)": 2,
        "Large (100+ employees)":   3,
    }
    firm_size_num = firm_size_map[firm_size]

with colB3:
    is_exporting = st.checkbox(
        "The business exports goods or services (>5% of sales abroad)",
        value=False
    )
    export_pct = 0
    if is_exporting:
        export_pct = st.slider("Approximate export share (% of total sales)",
                               min_value=5, max_value=100, value=15)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — WORKFORCE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 👥 Section C — Workforce")

colC1, colC2, colC3 = st.columns(3)

with colC1:
    employees_now = st.number_input(
        "Current Number of Full-Time Employees",
        min_value=1, max_value=10000, value=15, step=1,
        help="Count only permanent, full-time paid employees."
    )

with colC2:
    employees_3yr = st.number_input(
        "Employees 3 Years Ago (or at founding if newer)",
        min_value=0, max_value=10000, value=12, step=1,
        help="If the business is less than 3 years old, enter the founding employee count."
    )

with colC3:
    if employees_3yr > 0:
        growth = (employees_now - employees_3yr) / employees_3yr * 100
        if growth > 10:
            trend   = f"📈 Growing ({growth:+.1f}%)"
            tcolour = "#2196F3"
        elif growth < -10:
            trend   = f"📉 Shrinking ({growth:+.1f}%)"
            tcolour = "#F44336"
        else:
            trend   = f"➡️ Stable ({growth:+.1f}%)"
            tcolour = "#FF9800"
        st.markdown(f"""
        <div style="background:#F4F6F8;border-radius:8px;padding:14px;
                    border-left:4px solid {tcolour};margin-top:28px;">
            <div style="font-size:0.8rem;color:#64748B;">Employment Trend</div>
            <div style="font-size:1.1rem;font-weight:700;color:{tcolour};">{trend}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Enter employees 3 years ago to see the employment trend.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — FINANCIAL PROFILE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 💰 Section D — Financial Profile")

colD1, colD2 = st.columns(2)

with colD1:
    st.markdown("**Annual Sales Revenue**")
    currency_map = {
        "Kenya": "KES", "Uganda": "UGX", "Tanzania": "TZS",
        "Ghana": "GHS", "Ethiopia": "ETB", "Nigeria": "NGN",
        "Rwanda": "RWF", "South Africa": "ZAR",
    }
    currency = currency_map.get(country, "Local Currency")

    annual_sales = st.number_input(
        f"Annual Sales Revenue ({currency})",
        min_value=0, max_value=10_000_000_000,
        value=5_000_000, step=100_000,
        help=f"Total revenue from sales in the last fiscal year, in {currency}."
    )

    st.markdown("**Financing Sources** (must total 100%)")
    pct_internal = st.slider("% Financed Internally (owner savings, retained earnings)", 0, 100, 70)
    pct_bank     = st.slider("% Financed by Banks / Microfinance",                       0, 100, 15)
    pct_supplier = st.slider("% Financed by Supplier / Customer Credit",                 0, 100, 10)
    pct_other    = 100 - pct_internal - pct_bank - pct_supplier
    if pct_other < 0:
        st.warning(f"⚠️ Financing percentages exceed 100% by {abs(pct_other)}%.")
        pct_other = 0
    else:
        st.caption(f"Other financing (equity, NGOs, family): {pct_other}%")

with colD2:
    st.markdown("**Access to Finance**")
    has_loan = st.radio(
        "Does the business currently have a bank loan or line of credit?",
        ["Yes", "No"], horizontal=True
    ) == "Yes"

    needs_finance = st.radio(
        "Does the business need external financing to operate or grow?",
        ["Yes", "No"], horizontal=True
    ) == "Yes"

    if has_loan:
        st.markdown("""
        <div style="background:#E3F2FD;border-left:4px solid #2196F3;
                    border-radius:8px;padding:12px;margin-top:8px;">
            <b>✅ Finance Access — Has Active Loan</b><br/>
            <small>Access to credit is a protective factor against distress.</small>
        </div>
        """, unsafe_allow_html=True)
    elif needs_finance:
        st.markdown("""
        <div style="background:#FFF3E0;border-left:4px solid #FF9800;
                    border-radius:8px;padding:12px;margin-top:8px;">
            <b>⚠️ Needs Finance — No Current Loan</b><br/>
            <small>The model will factor in this financing gap alongside other indicators.</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#F0FFF4;border-left:4px solid #4CAF50;
                    border-radius:8px;padding:12px;margin-top:8px;">
            <b>➡️ Does Not Need External Finance</b><br/>
            <small>Self-sufficient financing — neutral signal.</small>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E — MANAGEMENT AND LEADERSHIP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 👔 Section E — Management and Leadership")

colE1, colE2, colE3 = st.columns(3)

with colE1:
    manager_female = st.radio(
        "Gender of Top Manager / Owner",
        ["Male", "Female", "Not specified"], horizontal=True
    ) == "Female"

with colE2:
    manager_university = st.radio(
        "Does the top manager hold a university degree?",
        ["Yes", "No", "Not known"], horizontal=True
    ) == "Yes"

with colE3:
    st.markdown("**Management Practices**")
    mgmt_targets  = st.checkbox("Uses performance targets (KPIs, production targets)")
    mgmt_shares   = st.checkbox("Shares performance targets with employees")
    mgmt_pay_perf = st.checkbox("Non-manager salaries linked to performance")
    mgmt_score    = sum([mgmt_targets, mgmt_shares, mgmt_pay_perf])
    mgmt_label    = ["Weak (0/3)", "Basic (1/3)", "Moderate (2/3)", "Strong (3/3)"][mgmt_score]
    st.caption(f"Management quality: **{mgmt_label}**")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F — BUSINESS ENVIRONMENT & OBSTACLES
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 🚧 Section F — Business Environment & Obstacles")
st.caption("Rate how severely each factor affects your business operations.")

colF1, colF2 = st.columns(2)

obstacle_inputs = {}

with colF1:
    obstacle_inputs["electricity"] = st.select_slider(
        "⚡ Electricity Supply",
        options=OBSTACLE_SEVERITY_OPTIONS, value="Minor obstacle"
    )
    obstacle_inputs["transport"] = st.select_slider(
        "🚛 Transport and Roads",
        options=OBSTACLE_SEVERITY_OPTIONS, value="No obstacle"
    )
    obstacle_inputs["finance"] = st.select_slider(
        "💳 Access to Finance",
        options=OBSTACLE_SEVERITY_OPTIONS, value="Moderate obstacle"
    )
    obstacle_inputs["customs"] = st.select_slider(
        "📦 Customs and Trade Regulations",
        options=OBSTACLE_SEVERITY_OPTIONS, value="No obstacle"
    )

with colF2:
    obstacle_inputs["corruption"] = st.select_slider(
        "🏛️ Corruption / Bribery",
        options=OBSTACLE_SEVERITY_OPTIONS, value="Minor obstacle"
    )
    obstacle_inputs["informality"] = st.select_slider(
        "🔄 Competition from Informal Sector",
        options=OBSTACLE_SEVERITY_OPTIONS, value="Minor obstacle"
    )
    obstacle_inputs["workforce"] = st.select_slider(
        "🎓 Inadequately Educated Workforce",
        options=OBSTACLE_SEVERITY_OPTIONS, value="No obstacle"
    )

biggest_obstacle = st.selectbox(
    "Which single factor is the BIGGEST obstacle to your business today?",
    OBSTACLE_OPTIONS, index=0
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION G — ADDITIONAL CONTEXT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📝 Section G — Additional Context")

colG1, colG2, colG3 = st.columns(3)

with colG1:
    has_bank_account   = st.radio("Does the business have a formal bank account?",
                                  ["Yes", "No"], horizontal=True) == "Yes"
    has_external_audit = st.radio("Are financial statements reviewed by an external auditor?",
                                  ["Yes", "No"], horizontal=True) == "Yes"

with colG2:
    has_website      = st.radio("Does the business have a website or active social media presence?",
                                ["Yes", "No"], horizontal=True) == "Yes"
    has_business_plan = st.radio("Does the business have a formal written business plan?",
                                 ["Yes", "No"], horizontal=True) == "Yes"

with colG3:
    years_operating = 2025 - year_founded
    st.markdown(f"""
    <div style="background:#F4F6F8;border-radius:8px;padding:14px;
                border-left:4px solid #0F7B8C;">
        <div style="font-size:0.8rem;color:#64748B;">Years in Operation</div>
        <div style="font-size:1.4rem;font-weight:700;color:#0D1B3E;">
            {years_operating} year{'s' if years_operating != 1 else ''}
        </div>
        <div style="font-size:0.75rem;color:#94A3B8;margin-top:4px;">
            Founded {year_founded}
        </div>
    </div>
    """, unsafe_allow_html=True)

    additional_notes = st.text_area(
        "Any additional context about this business (optional)",
        placeholder="e.g. Recently expanded to a second location. Lost a major client in 2024...",
        height=80
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
predict_btn = st.button(
    "🔍  Run Distress Assessment",
    type="primary",
    use_container_width=True
)

if predict_btn:
    inputs = {
        "country":            country,
        "region":             region,
        "broad_sector":       broad_sector,
        "detailed_sector":    detailed_sector,
        "legal_status":       legal_status,
        "firm_size_num":      firm_size_num,
        "employees_now":      employees_now,
        "employees_3yr":      employees_3yr,
        "annual_sales":       annual_sales,
        "has_loan":           has_loan,
        "needs_finance":      needs_finance,
        "pct_internal":       pct_internal,
        "pct_bank":           pct_bank,
        "pct_supplier":       pct_supplier,
        "export_pct":         export_pct,
        "manager_female":     manager_female,
        "manager_university": manager_university,
        "mgmt_targets":       mgmt_targets,
        "mgmt_shares":        mgmt_shares,
        "mgmt_pay_perf":      mgmt_pay_perf,
        "obstacles":          obstacle_inputs,
        "biggest_obstacle":   biggest_obstacle,
        "year_founded":       year_founded,
    }

    with st.spinner("Analysing business profile against 14,688 reference businesses…"):
        encoded = encode_input(inputs)
        result  = predict(models, encoded)

    prediction = result["prediction"]
    probs      = result["probs"]
    pred_idx   = result["pred_idx"]

    st.markdown("---")
    st.markdown("## 📊 Assessment Results")

    # ── Verdict banner ────────────────────────────────────────────────────────
    icon   = CLASS_ICONS[prediction]
    colour = CLASS_COLOURS[prediction]

    css_class = {
        "Stable":        "verdict-stable",
        "Moderate Risk": "verdict-moderate",
        "High Risk":     "verdict-highrisk",
    }[prediction]

    biz_display = f"**{business_name}**" if business_name else "This business profile"

    if prediction == "Stable":
        verdict_text = (
            f"{biz_display} shows **no significant distress signals**. "
            f"The profile is consistent with {probs['Stable']*100:.1f}% confidence of "
            f"stable business operations based on comparable firms in the dataset."
        )
    elif prediction == "Moderate Risk":
        verdict_text = (
            f"{biz_display} shows **vulnerability patterns** consistent with moderate-risk firms. "
            f"Model confidence: {probs['Moderate Risk']*100:.1f}%. "
            f"Targeted support or monitoring is recommended before committing investment."
        )
    else:
        verdict_text = (
            f"{biz_display} shows **multiple simultaneous distress indicators**. "
            f"This profile matches High Risk patterns with {probs['High Risk']*100:.1f}% confidence. "
            f"Significant intervention or restructuring would be required before this "
            f"business is suitable for investment without very high risk tolerance."
        )

    st.markdown(f"""
    <div class="{css_class}">
        <div style="font-size:1.5rem;font-weight:700;margin-bottom:8px;">
            {icon} &nbsp; {prediction}
        </div>
        <div style="font-size:1rem;color:#374151;">{verdict_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability breakdown ─────────────────────────────────────────────────
    st.markdown("### 📈 Probability Breakdown")
    col_probs, col_chart = st.columns([1, 1.4])

    with col_probs:
        for cls in CLASS_NAMES:
            p      = probs[cls]
            c      = CLASS_COLOURS[cls]
            marker = " ◀ predicted" if cls == prediction else ""
            bold   = "700" if cls == prediction else "400"
            st.markdown(f"""
            <div style="margin:8px 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                    <span style="font-weight:{bold};color:#0D1B3E;">
                        {CLASS_ICONS[cls]} {cls}{marker}
                    </span>
                    <span style="font-weight:700;color:{c};">{p*100:.1f}%</span>
                </div>
                <div style="background:#E2E8F0;border-radius:4px;height:12px;">
                    <div style="background:{c};width:{int(p*100)}%;
                                height:12px;border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_chart:
        fig, ax = plt.subplots(figsize=(5, 3))
        vals    = [probs[c] for c in CLASS_NAMES]
        colors  = [CLASS_COLOURS[c] for c in CLASS_NAMES]
        bars    = ax.bar(CLASS_NAMES, vals, color=colors, edgecolor="none", width=0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Risk Class Probabilities")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.015,
                    f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        if pred_idx is not None:
            bars[pred_idx].set_edgecolor("#0D1B3E")
            bars[pred_idx].set_linewidth(2.5)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Key risk drivers (from top v2 SHAP features) ──────────────────────────
    st.markdown("### 🔍 Key Risk Factors in This Profile")

    obs_sev_map = OBSTACLE_SEVERITY_MAP
    finance_score = obs_sev_map.get(obstacle_inputs.get("finance", "No obstacle"), 0)
    finance_double = needs_finance and finance_score >= 3
    emp_now_v = float(employees_now)
    log_s     = float(np.log1p(max(float(annual_sales), 0)))
    spe       = (log_s / emp_now_v) if emp_now_v > 0 else 0.0

    factors = []
    if needs_finance and not has_loan:
        factors.append(("⚠️", "Needs finance but has no loan",
                        "Unfulfilled financing need is the strongest predictor of distress in this dataset."))
    if has_loan:
        factors.append(("✅", "Active bank loan / line of credit",
                        "Finance access is a significant protective factor."))
    if finance_double:
        factors.append(("🚨", "Finance double stress",
                        "Needs finance AND rates access to finance as a major obstacle simultaneously."))
    if spe < 0.5:
        factors.append(("⚠️", f"Low sales per employee ({spe:.2f})",
                        "Labour productivity is below the dataset median — a key distress indicator."))
    total_obs = sum(1 for k in obstacle_inputs if obs_sev_map.get(obstacle_inputs[k], 0) >= 3)
    if total_obs >= 3:
        factors.append(("🚨", f"{total_obs} major obstacles active simultaneously",
                        "High cumulative obstacle burden is associated with High Risk classification."))
    if not factors:
        factors.append(("✅", "No major individual risk factors detected",
                        "Profile indicators are broadly within the Stable range."))

    for icon_f, title_f, detail_f in factors:
        bg = "#FFF5F5" if icon_f == "🚨" else ("#FFF3E0" if icon_f == "⚠️" else "#F0FFF4")
        bc = "#F44336" if icon_f == "🚨" else ("#FF9800" if icon_f == "⚠️" else "#4CAF50")
        st.markdown(f"""
        <div style="background:{bg};border-left:5px solid {bc};border-radius:8px;
                    padding:12px;margin:6px 0;">
            <div style="font-weight:700;color:#0D1B3E;">{icon_f} {title_f}</div>
            <div style="color:#64748B;font-size:0.85rem;margin-top:3px;">{detail_f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Investor guidance ─────────────────────────────────────────────────────
    st.markdown("### 💡 Investor Guidance")

    if prediction == "Stable":
        st.success("""
        **✅ Low-risk profile — suitable for standard due diligence process.**

        This business profile does not show structural distress indicators.
        Recommended next steps:
        - Proceed with standard financial due diligence
        - Verify the sales and employee figures with documentation
        - Assess market opportunity and competitive position
        - Standard loan/investment terms are appropriate
        """)
    elif prediction == "Moderate Risk":
        st.warning(f"""
        **⚠️ Moderate-risk profile — proceed with conditions and monitoring.**

        Recommended next steps:
        - Address the financing gap before finalising funding if present
        - Consider phased funding tied to milestone targets
        - Require quarterly reporting during the first year
        - Explore whether business support (not just capital) is needed
        """)
    else:
        st.error(f"""
        **🚨 High-risk profile — significant intervention required before funding.**

        Multiple distress indicators are simultaneously active.
        Recommended next steps:
        - Do NOT commit funding based on this profile alone
        - Commission a full independent financial and operational audit
        - Identify whether distress is temporary (cash flow) or structural (demand collapse)
        - If funding proceeds, require a turnaround plan with measurable milestones
        - Consider equity stake rather than debt to share the risk
        """)

    # ── Full profile summary ──────────────────────────────────────────────────
    st.markdown("### 📋 Business Profile Summary")
    with st.expander("Click to view full profile as submitted"):
        profile_df = pd.DataFrame([
            ("Business Name",          business_name or "Not provided"),
            ("Country",                country),
            ("Region",                 region),
            ("Year Founded",           year_founded),
            ("Years Operating",        f"{years_operating} years"),
            ("Legal Structure",        legal_status),
            ("Broad Sector",           broad_sector),
            ("Detailed Sector",        detailed_sector),
            ("Firm Size",              firm_size),
            ("Current Employees",      f"{employees_now:,}"),
            ("Employees 3 Years Ago",  f"{employees_3yr:,}"),
            ("Annual Sales",           f"{annual_sales:,.0f} {currency_map.get(country,'LC')}"),
            ("Has Bank Loan",          "Yes" if has_loan else "No"),
            ("Needs External Finance", "Yes" if needs_finance else "No"),
            ("Exports Goods/Services", "Yes" if is_exporting else "No"),
            ("Internal Finance %",     f"{pct_internal}%"),
            ("Bank Finance %",         f"{pct_bank}%"),
            ("Manager Gender",         "Female" if manager_female else "Male"),
            ("Manager Has Degree",     "Yes" if manager_university else "No"),
            ("Mgmt Quality Score",     f"{mgmt_score}/3 — {mgmt_label}"),
            ("Biggest Obstacle",       biggest_obstacle),
            ("Electricity Obstacle",   obstacle_inputs["electricity"]),
            ("Finance Obstacle",       obstacle_inputs["finance"]),
            ("Corruption Obstacle",    obstacle_inputs["corruption"]),
            ("Informality Obstacle",   obstacle_inputs["informality"]),
            ("Workforce Obstacle",     obstacle_inputs["workforce"]),
            ("Transport Obstacle",     obstacle_inputs["transport"]),
            ("Has Bank Account",       "Yes" if has_bank_account else "No"),
            ("External Audit",         "Yes" if has_external_audit else "No"),
            ("Has Website/Social",     "Yes" if has_website else "No"),
            ("Has Business Plan",      "Yes" if has_business_plan else "No"),
        ], columns=["Field", "Value"])
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.info("""
    **⚠️ Assessment Disclaimer:**
    This tool provides probabilistic risk scoring based on patterns in 14,688 real African
    businesses surveyed by the World Bank. It is a structured decision-support tool — not a
    definitive verdict. Predictions should be one input among several in any funding or
    investment decision. The model does not have access to proprietary financial records,
    market intelligence, or information about the entrepreneur's track record.
    Always complement this assessment with direct due diligence.
    """)
