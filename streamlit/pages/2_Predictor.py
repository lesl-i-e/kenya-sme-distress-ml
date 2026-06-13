"""Page 2 — Business Predictor (v2 — dual SME / Startup mode)"""
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
Complete the business profile below. The model compares your inputs against
**14,688 real East African businesses** surveyed by the World Bank and returns
a structured distress risk score with explanations.
""")

# ── Mode selector ─────────────────────────────────────────────────────────────
st.markdown("### Choose Assessment Mode")
mode = st.radio(
    "Select the type of business you are assessing",
    ["🏪 Running SME — has operating history",
     "🚀 Startup / New Venture — projections only"],
    horizontal=True,
    label_visibility="collapsed",
)
is_startup = mode.startswith("🚀")

if is_startup:
    st.info(
        "**Startup Mode:** Answer based on your business plan and financial projections. "
        "The model assesses your *projected risk profile* against patterns from 14,688 "
        "real African businesses. Results reflect how similar startup profiles tend to "
        "perform — not a guarantee of outcome."
    )
else:
    st.info(
        "**Running SME Mode:** A business with at least 6 months of operating history. "
        "Use real employee numbers, actual revenue, and observed financial behaviour."
    )

models = load_models()

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED FIELDS (appear in both modes)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("### 🏢 Business / Venture Profile")
c1, c2, c3 = st.columns(3)
with c1:
    country = st.selectbox("Country of Operation*", COUNTRIES)
with c2:
    region = st.selectbox("Region / City", COUNTRY_REGIONS.get(country, ["Other"]))
with c3:
    broad_sector = st.selectbox("Broad Sector*", BROAD_SECTORS)

c4, c5 = st.columns(2)
with c4:
    detailed_sector = st.selectbox(
        "Detailed Sector / Industry",
        DETAILED_SECTORS.get(broad_sector, ["Other"])
    )
with c5:
    legal_status = st.selectbox("Legal Structure", LEGAL_STATUSES)

# ═══════════════════════════════════════════════════════════════════════════════
#  SME-ONLY SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════
if not is_startup:
    st.markdown("---")
    st.markdown("### 👥 Workforce")
    w1, w2, w3 = st.columns(3)
    with w1:
        employees_now  = st.number_input("Current full-time employees*", min_value=1, max_value=10000, value=15, step=1)
        year_founded_s = st.number_input("Year business was founded*", min_value=1980, max_value=2025, value=2018, step=1)
    with w2:
        employees_3yr  = st.number_input("Employees 3 years ago*", min_value=0, max_value=10000, value=12, step=1,
                                          help="Enter 0 if business is less than 3 years old")
    with w3:
        if employees_3yr > 0:
            growth = (employees_now - employees_3yr) / employees_3yr * 100
            colour = "#2196F3" if growth > 10 else ("#F44336" if growth < -10 else "#FF9800")
            arrow  = "📈" if growth > 10 else ("📉" if growth < -10 else "➡️")
            st.markdown(f"""
            <div style="background:#F4F6F8;border-radius:8px;padding:14px;
                        border-left:4px solid {colour};margin-top:28px;">
                <div style="font-size:0.8rem;color:#64748B;">Employment Trend</div>
                <div style="font-size:1.1rem;font-weight:700;color:{colour};">
                    {arrow} {growth:+.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💰 Finance & Revenue")
    f1, f2 = st.columns(2)
    CURRENCY = {"Kenya":"KSh","Uganda":"UGX","Tanzania":"TZS","South Africa":"ZAR",
                "Ghana":"GH₵","Nigeria":"₦","Ethiopia":"ETB","Rwanda":"RF"}
    cur = CURRENCY.get(country, "USD")
    with f1:
        annual_sales = st.number_input(f"Annual revenue ({cur})*",
                                        min_value=0, max_value=10_000_000_000,
                                        value=5_000_000, step=100_000)
        has_loan      = st.radio("Active bank loan / overdraft?*", ["Yes","No"], horizontal=True) == "Yes"
        needs_finance = st.radio("Needs external financing to operate/grow?*", ["Yes","No"], horizontal=True) == "Yes"
    with f2:
        pct_internal = st.slider("% Financed internally (savings/retained earnings)", 0, 100, 70)
        pct_bank     = st.slider("% Financed by bank / microfinance",               0, 100, 15)
        pct_supplier = st.slider("% Financed by supplier / customer credit",         0, 100, 10)
        st.caption(f"Other / informal: {max(0, 100 - pct_internal - pct_bank - pct_supplier)}%")

    st.markdown("---")
    st.markdown("### ⚙️ Operations")
    o1, o2 = st.columns(2)
    with o1:
        capacity = st.slider("Capacity utilisation %*  (below 60% = distress signal)",
                              0, 100, 65, help="% of maximum production/service capacity currently being used")
        is_exporting = st.checkbox("Business exports goods/services (>5% of sales abroad)")
        export_pct   = st.slider("Export share % of total sales", 5, 100, 15) if is_exporting else 0
    with o2:
        profit_pos = st.selectbox("Profit position", ["Profitable","Breaking even","Loss-making"])

    st.markdown("---")
    st.markdown("### 👔 Management")
    m1, m2 = st.columns(2)
    with m1:
        manager_female     = st.radio("Top manager / owner gender", ["Male","Female","Not specified"], horizontal=True) == "Female"
        manager_university = st.radio("Manager holds university degree?", ["Yes","No","Unknown"], horizontal=True) == "Yes"
    with m2:
        mgmt_targets  = st.checkbox("Uses KPIs / performance targets")
        mgmt_shares   = st.checkbox("Shares targets with employees")
        mgmt_pay_perf = st.checkbox("Non-manager pay linked to performance")

    st.markdown("---")
    st.markdown("### 🌍 Business Environment")
    e1, e2 = st.columns(2)
    with e1:
        obs_electricity = st.select_slider("⚡ Electricity supply obstacle", OBSTACLE_SEVERITY_OPTIONS, value="Minor obstacle")
        obs_finance     = st.select_slider("💳 Access to finance obstacle",  OBSTACLE_SEVERITY_OPTIONS, value="Moderate obstacle")
        obs_transport   = st.select_slider("🚛 Transport / roads obstacle",  OBSTACLE_SEVERITY_OPTIONS, value="No obstacle")
        obs_customs     = st.select_slider("📦 Customs / trade obstacle",    OBSTACLE_SEVERITY_OPTIONS, value="No obstacle")
    with e2:
        obs_corruption  = st.select_slider("🏛️ Corruption / bribery obstacle",       OBSTACLE_SEVERITY_OPTIONS, value="Minor obstacle")
        obs_informality = st.select_slider("🔄 Informal sector competition obstacle", OBSTACLE_SEVERITY_OPTIONS, value="Minor obstacle")
        obs_workforce   = st.select_slider("🎓 Uneducated workforce obstacle",        OBSTACLE_SEVERITY_OPTIONS, value="No obstacle")

    biggest_obstacle = st.selectbox("Biggest single obstacle to your business today", OBSTACLE_OPTIONS)

    # ── Predict button (SME) ──────────────────────────────────────────────────
    st.markdown("---")
    sme_btn = st.button("🔍  Analyse This Business", type="primary", use_container_width=True)

    if sme_btn:
        if not country:
            st.error("Please select a country.")
        else:
            inputs = {
                "country": country, "region": region,
                "broad_sector": broad_sector, "detailed_sector": detailed_sector,
                "legal_status": legal_status,
                "firm_size_num": 1,
                "employees_now": employees_now,
                "employees_3yr": employees_3yr,
                "annual_sales":  annual_sales,
                "has_loan":      has_loan,
                "needs_finance": needs_finance,
                "pct_internal":  pct_internal,
                "pct_bank":      pct_bank,
                "pct_supplier":  pct_supplier,
                "export_pct":    export_pct,
                "year_founded":  year_founded_s,
                "manager_female":     manager_female,
                "manager_university": manager_university,
                "mgmt_targets":  mgmt_targets,
                "mgmt_shares":   mgmt_shares,
                "mgmt_pay_perf": mgmt_pay_perf,
                "obstacles": {
                    "electricity": obs_electricity, "finance": obs_finance,
                    "transport": obs_transport,     "customs": obs_customs,
                    "corruption": obs_corruption,   "informality": obs_informality,
                    "workforce": obs_workforce,
                },
                "biggest_obstacle": biggest_obstacle,
            }

            with st.spinner("Analysing against 14,688 reference businesses…"):
                encoded = encode_input(inputs)
                result  = predict(models, encoded)

            _render_result(result, inputs, mode="sme", capacity=capacity,
                           employees_now=employees_now, employees_3yr=employees_3yr)


# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP-ONLY SECTIONS
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("---")
    st.markdown("### 👥 Team & Founders")
    t1, t2 = st.columns(2)
    with t1:
        num_founders   = st.selectbox("Number of founders*", [1,2,3,4], index=1)
        team_size_proj = st.number_input("Planned Year-1 team size (incl. founders)*", min_value=1, max_value=200, value=4)
        stage          = st.selectbox("Venture stage*", [
            "💡 Idea / concept (pre-product)",
            "🛠️ MVP / prototype built",
            "🧪 Pilot / early customers (<6 months)",
            "📦 Early revenue (6–12 months)",
        ])
    with t2:
        experience     = st.selectbox("Founders' relevant industry experience*", [
            "None — first-time in this sector",
            "1–2 years", "3–5 years", "6–10 years", "10+ years",
        ])
        prior_biz      = st.radio("Has any founder previously run a business?*", ["Yes","No"], horizontal=True) == "Yes"
        manager_female = st.radio("Lead founder gender", ["Male","Female","Not specified"], horizontal=True) == "Female"
        manager_univ   = st.radio("Founder holds university degree?", ["Yes","No"], horizontal=True) == "Yes"

    st.markdown("---")
    st.markdown("### 💰 Financial Projections")
    p1, p2 = st.columns(2)
    CURRENCY = {"Kenya":"KSh","Uganda":"UGX","Tanzania":"TZS","South Africa":"ZAR",
                "Ghana":"GH₵","Nigeria":"₦","Ethiopia":"ETB","Rwanda":"RF"}
    cur = CURRENCY.get(country, "USD")
    with p1:
        capital_raised = st.selectbox("Startup capital available / raised*", [
            "None — bootstrapping",
            f"< {cur} 500K (personal savings)",
            f"{cur} 500K – 2.5M",
            f"{cur} 2.5M – 10M",
            f"{cur} 10M – 50M",
            f"> {cur} 50M (institutional)",
        ])
        fund_source    = st.selectbox("Primary funding source*", [
            "Self-funded / bootstrapped","Family & friends","Angel investor",
            "Grant / competition prize","Bank loan","SACCO / microfinance",
            "Venture capital","Multiple sources",
        ])
        needs_finance  = st.radio("Needs external financing to launch/grow?*", ["Yes","No"], horizontal=True) == "Yes"
        has_loan       = st.radio("Loan / credit line already secured?*",       ["Yes","No"], horizontal=True) == "Yes"
    with p2:
        rev_y1      = st.number_input(f"Projected Year-1 revenue ({cur})",   min_value=0, value=1_000_000, step=100_000)
        rev_y3      = st.number_input(f"Projected Year-3 revenue ({cur})",   min_value=0, value=5_000_000, step=100_000)
        breakeven_m = st.selectbox("Projected months to break-even", [
            "0–6 months","6–12 months","12–18 months","18–24 months",
            "2–3 years","More than 3 years / uncertain",
        ])
        cap_proj    = st.slider("Planned capacity utilisation at launch (%)", 0, 100, 50,
                                 help="Realistic estimate of % of potential output you will actually sell at launch")

    year_founded_t = st.number_input("Proposed / planned start year", min_value=2024, max_value=2030, value=2025, step=1)

    st.markdown("---")
    st.markdown("### 🎯 Market & Product")
    mk1, mk2 = st.columns(2)
    with mk1:
        has_sales    = st.radio("Has the venture made any sales (even pilot)?*", ["Yes","No"], horizontal=True) == "Yes"
        has_loi      = st.radio("Has paying customer commitment / LOI?", ["Yes","No"], horizontal=True) == "Yes"
        demand       = st.selectbox("Demand strength for your product/service*", [
            "✅ Strong — customers actively looking for this",
            "⚠️ Moderate — some demand, awareness needed",
            "❌ Weak — creating new behaviour / educating market",
        ])
    with mk2:
        competition  = st.selectbox("Competition level in target market*", [
            "Low — few or no direct competitors",
            "Moderate — some established players",
            "High — crowded market",
            "Dominated — large incumbents difficult to displace",
        ])
        mkt_research = st.selectbox("Market size assessment*", [
            "No formal assessment done",
            "Rough estimate only",
            "Basic research completed",
            "Detailed TAM/SAM/SOM analysis done",
        ])
        is_exporting = st.checkbox("Plans to export or serve international clients")
        export_pct   = 10 if is_exporting else 0

    st.markdown("---")
    st.markdown("### 🌍 Operating Environment")
    env1, env2 = st.columns(2)
    with env1:
        obs_electricity = st.select_slider("⚡ Expected electricity / infrastructure challenge",
                                            OBSTACLE_SEVERITY_OPTIONS, value="Minor obstacle")
        obs_finance     = st.select_slider("💳 Access to finance barrier",
                                            OBSTACLE_SEVERITY_OPTIONS, value="Moderate obstacle")
        obs_informality = st.select_slider("🔄 Informal sector competition",
                                            OBSTACLE_SEVERITY_OPTIONS, value="Minor obstacle")
    with env2:
        obs_transport   = st.select_slider("🚛 Transport / infrastructure",
                                            OBSTACLE_SEVERITY_OPTIONS, value="No obstacle")
        obs_corruption  = st.select_slider("🏛️ Corruption / regulatory burden",
                                            OBSTACLE_SEVERITY_OPTIONS, value="No obstacle")
        obs_workforce   = st.select_slider("🎓 Workforce skills gap",
                                            OBSTACLE_SEVERITY_OPTIONS, value="No obstacle")
        obs_customs     = st.select_slider("📦 Customs / trade regulations",
                                            OBSTACLE_SEVERITY_OPTIONS, value="No obstacle")

    mgmt_targets  = True   # startups with a plan inherently have targets
    mgmt_shares   = prior_biz
    mgmt_pay_perf = False

    biggest_obstacle = st.selectbox("Expected biggest obstacle at launch", OBSTACLE_OPTIONS)

    # ── Predict button (Startup) ──────────────────────────────────────────────
    st.markdown("---")
    startup_btn = st.button("🚀  Assess This Startup", type="primary", use_container_width=True)

    if startup_btn:
        if not country:
            st.error("Please select a country.")
        else:
            # Map startup inputs to model-compatible format
            exp_map   = {"None — first-time in this sector":0,"1–2 years":1,"3–5 years":3,"6–10 years":7,"10+ years":10}
            exp_yrs   = exp_map.get(experience, 0)
            annual_sales_est = rev_y1

            inputs = {
                "country": country, "region": region,
                "broad_sector": broad_sector, "detailed_sector": detailed_sector,
                "legal_status": legal_status,
                "firm_size_num": 1,
                "employees_now": float(team_size_proj),
                "employees_3yr": float(num_founders),
                "annual_sales":  annual_sales_est,
                "has_loan":      has_loan,
                "needs_finance": needs_finance,
                "pct_internal":  80 if "bootstrap" in fund_source.lower() else 40,
                "pct_bank":      30 if "bank" in fund_source.lower() else 5,
                "pct_supplier":  5,
                "export_pct":    export_pct,
                "year_founded":  year_founded_t,
                "manager_female":     manager_female,
                "manager_university": manager_univ,
                "mgmt_targets":  mgmt_targets,
                "mgmt_shares":   mgmt_shares,
                "mgmt_pay_perf": mgmt_pay_perf,
                "obstacles": {
                    "electricity": obs_electricity, "finance": obs_finance,
                    "transport": obs_transport,     "customs": obs_customs,
                    "corruption": obs_corruption,   "informality": obs_informality,
                    "workforce": obs_workforce,
                },
                "biggest_obstacle": biggest_obstacle,
            }

            with st.spinner("Analysing startup profile…"):
                encoded = encode_input(inputs)
                result  = predict(models, encoded)

            _render_result(result, inputs, mode="startup",
                           capacity=cap_proj,
                           employees_now=team_size_proj,
                           employees_3yr=num_founders,
                           startup_extras={
                               "stage": stage, "has_sales": has_sales,
                               "demand": demand, "competition": competition,
                               "prior_biz": prior_biz, "exp_yrs": exp_yrs,
                               "breakeven": breakeven_m,
                           })


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED RESULT RENDERER
# ═══════════════════════════════════════════════════════════════════════════════
def _render_result(result, inputs, mode, capacity,
                   employees_now, employees_3yr, startup_extras=None):

    prediction = result["prediction"]
    probs      = result["probs"]
    pred_idx   = result["pred_idx"]

    COUNTRY_RISK = {
        "South Africa": ("18.4%","South Africa has the highest High Risk rate in the dataset."),
        "Uganda":       ("13.1%","Above average — credit access and employment trends are key."),
        "Kenya":        ("12.2%","Above median — services sector shows elevated vulnerability."),
        "Ghana":        ("10.9%","Moderate — currency and infrastructure are key stressors."),
        "Nigeria":      ("5.4%", "Lower observed distress rate, but informal competition is significant."),
        "Ethiopia":     ("5.1%", "Low but growing — access to finance is the primary constraint."),
        "Tanzania":     ("4.6%", "Low distress rate — stable macro conditions."),
        "Rwanda":       ("1.8%", "Lowest in dataset — strong regulatory environment."),
    }

    colour_map = {"Stable":"#2196F3","Moderate Risk":"#FF9800","High Risk":"#F44336"}
    bg_map     = {"Stable":"#E3F2FD","Moderate Risk":"#FFF3E0","High Risk":"#FFEBEE"}
    css_map    = {"Stable":"verdict-stable","Moderate Risk":"verdict-moderate","High Risk":"verdict-highrisk"}
    icon_map   = CLASS_ICONS

    st.markdown("---")
    st.markdown("## 📊 Assessment Result")

    colour  = colour_map[prediction]
    bg      = bg_map[prediction]
    icon    = icon_map[prediction]
    mode_tag = "🚀 Startup" if mode == "startup" else "🏪 Running SME"

    st.markdown(f"""
    <div class="{css_map[prediction]}">
        <div style="font-size:1.6rem;font-weight:800;margin-bottom:6px;">
            {icon} &nbsp; {prediction}
            &nbsp;<span style="font-size:0.7rem;background:rgba(0,0,0,0.08);
                   padding:3px 10px;border-radius:999px;">{mode_tag}</span>
        </div>
        <div style="color:#374151;font-size:0.95rem;">
            {'No significant structural distress signals detected in this profile.' if prediction=='Stable'
             else 'Vulnerability signals present — monitor and address finance access.' if prediction=='Moderate Risk'
             else 'Multiple distress indicators converging — significant intervention required.'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("### 📈 Risk Class Probabilities")
    col_bars, col_chart = st.columns([1, 1.4])
    with col_bars:
        for cls in CLASS_NAMES:
            p = probs[cls]; c = colour_map[cls]
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
        vals   = [probs[c] for c in CLASS_NAMES]
        colors = [colour_map[c] for c in CLASS_NAMES]
        bars   = ax.bar(CLASS_NAMES, vals, color=colors, edgecolor="none", width=0.5)
        ax.set_ylim(0, 1); ax.set_ylabel("Probability")
        ax.set_title("Risk Class Probabilities")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                    f"{v*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
        bars[pred_idx].set_edgecolor("#0D1B3E"); bars[pred_idx].set_linewidth(2.5)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Key risk signals
    st.markdown("### 🔍 Key Risk Factors")
    obs        = inputs.get("obstacles", {})
    fin_sev    = OBSTACLE_SEVERITY_MAP.get(obs.get("finance","No obstacle"), 0)
    fin_double = inputs.get("needs_finance") and fin_sev >= 3
    log_s      = np.log1p(max(float(inputs.get("annual_sales", 0)), 0))
    emp        = float(employees_now)
    spe        = log_s / emp if emp > 0 else 0
    n_obs      = sum(1 for k,v in obs.items() if OBSTACLE_SEVERITY_MAP.get(v,0) >= 3)

    def risk_card(emoji, title, triggered, detail, projected=False):
        cls   = "projected" if projected else ("triggered" if triggered else "clear")
        chip  = "PROJECTED ⚡" if projected else ("TRIGGERED ✗" if triggered else "ALL CLEAR ✓")
        bg2   = ("#fff5f5" if triggered and not projected
                 else "#fffbeb" if projected else "#f0fdf4")
        bc    = "#F44336" if triggered and not projected else ("#F59E0B" if projected else "#10B981")
        return f"""<div style="background:{bg2};border-left:5px solid {bc};border-radius:8px;
                    padding:12px;margin:6px 0;">
            <div style="font-weight:700;font-size:0.9rem;">{emoji} {title}
                <span style="font-size:0.65rem;font-weight:700;background:rgba(0,0,0,0.08);
                      padding:2px 8px;border-radius:999px;margin-left:8px;">{chip}</span>
            </div>
            <div style="color:#6B7280;font-size:0.8rem;margin-top:5px;">{detail}</div>
        </div>"""

    needs = inputs.get("needs_finance", False)
    has   = inputs.get("has_loan", False)
    credit_gap = needs and not has

    if mode == "sme":
        emp_growth = (float(employees_now) - float(employees_3yr)) / float(employees_3yr) * 100 if float(employees_3yr) > 0 else 0
        emp_shrink = emp_growth < -10
        low_cap    = capacity < 60
        signals_html = "".join([
            risk_card("💳","Credit Gap (Needs Finance, No Loan)", credit_gap,
                      "Unfulfilled financing need is the #1 distress predictor (SHAP 0.481)."),
            risk_card("👥","Employment Shrinkage (>10% decline)", emp_shrink,
                      f"Workforce change: {emp_growth:+.1f}%. Threshold: −10%."),
            risk_card("🏭","Low Capacity Utilisation (<60%)", low_cap,
                      f"Currently at {capacity}%. Below 60% signals operational distress."),
        ])
    else:
        se = startup_extras or {}
        weak_demand  = "Weak" in se.get("demand","")
        high_comp    = "Dominated" in se.get("competition","") or "High" in se.get("competition","")
        signals_html = "".join([
            risk_card("💳","Finance Gap (Needs Finance, No Loan)", credit_gap,
                      "Securing finance before launch is the highest-leverage risk reduction action."),
            risk_card("📈","Market Demand Strength", weak_demand,
                      "Weak demand means educating the market — expensive and slow for a new venture.", projected=not weak_demand),
            risk_card("⚔️","Competitive Environment", high_comp,
                      f"Competition level: {se.get('competition','—')}.", projected=not high_comp),
        ])

    st.markdown(signals_html, unsafe_allow_html=True)

    # Country benchmark
    st.markdown("### 🌍 Country Benchmark")
    bench = COUNTRY_RISK.get(inputs.get("country",""), None)
    if bench:
        pct, note = bench
        st.info(f"**{inputs['country']}** — High Risk rate in dataset: **{pct}**. {note}")
    else:
        st.info("High Risk rates range from **1.8%** (Rwanda) to **18.4%** (South Africa) across the dataset.")

    # Recommendations
    st.markdown("### ✅ Recommended Actions")
    if mode == "sme":
        if prediction == "Stable":
            recos = [
                ("Protect your credit access","Maintain banking relationships proactively — never wait until you need emergency credit."),
                ("Build a 3-month cash buffer","Target 90 days of operating expenses as a reserve before expanding."),
                ("Push capacity utilisation above 80%" if capacity < 80 else "Plan for capacity expansion",
                 "You have headroom to grow before needing to invest in new infrastructure." if capacity < 80 else "You are operating at high capacity — now is the right time to plan expansion."),
            ]
        elif prediction == "Moderate Risk":
            recos = [
                ("Resolve the financing gap first" if credit_gap else "Maintain your loan relationship",
                 "Explore SACCOs, development finance, or invoice discounting. The gap is the #1 distress predictor." if credit_gap else "Keep repayments current and explore increasing your credit facility."),
                ("Get capacity above 60%" if capacity < 60 else "Protect your capacity advantage",
                 "Low utilisation signals demand or cost structure problems — investigate pricing and marketing." if capacity < 60 else "Above 60% keeps you out of the High Risk category. Invest in preventive maintenance."),
                ("Document your stability for investors","A clear financial narrative will open doors to better-priced credit and investment."),
            ]
        else:
            recos = [
                ("Address the financing gap immediately",
                 "Contact your nearest development bank or SACCO this week." if credit_gap else "Renegotiate loan terms before default — a restructured loan is better than a defaulted one."),
                ("Commission an independent financial audit",
                 "Identify which parts of the business are viable before committing any new capital."),
                ("Seek specialist business support",
                 "MSEA (Kenya), NBSSI (Ghana), SMEDAN (Nigeria), BDS providers in your region."),
            ]
    else:
        se     = startup_extras or {}
        recos  = []
        if credit_gap:
            recos.append(("Secure financing before launch",
                           "Starting with unresolved finance need is the #1 predictor of early failure. Explore angels, SACCOs, grant competitions."))
        if not se.get("has_sales"):
            recos.append(("Get to paying customers first",
                           "Even 3 paying pilot customers de-risk your venture more than any plan document."))
        if "Weak" in se.get("demand",""):
            recos.append(("Validate demand before scaling investment",
                           "Narrow your target segment to the most motivated early adopters before spending on marketing."))
        if "Dominated" in se.get("competition","") or "High" in se.get("competition",""):
            recos.append(("Find a defensible niche",
                           "In a crowded market, competing broadly is fatal for startups. Be dominant in one geography, price tier, or service model first."))
        if len(recos) < 2:
            recos.append(("Formalise your financial projections",
                           "A 3-year model with clear assumptions is expected by any investor or lender — build it now."))
        recos = recos[:3]

    for i, (title, detail) in enumerate(recos, 1):
        st.markdown(f"""
        <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;
                    padding:14px;margin:8px 0;display:flex;gap:14px;align-items:flex-start;">
            <div style="background:linear-gradient(135deg,#1e40af,#0f766e);color:white;
                        font-weight:700;font-size:0.75rem;width:28px;height:28px;
                        border-radius:8px;display:flex;align-items:center;
                        justify-content:center;flex-shrink:0;">{i}</div>
            <div>
                <div style="font-weight:700;color:#0D1B3E;">{title}</div>
                <div style="color:#6B7280;font-size:0.85rem;margin-top:3px;">{detail}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "⚠️ Assessment Disclaimer: This tool provides probabilistic risk scoring based on 14,688 real "
        "African business profiles (World Bank Enterprise Surveys, v2 leakage-corrected model, "
        "ROC-AUC 0.8636). It is decision support — not a definitive verdict. Always complement "
        "with independent due diligence."
    )
