"""
Settlement Risk Ops Dashboard.

An analyst-facing view: a queue of at-risk trades sorted by predicted fail
probability, with a drill-down into WHY each trade was flagged and WHAT to
do about it. Also maintains a lightweight audit log -- every trade that gets
viewed/actioned is timestamped, because "what did the system decide and
when" is exactly the kind of trail real settlement ops tooling needs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from explain import SettlementRiskExplainer

st.set_page_config(page_title="Settlement Risk Monitor", layout="wide")

# ---------- Visual design system ----------
# Deliberately not the default Streamlit look: a dark navy / warm-gold
# palette (nodding to the kind of institutional custody-bank UI this is
# modeled on), condensed data-grade typography, and text-based severity
# badges instead of colored emoji circles.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background-color: #0A1420; }

/* Eyebrow + title block */
.srm-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: #C9A24B;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.srm-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #EDF1F7;
    margin: 0 0 0.35rem 0;
    letter-spacing: -0.01em;
}
.srm-subtitle {
    color: #8A97AC;
    font-size: 0.95rem;
    max-width: 720px;
    line-height: 1.5;
    margin-bottom: 1.4rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #101C2E;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 6px;
    padding: 0.9rem 1rem 0.7rem 1rem;
}
div[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8A97AC;
}
div[data-testid="stMetricValue"] {
    color: #EDF1F7;
    font-family: 'IBM Plex Mono', monospace;
}

/* Severity badges -- text-based, not emoji */
.risk-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.18rem 0.55rem;
    border-radius: 3px;
    border: 1px solid;
}
.risk-badge.high { color: #E2685C; border-color: #E2685C; background: rgba(226,104,92,0.08); }
.risk-badge.medium { color: #C9A24B; border-color: #C9A24B; background: rgba(201,162,75,0.08); }
.risk-badge.low { color: #5FA37E; border-color: #5FA37E; background: rgba(95,163,126,0.08); }

.srm-tradeline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.3rem;
    color: #EDF1F7;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

/* Custom driver bars, replacing the default st.bar_chart look */
.driver-row { margin-bottom: 0.85rem; }
.driver-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #C7CEDA;
    margin-bottom: 0.25rem;
    display: flex;
    justify-content: space-between;
}
.driver-track {
    background: rgba(255,255,255,0.05);
    border-radius: 3px;
    height: 10px;
    overflow: hidden;
}
.driver-fill {
    background: linear-gradient(90deg, #C9A24B, #E2685C);
    height: 100%;
    border-radius: 3px;
}

/* Filter tags -- override Streamlit's default pink multiselect chips */
span[data-baseweb="tag"] {
    background-color: #1B2A40 !important;
    border: 1px solid #C9A24B !important;
}
span[data-baseweb="tag"] span { color: #EDF1F7 !important; }

/* Highlight the primary metric card */
div[data-testid="stMetric"]:first-of-type {
    border-color: rgba(201,162,75,0.35);
}

hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)


FEATURE_DISPLAY_NAMES = {
    "ssi_mismatch_flag": "SSI Mismatch",
    "counterparty_historical_fail_rate": "Counterparty Fail History",
    "instrument_liquidity_score": "Instrument Liquidity",
    "settlement_cycle_days": "Settlement Cycle Length",
    "is_cross_border": "Cross-Border",
    "is_cross_currency": "Cross-Currency",
    "trade_value_usd_log": "Trade Value",
}


def render_driver_bars(drivers_df: pd.DataFrame):
    if drivers_df.empty:
        st.caption("No strong individual driver — elevated baseline risk.")
        return
    max_val = drivers_df["shap_contribution"].max()
    rows_html = ""
    for _, r in drivers_df.iterrows():
        label = FEATURE_DISPLAY_NAMES.get(r["feature"], r["feature"])
        pct = max(4, (r["shap_contribution"] / max_val) * 100)
        rows_html += f"""
        <div class="driver-row">
            <div class="driver-label"><span>{label}</span><span>+{r['shap_contribution']:.3f}</span></div>
            <div class="driver-track"><div class="driver-fill" style="width:{pct:.0f}%"></div></div>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)


def risk_badge_html(severity: str) -> str:
    return f'<span class="risk-badge {severity.lower()}">{severity} risk</span>'


# ---------- Data / model loading (cached) ----------
@st.cache_resource
def load_explainer():
    return SettlementRiskExplainer()

@st.cache_data
def load_data():
    df = pd.read_csv("trades.csv")
    df["trade_value_usd_log"] = np.log1p(df["trade_value_usd"])
    return df

explainer = load_explainer()
df = load_data()

X_all = df[explainer.features].astype(float)
df["risk_score"] = explainer.model.predict_proba(X_all)[:, 1]
df["severity"] = pd.cut(df["risk_score"], bins=[-0.01, 0.25, 0.5, 1.01], labels=["LOW", "MEDIUM", "HIGH"])

if "audit_log" not in st.session_state:
    st.session_state.audit_log = []

# ---------- Header ----------
st.markdown('<div class="srm-eyebrow">Settlement Operations · Risk Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="srm-title">Settlement Risk Monitor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="srm-subtitle">Explainable early-warning system for settlement fails — '
    'every score traces back to specific, ranked drivers with a suggested remediation action.</div>',
    unsafe_allow_html=True,
)

# ---------- Summary metrics ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Trades Monitored", f"{len(df):,}")
col2.metric("High Severity", f"{(df['severity'] == 'HIGH').sum():,}")
col3.metric("Medium Severity", f"{(df['severity'] == 'MEDIUM').sum():,}")
col4.metric("Portfolio Fail Rate", f"{df['risk_score'].mean():.1%}")

st.divider()

# ---------- Filters ----------
fcol1, fcol2, fcol3 = st.columns([1, 1, 2])
with fcol1:
    severity_filter = st.multiselect("Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
with fcol2:
    region_filter = st.multiselect("Counterparty region", sorted(df["counterparty_region"].unique().tolist()))
with fcol3:
    search = st.text_input("Search trade ID or counterparty")

filtered = df[df["severity"].isin(severity_filter)] if severity_filter else df
if region_filter:
    filtered = filtered[filtered["counterparty_region"].isin(region_filter)]
if search:
    mask = (
        filtered["trade_id"].str.contains(search, case=False, na=False)
        | filtered["counterparty_name"].str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

filtered = filtered.sort_values("risk_score", ascending=False)

st.subheader(f"At-risk trade queue ({len(filtered)} trades)")

display_cols = [
    "trade_id", "counterparty_name", "counterparty_region", "instrument_type",
    "currency", "trade_value_usd", "settlement_date", "risk_score", "severity",
]
COLUMN_LABELS = {
    "trade_id": "Trade ID",
    "counterparty_name": "Counterparty",
    "counterparty_region": "Region",
    "instrument_type": "Instrument",
    "currency": "Ccy",
    "trade_value_usd": "Value (USD)",
    "settlement_date": "Settlement Date",
    "risk_score": "Fail Probability",
    "severity": "Severity",
}
display_df = filtered[display_cols].head(200).rename(columns=COLUMN_LABELS)
st.dataframe(
    display_df.style.format({"Fail Probability": "{:.1%}", "Value (USD)": "${:,.0f}"}),
    use_container_width=True,
    height=350,
)

st.divider()

# ---------- Drill-down ----------
st.subheader("Investigate a trade")
trade_options = filtered["trade_id"].head(200).tolist()
if trade_options:
    selected_trade = st.selectbox("Select a trade to explain", trade_options)

    if selected_trade:
        row = df[df["trade_id"] == selected_trade].iloc[0]
        explanation = explainer.explain_trade(row)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f'<div class="srm-tradeline">{selected_trade}</div>', unsafe_allow_html=True)
            st.markdown(risk_badge_html(explanation["severity"]), unsafe_allow_html=True)
            st.write("")
            st.write(explanation["narrative"])

            st.markdown("**Top contributing factors**")
            drivers_df = pd.DataFrame(explanation["top_drivers"])
            render_driver_bars(drivers_df)

            st.markdown("**Suggested remediation actions**")
            for action in explanation["suggested_actions"]:
                st.checkbox(action, key=f"{selected_trade}_{action[:20]}")

        with c2:
            st.markdown("**Trade details**")
            st.json({
                "counterparty": row["counterparty_name"],
                "region": row["counterparty_region"],
                "instrument": row["instrument_type"],
                "currency": row["currency"],
                "value_usd": row["trade_value_usd"],
                "settlement_date": str(row["settlement_date"]),
                "ssi_mismatch": bool(row["ssi_mismatch_flag"]),
            })

        if st.button("Log review of this trade"):
            st.session_state.audit_log.append({
                "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                "trade_id": selected_trade,
                "risk_score": explanation["risk_score"],
                "severity": explanation["severity"],
                "top_driver": explanation["top_drivers"][0]["feature"] if explanation["top_drivers"] else None,
            })
            st.success("Logged.")
else:
    st.info("No trades match the current filters.")

st.divider()

# ---------- Audit log ----------
st.subheader("Audit log (this session)")
if st.session_state.audit_log:
    st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True)
else:
    st.caption("No trades reviewed yet in this session — select a trade above and click \"Log review\".")
