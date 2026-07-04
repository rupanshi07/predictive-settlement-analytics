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
import datetime
from explain import SettlementRiskExplainer

st.set_page_config(page_title="Settlement Risk Monitor", layout="wide")

# ---------- Visual design system ----------
# Flat, single-page layout: pure black background, white text, thin gray
# borders for structure. No sidebar, no color accents -- severity is
# communicated through fill/weight (a filled white pill vs. an outlined
# one) rather than hue.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #000000; }
.block-container { padding-top: 2rem; }

/* Header block */
.srm-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    letter-spacing: 0.18em; color: #999999; text-transform: uppercase; margin-bottom: 0.25rem;
}
.srm-title { font-size: 2.1rem; font-weight: 700; color: #FFFFFF; margin: 0 0 0.3rem 0; letter-spacing: -0.01em; }
.srm-subtitle { color: #999999; font-size: 0.92rem; max-width: 700px; line-height: 1.5; margin-bottom: 1.4rem; }

/* Metric cards -- symmetric, bordered, no fill */
.metric-card {
    border: 1px solid #333333; border-radius: 6px; padding: 1rem 1.2rem; height: 100%;
}
.metric-label { font-size: 0.85rem; color: #AAAAAA; margin-bottom: 0.5rem; }
.metric-value { font-family: 'IBM Plex Mono', monospace; font-size: 2.1rem; font-weight: 600; color: #FFFFFF; }

/* Severity pills / badges -- fill vs outline distinguishes emphasis */
.pill {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
    font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.28rem 0.7rem; border: 1px solid #FFFFFF;
}
.pill.high { background: #FFFFFF; color: #000000; }
.pill.medium { background: transparent; color: #FFFFFF; }
.pill.low { background: transparent; color: #777777; border-color: #444444; }

.risk-badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.22rem 0.65rem; border: 1px solid #FFFFFF;
}
.risk-badge.high { background: #FFFFFF; color: #000000; }
.risk-badge.medium { background: transparent; color: #FFFFFF; }
.risk-badge.low { background: transparent; color: #777777; border-color: #444444; }

.srm-tradeline { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; color: #FFFFFF; font-weight: 700; margin-bottom: 0.5rem; }

/* Driver bars */
.driver-row { margin-bottom: 0.85rem; }
.driver-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #CCCCCC; margin-bottom: 0.25rem; display: flex; justify-content: space-between; }
.driver-track { background: #1A1A1A; height: 9px; overflow: hidden; }
.driver-fill { background: #FFFFFF; height: 100%; }

/* Filter tags */
span[data-baseweb="tag"] { background-color: #1A1A1A !important; border: 1px solid #FFFFFF !important; }
span[data-baseweb="tag"] span { color: #FFFFFF !important; }

/* Buttons */
.stButton button { background-color: #FFFFFF; color: #000000; border: none; font-weight: 600; border-radius: 4px; }
.stButton button:hover { background-color: #DDDDDD; color: #000000; }

div[data-testid="stDataFrame"] table tbody tr:nth-child(even) { background-color: rgba(255,255,255,0.02); }
hr { border-color: #333333; }
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
    '<div class="srm-subtitle">Explainable early-warning system for settlement fails. Each trade is scored for fail probability, and'
    'every score traces back to specific, ranked drivers with a suggested remediation action.</div>',
    unsafe_allow_html=True,
)

# ---------- Summary metrics: four equal cards ----------
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
metric_defs = [
    ("Trades Monitored", f"{len(df):,}"),
    ("High Severity", f"{(df['severity'] == 'HIGH').sum():,}"),
    ("Medium Severity", f"{(df['severity'] == 'MEDIUM').sum():,}"),
    ("Portfolio Fail Rate", f"{df['risk_score'].mean():.1%}"),
]
for col, (label, value) in zip([mcol1, mcol2, mcol3, mcol4], metric_defs):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

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
    "trade_id": "Trade ID", "counterparty_name": "Counterparty", "counterparty_region": "Region",
    "instrument_type": "Instrument", "currency": "Ccy", "trade_value_usd": "Value (USD)",
    "settlement_date": "Settlement Date", "risk_score": "Fail Probability", "severity": "Severity",
}
display_df = filtered[display_cols].head(200).rename(columns=COLUMN_LABELS).reset_index(drop=True)
st.dataframe(
    display_df.style.format({"Fail Probability": "{:.1%}", "Value (USD)": "${:,.0f}"}),
    use_container_width=True, height=350, hide_index=True,
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
    st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True, hide_index=True)
else:
    st.caption("No trades reviewed yet in this session, select a trade above and click \"Log review\".")
