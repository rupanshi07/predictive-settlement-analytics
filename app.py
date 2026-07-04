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
# Dark, rounded-card SaaS-dashboard style: near-black content area, a
# pure-black sidebar rail, pill-shaped tabs and buttons, and segmented
# "token" progress bars -- kept monochrome (black / white / warm beige)
# so severity/emphasis comes through via fill and weight, not color.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color: #0A0A09; }
.block-container { padding-top: 2rem; }
body, p, span, div, label { color: #EDE7DA; }

section[data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #201E1A; }
section[data-testid="stSidebar"] * { color: #EDE7DA !important; }

/* Sidebar brand */
.srm-brand {
    display: flex; align-items: center; gap: 0.6rem;
    font-weight: 800; color: #F5F0E6 !important; font-size: 1.05rem; margin-bottom: 1.6rem;
}
.srm-brand-mark {
    width: 30px; height: 30px; border-radius: 9px; background: #D9CBB0;
    display: inline-block;
}
.srm-nav-section {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
    letter-spacing: 0.14em; text-transform: uppercase; color: #8A8478 !important;
    margin: 1.2rem 0 0.5rem 0;
}
div[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.5rem 0.7rem; border-radius: 10px; margin-bottom: 0.2rem;
}
div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background-color: rgba(217,203,176,0.14);
}

/* Header block */
.srm-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.2em; color: #A39C8C; text-transform: uppercase; margin-bottom: 0.3rem;
}
.srm-title { font-size: 2.4rem; font-weight: 800; color: #F5F0E6; margin: 0 0 0.3rem 0; letter-spacing: -0.02em; }
.srm-subtitle { color: #9C9585; font-size: 0.95rem; max-width: 700px; line-height: 1.5; margin-bottom: 1.4rem; }

/* Rounded card base */
.card {
    background: #141311; border: 1px solid #262319; border-radius: 20px; padding: 1.3rem 1.5rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.3); height: 100%;
}
.card-dark { background: #E9DFCB; color: #12100C; border: 1px solid #E9DFCB; }

/* Hero + mini stats */
.hero-stat-label { font-size: 0.82rem; font-weight: 600; color: #9C9585; margin-bottom: 0.7rem; }
.hero-stat-value-row { display: flex; align-items: baseline; gap: 0.6rem; }
.hero-stat-value { font-size: 2.7rem; font-weight: 800; color: #F5F0E6; line-height: 1; }
.hero-stat-context { color: #7C7668; font-size: 0.78rem; margin-top: 0.6rem; }

/* Segmented token progress bar */
.token-row { display: flex; gap: 5px; margin-top: 0.9rem; }
.token { width: 22px; height: 14px; border-radius: 4px; background: #262319; }
.token.filled { background: #D9CBB0; }

.mini-stat-label { font-size: 0.78rem; font-weight: 600; color: #9C9585; margin-bottom: 0.5rem; }
.mini-stat-value { font-size: 1.6rem; font-weight: 800; color: #F5F0E6; }

/* Insight callout -- beige inverted card, mirrors reference's promo tile */
.insight-card { margin-bottom: 1.2rem; }
.insight-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase; color: #6B6555; margin-bottom: 0.5rem; }
.insight-text { color: #16140F; font-size: 0.95rem; line-height: 1.55; font-weight: 500; }

/* Pill tabs */
.pill-tag {
    display: inline-block; font-size: 0.72rem; font-weight: 700;
    padding: 0.35rem 0.85rem; border-radius: 999px; background: #D9CBB0; color: #12100C;
    margin-bottom: 0.9rem;
}

/* Trade row cards */
.trade-card {
    display: flex; align-items: center; justify-content: space-between;
    background: #141311; border: 1px solid #262319; border-radius: 16px; box-shadow: 0 1px 2px rgba(0,0,0,0.3);
    padding: 0.9rem 1.2rem; margin-bottom: 0.6rem;
}
.trade-card-left { display: flex; flex-direction: column; gap: 0.15rem; }
.trade-card-id { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #7C7668; }
.trade-card-name { font-weight: 700; color: #F5F0E6; font-size: 0.95rem; }
.trade-card-sub { font-size: 0.78rem; color: #8A8478; }
.trade-card-right { display: flex; align-items: center; gap: 1.1rem; }
.trade-card-prob { font-family: 'IBM Plex Mono', monospace; font-weight: 700; color: #F5F0E6; text-align: right; }

/* Severity pills -- fill/weight distinguishes emphasis, no color */
.pill {
    display: inline-block; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.03em; text-transform: uppercase;
    padding: 0.32rem 0.8rem; border-radius: 999px;
}
.pill.high { background: #D9CBB0; color: #12100C; }
.pill.medium { background: #2A271F; color: #D9CBB0; }
.pill.low { background: #1A1814; color: #7C7668; }

.risk-badge {
    display: inline-block; font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.03em; text-transform: uppercase; padding: 0.3rem 0.8rem; border-radius: 999px;
}
.risk-badge.high { background: #D9CBB0; color: #12100C; }
.risk-badge.medium { background: #2A271F; color: #D9CBB0; }
.risk-badge.low { background: #1A1814; color: #7C7668; }

.srm-tradeline { font-size: 1.4rem; color: #F5F0E6; font-weight: 800; margin-bottom: 0.5rem; }

/* Driver bars -- capsule style, echoing the reference's rounded bar chart */
.driver-row { margin-bottom: 0.9rem; }
.driver-label { font-size: 0.82rem; font-weight: 600; color: #C7C0B0; margin-bottom: 0.35rem; display: flex; justify-content: space-between; }
.driver-track { background: #262319; height: 10px; border-radius: 999px; overflow: hidden; }
.driver-fill { background: #D9CBB0; height: 100%; border-radius: 999px; }

/* Filter tags */
span[data-baseweb="tag"] { background-color: #D9CBB0 !important; border-radius: 999px !important; }
span[data-baseweb="tag"] span { color: #12100C !important; }

/* Buttons -- solid beige pill, matches reference's "Create a New Scenario" button */
.stButton button { background-color: #D9CBB0; color: #12100C; border: none; font-weight: 700; border-radius: 999px; padding: 0.5rem 1.3rem; }
.stButton button:hover { background-color: #EAE0CB; color: #12100C; }

/* Inputs */
div[data-testid="stTextInput"] input, div[data-baseweb="select"] > div {
    background-color: #141311 !important; color: #EDE7DA !important; border: 1px solid #262319 !important;
}
div[data-testid="stSidebar"] div[data-baseweb="select"] > div { background-color: #141311 !important; }

div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; border: 1px solid #262319; }
hr { border-color: #262319; }
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


def token_row_html(filled: int, total: int = 8) -> str:
    tokens = "".join(
        f'<div class="token {"filled" if i < filled else ""}"></div>' for i in range(total)
    )
    return f'<div class="token-row">{tokens}</div>'


def trade_card_html(row) -> str:
    return f"""
    <div class="trade-card">
        <div class="trade-card-left">
            <div class="trade-card-name">{row['counterparty_name']}</div>
            <div class="trade-card-sub">{row['trade_id']} · {row['instrument_type']} · {row['currency']} · {row['counterparty_region']}</div>
        </div>
        <div class="trade-card-right">
            <div class="trade-card-prob">${row['trade_value_usd']:,.0f}<div class="trade-card-id">{row['settlement_date']}</div></div>
            <div style="min-width:60px; text-align:right; font-family:'IBM Plex Mono',monospace; color:#F5F0E6; font-weight:700;">{row['risk_score']:.1%}</div>
            <span class="pill {row['severity'].lower()}">{row['severity']}</span>
        </div>
    </div>
    """


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

# ---------- Sidebar: brand, nav, filters ----------
with st.sidebar:
    st.markdown(
        '<div class="srm-brand"><span class="srm-brand-mark"></span> SETTLEMENT RISK</div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigate", ["Overview", "Audit Log"], label_visibility="collapsed")

    st.markdown('<div class="srm-nav-section">Filters</div>', unsafe_allow_html=True)
    severity_filter = st.multiselect("Severity", ["HIGH", "MEDIUM", "LOW"], default=["HIGH", "MEDIUM"])
    region_filter = st.multiselect("Counterparty region", sorted(df["counterparty_region"].unique().tolist()))
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

# ---------- Header ----------
st.markdown('<div class="srm-eyebrow">Settlement Operations · Risk Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="srm-title">Settlement Risk Monitor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="srm-subtitle">Explainable early-warning system for settlement fails — '
    'every score traces back to specific, ranked drivers with a suggested remediation action.</div>',
    unsafe_allow_html=True,
)

if page == "Overview":
    # ---------- Summary metrics: rounded cards with token progress bars ----------
    mcol1, mcol2, mcol3 = st.columns([1.3, 1, 1])

    with mcol1:
        filled_tokens = round(df['risk_score'].mean() * 8 / 0.3)  # visual scale, capped below
        filled_tokens = max(1, min(8, filled_tokens))
        st.markdown(f"""
        <div class="card">
            <div class="hero-stat-label">Portfolio Fail Rate</div>
            <div class="hero-stat-value-row"><div class="hero-stat-value">{df['risk_score'].mean():.1%}</div></div>
            {token_row_html(filled_tokens)}
            <div class="hero-stat-context">Average predicted fail probability across {len(df):,} monitored trades</div>
        </div>
        """, unsafe_allow_html=True)

    with mcol2:
        n_high = (df['severity'] == 'HIGH').sum()
        n_med = (df['severity'] == 'MEDIUM').sum()
        st.markdown(f"""
        <div class="card" style="margin-bottom:1rem;">
            <div class="mini-stat-label">High Severity</div>
            <div class="mini-stat-value">{n_high:,}</div>
            {token_row_html(round(n_high / len(df) * 8 / 0.15), total=8)}
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div class="mini-stat-label">Medium Severity</div>
            <div class="mini-stat-value">{n_med:,}</div>
            {token_row_html(round(n_med / len(df) * 8 / 0.4), total=8)}
        </div>
        """, unsafe_allow_html=True)

    with mcol3:
        high_value_at_risk = df.loc[df["severity"] == "HIGH", "trade_value_usd"].sum()
        ssi_share = df["ssi_mismatch_flag"].mean()
        st.markdown(f"""
        <div class="card" style="margin-bottom:1rem;">
            <div class="mini-stat-label">Value at Risk (High Sev.)</div>
            <div class="mini-stat-value">${high_value_at_risk/1e6:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
            <div class="mini-stat-label">Trades w/ SSI Mismatch</div>
            <div class="mini-stat-value">{ssi_share:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Insight callout: beige inverted card, mirroring the reference's promo tile ----------
    high_df = df[df["severity"] == "HIGH"]
    ssi_share_high = high_df["ssi_mismatch_flag"].mean() if len(high_df) else 0
    st.markdown(f"""
    <div class="card card-dark insight-card">
        <div class="insight-label">Risk Insight</div>
        <div class="insight-text">SSI mismatches are the dominant driver of high-severity trades — present in
        {ssi_share_high:.0%} of trades currently flagged HIGH risk. Prioritizing SSI re-affirmation workflows
        would address the largest single source of predicted fails.</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # ---------- Top at-risk trades: card list ----------
    st.markdown('<span class="pill-tag">TOP AT-RISK TRADES</span>', unsafe_allow_html=True)
    top_n = filtered.head(6)
    if top_n.empty:
        st.info("No trades match the current filters.")
    else:
        cards_html = "".join(trade_card_html(row) for _, row in top_n.iterrows())
        st.markdown(cards_html, unsafe_allow_html=True)

    st.write("")

    # ---------- Full trade table ----------
    st.markdown(f'<span class="pill-tag">FULL TRADE QUEUE ({len(filtered)})</span>', unsafe_allow_html=True)
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

    st.write("")

    # ---------- Drill-down ----------
    st.markdown('<span class="pill-tag">INVESTIGATE A TRADE</span>', unsafe_allow_html=True)
    trade_options = filtered["trade_id"].head(200).tolist()
    if trade_options:
        selected_trade = st.selectbox("Select a trade to explain", trade_options, label_visibility="collapsed")

        if selected_trade:
            row = df[df["trade_id"] == selected_trade].iloc[0]
            explanation = explainer.explain_trade(row)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
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
                st.markdown('</div>', unsafe_allow_html=True)

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

else:  # Audit Log page
    st.markdown('<span class="pill-tag">AUDIT LOG (THIS SESSION)</span>', unsafe_allow_html=True)
    if st.session_state.audit_log:
        st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True, hide_index=True)
    else:
        st.caption("No trades reviewed yet in this session — go to Overview, select a trade, and click \"Log review\".")