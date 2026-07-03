"""
Explanation + remediation-suggestion layer.

This is the piece that turns "a model that scores trades" into "a system an
ops analyst could actually act on." Two deliberate design choices:

1. SHAP for the explanation, not an LLM. The RISK DECISION and its
   justification must be deterministic, reproducible, and auditable --
   you cannot have a compliance-relevant number vary because an LLM felt
   like phrasing it differently today. SHAP values are additive and exact:
   they tell you precisely how much each feature pushed a specific trade's
   score up or down, and they're the same every time you ask.

2. Remediation suggestions come from a small deterministic rule base keyed
   off the SAME SHAP-identified top drivers, not a free-form LLM
   generation. This keeps the whole decision chain (score -> explanation ->
   suggested action) traceable end to end, which is exactly the kind of
   auditability real settlement/ops risk tooling needs. An LLM could later
   be layered on top purely to smooth the wording of the narrative --
   never to change the underlying decision -- and that boundary is itself
   worth explaining in an interview as a deliberate trust design choice.
"""

import pandas as pd
import numpy as np
import joblib
import json
import shap

FEATURE_LABELS = {
    "ssi_mismatch_flag": "a Standing Settlement Instruction (SSI) mismatch",
    "counterparty_historical_fail_rate": "the counterparty's historical settlement fail rate",
    "instrument_liquidity_score": "low liquidity in the instrument",
    "settlement_cycle_days": "a short settlement cycle",
    "is_cross_border": "the trade being cross-border",
    "is_cross_currency": "the trade being cross-currency",
    "trade_value_usd_log": "a high trade value",
}

REMEDIATIONS = {
    "ssi_mismatch_flag": "Contact the counterparty's settlements desk to confirm and re-affirm SSIs before the settlement date.",
    "counterparty_historical_fail_rate": "Flag for enhanced monitoring; consider pre-settlement confirmation call given this counterparty's fail history.",
    "instrument_liquidity_score": "Check securities lending / borrow availability early in case the position needs to be sourced.",
    "settlement_cycle_days": "Prioritize same-day instruction matching given the compressed settlement window.",
    "is_cross_border": "Verify local market cut-off times and any local custodian requirements in advance.",
    "is_cross_currency": "Confirm FX funding is arranged ahead of settlement to avoid funding-related delays.",
    "trade_value_usd_log": "Route to senior ops review given the size of the position at risk.",
}


class SettlementRiskExplainer:
    def __init__(self, model_path="risk_model.joblib", feature_list_path="feature_list.json"):
        self.model = joblib.load(model_path)
        with open(feature_list_path) as f:
            self.features = json.load(f)
        self.explainer = shap.TreeExplainer(self.model)

    def explain_trade(self, trade_row: pd.Series, top_k=3):
        X = trade_row[self.features].to_frame().T.astype(float)
        risk_score = float(self.model.predict_proba(X)[0, 1])

        shap_values = self.explainer.shap_values(X)
        contributions = dict(zip(self.features, shap_values[0]))
        # sort by absolute contribution, keep only risk-increasing drivers for the narrative
        sorted_drivers = sorted(contributions.items(), key=lambda kv: -kv[1])
        top_drivers = [d for d in sorted_drivers if d[1] > 0][:top_k]

        driver_sentences = []
        remediation_actions = []
        for feat, contrib in top_drivers:
            label = FEATURE_LABELS.get(feat, feat)
            driver_sentences.append(f"{label} (contributed +{contrib:.3f} to the risk score)")
            remediation_actions.append(REMEDIATIONS.get(feat))

        if risk_score >= 0.5:
            severity = "HIGH"
        elif risk_score >= 0.25:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        narrative = (
            f"Trade {trade_row['trade_id']} flagged at {severity} risk "
            f"(predicted fail probability: {risk_score:.1%}). "
            f"Primary driver(s): {'; '.join(driver_sentences) if driver_sentences else 'no strong individual driver -- elevated baseline risk'}."
        )

        return {
            "trade_id": trade_row["trade_id"],
            "risk_score": round(risk_score, 4),
            "severity": severity,
            "top_drivers": [{"feature": f, "shap_contribution": round(float(c), 4)} for f, c in top_drivers],
            "narrative": narrative,
            "suggested_actions": [a for a in remediation_actions if a],
        }


if __name__ == "__main__":
    df = pd.read_csv("trades.csv")
    df["trade_value_usd_log"] = np.log1p(df["trade_value_usd"])

    explainer = SettlementRiskExplainer()

    # Demo on a handful of trades, sorted by predicted risk, so we see the
    # highest-risk cases first -- this is what the dashboard will show.
    X_all = df[explainer.features].astype(float)
    df["_risk_score"] = explainer.model.predict_proba(X_all)[:, 1]
    sample = df.sort_values("_risk_score", ascending=False).head(5)

    results = [explainer.explain_trade(row) for _, row in sample.iterrows()]
    for r in results:
        print(json.dumps(r, indent=2))
        print("-" * 60)

    with open("sample_explanations.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved sample_explanations.json")
