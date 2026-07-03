🔗 **[Live Demo](https://predictive-settlement-analytics.streamlit.app/)** · [GitHub Repo](https://github.com/rupanshi07/predictive-settlement-analytics)

# Explainable Settlement Risk Monitor

An early-warning system for securities settlement fails that doesn't just
score risk but also explains *why* a trade is flagged, ranks the specific
contributing factors, and suggests a concrete remediation action with
every decision traceable and auditable.

Built to mirror the kind of "predictive trade analytics" (settlement risk
detection + autonomous remediation suggestions) that custody banks are
actively investing in for their operations teams.

## Why this project, and why it's not a tutorial clone

Most ML projects stop at "train a classifier, report accuracy."
This one deliberately adds two layers that most skip, because
they're the layers that matter in a regulated financial context:

1. **A synthetic data generator with a designed causal structure**
   (`generate_data.py`), instead of downloading a Kaggle CSV. The fail
   probability is built from real, named settlement-risk drivers (SSI
   mismatches, counterparty reliability, cross border/currency friction,
   instrument liquidity, settlement cycle length, trade size) combined in
   log odds space, with irreducible random noise so the dataset behaves
   like a genuinely hard, realistic rare event problem (~10% fail rate,
   ROC-AUC ~0.80) instead of an artificially easy one.

2. **A deterministic, auditable explanation layer** (`explain.py`), instead
   of a black-box score. Every prediction is decomposed with SHAP into
   exact per feature contributions, and remediation suggestions are looked
   up from those same top ranked drivers so the full chain (score →
   explanation → suggested action) is reproducible and defensible, which
   is exactly what a compliance-sensitive ops tool needs. An LLM could be
   layered on top purely to smooth the wording later, but the underlying
   *decision* never depends on it that boundary is a deliberate trust
   design choice, not a limitation.

## Architecture

```
generate_data.py   → trades.csv               (synthetic trade/settlement data)
train_model.py      → risk_model.joblib         (XGBoost classifier, class-imbalance aware)
                    → feature_list.json, metrics.json
explain.py           → SettlementRiskExplainer    (SHAP-based per-trade explanation + remediation)
app.py               → Streamlit dashboard        (risk queue, drill-down, audit log)
```

## Key design decisions (talking points)

- **XGBoost over deep learning**: tabular data, modest feature count and
  gradient-boosted trees give better accuracy *and* native, exact
  explainability (via SHAP) for no real cost versus a neural net.
- **Logistic (log-odds) risk formulation** in the data generator rather
  than linear-then-clip: clipping destroys rank information at the
  boundaries and artificially caps achievable model performance.
- **`scale_pos_weight` for class imbalance** (fails are ~10% of trades)
  instead of naive oversampling keeps the natural test distribution for
  honest evaluation.
- **SHAP over LIME or plain feature importances**: SHAP values are
  additive and exact for tree models, so a specific trade's explanation
  is reproducible and mathematically justified, not an approximation.
- **Rule-based remediation, not free-form LLM generation**: keeps the
  suggested action deterministic and traceable back to the specific SHAP
  driver that triggered it.

## Running it

```bash
pip install -r requirements.txt
python3 generate_data.py     # generates trades.csv
python3 train_model.py       # trains and saves the model
python3 explain.py           # sanity-check: prints explanations for top-5 riskiest trades
streamlit run app.py         # launches the dashboard
```

## Results

- ROC-AUC: ~0.80, PR-AUC (average precision): ~0.51 on an ~12% positive
  class base rate which is evaluated on a held-out test set.
- Top global driver: SSI mismatch, consistent with real-world settlement
  ops experience where instruction mismatches are the single largest
  cause of fails.

## Honest limitations (worth stating up front in an interview)

- Data is synthetic, not real trade data the *causal structure* is
  designed to be realistic, but the exact coefficients are illustrative,
  not fitted to real settlement statistics.
- No real-time data feed; this is a batch-scored snapshot, not a
  streaming system.
- The remediation rule base is intentionally small a production version
  would need a much larger, ops-team-validated action library.

## Possible extensions

- Feed the audit log into a simple feedback loop: track which suggested
  actions actually correlated with the trade later settling successfully.
- Add a second model stage that clusters flagged trades by root-cause
  pattern, so ops teams can batch-handle systemic issues (e.g. one
  counterparty having a bad SSI data week) instead of one trade at a time.
- Swap the synthetic generator for a real (anonymized) settlement dataset
  if one becomes available, and re-validate the causal assumptions.
