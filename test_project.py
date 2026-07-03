"""
Unit tests for the settlement risk model and explanation layer.

Run with: python -m pytest test_project.py -v
(or: python -m unittest test_project.py -v)
"""

import unittest
import numpy as np
import pandas as pd
import joblib
import json
import os

from explain import SettlementRiskExplainer


class TestRiskModel(unittest.TestCase):
    """Sanity checks on the trained model artifact itself."""

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("risk_model.joblib")
        with open("feature_list.json") as f:
            cls.features = json.load(f)

    def test_model_loads(self):
        self.assertIsNotNone(self.model)

    def test_feature_list_matches_model(self):
        # The features the model was trained on should match what we're
        # about to feed it at inference time -- a mismatch here would
        # silently produce garbage predictions.
        self.assertEqual(len(self.features), self.model.n_features_in_)

    def test_predicts_probability_in_valid_range(self):
        # Build one synthetic "safe" trade and one "risky" trade by hand,
        # confirm predicted probabilities land in [0, 1] and behave in the
        # expected direction (more risk factors -> higher score).
        safe_trade = pd.DataFrame([{
            "settlement_cycle_days": 3,
            "instrument_liquidity_score": 0.95,
            "is_cross_border": 0,
            "is_cross_currency": 0,
            "ssi_mismatch_flag": 0,
            "counterparty_historical_fail_rate": 0.02,
            "trade_value_usd_log": np.log1p(10_000),
        }])[self.features]

        risky_trade = pd.DataFrame([{
            "settlement_cycle_days": 1,
            "instrument_liquidity_score": 0.10,
            "is_cross_border": 1,
            "is_cross_currency": 1,
            "ssi_mismatch_flag": 1,
            "counterparty_historical_fail_rate": 0.45,
            "trade_value_usd_log": np.log1p(8_000_000),
        }])[self.features]

        p_safe = self.model.predict_proba(safe_trade)[0, 1]
        p_risky = self.model.predict_proba(risky_trade)[0, 1]

        self.assertGreaterEqual(p_safe, 0.0)
        self.assertLessEqual(p_safe, 1.0)
        self.assertGreaterEqual(p_risky, 0.0)
        self.assertLessEqual(p_risky, 1.0)
        # This is the core sanity check: the model should agree with
        # domain knowledge that an SSI-mismatched, illiquid, cross-border
        # trade is riskier than a clean one.
        self.assertGreater(p_risky, p_safe)


class TestExplainer(unittest.TestCase):
    """Checks on the SHAP-based explanation layer."""

    @classmethod
    def setUpClass(cls):
        cls.explainer = SettlementRiskExplainer()
        df = pd.read_csv("trades.csv")
        df["trade_value_usd_log"] = np.log1p(df["trade_value_usd"])
        cls.sample_row = df.iloc[0]

    def test_explanation_has_required_fields(self):
        result = self.explainer.explain_trade(self.sample_row)
        for key in ["trade_id", "risk_score", "severity", "top_drivers", "narrative", "suggested_actions"]:
            self.assertIn(key, result)

    def test_severity_matches_risk_score(self):
        result = self.explainer.explain_trade(self.sample_row)
        score = result["risk_score"]
        severity = result["severity"]
        if score >= 0.5:
            self.assertEqual(severity, "HIGH")
        elif score >= 0.25:
            self.assertEqual(severity, "MEDIUM")
        else:
            self.assertEqual(severity, "LOW")

    def test_top_drivers_are_sorted_descending(self):
        # SHAP contributions in top_drivers should be in strictly
        # non-increasing order -- this is what makes "primary driver" in
        # the narrative meaningful rather than arbitrary.
        result = self.explainer.explain_trade(self.sample_row)
        contributions = [d["shap_contribution"] for d in result["top_drivers"]]
        self.assertEqual(contributions, sorted(contributions, reverse=True))

    def test_shap_contributions_are_all_positive_in_top_drivers(self):
        # top_drivers is defined as risk-INCREASING factors only (per
        # explain.py's design) -- every entry should have contrib > 0.
        result = self.explainer.explain_trade(self.sample_row)
        for d in result["top_drivers"]:
            self.assertGreater(d["shap_contribution"], 0)

    def test_every_top_driver_has_a_suggested_action(self):
        # Every risk-increasing driver we surface should map to a concrete
        # remediation suggestion -- an explanation with no action isn't
        # useful to an ops analyst.
        result = self.explainer.explain_trade(self.sample_row)
        self.assertEqual(len(result["top_drivers"]), len(result["suggested_actions"]))

    def test_explanation_is_deterministic(self):
        # Same trade, called twice, should give the identical result --
        # this is the whole point of using SHAP over free-form generation.
        result1 = self.explainer.explain_trade(self.sample_row)
        result2 = self.explainer.explain_trade(self.sample_row)
        self.assertEqual(result1["risk_score"], result2["risk_score"])
        self.assertEqual(result1["narrative"], result2["narrative"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
