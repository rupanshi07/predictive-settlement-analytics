"""
Train a gradient-boosted settlement risk model.

Design decisions worth explaining in an interview:
1. XGBoost over deep learning: this is tabular data with a modest number of
   engineered features -- gradient boosted trees are the industry-standard
   choice here (better accuracy AND better native explainability than a
   neural net would give us, for no real cost).
2. Class imbalance (12% positive class) is handled via THRESHOLD TUNING,
   not scale_pos_weight. scale_pos_weight reweights the training loss,
   which improves recall at a fixed 0.5 cutoff but systematically inflates
   the raw predicted probabilities above their true calibrated values --
   e.g. an average predicted fail probability of 30%+ on a dataset with a
   true 12% fail rate. That mismatch is exactly the kind of thing an
   auditor (or interviewer) would immediately flag as broken. Since this
   system reports the probability itself to an analyst (not just a
   yes/no), the probabilities need to mean what they say. So: train
   without reweighting to keep calibration honest, and instead choose the
   HIGH/MEDIUM/LOW severity cutoffs based on the precision-recall curve.
3. SHAP values for explainability -- every single prediction gets decomposed
   into per-feature contributions, which is exactly what explain.py needs to
   generate a traceable, auditable explanation (not just a black-box score).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    average_precision_score, confusion_matrix, f1_score
)
import xgboost as xgb
import joblib
import json

df = pd.read_csv("trades.csv")

FEATURES = [
    "settlement_cycle_days",
    "instrument_liquidity_score",
    "is_cross_border",
    "is_cross_currency",
    "trade_value_usd",
    "ssi_mismatch_flag",
    "counterparty_historical_fail_rate",
]
TARGET = "settlement_failed"

# log-transform trade value: heavy-tailed, model learns better on log scale
df["trade_value_usd_log"] = np.log1p(df["trade_value_usd"])
FEATURES = [f for f in FEATURES if f != "trade_value_usd"] + ["trade_value_usd_log"]

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training base rate: {y_train.mean():.2%}  |  Test base rate: {y_test.mean():.2%}")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
)
model.fit(X_train, y_train)

# --- Evaluation ---
y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
print(f"\nROC-AUC: {auc:.3f}")
print(f"Average Precision (PR-AUC): {ap:.3f}")
print(f"Mean predicted probability: {y_prob.mean():.2%}  (should track the true base rate above -- this is the calibration sanity check)")

# --- Threshold tuning: pick the cutoff that maximizes F1 on the PR curve,
# rather than defaulting to 0.5 (which is a poor choice on imbalanced data
# with honestly-calibrated probabilities -- almost everything would be
# predicted negative at 0.5 since the true rate is only ~12%). ---
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = float(thresholds[best_idx])
print(f"\nBest F1-optimal decision threshold: {best_threshold:.3f}")

y_pred = (y_prob >= best_threshold).astype(int)
print(f"\nClassification report (threshold={best_threshold:.3f}):")
print(classification_report(y_test, y_pred, digits=3))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# --- Feature importance (global) ---
importances = dict(zip(FEATURES, model.feature_importances_.tolist()))
importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
print("\nGlobal feature importances:")
for f, v in importances.items():
    print(f"  {f}: {v:.3f}")

# --- Save model + metadata ---
joblib.dump(model, "risk_model.joblib")
with open("feature_list.json", "w") as f:
    json.dump(FEATURES, f)
with open("metrics.json", "w") as f:
    json.dump({
        "roc_auc": auc,
        "average_precision": ap,
        "mean_predicted_probability": float(y_prob.mean()),
        "test_base_rate": float(y_test.mean()),
        "decision_threshold": best_threshold,
        "global_feature_importances": importances,
    }, f, indent=2)

print("\nSaved risk_model.joblib, feature_list.json, metrics.json")
