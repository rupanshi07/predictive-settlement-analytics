"""
Train a gradient-boosted settlement risk model.

Design decisions worth explaining in an interview:
1. XGBoost over deep learning: this is tabular data with a modest number of
   engineered features -- gradient boosted trees are the industry-standard
   choice here (better accuracy AND better native explainability than a
   neural net would give us, for no real cost).
2. scale_pos_weight to handle class imbalance (12.6% positive class) instead
   of naive resampling -- keeps the natural data distribution for evaluation
   while still training an unbiased classifier.
3. SHAP values for explainability -- every single prediction gets decomposed
   into per-feature contributions, which is exactly what explain.py needs to
   generate a traceable, auditable explanation (not just a black-box score).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    average_precision_score, confusion_matrix
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

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Class imbalance ratio (neg/pos): {pos_weight:.2f}")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    eval_metric="aucpr",
    random_state=42,
)
model.fit(X_train, y_train)

# --- Evaluation ---
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
print(f"\nROC-AUC: {auc:.3f}")
print(f"Average Precision (PR-AUC): {ap:.3f}")
print("\nClassification report (threshold=0.5):")
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
        "global_feature_importances": importances,
    }, f, indent=2)

print("\nSaved risk_model.joblib, feature_list.json, metrics.json")
