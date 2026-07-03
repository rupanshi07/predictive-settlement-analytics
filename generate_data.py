"""
Synthetic securities settlement dataset generator.

Design rationale (this is the part worth explaining in an interview):
Real settlement fails cluster around a handful of known root causes:
  1. SSI (Standing Settlement Instruction) mismatches between counterparties
  2. Counterparties with historically weak operational reliability
  3. Cross-border / cross-currency trades (more hops = more friction)
  4. Short settlement cycles (less time to catch/fix problems)
  5. Illiquid instruments (harder to source securities in time)
  6. High trade value (more scrutiny, more likely to get held up)

Rather than randomly labeling rows, this generator builds fail probability
from a weighted combination of these factors, so a model trained on it is
learning genuine (simulated) causal structure -- and every prediction can be
traced back to a real, explainable driver. This traceability is what the
explanation layer in explain.py depends on.
"""

import numpy as np
import pandas as pd
from faker import Faker
import random

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

N_TRADES = 8000
N_COUNTERPARTIES = 120
N_INSTRUMENTS = 300

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "SGD", "INR", "BRL", "ZAR"]
INSTRUMENT_TYPES = ["Equity", "Corporate Bond", "Government Bond", "ETF", "Money Market", "ADR"]
MARKETS = ["US", "UK", "EU", "JP", "SG", "AU", "BR", "ZA", "IN", "CH"]

# --- Build a counterparty universe with persistent "reliability" traits ---
# In real life, some counterparties are just chronically worse at settlement
# ops than others. We bake that in as a latent trait the model has to learn
# indirectly through historical_fail_rate, not as a direct label leak.
counterparties = []
for i in range(N_COUNTERPARTIES):
    base_reliability = np.random.beta(6, 2)  # most are decent, some are poor
    counterparties.append({
        "counterparty_id": f"CPTY{i:04d}",
        "counterparty_name": fake.company(),
        "base_reliability": base_reliability,
        "region": random.choice(MARKETS),
    })
cpty_df = pd.DataFrame(counterparties)

# --- Build an instrument universe with persistent liquidity traits ---
instruments = []
for i in range(N_INSTRUMENTS):
    instruments.append({
        "instrument_id": f"ISIN{i:06d}",
        "instrument_type": random.choice(INSTRUMENT_TYPES),
        "liquidity_score": np.random.beta(5, 2),  # 0=illiquid, 1=very liquid
    })
instr_df = pd.DataFrame(instruments)


def generate_trade(trade_idx):
    cpty = cpty_df.sample(1).iloc[0]
    instr = instr_df.sample(1).iloc[0]

    trade_date = fake.date_between(start_date="-2y", end_date="today")
    settlement_cycle_days = random.choice([1, 1, 1, 2, 2, 3])  # T+1 dominant, some T+2/T+3
    settlement_date = pd.Timestamp(trade_date) + pd.Timedelta(days=settlement_cycle_days)

    currency = random.choice(CURRENCIES)
    is_cross_border = 1 if cpty["region"] != "US" else 0
    is_cross_currency = 1 if currency != "USD" else 0

    trade_value_usd = round(np.random.lognormal(mean=11, sigma=1.6), 2)  # heavy tail -> some huge trades

    # SSI mismatch is more likely for counterparties we haven't traded with
    # recently / less reliable ones -- modeled via a random flag correlated
    # with reliability.
    ssi_mismatch_prob = 0.35 * (1 - cpty["base_reliability"])
    ssi_mismatch = 1 if np.random.rand() < ssi_mismatch_prob else 0

    # Historical fail rate: noisy observation of the counterparty's latent
    # reliability -- this is what the model actually gets to see (not the
    # true base_reliability), simulating realistic imperfect features.
    historical_fail_rate = np.clip(
        (1 - cpty["base_reliability"]) * 0.6 + np.random.normal(0, 0.05), 0, 1
    )

    days_to_settle = settlement_cycle_days

    # --- True fail probability model (the "ground truth" data-generating process) ---
    # Uses a logistic-linear formulation (log-odds space) rather than a
    # linear-then-clip formulation: linear+clip destroys rank information
    # whenever many trades saturate at the clip boundaries, which produces
    # an artificially low ceiling on achievable AUC regardless of model
    # quality. Logistic-linear is also simply the standard way real risk
    # models (credit risk, fraud, settlement risk) are actually built.
    logit = -5.0  # intercept, calibrated for ~10% base rate
    logit += 3.2 * ssi_mismatch
    logit += 2.4 * (1 - cpty["base_reliability"])
    logit += 1.4 * ssi_mismatch * (1 - cpty["base_reliability"])  # compounding effect
    logit += 0.55 * is_cross_border
    logit += 0.45 * is_cross_currency
    logit += 1.3 * (1 - instr["liquidity_score"])
    logit += 0.35 * (1 if days_to_settle <= 1 else 0)
    logit += 0.7 * (1 if trade_value_usd > 5_000_000 else 0)
    logit += np.random.normal(0, 0.4)  # irreducible randomness (real events are stochastic)
    risk = 1 / (1 + np.exp(-logit))

    failed = 1 if np.random.rand() < risk else 0

    return {
        "trade_id": f"TRD{trade_idx:07d}",
        "trade_date": trade_date,
        "settlement_date": settlement_date.date(),
        "settlement_cycle_days": settlement_cycle_days,
        "counterparty_id": cpty["counterparty_id"],
        "counterparty_name": cpty["counterparty_name"],
        "counterparty_region": cpty["region"],
        "instrument_id": instr["instrument_id"],
        "instrument_type": instr["instrument_type"],
        "instrument_liquidity_score": round(instr["liquidity_score"], 3),
        "currency": currency,
        "is_cross_border": is_cross_border,
        "is_cross_currency": is_cross_currency,
        "trade_value_usd": trade_value_usd,
        "ssi_mismatch_flag": ssi_mismatch,
        "counterparty_historical_fail_rate": round(historical_fail_rate, 3),
        "settlement_failed": failed,
    }


rows = [generate_trade(i) for i in range(N_TRADES)]
df = pd.DataFrame(rows)

print(f"Generated {len(df)} trades")
print(f"Overall fail rate: {df['settlement_failed'].mean():.2%}")
print(df.head())

df.to_csv("trades.csv", index=False)
print("\nSaved to trades.csv")
