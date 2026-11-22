"""
PharmaPulse - Expiry Intelligence Predictor & Risk Quantifier
=============================================================
Provides production inference for batch-level expiration risk and financial exposure.

Data Source:
  - Uses ACTIVE batches from data/raw/batches.csv (real expiry_date → real days_to_expiry)
  - Uses data/raw/daily_inventory.csv for demand-based features (last 30d)
  - This gives a meaningful days_to_expiry range (19–149 days) so the
    Expiry Horizon slider on the dashboard produces visible, dynamic results.

Dual-Task ML Inference:
  Task A: XGBoost Classifier  → probability of expiring with unsold units
  Task B: Gradient Boosting   → predicted quantity of unsold units at expiry

Physical Domain Constraints Enforced:
  1. Non-negative: predicted_unsold >= 0
  2. Stock ceiling: predicted_unsold <= remaining_batch_quantity
  3. Velocity clearance: if days_left > coverage * 1.3 and prob < 0.20 → predicted_unsold = 0
  4. Imminent expiry: if days_left <= 7 and prob > 0.40 → predicted_unsold >= 80% of stock

Risk Tiers (4-tier system):
  Critical: >= 25% batch loss OR (days_left <= 7 AND unsold_pct >= 15%)
  High:     12% – 25% projected loss
  Medium:   3%  – 12% projected loss
  Low:      < 3% projected loss OR no unsold units
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR  = os.path.join(PROJECT_ROOT, "models", "expiry")
DATA_DIR    = os.path.join(PROJECT_ROOT, "data")
RAW_DIR     = os.path.join(DATA_DIR, "raw")

# Simulation reference date (last date in daily_inventory.csv)
SIM_END_DATE = pd.Timestamp("2026-12-30")

FEATURE_COLS = [
    "days_to_expiry",
    "initial_quantity",
    "remaining_batch_quantity",
    "batch_age_days",
    "historical_medicine_demand_7d",
    "historical_medicine_demand_14d",
    "historical_medicine_demand_30d",
    "historical_sales_7d",
    "historical_sales_14d",
    "historical_sales_30d",
    "historical_daily_demand",
    "stock_coverage_estimate",
]


# ---------------------------------------------------------------------------
# Risk / Action Helpers
# ---------------------------------------------------------------------------

def map_expiry_risk_tier(unsold_pct: float, days_left: int, predicted_unsold: float) -> str:
    """
    Classifies a batch into one of 4 standardised risk tiers.
      Critical : >= 25% batch loss  OR (days_left <= 7 AND unsold_pct >= 15%)
      High     : 12% – 25% projected loss
      Medium   : 3%  – 12% projected loss
      Low      : < 3% OR zero predicted unsold
    """
    if predicted_unsold <= 0 or unsold_pct < 3.0:
        return "Low"
    if unsold_pct >= 25.0 or (days_left <= 7 and unsold_pct >= 15.0):
        return "Critical"
    if unsold_pct >= 12.0:
        return "High"
    return "Medium"


def map_mitigation_action(days_left: int, risk_tier: str) -> str:
    """Returns the operational mitigation recommendation for a batch."""
    if risk_tier == "Low":
        return "Standard FEFO Dispensing (Healthy Velocity)"
    if days_left > 90:
        return "Tier 1: Supplier Return / Swap (Contract Window)"
    if days_left >= 30:
        return "Tier 2: 25% Clearance Markdown / Fast-Track FEFO"
    return "Tier 3: Inter-Branch Reallocation / Immediate Clinical Swap"


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _build_active_batch_features() -> pd.DataFrame:
    """
    Constructs the ML feature matrix for all ACTIVE pharmacy batches using
    real expiry dates and live demand signals from daily_inventory.csv.

    Returns a DataFrame with FEATURE_COLS columns plus helper identifiers.
    """
    # --- Load raw data -------------------------------------------------------
    df_batches = pd.read_csv(os.path.join(RAW_DIR, "batches.csv"))
    df_inv     = pd.read_csv(os.path.join(RAW_DIR, "daily_inventory.csv"))
    df_inv["date"] = pd.to_datetime(df_inv["date"])

    # --- ACTIVE batches only -------------------------------------------------
    df_active = df_batches[df_batches["batch_status"] == "ACTIVE"].copy()
    df_active["expiry_date_dt"]   = pd.to_datetime(df_active["expiry_date"])
    df_active["received_date_dt"] = pd.to_datetime(df_active["received_date"])

    # Real days to expiry relative to simulation end date
    df_active["days_to_expiry"] = (df_active["expiry_date_dt"] - SIM_END_DATE).dt.days

    # Keep only batches that have not yet expired
    df_active = df_active[df_active["days_to_expiry"] > 0].reset_index(drop=True)

    # Batch age = days since it was received
    df_active["batch_age_days"] = (SIM_END_DATE - df_active["received_date_dt"]).dt.days.clip(lower=0)

    # remaining_batch_quantity ≈ initial_quantity for ACTIVE batches
    # (we don't have per-batch real-time counts, so use initial as upper bound)
    df_active["remaining_batch_quantity"] = df_active["initial_quantity"]

    # --- Demand features from daily_inventory (last N days) ------------------
    last_date = df_inv["date"].max()

    def demand_window(n_days: int) -> pd.Series:
        cutoff = last_date - pd.Timedelta(days=n_days)
        return (
            df_inv[df_inv["date"] >= cutoff]
            .groupby("medicine_id")["units_sold"]
            .sum()
        )

    dem_7d  = demand_window(7).rename("historical_medicine_demand_7d")
    dem_14d = demand_window(14).rename("historical_medicine_demand_14d")
    dem_30d = demand_window(30).rename("historical_medicine_demand_30d")
    dem_avg = (
        df_inv[df_inv["date"] >= last_date - pd.Timedelta(days=30)]
        .groupby("medicine_id")["units_sold"]
        .mean()
        .rename("historical_daily_demand")
    )

    # Join demand signals onto batch DataFrame
    df_active = df_active.join(dem_7d,  on="medicine_id")
    df_active = df_active.join(dem_14d, on="medicine_id")
    df_active = df_active.join(dem_30d, on="medicine_id")
    df_active = df_active.join(dem_avg, on="medicine_id")

    # Fill any missing demand (new medicines with no inventory history)
    for col in ["historical_medicine_demand_7d", "historical_medicine_demand_14d",
                "historical_medicine_demand_30d", "historical_daily_demand"]:
        df_active[col] = df_active[col].fillna(0.0)

    # historical_sales_* mirrors medicine demand (batch-level sales ≈ medicine demand)
    df_active["historical_sales_7d"]  = df_active["historical_medicine_demand_7d"]
    df_active["historical_sales_14d"] = df_active["historical_medicine_demand_14d"]
    df_active["historical_sales_30d"] = df_active["historical_medicine_demand_30d"]

    # stock_coverage_estimate: how many days the current stock will last at avg demand
    df_active["stock_coverage_estimate"] = (
        df_active["remaining_batch_quantity"]
        / df_active["historical_daily_demand"].replace(0, 1)
    ).clip(upper=365)

    return df_active


# ---------------------------------------------------------------------------
# Main Inference Function
# ---------------------------------------------------------------------------

def predict_batch_expiry(df_input: pd.DataFrame = None) -> pd.DataFrame:
    """
    Runs dual-task ML inference across all ACTIVE pharmacy batches.

    Returns a DataFrame with per-batch predictions:
      - days_to_expiry           (real, from batches.csv expiry_date)
      - predicted_unsold_units   (physically constrained)
      - unsold_probability       (classifier confidence)
      - financial_exposure       (PKR at cost price)
      - risk_level               (Critical / High / Medium / Low)
      - recommended_action       (3-tier mitigation playbook)
      - medicine_name, category_id, category_name
    """
    # Load ML models
    clf_path  = os.path.join(MODELS_DIR, "xgboost_expiry_clf.joblib")
    reg_path  = os.path.join(MODELS_DIR, "gradient_boosting_reg.joblib")
    feat_path = os.path.join(MODELS_DIR, "feature_cols.joblib")

    if not (os.path.exists(clf_path) and os.path.exists(reg_path)):
        from modules.expiry.classification import train_expiry_classification
        from modules.expiry.regression     import train_expiry_regression
        train_expiry_classification(PROJECT_ROOT)
        train_expiry_regression(PROJECT_ROOT)

    clf          = joblib.load(clf_path)
    reg          = joblib.load(reg_path)
    feature_cols = joblib.load(feat_path) if os.path.exists(feat_path) else FEATURE_COLS

    # Build feature matrix from live ACTIVE batch data
    if df_input is None:
        df = _build_active_batch_features()
    else:
        df = df_input.copy()

    # --- Enrich with medicine economics & taxonomy ---------------------------
    df_meds = pd.read_csv(os.path.join(RAW_DIR, "medicines.csv"))
    df_cats = pd.read_csv(os.path.join(RAW_DIR, "categories.csv"))

    med_cost_dict  = df_meds.set_index("medicine_id")["unit_cost"].to_dict()
    med_price_dict = df_meds.set_index("medicine_id")["selling_price"].to_dict()
    med_name_dict  = df_meds.set_index("medicine_id")["medicine_name"].to_dict()
    med_cat_dict   = df_meds.set_index("medicine_id")["category_id"].to_dict()
    cat_name_dict  = df_cats.set_index("category_id")["category_name"].to_dict()

    df["medicine_name"] = df["medicine_id"].map(med_name_dict).fillna(df["medicine_id"])
    df["category_id"]   = df["medicine_id"].map(med_cat_dict).fillna("CAT001")
    df["category_name"] = df["category_id"].map(cat_name_dict).fillna("General")
    df["unit_cost"]     = df["medicine_id"].map(med_cost_dict).fillna(20.0).astype(float)
    df["selling_price"] = df["medicine_id"].map(med_price_dict).fillna(30.0).astype(float)

    # --- ML Inference --------------------------------------------------------
    X          = df[feature_cols].copy()
    pred_prob  = clf.predict_proba(X)[:, 1]
    pred_raw   = reg.predict(X)

    # --- Apply physical domain constraints -----------------------------------
    predicted_unsold = []
    for idx, (_, row) in enumerate(df.iterrows()):
        rem_qty   = float(row["remaining_batch_quantity"])
        days_left = float(row["days_to_expiry"])
        coverage  = float(row.get("stock_coverage_estimate", 30.0))
        prob      = float(pred_prob[idx])
        raw_qty   = float(pred_raw[idx])

        qty = max(0.0, raw_qty)              # Rule 1: non-negative
        qty = min(rem_qty, qty)              # Rule 2: stock ceiling

        # Rule 3: Velocity clearance — stock will sell out well before expiry
        if days_left > coverage * 1.3 and prob < 0.20:
            qty = 0.0

        # Rule 4: Imminent expiry (<= 7 days) with meaningful remaining stock
        if days_left <= 7 and rem_qty > 0 and prob > 0.40:
            qty = max(qty, rem_qty * 0.80)

        predicted_unsold.append(round(qty, 1))

    df = df.reset_index(drop=True)
    df["predicted_unsold_units"] = predicted_unsold
    df["unsold_probability"]     = np.round(pred_prob, 3)
    df["financial_exposure"]     = (df["predicted_unsold_units"] * df["unit_cost"]).round(2)

    # Unsold percentage relative to initial batch size
    init_q        = df["initial_quantity"].replace(0, 1)
    df["unsold_pct"] = (df["predicted_unsold_units"] / init_q * 100.0).round(2)

    # --- Risk tier & mitigation mapping --------------------------------------
    risk_levels = []
    actions     = []
    for _, row in df.iterrows():
        pct    = float(row["unsold_pct"])
        days   = int(row["days_to_expiry"])
        unsold = float(row["predicted_unsold_units"])
        tier   = map_expiry_risk_tier(pct, days, unsold)
        action = map_mitigation_action(days, tier)
        risk_levels.append(tier)
        actions.append(action)

    df["risk_level"]         = risk_levels
    df["recommended_action"] = actions

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_res = predict_batch_expiry()
    print("Expiry Predictions Generated Successfully:")
    print("  Total ACTIVE batches  :", len(df_res))
    print("  days_to_expiry range  :", df_res["days_to_expiry"].min(), "–", df_res["days_to_expiry"].max())
    print("  Risk Level Counts:\n", df_res["risk_level"].value_counts())
    print("  Total Financial Exposure: Rs.", f"{df_res['financial_exposure'].sum():,.2f}")
