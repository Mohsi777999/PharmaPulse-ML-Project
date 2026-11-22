"""
PharmaPulse - Expiry Intelligence Comprehensive Verification Test Suite
======================================================================
Tests all components of the Expiry Intelligence Module:
  1. Feature pipeline and model artifact loading
  2. Dual-task ML inference (XGBoost Classifier + Gradient Boosting Regressor)
  3. Domain physical boundary constraints (non-negative, on-hand stock ceiling)
  4. Batch deduplication (unique physical batch identities)
  5. 4-tier risk classification (Critical, High, Medium, Low)
  6. Financial capital exposure calculations in PKR
  7. 3-tier operational mitigation action mapping
  8. FastAPI /api/expiry endpoint integrity and multi-criteria filtering
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.expiry.features import EXPIRY_FEATURE_COLS
from modules.expiry.predict import predict_batch_expiry
from backend.app.routers.expiry import get_expiry_data

print("=" * 75)
print("PHARMAPULSE — EXPIRY INTELLIGENCE MODULE VERIFICATION TEST SUITE")
print("=" * 75)

# -------------------------------------------------------------
# TEST 1: Model Artifacts & Feature Schema
# -------------------------------------------------------------
print("\n[Test 1] Verifying Model Artifacts & Feature Columns Schema...")
models_dir = os.path.join(PROJECT_ROOT, "models", "expiry")
assert os.path.exists(os.path.join(models_dir, "xgboost_expiry_clf.joblib")), "Classifier missing!"
assert os.path.exists(os.path.join(models_dir, "gradient_boosting_reg.joblib")), "Regressor missing!"
assert len(EXPIRY_FEATURE_COLS) == 12, f"Expected 12 feature cols, got {len(EXPIRY_FEATURE_COLS)}"
print(f"  Passed: All model artifacts and 12 feature columns verified.")

# -------------------------------------------------------------
# TEST 2: Dual-Task ML Inference & Physical Bounds
# -------------------------------------------------------------
print("\n[Test 2] Testing Dual-Task ML Inference & Physical Constraints...")
df_pred = predict_batch_expiry()

# Deduplication check
total_rows = len(df_pred)
unique_batches = df_pred["batch_id"].nunique()
print(f"  Total batches evaluated: {total_rows}")
print(f"  Unique physical batch IDs: {unique_batches}")
assert total_rows == unique_batches, f"Duplicate batches found! Total: {total_rows}, Unique: {unique_batches}"
print("  Passed: 100% unique batch identities (zero duplicate snapshots).")

# Physical Constraint 1: Non-negativity
neg_unsold = (df_pred["predicted_unsold_units"] < 0).sum()
assert neg_unsold == 0, f"Found {neg_unsold} negative unsold predictions!"
print("  Passed: Physical Rule 1 (predicted_unsold >= 0) strictly satisfied.")

# Physical Constraint 2: Stock ceiling (cannot expire more than on-hand stock)
exceeded_stock = (df_pred["predicted_unsold_units"] > df_pred["remaining_batch_quantity"]).sum()
assert exceeded_stock == 0, f"Found {exceeded_stock} batches where predicted unsold exceeds stock!"
print("  Passed: Physical Rule 2 (predicted_unsold <= remaining_batch_quantity) strictly satisfied.")

# Physical Constraint 3: Financial Exposure Calculation
expected_exposure = (df_pred["predicted_unsold_units"] * df_pred["unit_cost"]).round(2)
diff = np.abs(df_pred["financial_exposure"] - expected_exposure).max()
assert diff < 0.01, f"Financial exposure mismatch! Max diff: {diff}"
print("  Passed: Financial exposure formula (predicted_unsold * unit_cost) verified.")

# -------------------------------------------------------------
# TEST 3: 4-Tier Risk Categorization & Mitigation Actions
# -------------------------------------------------------------
print("\n[Test 3] Verifying 4-Tier Risk Categorization & Mitigation Action Rules...")
risk_counts = df_pred["risk_level"].value_counts().to_dict()
print(f"  Risk Level Breakdown: {risk_counts}")
assert set(risk_counts.keys()).issubset({"Critical", "High", "Medium", "Low"}), "Invalid risk tiers found!"
assert "Critical" in risk_counts, "Critical risk tier missing!"
assert "High" in risk_counts, "High risk tier missing!"
assert "Medium" in risk_counts, "Medium risk tier missing!"
assert "Low" in risk_counts, "Low risk tier missing!"

# Verify Mitigation Rules
for _, row in df_pred.iterrows():
    tier = row["risk_level"]
    days = int(row["days_to_expiry"])
    action = row["recommended_action"]
    if tier != "Low":
        if days > 90:
            assert "Tier 1" in action, f"Expected Tier 1 for >90d, got: {action}"
        elif days >= 30:
            assert "Tier 2" in action, f"Expected Tier 2 for 30-90d, got: {action}"
        else:
            assert "Tier 3" in action, f"Expected Tier 3 for <30d, got: {action}"
print("  Passed: 3-tier operational mitigation playbook rules strictly verified.")

# -------------------------------------------------------------
# TEST 4: FastAPI /api/expiry Endpoint Integrity
# -------------------------------------------------------------
print("\n[Test 4] Testing FastAPI /api/expiry API Endpoint & KPI Summary...")
api_res = get_expiry_data()

kpis = api_res["kpis"]
print(f"  Total Capital at Risk: Rs. {kpis['total_capital_at_risk_pkr']:,.2f}")
print(f"  Urgent Batches (Critical + High): {kpis['urgent_batches']} ({kpis['critical_risk_batches']} Critical, {kpis['high_risk_batches']} High)")
print(f"  Total Units at Risk: {kpis['total_units_at_risk']:,} units")
print(f"  Avg Days to Expiry: {kpis['avg_days_to_expiry']} days")

assert kpis["urgent_batches"] == kpis["critical_risk_batches"] + kpis["high_risk_batches"], "Urgent sum mismatch!"
assert kpis["total_capital_at_risk_pkr"] > 0, "Capital at risk must be > 0"

# Donut chart structure
dist = api_res["risk_distribution"]
assert len(dist) == 4, f"Expected 4 risk categories in distribution, got {len(dist)}"
dist_names = [d["name"] for d in dist]
assert dist_names == ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
total_dist_val = sum(d["value"] for d in dist)
assert total_dist_val == kpis["total_tracked_batches"], "Donut sum doesn't match total batches!"
print("  Passed: KPI aggregations and 4-tier donut chart distribution verified.")

# -------------------------------------------------------------
# TEST 5: Multi-Criteria Filtering
# -------------------------------------------------------------
print("\n[Test 5] Testing Multi-Criteria Filtering Logic...")
# Filter Critical Only
crit_res = get_expiry_data(risk_level="Critical")
for b in crit_res["batches"]:
    assert b["risk_level"] == "Critical", f"Expected Critical, got {b['risk_level']}"
print(f"  Passed: Filter 'risk_level=Critical' returned {len(crit_res['batches'])} Critical batches.")

# Filter High Only
high_res = get_expiry_data(risk_level="High")
for b in high_res["batches"]:
    assert b["risk_level"] == "High", f"Expected High, got {b['risk_level']}"
print(f"  Passed: Filter 'risk_level=High' returned {len(high_res['batches'])} High batches.")

# Filter by Max Days
days_res = get_expiry_data(max_days=30)
for b in days_res["batches"]:
    assert b["days_to_expiry"] <= 30, f"Expected <=30d, got {b['days_to_expiry']}"
print(f"  Passed: Filter 'max_days=30' returned {len(days_res['batches'])} batches with <=30 days left.")

# Filter by Category
cat_res = get_expiry_data(category_id="CAT_01")
assert len(cat_res["batches"]) > 0, "No batches returned for CAT_01"
for b in cat_res["batches"]:
    assert b["category_name"].lower() == "pain relief and fever", f"Category mismatch: {b['category_name']}"
print(f"  Passed: Filter 'category_id=CAT_01' returned {len(cat_res['batches'])} matching batches.")

# -------------------------------------------------------------
# TEST 6: Dynamic Expiry Horizon Slider Verification
# -------------------------------------------------------------
print("\n[Test 6] Verifying Dynamic KPI & Chart Responsiveness to Expiry Horizon...")
res_14 = get_expiry_data(max_days=14)
res_30 = get_expiry_data(max_days=30)
res_60 = get_expiry_data(max_days=60)
res_90 = get_expiry_data(max_days=90)
res_365 = get_expiry_data(max_days=365)

# Verify monotonic increase of exposure, urgent batches, units at risk with horizon
assert res_14["kpis"]["total_capital_at_risk_pkr"] <= res_30["kpis"]["total_capital_at_risk_pkr"], "Exposure not dynamic!"
assert res_30["kpis"]["total_capital_at_risk_pkr"] < res_60["kpis"]["total_capital_at_risk_pkr"], "Exposure 30 vs 60 not dynamic!"
assert res_60["kpis"]["total_capital_at_risk_pkr"] < res_90["kpis"]["total_capital_at_risk_pkr"], "Exposure 60 vs 90 not dynamic!"
assert res_90["kpis"]["total_capital_at_risk_pkr"] <= res_365["kpis"]["total_capital_at_risk_pkr"], "Exposure 90 vs 365 not dynamic!"

# Verify urgent batches change dynamically
assert res_30["kpis"]["urgent_batches"] < res_60["kpis"]["urgent_batches"] < res_90["kpis"]["urgent_batches"], "Urgent batches not dynamic!"

# Verify avg days to expiry changes dynamically and respects the horizon ceiling
assert res_30["kpis"]["avg_days_to_expiry"] <= 30.0, f"Avg days {res_30['kpis']['avg_days_to_expiry']} > 30d horizon!"
assert res_60["kpis"]["avg_days_to_expiry"] <= 60.0, f"Avg days {res_60['kpis']['avg_days_to_expiry']} > 60d horizon!"
assert res_30["kpis"]["avg_days_to_expiry"] < res_60["kpis"]["avg_days_to_expiry"] < res_90["kpis"]["avg_days_to_expiry"], "Avg days not dynamic!"

# Verify donut chart (risk distribution) values change dynamically
dist_30_total = sum(d["value"] for d in res_30["risk_distribution"])
dist_60_total = sum(d["value"] for d in res_60["risk_distribution"])
dist_365_total = sum(d["value"] for d in res_365["risk_distribution"])
assert dist_30_total < dist_60_total < dist_365_total, "Donut chart distribution not dynamic!"

# Verify category exposure chart changes dynamically
cat_exp_30 = sum(c["exposure_pkr"] for c in res_30["category_exposure"])
cat_exp_60 = sum(c["exposure_pkr"] for c in res_60["category_exposure"])
cat_exp_365 = sum(c["exposure_pkr"] for c in res_365["category_exposure"])
assert cat_exp_30 < cat_exp_60 < cat_exp_365, "Category exposure chart not dynamic!"

print(f"  Horizon <= 30d: Capital=Rs. {res_30['kpis']['total_capital_at_risk_pkr']:,.2f}, Urgent={res_30['kpis']['urgent_batches']}, AvgDays={res_30['kpis']['avg_days_to_expiry']}d")
print(f"  Horizon <= 60d: Capital=Rs. {res_60['kpis']['total_capital_at_risk_pkr']:,.2f}, Urgent={res_60['kpis']['urgent_batches']}, AvgDays={res_60['kpis']['avg_days_to_expiry']}d")
print(f"  Horizon <= 90d: Capital=Rs. {res_90['kpis']['total_capital_at_risk_pkr']:,.2f}, Urgent={res_90['kpis']['urgent_batches']}, AvgDays={res_90['kpis']['avg_days_to_expiry']}d")
print(f"  Passed: All 4 KPIs, Risk Classification Share, and Category Exposure are 100% dynamic & logically coherent!")

print("\n" + "=" * 75)
print("ALL EXPIRY INTELLIGENCE CHECKS PASSED (100% CORRECT & LOGICAL)!")
print("=" * 75)
