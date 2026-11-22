"""
PharmaPulse - Stockout Intelligence Comprehensive Verification Test Suite
=========================================================================
Tests all components of the Stockout Intelligence Module (Module 3):
  1. Feature pipeline and model artifact loading (xgboost_stockout_clf.joblib)
  2. Authentic ML inference on pending inbound purchase orders
  3. Real-world active formulary inventory assessment across all 150 SKUs
  4. Real supplier performance metrics & dynamic safety buffer multipliers (10 distinct suppliers)
  5. Multi-criteria dynamic filtering and KPI recalculation (/api/stockout)
  6. Data diversity & non-static behavior across suppliers and risk tiers
"""

import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.stockout.features import STOCKOUT_FEATURE_COLS
from modules.stockout.predict import predict_stockout_risks
from backend.app.routers.stockout import get_stockout_data

print("=" * 75)
print("PHARMAPULSE — STOCKOUT INTELLIGENCE MODULE VERIFICATION TEST SUITE")
print("=" * 75)

# -------------------------------------------------------------
# TEST 1: Model Artifacts & Feature Schema
# -------------------------------------------------------------
print("\n[Test 1] Verifying Stockout Model Artifacts & Feature Schema...")
models_dir = os.path.join(PROJECT_ROOT, "models", "stockout")
assert os.path.exists(os.path.join(models_dir, "xgboost_stockout_clf.joblib")), "XGBoost model missing!"
assert os.path.exists(os.path.join(models_dir, "feature_cols.joblib")), "Feature cols missing!"
assert len(STOCKOUT_FEATURE_COLS) == 13, f"Expected 13 feature cols, got {len(STOCKOUT_FEATURE_COLS)}"
print(f"  Passed: All stockout model artifacts and 13 feature columns verified.")

# -------------------------------------------------------------
# TEST 2: Real-World Formulary Inventory & Demand Burn Rate Assessment
# -------------------------------------------------------------
print("\n[Test 2] Testing Active Formulary Inventory Assessment (150 Medicines)...")
raw_output = predict_stockout_risks()
items = raw_output["items"]

assert len(items) == 150, f"Expected 150 formulary medicines, got {len(items)}"

# Check for non-zero stock diversity
stock_values = [it["current_stock"] for it in items]
assert min(stock_values) >= 0, "Negative stock found!"
assert max(stock_values) > 1000, f"Expected maximum stock > 1000, got {max(stock_values)}"
assert len(set(stock_values)) > 50, "Stock values lack diversity!"

# Check daily burn rate diversity (no hardcoded 5.0)
burn_values = [it["burn_rate"] for it in items]
assert len(set(burn_values)) > 30, "Burn rates lack diversity!"
assert min(burn_values) > 0, "Zero or negative burn rate found!"

# Check Days of Supply
dos_values = [it["days_of_supply"] for it in items]
assert min(dos_values) >= 0.0, "Negative days of supply found!"
assert len(set(dos_values)) > 50, "Days of supply lacks diversity!"

print(f"  Formulary SKUs Evaluated : {len(items)}")
print(f"  Stock Range              : {min(stock_values)} to {max(stock_values)} units (mean: {np.mean(stock_values):.1f})")
print(f"  Burn Rate Range          : {min(burn_values):.1f} to {max(burn_values):.1f} u/d (mean: {np.mean(burn_values):.1f})")
print(f"  Days of Supply Range     : {min(dos_values):.1f}d to {max(dos_values):.1f}d (mean: {np.mean(dos_values):.1f})")
print("  Passed: Formulary inventory & burn rates reflect authentic pharmacy data.")

# -------------------------------------------------------------
# TEST 3: Authentic ML Inference on Pending Purchase Orders
# -------------------------------------------------------------
print("\n[Test 3] Verifying ML Inference on Pending Inbound POs...")
pos = raw_output["recent_pos"]
pending_pos = [p for p in pos if p["status"].lower() == "pending"]
assert len(pending_pos) == 87, f"Expected 87 pending POs, got {len(pending_pos)}"

# Verify that medicines with pending POs received ML probabilities
items_with_ml = [it for it in items if it["ml_stockout_prob"] is not None]
assert len(items_with_ml) > 0, "No medicines received ML inference probabilities!"
ml_probs = [it["ml_stockout_prob"] for it in items_with_ml]
assert min(ml_probs) >= 0.0 and max(ml_probs) <= 1.0, "Invalid ML probability bounds!"
print(f"  Total Inbound POs Tracked : {len(pos)} ({len(pending_pos)} Pending)")
print(f"  Medicines Scored with ML  : {len(items_with_ml)} SKUs")
print(f"  ML Probability Range      : {min(ml_probs):.3f} to {max(ml_probs):.3f}")
print("  Passed: Authentic ML inference executed across replenishment pipeline.")

# -------------------------------------------------------------
# TEST 4: Supplier Reliability & Dynamic Safety Buffer Multipliers
# -------------------------------------------------------------
print("\n[Test 4] Verifying Supplier Performance & Buffer Multipliers...")
suppliers = raw_output["suppliers"]
assert len(suppliers) == 10, f"Expected 10 qualified suppliers, got {len(suppliers)}"

# Verify no identical dummy fallback values (e.g. all 85.0%)
rel_values = [s["reliability_pct"] for s in suppliers]
lt_values = [s["lead_time_days"] for s in suppliers]
buffer_values = [s["buffer_multiplier"] for s in suppliers]

assert len(set(rel_values)) >= 8, f"Supplier reliability lacks diversity! Found: {set(rel_values)}"
assert len(set(lt_values)) >= 7, f"Supplier lead times lack diversity! Found: {set(lt_values)}"
assert len(set(buffer_values)) >= 7, f"Buffer multipliers lack diversity! Found: {set(buffer_values)}"

# Verify buffer multiplier formula: buffer = 1.0 + ((100 - rel)/100) * 1.5
for s in suppliers:
    expected_buf = round(1.0 + ((100.0 - s["reliability_pct"]) / 100.0) * 1.5, 2)
    assert abs(s["buffer_multiplier"] - expected_buf) < 0.02, f"Buffer formula mismatch for {s['supplier_name']}"

print("  Sample Supplier Metrics:")
for s in suppliers[:4]:
    print(f"    {s['supplier_id']} ({s['supplier_name'][:25]:25s}): Rel={s['reliability_pct']}%, LT={s['lead_time_days']}d, Buffer={s['buffer_multiplier']}x, {s['performance_tier']}")
print("  Passed: Real supplier delivery volatility and dynamic buffers verified.")

# -------------------------------------------------------------
# TEST 5: FastAPI /api/stockout Endpoint & Dynamic Filter Responsiveness
# -------------------------------------------------------------
print("\n[Test 5] Testing Dynamic Multi-Criteria Filtering & KPI Recalculation...")
# Base unfiltered
res_all = get_stockout_data()
k_all = res_all["kpis"]
assert k_all["total_monitored_events"] == 150
assert k_all["high_risk_threats"] > 0
assert k_all["vulnerability_rate_pct"] > 0
assert k_all["avg_burn_rate_daily"] > 0
assert k_all["avg_days_of_supply"] > 0
assert k_all["active_suppliers_count"] == 10

# Supplier 1: SUP001 (Premier Pharma Distribution - short lead time, high reliability)
res_sup1 = get_stockout_data(supplier_id="SUP001")
k_sup1 = res_sup1["kpis"]
assert k_sup1["total_monitored_events"] == 15
assert k_sup1["active_suppliers_count"] == 1
assert all(it["supplier_id"] == "SUP001" for it in res_sup1["items"])
assert all(p["supplier_id"] == "SUP001" for p in res_sup1["recent_pos"])

# Supplier 9: SUP009 (Chiltan Med Logistics - long lead time, lower reliability)
res_sup9 = get_stockout_data(supplier_id="SUP009")
k_sup9 = res_sup9["kpis"]
assert k_sup9["total_monitored_events"] == 15
assert k_sup9["active_suppliers_count"] == 1
assert all(it["supplier_id"] == "SUP009" for it in res_sup9["items"])

# Verify dynamic KPI changes between suppliers
assert k_sup9["high_risk_threats"] > k_sup1["high_risk_threats"], "SUP009 should have higher stockout threats than SUP001!"
assert k_sup9["vulnerability_rate_pct"] > k_sup1["vulnerability_rate_pct"], "Vulnerability rate not dynamic across suppliers!"
assert k_sup9["avg_days_of_supply"] < k_sup1["avg_days_of_supply"], "Avg DOS not dynamic across suppliers!"

# Risk Level Filter: High Risk Only
res_hr = get_stockout_data(risk_level="High Risk")
k_hr = res_hr["kpis"]
assert k_hr["vulnerability_rate_pct"] == 100.0, "High Risk filter should have 100% vulnerability rate"
assert k_hr["total_monitored_events"] == k_hr["high_risk_threats"]
assert all(it["risk_flag"] == "High Risk" for it in res_hr["items"])

# Risk Level Filter: Low Risk Only
res_lr = get_stockout_data(risk_level="Low Risk")
k_lr = res_lr["kpis"]
assert k_lr["vulnerability_rate_pct"] == 0.0, "Low Risk filter should have 0% vulnerability rate"
assert k_lr["high_risk_threats"] == 0
assert all(it["risk_flag"] == "Low Risk" for it in res_lr["items"])

print(f"  Unfiltered KPIs        : Total={k_all['total_monitored_events']}, Threats={k_all['high_risk_threats']}, Vuln={k_all['vulnerability_rate_pct']}%, DOS={k_all['avg_days_of_supply']}d")
print(f"  SUP001 Filtered KPIs   : Total={k_sup1['total_monitored_events']}, Threats={k_sup1['high_risk_threats']}, Vuln={k_sup1['vulnerability_rate_pct']}%, DOS={k_sup1['avg_days_of_supply']}d")
print(f"  SUP009 Filtered KPIs   : Total={k_sup9['total_monitored_events']}, Threats={k_sup9['high_risk_threats']}, Vuln={k_sup9['vulnerability_rate_pct']}%, DOS={k_sup9['avg_days_of_supply']}d")
print(f"  High Risk Filtered KPIs: Total={k_hr['total_monitored_events']}, Threats={k_hr['high_risk_threats']}, Vuln={k_hr['vulnerability_rate_pct']}%, DOS={k_hr['avg_days_of_supply']}d")
print("  Passed: All 4 KPIs dynamically update and reflect filtered operational subsets.")

# -------------------------------------------------------------
# TEST 6: Domain Physical & Action Consistency
# -------------------------------------------------------------
print("\n[Test 6] Verifying Domain Actions & Physical Rules Consistency...")
for it in res_all["items"]:
    stock = it["current_stock"]
    dos = it["days_of_supply"]
    lt = it["lead_time_days"]
    risk = it["risk_flag"]
    action = it["recommended_action"]

    if stock == 0:
        assert risk == "High Risk", f"Zero stock item {it['medicine_name']} must be High Risk!"
        assert "Active Stockout" in action, f"Zero stock item missing Active Stockout action: {action}"
    elif dos <= lt:
        assert risk == "High Risk", f"Item with DOS ({dos}) <= LT ({lt}) must be High Risk!"

print("  Passed: Domain physical consistency strictly enforced across all formulary items.")

print("\n" + "=" * 75)
print("ALL STOCKOUT INTELLIGENCE CHECKS PASSED (100% CORRECT & DYNAMIC)!")
print("=" * 75)
