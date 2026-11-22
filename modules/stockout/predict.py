"""
PharmaPulse - Stockout Intelligence Production Inference Engine
===============================================================
Provides production inference for inventory stockout risk and replenishment pipeline monitoring.

Real-World Domain Architecture:
  1. Active Formulary Inventory Assessment (150 Medicines):
     - Uses live on-hand usable inventory from data/raw/daily_inventory.csv (closing_stock at 2026-12-30).
     - Computes real daily burn rate (units/day) from rolling 30-day historical consumption.
     - Calculates Days of Supply (DOS = current_stock / burn_rate).
     - Incorporates preferred vendor lead time and delivery reliability from data/raw/suppliers.csv.

  2. Replenishment Pipeline ML Inference (87 Pending Inbound POs):
     - Uses trained XGBoost Classifier (xgboost_stockout_clf.joblib) on the 13 feature schema.
     - Predicts the probability of inventory exhausting before vendor delivery arrives.
     - Identifies high-threat shipments (P(Stockout) >= 0.50).

  3. Domain Physical Stockout Classification:
     - High Risk / High Threat:
       * On-hand stock is 0 (active stockout), OR
       * DOS <= vendor lead time (stock exhausts before normal replenishment), OR
       * Active pending inbound PO has ML predicted stockout probability >= 0.50.
     - Low Risk / Adequate:
       * DOS > 1.5 * vendor lead time and inbound pipeline is healthy.

  4. Actionable Supply Chain & Clinical Protocols:
     - Active stockout: "Critical: Active Stockout! Issue Emergency Transfer / Expedite PO"
     - Imminent exhaustion: "Urgent: Expedite In-Transit PO / Place Emergency Buffer"
     - Approaching reorder: "Warning: Stock Approaching Reorder Point (< 1.5x Lead Time)"
     - Healthy inventory: "Stable: Inventory Adequate for Lead Time Cycle"
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "stockout")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Simulation reference date (latest date in daily_inventory.csv)
SIM_REF_DATE = pd.Timestamp("2026-12-30")

STOCKOUT_FEATURES = [
    "usable_stock_at_order", "outstanding_order_quantity_at_order",
    "historical_demand_7d", "historical_demand_14d", "historical_demand_30d",
    "average_daily_demand_7d", "average_daily_demand_30d", "demand_std_30d",
    "reorder_level", "target_stock_days", "supplier_lead_time_days",
    "supplier_reliability", "days_until_expected_delivery"
]


def predict_stockout_risks() -> Dict[str, Any]:
    """
    Executes end-to-end stockout intelligence evaluation:
      - ML inference on pending inbound purchase orders
      - Formulary inventory assessment across all 150 SKUs
      - Supplier reliability and dynamic safety buffer matrix
    """
    # ------------------------------------------------------------------
    # Step 1: Load Raw Datasets
    # ------------------------------------------------------------------
    df_meds = pd.read_csv(os.path.join(RAW_DIR, "medicines.csv"))
    df_sups = pd.read_csv(os.path.join(RAW_DIR, "suppliers.csv"))
    df_cats = pd.read_csv(os.path.join(RAW_DIR, "categories.csv"))
    df_inv = pd.read_csv(os.path.join(RAW_DIR, "daily_inventory.csv"))
    df_pos = pd.read_csv(os.path.join(RAW_DIR, "purchase_orders.csv"))

    # Load ML Classifier and Feature Schema
    clf_path = os.path.join(MODELS_DIR, "xgboost_stockout_clf.joblib")
    feat_path = os.path.join(MODELS_DIR, "feature_cols.joblib")

    if not os.path.exists(clf_path):
        from modules.stockout.train import train_stockout_models
        train_stockout_models(PROJECT_ROOT)

    clf = joblib.load(clf_path)
    feat_cols = joblib.load(feat_path) if os.path.exists(feat_path) else STOCKOUT_FEATURES

    # ------------------------------------------------------------------
    # Step 2: ML Inference on Pending Inbound Purchase Orders
    # ------------------------------------------------------------------
    p_so_data = os.path.join(PROCESSED_DIR, "stockout_prediction_data.csv")
    po_risk_map = {}  # medicine_id -> highest predicted stockout probability on pending orders

    if os.path.exists(p_so_data):
        df_so_all = pd.read_csv(p_so_data)
        # Select pending PO events (censored target)
        df_pending_so = df_so_all[df_so_all["target_stockout_before_arrival"].isna()].copy()
        if len(df_pending_so) > 0:
            X_pending = df_pending_so[feat_cols].copy()
            pred_probs = clf.predict_proba(X_pending)[:, 1]
            df_pending_so["predicted_stockout_prob"] = pred_probs.round(3)
            df_pending_so["predicted_stockout_flag"] = (pred_probs >= 0.50).astype(int)

            for _, r in df_pending_so.iterrows():
                mid = str(r["medicine_id"])
                p = float(r["predicted_stockout_prob"])
                if mid not in po_risk_map or p > po_risk_map[mid]:
                    po_risk_map[mid] = p

    # ------------------------------------------------------------------
    # Step 3: Formulary Inventory & Demand Burn Rate Calculation
    # ------------------------------------------------------------------
    latest_date_str = df_inv["date"].max()
    inv_latest = df_inv[df_inv["date"] == latest_date_str].set_index("medicine_id")["closing_stock"].to_dict()

    df_inv["date_dt"] = pd.to_datetime(df_inv["date"])
    cutoff_30d = pd.to_datetime(latest_date_str) - pd.Timedelta(days=30)
    df_30 = df_inv[df_inv["date_dt"] >= cutoff_30d]
    burn_rates_30d = (df_30.groupby("medicine_id")["units_sold"].sum() / 30.0).round(2).to_dict()

    # Supplier lookups
    sup_dict = df_sups.set_index("supplier_id").to_dict(orient="index")
    cat_dict = df_cats.set_index("category_id")["category_name"].to_dict()

    # ------------------------------------------------------------------
    # Step 4: Evaluate All 150 Medicines in Active Formulary
    # ------------------------------------------------------------------
    items_list = []
    for _, med in df_meds.iterrows():
        mid = str(med["medicine_id"])
        mname = str(med["medicine_name"])
        cid = str(med["category_id"])
        cname = cat_dict.get(cid, "General")
        preferred_sup_id = str(med.get("preferred_supplier_id", "SUP001"))
        sup_info = sup_dict.get(preferred_sup_id, {})
        sname = sup_info.get("supplier_name", "Primary Distributor")
        lead_time = int(sup_info.get("lead_time_days", 5))

        stock = int(inv_latest.get(mid, 0))
        burn = float(burn_rates_30d.get(mid, 5.0))
        burn = max(burn, 0.5)  # ensure non-zero denominator

        dos = round(stock / burn, 1)

        # ML Stockout Probability from inbound pending PO (if any)
        ml_prob = po_risk_map.get(mid, None)

        # Domain Risk Rules:
        # 1. Critical if zero stock (already stocked out)
        # 2. High Risk if DOS <= lead time (stock runs out before replenishment)
        # 3. High Risk if pending PO has ML stockout probability >= 0.50
        is_active_stockout = (stock == 0)
        is_burnout_before_lead = (dos <= lead_time)
        is_ml_threat = (ml_prob is not None and ml_prob >= 0.50)

        is_high_risk = is_active_stockout or is_burnout_before_lead or is_ml_threat

        if is_active_stockout:
            action = "Critical: Active Stockout! Issue Emergency Transfer / Expedite PO"
            risk_flag = "High Risk"
        elif is_burnout_before_lead:
            action = f"Urgent: Expedite In-Transit PO / Place Emergency Buffer ({dos}d supply <= {lead_time}d lead time)"
            risk_flag = "High Risk"
        elif is_ml_threat:
            action = f"Urgent: ML Alert ({int(ml_prob*100)}% stockout prob on inbound shipment)"
            risk_flag = "High Risk"
        elif dos <= lead_time * 1.5:
            action = f"Warning: Review Reorder Level & Lead Time ({dos}d supply approaching {lead_time}d lead time)"
            risk_flag = "Low Risk"
        else:
            action = "Stable: Inventory Adequate for Replenishment Cycle"
            risk_flag = "Low Risk"

        items_list.append({
            "medicine_id": mid,
            "medicine_name": mname,
            "category_id": cid,
            "category_name": cname,
            "supplier_id": preferred_sup_id,
            "supplier_name": sname,
            "current_stock": stock,
            "burn_rate": round(burn, 1),
            "days_of_supply": dos,
            "lead_time_days": lead_time,
            "risk_flag": risk_flag,
            "is_high_risk": is_high_risk,
            "recommended_action": action,
            "ml_stockout_prob": ml_prob
        })

    # Sort items by days_of_supply ascending (most critical first)
    items_list.sort(key=lambda x: (x["current_stock"] > 0, x["days_of_supply"]))

    # ------------------------------------------------------------------
    # Step 5: Supplier Reliability & Dynamic Safety Buffer Matrix
    # ------------------------------------------------------------------
    suppliers_summary = []
    for _, sup in df_sups.iterrows():
        sid = str(sup["supplier_id"])
        sname = str(sup["supplier_name"])
        reliability = float(sup.get("delivery_reliability", 0.90)) * 100.0
        lead_time = float(sup.get("lead_time_days", 7.0))
        rating = float(sup.get("supplier_rating", 4.5))

        # Dynamic Safety Buffer Multiplier:
        # Buffer multiplier = 1.0 + ((100 - Reliability) / 100) * 1.5
        # Example: 98% reliability -> 1.03x buffer; 86% reliability -> 1.21x buffer
        buffer_mult = round(1.0 + ((100.0 - reliability) / 100.0) * 1.5, 2)

        if reliability >= 95.0:
            tier = "Tier A (Preferred)"
        elif reliability >= 90.0:
            tier = "Tier B (Standard)"
        else:
            tier = "Tier C (Watchlist)"

        suppliers_summary.append({
            "supplier_id": sid,
            "supplier_name": sname,
            "reliability_pct": round(reliability, 1),
            "lead_time_days": round(lead_time, 1),
            "supplier_rating": rating,
            "buffer_multiplier": buffer_mult,
            "performance_tier": tier
        })

    # ------------------------------------------------------------------
    # Step 6: Inbound Purchase Orders Ledger (87 Pending + Recent POs)
    # ------------------------------------------------------------------
    pos_data = []
    if not df_pos.empty:
        # Pending POs first, followed by recent delivered orders
        pending_pos = df_pos[df_pos["purchase_order_status"] == "PENDING"].sort_values("expected_delivery_date")
        delivered_pos = df_pos[df_pos["purchase_order_status"] == "DELIVERED"].tail(50).sort_values("order_date", ascending=False)
        combined_pos = pd.concat([pending_pos, delivered_pos], ignore_index=True)

        med_name_dict = df_meds.set_index("medicine_id")["medicine_name"].to_dict()
        sup_name_dict = df_sups.set_index("supplier_id")["supplier_name"].to_dict()

        for _, p in combined_pos.iterrows():
            mid = str(p["medicine_id"])
            sid = str(p["supplier_id"])
            pos_data.append({
                "po_id": str(p["purchase_order_id"]),
                "medicine_id": mid,
                "medicine_name": med_name_dict.get(mid, mid),
                "supplier_id": sid,
                "supplier_name": sup_name_dict.get(sid, sid),
                "quantity": int(p.get("ordered_quantity", 100)),
                "order_date": str(p["order_date"]),
                "expected_delivery": str(p["expected_delivery_date"]),
                "status": str(p.get("purchase_order_status", "DELIVERED")).capitalize()
            })

    return {
        "items": items_list,
        "suppliers": suppliers_summary,
        "recent_pos": pos_data
    }


if __name__ == "__main__":
    res = predict_stockout_risks()
    items = res["items"]
    high_count = sum(1 for it in items if it["risk_flag"] == "High Risk")
    print("Stockout Intelligence Engine Executed Successfully:")
    print(f"  Total Formulary Items : {len(items)}")
    print(f"  High-Risk Threats     : {high_count} ({round(high_count/len(items)*100, 1)}%)")
    print(f"  Qualified Suppliers   : {len(res['suppliers'])}")
    print(f"  Tracked Inbound POs   : {len(res['recent_pos'])}")
    print("\nTop 5 Most Urgent Items:")
    for it in items[:5]:
        print(f"  {it['medicine_id']} | {it['medicine_name'][:30]:30s} | Stock: {it['current_stock']:4d} | Burn: {it['burn_rate']:4.1f}u/d | DOS: {it['days_of_supply']:4.1f}d | LT: {it['lead_time_days']:2d}d | {it['risk_flag']}")
