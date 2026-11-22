"""
PharmaPulse - Stockout Intelligence API Router
==============================================
Provides dynamic stockout intelligence endpoints with multi-criteria filtering:
  - Vendor / Supplier filter (supplier_id: ALL, SUP001..SUP010)
  - Risk Classification filter (risk_level: ALL, High Risk, Low Risk)

All KPIs, formulary threat items, supplier buffer multipliers, and inbound POs
dynamically respond to the applied filters and execute authentic ML inference.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from fastapi import APIRouter, Query

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_model_evaluations
from modules.stockout.predict import predict_stockout_risks

router = APIRouter(prefix="/api/stockout", tags=["Stockout Intelligence"])

# In-memory cache for stockout predictions to ensure sub-millisecond filter responsiveness
_cached_stockout_data: Optional[Dict[str, Any]] = None


def get_cached_stockout_predictions(force_reload: bool = False) -> Dict[str, Any]:
    """Returns cached stockout intelligence data or recomputes if None."""
    global _cached_stockout_data
    if _cached_stockout_data is None or force_reload:
        _cached_stockout_data = predict_stockout_risks()
    return _cached_stockout_data


def clear_stockout_cache():
    """Invalidates the in-memory cache (e.g. after model retraining or inventory update)."""
    global _cached_stockout_data
    _cached_stockout_data = None


@router.get("")
def get_stockout_data(
    supplier_id: str = "ALL",
    risk_level: str = "ALL",
    force_reload: bool = False
):
    """
    Returns dynamically aggregated stockout intelligence metrics:
      - 4 Summary KPIs responding dynamically to active filters
      - 10 Qualified Suppliers with real reliability & buffer multipliers
      - 7-Day Replenishment Threat Register (active formulary inventory items)
      - Inbound Purchase Orders Tracking Ledger (pending shipments)
      - ML Model Benchmark Evaluations
    """
    data = get_cached_stockout_predictions(force_reload=force_reload)
    all_items = data["items"]
    all_suppliers = data["suppliers"]
    all_recent_pos = data["recent_pos"]
    df_evals = load_model_evaluations()

    # ------------------------------------------------------------------
    # Step 1: Filter Items based on supplier_id and risk_level
    # ------------------------------------------------------------------
    filtered_items = all_items.copy()

    # Normalize supplier_id
    if supplier_id != "ALL":
        filtered_items = [it for it in filtered_items if it["supplier_id"].upper() == supplier_id.upper()]

    # Normalize risk_level
    if risk_level != "ALL":
        clean_risk = risk_level.strip().lower()
        if "high" in clean_risk or "threat" in clean_risk:
            filtered_items = [it for it in filtered_items if it["risk_flag"] == "High Risk"]
        elif "low" in clean_risk or "adequate" in clean_risk:
            filtered_items = [it for it in filtered_items if it["risk_flag"] == "Low Risk"]

    # ------------------------------------------------------------------
    # Step 2: Compute Dynamic KPIs from Filtered Items
    # ------------------------------------------------------------------
    total_monitored = len(filtered_items)
    high_risk_threats = sum(1 for it in filtered_items if it["risk_flag"] == "High Risk")
    vulnerability_rate_pct = round((high_risk_threats / max(total_monitored, 1)) * 100.0, 1) if total_monitored > 0 else 0.0

    if total_monitored > 0:
        avg_burn_rate = round(sum(it["burn_rate"] for it in filtered_items) / total_monitored, 1)
        avg_days_of_supply = round(sum(it["days_of_supply"] for it in filtered_items) / total_monitored, 1)
        active_sups_in_scope = len(set(it["supplier_id"] for it in filtered_items))
    else:
        avg_burn_rate = 0.0
        avg_days_of_supply = 0.0
        active_sups_in_scope = 0

    # ------------------------------------------------------------------
    # Step 3: Inbound Purchase Orders Filtering
    # ------------------------------------------------------------------
    filtered_pos = all_recent_pos.copy()
    if supplier_id != "ALL":
        filtered_pos = [p for p in filtered_pos if p["supplier_id"].upper() == supplier_id.upper()]

    # ------------------------------------------------------------------
    # Step 4: ML Evaluation Metrics
    # ------------------------------------------------------------------
    so_evals = []
    if not df_evals.empty:
        so_evals = df_evals[df_evals["module"].str.contains("stockout", case=False, na=False)].to_dict(orient="records")

    return {
        "kpis": {
            "total_monitored_events": total_monitored,
            "high_risk_threats": high_risk_threats,
            "vulnerability_rate_pct": vulnerability_rate_pct,
            "avg_burn_rate_daily": avg_burn_rate,
            "avg_days_of_supply": avg_days_of_supply,
            "active_suppliers_count": active_sups_in_scope if supplier_id != "ALL" else len(all_suppliers)
        },
        "suppliers": all_suppliers,
        "items": filtered_items,
        "recent_pos": filtered_pos,
        "evaluations": so_evals
    }
