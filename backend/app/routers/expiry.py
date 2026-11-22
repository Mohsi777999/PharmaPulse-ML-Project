"""
PharmaPulse - Expiry Intelligence API Router
============================================
Provides dynamic batch expiry intelligence endpoints with multi-criteria filtering:
  - Expiry Horizon slider (max_days: 14 to 365 days)
  - Therapeutic Category filter (category_id: ALL, CAT001..CAT006)
  - Risk Level filter (risk_level: ALL, Critical, High, Medium, Low)

All KPIs, risk distribution (donut chart), and category financial exposure (bar chart)
are dynamically recomputed based on the active horizon and category filters.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Optional
from fastapi import APIRouter, Query

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_model_evaluations
from modules.expiry.predict import predict_batch_expiry

router = APIRouter(prefix="/api/expiry", tags=["Expiry Intelligence"])

# In-memory cache for batch predictions to ensure sub-millisecond filter responsiveness
_cached_predictions: Optional[pd.DataFrame] = None

# Category master taxonomy for consistent bar chart representation
CATEGORIES_METADATA = [
    {"category_id": "CAT001", "category_name": "Pain Relief & Fever"},
    {"category_id": "CAT002", "category_name": "Anti-Infectives"},
    {"category_id": "CAT003", "category_name": "Respiratory & Cold"},
    {"category_id": "CAT004", "category_name": "Gastrointestinal"},
    {"category_id": "CAT005", "category_name": "Chronic Care"},
    {"category_id": "CAT006", "category_name": "Vitamins & Supplements"},
]


def get_cached_predictions(force_reload: bool = False) -> pd.DataFrame:
    """Returns cached ML predictions across active batches or recomputes if None."""
    global _cached_predictions
    if _cached_predictions is None or force_reload:
        _cached_predictions = predict_batch_expiry()
    return _cached_predictions.copy()


def clear_expiry_cache():
    """Invalidates the in-memory predictions cache (e.g. after model retraining)."""
    global _cached_predictions
    _cached_predictions = None


@router.get("")
def get_expiry_data(
    category_id: str = "ALL",
    risk_level: str = "ALL",
    max_days: int = 365,
    force_reload: bool = False
):
    """
    Returns batch-level expiry predictions, dynamically aggregated KPIs,
    risk classification share, and category exposure breakdown.

    All aggregations strictly respond to `max_days` (Expiry Horizon)
    and `category_id` to provide dynamic, logically correct dashboard metrics.
    """
    df = get_cached_predictions(force_reload=force_reload)
    df_evals = load_model_evaluations()

    # ------------------------------------------------------------------
    # Step 1: Apply Expiry Horizon filter (max_days)
    # ------------------------------------------------------------------
    horizon_df = df[df["days_to_expiry"] <= max_days].copy()

    # ------------------------------------------------------------------
    # Step 2: Apply Therapeutic Category filter for scoped metrics
    # ------------------------------------------------------------------
    clean_cat = category_id
    if category_id != "ALL":
        cat_num = "".join(filter(str.isdigit, category_id))
        if cat_num:
            clean_cat = f"CAT{int(cat_num):03d}"
            scoped_df = horizon_df[horizon_df["category_id"] == clean_cat].copy()
        else:
            scoped_df = horizon_df[horizon_df["category_id"].str.lower() == category_id.lower()].copy()
    else:
        scoped_df = horizon_df.copy()

    # ------------------------------------------------------------------
    # Step 3: Compute Dynamic KPIs on the scoped dataset
    # ------------------------------------------------------------------
    has_data = len(scoped_df) > 0
    total_exposure_pkr = round(float(scoped_df["financial_exposure"].sum()), 2) if has_data else 0.0
    critical_count = int((scoped_df["risk_level"] == "Critical").sum()) if has_data else 0
    high_count = int((scoped_df["risk_level"] == "High").sum()) if has_data else 0
    med_count = int((scoped_df["risk_level"] == "Medium").sum()) if has_data else 0
    low_count = int((scoped_df["risk_level"] == "Low").sum()) if has_data else 0
    urgent_count = critical_count + high_count

    avg_days_to_expiry = (
        round(float(scoped_df["days_to_expiry"].mean()), 1) if has_data else 0.0
    )
    total_units_at_risk = (
        int(round(float(scoped_df["predicted_unsold_units"].sum()))) if has_data else 0
    )
    total_tracked_batches = len(scoped_df)

    # ------------------------------------------------------------------
    # Step 4: Risk Distribution Breakdown (Donut Chart)
    # Reflects the 4-tier risk classification share within the selected scope
    # ------------------------------------------------------------------
    risk_distribution = [
        {"name": "Critical Risk", "value": critical_count, "color": "#EF4444"},
        {"name": "High Risk", "value": high_count, "color": "#F87171"},
        {"name": "Medium Risk", "value": med_count, "color": "#FBBF24"},
        {"name": "Low Risk", "value": low_count, "color": "#34D399"}
    ]

    # ------------------------------------------------------------------
    # Step 5: Financial Exposure by Therapeutic Category (Bar Chart)
    # Dynamically recomputed from horizon_df for all categories within the horizon
    # ------------------------------------------------------------------
    cat_exp_sums = horizon_df.groupby("category_id")["financial_exposure"].sum().to_dict()
    category_exposure = []
    for cat in CATEGORIES_METADATA:
        cid = cat["category_id"]
        cname = cat["category_name"]
        # If filtered to a specific category, show only that category or keep context
        if category_id != "ALL" and cid != clean_cat:
            continue
        exp_val = float(cat_exp_sums.get(cid, 0.0))
        category_exposure.append({
            "category_id": cid,
            "category_name": cname,
            "exposure_pkr": round(exp_val, 2),
            "exposure_k": round(exp_val / 1000.0, 1)
        })

    # ------------------------------------------------------------------
    # Step 6: Filter Batches for the Register Table
    # Applies risk_level filter on top of horizon & category
    # ------------------------------------------------------------------
    table_df = scoped_df.copy()
    if risk_level != "ALL":
        table_df = table_df[table_df["risk_level"].str.lower() == risk_level.lower()]

    table_df = table_df.sort_values(by="financial_exposure", ascending=False)

    batches_list = []
    for _, r in table_df.head(150).iterrows():
        batches_list.append({
            "batch_id": str(r["batch_id"]),
            "medicine_id": str(r["medicine_id"]),
            "medicine_name": str(r["medicine_name"]),
            "category_name": str(r["category_name"]),
            "days_to_expiry": int(r["days_to_expiry"]),
            "current_quantity": int(r["remaining_batch_quantity"]),
            "predicted_unsold": round(float(r["predicted_unsold_units"]), 1),
            "unsold_probability": round(float(r.get("unsold_probability", 0.0)), 2),
            "financial_exposure": float(r["financial_exposure"]),
            "risk_level": str(r["risk_level"]),
            "recommended_action": str(r["recommended_action"])
        })

    # Model evaluation metrics for Expiry module
    exp_evals = df_evals[df_evals["module"].str.contains("expiry", case=False, na=False)].to_dict(orient="records")

    return {
        "kpis": {
            "total_capital_at_risk_pkr": total_exposure_pkr,
            "total_capital_at_risk_k": round(total_exposure_pkr / 1000.0, 1),
            "critical_risk_batches": critical_count,
            "high_risk_batches": high_count,
            "medium_risk_batches": med_count,
            "low_risk_batches": low_count,
            "urgent_batches": urgent_count,
            "avg_days_to_expiry": avg_days_to_expiry,
            "total_units_at_risk": total_units_at_risk,
            "total_tracked_batches": total_tracked_batches
        },
        "risk_distribution": risk_distribution,
        "category_exposure": category_exposure,
        "batches": batches_list,
        "evaluations": exp_evals
    }
