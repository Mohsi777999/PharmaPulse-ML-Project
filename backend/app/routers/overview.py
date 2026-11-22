import os
import sys
import pandas as pd
import numpy as np
from fastapi import APIRouter, Query
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import (
    load_medicines, load_categories, load_daily_demand_sales,
    load_daily_inventory, load_batches, load_validation_report
)
from modules.intelligence.insights import get_pharmacy_intelligence_summary

router = APIRouter(prefix="/api/overview", tags=["Overview"])


@router.get("")
def get_overview_data(
    time_range: str = "ALL",
    category_id: str = "ALL"
):
    """
    Returns dynamic, fully functional executive operations overview data.
    Supports filtering by time_range ('ALL', '12M', '6M', '30D') and category_id.
    """
    df_sales = load_daily_demand_sales()
    df_inv = load_daily_inventory()
    df_meds = load_medicines()
    df_cats = load_categories()
    df_batches = load_batches()
    val_report = load_validation_report()

    cat_dict = df_cats.set_index("category_id")["category_name"].to_dict()
    categories_meta = [{"category_id": "ALL", "category_name": "All Categories"}]
    for _, c in df_cats.iterrows():
        categories_meta.append({
            "category_id": c["category_id"],
            "category_name": c["category_name"]
        })

    # 1. Filter by Category
    f_meds = df_meds.copy()
    f_sales = df_sales.copy()
    f_inv = df_inv.copy()
    f_batches = df_batches.copy()

    selected_category_name = "All Categories"
    if category_id and category_id.upper() != "ALL":
        target_cat = category_id.upper()
        selected_category_name = cat_dict.get(target_cat, target_cat)
        f_meds = f_meds[f_meds["category_id"].str.upper() == target_cat]
        valid_med_ids = set(f_meds["medicine_id"])
        f_sales = f_sales[f_sales["medicine_id"].isin(valid_med_ids)]
        f_inv = f_inv[f_inv["medicine_id"].isin(valid_med_ids)]
        f_batches = f_batches[f_batches["medicine_id"].isin(valid_med_ids)]
    else:
        valid_med_ids = set(f_meds["medicine_id"])

    # 2. Filter by Time Range
    min_date = None
    if time_range == "12M":
        min_date = "2026-01-01"
    elif time_range == "6M":
        min_date = "2026-07-01"
    elif time_range == "30D":
        min_date = "2026-12-01"

    if min_date:
        f_sales = f_sales[f_sales["date"] >= min_date]
        f_inv = f_inv[f_inv["date"] >= min_date]
        f_batches = f_batches[f_batches["expiry_date"] >= min_date]

    # 3. Dynamic KPIs
    rev_pkr = float(f_sales["revenue"].sum()) if "revenue" in f_sales.columns else 0.0
    cogs_pkr = float(f_sales["cost_of_goods_sold"].sum()) if "cost_of_goods_sold" in f_sales.columns else 0.0
    profit_pkr = rev_pkr - cogs_pkr
    gross_margin_pct = round(profit_pkr / max(rev_pkr, 1.0) * 100.0, 1)

    units_sold = int(f_sales["units_sold"].sum()) if "units_sold" in f_sales.columns else 0
    actual_demand = int(f_sales["actual_demand"].sum()) if "actual_demand" in f_sales.columns else units_sold
    fulfillment_rate_pct = round(units_sold / max(actual_demand, 1) * 100.0, 1)

    expired_batches_count = int((f_batches["batch_status"] == "EXPIRED").sum())
    expired_units = int(f_inv["expired_quantity"].sum()) if "expired_quantity" in f_inv.columns else 0
    expired_loss_pct = round(expired_units / max(units_sold + expired_units, 1) * 100.0, 2)

    stockout_days = int(f_inv["stockout_flag"].sum()) if "stockout_flag" in f_inv.columns else 0
    stockout_pct = round(float(f_inv["stockout_flag"].mean() * 100.0), 2) if len(f_inv) > 0 else 0.0

    kpis = {
        "active_medicines": len(f_meds),
        "total_categories": len(df_cats),
        "selected_category_id": category_id,
        "selected_category_name": selected_category_name,
        "time_range": time_range,
        "gross_revenue_pkr": round(rev_pkr, 2),
        "gross_revenue_m": round(rev_pkr / 1e6, 2),
        "gross_profit_pkr": round(profit_pkr, 2),
        "gross_profit_m": round(profit_pkr / 1e6, 2),
        "gross_margin_pct": gross_margin_pct,
        "units_sold": units_sold,
        "actual_demand": actual_demand,
        "fulfillment_rate_pct": fulfillment_rate_pct,
        "batches_tracked": len(f_batches),
        "expired_batches": expired_batches_count,
        "expired_units": expired_units,
        "expired_loss_pct": expired_loss_pct,
        "stockout_days": stockout_days,
        "stockout_pct": stockout_pct,
        "audit_status": val_report.get("overall_status", "PASS")
    }

    # 4. Dispensing Velocity Trajectory (Monthly or Daily for 30D)
    monthly_data = []
    if time_range == "30D":
        # Group by day for fine-grained 30-day velocity
        daily = f_sales.groupby("date").agg(
            units=("actual_demand", "sum"),
            revenue_pkr=("revenue", "sum")
        ).reset_index().sort_values("date")

        for _, row in daily.iterrows():
            monthly_data.append({
                "month": str(row["date"]),
                "units": int(row["units"]),
                "revenue_k": round(float(row["revenue_pkr"]) / 1000.0, 1)
            })
    else:
        # Group by month
        f_sales["month_period"] = pd.to_datetime(f_sales["date"]).dt.to_period("M").astype(str)
        monthly = f_sales.groupby("month_period").agg(
            units=("actual_demand", "sum"),
            revenue_pkr=("revenue", "sum")
        ).reset_index().sort_values("month_period")

        for _, row in monthly.iterrows():
            monthly_data.append({
                "month": str(row["month_period"]),
                "units": int(row["units"]),
                "revenue_k": round(float(row["revenue_pkr"]) / 1000.0, 1)
            })

    # 5. Category Breakdown / Top Medicines Breakdown
    med_cat_map = df_meds.set_index("medicine_id")["category_id"].to_dict()
    f_sales["category_id"] = f_sales["medicine_id"].map(med_cat_map)
    total_filtered_units = max(units_sold, 1)

    categories_data = []
    if category_id.upper() == "ALL":
        cat_counts = df_meds["category_id"].value_counts().to_dict()
        cat_sales = f_sales.groupby("category_id").agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum")
        ).reset_index()

        for _, row in cat_sales.iterrows():
            cid = row["category_id"]
            cname = cat_dict.get(cid, cid)
            u = int(row["units_sold"])
            rev = float(row["revenue"])
            categories_data.append({
                "category_id": cid,
                "category_name": cname,
                "sku_count": int(cat_counts.get(cid, 0)),
                "units_sold": u,
                "revenue_pkr": round(rev, 2),
                "revenue_m": round(rev / 1e6, 2),
                "share_pct": round(u / total_filtered_units * 100.0, 1)
            })
        categories_data.sort(key=lambda x: x["units_sold"], reverse=True)
    else:
        # Show top medicines in this category
        med_dict = df_meds.set_index("medicine_id")["medicine_name"].to_dict()
        top_med_sales = f_sales.groupby("medicine_id").agg(
            units_sold=("units_sold", "sum"),
            revenue=("revenue", "sum")
        ).reset_index().sort_values("units_sold", ascending=False).head(6)

        for _, row in top_med_sales.iterrows():
            mid = row["medicine_id"]
            mname = med_dict.get(mid, mid)
            u = int(row["units_sold"])
            rev = float(row["revenue"])
            categories_data.append({
                "category_id": mid,
                "category_name": mname,
                "sku_count": 1,
                "units_sold": u,
                "revenue_pkr": round(rev, 2),
                "revenue_m": round(rev / 1e6, 2),
                "share_pct": round(u / total_filtered_units * 100.0, 1)
            })

    # 6. Critical Alerts from Intelligence Layer
    insights = get_pharmacy_intelligence_summary(category_id=category_id, time_range=time_range)

    return {
        "kpis": kpis,
        "available_categories": categories_meta,
        "monthly_velocity": monthly_data,
        "categories": categories_data,
        "critical_expiry_batches": insights.get("critical_expiry_batches", []),
        "urgent_stockouts": insights.get("urgent_stockout_risks", []),
        "top_growing": insights.get("top_growing_medicines", []),
        "top_declining": insights.get("top_declining_medicines", [])
    }
