"""
PharmaPulse - Integrated Intelligence Layer
Combines predictions and historical signals across Demand Forecasting, Expiry Risk,
Stockout Prediction, and Sales Clustering into actionable pharmacy-level decision support alerts.
"""

import os
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def get_pharmacy_intelligence_summary(category_id: str = "ALL", time_range: str = "ALL"):
    """
    Synthesizes cross-module intelligence for the active pharmacy branch:
    1. Demand trajectory anomalies and growth leaders
    2. Critical batch expiry risks with PKR financial exposure
    3. Urgent stockout warnings and replenishment vulnerabilities
    4. SKU behavioural segment recommendations
    Supports optional category_id filtering.
    """
    df_meds = pd.read_csv(os.path.join(RAW_DIR, "medicines.csv"))
    df_inv = pd.read_csv(os.path.join(RAW_DIR, "daily_inventory.csv"))
    df_sales = pd.read_csv(os.path.join(RAW_DIR, "daily_demand_sales.csv"))
    df_stockout = pd.read_csv(os.path.join(PROCESSED_DIR, "stockout_prediction_data.csv"))
    df_snapshots = pd.read_csv(os.path.join(PROCESSED_DIR, "expiry_prediction_snapshots.csv"))
    df_cats = pd.read_csv(os.path.join(RAW_DIR, "categories.csv"))
    cat_dict = df_cats.set_index("category_id")["category_name"].to_dict()

    # Category filter if provided
    if category_id and category_id.upper() != "ALL":
        target_cat = category_id.upper()
        df_meds = df_meds[df_meds["category_id"].str.upper() == target_cat].copy()
        valid_med_ids = set(df_meds["medicine_id"])
        df_sales = df_sales[df_sales["medicine_id"].isin(valid_med_ids)].copy()
        df_inv = df_inv[df_inv["medicine_id"].isin(valid_med_ids)].copy()
        df_stockout = df_stockout[df_stockout["medicine_id"].isin(valid_med_ids)].copy()
        df_snapshots = df_snapshots[df_snapshots["medicine_id"].isin(valid_med_ids)].copy()

    med_dict = df_meds.set_index("medicine_id").to_dict(orient="index")

    # ── 1. Demand Intelligence: Growth vs Decline ────────────────────────────
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    all_dates = sorted(df_sales["date"].unique())
    
    top_growing = []
    top_declining = []

    if len(all_dates) >= 60:
        last_30_dates = set(all_dates[-30:])
        prev_30_dates = set(all_dates[-60:-30])

        sales_last_30 = df_sales[df_sales["date"].isin(last_30_dates)].groupby("medicine_id")["actual_demand"].sum()
        sales_prev_30 = df_sales[df_sales["date"].isin(prev_30_dates)].groupby("medicine_id")["actual_demand"].sum()

        all_candidate_meds = sales_last_30.index.union(sales_prev_30.index)
        sales_last_30 = sales_last_30.reindex(all_candidate_meds, fill_value=0)
        sales_prev_30 = sales_prev_30.reindex(all_candidate_meds, fill_value=0)

        growth_ratio = (sales_last_30 - sales_prev_30) / np.maximum(sales_prev_30, 1.0)

        # Gainers
        gainers = growth_ratio[growth_ratio > 0].nlargest(5).reset_index()
        gainers.columns = ["medicine_id", "growth_ratio"]
        gainers["medicine_name"] = gainers["medicine_id"].map(
            lambda m: med_dict.get(m, {}).get("medicine_name", m)
        )
        gainers["growth_pct"] = (gainers["growth_ratio"] * 100.0).round(1)
        top_growing = gainers.to_dict(orient="records")

        # Decliners
        decliners = growth_ratio[growth_ratio < 0].nsmallest(5).reset_index()
        decliners.columns = ["medicine_id", "decline_ratio"]
        decliners["medicine_name"] = decliners["medicine_id"].map(
            lambda m: med_dict.get(m, {}).get("medicine_name", m)
        )
        decliners["decline_pct"] = (decliners["decline_ratio"] * 100.0).round(1)
        top_declining = decliners.to_dict(orient="records")

    # ── 2. Expiry Intelligence & Financial Exposure ──────────────────────────
    near_expiry = df_snapshots[df_snapshots["days_to_expiry"] <= 30].copy()
    if not near_expiry.empty:
        near_expiry["unit_cost"] = near_expiry["medicine_id"].map(
            lambda m: med_dict.get(m, {}).get("unit_cost", 0)
        )
        near_expiry["medicine_name"] = near_expiry["medicine_id"].map(
            lambda m: med_dict.get(m, {}).get("medicine_name", m)
        )
        near_expiry["category_id"] = near_expiry["medicine_id"].map(
            lambda m: med_dict.get(m, {}).get("category_id", "")
        )
        near_expiry["category_name"] = near_expiry["category_id"].map(
            lambda c: cat_dict.get(c, c)
        )

        # Financial exposure: estimated unsold units * cost
        est_unsold = near_expiry["target_unsold_quantity"].fillna(
            near_expiry["remaining_batch_quantity"] * 0.25
        )
        near_expiry["financial_exposure"] = (est_unsold * near_expiry["unit_cost"]).round(2)

        # Filter High & Critical risk
        crit_candidates = near_expiry[
            near_expiry["target_expiry_risk_class"].isin(["High", "Critical"])
        ].copy()

        # Deduplicate on batch_id keeping the highest exposure / lowest days_to_expiry
        crit_sorted = crit_candidates.sort_values(
            ["financial_exposure", "days_to_expiry"], ascending=[False, True]
        )
        deduped_batches = crit_sorted.drop_duplicates(subset=["batch_id"]).head(10)
        
        critical_batches = deduped_batches[[
            "batch_id", "medicine_id", "medicine_name", "category_name",
            "days_to_expiry", "remaining_batch_quantity",
            "target_expiry_risk_class", "financial_exposure"
        ]].to_dict(orient="records")
        total_expiry_exposure = float(deduped_batches["financial_exposure"].sum())
    else:
        critical_batches = []
        total_expiry_exposure = 0.0

    # ── 3. Stockout Intelligence: Replenishment Alerts ───────────────────────
    # Look across ALL high-risk orders in dataset
    high_risk_pos = df_stockout[df_stockout["target_stockout_before_arrival"] == 1.0].copy()
    urgent_stockout_list = []

    if not high_risk_pos.empty:
        high_risk_pos["medicine_name"] = high_risk_pos["medicine_id"].map(
            lambda m: med_dict.get(m, {}).get("medicine_name", m)
        )
        high_risk_pos["category_id"] = high_risk_pos["medicine_id"].map(
            lambda m: med_dict.get(m, {}).get("category_id", "")
        )
        high_risk_pos["category_name"] = high_risk_pos["category_id"].map(
            lambda c: cat_dict.get(c, c)
        )

        # Sort by latest order date
        if "order_date" in high_risk_pos.columns:
            high_risk_pos["order_date"] = pd.to_datetime(high_risk_pos["order_date"])
            high_risk_pos = high_risk_pos.sort_values("order_date", ascending=False)

        # Deduplicate on medicine_id to get diverse alerts
        deduped_pos = high_risk_pos.drop_duplicates(subset=["medicine_id"]).head(8)

        for _, row in deduped_pos.iterrows():
            avg_demand = max(float(row.get("average_daily_demand_7d", 0.1)), 0.1)
            usable_stock = float(row.get("usable_stock_at_order", 0))
            lead_time = int(row.get("supplier_lead_time_days", 10))
            days_of_supply = round(usable_stock / avg_demand, 1)
            deficit_days = round(lead_time - days_of_supply, 1)

            urgency_status = "CRITICAL" if days_of_supply <= 2.0 else "EXPEDITE PO"

            urgent_stockout_list.append({
                "purchase_order_id": str(row.get("purchase_order_id", "PO-N/A")),
                "medicine_id": str(row.get("medicine_id", "")),
                "medicine_name": str(row.get("medicine_name", "")),
                "category_name": str(row.get("category_name", "")),
                "days_of_supply": float(days_of_supply),
                "expected_lead_time_days": int(lead_time),
                "usable_stock_at_order": int(usable_stock),
                "supplier_id": str(row.get("supplier_id", "SUP-N/A")),
                "urgency_status": urgency_status,
                "deficit_days": float(deficit_days)
            })

    # Closing stock coverage on final day
    df_inv["date"] = pd.to_datetime(df_inv["date"])
    last_date = df_inv["date"].max()
    final_day_inv = df_inv[df_inv["date"] == last_date].copy()
    final_day_inv["medicine_name"] = final_day_inv["medicine_id"].map(
        lambda m: med_dict.get(m, {}).get("medicine_name", m)
    )
    low_coverage_meds = final_day_inv.sort_values("days_of_inventory_remaining").head(8)

    # ── 4. Behavioral Clustering Profile ────────────────────────────────────
    cluster_profiles_path = os.path.join(MODELS_DIR, "clustering", "cluster_profiles.joblib")
    cluster_profiles = joblib.load(cluster_profiles_path) if os.path.exists(cluster_profiles_path) else {}

    return {
        "top_growing_medicines": top_growing,
        "top_declining_medicines": top_declining,
        "critical_expiry_batches": critical_batches,
        "total_critical_exposure_pkr": round(total_expiry_exposure, 2),
        "urgent_stockout_risks": urgent_stockout_list,
        "high_risk_stockouts": urgent_stockout_list,
        "low_coverage_inventory": low_coverage_meds[[
            "medicine_id", "medicine_name", "closing_stock", "days_of_inventory_remaining"
        ]].to_dict(orient="records"),
        "cluster_profiles": cluster_profiles
    }
