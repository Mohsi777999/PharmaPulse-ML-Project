import os
import sys
import io
import pandas as pd
from fastapi import APIRouter, Response, HTTPException

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import (
    load_medicines, load_categories, load_suppliers, load_daily_demand_sales,
    load_daily_inventory, load_batches, load_purchase_orders,
    load_model_evaluations, load_model_predictions, load_dataset_summary
)
from modules.intelligence.insights import get_pharmacy_intelligence_summary

router = APIRouter(prefix="/api/reports", tags=["Operational Reports"])

REPORTS_CATALOG = [
    {"id": "demand_forecasting", "name": "1. Demand Forecasting Audit Report", "description": "Aggregated daily SKU demand velocity, variance, and forecast predictability scorecards."},
    {"id": "expiry_risk", "name": "2. Expiry Risk & Financial Exposure Report", "description": "Perishable batch registry, days-to-expiry timeline, and capital at risk in PKR."},
    {"id": "stockout_risk", "name": "3. Stockout Risk & Replenishment Pipeline Report", "description": "Active inventory depletion burn rates, vendor lead-time risks, and replenishment urgency."},
    {"id": "sales_clustering", "name": "4. Sales Behaviour Clustering & Inventory Policies", "description": "150 SKU segmentation cohorts, demand volatility (CV), and targeted reordering rules."},
    {"id": "model_evaluation", "name": "5. Comparative Machine Learning Benchmark Report", "description": "Out-of-sample holdout test partition validation metrics across all 4 production ML engines."},
    {"id": "executive_summary", "name": "6. Integrated Pharmacy Operations Intelligence Report", "description": "Executive high-level summary of financial turnover, stock availability, and mitigation playbooks."}
]

def generate_report_dataframe(report_id: str) -> pd.DataFrame:
    df_meds = load_medicines()
    df_cats = load_categories()
    cat_dict = df_cats.set_index("category_id")["category_name"].to_dict()

    if report_id == "demand_forecasting":
        df_sales = load_daily_demand_sales()
        sales_agg = df_sales.groupby("medicine_id").agg(
            total_units_sold=("actual_demand", "sum"),
            mean_daily_sales=("actual_demand", "mean"),
            std_daily_sales=("actual_demand", "std")
        ).reset_index()
        sales_agg["cv_volatility"] = (sales_agg["std_daily_sales"] / sales_agg["mean_daily_sales"].clip(lower=0.1)).round(2)
        sales_agg["medicine_name"] = sales_agg["medicine_id"].map(df_meds.set_index("medicine_id")["medicine_name"])
        sales_agg["category"] = sales_agg["medicine_id"].map(df_meds.set_index("medicine_id")["category_id"]).map(cat_dict)
        return sales_agg[["medicine_id", "medicine_name", "category", "total_units_sold", "mean_daily_sales", "cv_volatility"]]

    elif report_id == "expiry_risk":
        df_batches = load_batches()
        df_batches["medicine_name"] = df_batches["medicine_id"].map(df_meds.set_index("medicine_id")["medicine_name"])
        df_batches["financial_exposure_pkr"] = (df_batches["initial_quantity"] * df_batches["unit_cost"]).round(2)
        return df_batches[["batch_id", "medicine_id", "medicine_name", "received_date", "expiry_date", "initial_quantity", "unit_cost", "batch_status", "financial_exposure_pkr"]]

    elif report_id == "stockout_risk":
        df_inv = load_daily_inventory()
        latest_date = df_inv["date"].max()
        inv_latest = df_inv[df_inv["date"] == latest_date].copy()
        inv_latest["medicine_name"] = inv_latest["medicine_id"].map(df_meds.set_index("medicine_id")["medicine_name"])
        inv_latest["category"] = inv_latest["medicine_id"].map(df_meds.set_index("medicine_id")["category_id"]).map(cat_dict)
        return inv_latest[["date", "medicine_id", "medicine_name", "category", "closing_stock", "stockout_flag", "days_of_inventory_remaining"]]

    elif report_id == "sales_clustering":
        cluster_df_path = os.path.join(PROJECT_ROOT, "models", "clustering", "clustered_medicines.joblib")
        if os.path.exists(cluster_df_path):
            import joblib
            df_c = joblib.load(cluster_df_path)
            cluster_names = {0: "Core Staples", 1: "Seasonal & Cyclic", 2: "Erratic Spikers"}
            c_col = "cluster_id" if "cluster_id" in df_c.columns else "cluster"
            l_col = "cluster_label" if "cluster_label" in df_c.columns else "cluster_name"
            if l_col not in df_c.columns:
                df_c[l_col] = df_c[c_col].map(cluster_names)
            m_col = "average_daily_demand" if "average_daily_demand" in df_c.columns else "mean_daily_sales"
            v_col = "demand_coefficient_of_variation" if "demand_coefficient_of_variation" in df_c.columns else "cv_daily_sales"
            return df_c[["medicine_id", "medicine_name", "category_id", c_col, l_col, m_col, v_col]]
        return pd.DataFrame()

    elif report_id == "model_evaluation":
        return load_model_evaluations()

    elif report_id == "executive_summary":
        summary = load_dataset_summary()
        summary_rows = [{"Metric Name": k.replace("_", " ").title(), "Metric Value": str(v)} for k, v in summary.items()]
        return pd.DataFrame(summary_rows)

    raise HTTPException(status_code=404, detail="Report not found")

@router.get("/list")
def list_reports():
    return REPORTS_CATALOG

@router.get("/{report_id}")
def get_report(report_id: str):
    df = generate_report_dataframe(report_id)
    # Return first 100 rows preview and column schema
    preview = df.head(100).to_dict(orient="records")
    columns = list(df.columns)
    return {
        "report_id": report_id,
        "total_records": len(df),
        "columns": columns,
        "preview": preview
    }

@router.get("/{report_id}/csv")
def download_report_csv(report_id: str):
    df = generate_report_dataframe(report_id)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=pharmapulse_{report_id}.csv"}
    )
