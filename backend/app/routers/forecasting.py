import os
import sys
import pandas as pd
import numpy as np
from fastapi import APIRouter, Query

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_medicines, load_categories, load_model_evaluations
from modules.forecasting.predict import generate_forecast

router = APIRouter(prefix="/api/forecasting", tags=["Demand Forecasting"])

@router.get("/medicines")
def get_medicines_list():
    df_meds = load_medicines()
    df_cats = load_categories()
    cat_dict = df_cats.set_index("category_id")["category_name"].to_dict()
    
    meds = []
    for _, row in df_meds.iterrows():
        meds.append({
            "medicine_id": str(row["medicine_id"]),
            "medicine_name": str(row["medicine_name"]),
            "category_id": str(row["category_id"]),
            "category_name": str(cat_dict.get(row["category_id"], row["category_id"])),
            "unit_cost": float(row["unit_cost"]),
            "selling_price": float(row["selling_price"]),
            "reorder_level": int(row.get("reorder_level", 50)),
            "demand_pattern": str(row.get("demand_pattern", "Consistent"))
        })
    return meds

import json

@router.get("/predict")
def predict_demand(
    medicine_id: str = "MED001",
    horizon_days: int = 30,
    model_type: str = "champion"
):
    df_meds = load_medicines()
    med_row = df_meds[df_meds["medicine_id"] == medicine_id]
    if med_row.empty:
        # Default fallback to first medicine
        med_row = df_meds.iloc[0:1]
        medicine_id = med_row["medicine_id"].iloc[0]

    med_info = med_row.iloc[0].to_dict()
    unit_cost = float(med_info.get("unit_cost", 25.0))
    selling_price = float(med_info.get("selling_price", 35.0))
    gross_margin_pct = round(((selling_price - unit_cost) / max(selling_price, 0.01)) * 100.0, 1)

    # Load champion info metadata if available
    champion_info_path = os.path.join(PROJECT_ROOT, "models", "forecasting", "champion_info.json")
    champion_name = "Linear Regression"
    champion_wape = 53.24
    if os.path.exists(champion_info_path):
        try:
            with open(champion_info_path, "r") as f:
                c_data = json.load(f)
                champion_name = c_data.get("champion_model", "Linear Regression")
                champion_wape = c_data.get("test_metrics", {}).get("WAPE", 53.24)
        except Exception:
            pass

    # Generate forecast using the selected champion model
    fc_df, hist_df = generate_forecast(medicine_id, horizon_days=horizon_days, model_type=model_type)

    hist_data = []
    for _, r in hist_df.iterrows():
        hist_data.append({
            "date": str(r["date"]),
            "actual_demand": float(r["actual_demand"])
        })

    fc_data = []
    total_forecast_units = 0.0
    for _, r in fc_df.iterrows():
        pred = float(r["predicted_demand"])
        total_forecast_units += pred
        fc_data.append({
            "date": str(r["date"]),
            "predicted_demand": round(pred, 1),
            "lower_bound": round(float(r["lower_bound"]), 1),
            "upper_bound": round(float(r["upper_bound"]), 1)
        })

    avg_daily_demand = round(total_forecast_units / max(horizon_days, 1), 1)
    est_revenue = round(total_forecast_units * selling_price, 2)
    safety_stock_buffer = int(np.ceil(avg_daily_demand * 5 * 1.25))

    # Benchmark comparison metrics from model_evaluations (strictly the 3 candidates)
    df_evals = load_model_evaluations()
    benchmarks = []
    if not df_evals.empty:
        eval_df = df_evals[
            (df_evals["module"].str.lower() == "forecasting") &
            (df_evals["evaluation_split"] == "Test")
        ]
        if not eval_df.empty:
            pivoted = eval_df.pivot(index="model_name", columns="metric_name", values="metric_value").reset_index()
            pivoted.columns.name = None
            for _, r in pivoted.iterrows():
                m_name = str(r.get("model_name", "N/A"))
                is_champ = (m_name == champion_name)
                benchmarks.append({
                    "model_name": m_name,
                    "wape": round(float(r.get("WAPE", 0.0) or 0.0), 2),
                    "mae": round(float(r.get("MAE", 0.0) or 0.0), 2),
                    "rmse": round(float(r.get("RMSE", 0.0) or 0.0), 2),
                    "r2": round(float(r.get("R2", 0.0) or 0.0), 4),
                    "status": "CHAMPION" if is_champ else "BENCHMARK",
                    "is_champion": is_champ
                })
            # Sort with champion first, then by lowest WAPE
            benchmarks.sort(key=lambda x: (not x["is_champion"], x["wape"]))

    return {
        "medicine": {
            "medicine_id": str(med_info["medicine_id"]),
            "medicine_name": str(med_info["medicine_name"]),
            "category_id": str(med_info.get("category_id", "CAT_01")),
            "unit_cost": unit_cost,
            "selling_price": selling_price,
            "gross_margin_pct": gross_margin_pct,
            "reorder_level": int(med_info.get("reorder_level", 50))
        },
        "champion": {
            "model_name": champion_name,
            "wape": champion_wape,
            "status": "CHAMPION",
            "selection_rationale": f"Lowest out-of-sample Test WAPE ({champion_wape}%) across 165 holdout test days among all 3 evaluated architectures."
        },
        "parameters": {
            "horizon_days": horizon_days,
            "model_used": champion_name
        },
        "summary": {
            "total_forecast_units": round(total_forecast_units, 0),
            "avg_daily_demand": avg_daily_demand,
            "estimated_revenue_pkr": est_revenue,
            "safety_stock_recommended": safety_stock_buffer
        },
        "history": hist_data,
        "forecast": fc_data,
        "benchmarks": benchmarks
    }
