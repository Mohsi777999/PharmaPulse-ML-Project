"""
PharmaPulse - Forecasting Predictor
Generates multi-horizon forward forecasts (7-day, 30-day, up to 180-day) for any selected medicine.
Incorporates expanding forecast uncertainty bounds over time.
"""

import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from .features import FEATURE_COLS, engineer_forecasting_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "forecasting")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

def generate_forecast(medicine_id, horizon_days=30, model_type="champion"):
    """
    Generates forward forecast up to horizon_days for a specific medicine.
    Uses the best-performing Champion Model selected through empirical comparative analysis.
    Returns:
    forecast_df: DataFrame with [date, predicted_demand, lower_bound, upper_bound]
    historical_df: DataFrame with recent 60-day historical observations for plotting
    """
    champion_path = os.path.join(MODELS_DIR, "champion_forecasting.joblib")
    
    # Determine model artifact
    if model_type == "champion" or model_type is None:
        if os.path.exists(champion_path):
            model_path = champion_path
        else:
            model_path = os.path.join(MODELS_DIR, "linear_regression.joblib")
    elif model_type in ["linear", "linear_regression"]:
        model_path = os.path.join(MODELS_DIR, "linear_regression.joblib")
    elif model_type in ["rf", "random_forest"]:
        model_path = os.path.join(MODELS_DIR, "random_forest.joblib")
    elif model_type in ["xgboost", "xgb"]:
        model_path = os.path.join(MODELS_DIR, "xgboost_forecasting.joblib")
    else:
        model_path = champion_path if os.path.exists(champion_path) else os.path.join(MODELS_DIR, "linear_regression.joblib")

    if not os.path.exists(model_path):
        from .train import train_forecasting_models
        train_forecasting_models(PROJECT_ROOT)
        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_DIR, "champion_forecasting.joblib")

    model = joblib.load(model_path)
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_cols.joblib"))

    # Load sales data
    df_sales = pd.read_csv(os.path.join(RAW_DIR, "daily_demand_sales.csv"))
    med_sales = df_sales[df_sales["medicine_id"] == medicine_id].sort_values("date").copy()
    
    # Feature engineering for this medicine
    df_feat = engineer_forecasting_features(med_sales)
    
    # Recent history (last 60 days)
    recent_history = med_sales.tail(60)[["date", "actual_demand"]].copy()

    # Recursive / auto-regressive forward prediction
    last_known_date = pd.to_datetime(med_sales["date"].iloc[-1])
    demand_history = list(med_sales["actual_demand"].values)

    forecast_records = []
    base_std = float(np.std(demand_history[-30:])) if len(demand_history) >= 30 else 3.0

    for step in range(1, horizon_days + 1):
        fc_date = last_known_date + timedelta(days=step)
        dow = fc_date.weekday()
        dom = fc_date.day
        month = fc_date.month
        quarter = (month - 1) // 3 + 1
        is_weekend = 1 if dow in [5, 6] else 0

        sin_dow = np.sin(2 * np.pi * dow / 7.0)
        cos_dow = np.cos(2 * np.pi * dow / 7.0)
        sin_month = np.sin(2 * np.pi * (month - 1) / 12.0)
        cos_month = np.cos(2 * np.pi * (month - 1) / 12.0)

        # Lags from historical + forecasted series
        l1 = demand_history[-1]
        l7 = demand_history[-7] if len(demand_history) >= 7 else demand_history[-1]
        l14 = demand_history[-14] if len(demand_history) >= 14 else demand_history[-1]
        l28 = demand_history[-28] if len(demand_history) >= 28 else demand_history[-1]

        rm7 = float(np.mean(demand_history[-7:]))
        rm14 = float(np.mean(demand_history[-14:]))
        rm28 = float(np.mean(demand_history[-28:]))
        rstd7 = float(np.std(demand_history[-7:]))
        rstd28 = float(np.std(demand_history[-28:]))

        row_dict = {
            "day_of_week": dow, "day_of_month": dom, "month": month, "quarter": quarter, "is_weekend": is_weekend,
            "sin_dow": sin_dow, "cos_dow": cos_dow, "sin_month": sin_month, "cos_month": cos_month,
            "lag_1": l1, "lag_7": l7, "lag_14": l14, "lag_28": l28,
            "rolling_mean_7": rm7, "rolling_mean_14": rm14, "rolling_mean_28": rm28,
            "rolling_std_7": rstd7, "rolling_std_28": rstd28
        }
        X_step = pd.DataFrame([row_dict])[feature_cols]
        pred_val = float(model.predict(X_step)[0])
        pred_val = max(0.0, pred_val)

        # Expanding confidence bounds: uncertainty grows with sqrt(horizon)
        uncertainty = 1.28 * base_std * np.sqrt(step / 7.0)
        lower_b = max(0.0, pred_val - uncertainty)
        upper_b = pred_val + uncertainty

        forecast_records.append({
            "date": fc_date.strftime("%Y-%m-%d"),
            "predicted_demand": round(pred_val, 1),
            "lower_bound": round(lower_b, 1),
            "upper_bound": round(upper_b, 1)
        })

        demand_history.append(pred_val)

    return pd.DataFrame(forecast_records), recent_history
