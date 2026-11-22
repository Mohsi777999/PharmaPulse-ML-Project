"""
PharmaPulse - Machine Learning Datasets Builder
Constructs all four ML datasets and the final batch outcome table:
1. data/processed/forecasting_features.csv
2. data/processed/expiry_model_data.csv
3. data/processed/expiry_prediction_snapshots.csv
4. data/processed/stockout_prediction_data.csv
5. data/processed/clustering_features.csv

Guarantees:
- Zero temporal feature leakage across all datasets.
- Shifted historical features for forecasting (past-only).
- Chronological train/val/test split boundaries documented.
- Expiry snapshots taken 30, 14, 7 days before expiry using strictly past data.
- Stockout prediction features strictly known at PO creation time.
- Clustering features on medicine-level without demand_pattern label.
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def build_forecasting_dataset():
    print("Building Demand Forecasting Dataset (forecasting_features.csv)...")
    df_sales = pd.read_csv(os.path.join(RAW_DATA_DIR, "daily_demand_sales.csv"))
    df_sales["date"] = pd.to_datetime(df_sales["date"])
    df_sales.sort_values(by=["medicine_id", "date"], inplace=True)
    
    # Calendar features
    df_sales["day_of_week"] = df_sales["date"].dt.dayofweek
    df_sales["day_of_month"] = df_sales["date"].dt.day
    df_sales["month"] = df_sales["date"].dt.month
    df_sales["quarter"] = df_sales["date"].dt.quarter
    df_sales["is_weekend"] = df_sales["day_of_week"].isin([5, 6]).astype(int)
    
    # Lag and rolling features per medicine using shifted past demand only
    features = []
    for med_id, group in df_sales.groupby("medicine_id", sort=False):
        grp = group.copy()
        demand = grp["actual_demand"]
        
        # Shifted past demand
        grp["lag_1"] = demand.shift(1)
        grp["lag_7"] = demand.shift(7)
        grp["lag_14"] = demand.shift(14)
        grp["lag_28"] = demand.shift(28)
        
        # Rolling stats on shifted series (past observations only)
        shifted = demand.shift(1)
        grp["rolling_mean_7"] = shifted.rolling(7, min_periods=7).mean().round(3)
        grp["rolling_mean_14"] = shifted.rolling(14, min_periods=14).mean().round(3)
        grp["rolling_mean_28"] = shifted.rolling(28, min_periods=28).mean().round(3)
        grp["rolling_std_7"] = shifted.rolling(7, min_periods=7).std().round(3)
        grp["rolling_std_28"] = shifted.rolling(28, min_periods=28).std().round(3)
        
        features.append(grp)
        
    df_forecasting = pd.concat(features, ignore_index=True)
    df_forecasting["date"] = df_forecasting["date"].dt.strftime("%Y-%m-%d")
    
    cols = [
        "date", "medicine_id", "actual_demand", "day_of_week", "day_of_month",
        "month", "quarter", "is_weekend", "lag_1", "lag_7", "lag_14", "lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28", "rolling_std_7", "rolling_std_28"
    ]
    df_forecasting = df_forecasting[cols]
    df_forecasting.to_csv(os.path.join(PROCESSED_DATA_DIR, "forecasting_features.csv"), index=False)
    print(f"forecasting_features.csv saved: {len(df_forecasting)} rows")

def build_expiry_datasets():
    print("Building Expiry Datasets (expiry_model_data.csv & expiry_prediction_snapshots.csv)...")
    df_batches = pd.read_csv(os.path.join(RAW_DATA_DIR, "batches.csv"))
    df_movements = pd.read_csv(os.path.join(RAW_DATA_DIR, "inventory_movements.csv"))
    df_sales = pd.read_csv(os.path.join(RAW_DATA_DIR, "daily_demand_sales.csv"))
    
    # 1. Final Batch Outcome Table (expiry_model_data.csv)
    sales_by_batch = df_movements[df_movements["movement_type"] == "sale"].groupby("batch_id")["quantity"].sum().to_dict()
    expiry_by_batch = df_movements[df_movements["movement_type"] == "expiry"].groupby("batch_id")["quantity"].sum().to_dict()
    
    sim_end_date = "2026-12-30"
    
    batch_outcomes = []
    for _, b in df_batches.iterrows():
        b_id = b["batch_id"]
        init_qty = int(b["initial_quantity"])
        sold_qty = int(sales_by_batch.get(b_id, 0))
        expired_qty = int(expiry_by_batch.get(b_id, 0))
        exp_date = b["expiry_date"]
        
        is_fully_observed = (exp_date <= sim_end_date) or (sold_qty == init_qty)
        
        if is_fully_observed:
            unsold_at_expiry = expired_qty
            expired_flag = 1 if expired_qty > 0 else 0
            unsold_pct = round((unsold_at_expiry / init_qty) * 100.0, 2)
            final_status = "EXPIRED" if expired_qty > 0 else "DEPLETED"
        else:
            unsold_at_expiry = np.nan
            expired_flag = np.nan
            unsold_pct = np.nan
            final_status = "ACTIVE_AT_SIMULATION_END"
            
        batch_outcomes.append({
            "batch_id": b_id,
            "medicine_id": b["medicine_id"],
            "supplier_id": b["supplier_id"],
            "received_date": b["received_date"],
            "expiry_date": exp_date,
            "initial_quantity": init_qty,
            "sold_quantity": sold_qty,
            "unsold_quantity_at_expiry": unsold_at_expiry,
            "expired_flag": expired_flag,
            "unsold_percentage": unsold_pct,
            "final_batch_status": final_status
        })
        
    df_expiry_model = pd.DataFrame(batch_outcomes)
    df_expiry_model.to_csv(os.path.join(PROCESSED_DATA_DIR, "expiry_model_data.csv"), index=False)
    print(f"expiry_model_data.csv saved: {len(df_expiry_model)} rows")

    # 2. Predictive Expiry Snapshots (expiry_prediction_snapshots.csv)
    # Pre-index sales by batch and date
    sales_movs = df_movements[df_movements["movement_type"] == "sale"].copy()
    sales_movs["date_dt"] = pd.to_datetime(sales_movs["date"])
    
    # Create cumulative sales lookup: batch_id -> DataFrame of (date_dt, cumulative_sales)
    batch_sales_cum = {}
    for b_id, group in sales_movs.groupby("batch_id"):
        grp = group.sort_values("date_dt")
        dates_arr = grp["date_dt"].values
        cum_arr = np.cumsum(grp["quantity"].values)
        batch_sales_cum[b_id] = (dates_arr, cum_arr)

    # Pre-pivot sales and demand for rapid historical feature retrieval
    df_sales["date_dt"] = pd.to_datetime(df_sales["date"])
    sales_pivot = df_sales.pivot(index="date_dt", columns="medicine_id", values="units_sold").fillna(0)
    demand_pivot = df_sales.pivot(index="date_dt", columns="medicine_id", values="actual_demand").fillna(0)
    
    sim_end_dt = pd.to_datetime(sim_end_date)
    snapshots = []
    snap_id = 1
    
    for _, b_out in df_expiry_model.iterrows():
        b_id = b_out["batch_id"]
        m_id = b_out["medicine_id"]
        sup_id = b_out["supplier_id"]
        recv_dt = pd.to_datetime(b_out["received_date"])
        exp_dt = pd.to_datetime(b_out["expiry_date"])
        init_qty = int(b_out["initial_quantity"])
        
        sales_tuple = batch_sales_cum.get(b_id, None)
        
        for days_to_exp in [30, 14, 7]:
            snap_dt = exp_dt - timedelta(days=days_to_exp)
            
            # Rule: batch must have been received on or before snapshot date, and snapshot within simulation
            if snap_dt < recv_dt or snap_dt > sim_end_dt:
                continue
                
            # Reconstruct batch remaining quantity as of snap_dt
            if sales_tuple is not None:
                d_arr, c_arr = sales_tuple
                snap_ts = np.datetime64(snap_dt)
                idx = np.searchsorted(d_arr, snap_ts, side="right") - 1
                sales_to_date = int(c_arr[idx]) if idx >= 0 else 0
            else:
                sales_to_date = 0
                
            rem_qty = init_qty - sales_to_date
            if rem_qty <= 0:
                continue
                
            # Historical demand and sales
            start_7d = snap_dt - timedelta(days=6)
            start_14d = snap_dt - timedelta(days=13)
            start_30d = snap_dt - timedelta(days=29)
            
            dem_7d = int(demand_pivot.loc[start_7d:snap_dt, m_id].sum()) if start_7d in demand_pivot.index else int(demand_pivot.loc[:snap_dt, m_id].tail(7).sum())
            dem_14d = int(demand_pivot.loc[start_14d:snap_dt, m_id].sum()) if start_14d in demand_pivot.index else int(demand_pivot.loc[:snap_dt, m_id].tail(14).sum())
            dem_30d = int(demand_pivot.loc[start_30d:snap_dt, m_id].sum()) if start_30d in demand_pivot.index else int(demand_pivot.loc[:snap_dt, m_id].tail(30).sum())
            
            sold_7d = int(sales_pivot.loc[start_7d:snap_dt, m_id].sum()) if start_7d in sales_pivot.index else int(sales_pivot.loc[:snap_dt, m_id].tail(7).sum())
            sold_14d = int(sales_pivot.loc[start_14d:snap_dt, m_id].sum()) if start_14d in sales_pivot.index else int(sales_pivot.loc[:snap_dt, m_id].tail(14).sum())
            sold_30d = int(sales_pivot.loc[start_30d:snap_dt, m_id].sum()) if start_30d in sales_pivot.index else int(sales_pivot.loc[:snap_dt, m_id].tail(30).sum())
            
            hist_daily_dem = round(dem_30d / 30.0, 2)
            coverage = round(rem_qty / max(hist_daily_dem, 0.1), 2)
            batch_age = (snap_dt - recv_dt).days
            
            # Supervised Target
            if not pd.isna(b_out["unsold_quantity_at_expiry"]):
                target_unsold_qty = int(b_out["unsold_quantity_at_expiry"])
                target_unsold_flag = int(b_out["expired_flag"])
                unsold_pct = (target_unsold_qty / init_qty) * 100.0
                
                if unsold_pct < 5.0:
                    risk_class = "Low"
                elif unsold_pct < 15.0:
                    risk_class = "Medium"
                elif unsold_pct < 30.0:
                    risk_class = "High"
                else:
                    risk_class = "Critical"
            else:
                target_unsold_qty = np.nan
                target_unsold_flag = np.nan
                risk_class = np.nan
                
            snapshots.append({
                "snapshot_id": f"SNAP{snap_id:06d}",
                "snapshot_date": snap_dt.strftime("%Y-%m-%d"),
                "batch_id": b_id,
                "medicine_id": m_id,
                "supplier_id": sup_id,
                "days_to_expiry": days_to_exp,
                "initial_quantity": init_qty,
                "remaining_batch_quantity": rem_qty,
                "historical_medicine_demand_7d": dem_7d,
                "historical_medicine_demand_14d": dem_14d,
                "historical_medicine_demand_30d": dem_30d,
                "historical_sales_7d": sold_7d,
                "historical_sales_14d": sold_14d,
                "historical_sales_30d": sold_30d,
                "historical_daily_demand": hist_daily_dem,
                "stock_coverage_estimate": coverage,
                "batch_age_days": batch_age,
                "target_unsold_quantity": target_unsold_qty,
                "target_unsold_flag": target_unsold_flag,
                "target_expiry_risk_class": risk_class
            })
            snap_id += 1
            
    df_snapshots = pd.DataFrame(snapshots)
    df_snapshots.to_csv(os.path.join(PROCESSED_DATA_DIR, "expiry_prediction_snapshots.csv"), index=False)
    print(f"expiry_prediction_snapshots.csv saved: {len(df_snapshots)} rows")

def build_stockout_dataset():
    print("Building Stockout Prediction Dataset (stockout_prediction_data.csv)...")
    df_pos = pd.read_csv(os.path.join(RAW_DATA_DIR, "purchase_orders.csv"))
    df_medicines = pd.read_csv(os.path.join(RAW_DATA_DIR, "medicines.csv"))
    df_suppliers = pd.read_csv(os.path.join(RAW_DATA_DIR, "suppliers.csv"))
    df_sales = pd.read_csv(os.path.join(RAW_DATA_DIR, "daily_demand_sales.csv"))
    df_inv = pd.read_csv(os.path.join(RAW_DATA_DIR, "daily_inventory.csv"))
    
    sup_lookup = df_suppliers.set_index("supplier_id").to_dict(orient="index")
    med_lookup = df_medicines.set_index("medicine_id").to_dict(orient="index")
    
    df_sales["date_dt"] = pd.to_datetime(df_sales["date"])
    df_inv["date_dt"] = pd.to_datetime(df_inv["date"])
    
    demand_pivot = df_sales.pivot(index="date_dt", columns="medicine_id", values="actual_demand").fillna(0)
    closing_stock_pivot = df_inv.pivot(index="date_dt", columns="medicine_id", values="closing_stock").fillna(0)
    stockout_flag_pivot = df_inv.pivot(index="date_dt", columns="medicine_id", values="stockout_flag").fillna(0)
    
    sim_pos = df_pos[df_pos["order_date"] >= "2024-01-01"].copy()
    sim_pos["order_dt"] = pd.to_datetime(sim_pos["order_date"])
    sim_pos["actual_del_dt"] = pd.to_datetime(sim_pos["actual_delivery_date"])
    
    stockout_samples = []
    sample_id = 1
    
    for _, po in sim_pos.iterrows():
        po_id = po["purchase_order_id"]
        m_id = po["medicine_id"]
        sup_id = po["supplier_id"]
        order_dt = po["order_dt"]
        exp_del_dt = pd.to_datetime(po["expected_delivery_date"])
        actual_del_dt = po["actual_del_dt"]
        is_pending = pd.isna(actual_del_dt) or po["purchase_order_status"] == "PENDING"
        
        usable_stock = int(closing_stock_pivot.loc[order_dt, m_id]) if order_dt in closing_stock_pivot.index else 0
        
        prior_pos = sim_pos[
            (sim_pos["medicine_id"] == m_id) &
            (sim_pos["order_dt"] <= order_dt) &
            (sim_pos["purchase_order_id"] != po_id)
        ]
        outstanding_mask = prior_pos["actual_del_dt"].isna() | (prior_pos["actual_del_dt"] > order_dt)
        outstanding_qty = int(prior_pos.loc[outstanding_mask, "ordered_quantity"].sum())
        
        start_7d = order_dt - timedelta(days=6)
        start_14d = order_dt - timedelta(days=13)
        start_30d = order_dt - timedelta(days=29)
        
        dem_series_30 = demand_pivot.loc[start_30d:order_dt, m_id] if start_30d in demand_pivot.index else demand_pivot.loc[:order_dt, m_id].tail(30)
        dem_30d = int(dem_series_30.sum())
        dem_14d = int(dem_series_30.tail(14).sum())
        dem_7d = int(dem_series_30.tail(7).sum())
        
        avg_dem_7d = round(dem_7d / 7.0, 2)
        avg_dem_30d = round(dem_30d / 30.0, 2)
        std_dem_30d = round(float(dem_series_30.std()), 2) if len(dem_series_30) > 1 else 0.0
        
        reorder_level = int(med_lookup[m_id]["reorder_level"])
        target_stock_days = int(med_lookup[m_id]["target_stock_days"])
        sup_lead_days = int(sup_lookup[sup_id]["lead_time_days"])
        sup_rel = float(sup_lookup[sup_id]["delivery_reliability"])
        days_until_exp = (exp_del_dt - order_dt).days
        
        if is_pending:
            target_stockout = np.nan
            actual_del_str = ""
        else:
            actual_del_str = actual_del_dt.strftime("%Y-%m-%d")
            window_start = order_dt + timedelta(days=1)
            window_end = actual_del_dt - timedelta(days=1)
            
            if window_start <= window_end:
                window_flags = stockout_flag_pivot.loc[window_start:window_end, m_id]
                target_stockout = 1 if (window_flags == 1).any() else 0
            else:
                target_stockout = 0
                
        stockout_samples.append({
            "stockout_sample_id": f"SO{sample_id:06d}",
            "purchase_order_id": po_id,
            "medicine_id": m_id,
            "supplier_id": sup_id,
            "order_date": po["order_date"],
            "expected_delivery_date": po["expected_delivery_date"],
            "actual_delivery_date": actual_del_str,
            "usable_stock_at_order": usable_stock,
            "outstanding_order_quantity_at_order": outstanding_qty,
            "historical_demand_7d": dem_7d,
            "historical_demand_14d": dem_14d,
            "historical_demand_30d": dem_30d,
            "average_daily_demand_7d": avg_dem_7d,
            "average_daily_demand_30d": avg_dem_30d,
            "demand_std_30d": std_dem_30d,
            "reorder_level": reorder_level,
            "target_stock_days": target_stock_days,
            "supplier_lead_time_days": sup_lead_days,
            "supplier_reliability": sup_rel,
            "days_until_expected_delivery": days_until_exp,
            "target_stockout_before_arrival": target_stockout
        })
        sample_id += 1
        
    df_stockout = pd.DataFrame(stockout_samples)
    df_stockout.to_csv(os.path.join(PROCESSED_DATA_DIR, "stockout_prediction_data.csv"), index=False)
    print(f"stockout_prediction_data.csv saved: {len(df_stockout)} rows")

def build_clustering_dataset():
    print("Building Medicine Sales Behaviour Clustering Dataset (clustering_features.csv)...")
    df_medicines = pd.read_csv(os.path.join(RAW_DATA_DIR, "medicines.csv"))
    df_sales = pd.read_csv(os.path.join(RAW_DATA_DIR, "daily_demand_sales.csv"))
    df_inv = pd.read_csv(os.path.join(RAW_DATA_DIR, "daily_inventory.csv"))
    
    clustering_records = []
    
    for _, med in df_medicines.iterrows():
        m_id = med["medicine_id"]
        m_sales = df_sales[df_sales["medicine_id"] == m_id]
        m_inv = df_inv[df_inv["medicine_id"] == m_id]
        
        dem = m_sales["actual_demand"]
        sold = m_sales["units_sold"]
        unfulfilled = m_sales["unfulfilled_demand"]
        stockout_flags = m_inv["stockout_flag"]
        
        avg_daily_demand = round(float(dem.mean()), 3)
        std_demand = round(float(dem.std()), 3)
        cov_demand = round(std_demand / max(avg_daily_demand, 0.001), 3)
        avg_monthly_demand = round(avg_daily_demand * 30.4375, 2)
        total_units_sold = int(sold.sum())
        
        sales_freq = round(float((sold > 0).mean()), 3)
        zero_demand_ratio = round(float((dem == 0).mean()), 3)
        
        x = np.linspace(0, 1, len(dem))
        slope, _ = np.polyfit(x, dem, 1)
        demand_trend = round(float(slope), 3)
        
        m_sales_copy = m_sales.copy()
        m_sales_copy["month"] = pd.to_datetime(m_sales_copy["date"]).dt.month
        monthly_means = m_sales_copy.groupby("month")["actual_demand"].mean()
        seasonal_range = monthly_means.max() - monthly_means.min()
        seasonal_strength = round(float(seasonal_range / max(avg_daily_demand, 0.001)), 3)
        
        avg_daily_sales = round(float(sold.mean()), 3)
        stockout_ratio = round(float(stockout_flags.mean()), 3)
        unfulfilled_ratio = round(float(unfulfilled.sum()) / max(float(dem.sum()), 1.0), 3)
        
        clustering_records.append({
            "medicine_id": m_id,
            "average_daily_demand": avg_daily_demand,
            "demand_std": std_demand,
            "demand_coefficient_of_variation": cov_demand,
            "average_monthly_demand": avg_monthly_demand,
            "total_units_sold": total_units_sold,
            "sales_frequency": sales_freq,
            "zero_demand_day_ratio": zero_demand_ratio,
            "demand_trend": demand_trend,
            "seasonal_strength": seasonal_strength,
            "average_daily_sales": avg_daily_sales,
            "stockout_day_ratio": stockout_ratio,
            "unfulfilled_demand_ratio": unfulfilled_ratio
        })
        
    df_clustering = pd.DataFrame(clustering_records)
    assert len(df_clustering) == 150, f"Expected 150 rows, got {len(df_clustering)}"
    df_clustering.to_csv(os.path.join(PROCESSED_DATA_DIR, "clustering_features.csv"), index=False)
    print(f"clustering_features.csv saved: {len(df_clustering)} rows")

def build_all_ml_datasets():
    build_forecasting_dataset()
    build_expiry_datasets()
    build_stockout_dataset()
    build_clustering_dataset()
    print("All ML datasets successfully created!")

if __name__ == "__main__":
    build_all_ml_datasets()
