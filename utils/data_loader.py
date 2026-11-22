"""
PharmaPulse - Data Loader Utility
Provides cached, centralized data loading routines for all CSV files and summaries.
Compatible with FastAPI REST application services and standalone batch script executions.
"""

import os
import sys
import json
import functools
import pandas as pd

# Centralized in-memory caching decorator for high-performance API reads
cache_decorator = functools.lru_cache(maxsize=32)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "data", "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

@cache_decorator
def load_medicines():
    return pd.read_csv(os.path.join(RAW_DIR, "medicines.csv"))

@cache_decorator
def load_categories():
    return pd.read_csv(os.path.join(RAW_DIR, "categories.csv"))

@cache_decorator
def load_suppliers():
    return pd.read_csv(os.path.join(RAW_DIR, "suppliers.csv"))

@cache_decorator
def load_medicine_suppliers():
    return pd.read_csv(os.path.join(RAW_DIR, "medicine_suppliers.csv"))

@cache_decorator
def load_batches():
    return pd.read_csv(os.path.join(RAW_DIR, "batches.csv"))

@cache_decorator
def load_purchase_orders():
    return pd.read_csv(os.path.join(RAW_DIR, "purchase_orders.csv"))

@cache_decorator
def load_deliveries():
    return pd.read_csv(os.path.join(RAW_DIR, "deliveries.csv"))

@cache_decorator
def load_daily_demand_sales():
    return pd.read_csv(os.path.join(RAW_DIR, "daily_demand_sales.csv"))

@cache_decorator
def load_daily_inventory():
    return pd.read_csv(os.path.join(RAW_DIR, "daily_inventory.csv"))

@cache_decorator
def load_inventory_movements():
    return pd.read_csv(os.path.join(RAW_DIR, "inventory_movements.csv"))

@cache_decorator
def load_forecasting_features():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "forecasting_features.csv"))

@cache_decorator
def load_expiry_model_data():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "expiry_model_data.csv"))

@cache_decorator
def load_expiry_prediction_snapshots():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "expiry_prediction_snapshots.csv"))

@cache_decorator
def load_stockout_prediction_data():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "stockout_prediction_data.csv"))

@cache_decorator
def load_clustering_features():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "clustering_features.csv"))

@cache_decorator
def load_data_dictionary():
    return pd.read_csv(os.path.join(OUTPUTS_DIR, "data_dictionary.csv"))

@cache_decorator
def load_dataset_summary():
    with open(os.path.join(OUTPUTS_DIR, "dataset_summary.json"), "r") as f:
        return json.load(f)

@cache_decorator
def load_validation_report():
    with open(os.path.join(OUTPUTS_DIR, "validation_report.json"), "r") as f:
        return json.load(f)

@cache_decorator
def load_model_evaluations():
    # Try outputs directory first, then root data directory
    p1 = os.path.join(OUTPUTS_DIR, "model_evaluations.csv")
    p2 = os.path.join(DATA_DIR, "model_evaluations.csv")
    if os.path.exists(p1):
        return pd.read_csv(p1)
    elif os.path.exists(p2):
        return pd.read_csv(p2)
    return pd.DataFrame()

@cache_decorator
def load_model_predictions():
    p1 = os.path.join(OUTPUTS_DIR, "model_predictions.csv")
    p2 = os.path.join(DATA_DIR, "model_predictions.csv")
    if os.path.exists(p1):
        return pd.read_csv(p1)
    elif os.path.exists(p2):
        return pd.read_csv(p2)
    return pd.DataFrame()
