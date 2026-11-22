"""
PharmaPulse - Clustering Feature Pipeline
=========================================
Extracts 90-day rolling behavioural demand features at SKU level.
Excludes simulator demand_pattern and metadata during feature extraction
to guarantee authentic, unsupervised machine learning discovery.
Enriches with authentic medicine economics (unit cost, selling price, gross margin).
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CLUSTERING_FEATURE_COLS = [
    "average_daily_demand", "demand_std", "demand_coefficient_of_variation",
    "average_monthly_demand", "total_units_sold", "sales_frequency",
    "zero_demand_day_ratio", "demand_trend", "seasonal_strength",
    "average_daily_sales", "stockout_day_ratio", "unfulfilled_demand_ratio"
]


def load_and_scale_clustering_features(project_root=None):
    """
    Loads clustering features from data/processed/clustering_features.csv,
    enriches with medicine catalog economics (unit cost, selling price, gross margin),
    scales numerical columns using StandardScaler, and returns scaled array + enriched dataframe.
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    p_data = os.path.join(project_root, "data", "processed", "clustering_features.csv")
    df = pd.read_csv(p_data)

    # Load complete medicine metadata and economics
    p_meds = os.path.join(project_root, "data", "raw", "medicines.csv")
    df_meds = pd.read_csv(p_meds)[["medicine_id", "medicine_name", "category_id", "unit_cost", "selling_price"]]
    
    # Calculate authentic gross margin percentage
    df_meds["unit_cost"] = df_meds["unit_cost"].astype(float)
    df_meds["selling_price"] = df_meds["selling_price"].astype(float)
    df_meds["profit_margin"] = (
        (df_meds["selling_price"] - df_meds["unit_cost"]) / df_meds["selling_price"].replace(0, 1.0)
    ).round(4)
    df_meds["profit_margin_pct"] = (df_meds["profit_margin"] * 100.0).round(1)

    df_merged = df.merge(df_meds, on="medicine_id", how="left")

    X_raw = df_merged[CLUSTERING_FEATURE_COLS].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    return df_merged, X_scaled, scaler
