"""
PharmaPulse - Stockout Feature Pipeline & Chronological Partitioning
Prepares order-time features and provides chronological order-date splitting.
Guarantees pending orders with unobserved outcomes remain censored.
"""

import os
import numpy as np
import pandas as pd

STOCKOUT_FEATURE_COLS = [
    "usable_stock_at_order", "outstanding_order_quantity_at_order",
    "historical_demand_7d", "historical_demand_14d", "historical_demand_30d",
    "average_daily_demand_7d", "average_daily_demand_30d", "demand_std_30d",
    "reorder_level", "target_stock_days", "supplier_lead_time_days",
    "supplier_reliability", "days_until_expected_delivery"
]

def load_and_prep_stockout_data(project_root=None):
    """
    Loads stockout replenishment event samples.
    Separates completed orders (supervised) from pending orders (held-out censored).
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    p_data = os.path.join(project_root, "data", "processed", "stockout_prediction_data.csv")
    df = pd.read_csv(p_data)

    df_supervised = df[df["target_stockout_before_arrival"].notna()].copy()
    df_pending = df[df["target_stockout_before_arrival"].isna()].copy()

    return df_supervised, df_pending

def get_chronological_stockout_splits(df_supervised, train_ratio=0.70, val_ratio=0.15):
    """
    Splits stockout purchase order events chronologically by order_date.
    """
    df = df_supervised.sort_values("order_date").copy()
    unique_dates = np.sort(df["order_date"].unique())
    n_dates = len(unique_dates)

    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    train_df = df[df["order_date"].isin(train_dates)].copy()
    val_df = df[df["order_date"].isin(val_dates)].copy()
    test_df = df[df["order_date"].isin(test_dates)].copy()

    split_info = {
        "train_orders": len(train_df),
        "val_orders": len(val_df),
        "test_orders": len(test_df),
        "train_date_range": (unique_dates[0], unique_dates[train_end - 1]),
        "val_date_range": (unique_dates[train_end], unique_dates[val_end - 1]),
        "test_date_range": (unique_dates[val_end], unique_dates[-1])
    }
    return train_df, val_df, test_df, split_info
