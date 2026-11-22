"""
PharmaPulse - Forecasting Feature Pipeline
Generates lag, rolling, and calendar features with zero future leakage.
Provides chronological 70/15/15 split partitioners.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "day_of_week", "day_of_month", "month", "quarter", "is_weekend",
    "sin_dow", "cos_dow", "sin_month", "cos_month",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_28"
]

def engineer_forecasting_features(df_sales):
    """
    Transforms daily sales/demand dataframe into feature matrix with shifted past predictors.
    """
    df = df_sales.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["medicine_id", "date"], inplace=True)

    # Calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Cyclic calendar features
    df["sin_dow"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0).round(4)
    df["cos_dow"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0).round(4)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0).round(4)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0).round(4)

    # Shifted lag and rolling statistics per medicine
    processed_groups = []
    for med_id, grp in df.groupby("medicine_id", sort=False):
        g = grp.copy()
        demand = g["actual_demand"]
        g["lag_1"] = demand.shift(1)
        g["lag_7"] = demand.shift(7)
        g["lag_14"] = demand.shift(14)
        g["lag_28"] = demand.shift(28)

        shifted = demand.shift(1)
        g["rolling_mean_7"] = shifted.rolling(7, min_periods=7).mean().round(3)
        g["rolling_mean_14"] = shifted.rolling(14, min_periods=14).mean().round(3)
        g["rolling_mean_28"] = shifted.rolling(28, min_periods=28).mean().round(3)
        g["rolling_std_7"] = shifted.rolling(7, min_periods=7).std().round(3)
        g["rolling_std_28"] = shifted.rolling(28, min_periods=28).std().round(3)
        processed_groups.append(g)

    df_out = pd.concat(processed_groups, ignore_index=True)
    df_out["date_str"] = df_out["date"].dt.strftime("%Y-%m-%d")
    return df_out

def get_chronological_splits(df_features, train_ratio=0.70, val_ratio=0.15):
    """
    Partitions dataset chronologically into Train (70%), Validation (15%), and Test (15%).
    Date boundaries for 1,095 days (2024-01-01 to 2026-12-30):
    - Train: 2024-01-01 to 2026-02-05 (766 days)
    - Val:   2026-02-06 to 2026-07-19 (164 days)
    - Test:  2026-07-20 to 2026-12-30 (165 days)
    """
    df = df_features.copy()
    unique_dates = np.sort(df["date_str"].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))

    train_dates = set(unique_dates[:train_end_idx])
    val_dates = set(unique_dates[train_end_idx:val_end_idx])
    test_dates = set(unique_dates[val_end_idx:])

    train_df = df[df["date_str"].isin(train_dates)].dropna(subset=FEATURE_COLS).copy()
    val_df = df[df["date_str"].isin(val_dates)].dropna(subset=FEATURE_COLS).copy()
    test_df = df[df["date_str"].isin(test_dates)].dropna(subset=FEATURE_COLS).copy()

    split_info = {
        "train_start": unique_dates[0],
        "train_end": unique_dates[train_end_idx - 1],
        "val_start": unique_dates[train_end_idx],
        "val_end": unique_dates[val_end_idx - 1],
        "test_start": unique_dates[val_end_idx],
        "test_end": unique_dates[-1],
        "train_days": len(train_dates),
        "val_days": len(val_dates),
        "test_days": len(test_dates)
    }
    return train_df, val_df, test_df, split_info
