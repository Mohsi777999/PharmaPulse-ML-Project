"""
PharmaPulse - Expiry Feature Pipeline & Grouped Split Partitioning
Prepares predictive snapshot feature matrices and guarantees that all snapshots
from the same batch remain strictly within one data split to prevent leakage.
"""

import os
import numpy as np
import pandas as pd

EXPIRY_FEATURE_COLS = [
    "days_to_expiry", "initial_quantity", "remaining_batch_quantity",
    "batch_age_days", "historical_medicine_demand_7d", "historical_medicine_demand_14d",
    "historical_medicine_demand_30d", "historical_sales_7d", "historical_sales_14d",
    "historical_sales_30d", "historical_daily_demand", "stock_coverage_estimate"
]

def load_and_prep_expiry_data(project_root=None):
    """
    Loads expiry snapshots and filters to fully observed ground-truth batches.
    Batches expiring after simulation end (NaN target) are held out as unobserved.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    p_data = os.path.join(project_root, "data", "processed", "expiry_prediction_snapshots.csv")
    df_snapshots = pd.read_csv(p_data)

    # Observed samples for supervised learning
    df_supervised = df_snapshots[df_snapshots["target_unsold_flag"].notna()].copy()
    df_unobserved = df_snapshots[df_snapshots["target_unsold_flag"].isna()].copy()

    return df_supervised, df_unobserved

def get_grouped_expiry_splits(df_supervised, train_ratio=0.70, val_ratio=0.15, random_seed=42):
    """
    Performs batch-level grouped splitting to prevent leakage between multiple snapshots
    of the same physical batch across train, validation, and test splits.
    """
    unique_batches = np.sort(df_supervised["batch_id"].unique())
    np.random.seed(random_seed)
    shuffled_batches = np.random.permutation(unique_batches)

    n_batches = len(shuffled_batches)
    train_end = int(n_batches * train_ratio)
    val_end = int(n_batches * (train_ratio + val_ratio))

    train_b_set = set(shuffled_batches[:train_end])
    val_b_set = set(shuffled_batches[train_end:val_end])
    test_b_set = set(shuffled_batches[val_end:])

    train_df = df_supervised[df_supervised["batch_id"].isin(train_b_set)].copy()
    val_df = df_supervised[df_supervised["batch_id"].isin(val_b_set)].copy()
    test_df = df_supervised[df_supervised["batch_id"].isin(test_b_set)].copy()

    split_info = {
        "train_batches": len(train_b_set),
        "val_batches": len(val_b_set),
        "test_batches": len(test_b_set),
        "train_snapshots": len(train_df),
        "val_snapshots": len(val_df),
        "test_snapshots": len(test_df)
    }
    return train_df, val_df, test_df, split_info
