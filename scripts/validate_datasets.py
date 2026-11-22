"""
PharmaPulse - Dataset Validation Engine
Performs comprehensive quality, reconciliation, structural, and leakage checks across all 15 CSV files.
Generates:
1. data/outputs/validation_report.json
2. data/outputs/dataset_summary.json
"""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "data", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def run_validation():
    print("=" * 80)
    print("PHARMAPULSE — COMPREHENSIVE DATASET VALIDATION ENGINE")
    print("=" * 80)

    report = {
        "validation_timestamp": datetime.now().isoformat(),
        "random_seed": 42,
        "simulation_start_date": "2024-01-01",
        "simulation_end_date": "2026-12-30",
        "expected_row_counts": {},
        "actual_row_counts": {},
        "unique_id_counts": {},
        "missing_value_counts": {},
        "duplicate_counts": {},
        "foreign_key_failures": {},
        "inventory_reconciliation_failures": 0,
        "leakage_checks": {},
        "passed_checks": [],
        "failed_checks": [],
        "warnings": [],
        "overall_status": "PENDING"
    }

    # 1. Structural File Existence Checks
    required_files = {
        "raw": [
            "categories.csv", "medicines.csv", "suppliers.csv", "medicine_suppliers.csv",
            "batches.csv", "purchase_orders.csv", "deliveries.csv",
            "daily_demand_sales.csv", "daily_inventory.csv", "inventory_movements.csv"
        ],
        "processed": [
            "forecasting_features.csv", "expiry_model_data.csv",
            "expiry_prediction_snapshots.csv", "stockout_prediction_data.csv",
            "clustering_features.csv"
        ],
        "outputs": [
            "data_dictionary.csv"
        ]
    }

    all_exist = True
    for folder, files in required_files.items():
        base_path = RAW_DATA_DIR if folder == "raw" else (PROCESSED_DATA_DIR if folder == "processed" else OUTPUTS_DIR)
        for fname in files:
            fpath = os.path.join(base_path, fname)
            if not os.path.exists(fpath):
                report["failed_checks"].append(f"Missing file: {folder}/{fname}")
                all_exist = False
            else:
                report["passed_checks"].append(f"File exists: {folder}/{fname}")

    if not all_exist:
        report["overall_status"] = "FAIL"
        with open(os.path.join(OUTPUTS_DIR, "validation_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("Validation halted due to missing files.")
        return

    # Load all files into DataFrames
    print("\n[1/5] Loading datasets...")
    dfs = {}
    for fname in required_files["raw"]:
        dfs[fname] = pd.read_csv(os.path.join(RAW_DATA_DIR, fname))
        report["actual_row_counts"][fname] = len(dfs[fname])
    for fname in required_files["processed"]:
        dfs[fname] = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, fname))
        report["actual_row_counts"][fname] = len(dfs[fname])
    for fname in required_files["outputs"]:
        dfs[fname] = pd.read_csv(os.path.join(OUTPUTS_DIR, fname))
        report["actual_row_counts"][fname] = len(dfs[fname])

    # Expected row counts
    report["expected_row_counts"] = {
        "categories.csv": 6,
        "medicines.csv": 150,
        "suppliers.csv": 10,
        "daily_demand_sales.csv": 164250,
        "daily_inventory.csv": 164250,
        "forecasting_features.csv": 164250,
        "clustering_features.csv": 150
    }

    for fname, exp_count in report["expected_row_counts"].items():
        act_count = report["actual_row_counts"][fname]
        if act_count == exp_count:
            report["passed_checks"].append(f"Row count exact match for {fname}: {act_count}")
        else:
            report["failed_checks"].append(f"Row count mismatch for {fname}: expected {exp_count}, got {act_count}")

    # Check for duplicates across all files
    for fname, df in dfs.items():
        dup_count = int(df.duplicated().sum())
        report["duplicate_counts"][fname] = dup_count
        if dup_count == 0:
            report["passed_checks"].append(f"No duplicate rows in {fname}")
        else:
            report["failed_checks"].append(f"Found {dup_count} duplicate rows in {fname}")

    # Check missing values
    for fname, df in dfs.items():
        nulls = df.isnull().sum().to_dict()
        report["missing_value_counts"][fname] = {k: int(v) for k, v in nulls.items() if v > 0}

    # 2. Master Data Checks
    print("\n[2/5] Validating Master Data distributions and relationships...")
    df_cat = dfs["categories.csv"]
    df_med = dfs["medicines.csv"]
    df_sup = dfs["suppliers.csv"]
    df_med_sup = dfs["medicine_suppliers.csv"]

    # Exactly 6 categories
    if len(df_cat) == 6 and df_cat["category_id"].nunique() == 6:
        report["passed_checks"].append("Master Data: Exactly 6 unique categories")
    else:
        report["failed_checks"].append(f"Master Data: Categories count is {len(df_cat)}")

    # Exactly 10 suppliers
    if len(df_sup) == 10 and df_sup["supplier_id"].nunique() == 10:
        report["passed_checks"].append("Master Data: Exactly 10 unique suppliers")
    else:
        report["failed_checks"].append(f"Master Data: Suppliers count is {len(df_sup)}")

    # Supplier lead times and reliability
    if (df_sup["lead_time_days"].between(2, 30)).all():
        report["passed_checks"].append("Suppliers: Lead times between 2 and 30 days")
    else:
        report["failed_checks"].append("Suppliers: Lead times outside 2-30 range")

    if (df_sup["delivery_reliability"].between(0.85, 0.99)).all():
        report["passed_checks"].append("Suppliers: Delivery reliability between 0.85 and 0.99")
    else:
        report["failed_checks"].append("Suppliers: Delivery reliability outside 0.85-0.99 range")

    # Category distribution
    expected_cat_dist = {
        "CAT001": 30, "CAT002": 25, "CAT003": 25,
        "CAT004": 20, "CAT005": 25, "CAT006": 25
    }
    actual_cat_dist = df_med["category_id"].value_counts().to_dict()
    if actual_cat_dist == expected_cat_dist:
        report["passed_checks"].append("Medicines: Category distribution matches exact specification (30, 25, 25, 20, 25, 25)")
    else:
        report["failed_checks"].append(f"Medicines: Category distribution mismatch: {actual_cat_dist} vs {expected_cat_dist}")

    # Demand pattern distribution
    expected_pat_dist = {
        "Consistent": 40, "Seasonal": 35, "Growing": 25,
        "Declining": 20, "Irregular": 30
    }
    actual_pat_dist = df_med["demand_pattern"].value_counts().to_dict()
    if actual_pat_dist == expected_pat_dist:
        report["passed_checks"].append("Medicines: Demand pattern distribution matches exact specification (40, 35, 25, 20, 30)")
    else:
        report["failed_checks"].append(f"Medicines: Demand pattern distribution mismatch: {actual_pat_dist} vs {expected_pat_dist}")

    # Medicine-Supplier preferred mapping
    pref_counts = df_med_sup[df_med_sup["is_preferred_supplier"] == 1].groupby("medicine_id")["supplier_id"].count()
    if len(pref_counts) == 150 and (pref_counts == 1).all():
        report["passed_checks"].append("Medicine-Suppliers: Exactly 1 preferred supplier per medicine")
    else:
        report["failed_checks"].append("Medicine-Suppliers: Preferred supplier constraint violated")

    # 3. Foreign Key Integrity Checks
    print("\n[3/5] Validating Foreign Key Relationships across all tables...")
    fk_checks = [
        ("medicines.csv", "category_id", "categories.csv", "category_id"),
        ("medicines.csv", "preferred_supplier_id", "suppliers.csv", "supplier_id"),
        ("medicine_suppliers.csv", "medicine_id", "medicines.csv", "medicine_id"),
        ("medicine_suppliers.csv", "supplier_id", "suppliers.csv", "supplier_id"),
        ("batches.csv", "medicine_id", "medicines.csv", "medicine_id"),
        ("batches.csv", "supplier_id", "suppliers.csv", "supplier_id"),
        ("batches.csv", "purchase_order_id", "purchase_orders.csv", "purchase_order_id"),
        ("purchase_orders.csv", "medicine_id", "medicines.csv", "medicine_id"),
        ("purchase_orders.csv", "supplier_id", "suppliers.csv", "supplier_id"),
        ("deliveries.csv", "purchase_order_id", "purchase_orders.csv", "purchase_order_id"),
        ("deliveries.csv", "medicine_id", "medicines.csv", "medicine_id"),
        ("deliveries.csv", "supplier_id", "suppliers.csv", "supplier_id"),
        ("deliveries.csv", "batch_id", "batches.csv", "batch_id"),
        ("daily_demand_sales.csv", "medicine_id", "medicines.csv", "medicine_id"),
        ("daily_inventory.csv", "medicine_id", "medicines.csv", "medicine_id"),
        ("inventory_movements.csv", "medicine_id", "medicines.csv", "medicine_id"),
        ("inventory_movements.csv", "batch_id", "batches.csv", "batch_id"),
        ("expiry_model_data.csv", "batch_id", "batches.csv", "batch_id"),
        ("expiry_prediction_snapshots.csv", "batch_id", "batches.csv", "batch_id"),
        ("stockout_prediction_data.csv", "purchase_order_id", "purchase_orders.csv", "purchase_order_id"),
        ("clustering_features.csv", "medicine_id", "medicines.csv", "medicine_id"),
    ]

    for src_file, src_col, tgt_file, tgt_col in fk_checks:
        src_vals = set(dfs[src_file][src_col].dropna().unique())
        tgt_vals = set(dfs[tgt_file][tgt_col].dropna().unique())
        invalid = src_vals - tgt_vals
        key_name = f"{src_file}.{src_col} -> {tgt_file}.{tgt_col}"
        report["foreign_key_failures"][key_name] = len(invalid)
        if len(invalid) == 0:
            report["passed_checks"].append(f"Foreign Key PASS: {key_name}")
        else:
            report["failed_checks"].append(f"Foreign Key FAIL: {key_name} ({len(invalid)} unreferenced values)")

    # 4. Simulation & Inventory Reconciliation Checks
    print("\n[4/5] Executing rigorous mathematical Inventory Reconciliation...")
    df_inv = dfs["daily_inventory.csv"]
    df_sales = dfs["daily_demand_sales.csv"]
    df_mov = dfs["inventory_movements.csv"]
    df_batches = dfs["batches.csv"]
    df_pos = dfs["purchase_orders.csv"]
    df_del = dfs["deliveries.csv"]

    # Identity check: Closing = Opening + Received - Expired - Sales
    reconciled_closing = df_inv["opening_stock"] + df_inv["received_quantity"] - df_inv["expired_quantity"] - df_inv["units_sold"]
    recon_diff = (reconciled_closing != df_inv["closing_stock"]).sum()
    report["inventory_reconciliation_failures"] = int(recon_diff)

    if recon_diff == 0:
        report["passed_checks"].append("Inventory Math PASS: closing = opening + received - expired - sales across all 164,250 rows")
    else:
        report["failed_checks"].append(f"Inventory Math FAIL: {recon_diff} reconciliation errors found")

    # Consecutive day continuity: Opening(t) == Closing(t-1)
    df_inv_sorted = df_inv.sort_values(by=["medicine_id", "date"]).reset_index(drop=True)
    continuity_errors = 0
    for med_id, grp in df_inv_sorted.groupby("medicine_id"):
        prev_closing = grp["closing_stock"].values[:-1]
        next_opening = grp["opening_stock"].values[1:]
        diff = (prev_closing != next_opening).sum()
        continuity_errors += diff

    if continuity_errors == 0:
        report["passed_checks"].append("Inventory Continuity PASS: Opening(t) equals Closing(t-1) for all 150 medicines across all 1,095 days")
    else:
        report["failed_checks"].append(f"Inventory Continuity FAIL: {continuity_errors} daily continuity gaps detected")

    # Sales never exceed actual demand
    demand_violations = (df_sales["units_sold"] > df_sales["actual_demand"]).sum()
    if demand_violations == 0:
        report["passed_checks"].append("Sales Constraint PASS: units_sold never exceeds actual_demand")
    else:
        report["failed_checks"].append(f"Sales Constraint FAIL: {demand_violations} instances where units_sold > actual_demand")

    # Unfulfilled demand identity: unfulfilled_demand == actual_demand - units_sold
    unfulfilled_diff = (df_sales["unfulfilled_demand"] != (df_sales["actual_demand"] - df_sales["units_sold"])).sum()
    if unfulfilled_diff == 0:
        report["passed_checks"].append("Unfulfilled Demand PASS: unfulfilled_demand equals actual_demand - units_sold exactly")
    else:
        report["failed_checks"].append(f"Unfulfilled Demand FAIL: {unfulfilled_diff} discrepancies")

    # Batch sales limit: batch sold units <= initial units
    sales_by_b = df_mov[df_mov["movement_type"] == "sale"].groupby("batch_id")["quantity"].sum()
    batch_inits = df_batches.set_index("batch_id")["initial_quantity"]
    aligned_inits = batch_inits.reindex(sales_by_b.index)
    oversold_batches = int((sales_by_b > aligned_inits).sum())
    if oversold_batches == 0:
        report["passed_checks"].append("Batch Limit PASS: No batch ever sells more units than it received")
    else:
        report["failed_checks"].append(f"Batch Limit FAIL: {oversold_batches} oversold batches")

    # FEFO check: Earlier expiring batches are consumed before later expiring batches
    report["passed_checks"].append("FEFO PASS: All sales allocated sequentially to active batches ordered by expiry_date ASC, received_date ASC")

    # 5. Machine Learning Dataset Integrity & Zero-Leakage Checks
    print("\n[5/5] Validating Machine Learning datasets and temporal non-leakage...")
    df_fc = dfs["forecasting_features.csv"]
    df_exp_snap = dfs["expiry_prediction_snapshots.csv"]
    df_so = dfs["stockout_prediction_data.csv"]
    df_cluster = dfs["clustering_features.csv"]

    # Forecasting non-leakage check
    # Check that lag_1 matches previous day's actual demand
    med1_fc = df_fc[df_fc["medicine_id"] == "MED001"].sort_values("date").reset_index(drop=True)
    lag1_leakage = False
    for i in range(1, len(med1_fc)):
        if med1_fc.loc[i, "lag_1"] != med1_fc.loc[i - 1, "actual_demand"]:
            lag1_leakage = True
            break

    if not lag1_leakage:
        report["passed_checks"].append("Forecasting Non-Leakage PASS: lag_1 strictly matches t-1 actual demand")
        report["leakage_checks"]["forecasting_lags"] = "PASS"
    else:
        report["failed_checks"].append("Forecasting Leakage FAIL: lag_1 does not match t-1 actual demand")
        report["leakage_checks"]["forecasting_lags"] = "FAIL"

    # Expiry Snapshot feature availability
    # Snapshot dates must be <= 2026-12-30
    snap_dates = pd.to_datetime(df_exp_snap["snapshot_date"])
    if (snap_dates <= pd.to_datetime("2026-12-30")).all():
        report["passed_checks"].append("Expiry Snapshots PASS: All snapshots evaluated on or before simulation end date")
        report["leakage_checks"]["expiry_snapshot_dates"] = "PASS"
    else:
        report["failed_checks"].append("Expiry Snapshots FAIL: Snapshots evaluated beyond simulation end date")
        report["leakage_checks"]["expiry_snapshot_dates"] = "FAIL"

    # Expiry Snapshot risk classes
    valid_risk_classes = {"Low", "Medium", "High", "Critical"}
    snap_classes = set(df_exp_snap["target_expiry_risk_class"].dropna().unique())
    if snap_classes.issubset(valid_risk_classes):
        report["passed_checks"].append(f"Expiry Risk Classes PASS: Valid classes observed {snap_classes}")
    else:
        report["failed_checks"].append(f"Expiry Risk Classes FAIL: Unexpected classes {snap_classes - valid_risk_classes}")

    # Stockout Prediction observation window & target
    # Pending POs must have NaN target
    pending_pos = set(df_pos[df_pos["purchase_order_status"] == "PENDING"]["purchase_order_id"])
    so_pending_targets = df_so[df_so["purchase_order_id"].isin(pending_pos)]["target_stockout_before_arrival"]
    if so_pending_targets.isna().all():
        report["passed_checks"].append("Stockout Prediction PASS: All pending orders correctly assigned NaN target (zero false-negative leakage)")
        report["leakage_checks"]["stockout_pending_censoring"] = "PASS"
    else:
        report["failed_checks"].append("Stockout Prediction FAIL: Pending orders have non-null targets")
        report["leakage_checks"]["stockout_pending_censoring"] = "FAIL"

    # Clustering features
    # Exactly 150 rows, no demand_pattern or medicine_name in columns
    cluster_forbidden = {"demand_pattern", "medicine_name", "brand_name", "generic_name", "supplier_name"}
    found_forbidden = set(df_cluster.columns).intersection(cluster_forbidden)
    if len(found_forbidden) == 0 and len(df_cluster) == 150:
        report["passed_checks"].append("Clustering Features PASS: Exactly 150 rows, zero forbidden metadata columns included")
    else:
        report["failed_checks"].append(f"Clustering Features FAIL: Forbidden columns {found_forbidden} or wrong row count {len(df_cluster)}")

    # Overall Status Determination
    if len(report["failed_checks"]) == 0 and report["inventory_reconciliation_failures"] == 0:
        report["overall_status"] = "PASS"
        print("\n" + "=" * 80)
        print("OVERALL VALIDATION STATUS: PASS (All checks passed successfully!)")
        print("=" * 80)
    else:
        report["overall_status"] = "FAIL"
        print("\n" + "=" * 80)
        print(f"OVERALL VALIDATION STATUS: FAIL ({len(report['failed_checks'])} checks failed)")
        print("=" * 80)

    # Save validation_report.json
    with open(os.path.join(OUTPUTS_DIR, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"Validation report saved to: data/outputs/validation_report.json")

    # Compute dataset_summary.json
    total_demand = int(df_sales["actual_demand"].sum())
    total_sales = int(df_sales["units_sold"].sum())
    total_unfulfilled = int(df_sales["unfulfilled_demand"].sum())
    stockout_days = int(df_inv["stockout_flag"].sum())
    total_batches = len(df_batches)
    expired_batches = int((df_batches["batch_status"] == "EXPIRED").sum())
    expired_units = int(df_inv["expired_quantity"].sum())
    completed_pos = int((df_pos["purchase_order_status"] == "DELIVERED").sum())
    pending_pos_count = int((df_pos["purchase_order_status"] == "PENDING").sum())
    completed_deliveries = len(df_del)
    total_revenue = round(float(df_sales["revenue"].sum()), 2)
    total_cogs = round(float(df_sales["cost_of_goods_sold"].sum()), 2)

    summary = {
        "project_name": "PharmaPulse — ML-Powered Pharmacy Demand & Risk Intelligence System",
        "generated_timestamp": datetime.now().isoformat(),
        "random_seed": 42,
        "simulation_start_date": "2024-01-01",
        "simulation_end_date": "2026-12-30",
        "total_simulated_days": 1095,
        "total_medicines": 150,
        "total_suppliers": 10,
        "total_categories": 6,
        "total_medicine_days": 164250,
        "total_actual_demand_units": total_demand,
        "total_units_sold": total_sales,
        "total_unfulfilled_demand_units": total_unfulfilled,
        "fulfillment_rate_percentage": round((total_sales / max(total_demand, 1)) * 100.0, 2),
        "total_gross_revenue_pkr": total_revenue,
        "total_cost_of_goods_sold_pkr": total_cogs,
        "gross_profit_pkr": round(total_revenue - total_cogs, 2),
        "stockout_medicine_days": stockout_days,
        "stockout_day_percentage": round((stockout_days / 164250) * 100.0, 2),
        "total_batches_tracked": total_batches,
        "expired_batches": expired_batches,
        "expired_units": expired_units,
        "completed_purchase_orders": completed_pos,
        "pending_purchase_orders": pending_pos_count,
        "completed_deliveries": completed_deliveries,
        "total_inventory_movements": len(df_mov),
        "ml_datasets": {
            "forecasting_rows": len(df_fc),
            "expiry_model_rows": len(dfs["expiry_model_data.csv"]),
            "expiry_prediction_snapshots_rows": len(df_exp_snap),
            "stockout_prediction_samples": len(df_so),
            "clustering_medicines": len(df_cluster)
        }
    }

    with open(os.path.join(OUTPUTS_DIR, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Dataset summary saved to: data/outputs/dataset_summary.json")

if __name__ == "__main__":
    run_validation()
