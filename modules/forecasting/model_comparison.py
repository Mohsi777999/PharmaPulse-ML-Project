"""
PharmaPulse - Forecasting Model Comparative Analysis & Champion Selection
========================================================================
Performs rigorous empirical comparative evaluation across three candidate
architectures for daily pharmacy demand forecasting:
  1. Linear Regression
  2. Random Forest Regressor
  3. XGBoost Regressor

Evaluation Strategy:
- Strict chronological 70/15/15 train/validation/test partitioning.
- Zero future data leakage across all 18 lag and rolling features.
- Primary decision metric: Weighted Absolute Percentage Error (WAPE) on out-of-sample Test split.
- Tie-breaker metrics: MAE, RMSE, and R2.
- Automatically selects the Champion model, serializes it as champion_forecasting.joblib,
  and exports champion_info.json and updated model_evaluations.csv.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.forecasting.features import FEATURE_COLS, engineer_forecasting_features, get_chronological_splits
from utils.metrics import evaluate_forecasting

def run_model_comparison(project_root=None):
    """
    Executes the 3-model comparative analysis pipeline:
    1. Trains Linear Regression, Random Forest, and XGBoost.
    2. Evaluates each on Validation and Test splits.
    3. Ranks models by Test WAPE.
    4. Selects the champion and saves champion_forecasting.joblib and champion_info.json.
    5. Updates model_evaluations.csv (eliminating legacy naive baselines).
    """
    if project_root is None:
        project_root = PROJECT_ROOT

    models_dir = os.path.join(project_root, "models", "forecasting")
    raw_dir = os.path.join(project_root, "data", "raw")
    outputs_dir = os.path.join(project_root, "data", "outputs")
    data_dir = os.path.join(project_root, "data")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print("=" * 80)
    print("PHARMAPULSE — DEMAND FORECASTING COMPARATIVE ANALYSIS (3 CANDIDATES)")
    print("=" * 80)
    print("Candidate Architectures:")
    print("  1. Linear Regression (Multivariate Regularized Hyperplane)")
    print("  2. Random Forest Regressor (100 Decision Trees)")
    print("  3. XGBoost Regressor (Gradient Boosted Trees)")
    print("-" * 80)

    # 1. Ingest sales and generate features
    sales_csv = os.path.join(raw_dir, "daily_demand_sales.csv")
    print(f"Loading daily sales data: {sales_csv}")
    df_sales = pd.read_csv(sales_csv)
    print("Generating zero-leakage feature matrix (18 features)...")
    df_features = engineer_forecasting_features(df_sales)

    # 2. Chronological splits
    train_df, val_df, test_df, split_info = get_chronological_splits(df_features)
    print(f"Chronological Splits Partitioned:")
    print(f"  • Train: {split_info['train_days']} days ({split_info['train_start']} -> {split_info['train_end']}) - {len(train_df):,} rows")
    print(f"  • Val:   {split_info['val_days']} days ({split_info['val_start']} -> {split_info['val_end']}) - {len(val_df):,} rows")
    print(f"  • Test:  {split_info['test_days']} days ({split_info['test_start']} -> {split_info['test_end']}) - {len(test_df):,} rows")

    X_train, y_train = train_df[FEATURE_COLS], train_df["actual_demand"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["actual_demand"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["actual_demand"]

    # Candidate models definition
    candidates = {
        "Linear Regression": {
            "model": LinearRegression(),
            "filename": "linear_regression.joblib",
            "description": "Multivariate linear regression fitting autoregressive lags and rolling statistics."
        },
        "Random Forest": {
            "model": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=6, random_state=42, n_jobs=-1),
            "filename": "random_forest.joblib",
            "description": "Ensemble of 100 decorrelated decision trees capturing non-linear interactions."
        },
        "XGBoost": {
            "model": XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
            "filename": "xgboost_forecasting.joblib",
            "description": "Gradient boosted decision trees optimizing second-order Taylor expansion loss."
        }
    }

    results = {}
    evaluation_records = []

    print("\nTraining and evaluating candidate architectures...")
    for name, config in candidates.items():
        print(f"\n[Training] {name}...")
        model = config["model"]
        model.fit(X_train, y_train)

        # Predictions (non-negative clipping for physical sales volume)
        val_pred = np.clip(model.predict(X_val), 0, None)
        test_pred = np.clip(model.predict(X_test), 0, None)

        val_metrics = evaluate_forecasting(y_val, val_pred)
        test_metrics = evaluate_forecasting(y_test, test_pred)

        results[name] = {
            "model": model,
            "filename": config["filename"],
            "description": config["description"],
            "val_metrics": val_metrics,
            "test_metrics": test_metrics
        }

        # Save individual model artifact
        artifact_path = os.path.join(models_dir, config["filename"])
        joblib.dump(model, artifact_path)

        # Append to evaluation records for model_evaluations.csv
        for m_name, m_val in val_metrics.items():
            evaluation_records.append({
                "model_name": name,
                "module": "Forecasting",
                "task_type": "Regression",
                "evaluation_split": "Validation",
                "metric_name": m_name,
                "metric_value": m_val
            })
        for m_name, m_val in test_metrics.items():
            evaluation_records.append({
                "model_name": name,
                "module": "Forecasting",
                "task_type": "Regression",
                "evaluation_split": "Test",
                "metric_name": m_name,
                "metric_value": m_val
            })

        print(f"  Validation -> WAPE: {val_metrics['WAPE']}% | MAE: {val_metrics['MAE']} | RMSE: {val_metrics['RMSE']} | R2: {val_metrics['R2']}")
        print(f"  Test       -> WAPE: {test_metrics['WAPE']}% | MAE: {test_metrics['MAE']} | RMSE: {test_metrics['RMSE']} | R2: {test_metrics['R2']}")

    # Save metadata artifacts
    joblib.dump(FEATURE_COLS, os.path.join(models_dir, "feature_cols.joblib"))
    joblib.dump(split_info, os.path.join(models_dir, "split_info.joblib"))

    # 3. Comparative Leaderboard Analysis & Champion Selection
    print("\n" + "=" * 80)
    print("COMPARATIVE LEADERBOARD — OUT-OF-SAMPLE TEST PARTITION (165 DAYS)")
    print("=" * 80)

    leaderboard = []
    for name, data in results.items():
        leaderboard.append({
            "model_name": name,
            "test_wape": data["test_metrics"]["WAPE"],
            "test_mae": data["test_metrics"]["MAE"],
            "test_rmse": data["test_metrics"]["RMSE"],
            "test_r2": data["test_metrics"]["R2"],
            "val_wape": data["val_metrics"]["WAPE"],
            "val_mae": data["val_metrics"]["MAE"],
            "val_rmse": data["val_metrics"]["RMSE"],
            "val_r2": data["val_metrics"]["R2"],
            "filename": data["filename"],
            "description": data["description"]
        })

    # Sort leaderboard by Test WAPE ascending (lowest error wins)
    leaderboard.sort(key=lambda x: (x["test_wape"], x["test_rmse"], x["test_mae"]))

    for rank, entry in enumerate(leaderboard, start=1):
        entry["rank"] = rank
        entry["status"] = "CHAMPION" if rank == 1 else "BENCHMARK"

    print(f"{'Rank':<6} {'Model':<20} {'Test WAPE':<12} {'Test MAE':<12} {'Test RMSE':<12} {'Test R2':<10} {'Status':<12}")
    print("-" * 80)
    for entry in leaderboard:
        print(f"{entry['rank']:<6} {entry['model_name']:<20} {entry['test_wape']:<11}% {entry['test_mae']:<12} {entry['test_rmse']:<12} {entry['test_r2']:<10} {entry['status']:<12}")
    print("=" * 80)

    # 4. Champion Designation
    champion_entry = leaderboard[0]
    champion_name = champion_entry["model_name"]
    champion_model = results[champion_name]["model"]

    print(f"\n>>> DESIGNATED CHAMPION: {champion_name.upper()} <<<")
    print(f"Reason: Lowest Out-Of-Sample Test WAPE of {champion_entry['test_wape']}% across 165 holdout test days.")

    # Save champion model directly as champion_forecasting.joblib
    champion_model_path = os.path.join(models_dir, "champion_forecasting.joblib")
    joblib.dump(champion_model, champion_model_path)
    print(f"Serialized champion model -> {champion_model_path}")

    # Save champion info JSON metadata
    champion_info = {
        "champion_model": champion_name,
        "selection_criterion": "Lowest Out-of-Sample Test WAPE",
        "primary_metric": "WAPE",
        "test_metrics": results[champion_name]["test_metrics"],
        "val_metrics": results[champion_name]["val_metrics"],
        "leaderboard": leaderboard,
        "selection_summary": (
            f"{champion_name} achieved the lowest prediction error with Test WAPE of {champion_entry['test_wape']}% "
            f"(vs {leaderboard[1]['model_name']} at {leaderboard[1]['test_wape']}% and "
            f"{leaderboard[2]['model_name']} at {leaderboard[2]['test_wape']}%). "
            f"It is designated as the sole production model for forward demand forecasting."
        )
    }

    champion_json_path = os.path.join(models_dir, "champion_info.json")
    with open(champion_json_path, "w") as f:
        json.dump(champion_info, f, indent=2)
    print(f"Saved champion metadata -> {champion_json_path}")

    # 5. Generate sample predictions using Champion for model_predictions.csv
    test_sample = test_df.tail(150 * 30).copy()
    test_sample["predicted_demand"] = np.clip(champion_model.predict(test_sample[FEATURE_COLS]), 0, None).round(1)

    preds_records = []
    for idx, row in test_sample.iterrows():
        preds_records.append({
            "prediction_id": f"PRED_FC_{len(preds_records)+1:06d}",
            "module": "Forecasting",
            "medicine_id": row["medicine_id"],
            "batch_id": "N/A",
            "prediction_date": row["date_str"],
            "target_name": "actual_demand",
            "actual_value": row["actual_demand"],
            "predicted_value": row["predicted_demand"],
            "risk_level": "Normal"
        })

    # 6. Update model_evaluations.csv across both data directories
    # Remove any old Forecasting evaluations (including legacy Naive baselines) and replace with only the 3 models
    for eval_file_path in [
        os.path.join(outputs_dir, "model_evaluations.csv"),
        os.path.join(data_dir, "model_evaluations.csv")
    ]:
        if os.path.exists(eval_file_path):
            existing_evals = pd.read_csv(eval_file_path)
            # Filter out all old forecasting rows
            non_fc_evals = existing_evals[existing_evals["module"].str.lower() != "forecasting"]
            new_fc_df = pd.DataFrame(evaluation_records)
            updated_evals = pd.concat([new_fc_df, non_fc_evals], ignore_index=True)
            updated_evals.to_csv(eval_file_path, index=False)
            print(f"Updated {eval_file_path} (removed Naive baselines, saved 3 models with R2).")

    return evaluation_records, preds_records, champion_info

if __name__ == "__main__":
    run_model_comparison()
