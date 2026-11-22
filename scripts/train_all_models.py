"""
PharmaPulse - Master Model Training Pipeline
Orchestrates training and comparative evaluation across all four ML modules:
1. Demand Forecasting
2. Expiry Risk (Classification + Regression)
3. Stockout Prediction
4. Sales Behaviour Clustering

Generates:
- data/outputs/model_evaluations.csv & data/model_evaluations.csv
- data/outputs/model_predictions.csv & data/model_predictions.csv
- Serialized joblib artifacts in models/
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from modules.forecasting.train import train_forecasting_models
from modules.expiry.classification import train_expiry_classification
from modules.expiry.regression import train_expiry_regression
from modules.stockout.train import train_stockout_models
from modules.clustering.train import train_clustering_models

def run_all_training():
    print("=" * 80)
    print("PHARMAPULSE — MASTER ML TRAINING & BENCHMARKING PIPELINE")
    print("=" * 80)

    outputs_dir = os.path.join(PROJECT_ROOT, "data", "outputs")
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(outputs_dir, exist_ok=True)

    all_evaluations = []
    all_predictions = []

    # 1. Demand Forecasting
    fc_evals, fc_preds = train_forecasting_models(PROJECT_ROOT)
    all_evaluations.extend(fc_evals)
    all_predictions.extend(fc_preds)

    # 2. Expiry Risk Classification
    exp_clf_evals, _ = train_expiry_classification(PROJECT_ROOT)
    all_evaluations.extend(exp_clf_evals)

    # 3. Expiry Risk Regression
    exp_reg_evals, exp_preds, _ = train_expiry_regression(PROJECT_ROOT)
    all_evaluations.extend(exp_reg_evals)
    all_predictions.extend(exp_preds)

    # 4. Stockout Prediction
    so_evals, so_preds, _ = train_stockout_models(PROJECT_ROOT)
    all_evaluations.extend(so_evals)
    all_predictions.extend(so_preds)

    # 5. Sales Behaviour Clustering
    cluster_evals, _, _ = train_clustering_models(PROJECT_ROOT)
    all_evaluations.extend(cluster_evals)

    # Export evaluations
    df_evals = pd.DataFrame(all_evaluations)
    df_evals.to_csv(os.path.join(outputs_dir, "model_evaluations.csv"), index=False)
    df_evals.to_csv(os.path.join(data_dir, "model_evaluations.csv"), index=False)
    print(f"\nSaved model_evaluations.csv: {len(df_evals)} benchmark records.")

    # Export predictions
    df_preds = pd.DataFrame(all_predictions)
    df_preds.to_csv(os.path.join(outputs_dir, "model_predictions.csv"), index=False)
    df_preds.to_csv(os.path.join(data_dir, "model_predictions.csv"), index=False)
    print(f"Saved model_predictions.csv: {len(df_preds)} prediction records.")

    print("\n" + "=" * 80)
    print("ALL ML MODULES SUCCESSFULLY TRAINED, EVALUATED, AND SERIALIZED!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_training()
