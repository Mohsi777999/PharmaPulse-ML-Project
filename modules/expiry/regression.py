"""
PharmaPulse - Expiry Unsold Quantity Regression Module
Predicts the exact number of unsold units remaining at batch expiration (Continuous Task).
Candidate algorithms:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Gradient Boosting Regressor

Computes financial exposure: Predicted Unsold Units * Unit Cost.
Maps predicted unsold percentage to Risk Levels: Low, Medium, High, Critical.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .features import EXPIRY_FEATURE_COLS, load_and_prep_expiry_data, get_grouped_expiry_splits
from utils.metrics import evaluate_regression

def map_risk_level(unsold_pct):
    if unsold_pct < 5.0:
        return "Low"
    elif unsold_pct < 15.0:
        return "Medium"
    elif unsold_pct < 30.0:
        return "High"
    else:
        return "Critical"

def train_expiry_regression(project_root=None):
    print("=" * 70)
    print("TRAINING EXPIRY UNSOLD QUANTITY REGRESSION MODELS")
    print("=" * 70)

    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    models_dir = os.path.join(project_root, "models", "expiry")
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(models_dir, exist_ok=True)

    df_supervised, _ = load_and_prep_expiry_data(project_root)
    df_meds = pd.read_csv(os.path.join(raw_dir, "medicines.csv")).set_index("medicine_id")["unit_cost"].to_dict()

    train_df, val_df, test_df, split_info = get_grouped_expiry_splits(df_supervised)

    X_train, y_train = train_df[EXPIRY_FEATURE_COLS], train_df["target_unsold_quantity"].astype(float)
    X_val, y_val = val_df[EXPIRY_FEATURE_COLS], val_df["target_unsold_quantity"].astype(float)
    X_test, y_test = test_df[EXPIRY_FEATURE_COLS], test_df["target_unsold_quantity"].astype(float)

    evaluation_records = []

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    val_pred_lr = np.clip(lr.predict(X_val), 0, None)
    test_pred_lr = np.clip(lr.predict(X_test), 0, None)
    eval_val_lr = evaluate_regression(y_val, val_pred_lr)
    eval_test_lr = evaluate_regression(y_test, test_pred_lr)
    for m_name, m_val in eval_val_lr.items():
        evaluation_records.append({"model_name": "Linear Regression", "module": "Expiry Risk", "task_type": "Regression", "evaluation_split": "Validation", "metric_name": m_name, "metric_value": m_val})
    for m_name, m_val in eval_test_lr.items():
        evaluation_records.append({"model_name": "Linear Regression", "module": "Expiry Risk", "task_type": "Regression", "evaluation_split": "Test", "metric_name": m_name, "metric_value": m_val})
    joblib.dump(lr, os.path.join(models_dir, "linear_regression_reg.joblib"))
    print(f"Linear Regression -> Val MAE: {eval_val_lr['MAE']}, Test MAE: {eval_test_lr['MAE']}, R2: {eval_test_lr['R2']}")

    # 2. Decision Tree Regressor
    dt = DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=42)
    dt.fit(X_train, y_train)
    val_pred_dt = np.clip(dt.predict(X_val), 0, None)
    test_pred_dt = np.clip(dt.predict(X_test), 0, None)
    eval_val_dt = evaluate_regression(y_val, val_pred_dt)
    eval_test_dt = evaluate_regression(y_test, test_pred_dt)
    for m_name, m_val in eval_val_dt.items():
        evaluation_records.append({"model_name": "Decision Tree", "module": "Expiry Risk", "task_type": "Regression", "evaluation_split": "Validation", "metric_name": m_name, "metric_value": m_val})
    for m_name, m_val in eval_test_dt.items():
        evaluation_records.append({"model_name": "Decision Tree", "module": "Expiry Risk", "task_type": "Regression", "evaluation_split": "Test", "metric_name": m_name, "metric_value": m_val})
    joblib.dump(dt, os.path.join(models_dir, "decision_tree_reg.joblib"))
    print(f"Decision Tree -> Val MAE: {eval_val_dt['MAE']}, Test MAE: {eval_test_dt['MAE']}, R2: {eval_test_dt['R2']}")

    # 3. Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_split=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_pred_rf = np.clip(rf.predict(X_val), 0, None)
    test_pred_rf = np.clip(rf.predict(X_test), 0, None)
    eval_val_rf = evaluate_regression(y_val, val_pred_rf)
    eval_test_rf = evaluate_regression(y_test, test_pred_rf)
    for m_name, m_val in eval_val_rf.items():
        evaluation_records.append({"model_name": "Random Forest", "module": "Expiry Risk", "task_type": "Regression", "evaluation_split": "Validation", "metric_name": m_name, "metric_value": m_val})
    for m_name, m_val in eval_test_rf.items():
        evaluation_records.append({"model_name": "Random Forest", "module": "Expiry Risk", "task_type": "Regression", "evaluation_split": "Test", "metric_name": m_name, "metric_value": m_val})
    joblib.dump(rf, os.path.join(models_dir, "random_forest_reg.joblib"))
    print(f"Random Forest -> Val MAE: {eval_val_rf['MAE']}, Test MAE: {eval_test_rf['MAE']}, R2: {eval_test_rf['R2']}")

    # 4. Gradient Boosting Regressor (Winning Candidate)
    gb = GradientBoostingRegressor(n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42)
    gb.fit(X_train, y_train)
    val_pred_gb = np.clip(gb.predict(X_val), 0, None)
    test_pred_gb = np.clip(gb.predict(X_test), 0, None)
    eval_val_gb = evaluate_regression(y_val, val_pred_gb)
    eval_test_gb = evaluate_regression(y_test, test_pred_gb)
    for m_name, m_val in eval_val_gb.items():
        evaluation_records.append({"model_name": "Gradient Boosting", "module": "Expiry Risk", "task_type": "Regression", "evaluation_split": "Validation", "metric_name": m_name, "metric_value": m_val})
    for m_name, m_val in eval_test_gb.items():
        evaluation_records.append({"model_name": "Gradient Boosting", "module": "Expiry Risk", "task_type": "Regression", "evaluation_split": "Test", "metric_name": m_name, "metric_value": m_val})
    joblib.dump(gb, os.path.join(models_dir, "gradient_boosting_reg.joblib"))
    print(f"Gradient Boosting -> Val MAE: {eval_val_gb['MAE']}, Test MAE: {eval_test_gb['MAE']}, R2: {eval_test_gb['R2']}")

    # Generate prediction records for model_predictions.csv
    test_res = test_df.copy()
    test_res["predicted_unsold_units"] = test_pred_gb.round(1)
    test_res["unit_cost"] = test_res["medicine_id"].map(df_meds).fillna(20.0)
    test_res["financial_exposure"] = (test_res["predicted_unsold_units"] * test_res["unit_cost"]).round(2)
    test_res["predicted_unsold_pct"] = (test_res["predicted_unsold_units"] / test_res["initial_quantity"] * 100.0).round(2)
    test_res["predicted_risk_level"] = test_res["predicted_unsold_pct"].apply(map_risk_level)

    preds_records = []
    for idx, row in test_res.iterrows():
        preds_records.append({
            "prediction_id": f"PRED_EXP_{len(preds_records)+1:06d}",
            "module": "Expiry Risk",
            "medicine_id": row["medicine_id"],
            "batch_id": row["batch_id"],
            "prediction_date": row["snapshot_date"],
            "target_name": "unsold_quantity",
            "actual_value": row["target_unsold_quantity"],
            "predicted_value": row["predicted_unsold_units"],
            "risk_level": row["predicted_risk_level"]
        })

    return evaluation_records, preds_records, test_res
