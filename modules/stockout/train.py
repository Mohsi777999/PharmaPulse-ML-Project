"""
PharmaPulse - Stockout Prediction Training & Comparative Evaluation
Trains candidate classification algorithms to predict stockout occurrence before delivery arrival.
Candidate algorithms:
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. XGBoost Classifier

Decision Tree is explicitly excluded per blueprint specifications.
Focus is placed on optimizing Recall to minimize costly uncaptured stockouts.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .features import STOCKOUT_FEATURE_COLS, load_and_prep_stockout_data, get_chronological_stockout_splits
from utils.metrics import evaluate_classification

def train_stockout_models(project_root=None):
    print("=" * 70)
    print("TRAINING STOCKOUT PREDICTION MODELS")
    print("=" * 70)

    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    models_dir = os.path.join(project_root, "models", "stockout")
    os.makedirs(models_dir, exist_ok=True)

    df_supervised, df_pending = load_and_prep_stockout_data(project_root)
    train_df, val_df, test_df, split_info = get_chronological_stockout_splits(df_supervised)
    print(f"Chronological Splits: Train {split_info['train_orders']} POs, Val {split_info['val_orders']} POs, Test {split_info['test_orders']} POs")

    X_train, y_train = train_df[STOCKOUT_FEATURE_COLS], train_df["target_stockout_before_arrival"].astype(int)
    X_val, y_val = val_df[STOCKOUT_FEATURE_COLS], val_df["target_stockout_before_arrival"].astype(int)
    X_test, y_test = test_df[STOCKOUT_FEATURE_COLS], test_df["target_stockout_before_arrival"].astype(int)

    # Standard Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(models_dir, "stockout_scaler.joblib"))

    evaluation_records = []

    # Class weight calculation to boost recall
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_weight = float(neg_count / max(pos_count, 1))

    # 1. Logistic Regression
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    val_pred_lr = lr.predict(X_val_scaled)
    val_prob_lr = lr.predict_proba(X_val_scaled)
    test_pred_lr = lr.predict(X_test_scaled)
    test_prob_lr = lr.predict_proba(X_test_scaled)

    eval_val_lr = evaluate_classification(y_val, val_pred_lr, val_prob_lr)
    eval_test_lr = evaluate_classification(y_test, test_pred_lr, test_prob_lr)
    for m_name, m_val in eval_val_lr.items():
        if m_name != "Confusion_Matrix":
            evaluation_records.append({"model_name": "Logistic Regression", "module": "Stockout Prediction", "task_type": "Classification", "evaluation_split": "Validation", "metric_name": m_name, "metric_value": m_val})
    for m_name, m_val in eval_test_lr.items():
        if m_name != "Confusion_Matrix":
            evaluation_records.append({"model_name": "Logistic Regression", "module": "Stockout Prediction", "task_type": "Classification", "evaluation_split": "Test", "metric_name": m_name, "metric_value": m_val})
    joblib.dump(lr, os.path.join(models_dir, "logistic_regression.joblib"))
    print(f"Logistic Regression -> Val Recall: {eval_val_lr['Recall']}, F1: {eval_val_lr['F1-Score']}; Test Recall: {eval_test_lr['Recall']}, F1: {eval_test_lr['F1-Score']}")

    # 2. Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=120, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    val_pred_rf = rf.predict(X_val)
    val_prob_rf = rf.predict_proba(X_val)
    test_pred_rf = rf.predict(X_test)
    test_prob_rf = rf.predict_proba(X_test)

    eval_val_rf = evaluate_classification(y_val, val_pred_rf, val_prob_rf)
    eval_test_rf = evaluate_classification(y_test, test_pred_rf, test_prob_rf)
    for m_name, m_val in eval_val_rf.items():
        if m_name != "Confusion_Matrix":
            evaluation_records.append({"model_name": "Random Forest", "module": "Stockout Prediction", "task_type": "Classification", "evaluation_split": "Validation", "metric_name": m_name, "metric_value": m_val})
    for m_name, m_val in eval_test_rf.items():
        if m_name != "Confusion_Matrix":
            evaluation_records.append({"model_name": "Random Forest", "module": "Stockout Prediction", "task_type": "Classification", "evaluation_split": "Test", "metric_name": m_name, "metric_value": m_val})
    joblib.dump(rf, os.path.join(models_dir, "random_forest_clf.joblib"))
    print(f"Random Forest -> Val Recall: {eval_val_rf['Recall']}, F1: {eval_val_rf['F1-Score']}; Test Recall: {eval_test_rf['Recall']}, F1: {eval_test_rf['F1-Score']}")

    # 3. Gradient Boosting Classifier
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42)
    gb.fit(X_train, y_train)
    val_pred_gb = gb.predict(X_val)
    val_prob_gb = gb.predict_proba(X_val)
    test_pred_gb = gb.predict(X_test)
    test_prob_gb = gb.predict_proba(X_test)

    eval_val_gb = evaluate_classification(y_val, val_pred_gb, val_prob_gb)
    eval_test_gb = evaluate_classification(y_test, test_pred_gb, test_prob_gb)
    for m_name, m_val in eval_val_gb.items():
        if m_name != "Confusion_Matrix":
            evaluation_records.append({"model_name": "Gradient Boosting", "module": "Stockout Prediction", "task_type": "Classification", "evaluation_split": "Validation", "metric_name": m_name, "metric_value": m_val})
    for m_name, m_val in eval_test_gb.items():
        if m_name != "Confusion_Matrix":
            evaluation_records.append({"model_name": "Gradient Boosting", "module": "Stockout Prediction", "task_type": "Classification", "evaluation_split": "Test", "metric_name": m_name, "metric_value": m_val})
    joblib.dump(gb, os.path.join(models_dir, "gradient_boosting_clf.joblib"))
    print(f"Gradient Boosting -> Val Recall: {eval_val_gb['Recall']}, F1: {eval_val_gb['F1-Score']}; Test Recall: {eval_test_gb['Recall']}, F1: {eval_test_gb['F1-Score']}")

    # 4. XGBoost Classifier (Optimized for High Recall & F1)
    xgb = XGBClassifier(n_estimators=120, max_depth=5, learning_rate=0.08, scale_pos_weight=scale_weight, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    val_pred_xgb = xgb.predict(X_val)
    val_prob_xgb = xgb.predict_proba(X_val)
    test_pred_xgb = xgb.predict(X_test)
    test_prob_xgb = xgb.predict_proba(X_test)

    eval_val_xgb = evaluate_classification(y_val, val_pred_xgb, val_prob_xgb)
    eval_test_xgb = evaluate_classification(y_test, test_pred_xgb, test_prob_xgb)
    for m_name, m_val in eval_val_xgb.items():
        if m_name != "Confusion_Matrix":
            evaluation_records.append({"model_name": "XGBoost", "module": "Stockout Prediction", "task_type": "Classification", "evaluation_split": "Validation", "metric_name": m_name, "metric_value": m_val})
    for m_name, m_val in eval_test_xgb.items():
        if m_name != "Confusion_Matrix":
            evaluation_records.append({"model_name": "XGBoost", "module": "Stockout Prediction", "task_type": "Classification", "evaluation_split": "Test", "metric_name": m_name, "metric_value": m_val})
    joblib.dump(xgb, os.path.join(models_dir, "xgboost_stockout_clf.joblib"))
    joblib.dump(eval_test_xgb["Confusion_Matrix"], os.path.join(models_dir, "xgb_test_confusion_matrix.joblib"))
    print(f"XGBoost -> Val Recall: {eval_val_xgb['Recall']}, F1: {eval_val_xgb['F1-Score']}; Test Recall: {eval_test_xgb['Recall']}, F1: {eval_test_xgb['F1-Score']}")

    joblib.dump(STOCKOUT_FEATURE_COLS, os.path.join(models_dir, "feature_cols.joblib"))

    # Generate sample prediction records for model_predictions.csv
    test_res = test_df.copy()
    test_res["predicted_stockout"] = test_pred_xgb
    test_res["stockout_probability"] = test_prob_xgb[:, 1].round(3)

    preds_records = []
    for idx, row in test_res.iterrows():
        preds_records.append({
            "prediction_id": f"PRED_SO_{len(preds_records)+1:06d}",
            "module": "Stockout Prediction",
            "medicine_id": row["medicine_id"],
            "batch_id": "N/A",
            "prediction_date": row["order_date"],
            "target_name": "stockout_before_arrival",
            "actual_value": int(row["target_stockout_before_arrival"]),
            "predicted_value": int(row["predicted_stockout"]),
            "risk_level": "High" if row["stockout_probability"] >= 0.50 else "Low"
        })

    return evaluation_records, preds_records, test_res
