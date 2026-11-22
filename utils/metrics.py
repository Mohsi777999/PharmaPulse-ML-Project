"""
PharmaPulse - Metrics Utility Module
Calculates standard and domain-safe metrics for Demand Forecasting, Expiry Risk,
Stockout Prediction, and Sales Behaviour Clustering.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# -------------------------------------------------------------
# 1. Demand Forecasting Metrics
# -------------------------------------------------------------
def calculate_wape(y_true, y_pred):
    """
    Weighted Absolute Percentage Error (WAPE):
    WAPE = sum(|y_true - y_pred|) / sum(y_true) * 100
    Preferred in retail and pharmacy forecasting because it avoids division by zero on low-demand days.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    total_actual = np.sum(y_true)
    if total_actual == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_actual * 100.0)

def calculate_safe_mape(y_true, y_pred, epsilon=1.0):
    """
    Mean Absolute Percentage Error (MAPE) with epsilon threshold for near-zero demand:
    Safe MAPE = mean(|y_true - y_pred| / max(y_true, epsilon)) * 100
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(y_true, epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

def evaluate_forecasting(y_true, y_pred):
    """
    Returns MAE, RMSE, WAPE, Safe MAPE, and R2 in a dictionary.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    wape = calculate_wape(y_true, y_pred)
    mape = calculate_safe_mape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE": round(wape, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4)
    }

# -------------------------------------------------------------
# 2. Classification Metrics (Stockout & Expiry Classification)
# -------------------------------------------------------------
def evaluate_classification(y_true, y_pred, y_prob=None, average="binary"):
    """
    Returns Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    rec = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    auc = None
    if y_prob is not None:
        try:
            if average == "binary":
                # Probability of positive class
                if y_prob.ndim == 2:
                    y_prob_eval = y_prob[:, 1]
                else:
                    y_prob_eval = y_prob
                auc = float(roc_auc_score(y_true, y_prob_eval))
            else:
                auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            auc = None

    cm = confusion_matrix(y_true, y_pred).tolist()

    res = {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4) if auc is not None else "N/A",
        "Confusion_Matrix": cm
    }
    return res

# -------------------------------------------------------------
# 3. Regression Metrics (Expiry Unsold Quantity)
# -------------------------------------------------------------
def evaluate_regression(y_true, y_pred):
    """
    Returns MAE, RMSE, and R2.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "R2": round(r2, 4)
    }

# -------------------------------------------------------------
# 4. Clustering Metrics (Sales Behaviour Clustering)
# -------------------------------------------------------------
def evaluate_clustering(X, labels):
    """
    Returns Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
    """
    unique_labels = set(labels)
    # Ignore noise cluster (-1) if DBSCAN
    valid_labels = [lbl for lbl in unique_labels if lbl != -1]
    
    if len(valid_labels) < 2:
        return {
            "Silhouette_Score": -1.0,
            "Davies_Bouldin_Index": float("inf"),
            "Calinski_Harabasz_Index": 0.0,
            "Clusters_Found": len(valid_labels)
        }
        
    # If DBSCAN has noise (-1), filter for metric calculations if needed
    mask = (labels != -1)
    if mask.sum() < len(valid_labels) + 1:
        return {
            "Silhouette_Score": -1.0,
            "Davies_Bouldin_Index": float("inf"),
            "Calinski_Harabasz_Index": 0.0,
            "Clusters_Found": len(valid_labels)
        }
        
    sil = float(silhouette_score(X[mask], labels[mask]))
    db = float(davies_bouldin_score(X[mask], labels[mask]))
    ch = float(calinski_harabasz_score(X[mask], labels[mask]))
    
    return {
        "Silhouette_Score": round(sil, 4),
        "Davies_Bouldin_Index": round(db, 4),
        "Calinski_Harabasz_Index": round(ch, 2),
        "Clusters_Found": len(valid_labels)
    }
