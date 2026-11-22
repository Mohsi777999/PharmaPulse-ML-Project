"""
PharmaPulse - Forecasting Model Training & Comparative Evaluation
Trains candidate forecasting models:
1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

Evaluates across Train, Validation, and Test splits.
Selects best performing model as operational Champion.
"""

import os
import sys

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.forecasting.model_comparison import run_model_comparison

def train_forecasting_models(project_root=None):
    """
    Executes comparative training for Linear Regression, Random Forest, and XGBoost,
    ranks them by out-of-sample Test WAPE, designates the champion, and returns evaluations.
    """
    eval_records, pred_records, champion_info = run_model_comparison(project_root)
    return eval_records, pred_records

if __name__ == "__main__":
    train_forecasting_models()
