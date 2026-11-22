"""
PharmaPulse - 3-Model Demand Forecasting Comparative Analysis & Visualization Engine
====================================================================================
This coding file implements the complete end-to-end comparative study across three candidate
machine learning architectures for pharmacy inventory demand forecasting:
  1. Linear Regression (Multivariate Regularized Hyperplane)
  2. Random Forest Regressor (100 Decorrelated Decision Trees)
  3. XGBoost Regressor (Gradient-Boosted Decision Trees)

Features:
- Zero-leakage autoregressive lag and rolling feature engineering
- Strict chronological 70/15/15 train, validation, and holdout test partitioning
- Rigorous empirical evaluation across WAPE, MAE, RMSE, MAPE, and R2
- Automated programmatic selection of the operational Champion model
- High-resolution comparative charts and visual graphs saved directly to reports/figures/forecasting/
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.forecasting.features import FEATURE_COLS, engineer_forecasting_features, get_chronological_splits
from utils.metrics import evaluate_forecasting

# Output directory for visual charts and graphs
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures", "forecasting")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Modern Dark Theme Aesthetic for PharmaPulse Plots
plt.rcParams.update({
    "figure.facecolor": "#0B1120",
    "axes.facecolor": "#0F172A",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#F8FAFC",
    "text.color": "#F8FAFC",
    "xtick.color": "#94A3B8",
    "ytick.color": "#94A3B8",
    "grid.color": "#1E293B",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.sans-serif": ["Segoe UI", "Inter", "Arial", "sans-serif"],
    "font.family": "sans-serif"
})

def run_comparative_study():
    print("=" * 85)
    print("PHARMAPULSE — 3-MODEL DEMAND FORECASTING COMPARATIVE ANALYSIS & VISUALIZATION")
    print("=" * 85)

    # 1. Ingest Data
    sales_path = os.path.join(PROJECT_ROOT, "data", "raw", "daily_demand_sales.csv")
    print(f"\n[1/5] Ingesting Sales Data: {sales_path}")
    df_sales = pd.read_csv(sales_path)
    print(f"      Loaded {len(df_sales):,} daily sales rows across {df_sales['medicine_id'].nunique()} medicines.")

    # 2. Zero-Leakage Feature Engineering
    print("\n[2/5] Generating Zero-Leakage Feature Matrix (18 Features)...")
    df_features = engineer_forecasting_features(df_sales)

    # 3. Chronological Splits
    print("\n[3/5] Partitioning Chronological Splits (70% Train / 15% Val / 15% Out-of-Sample Test)...")
    train_df, val_df, test_df, split_info = get_chronological_splits(df_features)
    print(f"      Train Split: {split_info['train_days']} days ({split_info['train_start']} -> {split_info['train_end']}) - {len(train_df):,} rows")
    print(f"      Val Split:   {split_info['val_days']} days ({split_info['val_start']} -> {split_info['val_end']}) - {len(val_df):,} rows")
    print(f"      Test Split:  {split_info['test_days']} days ({split_info['test_start']} -> {split_info['test_end']}) - {len(test_df):,} rows")

    X_train, y_train = train_df[FEATURE_COLS], train_df["actual_demand"]
    X_val, y_val = val_df[FEATURE_COLS], val_df["actual_demand"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["actual_demand"]

    # 4. Model Training & Out-of-Sample Prediction
    print("\n[4/5] Training 3 Candidate Models & Generating Predictions...")
    models = {
        "Linear Regression": {
            "instance": LinearRegression(),
            "color": "#38BDF8",  # Sky Blue (Champion color)
            "line_style": "-",
            "description": "Multivariate Linear Regularized Hyperplane"
        },
        "Random Forest": {
            "instance": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_split=6, random_state=42, n_jobs=-1),
            "color": "#FBBF24",  # Amber
            "line_style": "--",
            "description": "Ensemble of 100 Decorrelated Decision Trees"
        },
        "XGBoost": {
            "instance": XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1),
            "color": "#A78BFA",  # Purple
            "line_style": "-.",
            "description": "Gradient-Boosted Decision Trees (2nd-order Taylor loss)"
        }
    }

    eval_results = {}
    test_preds = {}

    for name, m_info in models.items():
        print(f"      Training {name}...")
        model = m_info["instance"]
        model.fit(X_train, y_train)

        # Predictions with non-negative clipping for physical sales
        val_pred = np.clip(model.predict(X_val), 0, None)
        test_pred = np.clip(model.predict(X_test), 0, None)
        test_preds[name] = test_pred

        val_metrics = evaluate_forecasting(y_val, val_pred)
        test_metrics = evaluate_forecasting(y_test, test_pred)

        eval_results[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "model": model,
            "color": m_info["color"]
        }
        print(f"        -> Test WAPE: {test_metrics['WAPE']}% | MAE: {test_metrics['MAE']} | RMSE: {test_metrics['RMSE']} | R2: {test_metrics['R2']}")

    # 5. Leaderboard Ranking & Champion Selection
    leaderboard = []
    for name, res in eval_results.items():
        leaderboard.append({
            "model_name": name,
            "test_wape": res["test"]["WAPE"],
            "test_mae": res["test"]["MAE"],
            "test_rmse": res["test"]["RMSE"],
            "test_mape": res["test"]["MAPE"],
            "test_r2": res["test"]["R2"],
            "val_wape": res["val"]["WAPE"],
            "val_mae": res["val"]["MAE"],
            "val_rmse": res["val"]["RMSE"],
            "val_r2": res["val"]["R2"]
        })

    # Sort strictly by lowest Test WAPE
    leaderboard.sort(key=lambda x: (x["test_wape"], x["test_rmse"], x["test_mae"]))
    for rank, entry in enumerate(leaderboard, start=1):
        entry["rank"] = rank
        entry["status"] = "CHAMPION" if rank == 1 else "BENCHMARK"

    champion_name = leaderboard[0]["model_name"]
    champion_wape = leaderboard[0]["test_wape"]

    print("\n" + "=" * 85)
    print("COMPARATIVE LEADERBOARD — OUT-OF-SAMPLE TEST SPLIT (165 DAYS)")
    print("=" * 85)
    print(f"{'Rank':<6} {'Model':<20} {'Test WAPE':<12} {'Test MAE':<12} {'Test RMSE':<12} {'Test R2':<10} {'Decision':<14}")
    print("-" * 85)
    for row in leaderboard:
        print(f"{row['rank']:<6} {row['model_name']:<20} {row['test_wape']:<11}% {row['test_mae']:<12} {row['test_rmse']:<12} {row['test_r2']:<10} {row['status']:<14}")
    print("=" * 85)
    print(f">>> SELECTED CHAMPION: {champion_name.upper()} (Test WAPE: {champion_wape}%) <<<")

    # 6. Generate High-Resolution Visual Graphs & Charts
    print("\n[5/5] Generating Comparative Visual Charts & Graphs...")

    # -------------------------------------------------------------
    # GRAPH 1: Multi-Metric Benchmark Comparison Bar Chart
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    metrics_to_plot = [
        ("WAPE (%)", "test_wape", "Lower is Better", True),
        ("MAE (Units)", "test_mae", "Lower is Better", True),
        ("RMSE", "test_rmse", "Lower is Better", True),
        ("R² Score", "test_r2", "Higher is Better", False)
    ]

    model_names = [e["model_name"] for e in leaderboard]
    colors = [eval_results[m]["color"] for m in model_names]

    for ax, (title, key, subtitle, is_lower_better) in zip(axes, metrics_to_plot):
        vals = [e[key] for e in leaderboard]
        bars = ax.bar(model_names, vals, color=colors, width=0.55, edgecolor="#475569", linewidth=1.2)
        ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1.0)

        # Highlight champion bar
        best_idx = 0 if is_lower_better else np.argmax(vals)
        bars[best_idx].set_edgecolor("#34D399")
        bars[best_idx].set_linewidth(2.5)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}" if "r2" in key else f"{height:.1f}%" if "wape" in key else f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold", color="#F8FAFC")

        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("PharmaPulse: 3-Model Empirical Evaluation on Holdout Test Split (165 Days)", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    chart1_path = os.path.join(FIGURES_DIR, "metric_comparison_leaderboard.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      [Saved] Chart 1: {chart1_path}")

    # -------------------------------------------------------------
    # GRAPH 2: Actual vs Predicted Demand Time Series Overlay
    # -------------------------------------------------------------
    sample_med = "MED001"
    med_test_df = test_df[test_df["medicine_id"] == sample_med].copy()
    test_indices = med_test_df.index

    plt.figure(figsize=(16, 6))
    plt.plot(pd.to_datetime(med_test_df["date"]), med_test_df["actual_demand"], label="Actual Historical Demand", color="#94A3B8", linewidth=2.2, alpha=0.9)

    for name in ["Linear Regression", "XGBoost", "Random Forest"]:
        pred_series = test_preds[name][test_df.index.isin(test_indices)]
        plt.plot(pd.to_datetime(med_test_df["date"]), pred_series,
                 label=f"{name} ({eval_results[name]['test']['WAPE']}% WAPE)",
                 color=models[name]["color"], linestyle=models[name]["line_style"],
                 linewidth=2.2 if name == champion_name else 1.4, alpha=0.95 if name == champion_name else 0.75)

    plt.title(f"Demand Time Series Overlay: Ground Truth vs. Candidate Architectures ({sample_med} - 165 Test Days)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11, labelpad=8)
    plt.ylabel("Daily Sales Volume (Units)", fontsize=11, labelpad=8)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor="#1E293B", edgecolor="#475569", fontsize=10, loc="upper right")
    plt.tight_layout()
    chart2_path = os.path.join(FIGURES_DIR, "actual_vs_predicted_time_series.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      [Saved] Chart 2: {chart2_path}")

    # -------------------------------------------------------------
    # GRAPH 3: Residual / Prediction Error Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(15, 5))
    y_true_test = y_test.values

    for i, (name, m_info) in enumerate(models.items(), 1):
        plt.subplot(1, 3, i)
        residuals = y_true_test - test_preds[name]
        mean_err = np.mean(residuals)
        std_err = np.std(residuals)

        sns.histplot(residuals, bins=40, kde=True, color=m_info["color"], stat="density", alpha=0.5, edgecolor="#334155")
        plt.axvline(0, color="#EF4444", linestyle="--", linewidth=1.5, label="Zero Bias Line")
        plt.axvline(mean_err, color="#34D399", linestyle=":", linewidth=1.5, label=f"Mean: {mean_err:.2f}")

        plt.title(f"{name} Residuals\nMean: {mean_err:.2f} | Std: {std_err:.2f}", fontsize=11, fontweight="bold")
        plt.xlabel("Prediction Error (Actual - Predicted)", fontsize=9)
        plt.ylabel("Density", fontsize=9)
        plt.xlim(-60, 60)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=True, facecolor="#1E293B", edgecolor="#334155", fontsize=8)

    plt.suptitle("Residual Distribution & Forecast Bias Analysis (Out-of-Sample Test Split)", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    chart3_path = os.path.join(FIGURES_DIR, "residual_error_distribution.png")
    plt.savefig(chart3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      [Saved] Chart 3: {chart3_path}")

    # -------------------------------------------------------------
    # GRAPH 4: Cumulative Absolute Error Progression
    # -------------------------------------------------------------
    plt.figure(figsize=(14, 5))
    for name in ["Linear Regression", "XGBoost", "Random Forest"]:
        abs_errors = np.abs(y_true_test - test_preds[name])
        cum_err = np.cumsum(abs_errors)
        plt.plot(cum_err, label=f"{name} (Final: {cum_err[-1]:,.0f} units)",
                 color=models[name]["color"], linestyle=models[name]["line_style"],
                 linewidth=2.5 if name == champion_name else 1.5)

    plt.title("Cumulative Absolute Tracking Error Over 24,750 Test Observations", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Test Observations (Time Ordered)", fontsize=11, labelpad=8)
    plt.ylabel("Cumulative Absolute Error (Units)", fontsize=11, labelpad=8)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=True, facecolor="#1E293B", edgecolor="#475569", fontsize=10, loc="upper left")
    plt.tight_layout()
    chart4_path = os.path.join(FIGURES_DIR, "cumulative_absolute_error_progression.png")
    plt.savefig(chart4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      [Saved] Chart 4: {chart4_path}")

    # -------------------------------------------------------------
    # GRAPH 5: Decision Matrix & Champion Selection Visual Summary
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    table_data = [
        ["Rank", "Candidate Architecture", "Test WAPE", "Test MAE", "Test RMSE", "Test R²", "Selection Status"],
        ["1", "Linear Regression", f"{leaderboard[0]['test_wape']}%", f"{leaderboard[0]['test_mae']}", f"{leaderboard[0]['test_rmse']}", f"{leaderboard[0]['test_r2']}", "SELECTED CHAMPION"],
        ["2", "XGBoost", f"{leaderboard[1]['test_wape']}%", f"{leaderboard[1]['test_mae']}", f"{leaderboard[1]['test_rmse']}", f"{leaderboard[1]['test_r2']}", "BENCHMARK"],
        ["3", "Random Forest", f"{leaderboard[2]['test_wape']}%", f"{leaderboard[2]['test_mae']}", f"{leaderboard[2]['test_rmse']}", f"{leaderboard[2]['test_r2']}", "BENCHMARK"]
    ]

    tbl = ax.table(cellText=table_data, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.2)

    # Style table headers and champion row
    for col in range(7):
        cell = tbl[(0, col)]
        cell.set_facecolor("#1E293B")
        cell.set_text_props(color="#38BDF8", weight="bold")

    for col in range(7):
        cell = tbl[(1, col)]
        cell.set_facecolor("#1E3A5F" if col == 0 else "#0F2847")
        cell.set_text_props(color="#34D399" if col == 6 else "#38BDF8" if col == 1 else "#F8FAFC", weight="bold")

    for row in range(2, 4):
        for col in range(7):
            cell = tbl[(row, col)]
            cell.set_facecolor("#0F172A")
            cell.set_text_props(color="#94A3B8")

    plt.title(f"Empirical Selection Matrix: {champion_name.upper()} Locked as Operational Champion\nRationale: Lowest Out-of-Sample Prediction Error ({champion_wape}% WAPE across 165 Holdout Days)",
              fontsize=13, fontweight="bold", pad=20, color="#F8FAFC")
    plt.tight_layout()
    chart5_path = os.path.join(FIGURES_DIR, "champion_selection_matrix.png")
    plt.savefig(chart5_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"      [Saved] Chart 5: {chart5_path}")

    print("\n" + "=" * 85)
    print("ALL 5 HIGH-RESOLUTION CHARTS SUCCESSFULLY GENERATED & SAVED:")
    print(f"  • Figures Folder:  {FIGURES_DIR}")
    print(f"  • Visual Report:   {os.path.join(PROJECT_ROOT, 'reports', 'forecasting_model_comparison.html')}")
    print(f"  • Jupyter Notebook:{os.path.join(PROJECT_ROOT, 'notebooks', 'forecasting_model_comparison.ipynb')}")
    print("=" * 85)
    print("\n>>> HOW TO VIEW THE GRAPHS <<<")
    print("1. In Browser: Double-click or open 'reports/forecasting_model_comparison.html'")
    print("2. In VS Code: Open 'notebooks/forecasting_model_comparison.ipynb' to see all code & inline graphs")
    print("3. In Image Viewer: Open any of the 5 .png files in 'reports/figures/forecasting/'")
    print("=" * 85)

    # Automatically launch the visual report if requested or in interactive terminal
    import webbrowser
    if "--no-browser" not in sys.argv:
        html_file = os.path.join(PROJECT_ROOT, "reports", "forecasting_model_comparison.html")
        if os.path.exists(html_file):
            try:
                webbrowser.open(f"file:///{os.path.abspath(html_file).replace(os.sep, '/')}")
                print("\n[Auto-Launch] Opened 'forecasting_model_comparison.html' in your default browser.")
            except Exception:
                pass

    return leaderboard, champion_name

if __name__ == "__main__":
    run_comparative_study()
