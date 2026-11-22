import os
import sys
import json
import pandas as pd

sys.path.insert(0, '.')
from modules.forecasting.predict import generate_forecast
from backend.app.routers.forecasting import predict_demand
from backend.app.routers.performance import get_performance_metrics

print("=" * 70)
print("PHARMAPULSE — FORECASTING SYSTEM VERIFICATION TEST")
print("=" * 70)

# 1. Test Champion Model Forecast (default)
print("\n[Test 1] Testing Default Champion Forecast for MED001 (30 days)...")
fc_df, hist_df = generate_forecast('MED001', horizon_days=30)
print(f"  Historical observations loaded: {len(hist_df)} rows")
print(f"  Forecast steps produced:        {len(fc_df)} rows")
assert len(fc_df) == 30, f"Expected 30, got {len(fc_df)}"
assert len(hist_df) == 60, f"Expected 60, got {len(hist_df)}"
assert (fc_df['predicted_demand'] >= 0).all(), "Negative predictions found!"
assert (fc_df['lower_bound'] <= fc_df['predicted_demand']).all(), "Lower bound > predicted!"
assert (fc_df['upper_bound'] >= fc_df['predicted_demand']).all(), "Upper bound < predicted!"
print("  All assertions passed (non-negative, bounded uncertainty cones).")
print("\nFirst 5 days sample forecast:")
print(fc_df[['date', 'predicted_demand', 'lower_bound', 'upper_bound']].head().to_string(index=False))

# 2. Test Champion Metadata
print("\n[Test 2] Verifying Champion Model Selection Metadata...")
champ_json_path = os.path.join("models", "forecasting", "champion_info.json")
assert os.path.exists(champ_json_path), "champion_info.json does not exist!"
with open(champ_json_path, "r") as f:
    champ_info = json.load(f)
assert champ_info["champion_model"] == "Linear Regression", f"Expected Linear Regression, got {champ_info['champion_model']}"
assert len(champ_info["leaderboard"]) == 3, f"Expected 3 models in leaderboard, got {len(champ_info['leaderboard'])}"
models_in_board = [m["model_name"] for m in champ_info["leaderboard"]]
print("  Models in comparative leaderboard:", models_in_board)
assert set(models_in_board) == {"Linear Regression", "Random Forest", "XGBoost"}
print("  Designated Champion:", champ_info["champion_model"])
print(f"  Test WAPE: {champ_info['test_metrics']['WAPE']}%")

# 3. Test model_evaluations.csv has ZERO Naive baselines
print("\n[Test 3] Verifying model_evaluations.csv for Forecasting module...")
for p in ["data/outputs/model_evaluations.csv", "data/model_evaluations.csv"]:
    df_evals = pd.read_csv(p)
    fc_models = df_evals[df_evals["module"].str.lower() == "forecasting"]["model_name"].unique().tolist()
    print(f"  {p} forecasting models: {fc_models}")
    assert "Naive Baseline" not in fc_models, "Naive Baseline found in evals!"
    assert "Seasonal Naive (Lag 7)" not in fc_models, "Seasonal Naive found in evals!"
    assert set(fc_models) == {"Linear Regression", "Random Forest", "XGBoost"}, f"Unexpected models: {fc_models}"

# 4. Test API Router Endpoints
print("\n[Test 4] Testing FastAPI /api/forecasting/predict...")
api_res = predict_demand("MED001", 14)
assert api_res["champion"]["model_name"] == "Linear Regression"
assert len(api_res["forecast"]) == 14
assert len(api_res["benchmarks"]) == 3
assert [b["model_name"] for b in api_res["benchmarks"]] == ["Linear Regression", "XGBoost", "Random Forest"]
assert api_res["benchmarks"][0]["status"] == "CHAMPION"
assert api_res["benchmarks"][1]["status"] == "BENCHMARK"
assert api_res["benchmarks"][2]["status"] == "BENCHMARK"
print("  API response validated successfully.")

print("\n[Test 5] Testing Performance Router Output...")
perf = get_performance_metrics()
perf_fc_models = [r["model_name"] for r in perf["forecasting"]]
print("  Performance router models:", perf_fc_models)
assert set(perf_fc_models) == {"Linear Regression", "Random Forest", "XGBoost"}

print("\n" + "=" * 70)
print("ALL VERIFICATION SUITE CHECKS PASSED (100% CORRECT & COMPLIANT)!")
print("=" * 70)
