import os
import base64

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fig_dir = os.path.join(PROJECT_ROOT, "reports", "figures", "forecasting")

def get_b64(fname):
    p = os.path.join(fig_dir, fname)
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

c1 = get_b64("metric_comparison_leaderboard.png")
c2 = get_b64("actual_vs_predicted_time_series.png")
c3 = get_b64("residual_error_distribution.png")
c4 = get_b64("cumulative_absolute_error_progression.png")
c5 = get_b64("champion_selection_matrix.png")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PharmaPulse — 3-Model Demand Forecasting Comparative Analysis</title>
  <style>
    body {{
      margin: 0;
      padding: 30px;
      background: #0B1120;
      color: #F8FAFC;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    .header {{
      background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(15, 23, 42, 0.8) 100%);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 24px 30px;
      border-radius: 12px;
      margin-bottom: 28px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      background: rgba(56, 189, 248, 0.15);
      border: 1px solid #38BDF8;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
      color: #38BDF8;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 26px;
      color: #FFFFFF;
    }}
    p.subtitle {{
      margin: 0;
      font-size: 14px;
      color: #94A3B8;
      line-height: 1.5;
    }}
    .card {{
      background: #0F172A;
      border: 1px solid #1E293B;
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 26px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }}
    .card-title {{
      font-size: 16px;
      font-weight: 700;
      color: #38BDF8;
      margin-bottom: 6px;
    }}
    .card-desc {{
      font-size: 13px;
      color: #94A3B8;
      margin-bottom: 18px;
      line-height: 1.45;
    }}
    img.chart {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid #1E293B;
      display: block;
    }}
    table.leaderboard {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      font-size: 14px;
    }}
    table.leaderboard th {{
      background: #1E293B;
      color: #38BDF8;
      text-align: left;
      padding: 12px 14px;
      font-weight: 700;
      border-bottom: 2px solid #334155;
    }}
    table.leaderboard td {{
      padding: 12px 14px;
      border-bottom: 1px solid #1E293B;
      color: #F1F5F9;
    }}
    tr.champion-row {{
      background: rgba(14, 165, 233, 0.12);
      border-left: 4px solid #38BDF8;
    }}
    .pill-champ {{
      background: rgba(52, 211, 153, 0.15);
      border: 1px solid #34D399;
      color: #34D399;
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 11px;
    }}
    .pill-bench {{
      background: rgba(148, 163, 184, 0.15);
      border: 1px solid #64748B;
      color: #94A3B8;
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 11px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div class="badge">PharmaPulse Data Science Suite</div>
      <h1>3-Candidate Demand Forecasting Architecture Comparative Study</h1>
      <p class="subtitle">
        Empirical evaluation of <strong>Linear Regression</strong> vs. <strong>Random Forest Regressor</strong> vs. <strong>XGBoost Regressor</strong> on 165-day out-of-sample holdout test partition across 150 medicine SKUs.
      </p>
    </div>

    <!-- Leaderboard Table Card -->
    <div class="card">
      <div class="card-title">Empirical Benchmark Leaderboard (Out-of-Sample Test Split — 165 Days)</div>
      <div class="card-desc">Evaluated strictly on chronological holdout data with zero future leakage. Primary decision metric: Test WAPE.</div>
      <table class="leaderboard">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Architecture</th>
            <th>Test WAPE (%)</th>
            <th>Test MAE (Units)</th>
            <th>Test RMSE</th>
            <th>Test R² Score</th>
            <th>Validation WAPE (%)</th>
            <th>Decision</th>
          </tr>
        </thead>
        <tbody>
          <tr class="champion-row">
            <td style="font-weight: 700; color: #38BDF8;">1</td>
            <td style="font-weight: 700; color: #38BDF8;">Linear Regression</td>
            <td style="font-weight: 700; color: #34D399;">53.24%</td>
            <td>11.289</td>
            <td>19.586</td>
            <td>0.1546</td>
            <td>51.68%</td>
            <td><span class="pill-champ">CHAMPION (Selected)</span></td>
          </tr>
          <tr>
            <td>2</td>
            <td style="font-weight: 600;">XGBoost Regressor</td>
            <td>53.82%</td>
            <td>11.411</td>
            <td>19.730</td>
            <td>0.1420</td>
            <td>52.32%</td>
            <td><span class="pill-bench">BENCHMARK</span></td>
          </tr>
          <tr>
            <td>3</td>
            <td style="font-weight: 600;">Random Forest Regressor</td>
            <td>54.13%</td>
            <td>11.477</td>
            <td>19.738</td>
            <td>0.1413</td>
            <td>52.60%</td>
            <td><span class="pill-bench">BENCHMARK</span></td>
          </tr>
        </tbody>
      </table>
    </div>

    <!-- Chart 1 -->
    <div class="card">
      <div class="card-title">Chart 1: Multi-Metric Benchmark Comparison Leaderboard</div>
      <div class="card-desc">Side-by-side grouped bars for WAPE, MAE, RMSE, and R² comparing all three candidate architectures.</div>
      <img class="chart" src="data:image/png;base64,{c1}" alt="Metric Comparison Leaderboard">
    </div>

    <!-- Chart 2 -->
    <div class="card">
      <div class="card-title">Chart 2: Ground Truth Actual vs. Predicted Demand Time Series Overlay</div>
      <div class="card-desc">165-day continuous daily demand tracking Ground Truth historical sales against Linear Regression, Random Forest, and XGBoost predictions.</div>
      <img class="chart" src="data:image/png;base64,{c2}" alt="Actual vs Predicted Time Series">
    </div>

    <!-- Chart 3 -->
    <div class="card">
      <div class="card-title">Chart 3: Residual / Prediction Error Distribution & Forecast Bias Analysis</div>
      <div class="card-desc">Error histograms and KDE density curves verifying error symmetry and near-zero mean prediction bias for all 3 models.</div>
      <img class="chart" src="data:image/png;base64,{c3}" alt="Residual Error Distribution">
    </div>

    <!-- Chart 4 -->
    <div class="card">
      <div class="card-title">Chart 4: Cumulative Absolute Tracking Error Progression</div>
      <div class="card-desc">Cumulative error accumulation over 24,750 time-ordered test observations proving long-horizon stability.</div>
      <img class="chart" src="data:image/png;base64,{c4}" alt="Cumulative Error Progression">
    </div>

    <!-- Chart 5 -->
    <div class="card">
      <div class="card-title">Chart 5: Final Empirical Decision Matrix & Champion Declaration</div>
      <div class="card-desc">Visual scorecard summary declaring Linear Regression as the operational champion.</div>
      <img class="chart" src="data:image/png;base64,{c5}" alt="Champion Selection Matrix">
    </div>
  </div>
</body>
</html>
"""

html_path = os.path.join(PROJECT_ROOT, "reports", "forecasting_model_comparison.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Generated standalone visual HTML report: {html_path}")
