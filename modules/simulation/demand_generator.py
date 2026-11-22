"""
PharmaPulse - Simulation Demand Generator
Handles stochastic demand generation using Negative Binomial count distributions,
calendar factors, annual/seasonal cycles, long-term trends, and irregular burst dynamics.
"""

import math
import random
import numpy as np

def generate_daily_demand(base_demand, pattern, cat_id, day_idx, total_days, dow, month):
    """
    Generates stochastic customer demand strictly independent of available inventory.
    """
    # 1. Day of week effect
    dow_factor = 1.0 + 0.12 * math.sin(dow * math.pi / 3.0)

    # 2. Seasonality effect
    season_factor = 1.0
    if pattern == "Seasonal":
        if cat_id in ["CAT001", "CAT003"]:  # Cold & Fever winter peak (Dec-Feb)
            season_factor = 1.0 + 0.50 * math.cos(2 * math.pi * (month - 1) / 12.0)
        elif cat_id in ["CAT004", "CAT006"]:  # GI / Hydration summer peak (May-Jul)
            season_factor = 1.0 + 0.45 * math.sin(2 * math.pi * (month - 3) / 12.0)
        else:
            season_factor = 1.0 + 0.35 * math.sin(2 * math.pi * (month - 1) / 12.0)

    # 3. Trend effect
    trend_factor = 1.0
    if pattern == "Growing":
        trend_factor = 1.0 + 0.70 * (day_idx / total_days)
    elif pattern == "Declining":
        trend_factor = max(0.35, 1.0 - 0.50 * (day_idx / total_days))

    # 4. Irregular bursts or zero lulls
    irregular_multiplier = 1.0
    if pattern == "Irregular":
        r_val = random.random()
        if r_val < 0.30:
            irregular_multiplier = 0.1
        elif r_val > 0.88:
            irregular_multiplier = float(random.uniform(2.5, 4.2))
        else:
            irregular_multiplier = float(random.uniform(0.6, 1.4))

    # 5. Outbreak / random spikes
    spike_factor = 1.0
    if random.random() < 0.025:
        spike_factor = float(random.uniform(1.8, 3.2))

    expected_demand = base_demand * dow_factor * season_factor * trend_factor * irregular_multiplier * spike_factor
    expected_demand = max(0.1, expected_demand)

    # Negative Binomial count sampling (var > mean)
    alpha = 0.20 if pattern != "Irregular" else 0.55
    mean_d = expected_demand
    var_d = mean_d + alpha * (mean_d ** 2)
    p_nb = mean_d / var_d
    n_nb = (mean_d ** 2) / (var_d - mean_d)

    actual_demand = int(np.random.negative_binomial(n=max(0.5, n_nb), p=min(0.99, max(0.01, p_nb))))
    if pattern == "Irregular" and irregular_multiplier < 0.2 and random.random() < 0.65:
        actual_demand = 0

    return actual_demand
