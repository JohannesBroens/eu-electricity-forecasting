"""Outlier detection for electricity price data.

Domain notes:
- Negative prices are valid (wind/solar surplus exceeds demand).
- Extreme positive spikes (>500 EUR/MWh) may be real scarcity events;
  flag for review but do not auto-remove.
"""
import pandas as pd
import numpy as np

def detect_outliers(prices: pd.Series, window_days: int = 7, z_threshold: float = 3.0, extreme_high_threshold: float = 500.0) -> pd.DataFrame:
    window = window_days * 24
    rolling_mean = prices.rolling(window=window, min_periods=24, center=False).mean()
    rolling_std = prices.rolling(window=window, min_periods=24, center=False).std()
    rolling_std = rolling_std.replace(0, np.nan)
    z_score = (prices - rolling_mean) / rolling_std

    result = pd.DataFrame({
        "price": prices, "rolling_mean": rolling_mean, "rolling_std": rolling_std,
        "z_score": z_score, "is_outlier": False, "reason": "",
    }, index=prices.index)

    positive_spike_mask = z_score > z_threshold
    result.loc[positive_spike_mask, "is_outlier"] = True
    result.loc[positive_spike_mask, "reason"] = f"Positive spike: z-score > {z_threshold}"

    extreme_mask = prices > extreme_high_threshold
    result.loc[extreme_mask, "is_outlier"] = True
    result.loc[extreme_mask, "reason"] = f"Extreme high spike: > {extreme_high_threshold} EUR/MWh"

    return result
