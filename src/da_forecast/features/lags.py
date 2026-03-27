"""Lagged price features for electricity price forecasting.

All lags respect the day-ahead gate closure constraint: the model for
delivery day D can only use data available before 12:00 CET on D-1.
"""
import pandas as pd

def compute_lag_features(df: pd.DataFrame, price_col: str = "price_eur_mwh") -> pd.DataFrame:
    result = df[[price_col]].copy()
    prices = df[price_col]
    result["price_lag_1d"] = prices.shift(24)
    result["price_lag_2d"] = prices.shift(48)
    result["price_lag_7d"] = prices.shift(168)
    result["price_rolling_7d_mean"] = prices.shift(24).rolling(window=168, min_periods=24).mean()
    result["price_rolling_7d_std"] = prices.shift(24).rolling(window=168, min_periods=24).std()
    result["price_rolling_24h_min"] = prices.shift(24).rolling(window=24, min_periods=24).min()
    result["price_rolling_24h_max"] = prices.shift(24).rolling(window=24, min_periods=24).max()
    return result
