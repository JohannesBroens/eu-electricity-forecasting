"""Weather-related features for electricity price forecasting.

Uses ENTSO-E wind/solar generation forecasts (MW) as a proxy for raw weather
data. In production, raw weather forecasts from DMI/ECMWF would be converted
to expected generation.
"""
import pandas as pd
import numpy as np

def compute_weather_features(generation_forecast: pd.DataFrame, generation_actuals: pd.DataFrame | None = None) -> pd.DataFrame:
    result = generation_forecast.copy()
    wind_cols = [c for c in result.columns if "wind" in c.lower() and "actual" not in c.lower()]
    result["wind_total_mw"] = result[wind_cols].sum(axis=1)
    wind_max = result["wind_total_mw"].rolling(window=168, min_periods=24).max()
    result["wind_capacity_factor"] = np.where(wind_max > 0, result["wind_total_mw"] / wind_max, 0)
    if generation_actuals is not None and "wind_actual_mw" in generation_actuals.columns:
        error = generation_actuals["wind_actual_mw"] - result["wind_total_mw"]
        result["wind_forecast_error_lag1"] = error.shift(24)
    else:
        result["wind_forecast_error_lag1"] = np.nan
    return result
