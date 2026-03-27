"""Calendar features for electricity price forecasting.

Uses sin/cos encoding for cyclical features (hour, weekday) so that
adjacent values are close in feature space (e.g., hour 23 is near hour 0).

Includes Danish public holidays -- holiday load profiles resemble Sundays.
"""
import numpy as np
import pandas as pd
import holidays

def compute_calendar_features(df: pd.DataFrame, country: str = "DK") -> pd.DataFrame:
    result = df.copy()
    idx = df.index
    idx_cet = idx.tz_convert("Europe/Copenhagen")
    hour = idx_cet.hour
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    weekday = idx_cet.weekday
    result["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    result["is_weekend"] = weekday >= 5
    years = sorted(set(idx.year))
    dk_holidays = holidays.country_holidays(country, years=years)
    result["is_holiday"] = pd.Series([d.date() in dk_holidays for d in idx], index=idx)
    return result
