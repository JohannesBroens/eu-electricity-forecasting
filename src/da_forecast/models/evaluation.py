"""Model evaluation metrics for electricity price forecasting.

Uses sMAPE instead of standard MAPE because electricity prices frequently
hit zero or go negative, making standard MAPE undefined or misleading.
"""
import numpy as np
import pandas as pd


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denominator > 0
    if not mask.any():
        return 0.0
    return float(100 * np.mean(numerator[mask] / denominator[mask]))


def naive_baseline(prices: pd.Series) -> pd.Series:
    """Similar-day reference price (standard in EPF literature).

    - Tuesday through Friday: same hour yesterday (D-1, shift 24)
    - Saturday, Sunday, Monday: same hour last week (D-7, shift 168)

    This is NOT "market consensus" -- it's the simplest reasonable
    reference that a trader could construct from public data. Real
    market expectations would be harder to beat.

    Source: Lago et al. (2021), "Forecasting day-ahead electricity prices",
    Applied Energy 293. (epftoolbox implementation)
    """
    d1 = prices.shift(24)    # yesterday same hour
    d7 = prices.shift(168)   # last week same hour

    # Use D-1 for Tue-Fri, D-7 for Sat/Sun/Mon
    day_of_week = prices.index.dayofweek  # Mon=0, Sun=6
    use_weekly = (day_of_week == 0) | (day_of_week == 5) | (day_of_week == 6)

    result = d1.copy()
    result[use_weekly] = d7[use_weekly]
    return result


def evaluation_report(
    actual: np.ndarray,
    predicted: np.ndarray,
    hours: np.ndarray | None = None,
) -> dict:
    report = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "smape": smape(actual, predicted),
        "n_samples": len(actual),
    }
    if hours is not None:
        per_hour = {}
        for h in range(24):
            mask = hours == h
            if mask.sum() > 0:
                per_hour[h] = {
                    "mae": mae(actual[mask], predicted[mask]),
                    "rmse": rmse(actual[mask], predicted[mask]),
                    "n_samples": int(mask.sum()),
                }
        report["per_hour"] = per_hour
    return report
