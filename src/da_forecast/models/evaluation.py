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
    return prices.shift(168)


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
