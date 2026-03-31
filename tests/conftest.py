"""Shared fixtures for eu-electricity-forecasting test suite."""

import numpy as np
import pandas as pd
import pytest


TZ = "Europe/Copenhagen"


@pytest.fixture
def hourly_index():
    """168 hours (1 week) of hourly timestamps in Europe/Copenhagen."""
    return pd.date_range(
        "2025-01-06", periods=168, freq="h", tz=TZ
    )


@pytest.fixture
def sample_prices(hourly_index):
    """Realistic-looking price DataFrame with 'price_eur_mwh' column."""
    rng = np.random.default_rng(42)
    base = 40 + 20 * np.sin(2 * np.pi * np.arange(168) / 24)
    noise = rng.normal(0, 5, 168)
    prices = base + noise
    return pd.DataFrame(
        {"price_eur_mwh": prices},
        index=hourly_index,
    )


@pytest.fixture
def sample_prices_with_nan(sample_prices):
    """Price DataFrame with a 3-hour NaN gap."""
    df = sample_prices.copy()
    df.iloc[10:13, 0] = np.nan
    return df


@pytest.fixture
def sample_wind(hourly_index):
    """Wind generation DataFrame with 'wind_mw' column."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {"wind_mw": rng.uniform(200, 2000, len(hourly_index))},
        index=hourly_index,
    )


@pytest.fixture
def sample_solar(hourly_index):
    """Solar generation DataFrame with 'solar_mw' column."""
    rng = np.random.default_rng(11)
    hour = hourly_index.hour
    solar = np.where((hour >= 7) & (hour <= 18), rng.uniform(50, 500, len(hourly_index)), 0.0)
    return pd.DataFrame(
        {"solar_mw": solar},
        index=hourly_index,
    )


@pytest.fixture
def sample_load(hourly_index):
    """Load forecast DataFrame with 'load_mw' column."""
    rng = np.random.default_rng(13)
    base = 3000 + 500 * np.sin(2 * np.pi * np.arange(len(hourly_index)) / 24)
    return pd.DataFrame(
        {"load_mw": base + rng.normal(0, 100, len(hourly_index))},
        index=hourly_index,
    )


@pytest.fixture
def dst_spring_index():
    """48 hours spanning the spring DST transition (last Sunday of March 2025).

    In Europe/Copenhagen, clocks spring forward on 2025-03-30 at 02:00,
    so that day has only 23 hours.
    """
    return pd.date_range(
        "2025-03-29", periods=47, freq="h", tz=TZ
    )


@pytest.fixture
def dst_fall_index():
    """48 hours spanning the fall DST transition (last Sunday of October 2025).

    In Europe/Copenhagen, clocks fall back on 2025-10-26 at 03:00,
    so that day has 25 hours.
    """
    return pd.date_range(
        "2025-10-25", periods=49, freq="h", tz=TZ
    )


@pytest.fixture
def empty_price_df():
    """Empty DataFrame with the expected schema."""
    idx = pd.DatetimeIndex([], dtype="datetime64[ns, Europe/Copenhagen]", freq="h")
    return pd.DataFrame({"price_eur_mwh": pd.Series(dtype=float)}, index=idx)


@pytest.fixture
def feature_matrix(sample_prices):
    """Minimal feature matrix for model training: price + 2 numeric features."""
    rng = np.random.default_rng(99)
    df = sample_prices.copy()
    df["feature_a"] = rng.normal(0, 1, len(df))
    df["feature_b"] = rng.normal(0, 1, len(df))
    return df
