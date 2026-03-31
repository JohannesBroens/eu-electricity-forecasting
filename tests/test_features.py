"""Tests for feature engineering: lags, calendar, fundamental."""

import numpy as np
import pandas as pd
import pytest

from da_forecast.features.lags import compute_lag_features
from da_forecast.features.calendar import compute_calendar_features
from da_forecast.features.fundamental import compute_fundamental_features


TZ = "Europe/Copenhagen"


# --- compute_lag_features ---

class TestLagFeatures:
    def test_output_columns(self, sample_prices):
        result = compute_lag_features(sample_prices)
        expected = {
            "price_eur_mwh", "price_lag_1d", "price_lag_2d", "price_lag_7d",
            "price_rolling_7d_mean", "price_rolling_7d_std",
            "price_rolling_24h_min", "price_rolling_24h_max",
        }
        assert expected == set(result.columns)

    def test_lag_1d_is_24h_shift(self, sample_prices):
        result = compute_lag_features(sample_prices)
        # The 24th value of price_lag_1d should equal the 0th price
        expected_val = sample_prices["price_eur_mwh"].iloc[0]
        actual_val = result["price_lag_1d"].iloc[24]
        assert actual_val == pytest.approx(expected_val)

    def test_lag_2d_is_48h_shift(self, sample_prices):
        result = compute_lag_features(sample_prices)
        expected_val = sample_prices["price_eur_mwh"].iloc[0]
        actual_val = result["price_lag_2d"].iloc[48]
        assert actual_val == pytest.approx(expected_val)

    def test_lag_7d_is_168h_shift(self, sample_prices):
        result = compute_lag_features(sample_prices)
        # With 168 hours of data, position 168 doesn't exist (0-167),
        # but position 167 should have NaN for 7d lag
        assert result["price_lag_7d"].isna().sum() == 168  # all NaN for 1-week data

    def test_first_24_hours_have_nan_lag_1d(self, sample_prices):
        result = compute_lag_features(sample_prices)
        assert result["price_lag_1d"].iloc[:24].isna().all()
        assert result["price_lag_1d"].iloc[24:].notna().all()

    def test_preserves_index(self, sample_prices):
        result = compute_lag_features(sample_prices)
        pd.testing.assert_index_equal(result.index, sample_prices.index)

    def test_rolling_7d_mean_has_nans_at_start(self, sample_prices):
        result = compute_lag_features(sample_prices)
        # shift(24) + rolling(168, min_periods=24): first 47 values are NaN
        assert result["price_rolling_7d_mean"].iloc[:47].isna().all()

    def test_custom_price_column(self):
        idx = pd.date_range("2025-01-01", periods=48, freq="h", tz=TZ)
        df = pd.DataFrame({"my_price": np.arange(48, dtype=float)}, index=idx)
        result = compute_lag_features(df, price_col="my_price")
        assert "price_lag_1d" in result.columns
        assert "my_price" in result.columns


# --- compute_calendar_features ---

class TestCalendarFeatures:
    def test_output_columns(self, sample_prices):
        result = compute_calendar_features(sample_prices)
        expected_new = {"hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
                        "is_weekend", "is_holiday"}
        assert expected_new.issubset(set(result.columns))

    def test_hour_sin_cos_range(self, sample_prices):
        result = compute_calendar_features(sample_prices)
        assert result["hour_sin"].between(-1, 1).all()
        assert result["hour_cos"].between(-1, 1).all()

    def test_weekday_sin_cos_range(self, sample_prices):
        result = compute_calendar_features(sample_prices)
        assert result["weekday_sin"].between(-1, 1).all()
        assert result["weekday_cos"].between(-1, 1).all()

    def test_is_weekend_boolean(self, sample_prices):
        result = compute_calendar_features(sample_prices)
        # sample_prices starts on Monday 2025-01-06, so Saturday is day 5 (index 120-143)
        # and Sunday is day 6 (index 144-167)
        idx_cet = result.index.tz_convert(TZ)
        weekend_mask = idx_cet.weekday >= 5
        assert (result["is_weekend"] == weekend_mask).all()

    def test_holiday_detection(self):
        # January 1 is a holiday in Denmark
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz=TZ)
        df = pd.DataFrame({"price_eur_mwh": np.ones(24)}, index=idx)
        result = compute_calendar_features(df, country="DK")
        assert result["is_holiday"].all()

    def test_non_holiday_detection(self):
        # January 6 (Monday) is not a holiday in Denmark
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        df = pd.DataFrame({"price_eur_mwh": np.ones(24)}, index=idx)
        result = compute_calendar_features(df, country="DK")
        assert not result["is_holiday"].any()

    def test_preserves_original_columns(self, sample_prices):
        result = compute_calendar_features(sample_prices)
        assert "price_eur_mwh" in result.columns

    def test_hour_sin_midnight_near_zero(self):
        # At hour 0 in CET, sin(2*pi*0/24) = 0
        idx = pd.date_range("2025-01-06 00:00", periods=1, freq="h", tz=TZ)
        df = pd.DataFrame({"val": [1.0]}, index=idx)
        result = compute_calendar_features(df)
        # CET hour 0 -> hour_sin should be ~0
        assert abs(result["hour_sin"].iloc[0]) < 0.01


# --- compute_fundamental_features ---

class TestFundamentalFeatures:
    def test_residual_load_calculation(self, sample_load, sample_wind, sample_solar):
        result = compute_fundamental_features(sample_load, sample_wind, sample_solar)
        expected_residual = sample_load["load_mw"] - sample_wind["wind_mw"] - sample_solar["solar_mw"]
        pd.testing.assert_series_equal(
            result["residual_load_mw"], expected_residual, check_names=False
        )

    def test_output_columns(self, sample_load, sample_wind, sample_solar):
        result = compute_fundamental_features(sample_load, sample_wind, sample_solar)
        expected = {"load_mw", "wind_mw", "solar_mw", "residual_load_mw"}
        assert expected == set(result.columns)

    def test_index_matches_load(self, sample_load, sample_wind, sample_solar):
        result = compute_fundamental_features(sample_load, sample_wind, sample_solar)
        pd.testing.assert_index_equal(result.index, sample_load.index)

    def test_misaligned_indices_handled(self):
        idx1 = pd.date_range("2025-01-01", periods=24, freq="h", tz=TZ)
        idx2 = pd.date_range("2025-01-01 06:00", periods=24, freq="h", tz=TZ)
        load = pd.DataFrame({"load_mw": np.ones(24) * 3000}, index=idx1)
        wind = pd.DataFrame({"wind_mw": np.ones(24) * 500}, index=idx2)
        solar = pd.DataFrame({"solar_mw": np.ones(24) * 100}, index=idx2)
        result = compute_fundamental_features(load, wind, solar)
        # First 6 hours should have NaN wind/solar due to misalignment
        assert result["wind_mw"].iloc[:6].isna().all()
        # Overlapping hours should have valid values
        assert result["wind_mw"].iloc[6:].notna().all()
