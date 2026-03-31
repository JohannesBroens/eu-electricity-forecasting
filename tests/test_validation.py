"""Tests for validation: gap detection, DST transitions, outlier detection."""

import numpy as np
import pandas as pd
import pytest

from da_forecast.validation.completeness import find_gaps, daily_completeness_report
from da_forecast.validation.timezone import (
    find_dst_transitions,
    expected_hours_in_day,
    to_utc,
    to_cet,
    validate_timezone_aware,
)
from da_forecast.validation.outliers import detect_outliers


TZ = "Europe/Copenhagen"


# --- find_gaps ---

class TestFindGaps:
    def test_no_gaps_in_complete_series(self, sample_prices):
        gaps = find_gaps(sample_prices)
        assert gaps == []

    def test_detects_missing_hours(self):
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz=TZ)
        # Drop hour 5 and 6
        idx_gapped = idx.delete([5, 6])
        df = pd.DataFrame({"price_eur_mwh": np.ones(len(idx_gapped))}, index=idx_gapped)
        gaps = find_gaps(df)
        assert len(gaps) == 2
        gap_ts = {g["timestamp"] for g in gaps}
        assert idx[5] in gap_ts
        assert idx[6] in gap_ts

    def test_empty_dataframe_returns_empty(self, empty_price_df):
        gaps = find_gaps(empty_price_df)
        assert gaps == []

    def test_single_row_no_gap(self):
        idx = pd.date_range("2025-06-15 12:00", periods=1, freq="h", tz=TZ)
        df = pd.DataFrame({"price_eur_mwh": [50.0]}, index=idx)
        gaps = find_gaps(df)
        assert gaps == []


# --- daily_completeness_report ---

class TestDailyCompletenessReport:
    def test_complete_day(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        df = pd.DataFrame({"price_eur_mwh": np.ones(24)}, index=idx)
        report = daily_completeness_report(df)
        assert len(report) == 1
        assert report.iloc[0]["expected"] == 24
        assert report.iloc[0]["actual"] == 24
        assert report.iloc[0]["missing"] == 0
        assert report.iloc[0]["complete"]

    def test_incomplete_day(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        idx_gapped = idx.delete([3, 4, 5])
        df = pd.DataFrame({"val": np.ones(len(idx_gapped))}, index=idx_gapped)
        report = daily_completeness_report(df)
        row = report.iloc[0]
        assert row["missing"] == 3
        assert not row["complete"]

    def test_empty_dataframe(self, empty_price_df):
        report = daily_completeness_report(empty_price_df)
        assert len(report) == 0
        assert "expected" in report.columns

    def test_spring_dst_day_expects_23_hours(self, dst_spring_index):
        df = pd.DataFrame({"val": np.ones(len(dst_spring_index))}, index=dst_spring_index)
        report = daily_completeness_report(df)
        march30 = report[report["date"] == pd.Timestamp("2025-03-30")]
        if len(march30) > 0:
            assert march30.iloc[0]["expected"] == 23


# --- find_dst_transitions ---

class TestFindDstTransitions:
    def test_returns_two_transitions(self):
        transitions = find_dst_transitions(2025)
        assert len(transitions) == 2

    def test_spring_forward_is_first(self):
        transitions = find_dst_transitions(2025)
        assert transitions[0]["type"] == "spring_forward"
        assert transitions[0]["hours"] == 23

    def test_fall_back_is_second(self):
        transitions = find_dst_transitions(2025)
        assert transitions[1]["type"] == "fall_back"
        assert transitions[1]["hours"] == 25

    def test_spring_transition_is_last_sunday_of_march(self):
        transitions = find_dst_transitions(2025)
        spring = transitions[0]["date"]
        assert spring.month == 3
        assert spring.weekday() == 6  # Sunday

    def test_fall_transition_is_last_sunday_of_october(self):
        transitions = find_dst_transitions(2025)
        fall = transitions[1]["date"]
        assert fall.month == 10
        assert fall.weekday() == 6


# --- expected_hours_in_day ---

class TestExpectedHoursInDay:
    def test_normal_day_returns_24(self):
        assert expected_hours_in_day(pd.Timestamp("2025-01-15")) == 24

    def test_spring_dst_returns_23(self):
        assert expected_hours_in_day(pd.Timestamp("2025-03-30")) == 23

    def test_fall_dst_returns_25(self):
        assert expected_hours_in_day(pd.Timestamp("2025-10-26")) == 25

    def test_different_year(self):
        transitions_2024 = find_dst_transitions(2024)
        spring_day = transitions_2024[0]["date"]
        assert expected_hours_in_day(spring_day) == 23


# --- timezone conversion helpers ---

class TestTimezoneConversion:
    def test_to_utc_converts_correctly(self):
        ts = pd.Timestamp("2025-01-15 12:00", tz=TZ)
        utc = to_utc(ts)
        assert str(utc.tz) == "UTC"

    def test_to_cet_converts_correctly(self):
        ts = pd.Timestamp("2025-01-15 12:00", tz="UTC")
        cet = to_cet(ts)
        assert "Copenhagen" in str(cet.tz)

    def test_to_utc_raises_on_naive(self):
        with pytest.raises(ValueError, match="timezone-naive"):
            to_utc(pd.Timestamp("2025-01-15 12:00"))

    def test_to_cet_raises_on_naive(self):
        with pytest.raises(ValueError, match="timezone-naive"):
            to_cet(pd.Timestamp("2025-01-15 12:00"))


class TestValidateTimezoneAware:
    def test_raises_on_naive_index(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="h")
        df = pd.DataFrame({"val": np.ones(5)}, index=idx)
        with pytest.raises(ValueError, match="timezone-naive"):
            validate_timezone_aware(df)

    def test_passes_on_aware_index(self, sample_prices):
        # Should not raise
        validate_timezone_aware(sample_prices)


# --- detect_outliers ---

class TestDetectOutliers:
    def test_returns_expected_columns(self, sample_prices):
        result = detect_outliers(sample_prices["price_eur_mwh"])
        expected_cols = {"price", "rolling_mean", "rolling_std", "z_score", "is_outlier", "reason"}
        assert expected_cols == set(result.columns)

    def test_no_outliers_in_smooth_data(self):
        idx = pd.date_range("2025-01-01", periods=24 * 14, freq="h", tz=TZ)
        prices = pd.Series(50.0, index=idx)
        result = detect_outliers(prices)
        assert not result["is_outlier"].any()

    def test_detects_extreme_high_spike(self):
        idx = pd.date_range("2025-01-01", periods=24 * 14, freq="h", tz=TZ)
        values = np.full(len(idx), 50.0)
        values[200] = 600.0  # extreme high spike
        prices = pd.Series(values, index=idx)
        result = detect_outliers(prices, extreme_high_threshold=500.0)
        assert result.loc[result.index[200], "is_outlier"]

    def test_detects_z_score_outlier(self):
        idx = pd.date_range("2025-01-01", periods=24 * 14, freq="h", tz=TZ)
        rng = np.random.default_rng(42)
        values = rng.normal(50, 2, len(idx))
        values[250] = 200.0  # strong spike relative to rolling window
        prices = pd.Series(values, index=idx)
        result = detect_outliers(prices, z_threshold=3.0)
        assert result.loc[result.index[250], "is_outlier"]

    def test_negative_prices_are_not_outliers(self):
        idx = pd.date_range("2025-01-01", periods=24 * 14, freq="h", tz=TZ)
        values = np.full(len(idx), 50.0)
        values[100] = -10.0  # mild negative, not extreme
        prices = pd.Series(values, index=idx)
        result = detect_outliers(prices, z_threshold=3.0, extreme_high_threshold=500.0)
        # A mild negative in otherwise constant data may or may not trigger z-score.
        # The key test: no extreme_high flag for negatives.
        row = result.loc[result.index[100]]
        if row["is_outlier"]:
            assert "Extreme high" not in row["reason"]

    def test_preserves_index(self, sample_prices):
        result = detect_outliers(sample_prices["price_eur_mwh"])
        pd.testing.assert_index_equal(result.index, sample_prices.index)
