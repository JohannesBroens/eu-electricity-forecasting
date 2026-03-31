"""Tests for ParquetCache save/load/merge."""

import numpy as np
import pandas as pd
import pytest

from da_forecast.sources.cache import ParquetCache


@pytest.fixture
def cache(tmp_path):
    return ParquetCache(tmp_path)


@pytest.fixture
def sample_df():
    idx = pd.date_range("2025-01-01", periods=48, freq="h", tz="Europe/Copenhagen")
    return pd.DataFrame({"price_eur_mwh": np.arange(48, dtype=float)}, index=idx)


class TestParquetCacheSaveLoad:
    def test_save_and_load_roundtrip(self, cache, sample_df):
        cache.save("energinet", "DK_1", "day_ahead_prices", sample_df)
        loaded = cache.load("energinet", "DK_1", "day_ahead_prices")
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, sample_df)

    def test_load_nonexistent_returns_none(self, cache):
        result = cache.load("energinet", "DK_1", "day_ahead_prices")
        assert result is None

    def test_save_creates_nested_directories(self, cache, sample_df, tmp_path):
        cache.save("entsoe", "NO_1", "wind_solar_forecast", sample_df)
        expected_path = tmp_path / "entsoe" / "NO_1" / "wind_solar_forecast.parquet"
        assert expected_path.exists()

    def test_load_preserves_timezone(self, cache, sample_df):
        cache.save("energinet", "DK_1", "prices", sample_df)
        loaded = cache.load("energinet", "DK_1", "prices")
        assert loaded.index.tz is not None
        assert str(loaded.index.tz) == "Europe/Copenhagen"

    def test_load_infers_frequency(self, cache, sample_df):
        cache.save("energinet", "DK_1", "prices", sample_df)
        loaded = cache.load("energinet", "DK_1", "prices")
        assert loaded.index.freq is not None

    def test_save_overwrites_existing(self, cache, sample_df):
        cache.save("energinet", "DK_1", "prices", sample_df)
        new_df = sample_df.copy()
        new_df["price_eur_mwh"] = 999.0
        cache.save("energinet", "DK_1", "prices", new_df)
        loaded = cache.load("energinet", "DK_1", "prices")
        assert (loaded["price_eur_mwh"] == 999.0).all()


class TestParquetCacheMerge:
    def test_merge_into_empty_cache(self, cache, sample_df):
        cache.merge("energinet", "DK_1", "prices", sample_df)
        loaded = cache.load("energinet", "DK_1", "prices")
        assert loaded is not None
        assert len(loaded) == len(sample_df)

    def test_merge_appends_new_data(self, cache):
        idx1 = pd.date_range("2025-01-01", periods=24, freq="h", tz="Europe/Copenhagen")
        df1 = pd.DataFrame({"price_eur_mwh": np.arange(24, dtype=float)}, index=idx1)

        idx2 = pd.date_range("2025-01-02", periods=24, freq="h", tz="Europe/Copenhagen")
        df2 = pd.DataFrame({"price_eur_mwh": np.arange(100, 124, dtype=float)}, index=idx2)

        cache.save("energinet", "DK_1", "prices", df1)
        cache.merge("energinet", "DK_1", "prices", df2)
        loaded = cache.load("energinet", "DK_1", "prices")
        assert len(loaded) == 48
        assert loaded.index.is_monotonic_increasing

    def test_merge_overwrites_overlapping_timestamps(self, cache):
        idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="Europe/Copenhagen")
        df_old = pd.DataFrame({"price_eur_mwh": np.zeros(24)}, index=idx)
        df_new = pd.DataFrame({"price_eur_mwh": np.ones(24)}, index=idx)

        cache.save("energinet", "DK_1", "prices", df_old)
        cache.merge("energinet", "DK_1", "prices", df_new)
        loaded = cache.load("energinet", "DK_1", "prices")
        assert len(loaded) == 24
        assert (loaded["price_eur_mwh"] == 1.0).all()

    def test_merge_partial_overlap(self, cache):
        idx1 = pd.date_range("2025-01-01", periods=24, freq="h", tz="Europe/Copenhagen")
        df1 = pd.DataFrame({"price_eur_mwh": np.zeros(24)}, index=idx1)

        idx2 = pd.date_range("2025-01-01 12:00", periods=24, freq="h", tz="Europe/Copenhagen")
        df2 = pd.DataFrame({"price_eur_mwh": np.ones(24)}, index=idx2)

        cache.save("energinet", "DK_1", "prices", df1)
        cache.merge("energinet", "DK_1", "prices", df2)
        loaded = cache.load("energinet", "DK_1", "prices")
        # 12 from old (non-overlapping) + 24 from new
        assert len(loaded) == 36
        assert loaded.index.is_monotonic_increasing


class TestGetCachedRange:
    def test_returns_none_when_no_data(self, cache):
        result = cache.get_cached_range("energinet", "DK_1", "prices")
        assert result is None

    def test_returns_correct_range(self, cache, sample_df):
        cache.save("energinet", "DK_1", "prices", sample_df)
        result = cache.get_cached_range("energinet", "DK_1", "prices")
        assert result is not None
        assert result[0] == sample_df.index.min()
        assert result[1] == sample_df.index.max()

    def test_returns_none_for_empty_dataframe(self, cache):
        idx = pd.DatetimeIndex([], dtype="datetime64[ns, Europe/Copenhagen]")
        df = pd.DataFrame({"price_eur_mwh": pd.Series(dtype=float)}, index=idx)
        cache.save("energinet", "DK_1", "prices", df)
        result = cache.get_cached_range("energinet", "DK_1", "prices")
        assert result is None
