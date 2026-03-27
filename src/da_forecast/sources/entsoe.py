"""ENTSO-E Transparency Platform data source.

Wraps entsoe-py with caching, retry logic, and consistent column naming.
"""
import logging
import time
from pathlib import Path
import pandas as pd
from entsoe import EntsoePandasClient
from da_forecast.config import API_MAX_RETRIES, API_BACKOFF_SECONDS
from da_forecast.sources.cache import ParquetCache

logger = logging.getLogger(__name__)

def _retry(func, *args, **kwargs):
    """Retry API call with exponential backoff."""
    for attempt in range(API_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < API_MAX_RETRIES - 1:
                wait = API_BACKOFF_SECONDS[attempt]
                logger.warning(f"ENTSO-E API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

class EntsoeSource:
    """Fetch data from ENTSO-E Transparency Platform with local caching."""
    def __init__(self, api_key: str, cache_dir: Path | None = None):
        self.client = EntsoePandasClient(api_key=api_key)
        self.cache = ParquetCache(cache_dir) if cache_dir else None

    def _with_cache(self, zone: str, datatype: str, start: pd.Timestamp, end: pd.Timestamp, fetch_fn) -> pd.DataFrame:
        if self.cache:
            cached = self.cache.load("entsoe", zone, datatype)
            if cached is not None:
                cached_range = cached.loc[start:end]
                if not cached_range.empty and cached_range.index.min() <= start and cached_range.index.max() >= end - pd.Timedelta(hours=1):
                    return cached_range
        df = fetch_fn(start, end)
        if self.cache and df is not None and not df.empty:
            self.cache.merge("entsoe", zone, datatype, df)
        return df

    def fetch_day_ahead_prices(self, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        def _fetch(s, e):
            series = _retry(self.client.query_day_ahead_prices, zone, start=s, end=e)
            return pd.DataFrame({"price_eur_mwh": series})
        return self._with_cache(zone, "day_ahead_prices", start, end, _fetch)

    def fetch_load_forecast(self, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        def _fetch(s, e):
            df = _retry(self.client.query_load_forecast, zone, start=s, end=e)
            df.columns = ["load_mw"] if len(df.columns) == 1 else [f"load_mw_{i}" for i in range(len(df.columns))]
            if "load_mw_0" in df.columns:
                df = df.rename(columns={"load_mw_0": "load_mw"})
            return df[["load_mw"]]
        return self._with_cache(zone, "load_forecast", start, end, _fetch)

    def fetch_wind_solar_forecast(self, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        def _fetch(s, e):
            df = _retry(self.client.query_wind_and_solar_forecast, zone, start=s, end=e)
            col_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if "offshore" in col_lower:
                    col_map[col] = "wind_offshore_mw"
                elif "onshore" in col_lower or ("wind" in col_lower and "offshore" not in col_lower):
                    col_map[col] = "wind_onshore_mw"
                elif "solar" in col_lower:
                    col_map[col] = "solar_mw"
            df = df.rename(columns=col_map)
            for expected in ["wind_onshore_mw", "wind_offshore_mw", "solar_mw"]:
                if expected not in df.columns:
                    df[expected] = 0.0
            return df[["wind_onshore_mw", "wind_offshore_mw", "solar_mw"]]
        return self._with_cache(zone, "wind_solar_forecast", start, end, _fetch)

    def fetch_crossborder_flow(self, zone_from: str, zone_to: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        datatype = f"flow_{zone_from}_{zone_to}"
        def _fetch(s, e):
            series = _retry(self.client.query_crossborder_flows, zone_from, zone_to, start=s, end=e)
            return pd.DataFrame({"flow_mw": series})
        return self._with_cache(zone_from, datatype, start, end, _fetch)
