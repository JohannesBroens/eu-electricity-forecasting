"""Energinet DataHub (Energi Data Service) data source.

Free REST API -- no API key required.
Base URL: https://api.energidataservice.dk/dataset/{DatasetName}
"""
import logging
import time
from pathlib import Path
import pandas as pd
import requests
from da_forecast.config import API_BACKOFF_SECONDS, API_MAX_RETRIES, ENERGINET_BASE_URL
from da_forecast.sources.cache import ParquetCache

logger = logging.getLogger(__name__)

def _normalize_zone(zone: str) -> str:
    return zone.replace("_", "")

def _fetch_dataset(dataset: str, start: pd.Timestamp, end: pd.Timestamp, zone: str | None = None, columns: list[str] | None = None) -> list[dict]:
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M"),
        "end": end.strftime("%Y-%m-%dT%H:%M"),
        "sort": "HourUTC asc",
        "limit": 0,
        "timezone": "UTC",
    }
    if zone:
        params["filter"] = f'{{"PriceArea":["{_normalize_zone(zone)}"]}}'
    if columns:
        params["columns"] = ",".join(columns)
    url = f"{ENERGINET_BASE_URL}/{dataset}"
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()["records"]
        except Exception as e:
            if attempt < API_MAX_RETRIES - 1:
                wait = API_BACKOFF_SECONDS[attempt]
                logger.warning(f"Energinet API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

class EnerginetSource:
    def __init__(self, cache_dir: Path | None = None):
        self.cache = ParquetCache(cache_dir) if cache_dir else None

    def _records_to_df(self, records: list[dict], time_col: str = "HourUTC") -> pd.DataFrame:
        df = pd.DataFrame(records)
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col).sort_index()
        return df

    def fetch_spot_prices(self, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self.cache:
            cached = self.cache.load("energinet", zone, "spot_prices")
            if cached is not None:
                subset = cached.loc[start:end]
                if not subset.empty:
                    return subset
        records = _fetch_dataset("Elspotprices", start, end, zone)
        df = self._records_to_df(records)
        result = pd.DataFrame({"price_eur_mwh": df["SpotPriceEUR"].astype(float), "price_dkk_mwh": df["SpotPriceDKK"].astype(float)}, index=df.index)
        if self.cache and not result.empty:
            self.cache.merge("energinet", zone, "spot_prices", result)
        return result

    def fetch_production_and_exchange(self, zone: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        if self.cache:
            cached = self.cache.load("energinet", zone, "production_exchange")
            if cached is not None:
                subset = cached.loc[start:end]
                if not subset.empty:
                    return subset
        records = _fetch_dataset("ElectricityBalanceNonv", start, end, zone)
        df = self._records_to_df(records)
        col_map = {
            "TotalLoad": "total_load_mw", "OnshoreWindPower": "wind_onshore_mw",
            "OffshoreWindPower": "wind_offshore_mw", "SolarPower": "solar_mw",
            "FossilGas": "gas_mw", "FossilHardCoal": "coal_mw", "Biomass": "biomass_mw",
            "ExchangeContinent": "exchange_continent_mw", "ExchangeNordicCountries": "exchange_nordic_mw",
            "ExchangeGreatBelt": "exchange_great_belt_mw",
        }
        result = pd.DataFrame(index=df.index)
        for src_col, dst_col in col_map.items():
            if src_col in df.columns:
                result[dst_col] = pd.to_numeric(df[src_col], errors="coerce")
        if self.cache and not result.empty:
            self.cache.merge("energinet", zone, "production_exchange", result)
        return result
