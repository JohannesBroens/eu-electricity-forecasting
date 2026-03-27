#!/usr/bin/env python3
"""Fetch data from ENTSO-E Transparency Platform using the entsoe-py wrapper.

Requires ENTSOE_API_KEY in .env file.
Get your API key: https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188

Fetches for all configured zones:
- Day-ahead prices
- Day-ahead load forecasts
- Wind and solar generation forecasts
- Cross-border flows (DK interconnectors)

Skips zones/datatypes that already have cached data. Delete the parquet
files under data/raw/entsoe/ to force a re-fetch.

Run: uv run python scripts/fetch_entsoe_data.py
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from da_forecast.config import RAW_DIR, ZONES, INTERCONNECTORS
from da_forecast.sources.cache import ParquetCache
from da_forecast.sources.entsoe import EntsoeSource

load_dotenv()

MIN_CACHED_HOURS = 1000  # skip fetch if we already have this many hours


def _is_cached(cache: ParquetCache, zone: str, datatype: str) -> bool:
    df = cache.load("entsoe", zone, datatype)
    return df is not None and len(df) >= MIN_CACHED_HOURS


def main():
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        print("ENTSOE_API_KEY not found in .env file.")
        print()
        print("To get an API key, follow the instructions at:")
        print("  https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188")
        print()
        print("Then add it to .env:")
        print('  ENTSOE_API_KEY=your-key-here')
        sys.exit(1)

    cache = ParquetCache(RAW_DIR)
    source = EntsoeSource(api_key=api_key, cache_dir=RAW_DIR)

    start = pd.Timestamp("2025-03-01", tz="Europe/Copenhagen")
    end = pd.Timestamp("2025-10-01", tz="Europe/Copenhagen")

    print(f"Fetching ENTSO-E data: {start.date()} -> {end.date()}")
    print(f"API key: {'*' * 32}{api_key[-4:]}")
    print()

    # Day-ahead prices for all zones
    for zone in ZONES:
        if _is_cached(cache, zone, "day_ahead_prices"):
            print(f"  {zone} prices... CACHED (skipping)")
            continue
        print(f"  {zone} prices...", end=" ", flush=True)
        try:
            df = source.fetch_day_ahead_prices(zone, start, end)
            neg = (df["price_eur_mwh"] < 0).mean() * 100
            print(f"OK {len(df)} hours, mean={df['price_eur_mwh'].mean():.1f} EUR/MWh, {neg:.1f}% negative")
        except Exception as e:
            print(f"FAIL {e}")
        time.sleep(1)

    print()

    # Load forecasts for all zones
    for zone in ZONES:
        if _is_cached(cache, zone, "load_forecast"):
            print(f"  {zone} load forecast... CACHED (skipping)")
            continue
        print(f"  {zone} load forecast...", end=" ", flush=True)
        try:
            df = source.fetch_load_forecast(zone, start, end)
            print(f"OK {len(df)} hours, mean={df['load_mw'].mean():.0f} MW")
        except Exception as e:
            print(f"FAIL {e}")
        time.sleep(1)

    print()

    # Wind/solar forecasts for all zones
    for zone in ZONES:
        if _is_cached(cache, zone, "wind_solar_forecast"):
            print(f"  {zone} wind/solar forecast... CACHED (skipping)")
            continue
        print(f"  {zone} wind/solar forecast...", end=" ", flush=True)
        try:
            df = source.fetch_wind_solar_forecast(zone, start, end)
            wind = df[[c for c in df.columns if "wind" in c]].sum(axis=1).mean()
            solar = df["solar_mw"].mean() if "solar_mw" in df.columns else 0
            print(f"OK {len(df)} hours, wind={wind:.0f} MW, solar={solar:.0f} MW")
        except Exception as e:
            print(f"FAIL {e}")
        time.sleep(1)

    print()

    # Cross-border flows for DK interconnectors
    for from_zone, to_zone, capacity in INTERCONNECTORS:
        if "DK" not in from_zone:
            continue
        datatype = f"flow_{from_zone}_{to_zone}"
        if _is_cached(cache, from_zone, datatype):
            print(f"  {from_zone} -> {to_zone} flow... CACHED (skipping)")
            continue
        print(f"  {from_zone} -> {to_zone} flow...", end=" ", flush=True)
        try:
            df = source.fetch_crossborder_flow(from_zone, to_zone, start, end)
            print(f"OK {len(df)} hours, mean={df['flow_mw'].mean():.0f} MW")
        except Exception as e:
            print(f"FAIL {e}")
        time.sleep(1)

    print()
    total = sum(1 for _ in RAW_DIR.rglob("*.parquet") if "entsoe" in str(_))
    print(f"Done. ENTSO-E cache files: {total}")
    print(f"  Location: {RAW_DIR / 'entsoe'}")


if __name__ == "__main__":
    main()
