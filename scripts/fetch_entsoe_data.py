#!/usr/bin/env python3
"""Fetch data from ENTSO-E Transparency Platform using the entsoe-py wrapper.

Requires ENTSOE_API_KEY in .env file.
Get your API key: https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188

Fetches for all configured zones:
- Day-ahead prices
- Day-ahead load forecasts
- Wind and solar generation forecasts
- Cross-border flows (DK interconnectors)

Fetches in 3-month chunks to stay within API limits (max 1 year per request).
Uses cache merging -- safe to re-run; only missing periods are fetched.

Run: uv run python scripts/fetch_entsoe_data.py
     uv run python scripts/fetch_entsoe_data.py --start 2024-01-01 --end 2026-04-01
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from da_forecast.config import RAW_DIR, ZONES, ZONE_EIC, INTERCONNECTORS
from da_forecast.sources.cache import ParquetCache
from da_forecast.sources.entsoe import EntsoeSource

load_dotenv()

CHUNK_MONTHS = 3  # fetch in 3-month chunks to avoid API limits
TARGET_HOURS = 17000  # ~2 years of hourly data


def _cached_hours(cache: ParquetCache, zone: str, datatype: str) -> int:
    df = cache.load("entsoe", zone, datatype)
    return len(df) if df is not None else 0


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Yield (chunk_start, chunk_end) in CHUNK_MONTHS intervals."""
    current = start
    while current < end:
        chunk_end = current + pd.DateOffset(months=CHUNK_MONTHS)
        if chunk_end > end:
            chunk_end = end
        yield current, chunk_end
        current = chunk_end


def fetch_datatype(source, cache, zone, datatype, fetch_fn, start, end, label):
    """Fetch one datatype for one zone, in chunks, with cache merging."""
    existing = _cached_hours(cache, zone, datatype)
    if existing >= TARGET_HOURS:
        print(f"  {zone:6s} {label}... CACHED ({existing} hours)")
        return True

    print(f"  {zone:6s} {label}... ", end="", flush=True)
    total_new = 0
    for chunk_start, chunk_end in _date_chunks(start, end):
        try:
            df = fetch_fn(zone, chunk_start, chunk_end)
            if df is not None and not df.empty:
                cache.merge("entsoe", zone, datatype, df)
                total_new += len(df)
            time.sleep(0.5)  # rate limiting
        except Exception as e:
            err = str(e)
            if "No matching data" in err or "no data" in err.lower():
                continue  # some zones/periods have no data
            print(f"FAIL ({chunk_start.date()}-{chunk_end.date()}: {err})")
            time.sleep(2)
            continue

    final = _cached_hours(cache, zone, datatype)
    if total_new > 0:
        print(f"OK +{total_new} hours (total: {final})")
    elif final > 0:
        print(f"CACHED ({final} hours)")
    else:
        print(f"NO DATA")
    return final > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-01", help="Start date (default: 2024-01-01)")
    parser.add_argument("--end", default="2026-04-01", help="End date (default: 2026-04-01)")
    parser.add_argument("--zones", nargs="*", default=None, help="Specific zones (default: all)")
    args = parser.parse_args()

    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        print("ENTSOE_API_KEY not found in .env file.")
        print()
        print("To get an API key, follow the instructions at:")
        print("  https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188")
        print()
        print("Then add it to .env:")
        print("  ENTSOE_API_KEY=your-key-here")
        sys.exit(1)

    cache = ParquetCache(RAW_DIR)
    source = EntsoeSource(api_key=api_key, cache_dir=RAW_DIR)

    start = pd.Timestamp(args.start, tz="Europe/Copenhagen")
    end = pd.Timestamp(args.end, tz="Europe/Copenhagen")
    zones = args.zones if args.zones else ZONES

    print(f"ENTSO-E data fetch: {start.date()} -> {end.date()}")
    print(f"Zones: {len(zones)} ({', '.join(zones)})")
    print(f"API key: {'*' * 28}{api_key[-4:]}")
    print()

    # Prices
    print("Day-ahead prices:")
    for zone in zones:
        fetch_datatype(source, cache, zone, "day_ahead_prices",
                       source.fetch_day_ahead_prices, start, end, "prices")

    # Load forecasts
    print("\nLoad forecasts:")
    for zone in zones:
        fetch_datatype(source, cache, zone, "load_forecast",
                       source.fetch_load_forecast, start, end, "load")

    # Wind/solar forecasts
    print("\nWind/solar forecasts:")
    for zone in zones:
        fetch_datatype(source, cache, zone, "wind_solar_forecast",
                       source.fetch_wind_solar_forecast, start, end, "wind/solar")

    # Cross-border flows (DK interconnectors only)
    print("\nCross-border flows:")
    for from_zone, to_zone, capacity in INTERCONNECTORS:
        if "DK" not in from_zone:
            continue
        if from_zone not in zones:
            continue
        datatype = f"flow_{from_zone}_{to_zone}"

        def _fetch_flow(zone, s, e, fz=from_zone, tz=to_zone):
            return source.fetch_crossborder_flow(fz, tz, s, e)

        fetch_datatype(source, cache, from_zone, datatype,
                       _fetch_flow, start, end, f"-> {to_zone}")

    print()
    total = sum(1 for _ in RAW_DIR.rglob("*.parquet") if "entsoe" in str(_))
    print(f"Done. ENTSO-E cache files: {total}")
    print(f"Location: {RAW_DIR / 'entsoe'}")


if __name__ == "__main__":
    main()
