#!/usr/bin/env python3
"""Fetch historical weather data from Open-Meteo for all configured zones.

Open-Meteo is a free API -- no API key required.
Fetches: temperature_2m, wind_speed_10m, wind_speed_100m, direct_radiation,
         diffuse_radiation at hourly resolution.

Uses cache merging -- safe to re-run; only missing periods are fetched.

Run: uv run python scripts/fetch_weather_data.py
     uv run python scripts/fetch_weather_data.py --start 2024-01-01 --end 2026-04-01
     uv run python scripts/fetch_weather_data.py --zones DK_1 DK_2
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from da_forecast.config import RAW_DIR, ZONES
from da_forecast.sources.cache import ParquetCache
from da_forecast.sources.openmeteo import (
    ZONE_WEATHER_COORDS,
    fetch_weather,
)

TARGET_HOURS = 17000  # ~2 years of hourly data


def _cached_hours(cache: ParquetCache, zone: str) -> int:
    df = cache.load("openmeteo", zone, "weather")
    return len(df) if df is not None else 0


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical weather data from Open-Meteo."
    )
    parser.add_argument(
        "--start", default="2024-01-01", help="Start date (default: 2024-01-01)"
    )
    parser.add_argument(
        "--end", default="2026-04-01", help="End date (default: 2026-04-01)"
    )
    parser.add_argument(
        "--zones", nargs="*", default=None, help="Specific zones (default: all)"
    )
    args = parser.parse_args()

    start = pd.Timestamp(args.start, tz="Europe/Berlin")
    end = pd.Timestamp(args.end, tz="Europe/Berlin")
    zones = args.zones if args.zones else ZONES

    # Only include zones that have weather coordinates defined.
    zones = [z for z in zones if z in ZONE_WEATHER_COORDS]

    print(f"Open-Meteo weather fetch: {start.date()} -> {end.date()}")
    print(f"Zones: {len(zones)} ({', '.join(zones)})")
    print(f"Variables: temperature_2m, wind_speed_10m, wind_speed_100m, "
          f"direct_radiation, diffuse_radiation")
    print()

    cache = ParquetCache(RAW_DIR)

    for zone in zones:
        existing = _cached_hours(cache, zone)
        if existing >= TARGET_HOURS:
            print(f"  {zone:6s} weather... CACHED ({existing} hours)")
            continue

        print(f"  {zone:6s} weather... ", end="", flush=True)
        try:
            df = fetch_weather(zone, start, end, cache_dir=RAW_DIR)
            final = _cached_hours(cache, zone)
            new_hours = len(df) if df is not None and not df.empty else 0
            if new_hours > 0:
                print(f"OK +{new_hours} hours (total: {final})")
            else:
                print("NO DATA")
        except Exception as e:
            print(f"FAIL ({e})")

        time.sleep(0.5)  # polite rate-limiting between zones

    print()
    total = sum(1 for _ in RAW_DIR.rglob("*.parquet") if "openmeteo" in str(_))
    print(f"Done. Open-Meteo cache files: {total}")
    print(f"Location: {RAW_DIR / 'openmeteo'}")


if __name__ == "__main__":
    main()
