#!/usr/bin/env python3
"""Fetch Danish market data from Energinet Energi Data Service.

No API key required. Fetches DK1/DK2 data in monthly chunks to avoid
API limits, with retry logic and deduplication.

Datasets:
- Elspotprices: hourly EUR/MWh spot prices for DK1 and DK2
- ElectricityBalanceNonv: hourly wind, solar, load, exchange flows

Run: uv run python scripts/fetch_energinet_data.py
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from da_forecast.config import RAW_DIR, ENERGINET_BASE_URL
from da_forecast.sources.cache import ParquetCache

ZONES = ["DK1", "DK2"]
RATE_LIMIT = 2.0  # seconds between requests


def fetch_chunked(dataset: str, zone: str, start: str, end: str) -> pd.DataFrame:
    """Fetch data in monthly chunks to avoid API limits."""
    all_records = []
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)

    current = start_dt
    while current < end_dt:
        chunk_end = min(current + pd.DateOffset(months=1), end_dt)
        params = {
            "start": current.strftime("%Y-%m-%dT%H:%M"),
            "end": chunk_end.strftime("%Y-%m-%dT%H:%M"),
            "filter": f'{{"PriceArea":["{zone}"]}}',
            "sort": "HourUTC asc",
            "limit": 0,
            "timezone": "UTC",
        }
        url = f"{ENERGINET_BASE_URL}/{dataset}"

        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=60)
                resp.raise_for_status()
                records = resp.json().get("records", [])
                all_records.extend(records)
                print(f"      {current.date()} → {chunk_end.date()}: {len(records)} records")
                break
            except Exception as e:
                if attempt < 2:
                    print(f"      Retry {attempt+1}: {e}")
                    time.sleep(3)
                else:
                    print(f"      FAILED: {e}")

        time.sleep(RATE_LIMIT)
        current = chunk_end

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["HourUTC"] = pd.to_datetime(df["HourUTC"], utc=True)
    df = df.set_index("HourUTC").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch Danish market data from Energinet")
    parser.add_argument("--start", default="2024-01-01T00:00", help="Start date (default: 2024-01-01)")
    parser.add_argument("--end", default="2026-04-01T00:00", help="End date (default: 2026-04-01)")
    args = parser.parse_args()

    START = args.start
    END = args.end

    print(f"Fetching Danish market data from Energinet")
    print(f"Period: {START} → {END}")
    print(f"API: {ENERGINET_BASE_URL}\n")

    cache = ParquetCache(RAW_DIR)

    for zone in ZONES:
        zone_key = f"DK_{zone[-1]}"
        print(f"\n{'='*50}")
        print(f"  {zone} ({zone_key})")
        print(f"{'='*50}")

        # Check if data already exists
        existing = cache.load("energinet", zone_key, "day_ahead_prices")
        if existing is not None and len(existing) > 1000:
            print(f"  Already cached: {len(existing)} hours ({existing.index.min().date()} -> {existing.index.max().date()})")
            print(f"  Skipping fetch. Delete data/raw/energinet/{zone_key}/ to re-fetch.")
            continue

        # Spot prices
        print(f"\n  Spot prices (Elspotprices):")
        df = fetch_chunked("Elspotprices", zone, START, END)
        if not df.empty:
            prices = pd.DataFrame({
                "price_eur_mwh": pd.to_numeric(df["SpotPriceEUR"], errors="coerce"),
                "price_dkk_mwh": pd.to_numeric(df["SpotPriceDKK"], errors="coerce"),
            }, index=df.index)
            prices = prices.dropna(subset=["price_eur_mwh"])
            cache.save("energinet", zone_key, "day_ahead_prices", prices)
            neg_pct = (prices["price_eur_mwh"] < 0).mean() * 100
            print(f"    Saved {len(prices)} hours")
            print(f"      Mean: {prices['price_eur_mwh'].mean():.1f} EUR/MWh")
            print(f"      Min:  {prices['price_eur_mwh'].min():.1f} EUR/MWh")
            print(f"      Max:  {prices['price_eur_mwh'].max():.1f} EUR/MWh")
            print(f"      Negative: {neg_pct:.1f}%")

        # Production and exchange
        print(f"\n  Production & exchange (ElectricityBalanceNonv):")
        df = fetch_chunked("ElectricityBalanceNonv", zone, START, END)
        if not df.empty:
            col_map = {
                "TotalLoad": "total_load_mw",
                "OnshoreWindPower": "wind_onshore_mw",
                "OffshoreWindPower": "wind_offshore_mw",
                "SolarPower": "solar_mw",
                "FossilGas": "gas_mw",
                "FossilHardCoal": "coal_mw",
                "Biomass": "biomass_mw",
                "Waste": "waste_mw",
                "HydroPower": "hydro_mw",
                "ExchangeContinent": "exchange_continent_mw",
                "ExchangeNordicCountries": "exchange_nordic_mw",
                "ExchangeGreatBelt": "exchange_great_belt_mw",
            }
            prod = pd.DataFrame(index=df.index)
            for src, dst in col_map.items():
                if src in df.columns:
                    prod[dst] = pd.to_numeric(df[src], errors="coerce")

            # Resample to hourly if sub-hourly data is present
            if len(prod) > 0:
                freq = pd.infer_freq(prod.index[:100])
                if freq and "min" in str(freq).lower():
                    print(f"    Resampling from {freq} to hourly...")
                    prod = prod.resample("h").mean()

            cache.save("energinet", zone_key, "production_exchange", prod)
            print(f"    Saved {len(prod)} rows")

            # Wind/solar in ENTSO-E-compatible format
            ws = pd.DataFrame(index=prod.index)
            ws["wind_onshore_mw"] = prod.get("wind_onshore_mw", 0)
            ws["wind_offshore_mw"] = prod.get("wind_offshore_mw", 0)
            ws["solar_mw"] = prod.get("solar_mw", 0)
            cache.save("energinet", zone_key, "wind_solar_forecast", ws)

            # Load
            if "total_load_mw" in prod.columns:
                load = pd.DataFrame({"load_mw": prod["total_load_mw"]}, index=prod.index)
                cache.save("energinet", zone_key, "load_forecast", load)

            wind_total = ws["wind_onshore_mw"].fillna(0) + ws["wind_offshore_mw"].fillna(0)
            print(f"      Wind mean:  {wind_total.mean():.0f} MW")
            print(f"      Solar mean: {ws['solar_mw'].mean():.0f} MW")
            if "total_load_mw" in prod.columns:
                print(f"      Load mean:  {prod['total_load_mw'].mean():.0f} MW")

    # Summary
    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    total = sum(1 for _ in RAW_DIR.rglob("*.parquet") if "energinet" in str(_))
    print(f"  Files saved: {total}")
    print(f"  Location: {RAW_DIR}/energinet/")
    for zone_key in ["DK_1", "DK_2"]:
        p = cache.load("energinet", zone_key, "day_ahead_prices")
        if p is not None:
            print(f"  {zone_key} prices: {p.index.min().date()} → {p.index.max().date()} ({len(p)} hours)")


if __name__ == "__main__":
    main()
