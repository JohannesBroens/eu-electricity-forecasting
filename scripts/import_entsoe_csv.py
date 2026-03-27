#!/usr/bin/env python3
"""Import manually downloaded ENTSO-E CSV files into the pipeline cache.

ENTSO-E Transparency Platform allows CSV download without an API key.
Save CSV files to data/raw/entsoe_csv/ then run this script.

Supported data types (auto-detected from headers/filename):
- Day-ahead prices
- Wind/solar generation forecasts
- Total load forecasts
- Cross-border physical flows

Run: uv run python scripts/import_entsoe_csv.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from da_forecast.config import RAW_DIR
from da_forecast.sources.cache import ParquetCache

CSV_DIR = RAW_DIR / "entsoe_csv"

# ENTSO-E area code → internal zone code
AREA_MAP = {
    "NO2": "NO_2", "BZN|NO2": "NO_2", "10YNO-2--------T": "NO_2",
    "SE3": "SE_3", "BZN|SE3": "SE_3", "10Y1001A1001A46L": "SE_3",
    "SE4": "SE_4", "BZN|SE4": "SE_4", "10Y1001A1001A47J": "SE_4",
    "DE-LU": "DE_LU", "BZN|DE-LU": "DE_LU", "10Y1001A1001A82H": "DE_LU",
    "DK1": "DK_1", "BZN|DK1": "DK_1", "10YDK-1--------W": "DK_1",
    "DK2": "DK_2", "BZN|DK2": "DK_2", "10YDK-2--------M": "DK_2",
}


def detect_zone(df: pd.DataFrame, filename: str) -> str | None:
    """Detect bidding zone from CSV content or filename."""
    for col in ["Area", "MapCode", "AreaCode", "BiddingZone"]:
        if col in df.columns:
            area = str(df[col].iloc[0])
            if area in AREA_MAP:
                return AREA_MAP[area]

    fname = filename.upper()
    for code, zone in AREA_MAP.items():
        if code.upper().replace("|", "_") in fname or code.upper() in fname:
            return zone

    return None


def parse_entsoe_timestamp(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Parse ENTSO-E CSV timestamps across various formats."""
    # MTU columns (most common in ENTSO-E exports)
    for col in df.columns:
        if "mtu" in col.lower():
            times = df[col].astype(str).str.split(" - ").str[0]
            try:
                return pd.to_datetime(times, dayfirst=True, utc=True)
            except Exception:
                pass

    time_cols = ["DateTime", "Date", "Time (UTC)", "Timestamp"]
    for col in time_cols:
        if col in df.columns:
            return pd.to_datetime(df[col], utc=True, dayfirst=True)

    if "Date" in df.columns and "Time" in df.columns:
        return pd.to_datetime(df["Date"] + " " + df["Time"], utc=True, dayfirst=True)

    raise ValueError(f"Could not find/parse timestamp column. Columns: {list(df.columns)}")


def import_prices(df: pd.DataFrame, zone: str, cache: ParquetCache):
    """Import day-ahead price CSV."""
    idx = parse_entsoe_timestamp(df)
    price_col = None
    for col in df.columns:
        if "price" in col.lower() or "day-ahead" in col.lower():
            price_col = col
            break
    if price_col is None:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            price_col = numeric_cols[0]

    if price_col is None:
        print(f"    WARNING: Could not find price column in {list(df.columns)}")
        return

    prices = pd.DataFrame({
        "price_eur_mwh": pd.to_numeric(df[price_col], errors="coerce"),
    }, index=idx)
    prices = prices.dropna().sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    cache.merge("entsoe", zone, "day_ahead_prices", prices)
    print(f"    Prices: {len(prices)} hours, mean={prices['price_eur_mwh'].mean():.1f} EUR/MWh")


def import_generation_forecast(df: pd.DataFrame, zone: str, cache: ParquetCache):
    """Import wind/solar generation forecast CSV."""
    idx = parse_entsoe_timestamp(df)
    result = pd.DataFrame(index=idx)

    for col in df.columns:
        col_lower = col.lower()
        if "offshore" in col_lower and "wind" in col_lower:
            result["wind_offshore_mw"] = pd.to_numeric(df[col], errors="coerce")
        elif "onshore" in col_lower and "wind" in col_lower:
            result["wind_onshore_mw"] = pd.to_numeric(df[col], errors="coerce")
        elif "wind" in col_lower and ("forecast" in col_lower or "generation" in col_lower):
            result["wind_onshore_mw"] = pd.to_numeric(df[col], errors="coerce")
        elif "solar" in col_lower or "photovoltaic" in col_lower:
            result["solar_mw"] = pd.to_numeric(df[col], errors="coerce")

    for expected in ["wind_onshore_mw", "wind_offshore_mw", "solar_mw"]:
        if expected not in result.columns:
            result[expected] = 0.0

    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    cache.merge("entsoe", zone, "wind_solar_forecast", result)
    wind_total = result["wind_onshore_mw"].fillna(0) + result["wind_offshore_mw"].fillna(0)
    print(f"    Wind/solar: {len(result)} hours, wind mean={wind_total.mean():.0f} MW")


def import_load_forecast(df: pd.DataFrame, zone: str, cache: ParquetCache):
    """Import total load forecast CSV."""
    idx = parse_entsoe_timestamp(df)
    load_col = None
    for col in df.columns:
        if "load" in col.lower() or "forecast" in col.lower():
            load_col = col
            break
    if load_col is None:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            load_col = numeric_cols[0]

    if load_col is None:
        print(f"    WARNING: Could not find load column")
        return

    load = pd.DataFrame({"load_mw": pd.to_numeric(df[load_col], errors="coerce")}, index=idx)
    load = load.dropna().sort_index()
    load = load[~load.index.duplicated(keep="last")]
    cache.merge("entsoe", zone, "load_forecast", load)
    print(f"    Load: {len(load)} hours, mean={load['load_mw'].mean():.0f} MW")


def import_crossborder_flow(df: pd.DataFrame, cache: ParquetCache, filename: str):
    """Import cross-border physical flow CSV."""
    idx = parse_entsoe_timestamp(df)
    flow_col = None
    for col in df.columns:
        if "flow" in col.lower() or any(c.isdigit() for c in col):
            flow_col = col
            break
    if flow_col is None:
        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) > 0:
            flow_col = numeric_cols[0]

    if flow_col is None:
        print(f"    WARNING: Could not find flow column")
        return

    flow = pd.DataFrame({"flow_mw": pd.to_numeric(df[flow_col], errors="coerce")}, index=idx)
    flow = flow.dropna().sort_index()

    fname_upper = filename.upper()
    zone_from = zone_to = None
    for code, zone in AREA_MAP.items():
        if code.upper() in fname_upper:
            if zone_from is None:
                zone_from = zone
            else:
                zone_to = zone

    if zone_from and zone_to:
        label = f"{zone_from}_{zone_to}"
        cache.merge("entsoe", zone_from, f"flow_{label}", flow)
        print(f"    Flow {label}: {len(flow)} hours, mean={flow['flow_mw'].mean():.0f} MW")
    else:
        print(f"    WARNING: Could not detect zone pair from filename '{filename}'")
        print(f"    Rename file to include zone codes, e.g., 'flow_DK1_NO2.csv'")


def detect_data_type(df: pd.DataFrame, filename: str) -> str:
    """Determine CSV data type from content and filename."""
    fname = filename.lower()
    cols_lower = [c.lower() for c in df.columns]

    if "price" in fname or any("price" in c for c in cols_lower):
        return "prices"
    elif "generation" in fname or "wind" in fname or "solar" in fname:
        return "generation"
    elif "load" in fname:
        return "load"
    elif "flow" in fname or "physical" in fname:
        return "flow"

    if any("price" in c or "day-ahead" in c for c in cols_lower):
        return "prices"
    elif any("wind" in c or "solar" in c for c in cols_lower):
        return "generation"
    elif any("load" in c for c in cols_lower):
        return "load"

    return "unknown"


def main():
    print("ENTSO-E CSV Importer")
    print(f"Looking for CSV files in: {CSV_DIR}\n")

    if not CSV_DIR.exists():
        CSV_DIR.mkdir(parents=True)
        print(f"Created {CSV_DIR}/")
        print(f"\nDownload CSV files from https://transparency.entsoe.eu/ and save them here.")
        print(f"Then re-run this script.")
        return

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {CSV_DIR}")
        return

    cache = ParquetCache(RAW_DIR)

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file.name}")
        try:
            for sep in ["\t", ",", ";"]:
                try:
                    df = pd.read_csv(csv_file, sep=sep, encoding="utf-8")
                    if len(df.columns) > 1:
                        break
                except Exception:
                    continue
            else:
                df = pd.read_csv(csv_file)

            zone = detect_zone(df, csv_file.name)
            data_type = detect_data_type(df, csv_file.name)

            if zone:
                print(f"  Zone: {zone}, Type: {data_type}")
            else:
                print(f"  WARNING: Could not detect zone. Skipping.")
                print(f"  Columns: {list(df.columns)[:5]}...")
                continue

            if data_type == "prices":
                import_prices(df, zone, cache)
            elif data_type == "generation":
                import_generation_forecast(df, zone, cache)
            elif data_type == "load":
                import_load_forecast(df, zone, cache)
            elif data_type == "flow":
                import_crossborder_flow(df, cache, csv_file.name)
            else:
                print(f"  WARNING: Unknown data type. Columns: {list(df.columns)}")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nImport complete. Cache: {RAW_DIR}")


if __name__ == "__main__":
    main()
