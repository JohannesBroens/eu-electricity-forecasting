"""Convenient data loading for notebooks and scripts.

Tries data sources in priority order:
1. energinet -- real Danish market data (DK_1, DK_2 only)
2. entsoe -- manually imported ENTSO-E CSV data (all zones)

Includes data quality handling:
- Forward-fill imputation for small gaps (<=6 hours, configurable)
- Audit logging of every imputed value
- Date alignment across datasets (prices are the reference)
- Duplicate timestamp removal

Usage:
    from da_forecast.data import load_prices, load_wind_solar, load_load, load_all

    prices = load_prices("DK_1")

    data = load_all("DK_1")
    print(data["source"])  # "energinet" or "entsoe"
    print(data["imputation_log"])
"""

import logging

import pandas as pd

from da_forecast.config import RAW_DIR
from da_forecast.sources.cache import ParquetCache

logger = logging.getLogger(__name__)

_cache = ParquetCache(RAW_DIR)

# Priority order for data sources.
# If no real data exists for a zone, the load will return None rather than
# silently substituting fabricated data.
SOURCES = ["energinet", "entsoe"]


def _load(zone: str, datatype: str) -> tuple[pd.DataFrame | None, str]:
    """Load data trying each source in priority order. Returns (df, source_name)."""
    for source in SOURCES:
        df = _cache.load(source, zone, datatype)
        if df is not None and not df.empty:
            return df, source
    return None, "none"


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps and sort."""
    return df[~df.index.duplicated(keep="first")].sort_index()


def _impute_ffill(
    df: pd.DataFrame,
    zone: str,
    datatype: str,
    max_gap_hours: int = 6,
) -> tuple[pd.DataFrame, list[dict]]:
    """Forward-fill small NA gaps with audit logging.

    Forward fill preserves the last known state, which suits electricity
    generation regime shifts (wind on/off) better than interpolation.
    Gaps > max_gap_hours are left as NA.
    """
    log = []
    result = df.copy()

    for col in df.columns:
        na_mask = df[col].isna()
        if not na_mask.any():
            continue

        n_before = int(na_mask.sum())
        result[col] = df[col].ffill(limit=max_gap_hours)
        n_after = int(result[col].isna().sum())
        n_filled = n_before - n_after

        if n_filled > 0:
            filled_mask = na_mask & ~result[col].isna()
            for ts in df.index[filled_mask]:
                log.append({
                    "zone": zone,
                    "datatype": datatype,
                    "column": col,
                    "timestamp": ts,
                    "method": f"ffill (max {max_gap_hours}h)",
                })

        if n_after > 0:
            logger.debug(
                f"{zone}/{datatype}/{col}: {n_after} NAs remain "
                f"(gaps > {max_gap_hours}h)"
            )

    return result, log


def _align_to_index(
    df: pd.DataFrame, ref_index: pd.DatetimeIndex
) -> pd.DataFrame:
    """Align DataFrame to a reference index (intersection only)."""
    common = df.index.intersection(ref_index)
    return df.loc[common]


def load_prices(zone: str = "DK_1") -> pd.DataFrame:
    """Load day-ahead prices for a zone. Returns DataFrame with 'price_eur_mwh' column."""
    df, source = _load(zone, "day_ahead_prices")
    if df is None:
        raise FileNotFoundError(
            f"No price data found for {zone}. Run one of:\n"
            f"  uv run python scripts/fetch_energinet_data.py   (DK zones, no auth)\n"
            f"  uv run python scripts/fetch_entsoe_data.py      (all zones, needs API key)\n"
            f"\n"
            f"ENTSO-E API key: https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188"
        )
    df = _clean_index(df)
    print(f"Loaded {zone} prices from '{source}': {len(df)} hours "
          f"({df.index.min().date()} -> {df.index.max().date()})")
    return df


def load_wind_solar(zone: str = "DK_1") -> pd.DataFrame:
    """Load wind/solar generation data.

    Small NA gaps (<=6h) are forward-filled. Remaining NAs are dropped.
    """
    df, source = _load(zone, "wind_solar_forecast")
    if df is None:
        raise FileNotFoundError(f"No wind/solar data found for {zone}.")
    df = _clean_index(df)
    df, _ = _impute_ffill(df, zone, "wind_solar")
    df = df.dropna()
    print(f"Loaded {zone} wind/solar from '{source}': {len(df)} hours")
    return df


def load_load(zone: str = "DK_1") -> pd.DataFrame:
    """Load electricity demand/load data.

    Small NA gaps (<=6h) are forward-filled. Remaining NAs are dropped.
    """
    df, source = _load(zone, "load_forecast")
    if df is None:
        raise FileNotFoundError(f"No load data found for {zone}.")
    df = _clean_index(df)
    df, _ = _impute_ffill(df, zone, "load")
    df = df.dropna()
    print(f"Loaded {zone} load from '{source}': {len(df)} hours")
    return df


def load_production(zone: str = "DK_1") -> pd.DataFrame:
    """Load full production/exchange breakdown (Energinet only).

    Small NA gaps (<=6h) are forward-filled. Remaining NAs are dropped.
    """
    df, source = _load(zone, "production_exchange")
    if df is None:
        raise FileNotFoundError(f"No production data found for {zone}.")
    df = _clean_index(df)
    df, _ = _impute_ffill(df, zone, "production")
    df = df.dropna()
    print(f"Loaded {zone} production from '{source}': {len(df)} hours")
    return df


def load_all(zone: str = "DK_1") -> dict:
    """Load all available data for a zone with quality handling.

    Steps: deduplicate timestamps, forward-fill small gaps, drop remaining
    NAs, align all datasets to the prices index.

    Returns dict with keys: prices, wind_solar, load, production (if available),
    source, zone, imputation_log.
    """
    result = {"zone": zone, "imputation_log": []}

    prices, source = _load(zone, "day_ahead_prices")
    if prices is not None:
        prices = _clean_index(prices)
    result["prices"] = prices
    result["source"] = source

    for key, datatype in [
        ("wind_solar", "wind_solar_forecast"),
        ("load", "load_forecast"),
        ("production", "production_exchange"),
    ]:
        df, _ = _load(zone, datatype)
        if df is not None:
            df = _clean_index(df)
            df, log = _impute_ffill(df, zone, datatype)
            result["imputation_log"].extend(log)
            df = df.dropna()
            if prices is not None:
                df = _align_to_index(df, prices.index)
            result[key] = df
        else:
            result[key] = None

    src = result.get("source", "none")
    n_imputed = len(result["imputation_log"])
    print(f"\n{'='*50}")
    print(f"  {zone} data loaded from '{src}'")
    print(f"{'='*50}")
    for key in ["prices", "wind_solar", "load", "production"]:
        df = result.get(key)
        if df is not None:
            na_count = df.isna().sum().sum()
            na_str = f", {na_count} NAs" if na_count > 0 else ""
            print(f"  {key:15s}: {len(df):5d} hours ({df.index.min().date()} -> {df.index.max().date()}){na_str}")
        else:
            print(f"  {key:15s}: not available")
    if n_imputed > 0:
        print(f"  {'imputed':15s}: {n_imputed} values forward-filled")
    print()

    return result


def available_zones() -> dict[str, str]:
    """Show which zones have data and from which source."""
    from da_forecast.config import ZONES
    result = {}
    for zone in ZONES:
        for source in SOURCES:
            df = _cache.load(source, zone, "day_ahead_prices")
            if df is not None and not df.empty:
                result[zone] = source
                break
        else:
            result[zone] = "none"
    return result


def reconcile_sources(
    zone: str,
    datatype: str = "day_ahead_prices",
    col: str = "price_eur_mwh",
) -> pd.DataFrame:
    """Compare the same data across all available sources.

    Returns a DataFrame with one column per source plus pairwise diff columns.
    """
    frames = {}
    for source in SOURCES:
        df = _cache.load(source, zone, datatype)
        if df is not None and not df.empty and col in df.columns:
            series = _clean_index(df)[col]
            frames[source] = series

    if not frames:
        raise ValueError(f"No data for {zone}/{datatype}/{col}")

    combined = pd.DataFrame(frames)

    source_names = list(frames.keys())
    for i in range(len(source_names)):
        for j in range(i + 1, len(source_names)):
            s1, s2 = source_names[i], source_names[j]
            diff_col = f"diff_{s1}_{s2}"
            combined[diff_col] = combined[s1] - combined[s2]

    return combined


def load_reconciled(
    zone: str,
    datatype: str = "day_ahead_prices",
    col: str = "price_eur_mwh",
    primary_source: str | None = None,
) -> tuple[pd.Series, dict]:
    """Load data using primary source, filling gaps from fallback sources.

    Implements multi-source reconciliation: load from primary, fill missing
    hours from next source in priority order.

    Returns (series, report) where report describes what was used.
    """
    if primary_source is None:
        for s in SOURCES:
            df = _cache.load(s, zone, datatype)
            if df is not None and not df.empty and col in df.columns:
                primary_source = s
                break

    if primary_source is None:
        raise ValueError(f"No data for {zone}/{datatype}")

    primary_df = _cache.load(primary_source, zone, datatype)
    primary_df = _clean_index(primary_df)
    result = primary_df[col].copy()

    report = {
        "zone": zone,
        "datatype": datatype,
        "column": col,
        "primary_source": primary_source,
        "primary_hours": int((~result.isna()).sum()),
        "fallback_fills": [],
    }

    for fallback in SOURCES:
        if fallback == primary_source:
            continue
        if result.isna().sum() == 0:
            break

        fb_df = _cache.load(fallback, zone, datatype)
        if fb_df is None or fb_df.empty or col not in fb_df.columns:
            continue

        fb_df = _clean_index(fb_df)
        fb_series = fb_df[col]

        na_mask = result.isna()
        fill_mask = na_mask & fb_series.reindex(result.index).notna()
        n_filled = fill_mask.sum()

        if n_filled > 0:
            result[fill_mask] = fb_series.reindex(result.index)[fill_mask]
            report["fallback_fills"].append({
                "source": fallback,
                "hours_filled": int(n_filled),
            })

    report["total_hours"] = int((~result.isna()).sum())
    report["remaining_gaps"] = int(result.isna().sum())

    return result, report
