"""Open-Meteo historical weather data source.

Fetches hourly weather variables (temperature, wind speed, solar radiation) for
European bidding zone centroids.  Uses the free Archive API -- no API key needed.

API docs: https://open-meteo.com/en/docs/historical-weather-api
"""

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from da_forecast.config import API_MAX_RETRIES, API_BACKOFF_SECONDS
from da_forecast.sources.cache import ParquetCache

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "direct_radiation",
    "diffuse_radiation",
]

ZONE_WEATHER_COORDS: dict[str, tuple[float, float]] = {
    "DK_1": (56.0, 9.0),
    "DK_2": (55.5, 12.0),
    "NO_1": (60.0, 11.0),
    "NO_2": (58.5, 7.5),
    "NO_3": (63.5, 10.5),
    "NO_4": (69.0, 18.0),
    "NO_5": (60.5, 5.5),
    "SE_1": (66.5, 18.0),
    "SE_2": (63.0, 16.0),
    "SE_3": (59.5, 16.0),
    "SE_4": (56.5, 14.0),
    "FI": (63.0, 26.0),
    "DE_LU": (51.0, 10.0),
    "NL": (52.3, 5.0),
    "BE": (50.8, 4.5),
    "FR": (46.5, 2.5),
    "AT": (47.5, 14.0),
    "PL": (52.0, 20.0),
    "EE": (58.8, 25.5),
    "LV": (57.0, 24.5),
    "LT": (55.5, 24.0),
}

# Open-Meteo hourly archive caps at roughly 1 year per request.
MAX_DAYS_PER_REQUEST = 365


def _request_with_retry(params: dict) -> dict:
    """Send GET request to Open-Meteo with exponential back-off."""
    for attempt in range(API_MAX_RETRIES):
        try:
            resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data and data["error"]:
                raise ValueError(data.get("reason", "Unknown Open-Meteo error"))
            return data
        except (requests.RequestException, ValueError) as exc:
            if attempt < API_MAX_RETRIES - 1:
                wait = API_BACKOFF_SECONDS[attempt]
                logger.warning(
                    "Open-Meteo request failed (attempt %d): %s. Retrying in %ds...",
                    attempt + 1,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                raise


def _parse_response(data: dict) -> pd.DataFrame:
    """Convert Open-Meteo JSON response to a DataFrame with UTC DatetimeIndex."""
    hourly = data["hourly"]
    idx = pd.to_datetime(hourly["time"])
    # Open-Meteo returns timestamps in the requested timezone; convert to UTC.
    idx = idx.tz_localize("Europe/Berlin", ambiguous="infer").tz_convert("UTC")
    df = pd.DataFrame(
        {var: hourly[var] for var in HOURLY_VARIABLES if var in hourly},
        index=idx,
    )
    df.index.name = "utc_timestamp"
    return df


def _date_chunks(
    start: pd.Timestamp, end: pd.Timestamp
) -> list[tuple[str, str]]:
    """Split a date range into chunks of at most MAX_DAYS_PER_REQUEST days.

    Returns pairs of (start_date, end_date) formatted as YYYY-MM-DD strings.
    The *end* date is inclusive in the Open-Meteo API, so the last chunk's end
    is clamped to ``end - 1 day`` (we don't want to include the boundary day
    of the next chunk twice).
    """
    chunks: list[tuple[str, str]] = []
    current = start.normalize()
    final = (end - pd.Timedelta(days=1)).normalize()
    while current <= final:
        chunk_end = current + pd.Timedelta(days=MAX_DAYS_PER_REQUEST - 1)
        if chunk_end > final:
            chunk_end = final
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_weather(
    zone: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch historical weather data for *zone* between *start_date* and *end_date*.

    Checks the parquet cache first.  If the cached data already covers the
    requested range, the cached DataFrame is returned without hitting the API.
    New data is merged into the cache after fetching.

    Parameters
    ----------
    zone : str
        Bidding zone code (must be a key in ``ZONE_WEATHER_COORDS``).
    start_date, end_date : pd.Timestamp
        Half-open interval ``[start_date, end_date)``.
    cache_dir : Path
        Root directory for parquet cache files (typically ``data/raw``).

    Returns
    -------
    pd.DataFrame
        Hourly weather data with UTC DatetimeIndex.
    """
    if zone not in ZONE_WEATHER_COORDS:
        raise ValueError(f"Unknown zone '{zone}'. Must be one of {list(ZONE_WEATHER_COORDS)}")

    cache = ParquetCache(cache_dir)
    datatype = "weather"

    # Check if cache already covers the requested range.
    cached = cache.load("openmeteo", zone, datatype)
    if cached is not None and not cached.empty:
        start_utc = start_date.tz_convert("UTC") if start_date.tzinfo else start_date.tz_localize("UTC")
        end_utc = end_date.tz_convert("UTC") if end_date.tzinfo else end_date.tz_localize("UTC")
        if cached.index.min() <= start_utc and cached.index.max() >= end_utc - pd.Timedelta(hours=1):
            return cached.loc[start_utc:end_utc]

    lat, lon = ZONE_WEATHER_COORDS[zone]
    chunks = _date_chunks(start_date, end_date)

    frames: list[pd.DataFrame] = []
    for chunk_start, chunk_end in chunks:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start,
            "end_date": chunk_end,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": "Europe/Berlin",
        }
        data = _request_with_retry(params)
        df_chunk = _parse_response(data)
        if not df_chunk.empty:
            frames.append(df_chunk)
        time.sleep(0.3)  # polite rate-limiting

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    # Remove any duplicate timestamps from overlapping chunks.
    df = df[~df.index.duplicated(keep="first")]

    cache.merge("openmeteo", zone, datatype, df)
    return df
