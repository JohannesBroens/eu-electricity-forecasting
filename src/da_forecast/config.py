"""Pipeline configuration.

Zone codes, interconnector capacities, feature availability rules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# European bidding zones
ZONES = [
    "DK_1", "DK_2",
    "NO_1", "NO_2", "NO_3", "NO_4", "NO_5",
    "SE_1", "SE_2", "SE_3", "SE_4",
    "FI",
    "DE_LU", "NL", "BE", "FR", "AT", "PL",
    "EE", "LV", "LT",
]
PRIMARY_ZONES = ["DK_1", "DK_2"]

ZONE_LABELS = {
    "DK_1": "West Denmark",
    "DK_2": "East Denmark",
    "NO_1": "South-East Norway",
    "NO_2": "South Norway",
    "NO_3": "Middle Norway",
    "NO_4": "North Norway",
    "NO_5": "West Norway",
    "SE_1": "North Sweden",
    "SE_2": "North-Central Sweden",
    "SE_3": "Central Sweden",
    "SE_4": "South Sweden",
    "FI": "Finland",
    "DE_LU": "Germany-Luxembourg",
    "NL": "Netherlands",
    "BE": "Belgium",
    "FR": "France",
    "AT": "Austria",
    "PL": "Poland",
    "EE": "Estonia",
    "LV": "Latvia",
    "LT": "Lithuania",
}

# ENTSO-E EIC area codes
ZONE_EIC = {
    "DK_1": "10YDK-1--------W",
    "DK_2": "10YDK-2--------M",
    "NO_1": "10YNO-1--------2",
    "NO_2": "10YNO-2--------T",
    "NO_3": "10YNO-3--------J",
    "NO_4": "10YNO-4--------9",
    "NO_5": "10Y1001A1001A48H",
    "SE_1": "10Y1001A1001A44P",
    "SE_2": "10Y1001A1001A45N",
    "SE_3": "10Y1001A1001A46L",
    "SE_4": "10Y1001A1001A47J",
    "FI": "10YFI-1--------U",
    "DE_LU": "10Y1001A1001A82H",
    "NL": "10YNL----------L",
    "BE": "10YBE----------2",
    "FR": "10YFR-RTE------C",
    "AT": "10YAT-APG------L",
    "PL": "10YPL-AREA-----S",
    "EE": "10Y1001A1001A39I",
    "LV": "10YLV-1001A00074",
    "LT": "10YLT-1001A0008Q",
}

# Interconnectors: (from_zone, to_zone, nominal_capacity_mw)
# Sources: ENTSO-E TYNDP 2024, Energinet system data
INTERCONNECTORS = [
    # Denmark internal and neighbours
    ("DK_1", "DK_2", 600),
    ("DK_1", "DE_LU", 2500),
    ("DK_1", "NO_2", 1700),
    ("DK_1", "SE_3", 740),
    ("DK_2", "DE_LU", 600),
    ("DK_2", "SE_4", 1700),
    # Norway internal
    ("NO_1", "NO_2", 3500),
    ("NO_1", "NO_3", 1600),
    ("NO_1", "NO_5", 3900),
    ("NO_2", "NO_5", 3000),
    ("NO_3", "NO_4", 950),
    ("NO_3", "NO_5", 500),
    # Norway cross-border
    ("NO_1", "SE_3", 2145),
    ("NO_3", "SE_2", 1000),
    ("NO_4", "SE_1", 700),
    ("NO_4", "SE_2", 300),
    ("NO_2", "DE_LU", 1400),
    ("NO_2", "NL", 700),
    ("NO_5", "DE_LU", 1400),
    # Sweden internal
    ("SE_1", "SE_2", 3300),
    ("SE_2", "SE_3", 7300),
    ("SE_3", "SE_4", 5400),
    # Sweden cross-border
    ("SE_3", "FI", 1200),
    ("SE_4", "DE_LU", 615),
    ("SE_4", "PL", 600),
    ("SE_4", "LT", 700),
    # Finland cross-border
    ("FI", "EE", 1016),
    # Baltic internal
    ("EE", "LV", 1400),
    ("LV", "LT", 1400),
    ("LT", "PL", 500),
    # Central-West Europe
    ("DE_LU", "NL", 5000),
    ("DE_LU", "BE", 1000),
    ("DE_LU", "FR", 4800),
    ("DE_LU", "AT", 5000),
    ("DE_LU", "PL", 3000),
    ("NL", "BE", 2400),
    ("FR", "BE", 4300),
    ("AT", "PL", 600),
    ("NL", "DK_1", 700),
    ("FR", "AT", 150),
]

ENERGINET_BASE_URL = "https://api.energidataservice.dk/dataset"

# Feature availability rules — gate closure at 12:00 CET for day-ahead auction
FEATURE_AVAILABILITY = {
    "price_lag_1d": "D-1 13:00 CET",
    "price_lag_2d": "D-2 13:00 CET",
    "price_lag_7d": "D-7 13:00 CET",
    "price_rolling_7d_mean": "D-1 13:00 CET",
    "price_rolling_7d_std": "D-1 13:00 CET",
    "load_forecast": "D-1 10:00 CET",
    "wind_forecast": "D-1 10:00 CET",
    "solar_forecast": "D-1 10:00 CET",
    "wind_forecast_error_lag1": "D-1 end of day",
    "residual_load": "D-1 10:00 CET",
    "cross_border_flow": "D-1 14:00 CET",
    "hour_sin": "always",
    "hour_cos": "always",
    "weekday_sin": "always",
    "weekday_cos": "always",
    "is_weekend": "always",
    "is_holiday": "always",
}

DEFAULT_TRAINING_WINDOW_DAYS = 56
DEFAULT_XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

API_MAX_RETRIES = 3
API_BACKOFF_SECONDS = [2, 4, 8]
