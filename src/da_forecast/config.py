"""Pipeline configuration.

Zone codes, interconnector capacities, feature availability rules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Nordic-Continental bidding zones
ZONES = ["DK_1", "DK_2", "NO_2", "SE_3", "SE_4", "DE_LU"]
PRIMARY_ZONES = ["DK_1", "DK_2"]

ZONE_LABELS = {
    "DK_1": "West Denmark",
    "DK_2": "East Denmark",
    "NO_2": "South Norway",
    "SE_3": "Central Sweden",
    "SE_4": "South Sweden",
    "DE_LU": "Germany-Luxembourg",
}

# ENTSO-E EIC area codes
ZONE_EIC = {
    "DK_1": "10YDK-1--------W",
    "DK_2": "10YDK-2--------M",
    "NO_2": "10YNO-2--------T",
    "SE_3": "10Y1001A1001A46L",
    "SE_4": "10Y1001A1001A47J",
    "DE_LU": "10Y1001A1001A82H",
}

# Interconnectors: (from_zone, to_zone, nominal_capacity_mw)
# Sources: ENTSO-E TYNDP 2024, Energinet system data
INTERCONNECTORS = [
    ("DK_1", "DE_LU", 2500),
    ("DK_1", "NO_2", 1700),
    ("DK_1", "SE_3", 740),
    ("DK_2", "DE_LU", 600),
    ("DK_2", "SE_4", 1700),
    ("DK_1", "DK_2", 600),
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
