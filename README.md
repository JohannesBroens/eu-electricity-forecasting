# Day-Ahead Electricity Price Forecasting

XGBoost-based day-ahead price forecasting pipeline for 21 European electricity bidding zones across the Nordics, Baltics, and Central-West Europe.

## Architecture

```
src/da_forecast/
  config.py          Zone definitions, interconnectors, trading params, quality thresholds
  data.py            Multi-source data loading with quality handling + audit logging
  sources/
    energinet.py     Energinet REST API (DK zones)
    entsoe.py        ENTSO-E Transparency Platform (all EU zones)
    openmeteo.py     Open-Meteo weather data (temperature, wind speed, solar irradiance)
    cache.py         Parquet-based caching with incremental merge
  features/          Lag features (gate-closure aware), calendar, residual load, weather
  models/            XGBoost day-ahead forecaster (single + per-hour modes)
  validation/
    completeness.py  Gap detection, daily completeness reports
    timezone.py      DST transition handling (23/25-hour days)
    outliers.py      Rolling z-score outlier detection (negative prices preserved)
    schema.py        DataFrame schema validation (columns, dtypes, timezone)
  monitoring/
    drift.py         Model performance tracking + drift detection
  backtest/          Walk-forward engine, trading strategies, Sharpe/drawdown metrics
notebooks/           Analysis notebooks (00-09)
scripts/             Data fetching, pipeline runner, backtest, heatmap generation
tests/               108 tests (pytest)
```

## Data sources

| Source | Zones | Auth | Datasets |
|--------|-------|------|----------|
| [Energinet DataService](https://www.energidataservice.dk/) | DK1, DK2 | None (free) | Prices, production mix, load, wind/solar forecasts |
| [ENTSO-E Transparency](https://newtransparency.entsoe.eu/) | All 21 zones | [API key](https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188-How-to-get-security-token) | Prices, generation, load, cross-border flows |
| [Open-Meteo](https://open-meteo.com/) | All zones | None (free) | Temperature, wind speed (10m/100m), solar radiation |

### Supported bidding zones

| Region | Zones |
|--------|-------|
| Denmark | DK_1 (West), DK_2 (East) |
| Norway | NO_1 (South-East), NO_2 (South), NO_3 (Middle), NO_4 (North), NO_5 (West) |
| Sweden | SE_1 (North), SE_2 (North-Central), SE_3 (Central), SE_4 (South) |
| Finland | FI |
| Germany | DE_LU (Germany-Luxembourg) |
| Benelux | NL (Netherlands), BE (Belgium) |
| France | FR |
| Central Europe | AT (Austria), PL (Poland) |
| Baltics | EE (Estonia), LV (Latvia), LT (Lithuania) |

## Quick start

```bash
uv sync
cp .env.example .env  # add ENTSO-E API key if available

# Fetch data (fetchers skip zones that already have cached data)
uv run python scripts/fetch_energinet_data.py   # DK zones, no auth needed
uv run python scripts/fetch_entsoe_data.py       # All zones, needs API key
uv run python scripts/fetch_weather_data.py      # Weather data, no auth needed

# Run full pipeline (all zones)
uv run python scripts/run_pipeline.py

# Fast sampled backtest (all zones in ~2 minutes)
uv run python scripts/fast_backtest.py --minutes 3 --samples 50

# Generate zone analysis maps
uv run python scripts/generate_eu_heatmap.py
uv run python scripts/generate_earnings_map.py

# Run tests
uv run pytest tests/ -v

# Or explore via notebooks
jupyter notebook notebooks/
```

## Backtest simulation (Jan 2024 -- Mar 2026)

### What it does

The pipeline includes a walk-forward backtest. Each day, the model is retrained on a rolling window of past data (no look-ahead), predicts 24 hourly prices for the next day, then a rank-spread strategy buys the 4 hours the model predicts cheapest and sells the 4 hours it predicts most expensive. P&L comes from the actual price spread between those hours.

This tests one specific question: **does the model rank hours within a day better than random?** If the model correctly identifies which hours are cheap and which are expensive, the spread is positive.

### What it does NOT tell you

- Whether this strategy would work in practice. Real trading involves market entry mechanics, counterparty availability, and execution details that are not modelled here.
- Whether the model knows something the market doesn't. The backtest measures ranking accuracy against realised prices, but in practice you're competing against other participants who have access to similar data and models.
- What the actual P&L would be. The simulation uses the daily mean of actual prices as the settlement reference. Real settlement depends on contract structure, exchange mechanics, and the specific positions you can actually enter.
- Risk-adjusted viability. Sharpe ratios from the simulation are unrealistically high compared to real-world strategies (which typically achieve 1-3). Win rate alone is meaningless -- a 90% win rate strategy can still lose money if the 10% losses are large enough.

Risk metrics per zone (Sharpe, Sortino, Calmar, profit factor, max drawdown) are available in the pipeline output. These describe the simulation, not expected real-world performance.

![Simulated P&L](output/backtest_pnl.png)

### Zone analysis

![Zone attractiveness](output/eu_zone_heatmap.png)

## Pipeline features

- **Data quality gates**: warns when completeness < 90%, imputation > 5%, or prices outside expected range
- **Imputation audit log**: every forward-filled value is logged to `output/imputation_audit.csv`
- **Schema validation**: checks column names, dtypes, and timezone awareness on all loaded data
- **Model drift detection**: tracks daily MAE per zone, flags when 7-day rolling exceeds 2x 30-day rolling
- **No synthetic fallback**: pipeline returns `None` rather than silently substituting fabricated data
- **Gate-closure aware features**: all lag features shift by >=24h to respect the 12:00 CET auction deadline
- **Negative prices are valid**: outlier detection explicitly avoids flagging negative prices (wind surplus signal)
- **Walk-forward backtesting**: strict temporal separation, no look-ahead bias
- **Fast sampled backtest**: samples evenly-spaced days with caching for quick iteration (~2 min for all zones)
- **Multi-source reconciliation**: Energinet is authoritative for DK zones; ENTSO-E fills adjacent zones

## Testing

108 tests covering data loading, validation, feature engineering, model training, and backtesting:

```bash
uv run pytest tests/ -v
```

## Docker

```bash
docker build -t da-forecast .
docker run -v ./data:/app/data -v ./output:/app/output da-forecast
```
