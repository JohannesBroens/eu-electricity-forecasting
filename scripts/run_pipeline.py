#!/usr/bin/env python3
"""Run the complete DA forecast pipeline for all available zones.

Steps:
1. FETCH & VALIDATE -- Load data, check quality, log issues
2. BUILD FEATURES  -- Construct feature matrix per zone
3. TRAIN & EVALUATE -- XGBoost model vs naive baseline
4. BACKTEST        -- Walk-forward with realistic constraints
5. REPORT          -- Generate plots and quality log

Usage:
    uv run python scripts/run_pipeline.py                 # All zones, full pipeline
    uv run python scripts/run_pipeline.py --validate      # Only validation (no model)
    uv run python scripts/run_pipeline.py --zone DK_1     # Single zone

Outputs:
    output/pipeline_report.log    -- Full text log of all quality checks
    output/prices_all_zones.png   -- Price comparison across zones
    output/quality_summary.png    -- Data quality dashboard
    output/backtest_pnl.png       -- Cumulative P&L curve
    output/feature_importance.png -- Top features by importance
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from da_forecast.config import ZONES, ZONE_LABELS
from da_forecast.data import load_all, load_prices, available_zones
from da_forecast.validation.completeness import find_gaps, daily_completeness_report
from da_forecast.validation.outliers import detect_outliers
from da_forecast.validation.timezone import find_dst_transitions, expected_hours_in_day

OUTPUT_DIR = Path(__file__).parent.parent / "output"
sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging():
    OUTPUT_DIR.mkdir(exist_ok=True)
    log_path = OUTPUT_DIR / "pipeline_report.log"

    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_path


def log(msg=""):
    logging.getLogger("pipeline").info(msg)


def section(title):
    log(f"\n{'='*60}")
    log(f"  {title}")
    log(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Step 1: Fetch & Validate
# ---------------------------------------------------------------------------
def step_validate(zone: str) -> dict:
    section(f"VALIDATE -- {zone} ({ZONE_LABELS.get(zone, '')})")

    data = load_all(zone)
    prices = data["prices"]
    wind_solar = data["wind_solar"]
    load_df = data["load"]

    errors = []
    warnings = []

    if prices is None:
        errors.append(f"No price data for {zone}")
        data["errors"] = errors
        data["warnings"] = warnings
        return data

    log(f"  Prices: {len(prices)} hours ({prices.index.min().date()} -> {prices.index.max().date()})")

    n_days = (prices.index.max() - prices.index.min()).days + 1
    expected_hours = n_days * 24
    actual_hours = len(prices)
    completeness_pct = actual_hours / expected_hours * 100
    log(f"  Expected ~{expected_hours} hours, got {actual_hours} ({completeness_pct:.1f}%)")

    if wind_solar is not None:
        ws_gaps = find_gaps(pd.DataFrame({"v": wind_solar.iloc[:, 0]}))
        if ws_gaps:
            warnings.append(f"Wind/solar: {len(ws_gaps)} missing hours")
            log(f"  Wind/solar gaps: {len(ws_gaps)} hours")
    if load_df is not None:
        load_gaps = find_gaps(pd.DataFrame({"v": load_df.iloc[:, 0]}))
        if load_gaps:
            warnings.append(f"Load: {len(load_gaps)} missing hours")
            log(f"  Load gaps: {len(load_gaps)} hours")

    # DST transition checks
    years = sorted(set(prices.index.year))
    for year in years:
        for t in find_dst_transitions(year):
            t_date = t["date"].date()
            p_start = prices.index.min().date()
            p_end = prices.index.max().date()
            if t_date >= p_start and t_date <= p_end:
                day_data = prices.loc[prices.index.date == t_date]
                actual = len(day_data)
                expected = t["hours"]
                if actual != expected:
                    warnings.append(f"DST {t['type']} ({t['date'].date()}): got {actual}h, expected {expected}h")

    # Outlier detection
    outlier_flags = detect_outliers(prices["price_eur_mwh"])
    n_outliers = int(outlier_flags["is_outlier"].sum())
    n_negative = int((prices["price_eur_mwh"] < 0).sum())
    neg_pct = n_negative / len(prices) * 100

    log(f"  Outliers flagged: {n_outliers} (extreme positive spikes)")
    log(f"  Negative prices: {n_negative} hours ({neg_pct:.1f}%) -- valid, not errors")
    if n_outliers > 10:
        warnings.append(f"{n_outliers} price outliers -- review before trading")

    # Imputation audit
    n_imputed = len(data.get("imputation_log", []))
    if n_imputed > 0:
        log(f"  Imputed: {n_imputed} values forward-filled (audit log available)")
        from collections import Counter
        col_counts = Counter(e["column"] for e in data["imputation_log"])
        for col, count in col_counts.most_common(5):
            log(f"    {col}: {count} values")

    if errors:
        log(f"\n  ERRORS: {errors}")
    if warnings:
        log(f"  WARNINGS: {warnings}")
    if not errors and not warnings:
        log(f"  All quality checks passed")

    data["errors"] = errors
    data["warnings"] = warnings
    data["n_outliers"] = n_outliers
    data["n_negative"] = n_negative
    return data


# ---------------------------------------------------------------------------
# Step 2: Build features
# ---------------------------------------------------------------------------
def step_build_features(data: dict) -> pd.DataFrame:
    zone = data["zone"]
    log(f"\n  Building features for {zone}...")

    from da_forecast.features.build import build_feature_matrix
    from da_forecast.features.lags import compute_lag_features
    from da_forecast.features.calendar import compute_calendar_features

    prices = data["prices"]
    wind_solar = data["wind_solar"]
    load_df = data["load"]

    if prices is None:
        log(f"  No price data for {zone}. Skipping.")
        return pd.DataFrame()

    if wind_solar is not None and load_df is not None:
        features = build_feature_matrix(prices, load_df, wind_solar)
        mode = "full (price + wind/solar + load)"
    else:
        features = compute_lag_features(prices)
        cal = compute_calendar_features(prices)
        for col in ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend", "is_holiday"]:
            if col in cal.columns:
                features[col] = cal[col]
        mode = "price-only (lags + calendar)"

    n_before = len(features)
    features = features.dropna()

    log(f"  Feature matrix: {features.shape[0]} rows x {features.shape[1]} columns ({mode})")
    log(f"  Warmup dropped: {n_before - len(features)} rows")
    if not features.empty:
        log(f"  Period: {features.index.min().date()} -> {features.index.max().date()}")

    return features


# ---------------------------------------------------------------------------
# Step 3: Train & evaluate
# ---------------------------------------------------------------------------
def step_train_evaluate(features: pd.DataFrame, zone: str) -> dict | None:
    if features.empty:
        log(f"  No features for {zone}. Skipping model.")
        return None

    from da_forecast.models.xgboost_da import DayAheadForecaster
    from da_forecast.models.evaluation import mae, rmse, smape, naive_baseline

    target = "price_eur_mwh"
    feature_cols = [c for c in features.columns if c != target]

    split = int(len(features) * 0.7)
    train = features.iloc[:split]
    test = features.iloc[split:]

    log(f"\n  Train: {len(train)} hours ({train.index.min().date()} -> {train.index.max().date()})")
    log(f"  Test:  {len(test)} hours ({test.index.min().date()} -> {test.index.max().date()})")

    forecaster = DayAheadForecaster(per_hour=False)
    forecaster.train(train, target_col=target)
    predictions = forecaster.predict(test[feature_cols])

    actuals = test[target].values
    predicted = predictions.values

    baseline = naive_baseline(features[target]).reindex(test.index)
    baseline_valid = baseline.dropna()
    test_valid = test.loc[baseline_valid.index]

    model_mae = mae(actuals, predicted)
    model_rmse = rmse(actuals, predicted)
    baseline_mae = mae(test_valid[target].values, baseline_valid.values)
    improvement = (1 - model_mae / baseline_mae) * 100

    log(f"  MAE:  {model_mae:.2f} EUR/MWh (baseline: {baseline_mae:.2f}, improvement: {improvement:+.1f}%)")
    log(f"  RMSE: {model_rmse:.2f} EUR/MWh")

    importance = forecaster.feature_importance()
    log(f"  Top 3 features: {', '.join(f'{f} ({r.importance:.3f})' for f, r in importance.head(3).iterrows())}")

    return {
        "zone": zone, "model_mae": model_mae, "baseline_mae": baseline_mae,
        "improvement": improvement, "importance": importance,
        "forecaster": forecaster, "train": train, "test": test,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Step 4: Backtest
# ---------------------------------------------------------------------------
def step_backtest(features: pd.DataFrame, zone: str) -> dict | None:
    if features.empty:
        return None

    from da_forecast.backtest.engine import BacktestEngine
    from da_forecast.backtest.strategies import ThresholdStrategy
    from da_forecast.backtest.metrics import backtest_summary

    strategy = ThresholdStrategy(
        threshold_eur=5.0,
        transaction_cost_eur_mwh=0.04,
        max_daily_trades=12,
    )
    has_fundamentals = features.shape[1] > 14
    window = 56 if has_fundamentals else 30
    engine = BacktestEngine(strategy=strategy, training_window_days=window)
    results = engine.run(features, target_col="price_eur_mwh")

    if results.empty:
        log(f"  Backtest for {zone}: not enough data")
        return None

    s = backtest_summary(results)
    log(f"  Backtest {zone}: P&L={s['total_pnl']:.0f} EUR, Sharpe={s['sharpe_ratio']:.2f}, "
        f"Win={s['win_rate_pct']:.0f}%, Trades={s['n_trades']}, "
        f"Drawdown={s['max_drawdown']:.0f} EUR")

    return {"zone": zone, "results": results, "summary": s}


# ---------------------------------------------------------------------------
# Step 5: Generate plots
# ---------------------------------------------------------------------------
def generate_plots(all_zone_data: dict, model_results: list, backtest_results: list):
    section("GENERATING PLOTS")

    # All-zone price comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    for zone, data in all_zone_data.items():
        if data["prices"] is not None:
            weekly = data["prices"]["price_eur_mwh"].resample("W").mean()
            ax.plot(weekly.index, weekly.values, label=f"{zone} ({data['source']})", alpha=0.8)
    ax.set_ylabel("EUR/MWh (weekly avg)")
    ax.set_title("Day-Ahead Prices -- All Available Zones")
    ax.legend(loc="upper right")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "prices_all_zones.png", dpi=150)
    plt.close(fig)
    log(f"  Saved: output/prices_all_zones.png")

    # Feature importance
    if model_results:
        best = model_results[0]
        imp = best["importance"].head(10)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(imp.index[::-1], imp["importance"].values[::-1])
        ax.set_xlabel("Feature Importance")
        ax.set_title(f"Top 10 Features -- {best['zone']} XGBoost")
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)
        log(f"  Saved: output/feature_importance.png")

    # Backtest P&L curve
    if backtest_results:
        fig, axes = plt.subplots(len(backtest_results), 1,
                                  figsize=(14, 4 * len(backtest_results)),
                                  squeeze=False)
        for i, bt in enumerate(backtest_results):
            ax = axes[i, 0]
            cumulative = bt["results"]["pnl"].cumsum()
            peak = cumulative.cummax()
            ax.fill_between(cumulative.index, peak, cumulative,
                           alpha=0.3, color="red", label="Drawdown")
            ax.plot(cumulative.index, cumulative.values, color="darkblue", linewidth=1)
            s = bt["summary"]
            ax.set_title(f"{bt['zone']} -- P&L: {s['total_pnl']:.0f} EUR | "
                        f"Sharpe: {s['sharpe_ratio']:.2f} | "
                        f"Win: {s['win_rate_pct']:.0f}%")
            ax.set_ylabel("Cumulative P&L (EUR)")
            ax.legend()
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "backtest_pnl.png", dpi=150)
        plt.close(fig)
        log(f"  Saved: output/backtest_pnl.png")

    # Data quality dashboard
    zones_with_data = {z: d for z, d in all_zone_data.items() if d["prices"] is not None}
    if zones_with_data:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        neg_data = {z: (d["prices"]["price_eur_mwh"] < 0).mean() * 100
                    for z, d in zones_with_data.items()}
        axes[0].bar(neg_data.keys(), neg_data.values(), color="coral")
        axes[0].set_ylabel("% of hours")
        axes[0].set_title("Negative Price Frequency")
        axes[0].tick_params(axis="x", rotation=45)

        vol_data = {z: d["prices"]["price_eur_mwh"].std()
                    for z, d in zones_with_data.items()}
        axes[1].bar(vol_data.keys(), vol_data.values(), color="steelblue")
        axes[1].set_ylabel("EUR/MWh")
        axes[1].set_title("Price Volatility (Std Dev)")
        axes[1].tick_params(axis="x", rotation=45)

        comp_data = {}
        for z, d in zones_with_data.items():
            p = d["prices"]
            n_days = (p.index.max() - p.index.min()).days + 1
            comp_data[z] = len(p) / (n_days * 24) * 100
        axes[2].bar(comp_data.keys(), comp_data.values(), color="seagreen")
        axes[2].set_ylabel("% complete")
        axes[2].set_title("Hourly Completeness")
        axes[2].set_ylim(95, 101)
        axes[2].tick_params(axis="x", rotation=45)

        plt.suptitle("Data Quality Dashboard", fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "quality_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: output/quality_summary.png")


# ---------------------------------------------------------------------------
# Revenue estimation
# ---------------------------------------------------------------------------
def step_revenue_estimation(backtest_results: list[dict]):
    """Project annual revenue from backtest results under explicit assumptions.

    Assumptions:
      A1. Position size: 1 / 5 / 10 MW (= MWh per hour in day-ahead)
      A2. Trading fee: 0.04 EUR/MWh (Nord Pool day-ahead fee schedule 2025)
      A3. No market impact on clearing prices (valid for <50 MW positions)
      A4. Trading 365 days/year (electricity markets clear every day)
      A5. No balancing/imbalance costs (~5-15% drag in practice)
      A6. No collateral/margin costs
      A7. Model performance degrades out-of-sample vs backtest

    Haircut scenarios for A7:
      Optimistic:   100% of backtest P&L
      Base case:     50% of backtest P&L (typical live vs backtest ratio)
      Conservative:  25% of backtest P&L
    """
    section("REVENUE ESTIMATION")
    log("  Assumptions:")
    log("  A1. Position sizes: 1 MW, 5 MW, 10 MW")
    log("  A2. Trading fee: 0.04 EUR/MWh (Nord Pool fee schedule)")
    log("  A3. No market impact (valid for small positions)")
    log("  A4. 365 trading days/year")
    log("  A5. Balancing/imbalance costs excluded (~5-15% drag in practice)")
    log("  A6. Collateral costs excluded (margin capital earns 0%)")
    log("  A7. Model haircuts: 100% / 50% / 25% of backtest P&L")
    log("  Sources:")
    log("    - Nord Pool fee: https://www.nordpoolgroup.com/trading/fees/")
    log("    - Market impact threshold: ACER 2024 Wholesale Market Report")
    log("    - Live vs backtest ratio: De Prado, 'Advances in Financial ML', Ch. 11")
    log()

    position_sizes = [1, 5, 10]
    haircuts = [("Optimistic", 1.0), ("Base case", 0.5), ("Conservative", 0.25)]

    for bt in backtest_results:
        zone = bt["zone"]
        s = bt["summary"]
        backtest_pnl_per_mwh = s["total_pnl"]
        n_days = s["n_trading_days"]

        if n_days == 0:
            continue

        annual_factor = 365.0 / n_days

        log(f"  {zone} (backtest: {n_days} days, {backtest_pnl_per_mwh:.0f} EUR @ 1 MWh)")
        log(f"  {'Scenario':<16s}  {'1 MW':>12s}  {'5 MW':>12s}  {'10 MW':>12s}")
        log(f"  {'-'*56}")

        for scenario_name, factor in haircuts:
            row = f"  {scenario_name:<16s}"
            for mw in position_sizes:
                annual_eur = backtest_pnl_per_mwh * annual_factor * mw * factor
                row += f"  {annual_eur:>8,.0f} EUR"
            log(row)

        log(f"  {'':16s}  (multiply by ~7.46 for DKK)")
        log()

    if len(backtest_results) > 1:
        total_pnl = sum(bt["summary"]["total_pnl"] for bt in backtest_results)
        avg_days = np.mean([bt["summary"]["n_trading_days"] for bt in backtest_results])
        annual_factor = 365.0 / avg_days if avg_days > 0 else 0

        log(f"  COMBINED (all {len(backtest_results)} zones)")
        log(f"  {'Scenario':<16s}  {'1 MW':>12s}  {'5 MW':>12s}  {'10 MW':>12s}")
        log(f"  {'-'*56}")
        for scenario_name, factor in haircuts:
            row = f"  {scenario_name:<16s}"
            for mw in position_sizes:
                annual_eur = total_pnl * annual_factor * mw * factor
                row += f"  {annual_eur:>8,.0f} EUR"
            log(row)
        log(f"  {'':16s}  (multiply by ~7.46 for DKK)")
        log()

    log("  WARNING: These projections are indicative only.")
    log("  Real-world P&L is typically 25-50% of backtest estimates due to:")
    log("    - Model degradation on unseen market regimes")
    log("    - Imbalance settlement costs (not modelled)")
    log("    - Execution timing and partial fills")
    log("    - Collateral requirements reducing deployable capital")
    log("  The 'Base case' scenario is the most realistic starting point.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger, log_path = setup_logging()

    parser = argparse.ArgumentParser(description="Run the DA forecast pipeline")
    parser.add_argument("--zone", default=None, help="Single zone (default: all available)")
    parser.add_argument("--validate", action="store_true", help="Only validation")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtest (faster)")
    args = parser.parse_args()

    section("DA FORECAST PIPELINE")
    log(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    zones_available = available_zones()
    if args.zone:
        zones_to_run = [args.zone]
    else:
        zones_to_run = [z for z, src in zones_available.items() if src != "none"]

    log(f"  Zones: {', '.join(zones_to_run)}")
    log(f"  Mode: {'validate only' if args.validate else 'full pipeline'}")
    log(f"\n  Data sources:")
    for z, src in zones_available.items():
        marker = "->" if z in zones_to_run else " "
        log(f"    {marker} {z}: {src}")

    all_zone_data = {}
    model_results = []
    backtest_results = []

    for zone in zones_to_run:
        data = step_validate(zone)
        all_zone_data[zone] = data

        if args.validate:
            continue
        if data.get("errors"):
            log(f"  {zone}: quality errors -- skipping model/backtest")
            continue

        section(f"FEATURES + MODEL + BACKTEST -- {zone}")
        features = step_build_features(data)

        model_result = step_train_evaluate(features, zone)
        if model_result:
            model_results.append(model_result)

        if not args.no_backtest:
            bt_result = step_backtest(features, zone)
            if bt_result:
                backtest_results.append(bt_result)

    if not args.validate:
        generate_plots(all_zone_data, model_results, backtest_results)

    section("SUMMARY")
    log(f"  Zones processed: {len(zones_to_run)}")
    if model_results:
        log(f"\n  Model performance:")
        log(f"  {'Zone':>8s}  {'MAE':>8s}  {'Baseline':>8s}  {'Improvement':>12s}")
        log(f"  {'-'*42}")
        for r in model_results:
            log(f"  {r['zone']:>8s}  {r['model_mae']:8.2f}  {r['baseline_mae']:8.2f}  {r['improvement']:>+11.1f}%")

    if backtest_results:
        log(f"\n  Backtest results (with 0.04 EUR/MWh costs, max 12 trades/day):")
        log(f"  {'Zone':>8s}  {'P&L (EUR)':>10s}  {'Sharpe':>8s}  {'Win%':>6s}  {'Trades':>8s}")
        log(f"  {'-'*48}")
        for bt in backtest_results:
            s = bt["summary"]
            log(f"  {bt['zone']:>8s}  {s['total_pnl']:>10.0f}  {s['sharpe_ratio']:>8.2f}  "
                f"{s['win_rate_pct']:>5.0f}%  {s['n_trades']:>8d}")

    log(f"\n  Outputs saved to: {OUTPUT_DIR}/")
    log(f"  Full log: {log_path}")
    log(f"\n  Note: Sharpe ratios are calculated on daily P&L (annualized sqrt(365)).")
    log(f"  Real-world strategies typically achieve Sharpe 1-3.")
    log(f"  Values above 3 suggest the backtest is still optimistic")
    log(f"  (no market impact, no slippage, perfect execution).")

    if backtest_results:
        step_revenue_estimation(backtest_results)


if __name__ == "__main__":
    main()
