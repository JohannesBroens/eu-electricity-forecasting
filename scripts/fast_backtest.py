#!/usr/bin/env python3
"""Fast sampled backtest across all zones with time budget.

Instead of testing every single day (700+ days × 21 zones = hours of compute),
this script:
1. Samples evenly-spaced test days across the dataset
2. Runs all zones in parallel (one process per zone)
3. Caches results so subsequent runs only fill gaps
4. Stops when the time budget is exhausted

Results are saved to data/backtest_cache/ (gitignored).

Usage:
    uv run python scripts/fast_backtest.py                    # 15 min budget
    uv run python scripts/fast_backtest.py --minutes 30       # 30 min budget
    uv run python scripts/fast_backtest.py --samples 20       # 20 test days per zone
    uv run python scripts/fast_backtest.py --refresh          # ignore cache, start fresh
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from da_forecast.config import ZONES, ZONE_LABELS
from da_forecast.data import load_all, available_zones
from da_forecast.features.build import build_feature_matrix
from da_forecast.features.lags import compute_lag_features
from da_forecast.features.calendar import compute_calendar_features
from da_forecast.models.xgboost_da import DayAheadForecaster
from da_forecast.models.evaluation import naive_baseline
from da_forecast.backtest.strategies import ThresholdStrategy
from da_forecast.backtest.metrics import backtest_summary

CACHE_DIR = Path(__file__).parent.parent / "data" / "backtest_cache"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def build_features_for_zone(zone: str) -> pd.DataFrame:
    """Load data and build feature matrix for a zone."""
    data = load_all(zone)
    prices = data.get("prices")
    if prices is None:
        return pd.DataFrame()

    ws = data.get("wind_solar")
    load_df = data.get("load")

    if ws is not None and load_df is not None:
        features = build_feature_matrix(prices, load_df, ws)
    else:
        features = compute_lag_features(prices)
        cal = compute_calendar_features(prices)
        for col in ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend", "is_holiday"]:
            if col in cal.columns:
                features[col] = cal[col]

    return features.dropna()


def sample_test_days(features: pd.DataFrame, n_samples: int, training_window: int = 56) -> list:
    """Pick evenly-spaced test days across the dataset."""
    dates = features.index.normalize().unique().sort_values()
    min_date = dates[0] + pd.Timedelta(days=training_window)
    eligible = dates[dates >= min_date]

    if len(eligible) <= n_samples:
        return list(eligible)

    # Evenly spaced indices
    indices = np.linspace(0, len(eligible) - 1, n_samples, dtype=int)
    return [eligible[i] for i in indices]


def run_single_day(features, test_date, target_col, training_window, strategy):
    """Run backtest for a single day. Returns DataFrame of hourly results."""
    feature_cols = [c for c in features.columns if c != target_col]

    train_end = test_date
    train_start = train_end - pd.Timedelta(days=training_window)
    train_mask = (features.index >= train_start) & (features.index < train_end)
    train_data = features.loc[train_mask].dropna()
    test_mask = features.index.normalize() == test_date
    test_data = features.loc[test_mask]

    if train_data.empty or test_data.empty or len(train_data) < 48:
        return None

    forecaster = DayAheadForecaster(per_hour=False)
    forecaster.train(train_data, target_col=target_col)
    predictions = forecaster.predict(test_data[feature_cols])
    actuals = test_data[target_col]
    baseline = naive_baseline(features[target_col]).reindex(test_data.index)

    signals = strategy.generate_signals(predictions, baseline.fillna(predictions))
    pnl = signals * (actuals - baseline.fillna(actuals))

    if hasattr(strategy, "transaction_cost_eur_mwh") and strategy.transaction_cost_eur_mwh > 0:
        pnl -= signals.abs() * strategy.transaction_cost_eur_mwh

    return pd.DataFrame({
        "predicted_price": predictions,
        "actual_price": actuals,
        "baseline_price": baseline,
        "position": signals,
        "pnl": pnl,
    })


def backtest_zone(zone: str, n_samples: int, cache_dir: Path, refresh: bool) -> dict:
    """Run sampled backtest for one zone. Returns summary dict."""
    target_col = "price_eur_mwh"
    training_window = 56
    strategy = ThresholdStrategy(threshold_eur=5.0, transaction_cost_eur_mwh=0.04, max_daily_trades=12)

    cache_file = cache_dir / f"{zone}_backtest.parquet"
    cached_dates = set()
    cached_results = []

    if not refresh and cache_file.exists():
        cached_df = pd.read_parquet(cache_file)
        cached_dates = set(cached_df.index.normalize().unique())
        cached_results.append(cached_df)

    # Build features
    features = build_features_for_zone(zone)
    if features.empty:
        return {"zone": zone, "status": "no_data"}

    test_days = sample_test_days(features, n_samples, training_window)

    # Filter out already cached days
    new_days = [d for d in test_days if d not in cached_dates]

    new_results = []
    for day in new_days:
        result = run_single_day(features, day, target_col, training_window, strategy)
        if result is not None:
            new_results.append(result)

    # Merge with cache
    all_parts = cached_results + new_results
    if not all_parts:
        return {"zone": zone, "status": "no_results"}

    combined = pd.concat(all_parts).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]

    # Save cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_file)

    # Compute summary
    s = backtest_summary(combined)
    n_test_days = len(combined.index.normalize().unique())

    return {
        "zone": zone,
        "status": "ok",
        "total_pnl": s["total_pnl"],
        "sharpe_ratio": s["sharpe_ratio"],
        "win_rate_pct": s["win_rate_pct"],
        "n_trades": s["n_trades"],
        "n_test_days": n_test_days,
        "n_sampled": len(test_days),
        "n_cached": len(cached_dates),
        "n_new": len(new_days),
        "max_drawdown": s["max_drawdown"],
        "n_trading_days": s.get("n_trading_days", n_test_days),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=15, help="Time budget in minutes")
    parser.add_argument("--samples", type=int, default=30, help="Test days per zone")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--refresh", action="store_true", help="Ignore cache")
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + args.minutes * 60

    zones_available = available_zones()
    zones_to_run = [z for z, src in zones_available.items() if src != "none"]

    print(f"Fast backtest: {len(zones_to_run)} zones, {args.samples} sampled days each")
    print(f"Time budget: {args.minutes} min, workers: {args.workers}")
    print(f"Cache: {CACHE_DIR}")
    print()

    results = []

    # Run zones with ProcessPoolExecutor
    # But XGBoost already uses all cores, so we run sequentially to avoid contention
    for zone in zones_to_run:
        if time.time() > deadline:
            print(f"  {zone}: SKIPPED (time budget exceeded)")
            results.append({"zone": zone, "status": "timeout"})
            continue

        print(f"  {zone}...", end=" ", flush=True)
        t0 = time.time()
        try:
            r = backtest_zone(zone, args.samples, CACHE_DIR, args.refresh)
            elapsed = time.time() - t0
            if r["status"] == "ok":
                print(f"P&L={r['total_pnl']:,.0f} EUR, Sharpe={r['sharpe_ratio']:.2f}, "
                      f"Win={r['win_rate_pct']:.0f}%, Days={r['n_test_days']} "
                      f"({r['n_new']} new, {r['n_cached']} cached) [{elapsed:.0f}s]")
            else:
                print(f"{r['status']} [{elapsed:.0f}s]")
            results.append(r)
        except Exception as e:
            print(f"FAIL: {e}")
            results.append({"zone": zone, "status": f"error: {e}"})

    elapsed_total = time.time() - start_time

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]

    print(f"\n{'='*80}")
    print(f"  SAMPLED BACKTEST RESULTS ({len(ok_results)}/{len(zones_to_run)} zones, {elapsed_total:.0f}s)")
    print(f"{'='*80}")
    print(f"  {'Zone':>6s}  {'P&L (EUR)':>10s}  {'Sharpe':>8s}  {'Win%':>6s}  {'Trades':>8s}  {'Days':>6s}")
    print(f"  {'-'*52}")

    for r in sorted(ok_results, key=lambda x: x["total_pnl"], reverse=True):
        print(f"  {r['zone']:>6s}  {r['total_pnl']:>10,.0f}  {r['sharpe_ratio']:>8.2f}  "
              f"{r['win_rate_pct']:>5.0f}%  {r['n_trades']:>8d}  {r['n_test_days']:>6d}")

    if ok_results:
        total_pnl = sum(r["total_pnl"] for r in ok_results)
        print(f"\n  {'TOTAL':>6s}  {total_pnl:>10,.0f}")
        print(f"\n  Note: P&L is from sampled days only ({args.samples} per zone).")
        print(f"  Annualized estimates require scaling by (365 / n_test_days).")
        print(f"  Results cached in {CACHE_DIR}/ -- re-run to fill more days.")

    # Save summary JSON for earnings map
    summary_path = OUTPUT_DIR / "backtest_summary.json"
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(ok_results, f, indent=2)
    print(f"\n  Saved: {summary_path}")


if __name__ == "__main__":
    main()
