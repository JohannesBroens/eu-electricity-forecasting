"""Tests for BacktestEngine, ThresholdStrategy, and backtest metrics."""

import numpy as np
import pandas as pd
import pytest

from da_forecast.backtest.strategies import ThresholdStrategy
from da_forecast.backtest.metrics import (
    sharpe_ratio,
    max_drawdown,
    win_rate,
    backtest_summary,
)
from da_forecast.backtest.engine import BacktestEngine


TZ = "Europe/Copenhagen"


# --- ThresholdStrategy ---

class TestThresholdStrategy:
    def test_buy_signal_above_threshold(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        predictions = pd.Series(60.0, index=idx)
        baseline = pd.Series(50.0, index=idx)
        strategy = ThresholdStrategy(threshold_eur=5.0, position_mwh=1.0)
        signals = strategy.generate_signals(predictions, baseline)
        # Diff = +10, above threshold -> all buy (+1.0)
        assert (signals == 1.0).all()

    def test_sell_signal_below_threshold(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        predictions = pd.Series(40.0, index=idx)
        baseline = pd.Series(50.0, index=idx)
        strategy = ThresholdStrategy(threshold_eur=5.0, position_mwh=1.0)
        signals = strategy.generate_signals(predictions, baseline)
        # Diff = -10, below -threshold -> all sell (-1.0)
        assert (signals == -1.0).all()

    def test_no_signal_within_threshold(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        predictions = pd.Series(52.0, index=idx)
        baseline = pd.Series(50.0, index=idx)
        strategy = ThresholdStrategy(threshold_eur=5.0)
        signals = strategy.generate_signals(predictions, baseline)
        # Diff = +2, within threshold -> hold (0.0)
        assert (signals == 0.0).all()

    def test_mixed_signals(self):
        idx = pd.date_range("2025-01-06", periods=3, freq="h", tz=TZ)
        predictions = pd.Series([60.0, 50.0, 40.0], index=idx)
        baseline = pd.Series([50.0, 50.0, 50.0], index=idx)
        strategy = ThresholdStrategy(threshold_eur=5.0)
        signals = strategy.generate_signals(predictions, baseline)
        assert signals.iloc[0] == 1.0   # buy
        assert signals.iloc[1] == 0.0   # hold
        assert signals.iloc[2] == -1.0  # sell

    def test_max_daily_trades_limits_signals(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        predictions = pd.Series(60.0, index=idx)
        baseline = pd.Series(50.0, index=idx)
        strategy = ThresholdStrategy(threshold_eur=5.0, max_daily_trades=5)
        signals = strategy.generate_signals(predictions, baseline)
        active_trades = (signals != 0).sum()
        assert active_trades == 5

    def test_custom_position_size(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        predictions = pd.Series(60.0, index=idx)
        baseline = pd.Series(50.0, index=idx)
        strategy = ThresholdStrategy(threshold_eur=5.0, position_mwh=2.5)
        signals = strategy.generate_signals(predictions, baseline)
        assert (signals == 2.5).all()


# --- backtest metrics ---

class TestSharpeRatio:
    def test_constant_positive_pnl(self):
        idx = pd.date_range("2025-01-06", periods=48, freq="h", tz=TZ)
        results = pd.DataFrame({"pnl": np.ones(48)}, index=idx)
        sr = sharpe_ratio(results)
        # Constant daily PnL -> std=0 -> inf Sharpe
        assert sr == float("inf")

    def test_zero_pnl(self):
        idx = pd.date_range("2025-01-06", periods=48, freq="h", tz=TZ)
        results = pd.DataFrame({"pnl": np.zeros(48)}, index=idx)
        sr = sharpe_ratio(results)
        assert sr == 0.0

    def test_positive_sharpe_for_good_strategy(self):
        idx = pd.date_range("2025-01-01", periods=24 * 30, freq="h", tz=TZ)
        rng = np.random.default_rng(42)
        pnl = rng.normal(1.0, 0.5, len(idx))  # positive mean, low std
        results = pd.DataFrame({"pnl": pnl}, index=idx)
        sr = sharpe_ratio(results)
        assert sr > 0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        cumulative = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert max_drawdown(cumulative) == 0.0

    def test_known_drawdown(self):
        cumulative = pd.Series([10.0, 15.0, 8.0, 12.0])
        # Peak 15, trough 8 -> drawdown = 7
        assert max_drawdown(cumulative) == pytest.approx(7.0)

    def test_all_negative(self):
        cumulative = pd.Series([0.0, -5.0, -10.0, -3.0])
        # Peak 0, lowest -10 -> drawdown = 10
        assert max_drawdown(cumulative) == pytest.approx(10.0)


class TestWinRate:
    def test_all_winners(self):
        returns = pd.Series([1.0, 2.0, 3.0])
        assert win_rate(returns) == pytest.approx(100.0)

    def test_all_losers(self):
        returns = pd.Series([-1.0, -2.0, -3.0])
        assert win_rate(returns) == pytest.approx(0.0)

    def test_fifty_fifty(self):
        returns = pd.Series([1.0, -1.0, 2.0, -2.0])
        assert win_rate(returns) == pytest.approx(50.0)

    def test_no_trades(self):
        returns = pd.Series([0.0, 0.0, 0.0])
        assert win_rate(returns) == 0.0

    def test_empty_series(self):
        returns = pd.Series([], dtype=float)
        assert win_rate(returns) == 0.0


class TestBacktestSummary:
    def test_summary_keys(self):
        idx = pd.date_range("2025-01-06", periods=48, freq="h", tz=TZ)
        results = pd.DataFrame({
            "pnl": np.random.default_rng(42).normal(0, 1, 48),
            "position": np.random.default_rng(42).choice([-1, 0, 1], 48),
        }, index=idx)
        summary = backtest_summary(results)
        expected_keys = {
            "total_pnl", "daily_pnl_mean", "daily_pnl_std",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor",
            "max_drawdown", "win_rate_pct",
            "n_trades", "n_trading_days", "avg_win", "avg_loss",
        }
        assert expected_keys == set(summary.keys())

    def test_total_pnl_matches_sum(self):
        idx = pd.date_range("2025-01-06", periods=48, freq="h", tz=TZ)
        pnl_values = np.array([1.0] * 24 + [-0.5] * 24)
        results = pd.DataFrame({
            "pnl": pnl_values,
            "position": np.ones(48),
        }, index=idx)
        summary = backtest_summary(results)
        assert summary["total_pnl"] == pytest.approx(pnl_values.sum())

    def test_n_trades_counts_nonzero_positions(self):
        idx = pd.date_range("2025-01-06", periods=48, freq="h", tz=TZ)
        positions = np.zeros(48)
        positions[:10] = 1.0
        results = pd.DataFrame({
            "pnl": np.ones(48),
            "position": positions,
        }, index=idx)
        summary = backtest_summary(results)
        assert summary["n_trades"] == 10


# --- BacktestEngine ---

FAST_XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "learning_rate": 0.3,
    "n_estimators": 5,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "random_state": 42,
}


class TestBacktestEngine:
    @pytest.fixture(autouse=True)
    def _patch_default_params(self, monkeypatch):
        """Patch DEFAULT_XGBOOST_PARAMS so BacktestEngine trains fast."""
        import da_forecast.config as cfg
        monkeypatch.setattr(cfg, "DEFAULT_XGBOOST_PARAMS", FAST_XGBOOST_PARAMS)

    @pytest.fixture
    def small_feature_matrix(self):
        """Create a minimal feature matrix for backtesting.

        Need at least training_window_days + some test days of data.
        Use a small training window (3 days) to keep tests fast.
        """
        n_days = 7
        n_hours = n_days * 24
        idx = pd.date_range("2025-01-01", periods=n_hours, freq="h", tz=TZ)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "price_eur_mwh": 40 + 10 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + rng.normal(0, 2, n_hours),
            "feature_a": rng.normal(0, 1, n_hours),
            "feature_b": rng.normal(0, 1, n_hours),
        }, index=idx)
        return df

    def test_run_returns_dataframe(self, small_feature_matrix):
        engine = BacktestEngine(
            training_window_days=3,
            strategy=ThresholdStrategy(threshold_eur=5.0),
        )
        results = engine.run(small_feature_matrix)
        assert isinstance(results, pd.DataFrame)

    def test_result_columns(self, small_feature_matrix):
        engine = BacktestEngine(
            training_window_days=3,
            strategy=ThresholdStrategy(threshold_eur=5.0),
        )
        results = engine.run(small_feature_matrix)
        expected_cols = {"predicted_price", "actual_price", "baseline_price", "position", "pnl"}
        assert expected_cols == set(results.columns)

    def test_no_results_with_insufficient_data(self):
        # Only 2 days of data with 7-day training window -> no test days
        idx = pd.date_range("2025-01-01", periods=48, freq="h", tz=TZ)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "price_eur_mwh": rng.normal(50, 5, 48),
            "feature_a": rng.normal(0, 1, 48),
        }, index=idx)
        engine = BacktestEngine(training_window_days=7)
        results = engine.run(df)
        assert len(results) == 0

    def test_results_are_sorted(self, small_feature_matrix):
        engine = BacktestEngine(
            training_window_days=3,
            strategy=ThresholdStrategy(threshold_eur=5.0),
        )
        results = engine.run(small_feature_matrix)
        if len(results) > 0:
            assert results.index.is_monotonic_increasing

    def test_backtest_with_transaction_costs(self, small_feature_matrix):
        strategy_no_cost = ThresholdStrategy(threshold_eur=5.0, transaction_cost_eur_mwh=0.0)
        strategy_with_cost = ThresholdStrategy(threshold_eur=5.0, transaction_cost_eur_mwh=1.0)

        engine_no_cost = BacktestEngine(training_window_days=3, strategy=strategy_no_cost)
        engine_with_cost = BacktestEngine(training_window_days=3, strategy=strategy_with_cost)

        results_no_cost = engine_no_cost.run(small_feature_matrix)
        results_with_cost = engine_with_cost.run(small_feature_matrix)

        if len(results_no_cost) > 0 and len(results_with_cost) > 0:
            # Transaction costs should reduce total PnL
            pnl_no_cost = results_no_cost["pnl"].sum()
            pnl_with_cost = results_with_cost["pnl"].sum()
            assert pnl_with_cost <= pnl_no_cost
