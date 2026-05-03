"""Tests for BacktestEngine, RankSpreadStrategy, and backtest metrics."""

import numpy as np
import pandas as pd
import pytest

from da_forecast.backtest.strategies import RankSpreadStrategy
from da_forecast.backtest.metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    profit_factor,
    max_drawdown,
    win_rate,
    backtest_summary,
)
from da_forecast.backtest.engine import BacktestEngine


TZ = "Europe/Copenhagen"


# --- RankSpreadStrategy ---

class TestRankSpreadStrategy:
    def test_buys_cheapest_sells_most_expensive(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        # Clear ranking: hours 0-3 cheapest, hours 20-23 most expensive
        prices = pd.Series(np.arange(24, dtype=float) * 10, index=idx)
        strategy = RankSpreadStrategy(n_long=4, n_short=4, position_mwh=1.0, transaction_cost_eur_mwh=0.0)
        signals = strategy.generate_signals(prices)
        assert (signals.iloc[:4] == 1.0).all()    # buy cheapest 4
        assert (signals.iloc[20:] == -1.0).all()   # sell most expensive 4
        assert (signals.iloc[4:20] == 0.0).all()   # hold middle hours

    def test_balanced_positions(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        prices = pd.Series(np.random.randn(24), index=idx)
        strategy = RankSpreadStrategy(n_long=4, n_short=4)
        signals = strategy.generate_signals(prices)
        n_long = (signals > 0).sum()
        n_short = (signals < 0).sum()
        assert n_long == 4
        assert n_short == 4

    def test_pnl_positive_when_ranking_correct(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        # Predictions match actuals perfectly -> ranking is correct -> profit
        actuals = pd.Series(np.arange(24, dtype=float) * 10, index=idx)
        predictions = actuals.copy()
        strategy = RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=0.0)
        pnl = strategy.compute_pnl(predictions, actuals)
        assert pnl.sum() > 0

    def test_pnl_negative_when_ranking_reversed(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        actuals = pd.Series(np.arange(24, dtype=float) * 10, index=idx)
        # Predictions are reversed: model thinks cheap hours are expensive
        predictions = pd.Series(np.arange(23, -1, -1, dtype=float) * 10, index=idx)
        strategy = RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=0.0)
        pnl = strategy.compute_pnl(predictions, actuals)
        assert pnl.sum() < 0

    def test_transaction_costs_reduce_pnl(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        actuals = pd.Series(np.arange(24, dtype=float) * 10, index=idx)
        predictions = actuals.copy()
        s_free = RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=0.0)
        s_cost = RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=5.0)
        pnl_free = s_free.compute_pnl(predictions, actuals).sum()
        pnl_cost = s_cost.compute_pnl(predictions, actuals).sum()
        assert pnl_cost < pnl_free

    def test_custom_position_size(self):
        idx = pd.date_range("2025-01-06", periods=24, freq="h", tz=TZ)
        prices = pd.Series(np.arange(24, dtype=float), index=idx)
        strategy = RankSpreadStrategy(n_long=4, n_short=4, position_mwh=2.5)
        signals = strategy.generate_signals(prices)
        assert signals.max() == 2.5
        assert signals.min() == -2.5


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
            strategy=RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=0.0),
        )
        results = engine.run(small_feature_matrix)
        assert isinstance(results, pd.DataFrame)

    def test_result_columns(self, small_feature_matrix):
        engine = BacktestEngine(
            training_window_days=3,
            strategy=RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=0.0),
        )
        results = engine.run(small_feature_matrix)
        expected_cols = {"predicted_price", "actual_price", "position", "pnl"}
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
            strategy=RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=0.0),
        )
        results = engine.run(small_feature_matrix)
        if len(results) > 0:
            assert results.index.is_monotonic_increasing

    def test_backtest_with_transaction_costs(self, small_feature_matrix):
        strategy_no_cost = RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=0.0)
        strategy_with_cost = RankSpreadStrategy(n_long=4, n_short=4, transaction_cost_eur_mwh=1.0)

        engine_no_cost = BacktestEngine(training_window_days=3, strategy=strategy_no_cost)
        engine_with_cost = BacktestEngine(training_window_days=3, strategy=strategy_with_cost)

        results_no_cost = engine_no_cost.run(small_feature_matrix)
        results_with_cost = engine_with_cost.run(small_feature_matrix)

        if len(results_no_cost) > 0 and len(results_with_cost) > 0:
            # Transaction costs should reduce total PnL
            pnl_no_cost = results_no_cost["pnl"].sum()
            pnl_with_cost = results_with_cost["pnl"].sum()
            assert pnl_with_cost <= pnl_no_cost
