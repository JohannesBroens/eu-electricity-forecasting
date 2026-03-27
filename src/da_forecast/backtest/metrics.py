"""Backtest performance metrics.

Sharpe ratio uses daily aggregate P&L (not hourly) because consecutive
hourly returns are correlated -- using hourly with sqrt(8760) annualization
inflates Sharpe by ~5x.
"""
import numpy as np
import pandas as pd


def sharpe_ratio(results: pd.DataFrame) -> float:
    """Annualized Sharpe ratio based on daily P&L."""
    pnl = results["pnl"]
    daily_pnl = pnl.groupby(pnl.index.date).sum()
    if daily_pnl.std() == 0:
        return float("inf") if daily_pnl.mean() > 0 else 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(365))


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Maximum peak-to-trough drawdown in EUR."""
    peak = cumulative_pnl.cummax()
    drawdown = peak - cumulative_pnl
    return float(drawdown.max())


def win_rate(returns: pd.Series) -> float:
    """Percentage of profitable trades (non-zero returns only)."""
    trades = returns[returns != 0]
    if len(trades) == 0:
        return 0.0
    return float(100 * (trades > 0).sum() / len(trades))


def backtest_summary(results: pd.DataFrame) -> dict:
    """Comprehensive backtest summary from results with 'pnl' and 'position' columns."""
    pnl = results["pnl"]
    cumulative = pnl.cumsum()
    trade_returns = pnl[results["position"] != 0]

    daily_pnl = pnl.groupby(pnl.index.date).sum()

    return {
        "total_pnl": float(pnl.sum()),
        "daily_pnl_mean": float(daily_pnl.mean()),
        "daily_pnl_std": float(daily_pnl.std()),
        "sharpe_ratio": sharpe_ratio(results),
        "max_drawdown": max_drawdown(cumulative),
        "win_rate_pct": win_rate(trade_returns),
        "n_trades": int((results["position"] != 0).sum()),
        "n_trading_days": len(daily_pnl),
        "avg_win": float(trade_returns[trade_returns > 0].mean())
        if (trade_returns > 0).any()
        else 0,
        "avg_loss": float(trade_returns[trade_returns < 0].mean())
        if (trade_returns < 0).any()
        else 0,
    }
