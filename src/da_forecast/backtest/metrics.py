"""Backtest performance metrics.

Sharpe ratio uses daily aggregate P&L (not hourly) because consecutive
hourly returns are correlated -- using hourly with sqrt(8760) annualization
inflates Sharpe by ~5x.

IMPORTANT: All metrics here describe simulated backtest performance only.
They do NOT predict future returns. The backtest uses a simplistic reference
price and assumes perfect execution -- real-world results would differ
significantly. See docs/ for a full discussion of backtest limitations.
"""
import numpy as np
import pandas as pd


def sharpe_ratio(results: pd.DataFrame) -> float:
    """Annualized Sharpe ratio based on daily P&L.

    Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(365).
    Penalizes all volatility equally (up and down).
    """
    pnl = results["pnl"]
    daily_pnl = pnl.groupby(pnl.index.date).sum()
    if daily_pnl.std() == 0:
        return float("inf") if daily_pnl.mean() > 0 else 0.0
    return float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(365))


def sortino_ratio(results: pd.DataFrame) -> float:
    """Annualized Sortino ratio based on daily P&L.

    Like Sharpe but only penalizes downside (negative) volatility.
    More relevant for trading: big wins are good, only big losses matter.
    """
    pnl = results["pnl"]
    daily_pnl = pnl.groupby(pnl.index.date).sum()
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) == 0:
        return float("inf") if daily_pnl.mean() > 0 else 0.0
    downside_std = downside.std()
    if downside_std == 0:
        return float("inf") if daily_pnl.mean() > 0 else 0.0
    return float(daily_pnl.mean() / downside_std * np.sqrt(365))


def calmar_ratio(results: pd.DataFrame) -> float:
    """Calmar ratio: annualized return / max drawdown.

    Answers: how much worst-case pain per unit of annual return?
    Calmar 1.0 = your worst drawdown equals one year of returns.
    """
    pnl = results["pnl"]
    daily_pnl = pnl.groupby(pnl.index.date).sum()
    n_days = len(daily_pnl)
    if n_days == 0:
        return 0.0
    annual_pnl = daily_pnl.sum() * (365.0 / n_days)
    cumulative = pnl.cumsum()
    mdd = max_drawdown(cumulative)
    if mdd == 0:
        return float("inf") if annual_pnl > 0 else 0.0
    return float(annual_pnl / mdd)


def profit_factor(results: pd.DataFrame) -> float:
    """Gross winning P&L / gross losing P&L.

    More informative than win rate alone. A strategy with 90% win rate
    but profit factor < 1.0 still loses money (the 10% losses are huge).
    Profit factor > 1.0 = net positive. > 2.0 = strong.
    """
    pnl = results["pnl"]
    trades = pnl[pnl != 0]
    gross_wins = trades[trades > 0].sum()
    gross_losses = abs(trades[trades < 0].sum())
    if gross_losses == 0:
        return float("inf") if gross_wins > 0 else 0.0
    return float(gross_wins / gross_losses)


def max_drawdown(cumulative_pnl: pd.Series) -> float:
    """Maximum peak-to-trough drawdown in EUR."""
    peak = cumulative_pnl.cummax()
    drawdown = peak - cumulative_pnl
    return float(drawdown.max())


def win_rate(returns: pd.Series) -> float:
    """Percentage of profitable trades (non-zero returns only).

    NOTE: Win rate alone is meaningless. A 90% win rate with tiny wins
    and large losses has negative expected value. Always pair with
    profit_factor or avg_win/avg_loss ratio.
    """
    trades = returns[returns != 0]
    if len(trades) == 0:
        return 0.0
    return float(100 * (trades > 0).sum() / len(trades))


def backtest_summary(results: pd.DataFrame) -> dict:
    """Backtest summary from results with 'pnl' and 'position' columns.

    WARNING: These are simulated results, not real trading performance.
    The backtest uses a simplistic reference price and assumes perfect
    execution. Real-world performance would be significantly worse.
    """
    pnl = results["pnl"]
    cumulative = pnl.cumsum()
    trade_returns = pnl[results["position"] != 0]

    daily_pnl = pnl.groupby(pnl.index.date).sum()

    return {
        "total_pnl": float(pnl.sum()),
        "daily_pnl_mean": float(daily_pnl.mean()),
        "daily_pnl_std": float(daily_pnl.std()),
        "sharpe_ratio": sharpe_ratio(results),
        "sortino_ratio": sortino_ratio(results),
        "calmar_ratio": calmar_ratio(results),
        "profit_factor": profit_factor(results),
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
