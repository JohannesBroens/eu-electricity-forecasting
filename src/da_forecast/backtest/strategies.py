"""Trading strategies for backtesting.

Strategies convert price forecasts into positions (buy/sell/hold).
"""
import pandas as pd
import numpy as np


class RankSpreadStrategy:
    """Rank-based intra-day spread strategy. No reference price needed.

    Each day, the model ranks 24 hours by predicted price. The strategy
    buys the cheapest N hours and sells the most expensive N hours.
    P&L comes from the actual price spread between sell and buy hours.

    This is the most honest backtest: profit depends entirely on whether
    the model correctly identifies which hours are cheap vs expensive.
    No arbitrary baseline, no made-up reference price.

    Args:
        n_long: Number of cheapest hours to buy per day.
        n_short: Number of most expensive hours to sell per day.
        position_mwh: Size of each position in MWh.
        transaction_cost_eur_mwh: Cost per MWh traded.
    """

    def __init__(
        self,
        n_long: int = 4,
        n_short: int = 4,
        position_mwh: float = 1.0,
        transaction_cost_eur_mwh: float = 0.04,
    ):
        self.n_long = n_long
        self.n_short = n_short
        self.position_mwh = position_mwh
        self.transaction_cost_eur_mwh = transaction_cost_eur_mwh

    def generate_signals(self, predictions: pd.Series) -> pd.Series:
        """Generate positions from predictions alone. No baseline needed."""
        signals = pd.Series(0.0, index=predictions.index)
        dates = predictions.index.normalize()

        for day in dates.unique():
            day_mask = dates == day
            day_preds = predictions[day_mask]

            if len(day_preds) < self.n_long + self.n_short:
                continue

            # Rank hours by predicted price
            ranked = day_preds.rank()

            # Buy cheapest N hours (long), sell most expensive N hours (short)
            buy_hours = ranked.nsmallest(self.n_long).index
            sell_hours = ranked.nlargest(self.n_short).index

            signals.loc[buy_hours] = self.position_mwh
            signals.loc[sell_hours] = -self.position_mwh

        return signals

    def compute_pnl(self, predictions: pd.Series, actuals: pd.Series) -> pd.Series:
        """Compute P&L directly from positions and actual clearing prices.

        For long positions: profit when actual price is below daily mean
            (bought cheap, will sell at average or better)
        For short positions: profit when actual price is above daily mean
            (sold expensive, will buy back at average or lower)

        Long (buy) profits when actual < daily mean (bought cheap).
        Short (sell) profits when actual > daily mean (sold expensive).

        P&L per hour = -position * (actual_price - daily_mean_actual)
        """
        signals = self.generate_signals(predictions)

        # Use daily mean of actual prices as the settlement reference.
        # This is NOT a baseline prediction -- it's the arithmetic fact of
        # what the average price was that day.
        daily_mean = actuals.groupby(actuals.index.normalize()).transform("mean")

        # Long (buy cheap): profit = daily_mean - actual (positive when actual < mean)
        # Short (sell expensive): profit = actual - daily_mean (positive when actual > mean)
        pnl = -signals * (actuals - daily_mean)

        # Transaction costs
        if self.transaction_cost_eur_mwh > 0:
            pnl -= signals.abs() * self.transaction_cost_eur_mwh

        return pnl


class ThresholdStrategy:
    """Legacy: trade when predicted price diverges from baseline.

    DEPRECATED: This strategy requires an arbitrary reference price
    (baseline) which makes the backtest results unreliable. Use
    RankSpreadStrategy instead for honest evaluation.
    """

    def __init__(
        self,
        threshold_eur: float = 5.0,
        position_mwh: float = 1.0,
        transaction_cost_eur_mwh: float = 0.0,
        max_daily_trades: int | None = None,
    ):
        self.threshold_eur = threshold_eur
        self.position_mwh = position_mwh
        self.transaction_cost_eur_mwh = transaction_cost_eur_mwh
        self.max_daily_trades = max_daily_trades

    def generate_signals(
        self, predictions: pd.Series, baseline: pd.Series
    ) -> pd.Series:
        diff = predictions - baseline
        signals = pd.Series(0.0, index=predictions.index)
        signals[diff > self.threshold_eur] = self.position_mwh
        signals[diff < -self.threshold_eur] = -self.position_mwh

        if self.max_daily_trades is not None:
            dates = signals.index.normalize()
            for day in dates.unique():
                day_mask = dates == day
                day_signals = signals[day_mask]
                active = day_signals[day_signals != 0]
                if len(active) > self.max_daily_trades:
                    day_diff = diff[day_mask].abs()
                    ranked = day_diff[day_signals != 0].nlargest(self.max_daily_trades)
                    drop_mask = day_mask & (signals != 0) & (~signals.index.isin(ranked.index))
                    signals[drop_mask] = 0.0

        return signals
