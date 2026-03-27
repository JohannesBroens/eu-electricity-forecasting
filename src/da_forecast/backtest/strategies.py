"""Trading strategies for backtesting.

Strategies convert price forecasts into positions (buy/sell/hold).
"""
import pandas as pd
import numpy as np


class ThresholdStrategy:
    """Trade when predicted price diverges from baseline by more than threshold.

    Args:
        threshold_eur: Minimum predicted advantage to trigger a trade.
        position_mwh: Size of each trade in MWh.
        transaction_cost_eur_mwh: Cost per MWh traded (Nord Pool ~0.04 EUR/MWh).
        max_daily_trades: Maximum hourly trades per calendar day (capital constraint).
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


class SpreadStrategy:
    def __init__(
        self,
        threshold_eur: float = 5.0,
        lookback_days: int = 30,
        position_mwh: float = 1.0,
    ):
        self.threshold_eur = threshold_eur
        self.lookback_days = lookback_days
        self.position_mwh = position_mwh

    def generate_signals(
        self, predicted_spread: pd.Series, historical_spread: pd.Series
    ) -> pd.Series:
        rolling_avg = historical_spread.rolling(
            window=self.lookback_days * 24, min_periods=24
        ).mean()
        deviation = predicted_spread - rolling_avg.reindex(predicted_spread.index)
        signals = pd.Series(0.0, index=predicted_spread.index)
        signals[deviation > self.threshold_eur] = self.position_mwh
        signals[deviation < -self.threshold_eur] = -self.position_mwh
        return signals
