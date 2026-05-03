"""Walk-forward backtest engine.

Uses rank-based spread strategy by default: each day, buy the hours
the model predicts cheapest, sell the hours predicted most expensive.
P&L comes from actual price spreads -- no arbitrary reference price.
"""
import logging
import pandas as pd
import numpy as np
from da_forecast.models.xgboost_da import DayAheadForecaster
from da_forecast.backtest.strategies import RankSpreadStrategy

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        strategy: RankSpreadStrategy | None = None,
        training_window_days: int = 56,
        per_hour_models: bool = False,
    ):
        self.strategy = strategy or RankSpreadStrategy()
        self.training_window_days = training_window_days
        self.per_hour_models = per_hour_models

    def run(
        self, feature_matrix: pd.DataFrame, target_col: str = "price_eur_mwh"
    ) -> pd.DataFrame:
        feature_cols = [c for c in feature_matrix.columns if c != target_col]
        dates = feature_matrix.index.normalize().unique().sort_values()
        min_train_date = dates[0] + pd.Timedelta(days=self.training_window_days)
        test_dates = dates[dates >= min_train_date]
        all_results = []

        for pred_date in test_dates:
            train_end = pred_date
            train_start = train_end - pd.Timedelta(days=self.training_window_days)
            train_mask = (feature_matrix.index >= train_start) & (
                feature_matrix.index < train_end
            )
            train_data = feature_matrix.loc[train_mask].dropna()
            test_mask = feature_matrix.index.normalize() == pred_date
            test_data = feature_matrix.loc[test_mask]

            if train_data.empty or test_data.empty or len(train_data) < 48:
                continue

            forecaster = DayAheadForecaster(per_hour=self.per_hour_models)
            forecaster.train(train_data, target_col=target_col)
            X_test = test_data[feature_cols]
            predictions = forecaster.predict(X_test)
            actuals = test_data[target_col]

            # Rank-based: positions come from predictions alone
            signals = self.strategy.generate_signals(predictions)
            pnl = self.strategy.compute_pnl(predictions, actuals)

            day_results = pd.DataFrame(
                {
                    "predicted_price": predictions,
                    "actual_price": actuals,
                    "position": signals,
                    "pnl": pnl,
                }
            )
            all_results.append(day_results)

        if not all_results:
            return pd.DataFrame(
                columns=[
                    "predicted_price",
                    "actual_price",
                    "position",
                    "pnl",
                ]
            )
        return pd.concat(all_results).sort_index()
