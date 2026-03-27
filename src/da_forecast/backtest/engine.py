"""Walk-forward backtest engine."""
import logging
import pandas as pd
import numpy as np
from da_forecast.models.xgboost_da import DayAheadForecaster
from da_forecast.backtest.strategies import ThresholdStrategy
from da_forecast.models.evaluation import naive_baseline

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(
        self,
        strategy: ThresholdStrategy | None = None,
        training_window_days: int = 56,
        per_hour_models: bool = False,
    ):
        self.strategy = strategy or ThresholdStrategy()
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
            baseline = naive_baseline(feature_matrix[target_col]).reindex(
                test_data.index
            )
            signals = self.strategy.generate_signals(
                predictions, baseline.fillna(predictions)
            )
            # P&L: position * (actual - baseline). Measures directional accuracy.
            pnl = signals * (actuals - baseline.fillna(actuals))

            if hasattr(self.strategy, 'transaction_cost_eur_mwh'):
                cost = self.strategy.transaction_cost_eur_mwh
                if cost > 0:
                    pnl -= signals.abs() * cost

            day_results = pd.DataFrame(
                {
                    "predicted_price": predictions,
                    "actual_price": actuals,
                    "baseline_price": baseline,
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
                    "baseline_price",
                    "position",
                    "pnl",
                ]
            )
        return pd.concat(all_results).sort_index()
