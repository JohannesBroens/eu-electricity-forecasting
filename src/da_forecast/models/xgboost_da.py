"""XGBoost day-ahead price forecaster.

Two modes:
1. Single model: one XGBoost model predicts all 24 hours.
2. Per-hour models: 24 separate models, one per delivery hour.
"""
import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from da_forecast.config import DEFAULT_TRAINING_WINDOW_DAYS, DEFAULT_XGBOOST_PARAMS

logger = logging.getLogger(__name__)


class DayAheadForecaster:
    def __init__(
        self,
        per_hour: bool = False,
        params: dict | None = None,
        training_window_days: int = DEFAULT_TRAINING_WINDOW_DAYS,
    ):
        self.per_hour = per_hour
        self.params = params or DEFAULT_XGBOOST_PARAMS.copy()
        self.training_window_days = training_window_days
        self.models = None
        self.feature_columns: list[str] = []

    def _get_feature_cols(self, df: pd.DataFrame, target_col: str) -> list[str]:
        return [c for c in df.columns if c != target_col]

    def train(self, df: pd.DataFrame, target_col: str = "price_eur_mwh") -> None:
        self.feature_columns = self._get_feature_cols(df, target_col)
        X = df[self.feature_columns].values
        y = df[target_col].values
        if self.per_hour:
            self.models = {}
            hours = (
                df.index.hour
                if hasattr(df.index, "hour")
                else pd.DatetimeIndex(df.index).hour
            )
            for h in range(24):
                mask = hours == h
                if mask.sum() < 10:
                    logger.warning(f"Hour {h}: only {mask.sum()} samples, skipping")
                    continue
                model = xgb.XGBRegressor(**self.params)
                model.fit(X[mask], y[mask])
                self.models[h] = model
        else:
            self.models = xgb.XGBRegressor(**self.params)
            self.models.fit(X, y)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        X = df[self.feature_columns].values
        predictions = np.zeros(len(df))
        if self.per_hour and isinstance(self.models, dict):
            hours = (
                df.index.hour
                if hasattr(df.index, "hour")
                else pd.DatetimeIndex(df.index).hour
            )
            for h in range(24):
                mask = hours == h
                if mask.sum() > 0 and h in self.models:
                    predictions[mask] = self.models[h].predict(X[mask])
        else:
            predictions = self.models.predict(X)
        return pd.Series(predictions, index=df.index, name="predicted_price")

    def feature_importance(self) -> pd.DataFrame:
        if self.per_hour and isinstance(self.models, dict):
            importances = []
            for h, model in self.models.items():
                imp = pd.Series(
                    model.feature_importances_,
                    index=self.feature_columns,
                    name=f"hour_{h}",
                )
                importances.append(imp)
            avg = pd.concat(importances, axis=1).mean(axis=1)
            return pd.DataFrame({"importance": avg}).sort_values(
                "importance", ascending=False
            )
        else:
            imp = self.models.feature_importances_
            return pd.DataFrame(
                {"importance": imp}, index=self.feature_columns
            ).sort_values("importance", ascending=False)
