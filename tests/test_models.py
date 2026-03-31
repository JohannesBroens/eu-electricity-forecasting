"""Tests for DayAheadForecaster and evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from da_forecast.models.xgboost_da import DayAheadForecaster
from da_forecast.models.evaluation import mae, rmse, smape, naive_baseline, evaluation_report


TZ = "Europe/Copenhagen"


# --- evaluation metrics ---

class TestMAE:
    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert mae(actual, actual) == 0.0

    def test_known_value(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([12.0, 18.0, 33.0])
        # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7, mean = 7/3
        assert mae(actual, predicted) == pytest.approx(7 / 3)

    def test_symmetric(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert mae(a, b) == mae(b, a)


class TestRMSE:
    def test_perfect_prediction(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert rmse(actual, actual) == 0.0

    def test_known_value(self):
        actual = np.array([0.0, 0.0])
        predicted = np.array([3.0, 4.0])
        # sqrt((9 + 16) / 2) = sqrt(12.5)
        assert rmse(actual, predicted) == pytest.approx(np.sqrt(12.5))

    def test_rmse_geq_mae(self):
        a = np.array([1.0, 2.0, 3.0, 100.0])
        b = np.array([1.5, 2.5, 3.5, 50.0])
        assert rmse(a, b) >= mae(a, b)


class TestSMAPE:
    def test_perfect_prediction(self):
        actual = np.array([10.0, 20.0])
        assert smape(actual, actual) == pytest.approx(0.0)

    def test_both_zero(self):
        actual = np.array([0.0, 0.0])
        predicted = np.array([0.0, 0.0])
        assert smape(actual, predicted) == 0.0

    def test_bounded_by_200(self):
        actual = np.array([100.0, 0.0])
        predicted = np.array([0.0, 100.0])
        result = smape(actual, predicted)
        assert result <= 200.0

    def test_symmetric(self):
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([12.0, 18.0, 35.0])
        assert smape(a, b) == pytest.approx(smape(b, a))


class TestNaiveBaseline:
    def test_shifts_by_168_hours(self):
        idx = pd.date_range("2025-01-01", periods=336, freq="h", tz=TZ)
        prices = pd.Series(np.arange(336, dtype=float), index=idx)
        baseline = naive_baseline(prices)
        # Hour 168 should equal hour 0
        assert baseline.iloc[168] == pytest.approx(0.0)
        assert baseline.iloc[169] == pytest.approx(1.0)

    def test_first_168_hours_are_nan(self):
        idx = pd.date_range("2025-01-01", periods=336, freq="h", tz=TZ)
        prices = pd.Series(np.ones(336), index=idx)
        baseline = naive_baseline(prices)
        assert baseline.iloc[:168].isna().all()


class TestEvaluationReport:
    def test_report_keys(self):
        actual = np.array([10.0, 20.0, 30.0])
        predicted = np.array([11.0, 19.0, 32.0])
        report = evaluation_report(actual, predicted)
        assert "mae" in report
        assert "rmse" in report
        assert "smape" in report
        assert "n_samples" in report
        assert report["n_samples"] == 3

    def test_per_hour_report(self):
        actual = np.array([10.0, 20.0, 30.0, 40.0])
        predicted = np.array([11.0, 19.0, 32.0, 38.0])
        hours = np.array([0, 1, 0, 1])
        report = evaluation_report(actual, predicted, hours=hours)
        assert "per_hour" in report
        assert 0 in report["per_hour"]
        assert 1 in report["per_hour"]


# --- DayAheadForecaster ---

class TestDayAheadForecaster:
    @pytest.fixture
    def training_data(self):
        """Generate enough data for training (14 days = 336 hours)."""
        rng = np.random.default_rng(42)
        idx = pd.date_range("2025-01-01", periods=336, freq="h", tz=TZ)
        df = pd.DataFrame({
            "price_eur_mwh": 40 + 10 * np.sin(2 * np.pi * np.arange(336) / 24) + rng.normal(0, 3, 336),
            "feature_a": rng.normal(0, 1, 336),
            "feature_b": rng.normal(0, 1, 336),
        }, index=idx)
        return df

    def test_train_and_predict(self, training_data):
        model = DayAheadForecaster(
            params={"objective": "reg:squarederror", "n_estimators": 10, "max_depth": 3, "random_state": 42}
        )
        model.train(training_data, target_col="price_eur_mwh")
        predictions = model.predict(training_data)
        assert isinstance(predictions, pd.Series)
        assert len(predictions) == len(training_data)
        assert predictions.name == "predicted_price"

    def test_feature_columns_stored(self, training_data):
        model = DayAheadForecaster(
            params={"objective": "reg:squarederror", "n_estimators": 10, "random_state": 42}
        )
        model.train(training_data, target_col="price_eur_mwh")
        assert "feature_a" in model.feature_columns
        assert "feature_b" in model.feature_columns
        assert "price_eur_mwh" not in model.feature_columns

    def test_feature_importance(self, training_data):
        model = DayAheadForecaster(
            params={"objective": "reg:squarederror", "n_estimators": 10, "random_state": 42}
        )
        model.train(training_data, target_col="price_eur_mwh")
        imp = model.feature_importance()
        assert isinstance(imp, pd.DataFrame)
        assert "importance" in imp.columns
        assert len(imp) == 2  # feature_a, feature_b

    def test_per_hour_model(self, training_data):
        model = DayAheadForecaster(
            per_hour=True,
            params={"objective": "reg:squarederror", "n_estimators": 10, "max_depth": 3, "random_state": 42},
        )
        model.train(training_data, target_col="price_eur_mwh")
        assert isinstance(model.models, dict)
        predictions = model.predict(training_data)
        assert len(predictions) == len(training_data)

    def test_per_hour_feature_importance(self, training_data):
        model = DayAheadForecaster(
            per_hour=True,
            params={"objective": "reg:squarederror", "n_estimators": 10, "random_state": 42},
        )
        model.train(training_data, target_col="price_eur_mwh")
        imp = model.feature_importance()
        assert isinstance(imp, pd.DataFrame)
        assert "importance" in imp.columns

    def test_predictions_are_reasonable(self, training_data):
        model = DayAheadForecaster(
            params={"objective": "reg:squarederror", "n_estimators": 50, "max_depth": 4, "random_state": 42}
        )
        model.train(training_data, target_col="price_eur_mwh")
        predictions = model.predict(training_data)
        # Training predictions should be roughly in the right range
        assert predictions.mean() == pytest.approx(training_data["price_eur_mwh"].mean(), abs=20)

    def test_default_params_used(self):
        model = DayAheadForecaster()
        assert model.params["objective"] == "reg:squarederror"
        assert model.params["max_depth"] == 6
