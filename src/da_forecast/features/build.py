"""Feature matrix builder.

Assembles all feature families into a single DataFrame ready for model training.
The target variable (price_eur_mwh) is included for convenience.
"""
import pandas as pd
from da_forecast.features.lags import compute_lag_features
from da_forecast.features.calendar import compute_calendar_features
from da_forecast.features.fundamental import compute_fundamental_features
from da_forecast.features.weather import compute_weather_features

def build_feature_matrix(
    prices: pd.DataFrame,
    load_forecast: pd.DataFrame,
    wind_solar_forecast: pd.DataFrame,
    generation_actuals: pd.DataFrame | None = None,
    cross_border_flows: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    result = compute_lag_features(prices)
    cal = compute_calendar_features(prices)
    for col in ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend", "is_holiday"]:
        if col in cal.columns:
            result[col] = cal[col]

    wind_total = wind_solar_forecast.copy()
    wind_cols = [c for c in wind_total.columns if "wind" in c.lower()]
    wind_total["wind_mw"] = wind_total[wind_cols].sum(axis=1)
    solar_col = [c for c in wind_total.columns if "solar" in c.lower()]
    solar_df = pd.DataFrame({"solar_mw": wind_total[solar_col[0]] if solar_col else 0}, index=wind_total.index)
    wind_df = pd.DataFrame({"wind_mw": wind_total["wind_mw"]}, index=wind_total.index)
    fund = compute_fundamental_features(load_forecast, wind_df, solar_df)
    result["residual_load_mw"] = fund["residual_load_mw"].reindex(result.index)
    result["load_mw"] = fund["load_mw"].reindex(result.index)

    weather = compute_weather_features(wind_solar_forecast, generation_actuals)
    for col in ["wind_total_mw", "wind_capacity_factor", "wind_forecast_error_lag1"]:
        if col in weather.columns:
            series = weather[col].reindex(result.index)
            if not series.isna().all():
                result[col] = series
    if solar_col:
        result["solar_mw"] = wind_solar_forecast[solar_col[0]].reindex(result.index)

    if cross_border_flows:
        from da_forecast.config import INTERCONNECTORS
        for label, flow_df in cross_border_flows.items():
            if "flow_mw" in flow_df.columns:
                result[f"flow_{label}_mw"] = flow_df["flow_mw"].reindex(result.index)
                for ic in INTERCONNECTORS:
                    if f"{ic[0]}_{ic[1]}" == label or label == f"{ic[0]}_{ic[1]}":
                        result[f"util_{label}_pct"] = (flow_df["flow_mw"].abs() / ic[2] * 100).reindex(result.index)
    return result
