"""Fundamental features for electricity price forecasting.

Residual load = total load - renewable generation. This determines how much
thermal generation (gas/coal) must run, setting the marginal price via
the merit order.
"""
import pandas as pd

def compute_fundamental_features(load_forecast: pd.DataFrame, wind_forecast: pd.DataFrame, solar_forecast: pd.DataFrame) -> pd.DataFrame:
    combined = pd.DataFrame(index=load_forecast.index)
    combined["load_mw"] = load_forecast["load_mw"]
    combined["wind_mw"] = wind_forecast["wind_mw"].reindex(combined.index)
    combined["solar_mw"] = solar_forecast["solar_mw"].reindex(combined.index)
    combined["residual_load_mw"] = combined["load_mw"] - combined["wind_mw"] - combined["solar_mw"]
    return combined
