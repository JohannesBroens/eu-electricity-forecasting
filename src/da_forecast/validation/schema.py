"""Schema validation for pipeline DataFrames.

Validates column names, dtypes, and index properties for electricity
market data loaded by the pipeline.
"""
import pandas as pd


def validate_dataframe(
    df: pd.DataFrame | None,
    expected_columns: dict[str, str],
    name: str,
) -> list[str]:
    """Generic DataFrame schema validator.

    Parameters
    ----------
    df : DataFrame or None
        The data to validate.
    expected_columns : dict
        Mapping of column name -> expected dtype group. Accepted groups:
        "float", "int", "numeric" (float or int), "any".
    name : str
        Human-readable dataset name used in error messages.

    Returns
    -------
    list[str]
        Error messages. Empty list means the DataFrame is valid.
    """
    errors: list[str] = []

    if df is None:
        errors.append(f"{name}: DataFrame is None")
        return errors

    if not isinstance(df, pd.DataFrame):
        errors.append(f"{name}: expected pandas DataFrame, got {type(df).__name__}")
        return errors

    if df.empty:
        errors.append(f"{name}: DataFrame is empty")
        return errors

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"{name}: index is {type(df.index).__name__}, expected DatetimeIndex")
    elif df.index.tz is None:
        errors.append(f"{name}: DatetimeIndex is not timezone-aware")

    for col, dtype_group in expected_columns.items():
        if col not in df.columns:
            errors.append(f"{name}: missing column '{col}'")
            continue

        if dtype_group == "any":
            continue

        if dtype_group in ("float", "numeric") and not pd.api.types.is_float_dtype(df[col]):
            if dtype_group == "numeric" and pd.api.types.is_integer_dtype(df[col]):
                continue
            errors.append(
                f"{name}: column '{col}' has dtype '{df[col].dtype}', expected {dtype_group}"
            )
        elif dtype_group == "int" and not pd.api.types.is_integer_dtype(df[col]):
            errors.append(
                f"{name}: column '{col}' has dtype '{df[col].dtype}', expected int"
            )

    return errors


def validate_prices(df: pd.DataFrame | None) -> list[str]:
    """Validate the day-ahead prices DataFrame.

    Expects a timezone-aware DatetimeIndex and a float column
    ``price_eur_mwh``.
    """
    return validate_dataframe(
        df,
        expected_columns={"price_eur_mwh": "float"},
        name="prices",
    )


def validate_wind_solar(df: pd.DataFrame | None) -> list[str]:
    """Validate the wind/solar generation DataFrame.

    Expects a timezone-aware DatetimeIndex and at least one of
    ``wind_onshore_mw``, ``wind_offshore_mw``, ``solar_mw`` (all float).
    """
    errors: list[str] = []

    if df is None:
        errors.append("wind_solar: DataFrame is None")
        return errors

    if not isinstance(df, pd.DataFrame):
        errors.append(f"wind_solar: expected pandas DataFrame, got {type(df).__name__}")
        return errors

    if df.empty:
        errors.append("wind_solar: DataFrame is empty")
        return errors

    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(
            f"wind_solar: index is {type(df.index).__name__}, expected DatetimeIndex"
        )
    elif df.index.tz is None:
        errors.append("wind_solar: DatetimeIndex is not timezone-aware")

    renewable_cols = {"wind_onshore_mw", "wind_offshore_mw", "solar_mw"}
    present = renewable_cols & set(df.columns)

    if not present:
        errors.append(
            "wind_solar: must have at least one of "
            "'wind_onshore_mw', 'wind_offshore_mw', 'solar_mw'"
        )
    else:
        for col in present:
            if not pd.api.types.is_float_dtype(df[col]):
                errors.append(
                    f"wind_solar: column '{col}' has dtype '{df[col].dtype}', expected float"
                )

    return errors


def validate_load(df: pd.DataFrame | None) -> list[str]:
    """Validate the electricity load DataFrame.

    Expects a timezone-aware DatetimeIndex and a float column ``load_mw``.
    """
    return validate_dataframe(
        df,
        expected_columns={"load_mw": "float"},
        name="load",
    )
