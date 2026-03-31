"""Model performance tracking and drift detection.

Tracks daily prediction error (MAE) per zone over time and detects drift
using a rolling-window approach: if the 7-day rolling MAE exceeds 2x the
30-day rolling MAE, the zone is flagged as drifting.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HISTORY_PATH = Path(__file__).resolve().parents[3] / "output" / "model_performance_history.csv"

COLUMNS = ["timestamp", "zone", "mae"]


def _load_history() -> pd.DataFrame:
    """Load the persisted performance history, or return an empty frame."""
    if HISTORY_PATH.exists() and HISTORY_PATH.stat().st_size > 0:
        df = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
        return df[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def _save_history(df: pd.DataFrame) -> None:
    """Persist the full history to CSV (creates parent dirs if needed)."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(HISTORY_PATH, index=False)


def _append_observation(df: pd.DataFrame, zone: str, mae: float) -> pd.DataFrame:
    """Append a single observation and return the updated frame."""
    new_row = pd.DataFrame(
        [{"timestamp": datetime.now(timezone.utc).isoformat(), "zone": zone, "mae": mae}]
    )
    return pd.concat([df, new_row], ignore_index=True)


def check_drift(zone: str, current_mae: float) -> dict:
    """Check whether a zone's model performance is drifting.

    Loads the historical performance CSV, appends the new observation,
    computes rolling MAE windows, and returns a status dict.

    Parameters
    ----------
    zone : str
        Bidding-zone identifier (e.g. ``"DK_1"``).
    current_mae : float
        MAE from the most recent model evaluation.

    Returns
    -------
    dict
        Keys: zone, current_mae, rolling_7d_mae, rolling_30d_mae,
        is_drifting, message.
    """
    history = _load_history()
    history = _append_observation(history, zone, current_mae)
    _save_history(history)

    zone_history = history[history["zone"] == zone]["mae"].astype(float)

    n = len(zone_history)
    rolling_7d = zone_history.tail(7).mean() if n >= 1 else current_mae
    rolling_30d = zone_history.tail(30).mean() if n >= 1 else current_mae

    is_drifting = (n >= 7) and (rolling_7d > 2 * rolling_30d)

    if is_drifting:
        message = (
            f"DRIFT DETECTED for {zone}: 7-day MAE ({rolling_7d:.2f}) "
            f"exceeds 2x 30-day MAE ({rolling_30d:.2f})"
        )
    elif n < 7:
        message = f"{zone}: not enough history for drift detection ({n} observations)"
    else:
        message = f"{zone}: model performance stable (7d={rolling_7d:.2f}, 30d={rolling_30d:.2f})"

    return {
        "zone": zone,
        "current_mae": current_mae,
        "rolling_7d_mae": round(rolling_7d, 4),
        "rolling_30d_mae": round(rolling_30d, 4),
        "is_drifting": is_drifting,
        "message": message,
    }
