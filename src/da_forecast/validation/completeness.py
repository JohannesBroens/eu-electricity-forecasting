"""Data completeness validation.

Detects missing timestamps in hourly electricity market data, accounting
for DST transitions (23h and 25h days).
"""
import pandas as pd
from da_forecast.validation.timezone import expected_hours_in_day

def find_gaps(df: pd.DataFrame, expected_freq: str = "h") -> list[dict]:
    if df.empty:
        return []
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq, tz=df.index.tz)
    missing = full_idx.difference(df.index)
    return [{"timestamp": ts} for ts in missing]

def daily_completeness_report(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "expected", "actual", "missing", "complete"])
    daily_counts = df.groupby(df.index.date).size()
    rows = []
    for day, actual_count in daily_counts.items():
        day_ts = pd.Timestamp(day)
        expected = expected_hours_in_day(day_ts)
        rows.append({"date": day_ts, "expected": expected, "actual": actual_count,
                      "missing": expected - actual_count, "complete": actual_count >= expected})
    return pd.DataFrame(rows)
