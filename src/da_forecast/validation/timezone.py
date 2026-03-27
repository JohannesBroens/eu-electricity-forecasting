"""Timezone utilities for electricity market data.

All internal data is stored in UTC. The day-ahead auction and gate closure
times are in CET/CEST. DST transitions create 23-hour (spring) and 25-hour
(fall) days.
"""
import pandas as pd
import pytz

CET = pytz.timezone("Europe/Copenhagen")

def to_utc(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tz is None:
        raise ValueError(f"Timestamp {ts} is timezone-naive. Cannot convert to UTC safely.")
    return ts.tz_convert(pytz.UTC)

def to_cet(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tz is None:
        raise ValueError(f"Timestamp {ts} is timezone-naive. Cannot convert to CET safely.")
    return ts.tz_convert("Europe/Copenhagen")

def find_dst_transitions(year: int) -> list[dict]:
    transitions = []
    for month, expected_type in [(3, "spring_forward"), (10, "fall_back")]:
        last_day = pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(0)
        while last_day.weekday() != 6:
            last_day -= pd.Timedelta(days=1)
        hours = 23 if expected_type == "spring_forward" else 25
        transitions.append({"date": last_day, "type": expected_type, "hours": hours})
    return sorted(transitions, key=lambda x: x["date"])

def expected_hours_in_day(day: pd.Timestamp) -> int:
    year = day.year
    transitions = find_dst_transitions(year)
    day_date = day.normalize()
    for t in transitions:
        if t["date"].normalize() == day_date:
            return t["hours"]
    return 24

def validate_timezone_aware(df: pd.DataFrame) -> None:
    if df.index.tz is None:
        raise ValueError(
            "DataFrame index is timezone-naive. All data must have explicit timezone. "
            "Use df.tz_localize('UTC') or ensure the data source provides tz-aware timestamps."
        )
