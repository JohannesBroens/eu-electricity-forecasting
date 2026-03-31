"""Parquet-based cache for raw data.

One file per (source, zone, datatype) combination. Supports incremental
fetching by merging new data into existing cache files.
"""
from pathlib import Path
from typing import Optional
import pandas as pd

class ParquetCache:
    """Read/write/merge Parquet cache files for raw data."""
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def _path(self, source: str, zone: str, datatype: str) -> Path:
        return self.base_dir / source / zone / f"{datatype}.parquet"

    def save(self, source: str, zone: str, datatype: str, df: pd.DataFrame) -> None:
        path = self._path(source, zone, datatype)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def load(self, source: str, zone: str, datatype: str) -> Optional[pd.DataFrame]:
        path = self._path(source, zone, datatype)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        # Restore inferred frequency metadata lost during Parquet round-trip
        if isinstance(df.index, pd.DatetimeIndex) and df.index.freq is None and len(df) >= 3:
            inferred = pd.infer_freq(df.index)
            if inferred:
                df.index.freq = pd.tseries.frequencies.to_offset(inferred)
        return df

    def merge(self, source: str, zone: str, datatype: str, new_df: pd.DataFrame) -> None:
        existing = self.load(source, zone, datatype)
        if existing is None:
            self.save(source, zone, datatype, new_df)
            return
        overlap_mask = existing.index.isin(new_df.index)
        combined = pd.concat([existing[~overlap_mask], new_df]).sort_index()
        self.save(source, zone, datatype, combined)

    def get_cached_range(self, source: str, zone: str, datatype: str) -> Optional[tuple[pd.Timestamp, pd.Timestamp]]:
        df = self.load(source, zone, datatype)
        if df is None or df.empty:
            return None
        return df.index.min(), df.index.max()
