"""Parquet-backed bar cache and data sufficiency checks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


class InsufficientDataError(Exception):
    """Raised when available history cannot support the requested campaign."""


class BarCache:
    def __init__(self, data_dir: Path):
        self._dir = Path(data_dir) / "bars"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, timeframe: str) -> Path:
        return self._dir / f"{symbol}_{timeframe}.parquet"

    def save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        df.to_parquet(self._path(symbol, timeframe), index=False)

    def load(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._path(symbol, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def coverage(self, symbol: str, timeframe: str) -> tuple[datetime, datetime] | None:
        df = self.load(symbol, timeframe)
        if df is None or df.empty:
            return None
        return df["time"].iloc[0], df["time"].iloc[-1]


def check_sufficiency(df: pd.DataFrame | None, min_bars: int) -> None:
    if df is None or df.empty:
        raise InsufficientDataError(
            f"No price history available; at least {min_bars} bars are required."
        )
    if len(df) < min_bars:
        raise InsufficientDataError(
            f"Only {len(df)} bars of history available but at least "
            f"{min_bars} are required. Download more history or shorten the window."
        )
