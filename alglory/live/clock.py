"""Bar clock: hand the executor the latest *closed* bars.

MT5's copy_rates_from_pos(..., 0, n) includes the still-forming current bar as
the last row. The executor must act on the last *closed* bar (that's what the
backtester and the EA do), so the clock drops the forming bar. New-bar detection
lives in the executor manager, which compares the last closed timestamp.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class BarClock(Protocol):
    def latest_closed_bars(self, symbol: str, timeframe: str) -> pd.DataFrame | None: ...


class MT5BarClock:
    def __init__(self, source, count: int = 300) -> None:
        self._source = source
        self._count = count

    def latest_closed_bars(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        try:
            df = self._source.fetch_bars(symbol, timeframe, self._count + 1)
        except Exception:
            return None
        if df is None or len(df) < 2:
            return None
        # drop the still-forming last bar; the new last row is the last closed bar
        return df.iloc[:-1].reset_index(drop=True)
