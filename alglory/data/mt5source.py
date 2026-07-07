"""MetaTrader 5 adapter.

MetaTrader5 is a Windows-only optional dependency, so it is imported lazily
and every capability degrades to an explicit, message-carrying failure when
the package or terminal is unavailable — never an ImportError at module load.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_TF_ATTR = {"M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}


class MT5NotConnectedError(Exception):
    pass


@dataclass(frozen=True)
class ConnStatus:
    ok: bool
    message: str
    account: str | None = None


class MT5Source:
    def __init__(self) -> None:
        self._mt5 = None
        self._connected = False

    def available(self) -> bool:
        return importlib.util.find_spec("MetaTrader5") is not None

    def connect(self) -> ConnStatus:
        if not self.available():
            return ConnStatus(
                ok=False,
                message=(
                    "The MetaTrader5 Python package is not installed. "
                    "Install with: pip install alglory[mt5] (Windows only)."
                ),
            )
        if self._mt5 is None:
            import MetaTrader5 as mt5  # noqa: N813

            self._mt5 = mt5
        if not self._mt5.initialize():
            code, desc = self._mt5.last_error()
            return ConnStatus(
                ok=False,
                message=f"Could not attach to a running MT5 terminal ({code}: {desc}). "
                "Start MetaTrader 5 and log into an account, then retry.",
            )
        self._connected = True
        info = self._mt5.account_info()
        account = f"{info.login}@{info.server}" if info else None
        return ConnStatus(ok=True, message="Connected to MetaTrader 5.", account=account)

    def _require_connection(self) -> None:
        if not self._connected or self._mt5 is None:
            raise MT5NotConnectedError(
                "MT5 is not connected. Call connect() first (requires a running, "
                "logged-in MetaTrader 5 terminal)."
            )

    def fetch_bars(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        self._require_connection()
        if timeframe not in _TF_ATTR:
            raise ValueError(
                f"Unknown timeframe {timeframe!r}; expected one of {sorted(_TF_ATTR)}"
            )
        tf = getattr(self._mt5, _TF_ATTR[timeframe])
        rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            raise MT5NotConnectedError(
                f"MT5 returned no history for {symbol} {timeframe}. "
                "Open a chart for the symbol in the terminal to trigger a download."
            )
        info = self._mt5.symbol_info(symbol)
        point = info.point if info else 0.00001
        df = pd.DataFrame(np.asarray(rates))
        return pd.DataFrame(
            {
                "time": pd.to_datetime(df["time"], unit="s"),
                "open": df["open"].astype(np.float64),
                "high": df["high"].astype(np.float64),
                "low": df["low"].astype(np.float64),
                "close": df["close"].astype(np.float64),
                "volume": df["tick_volume"].astype(np.float64),
                "spread": (df["spread"] * point).astype(np.float64),
            }
        )

    def symbols(self) -> list[str]:
        self._require_connection()
        return [s.name for s in (self._mt5.symbols_get() or []) if s.visible]

    def symbol_spread(self, symbol: str) -> float:
        self._require_connection()
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol {symbol!r} not found in the MT5 terminal.")
        return float(info.spread * info.point)

    def experts_dir(self) -> Path | None:
        if not self._connected or self._mt5 is None:
            return None
        info = self._mt5.terminal_info()
        if info is None:
            return None
        path = Path(info.data_path) / "MQL5" / "Experts" / "Alglory"
        path.mkdir(parents=True, exist_ok=True)
        return path
