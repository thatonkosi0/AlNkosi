"""MetaTrader 5 adapter.

MetaTrader5 is a Windows-only optional dependency, so it is imported lazily
and every capability degrades to an explicit, message-carrying failure when
the package or terminal is unavailable — never an ImportError at module load.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from alglory.analysis.sizing import SymbolSpec

_TF_ATTR = {"M15": "TIMEFRAME_M15", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4", "D1": "TIMEFRAME_D1"}


class MT5NotConnectedError(Exception):
    pass


@dataclass(frozen=True)
class ConnStatus:
    ok: bool
    message: str
    account: str | None = None


@dataclass(frozen=True)
class OrderResult:
    ok: bool
    retcode: int
    message: str
    ticket: int | None = None
    price: float = 0.0
    volume: float = 0.0


@dataclass(frozen=True)
class PositionInfo:
    ticket: int
    symbol: str
    direction: int  # +1 long, -1 short
    volume: float
    price_open: float
    sl: float
    tp: float
    magic: int
    profit: float


@dataclass(frozen=True)
class AccountInfo:
    login: int
    balance: float
    equity: float
    currency: str
    is_demo: bool


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

    # ---- trading (live executor) --------------------------------------

    def account(self) -> AccountInfo:
        self._require_connection()
        a = self._mt5.account_info()
        if a is None:
            raise MT5NotConnectedError("MT5 returned no account info.")
        is_demo = a.trade_mode == self._mt5.ACCOUNT_TRADE_MODE_DEMO
        return AccountInfo(int(a.login), float(a.balance), float(a.equity), a.currency, is_demo)

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        """Broker contract spec for ``symbol``, ready for sizing.lots_for_risk."""
        from alglory.analysis.sizing import SymbolSpec

        self._require_connection()
        info = self._mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol {symbol!r} not found in the MT5 terminal.")
        return SymbolSpec(
            tick_value=float(info.trade_tick_value),
            tick_size=float(info.trade_tick_size),
            min_lot=float(info.volume_min),
            lot_step=float(info.volume_step),
            max_lot=float(info.volume_max),
        )

    def positions(self, magic: int | None = None) -> list[PositionInfo]:
        self._require_connection()
        out: list[PositionInfo] = []
        for p in self._mt5.positions_get() or []:
            if magic is not None and p.magic != magic:
                continue
            direction = 1 if p.type == self._mt5.POSITION_TYPE_BUY else -1
            out.append(
                PositionInfo(
                    int(p.ticket), p.symbol, direction, float(p.volume),
                    float(p.price_open), float(p.sl), float(p.tp), int(p.magic), float(p.profit),
                )
            )
        return out

    def place_market(
        self,
        symbol: str,
        direction: int,
        lots: float,
        *,
        sl: float = 0.0,
        tp: float = 0.0,
        magic: int = 0,
        comment: str = "ALGLORY",
    ) -> OrderResult:
        self._require_connection()
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return OrderResult(False, -1, f"No tick available for {symbol!r}.")
        is_buy = direction == 1
        price = tick.ask if is_buy else tick.bid
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lots),
            "type": self._mt5.ORDER_TYPE_BUY if is_buy else self._mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "magic": int(magic),
            "comment": comment,
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
            "deviation": 20,
        }
        return self._interpret(self._mt5.order_send(request), price, float(lots))

    def modify_position(self, symbol: str, magic: int, *, sl: float, tp: float) -> bool:
        self._require_connection()
        pos = next((p for p in self.positions(magic) if p.symbol == symbol), None)
        if pos is None:
            return False
        request = {
            "action": self._mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": pos.ticket,
            "sl": float(sl),
            "tp": float(tp),
            "magic": int(magic),
        }
        res = self._mt5.order_send(request)
        return res is not None and res.retcode == self._mt5.TRADE_RETCODE_DONE

    def close_position(self, symbol: str, magic: int) -> bool:
        self._require_connection()
        pos = next((p for p in self.positions(magic) if p.symbol == symbol), None)
        if pos is None:
            return False
        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            return False
        is_buy = pos.direction == 1
        # a long is closed by selling at the bid; a short by buying at the ask
        price = tick.bid if is_buy else tick.ask
        request = {
            "action": self._mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": pos.ticket,
            "volume": pos.volume,
            "type": self._mt5.ORDER_TYPE_SELL if is_buy else self._mt5.ORDER_TYPE_BUY,
            "price": price,
            "magic": int(magic),
            "comment": "ALGLORY close",
            "type_time": self._mt5.ORDER_TIME_GTC,
            "type_filling": self._mt5.ORDER_FILLING_IOC,
            "deviation": 20,
        }
        res = self._mt5.order_send(request)
        return res is not None and res.retcode == self._mt5.TRADE_RETCODE_DONE

    def _interpret(self, res, price: float, lots: float) -> OrderResult:
        if res is None:
            code, desc = self._mt5.last_error()
            return OrderResult(False, int(code), f"order_send failed: {desc}")
        ok = res.retcode == self._mt5.TRADE_RETCODE_DONE
        return OrderResult(
            ok,
            int(res.retcode),
            "done" if ok else f"rejected (retcode {res.retcode})",
            ticket=getattr(res, "order", None),
            price=price,
            volume=lots,
        )

    def experts_dir(self) -> Path | None:
        if not self._connected or self._mt5 is None:
            return None
        info = self._mt5.terminal_info()
        if info is None:
            return None
        path = Path(info.data_path) / "MQL5" / "Experts" / "Alglory"
        path.mkdir(parents=True, exist_ok=True)
        return path
