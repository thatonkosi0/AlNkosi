"""In-memory MetaTrader 5 double.

Implements the trading slice of MT5Source's interface — account, symbol_spec,
positions, place_market, modify_position, close_position — backed by plain
memory. It exists so the live executor (and its tests) can run without a real
terminal, and so the executor's ``dry_run`` mode has a broker that records
intended orders instead of sending them. Fills are instantaneous and frictionless
by design; this is a behavioural stand-in, not a market simulator.
"""

from __future__ import annotations

from dataclasses import replace

from alglory.analysis.sizing import EURUSD_STD, SymbolSpec
from alglory.data.mt5source import AccountInfo, ConnStatus, OrderResult, PositionInfo

_RETCODE_DONE = 10009


class FakeMT5Source:
    def __init__(
        self,
        *,
        balance: float = 10_000.0,
        currency: str = "USD",
        is_demo: bool = True,
        spec: SymbolSpec | None = None,
        price: float = 1.10,
    ) -> None:
        self._balance = balance
        self._equity = balance
        self._currency = currency
        self._is_demo = is_demo
        self._spec = spec or EURUSD_STD
        self._price = price
        self._positions: list[PositionInfo] = []
        self._next_ticket = 1
        self.orders: list[tuple] = []  # recorded intents: (kind, ...)

    # parity with MT5Source's connection surface
    def available(self) -> bool:
        return True

    def connect(self) -> ConnStatus:
        return ConnStatus(ok=True, message="Fake MT5 (in-memory).")

    def set_price(self, price: float) -> None:
        self._price = price

    def mark_price(self, price: float) -> list[int]:
        """Advance the market to ``price`` and close any position whose SL/TP is
        crossed, mirroring what a real broker does between bars. Returns the
        tickets closed."""
        self._price = price
        closed: list[int] = []
        for p in list(self._positions):
            hit_sl = p.sl > 0 and (
                (p.direction == 1 and price <= p.sl) or (p.direction == -1 and price >= p.sl)
            )
            hit_tp = p.tp > 0 and (
                (p.direction == 1 and price >= p.tp) or (p.direction == -1 and price <= p.tp)
            )
            if hit_sl or hit_tp:
                self._positions.remove(p)
                closed.append(p.ticket)
                self.orders.append(("broker_exit", p.symbol, p.magic))
        return closed

    def account(self) -> AccountInfo:
        return AccountInfo(0, self._balance, self._equity, self._currency, self._is_demo)

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        return self._spec

    def positions(self, magic: int | None = None) -> list[PositionInfo]:
        return [p for p in self._positions if magic is None or p.magic == magic]

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
        self.orders.append(("open", symbol, direction, float(lots), int(magic)))
        if lots <= 0:
            return OrderResult(False, -1, "zero volume — nothing placed.")
        ticket = self._next_ticket
        self._next_ticket += 1
        self._positions.append(
            PositionInfo(ticket, symbol, direction, float(lots), self._price,
                         float(sl), float(tp), int(magic), 0.0)
        )
        return OrderResult(True, _RETCODE_DONE, "done", ticket=ticket, price=self._price, volume=float(lots))

    def modify_position(self, symbol: str, magic: int, *, sl: float, tp: float) -> bool:
        for i, p in enumerate(self._positions):
            if p.symbol == symbol and p.magic == magic:
                self._positions[i] = replace(p, sl=float(sl), tp=float(tp))
                self.orders.append(("modify", symbol, int(magic), float(sl), float(tp)))
                return True
        return False

    def close_position(self, symbol: str, magic: int) -> bool:
        for i, p in enumerate(self._positions):
            if p.symbol == symbol and p.magic == magic:
                del self._positions[i]
                self.orders.append(("close", symbol, int(magic)))
                return True
        return False
