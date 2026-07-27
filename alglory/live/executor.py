"""Live executor (stage 2): trade vaulted strategies via order_send, no chart.

The executor decides on each *newly closed bar* using the SAME signal function
as the backtester (`alglory.backtest.engine.build_signals`), so its entry
signals are identical to the backtest's by construction — the property is
pinned by a parity test. Position state is read back from the broker (live
reality), guardrails are enforced in Python mirroring the generated EA, and lot
sizing reuses `alglory.analysis.sizing.lots_for_risk` with the broker's real
contract spec.

Modes:
  - dry_run: compute + emit intents, send nothing to the broker.
  - demo:    execute, but only against a demo account.
  - live:    execute; refused at construction unless confirm_live=True.

Bar-clock polling and the server/worker wiring are stage 3; this module is the
decision engine plus a `replay` harness for testing it against FakeMT5Source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from alglory.analysis.sizing import lots_for_risk
from alglory.backtest.engine import ATR_PERIOD, build_signals
from alglory.deploy.mql5 import Guardrails
from alglory.genome import Genome
from alglory.indicators import atr


class Mode(str, Enum):
    DRY_RUN = "dry_run"
    DEMO = "demo"
    LIVE = "live"


class Action(str, Enum):
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE = "close"
    HOLD = "hold"
    HALT = "halt"
    SKIP_TOO_SMALL = "skip_too_small"
    SKIP_CAP = "skip_cap"


class ExecutorError(Exception):
    pass


@dataclass
class StrategyRuntime:
    """One vaulted strategy on one symbol/timeframe, with its own magic + state."""

    genome: Genome
    symbol: str
    timeframe: str
    guard: Guardrails
    magic: int
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    day: int = -1
    halted: bool = False


@dataclass(frozen=True)
class Intent:
    action: Action
    magic: int
    symbol: str
    lots: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    direction: int = 0
    reason: str = ""


class LiveExecutor:
    def __init__(
        self,
        source,
        runtimes,
        *,
        mode: Mode = Mode.DRY_RUN,
        confirm_live: bool = False,
        max_open_positions: int | None = None,
    ):
        if mode == Mode.LIVE and not confirm_live:
            raise ExecutorError("live mode requires confirm_live=True — refusing to arm.")
        if max_open_positions is not None and max_open_positions < 1:
            raise ExecutorError("max_open_positions must be >= 1.")
        self.source = source
        self.runtimes = list(runtimes)
        self.mode = mode
        self._confirm_live = confirm_live
        self.max_open_positions = max_open_positions
        self.events: list[Intent] = []

    def _total_open(self) -> int:
        return sum(len(self.source.positions(rt.magic)) for rt in self.runtimes)

    # ---- decision (pure w.r.t. the broker; reads state, no side effects) ----

    def entry_signal(self, rt: StrategyRuntime, bars: pd.DataFrame) -> int:
        """Signal of the last CLOSED bar — identical to the backtester's."""
        sig = build_signals(rt.genome, bars)
        return int(sig[-1]) if len(sig) else 0

    def _guardrails_ok(self, rt: StrategyRuntime, equity: float, day_of_year: int) -> bool:
        if equity > rt.peak_equity:
            rt.peak_equity = equity
        if rt.halted:
            return False
        if rt.peak_equity > 0 and equity < rt.peak_equity * (1.0 - rt.guard.max_dd_pct):
            rt.halted = True
            return False
        if day_of_year != rt.day:
            rt.day = day_of_year
            rt.day_start_equity = equity
        if rt.day_start_equity > 0 and equity < rt.day_start_equity * (1.0 - rt.guard.daily_loss_pct):
            return False  # done for the day
        return True

    def decide(self, rt: StrategyRuntime, bars: pd.DataFrame) -> Intent:
        close = bars["close"].to_numpy()
        high = bars["high"].to_numpy()
        low = bars["low"].to_numpy()
        acct = self.source.account()
        doy = int(pd.to_datetime(bars["time"].iloc[-1]).dayofyear) if "time" in bars else -1
        pos = self.source.positions(rt.magic)

        if not self._guardrails_ok(rt, acct.equity, doy):
            if pos:
                return Intent(Action.CLOSE, rt.magic, rt.symbol, reason="guardrail halt")
            return Intent(Action.HALT, rt.magic, rt.symbol, reason="guardrail halt")

        d = self.entry_signal(rt, bars)

        if pos:
            if d == -pos[0].direction and d != 0:
                return Intent(Action.CLOSE, rt.magic, rt.symbol, reason="opposite signal")
            return Intent(Action.HOLD, rt.magic, rt.symbol, reason="in position")

        a = atr(high, low, close, ATR_PERIOD)
        entry_atr = a[-1]
        if d == 0 or np.isnan(entry_atr) or entry_atr <= 0:
            return Intent(Action.HOLD, rt.magic, rt.symbol, reason="no signal")

        sl_dist = rt.genome.management.sl_atr * entry_atr
        tp_dist = rt.genome.management.tp_atr * entry_atr
        spec = self.source.symbol_spec(rt.symbol)
        lots = lots_for_risk(acct.balance, rt.genome.risk.risk_pct, sl_dist, spec)
        if lots <= 0:
            return Intent(Action.SKIP_TOO_SMALL, rt.magic, rt.symbol,
                          reason="risk-based size below broker min lot")

        if self.max_open_positions is not None and self._total_open() >= self.max_open_positions:
            return Intent(Action.SKIP_CAP, rt.magic, rt.symbol,
                          reason=f"global cap of {self.max_open_positions} open positions reached")

        entry_px = float(close[-1])
        sl = entry_px - d * sl_dist
        tp = entry_px + d * tp_dist
        action = Action.OPEN_LONG if d == 1 else Action.OPEN_SHORT
        return Intent(action, rt.magic, rt.symbol, lots=lots, sl=sl, tp=tp, direction=d, reason="signal")

    # ---- execution ----------------------------------------------------------

    def on_bar(self, rt: StrategyRuntime, bars: pd.DataFrame) -> Intent:
        intent = self.decide(rt, bars)
        self.events.append(intent)
        if self.mode == Mode.DRY_RUN:
            return intent  # compute + log only, never touches the broker
        if self.mode == Mode.DEMO and not self.source.account().is_demo:
            raise ExecutorError("demo mode requires a demo account; refusing to trade.")
        self._execute(rt, intent)
        return intent

    def _execute(self, rt: StrategyRuntime, intent: Intent) -> None:
        if intent.action in (Action.OPEN_LONG, Action.OPEN_SHORT):
            direction = 1 if intent.action == Action.OPEN_LONG else -1
            self.source.place_market(
                rt.symbol, direction, intent.lots, sl=intent.sl, tp=intent.tp, magic=rt.magic
            )
        elif intent.action == Action.CLOSE:
            self.source.close_position(rt.symbol, rt.magic)

    def stop(self) -> None:
        """Kill switch: flatten every managed strategy and halt it."""
        for rt in self.runtimes:
            if self.source.positions(rt.magic):
                self.source.close_position(rt.symbol, rt.magic)
            rt.halted = True

    def status(self) -> list[dict]:
        return [
            {
                "magic": rt.magic,
                "symbol": rt.symbol,
                "halted": rt.halted,
                "open_positions": len(self.source.positions(rt.magic)),
                "peak_equity": rt.peak_equity,
            }
            for rt in self.runtimes
        ]


def replay(executor: LiveExecutor, rt: StrategyRuntime, bars: pd.DataFrame, *, warmup: int = 150):
    """Step the executor bar-by-bar over historical ``bars`` against a broker
    double, letting the broker close SL/TP between bars. Returns the executor's
    intents. This is the executor's 'backtest via itself' used in tests."""
    close = bars["close"].to_numpy()
    for i in range(warmup, len(bars)):
        price = float(close[i])
        if hasattr(executor.source, "mark_price"):
            executor.source.mark_price(price)  # broker-side SL/TP between bars
        executor.on_bar(rt, bars.iloc[: i + 1])
    return executor.events
