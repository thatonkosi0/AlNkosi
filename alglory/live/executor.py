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
    MODIFY = "modify"
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
    be_armed: bool = False
    entry_atr: float = 0.0
    bars_in_trade: int = 0


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
        self.execution_errors: list[dict] = []
        # Non-fatal anomalies the operator should see but which are not order
        # failures (drained by ExecutorManager.tick as executor_warning events).
        self.execution_warnings: list[dict] = []

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
        a = atr(high, low, close, ATR_PERIOD)
        bar_atr = a[-1]

        if pos:
            current = pos[0]
            if rt.entry_atr <= 0:
                self._adopt_position(rt, current, bar_atr)
            rt.bars_in_trade += 1
            mgmt = rt.genome.management
            if mgmt.max_bars is not None and rt.bars_in_trade >= mgmt.max_bars:
                return Intent(Action.CLOSE, rt.magic, rt.symbol, reason="time exit")
            if d == -current.direction and d != 0:
                return Intent(Action.CLOSE, rt.magic, rt.symbol, reason="opposite signal")
            new_sl, new_tp, reason = self._manage_stops(rt, current, high, low, bar_atr)
            if reason is not None:
                return Intent(
                    Action.MODIFY, rt.magic, rt.symbol, sl=new_sl, tp=new_tp,
                    direction=current.direction, reason=reason,
                )
            return Intent(Action.HOLD, rt.magic, rt.symbol, reason="in position")

        entry_atr = bar_atr
        if d == 0 or np.isnan(entry_atr) or entry_atr <= 0:
            return Intent(Action.HOLD, rt.magic, rt.symbol, reason="no signal")

        sl_dist = rt.genome.management.sl_atr * entry_atr
        tp_dist = rt.genome.management.tp_atr * entry_atr
        spec = self.source.symbol_spec(rt.symbol)
        # The guardrail preset is a ceiling on per-trade risk, not just the
        # daily-loss/max-DD halts — mirrors the generated EA, which always
        # sizes off InpRiskPct (the preset), never the genome's own risk_pct.
        risk_pct = min(rt.genome.risk.risk_pct, rt.guard.risk_pct)
        lots = lots_for_risk(acct.balance, risk_pct, sl_dist, spec)
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
        rt.entry_atr = entry_atr
        rt.be_armed = False
        rt.bars_in_trade = 0
        return Intent(action, rt.magic, rt.symbol, lots=lots, sl=sl, tp=tp, direction=d, reason="signal")

    def _adopt_position(self, rt: StrategyRuntime, current, bar_atr: float) -> None:
        """Take over a position that was already open before this runtime existed.

        ``POST /api/executor/arm`` rebuilds every ``StrategyRuntime`` from
        scratch (``entry_atr=0.0, be_armed=False, bars_in_trade=0``) and the
        magic number is deterministic (``990_000 + strategy_id``) — so if the
        server dies while a position is open and is then restarted and
        re-armed, it "adopts" that live position with state that does not
        describe it. ``entry_atr == 0.0`` is the tell, and it is actively
        dangerous: ``_manage_stops`` would compute
        ``price_open + d * breakeven_atr * 0.0``, i.e. a breakeven trigger
        exactly at ``price_open``, which the very next bar essentially always
        "reaches" — yanking the stop to breakeven no matter what the genome's
        ``breakeven_atr`` actually says.

        So: seed a real ATR from the current bar to clear the sentinel, and
        treat breakeven as already decided rather than arming it off state we
        never observed. Declining to arm is the conservative direction — it
        leaves the broker-side stop exactly where the operator's broker
        already has it instead of moving real money's risk on a guess.
        """
        seeded = float(bar_atr) if not np.isnan(bar_atr) and bar_atr > 0 else 0.0
        rt.entry_atr = seeded
        rt.be_armed = True  # skip breakeven arming: no real basis to arm it from
        side = "long" if current.direction == 1 else "short"
        atr_note = (
            f"seeded entry_atr={seeded:.6g} from the current bar"
            if seeded > 0
            else "no usable ATR on this bar yet, so entry_atr stays unset"
        )
        self._record_warning(
            rt,
            f"adopted an already-open {side} position (ticket {current.ticket}) with no "
            f"entry state — this happens after a crash/restart + re-arm. Breakeven "
            f"arming is declined for this position and the max_bars clock restarts "
            f"from this bar; {atr_note}.",
        )

    def _manage_stops(self, rt: StrategyRuntime, current, high, low, bar_atr: float):
        """Mirror engine.update_stops: arm breakeven, ratchet the trail.

        Returns (sl, tp, reason) where reason is None if nothing changed.
        """
        mgmt = rt.genome.management
        d = current.direction
        sl, tp = current.sl, current.tp
        reason = None

        if mgmt.breakeven_atr is not None and not rt.be_armed:
            trigger = current.price_open + d * mgmt.breakeven_atr * rt.entry_atr
            reached = high[-1] >= trigger if d == 1 else low[-1] <= trigger
            if reached:
                rt.be_armed = True
                if (d == 1 and current.price_open > sl) or (d == -1 and current.price_open < sl):
                    sl = current.price_open
                    reason = "be"

        if mgmt.trail_atr is not None:
            effective_atr = bar_atr if not np.isnan(bar_atr) else rt.entry_atr
            dist = mgmt.trail_atr * effective_atr
            candidate = high[-1] - dist if d == 1 else low[-1] + dist
            if (d == 1 and candidate > sl) or (d == -1 and candidate < sl):
                tp = tp + (candidate - sl)
                sl = candidate
                reason = "trail"

        return sl, tp, reason

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
            result = self.source.place_market(
                rt.symbol, direction, intent.lots, sl=intent.sl, tp=intent.tp, magic=rt.magic
            )
            if not result.ok:
                self._record_error(rt, intent, result.message)
        elif intent.action == Action.CLOSE:
            ok = self.source.close_position(rt.symbol, rt.magic)
            if not ok:
                self._record_error(rt, intent, "close_position rejected or position not found")
        elif intent.action == Action.MODIFY:
            ok = self.source.modify_position(rt.symbol, rt.magic, sl=intent.sl, tp=intent.tp)
            if not ok:
                self._record_error(rt, intent, "modify_position rejected or position not found")

    def _record_error(self, rt: StrategyRuntime, intent: Intent, message: str) -> None:
        self.execution_errors.append(
            {"magic": rt.magic, "symbol": rt.symbol, "action": intent.action.value, "message": message}
        )

    def _record_warning(self, rt: StrategyRuntime, message: str) -> None:
        """A non-fatal anomaly worth an operator's attention but not an order
        failure — same shape as the worker's bar-fetch executor_warning."""
        self.execution_warnings.append(
            {"magic": rt.magic, "symbol": rt.symbol, "message": message}
        )

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
