"""Stage 3: the threaded ExecutorManager + the MT5 bar clock."""

from __future__ import annotations

import pytest

from alglory.data.fake_mt5 import FakeMT5Source
from alglory.data.sample import generate_bars
from alglory.deploy.mql5 import GUARDRAIL_PRESETS
from alglory.genome import (
    FilterGene,
    Genome,
    ManagementGene,
    RiskGene,
    SignalGene,
)
from alglory.live.clock import MT5BarClock
from alglory.live.executor import LiveExecutor, Mode, StrategyRuntime
from alglory.server.executor_worker import ExecutorManager
from alglory.server.worker import BusyError

MAGIC = 990_001


def _genome():
    return Genome(
        "trend", "both",
        SignalGene("ma_cross", {"fast": 5, "slow": 15}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=1.5, tp_atr=3.0, trail_atr=None, max_bars=None),
        RiskGene(0.01),
    )


def _rt():
    return StrategyRuntime(
        genome=_genome(), symbol="EURUSD", timeframe="H1",
        guard=GUARDRAIL_PRESETS["personal"], magic=MAGIC,
    )


class _SeqClock:
    """Yields a new (longer) closed-bar window on each call — a new bar each tick."""

    def __init__(self, frames):
        self._frames = frames
        self.i = 0

    def latest_closed_bars(self, symbol, timeframe):
        frame = self._frames[min(self.i, len(self._frames) - 1)]
        self.i += 1
        return frame


class _FixedClock:
    def __init__(self, frame):
        self.frame = frame

    def latest_closed_bars(self, symbol, timeframe):
        return self.frame


# ---- ExecutorManager --------------------------------------------------


def test_manager_emits_one_intent_per_new_bar():
    bars = generate_bars("EURUSD", "H1", 300, seed=7)
    frames = [bars.iloc[:k].reset_index(drop=True) for k in range(200, 210)]
    ex = LiveExecutor(FakeMT5Source(balance=100_000), [_rt()], mode=Mode.DEMO)
    m = ExecutorManager()
    m.arm(ex, _SeqClock(frames), start=False)
    for _ in range(5):
        m.tick()
    events = m.drain()
    assert len(events) == 5
    assert all(e["type"] == "executor_intent" for e in events)
    assert events[0]["magic"] == MAGIC


def test_manager_dedups_same_bar():
    frame = generate_bars("EURUSD", "H1", 260, seed=7).iloc[:250].reset_index(drop=True)
    ex = LiveExecutor(FakeMT5Source(balance=100_000), [_rt()], mode=Mode.DEMO)
    m = ExecutorManager()
    m.arm(ex, _FixedClock(frame), start=False)
    m.tick()
    m.tick()
    m.tick()
    assert len(m.drain()) == 1  # same last closed bar -> only the first tick acts


def test_manager_arm_twice_is_busy():
    ex = LiveExecutor(FakeMT5Source(), [_rt()], mode=Mode.DRY_RUN)
    frame = generate_bars("EURUSD", "H1", 200, seed=1).reset_index(drop=True)
    m = ExecutorManager(poll_seconds=0.05)
    m.arm(ex, _FixedClock(frame), start=True)
    try:
        with pytest.raises(BusyError):
            m.arm(ex, _FixedClock(frame), start=True)
    finally:
        m.stop()
    assert m.is_running() is False


def test_manager_stop_flattens_and_halts():
    src = FakeMT5Source()
    rt = _rt()
    ex = LiveExecutor(src, [rt], mode=Mode.DEMO)
    src.place_market("EURUSD", 1, 0.1, sl=1.0, tp=1.2, magic=MAGIC)
    m = ExecutorManager()
    m.arm(ex, _FixedClock(generate_bars("EURUSD", "H1", 200, seed=1)), start=False)
    m.stop()
    assert src.positions(MAGIC) == []
    assert rt.halted is True


def test_manager_status_idle():
    s = ExecutorManager().status()
    assert s["running"] is False and s["armed"] is False and s["strategies"] == []


# ---- MT5BarClock ------------------------------------------------------


def test_bar_clock_drops_the_forming_bar():
    bars = generate_bars("EURUSD", "H1", 50, seed=1)

    class _StubSource:
        def fetch_bars(self, symbol, timeframe, count):
            return bars.tail(count).reset_index(drop=True)

    closed = MT5BarClock(_StubSource(), count=10).latest_closed_bars("EURUSD", "H1")
    assert len(closed) == 10  # asked for 11, dropped the forming one
    assert closed["time"].iloc[-1] == bars["time"].iloc[-2]  # last row is last CLOSED bar


def test_bar_clock_returns_none_on_fetch_error():
    class _BadSource:
        def fetch_bars(self, *a):
            raise RuntimeError("no terminal")

    assert MT5BarClock(_BadSource()).latest_closed_bars("EURUSD", "H1") is None
