"""Stage 2: the LiveExecutor decision engine + the signal-parity invariant."""

from __future__ import annotations

import pytest

from alglory.backtest.engine import build_signals
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
from alglory.live.executor import (
    Action,
    ExecutorError,
    LiveExecutor,
    Mode,
    StrategyRuntime,
    replay,
)

MAGIC = 990117


def _trading_genome():
    # frequent MA crosses, no filters -> reliably produces signals to compare
    return Genome(
        "trend", "both",
        SignalGene("ma_cross", {"fast": 5, "slow": 15}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=1.5, tp_atr=3.0, trail_atr=None, max_bars=None),
        RiskGene(0.01),
    )


def _rt(genome):
    return StrategyRuntime(
        genome=genome, symbol="EURUSD", timeframe="H1",
        guard=GUARDRAIL_PRESETS["personal"], magic=MAGIC,
    )


# ---- the core invariant ----------------------------------------------


def test_executor_entry_signal_matches_backtester_one_to_one():
    bars = generate_bars("EURUSD", "H1", 700, seed=7)
    g = _trading_genome()
    ref = build_signals(g, bars)
    ex = LiveExecutor(FakeMT5Source(), [], mode=Mode.DRY_RUN)
    rt = _rt(g)
    # incremental (as it runs live) must equal the backtester's batch signal
    for i in range(120, len(bars)):
        assert ex.entry_signal(rt, bars.iloc[: i + 1]) == int(ref[i]), f"mismatch at bar {i}"


def test_parity_holds_with_filters_engaged():
    bars = generate_bars("EURUSD", "H1", 700, seed=3)
    g = Genome(
        "trend", "short",
        SignalGene("macd_trend", {"fast": 9, "slow": 23, "signal": 7}),
        FilterGene(trend_ma=100, session=(6, 20), atr_regime="high"),
        ManagementGene(sl_atr=2.0, tp_atr=3.0, trail_atr=1.0, max_bars=None),
        RiskGene(0.01),
    )
    ref = build_signals(g, bars)
    ex = LiveExecutor(FakeMT5Source(), [], mode=Mode.DRY_RUN)
    rt = _rt(g)
    for i in range(200, len(bars)):
        assert ex.entry_signal(rt, bars.iloc[: i + 1]) == int(ref[i])


# ---- safety / modes --------------------------------------------------


def test_live_mode_refused_without_confirm():
    with pytest.raises(ExecutorError):
        LiveExecutor(FakeMT5Source(), [], mode=Mode.LIVE)
    # confirm_live=True is accepted
    LiveExecutor(FakeMT5Source(), [], mode=Mode.LIVE, confirm_live=True)


def test_demo_mode_refuses_non_demo_account():
    ex = LiveExecutor(FakeMT5Source(is_demo=False), [], mode=Mode.DEMO)
    bars = generate_bars("EURUSD", "H1", 200, seed=1)
    with pytest.raises(ExecutorError):
        ex.on_bar(_rt(_trading_genome()), bars)


def test_dry_run_never_touches_broker():
    src = FakeMT5Source()
    ex = LiveExecutor(src, [], mode=Mode.DRY_RUN)
    replay(ex, _rt(_trading_genome()), generate_bars("EURUSD", "H1", 600, seed=7))
    assert src.positions() == []  # nothing was ever placed
    assert any(o[0] == "broker_exit" for o in src.orders) is False
    assert len(ex.events) > 0  # but it did compute intents


def test_guardrail_halt_when_drawdown_breached():
    src = FakeMT5Source(balance=1000)  # equity 1000
    ex = LiveExecutor(src, [], mode=Mode.DEMO)
    rt = _rt(_trading_genome())
    rt.peak_equity = 2000.0  # a 50% drawdown vs a 25% max-DD guardrail
    intent = ex.decide(rt, generate_bars("EURUSD", "H1", 200, seed=1))
    assert intent.action == Action.HALT
    assert rt.halted is True


# ---- sizing gate -----------------------------------------------------


def test_big_account_opens_but_tiny_account_skips():
    bars = generate_bars("EURUSD", "H1", 1500, seed=7)
    g = _trading_genome()

    big = LiveExecutor(FakeMT5Source(balance=100_000), [], mode=Mode.DEMO)
    replay(big, _rt(g), bars)
    opens_big = [e for e in big.events if e.action in (Action.OPEN_LONG, Action.OPEN_SHORT)]
    assert len(opens_big) > 0  # a funded account actually trades

    tiny = LiveExecutor(FakeMT5Source(balance=5), [], mode=Mode.DEMO)
    replay(tiny, _rt(g), bars)
    opens_tiny = [e for e in tiny.events if e.action in (Action.OPEN_LONG, Action.OPEN_SHORT)]
    skips = [e for e in tiny.events if e.action == Action.SKIP_TOO_SMALL]
    assert opens_tiny == []  # too small to place anything
    assert len(skips) > 0  # and it says so


# ---- lifecycle -------------------------------------------------------


def test_replay_on_demo_places_and_broker_exits_run():
    src = FakeMT5Source(balance=100_000)
    ex = LiveExecutor(src, [], mode=Mode.DEMO)
    replay(ex, _rt(_trading_genome()), generate_bars("EURUSD", "H1", 1500, seed=7))
    assert any(o[0] == "open" for o in src.orders)
    # broker-side SL/TP closed at least one position during the replay
    assert any(o[0] == "broker_exit" for o in src.orders)


def _rt_magic(m):
    return StrategyRuntime(
        genome=_trading_genome(), symbol="EURUSD", timeframe="H1",
        guard=GUARDRAIL_PRESETS["personal"], magic=m,
    )


def _multi_replay(executor, runtimes, bars, warmup=150):
    close = bars["close"].to_numpy()
    max_open = 0
    for i in range(warmup, len(bars)):
        executor.source.mark_price(float(close[i]))
        for rt in runtimes:
            executor.on_bar(rt, bars.iloc[: i + 1])
        total = sum(len(executor.source.positions(rt.magic)) for rt in runtimes)
        max_open = max(max_open, total)
    return max_open


def test_max_open_positions_cap_is_enforced():
    bars = generate_bars("EURUSD", "H1", 1500, seed=7)
    rts = [_rt_magic(1), _rt_magic(2)]
    ex = LiveExecutor(FakeMT5Source(balance=100_000), rts, mode=Mode.DEMO, max_open_positions=1)
    max_open = _multi_replay(ex, rts, bars)
    assert max_open <= 1  # never more than the cap open at once
    assert any(e.action == Action.SKIP_CAP for e in ex.events)


def test_without_cap_both_strategies_can_hold_positions():
    bars = generate_bars("EURUSD", "H1", 1500, seed=7)
    rts = [_rt_magic(1), _rt_magic(2)]
    ex = LiveExecutor(FakeMT5Source(balance=100_000), rts, mode=Mode.DEMO)
    assert _multi_replay(ex, rts, bars) == 2


def test_bad_cap_rejected():
    with pytest.raises(ExecutorError):
        LiveExecutor(FakeMT5Source(), [], mode=Mode.DRY_RUN, max_open_positions=0)


def test_stop_flattens_and_halts():
    src = FakeMT5Source()
    rt = _rt(_trading_genome())
    ex = LiveExecutor(src, [rt], mode=Mode.DEMO)
    src.place_market("EURUSD", 1, 0.1, sl=1.0, tp=1.2, magic=MAGIC)
    assert src.positions(MAGIC)
    ex.stop()
    assert src.positions(MAGIC) == []
    assert rt.halted is True
    assert ex.status()[0]["halted"] is True
