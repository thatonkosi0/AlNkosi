"""Stage 2: the LiveExecutor decision engine + the signal-parity invariant."""

from __future__ import annotations

import numpy as np
import pytest

from alglory.backtest.engine import build_signals
from alglory.data.fake_mt5 import FakeMT5Source
from alglory.data.mt5source import OrderResult, PositionInfo
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


def test_sizing_respects_guardrail_preset_cap():
    bars = generate_bars("EURUSD", "H1", 1500, seed=7)
    g = _trading_genome()  # RiskGene(0.01)

    def first_open_lots(guard):
        rt = StrategyRuntime(genome=g, symbol="EURUSD", timeframe="H1", guard=guard, magic=MAGIC)
        ex = LiveExecutor(FakeMT5Source(balance=100_000), [], mode=Mode.DEMO)
        for i in range(120, len(bars)):
            intent = ex.decide(rt, bars.iloc[: i + 1])
            if intent.action in (Action.OPEN_LONG, Action.OPEN_SHORT):
                return intent.lots
        return None

    lots_personal = first_open_lots(GUARDRAIL_PRESETS["personal"])  # risk_pct 0.01, matches genome
    lots_conservative = first_open_lots(GUARDRAIL_PRESETS["prop_conservative"])  # risk_pct 0.005

    assert lots_personal is not None and lots_conservative is not None
    assert lots_conservative < lots_personal


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


# ---- stop management (breakeven, trailing, time exit) ------------------


def test_manage_stops_arms_breakeven():
    g = Genome(
        "trend", "long",
        SignalGene("ma_cross", {"fast": 5, "slow": 15}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=1.5, tp_atr=6.0, trail_atr=None, max_bars=None, breakeven_atr=0.5),
        RiskGene(0.01),
    )
    rt = StrategyRuntime(
        genome=g, symbol="EURUSD", timeframe="H1",
        guard=GUARDRAIL_PRESETS["personal"], magic=MAGIC,
    )
    rt.entry_atr = 0.0010
    ex = LiveExecutor(FakeMT5Source(), [], mode=Mode.DRY_RUN)
    current = PositionInfo(
        ticket=1, symbol="EURUSD", direction=1, volume=0.1,
        price_open=1.1000, sl=1.0985, tp=1.1060, magic=MAGIC, profit=0.0,
    )
    # breakeven trigger = 1.1000 + 0.5 * 0.0010 = 1.1005 -> bar's high reaches it
    high = np.array([1.1000, 1.1006])
    low = np.array([1.0995, 1.1000])
    sl, tp, reason = ex._manage_stops(rt, current, high, low, bar_atr=0.0012)
    assert reason == "be"
    assert sl == pytest.approx(1.1000)
    assert rt.be_armed is True


def test_manage_stops_ratchets_trail():
    g = Genome(
        "trend", "long",
        SignalGene("ma_cross", {"fast": 5, "slow": 15}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=1.5, tp_atr=6.0, trail_atr=1.0, max_bars=None, breakeven_atr=None),
        RiskGene(0.01),
    )
    rt = StrategyRuntime(
        genome=g, symbol="EURUSD", timeframe="H1",
        guard=GUARDRAIL_PRESETS["personal"], magic=MAGIC,
    )
    rt.entry_atr = 0.0010
    ex = LiveExecutor(FakeMT5Source(), [], mode=Mode.DRY_RUN)
    current = PositionInfo(
        ticket=1, symbol="EURUSD", direction=1, volume=0.1,
        price_open=1.1000, sl=1.0985, tp=1.1060, magic=MAGIC, profit=0.0,
    )
    high = np.array([1.1030])
    low = np.array([1.1010])
    # candidate = high[-1] - trail_atr * bar_atr = 1.1030 - 0.0010 = 1.1020 > sl(1.0985)
    sl, tp, reason = ex._manage_stops(rt, current, high, low, bar_atr=0.0010)
    assert reason == "trail"
    assert sl == pytest.approx(1.1020)
    assert tp == pytest.approx(1.1060 + (1.1020 - 1.0985))


def test_time_exit_closes_after_max_bars():
    g = Genome(
        "trend", "both",
        SignalGene("ma_cross", {"fast": 5, "slow": 15}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=1.5, tp_atr=6.0, trail_atr=None, max_bars=3, breakeven_atr=None),
        RiskGene(0.01),
    )
    bars = generate_bars("EURUSD", "H1", 1500, seed=7)
    src = FakeMT5Source(balance=100_000)
    ex = LiveExecutor(src, [], mode=Mode.DEMO)
    rt = StrategyRuntime(
        genome=g, symbol="EURUSD", timeframe="H1",
        guard=GUARDRAIL_PRESETS["personal"], magic=MAGIC,
    )
    replay(ex, rt, bars)
    time_exits = [e for e in ex.events if e.action == Action.CLOSE and e.reason == "time exit"]
    assert len(time_exits) > 0


def test_live_executor_ratchets_and_arms_over_a_full_replay():
    g = Genome(
        "trend", "both",
        SignalGene("ma_cross", {"fast": 5, "slow": 15}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=1.5, tp_atr=6.0, trail_atr=1.0, max_bars=40, breakeven_atr=0.5),
        RiskGene(0.01),
    )
    bars = generate_bars("EURUSD", "H1", 1500, seed=7)
    src = FakeMT5Source(balance=100_000)
    ex = LiveExecutor(src, [], mode=Mode.DEMO)
    rt = StrategyRuntime(
        genome=g, symbol="EURUSD", timeframe="H1",
        guard=GUARDRAIL_PRESETS["personal"], magic=MAGIC,
    )
    replay(ex, rt, bars)
    modifies = [e for e in ex.events if e.action == Action.MODIFY]
    assert len(modifies) > 0
    assert any(e.reason == "trail" for e in modifies)


# ---- adopted positions (server crash / restart / re-arm) -------------------


def _adoptable_genome(breakeven_atr=3.0, trail_atr=None, max_bars=None):
    return Genome(
        "trend", "long",  # long-only: a long position can never see an opposite signal
        SignalGene("ma_cross", {"fast": 5, "slow": 15}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(
            sl_atr=1.5, tp_atr=6.0, trail_atr=trail_atr,
            max_bars=max_bars, breakeven_atr=breakeven_atr,
        ),
        RiskGene(0.01),
    )


def _adopt_setup(genome):
    """A runtime that finds a position already open at the broker, exactly as
    /api/executor/arm does after a crash+restart: fresh StrategyRuntime state
    (entry_atr=0.0, be_armed=False, bars_in_trade=0) over a live position."""
    bars = generate_bars("EURUSD", "H1", 400, seed=7)
    src = FakeMT5Source(balance=100_000)
    # price_open sits exactly at the last bar's low, so a breakeven trigger
    # computed from a *zero* entry_atr (== price_open) is certainly "reached",
    # while the real 3-ATR trigger is not.
    price_open = float(bars["low"].iloc[-1])
    src.set_price(price_open)
    src.place_market("EURUSD", 1, 0.1, sl=price_open - 0.02, tp=price_open + 0.05, magic=MAGIC)
    rt = _rt(genome)
    assert rt.entry_atr == 0.0 and rt.be_armed is False  # the reconstructed state
    ex = LiveExecutor(src, [rt], mode=Mode.DRY_RUN)
    return ex, rt, bars


def test_adopted_position_does_not_spuriously_arm_breakeven():
    ex, rt, bars = _adopt_setup(_adoptable_genome())
    intent = ex.decide(rt, bars)
    assert not (intent.action == Action.MODIFY and intent.reason == "be"), (
        "an adopted position must not have its stop yanked to breakeven off a "
        "reconstructed entry_atr of 0.0"
    )


def test_adopted_position_seeds_entry_atr_and_clears_the_sentinel():
    ex, rt, bars = _adopt_setup(_adoptable_genome())
    ex.decide(rt, bars)
    assert rt.entry_atr > 0.0  # sentinel cleared with a real, current ATR
    assert rt.be_armed is True  # breakeven declined, not retroactively armed


def test_adopted_position_is_reported_as_a_warning():
    ex, rt, bars = _adopt_setup(_adoptable_genome())
    ex.decide(rt, bars)
    assert len(ex.execution_warnings) == 1
    warn = ex.execution_warnings[0]
    assert warn["magic"] == MAGIC and warn["symbol"] == "EURUSD"
    assert "adopted" in warn["message"]


def test_adoption_is_detected_only_once():
    ex, rt, bars = _adopt_setup(_adoptable_genome())
    ex.decide(rt, bars)
    ex.decide(rt, bars)
    ex.decide(rt, bars)
    assert len(ex.execution_warnings) == 1  # not re-warned every single bar


def test_normal_entry_still_arms_breakeven_after_adoption_guard():
    """Regression guard: the adoption branch must not suppress breakeven for a
    position this executor actually opened itself."""
    ex, rt, bars = _adopt_setup(_adoptable_genome())
    ex.source.close_position("EURUSD", MAGIC)  # no adoption: we open it ourselves
    rt.entry_atr = 0.0010
    rt.be_armed = False
    current = PositionInfo(
        ticket=1, symbol="EURUSD", direction=1, volume=0.1,
        price_open=1.1000, sl=1.0985, tp=1.1060, magic=MAGIC, profit=0.0,
    )
    high = np.array([1.1000, 1.1040])
    low = np.array([1.0995, 1.1000])
    sl, tp, reason = ex._manage_stops(rt, current, high, low, bar_atr=0.0010)
    assert reason == "be" and sl == pytest.approx(1.1000)


def test_self_opened_positions_never_look_adopted():
    """End-to-end guard: the adoption branch must not fire — and must not
    suppress real breakeven arming — for positions the executor opened itself."""
    g = Genome(
        "trend", "both",
        SignalGene("ma_cross", {"fast": 5, "slow": 15}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=1.5, tp_atr=6.0, trail_atr=None, max_bars=40, breakeven_atr=0.5),
        RiskGene(0.01),
    )
    ex = LiveExecutor(FakeMT5Source(balance=100_000), [], mode=Mode.DEMO)
    rt = _rt(g)
    replay(ex, rt, generate_bars("EURUSD", "H1", 1500, seed=7))
    assert ex.execution_warnings == []  # nothing here was adopted
    be = [e for e in ex.events if e.action == Action.MODIFY and e.reason == "be"]
    assert len(be) > 0  # and breakeven still arms normally


# ---- execution failures (errors surface instead of being discarded) --------


def test_execution_failure_recorded_on_rejected_open():
    class _RejectingSource(FakeMT5Source):
        def place_market(self, *a, **kw):
            return OrderResult(False, -1, "rejected: market closed")

    src = _RejectingSource(balance=100_000)
    ex = LiveExecutor(src, [], mode=Mode.DEMO)
    rt = _rt(_trading_genome())
    replay(ex, rt, generate_bars("EURUSD", "H1", 1500, seed=7))
    assert len(ex.execution_errors) > 0
    assert ex.execution_errors[0]["action"] in ("open_long", "open_short")
    assert "rejected" in ex.execution_errors[0]["message"]


def test_execution_failure_recorded_on_rejected_close():
    class _NoCloseSource(FakeMT5Source):
        def close_position(self, symbol, magic):
            return False

    src = _NoCloseSource(balance=100_000)
    rt = _rt(_trading_genome())
    ex = LiveExecutor(src, [rt], mode=Mode.DEMO)
    src.place_market("EURUSD", 1, 0.1, sl=1.05, tp=1.15, magic=MAGIC)
    rt.peak_equity = 2_000_000.0  # forces a guardrail halt -> CLOSE intent
    ex.on_bar(rt, generate_bars("EURUSD", "H1", 200, seed=1))
    assert any(e["action"] == "close" for e in ex.execution_errors)
