import numpy as np
import pandas as pd
import pytest

from alglory.backtest.engine import Costs, build_signals, run_backtest
from alglory.genome import (
    FilterGene,
    Genome,
    ManagementGene,
    RiskGene,
    SignalGene,
)


def make_bars(rows, start="2024-01-01"):
    """rows: list of (open, high, low, close) tuples, hourly bars."""
    o, h, l, c = (np.array(x, dtype=float) for x in zip(*rows))
    return pd.DataFrame(
        {
            "time": pd.date_range(start, periods=len(rows), freq="h"),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": np.ones(len(rows)),
            "spread": np.zeros(len(rows)),
        }
    )


WARMUP = [(100.0, 100.5, 99.5, 100.0)] * 21  # constant TR=1.0 -> ATR(14)=1.0
FLAT = (100.0, 100.4, 99.8, 100.0)


def genome_with(mgmt, direction="both", filters=None, risk=0.01):
    return Genome(
        tribe="trend",
        direction=direction,
        signal=SignalGene(kind="ma_cross", params={"fast": 5, "slow": 20}),
        filters=filters or FilterGene(trend_ma=None, session=None, atr_regime=None),
        management=mgmt,
        risk=RiskGene(risk_pct=risk),
    )


def inject(n, **at):
    """signals array of length n with values at given indices, e.g. i20=1."""
    s = np.zeros(n, dtype=np.int8)
    for key, val in at.items():
        s[int(key[1:])] = val
    return s


def test_long_tp_hit_exact_pnl():
    bars = make_bars(
        WARMUP
        + [
            (100.0, 100.4, 99.8, 100.2),  # bar 21: entry at open=100, no touch
            (100.2, 102.3, 100.0, 102.0),  # bar 22: TP=102 hit
            FLAT,
        ]
    )
    g = genome_with(ManagementGene(sl_atr=1.0, tp_atr=2.0, trail_atr=None, max_bars=None))
    res = run_backtest(g, bars, Costs(spread=0.0), signals=inject(len(bars), i20=1))
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_i == 21 and t.exit_i == 22
    assert t.direction == 1
    assert t.entry == pytest.approx(100.0)
    assert t.exit == pytest.approx(102.0)
    assert t.reason == "tp"
    assert t.pnl_pct == pytest.approx(0.02)  # 2R * 1% risk
    assert res.equity[-1] == pytest.approx(1.02)
    assert res.equity[0] == 1.0
    assert len(res.equity) == len(bars)


def test_sl_fills_first_when_both_hit():
    bars = make_bars(
        WARMUP
        + [
            (100.0, 100.4, 99.8, 100.2),
            (100.2, 102.5, 98.9, 100.0),  # both SL=99 and TP=102 in range
            FLAT,
        ]
    )
    g = genome_with(ManagementGene(sl_atr=1.0, tp_atr=2.0, trail_atr=None, max_bars=None))
    res = run_backtest(g, bars, Costs(spread=0.0), signals=inject(len(bars), i20=1))
    t = res.trades[0]
    assert t.reason == "sl"
    assert t.exit == pytest.approx(99.0)
    assert t.pnl_pct == pytest.approx(-0.01)


def test_spread_reduces_pnl():
    bars = make_bars(
        WARMUP
        + [
            (100.0, 100.4, 99.8, 100.2),
            (100.2, 102.3, 100.0, 102.0),
            FLAT,
        ]
    )
    g = genome_with(ManagementGene(sl_atr=1.0, tp_atr=2.0, trail_atr=None, max_bars=None))
    res = run_backtest(g, bars, Costs(spread=0.2), signals=inject(len(bars), i20=1))
    # effective entry 100.1, effective exit 101.9 -> 1.8R * 1% = 0.018
    assert res.trades[0].pnl_pct == pytest.approx(0.018)


def test_trailing_stop_ratchets_and_exits():
    bars = make_bars(
        WARMUP
        + [
            (100.0, 103.0, 99.5, 102.5),  # bar 21: entry, no exit (SL=98, TP=108); trail -> 102
            (102.5, 102.8, 101.5, 102.0),  # bar 22: low 101.5 <= trailed SL 102
            FLAT,
        ]
    )
    g = genome_with(ManagementGene(sl_atr=2.0, tp_atr=8.0, trail_atr=1.0, max_bars=None))
    res = run_backtest(g, bars, Costs(spread=0.0), signals=inject(len(bars), i20=1))
    t = res.trades[0]
    assert t.reason == "trail"
    assert t.exit == pytest.approx(102.0)
    assert t.pnl_pct == pytest.approx((102.0 - 100.0) / 2.0 * 0.01)


def test_max_bars_time_exit():
    bars = make_bars(
        WARMUP
        + [
            (100.0, 100.4, 99.8, 100.2),  # bar 21 entry
            (100.2, 100.6, 99.9, 100.3),  # bar 22 held
            (100.5, 100.9, 100.1, 100.6),  # bar 23: time exit at open
            FLAT,
        ]
    )
    g = genome_with(ManagementGene(sl_atr=5.0, tp_atr=8.0, trail_atr=None, max_bars=2))
    res = run_backtest(g, bars, Costs(spread=0.0), signals=inject(len(bars), i20=1))
    t = res.trades[0]
    assert t.reason == "time"
    assert t.exit_i == 23
    assert t.exit == pytest.approx(100.5)
    assert t.pnl_pct == pytest.approx(0.5 / 5.0 * 0.01)


def test_equity_compounds_across_trades():
    bars = make_bars(
        WARMUP
        + [
            (100.0, 100.4, 99.8, 100.2),  # 21: entry 1
            (100.2, 102.3, 100.0, 102.0),  # 22: TP +2%
            FLAT,  # 23
            FLAT,  # 24 (signal here)
            (100.0, 100.4, 99.8, 100.2),  # 25: entry 2
            (100.2, 100.5, 98.0, 98.2),  # 26: deep low guarantees SL hit
            FLAT,
        ]
    )
    g = genome_with(ManagementGene(sl_atr=1.0, tp_atr=2.0, trail_atr=None, max_bars=None))
    res = run_backtest(g, bars, Costs(spread=0.0), signals=inject(len(bars), i20=1, i24=1))
    assert len(res.trades) == 2
    assert res.trades[0].pnl_pct == pytest.approx(0.02)
    assert res.trades[1].pnl_pct < 0
    assert res.equity[-1] == pytest.approx(1.02 * (1 + res.trades[1].pnl_pct))


def test_end_of_data_closes_position():
    bars = make_bars(WARMUP + [(100.0, 100.4, 99.8, 100.3)])
    g = genome_with(ManagementGene(sl_atr=5.0, tp_atr=8.0, trail_atr=None, max_bars=None))
    res = run_backtest(g, bars, Costs(spread=0.0), signals=inject(len(bars), i20=1))
    assert len(res.trades) == 1
    assert res.trades[0].reason == "end"
    assert res.trades[0].exit == pytest.approx(100.3)


def test_short_trade_pnl_sign():
    bars = make_bars(
        WARMUP
        + [
            (100.0, 100.2, 99.6, 99.8),  # entry short at 100
            (99.8, 100.1, 97.9, 98.0),  # TP short = 98 (tp_atr=2)
            FLAT,
        ]
    )
    g = genome_with(ManagementGene(sl_atr=1.0, tp_atr=2.0, trail_atr=None, max_bars=None))
    res = run_backtest(g, bars, Costs(spread=0.0), signals=inject(len(bars), i20=-1))
    t = res.trades[0]
    assert t.direction == -1
    assert t.reason == "tp"
    assert t.pnl_pct == pytest.approx(0.02)


def test_build_signals_session_filter():
    rows = [(100.0, 100.5, 99.5, 100.0)] * 26 + [(100.0, 105.2, 99.9, 105.0)] * 3
    bars = make_bars(rows)  # jump bar is index 26 -> hour 02:00
    sig = SignalGene(kind="momentum_burst", params={"period": 5, "threshold": 0.02})
    mgmt = ManagementGene(sl_atr=1.0, tp_atr=2.0, trail_atr=None, max_bars=None)
    open_filters = FilterGene(trend_ma=None, session=None, atr_regime=None)
    g_open = Genome("momentum", "long", sig, open_filters, mgmt, RiskGene(0.01))
    g_closed = Genome(
        "momentum",
        "long",
        sig,
        FilterGene(trend_ma=None, session=(8, 12), atr_regime=None),
        mgmt,
        RiskGene(0.01),
    )
    s_open = build_signals(g_open, bars)
    s_closed = build_signals(g_closed, bars)
    assert s_open[26] == 1
    assert s_closed[26] == 0
    assert (s_closed == 0).all()


def test_build_signals_direction_long_only():
    rows = [(100.0, 100.5, 99.5, 100.0)] * 26 + [(100.0, 100.1, 94.8, 95.0)] * 3
    bars = make_bars(rows)  # 5% drop -> raw short signal
    sig = SignalGene(kind="momentum_burst", params={"period": 5, "threshold": 0.02})
    mgmt = ManagementGene(sl_atr=1.0, tp_atr=2.0, trail_atr=None, max_bars=None)
    filters = FilterGene(trend_ma=None, session=None, atr_regime=None)
    g_long = Genome("momentum", "long", sig, filters, mgmt, RiskGene(0.01))
    g_both = Genome("momentum", "both", sig, filters, mgmt, RiskGene(0.01))
    assert build_signals(g_both, bars)[26] == -1
    assert (build_signals(g_long, bars) >= 0).all()


def test_build_signals_is_causal(bars_h1):
    sig = SignalGene(kind="ma_cross", params={"fast": 10, "slow": 40})
    g = Genome(
        "trend",
        "both",
        sig,
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=2.0, tp_atr=3.0, trail_atr=None, max_bars=None),
        RiskGene(0.01),
    )
    full = build_signals(g, bars_h1)
    partial = build_signals(g, bars_h1.iloc[:3000].reset_index(drop=True))
    assert np.array_equal(full[:3000], partial)
