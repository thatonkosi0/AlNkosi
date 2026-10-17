from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from alglory.analysis.demo import account_prop, decide_stage, demo_stats

T0 = datetime(2026, 9, 1, 9, 0, tzinfo=UTC)


def _deals(rows):
    return pd.DataFrame(rows, columns=["time", "magic", "entry", "profit", "commission", "swap"])


def test_demo_stats_counts_only_its_magic_out_deals():
    d = _deals([
        (T0, 990_001, "in", 0.0, 0.0, 0.0),
        (T0, 990_001, "out", 10.0, 0.0, 0.0),
        (T0, 990_001, "out", -4.0, 0.0, 0.0),
        (T0, 990_002, "out", 99.0, 0.0, 0.0),
    ])
    s = demo_stats(d, 990_001)
    assert s == {"trades": 2, "wins": 1, "win_rate": 0.5, "net": 6.0}


def test_demo_stats_includes_commission_and_swap():
    d = _deals([(T0, 990_001, "out", 5.0, -7.0, 0.0)])
    s = demo_stats(d, 990_001)
    assert s["wins"] == 0 and s["net"] == -2.0


def test_demo_stats_empty():
    s = demo_stats(_deals([]), 990_001)
    assert s == {"trades": 0, "wins": 0, "win_rate": 0.0, "net": 0.0}


def test_account_prop_combines_strategies():
    d = _deals([
        (T0, 990_001, "out", -200.0, 0.0, 0.0),
        (T0 + timedelta(hours=1), 990_002, "out", -200.0, 0.0, 0.0),
    ])
    # balance after both losses is 9,600, so the day opened at 10,000
    m, days = account_prop(d, 9_600.0)
    assert m.worst_day_loss == pytest.approx(0.04)
    assert m.max_drawdown == pytest.approx(0.04)
    assert days[T0.date()] == pytest.approx(0.04)


def test_account_prop_ignores_non_alglory_pnl_but_uses_it_for_balance():
    d = _deals([
        (T0, 0, "out", 500.0, 0.0, 0.0),  # manual trade, opens day at 10,000
        (T0 + timedelta(hours=1), 990_001, "out", -100.0, 0.0, 0.0),
        (T0 + timedelta(days=1), 990_001, "out", 50.0, 0.0, 0.0),
    ])
    m, days = account_prop(d, 10_450.0)
    assert days[T0.date()] == pytest.approx(100 / 10_000)
    assert days[(T0 + timedelta(days=1)).date()] == 0.0
    assert m.worst_day_loss == pytest.approx(0.01)


def test_account_prop_empty():
    m, days = account_prop(_deals([]), 10_000.0)
    assert (m.worst_day_loss, m.max_drawdown) == (0.0, 0.0) and days == {}


def test_decide_stage_rules():
    start = datetime(2026, 9, 1, tzinfo=UTC)
    d10, d21 = start + timedelta(days=10), start + timedelta(days=21)
    good = {"trades": 25, "wins": 16, "win_rate": 0.64, "net": 100.0}
    assert decide_stage(good, start, d21, True) == ("demo_failed", "prop limit breached")
    assert decide_stage(good, start, d10, False) == ("on_demo", None)
    few = {**good, "trades": 19}
    assert decide_stage(few, start, d21, False) == ("on_demo", "needs more trades")
    assert decide_stage(good, start, d21, False) == ("demo_passed", None)
    edge = {**good, "trades": 20, "win_rate": 0.60}
    stage, why = decide_stage(edge, start, d21, False)
    assert stage == "demo_failed" and "win rate" in why
    loss = {**good, "net": -1.0}
    stage, why = decide_stage(loss, start, d21, False)
    assert stage == "demo_failed" and "net" in why
