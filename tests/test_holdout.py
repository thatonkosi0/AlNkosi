import numpy as np
import pandas as pd
import pytest

import alglory.evolve.holdout as holdout_mod
from alglory.backtest.engine import BacktestResult, Costs, Trade
from alglory.backtest.metrics import Metrics
from alglory.deploy.mql5 import GUARDRAIL_PRESETS
from alglory.evolve.fitness import HoldoutGate
from alglory.evolve.holdout import (
    Finalist,
    HoldoutResult,
    judge_finalist,
    keep_best,
)
from alglory.genome import random_genome

COSTS = Costs(spread=0.0001)


def _genome():
    return random_genome("trend", np.random.default_rng(3))


def _metrics(**kw):
    base = dict(net_profit=0.1, max_drawdown=0.05, profit_factor=1.5, sharpe=1.0,
                win_rate=0.65, trades=60, avg_yearly_return=0.1)
    base.update(kw)
    return Metrics(**base)


def _finalist(g=None, tf="H1"):
    g = g or _genome()
    return Finalist(genome=g, tribe=g.tribe, timeframe=tf,
                    is_metrics=_metrics(), select_metrics=_metrics())


def _bars(n):
    return pd.DataFrame({"time": pd.date_range("2024-01-01", periods=n, freq="h")})


def _result(pnls_by_exit):
    """BacktestResult from (exit_i, pnl_pct) pairs; equity compounds at exits."""
    trades, eq = [], [1.0]
    for exit_i, pnl in pnls_by_exit:
        trades.append(Trade(entry_i=max(exit_i - 1, 0), exit_i=exit_i, direction=1,
                            entry=1.0, exit=1.0, pnl_pct=pnl, reason="tp"))
        eq.append(eq[-1] * (1 + pnl))
    return BacktestResult(trades=trades, equity=np.array(eq))


def _patch_backtest(monkeypatch, result):
    calls = []

    def fake(genome, bars, costs):
        calls.append(len(bars))
        return result

    monkeypatch.setattr(holdout_mod, "run_backtest", fake)
    return calls


def test_finalist_json_roundtrip_and_key():
    f = _finalist(tf="H4")
    back = Finalist.from_json(f.to_json())
    assert back.key == f.key == f.genome.to_json() + "|H4"
    assert back.tribe == f.tribe and back.timeframe == "H4"
    assert back.is_metrics == f.is_metrics and back.select_metrics == f.select_metrics


def test_judge_rejects_high_win_rate_loser(monkeypatch):
    g = _genome()
    r = g.risk.risk_pct
    # 75% wins of +0.2R, 25% losses of -1R: high win rate, net loss
    pnls = [(i, (0.2 * r) if i % 4 else -r) for i in range(1, 81)]
    calls = _patch_backtest(monkeypatch, _result(pnls))
    res = judge_finalist(_finalist(g), _bars(100), COSTS, HoldoutGate(), None, 6240.0)
    assert len(calls) == 1  # exactly one backtest
    assert res.metrics.win_rate == pytest.approx(0.75)
    assert res.passed is False
    assert any("net profit" in reason for reason in res.reasons)


def test_judge_records_are_r_multiples_with_naive_iso_times(monkeypatch):
    g = _genome()
    r = g.risk.risk_pct
    _patch_backtest(monkeypatch, _result([(5, 2 * r), (7, -r)]))
    bars = _bars(10)
    res = judge_finalist(_finalist(g), bars, COSTS, HoldoutGate(min_trades=1), None, 6240.0)
    assert res.records == [
        {"exit_time": bars["time"].iloc[5].isoformat(), "r": pytest.approx(2.0)},
        {"exit_time": bars["time"].iloc[7].isoformat(), "r": pytest.approx(-1.0)},
    ]
    assert "+" not in res.records[0]["exit_time"]  # tz-naive


def test_judge_applies_prop_preset(monkeypatch):
    g = _genome()
    r = g.risk.risk_pct
    # day 1 (hours 0-8): nine straight -1R losses; then 51 winning days of +1R
    pnls = [(h, -r) for h in range(9)] + [(24 * d + 12, r) for d in range(1, 52)]
    _patch_backtest(monkeypatch, _result(pnls))
    bars = _bars(24 * 53)
    f = _finalist(g)

    personal = judge_finalist(f, bars, COSTS, HoldoutGate(), None, 6240.0)
    assert personal.passed is True
    assert personal.reasons == []

    prop = judge_finalist(f, bars, COSTS, HoldoutGate(), GUARDRAIL_PRESETS["prop_conservative"], 6240.0)
    assert prop.passed is False
    assert any("daily loss" in reason for reason in prop.reasons)


def test_judge_reasons_name_each_failed_gate_condition(monkeypatch):
    g = _genome()
    r = g.risk.risk_pct
    # 10 trades, 50% win, net positive: fails trades and win rate, not net profit
    pnls = [(i, (2 * r) if i % 2 else -r) for i in range(1, 11)]
    _patch_backtest(monkeypatch, _result(pnls))
    res = judge_finalist(_finalist(g), _bars(20), COSTS, HoldoutGate(), None, 6240.0)
    assert res.passed is False
    text = "; ".join(res.reasons)
    assert "win rate" in text and "trades" in text
    assert "net profit" not in text


def _hr(win, pf, passed=True):
    m = _metrics(win_rate=win, profit_factor=pf)
    return HoldoutResult(finalist=_finalist(), metrics=m, records=[], passed=passed,
                         reasons=[] if passed else ["x"])


def test_keep_best_caps_and_orders():
    passers = [_hr(0.62, 1.2), _hr(0.70, 2.0), _hr(0.65, 1.5), _hr(0.61, 3.0), _hr(0.80, 1.1)]
    failers = [_hr(0.95, 9.0, passed=False), _hr(0.99, 50.0, passed=False)]
    kept = keep_best(failers + passers, 3)
    assert len(kept) == 3
    assert all(k.passed for k in kept)
    scores = [k.metrics.win_rate * k.metrics.profit_factor for k in kept]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(0.61 * 3.0)
    assert scores[1] == pytest.approx(0.70 * 2.0)
    assert scores[2] == pytest.approx(0.65 * 1.5)


def test_holdout_gate_thresholds_pinned():
    g = HoldoutGate()
    assert (g.min_win_rate, g.min_pf, g.min_trades) == (0.60, 1.1, 50)


def test_judge_enforces_dd_cap_under_personal(monkeypatch):
    g = _genome()
    r = g.risk.risk_pct
    # 20 straight -1R losses (deep drawdown) then enough wins to pass the rest
    pnls = [(i, -r) for i in range(1, 21)] + [(i, 2 * r) for i in range(21, 121)]
    _patch_backtest(monkeypatch, _result(pnls))
    f = _finalist(g)
    res = judge_finalist(f, _bars(130), COSTS, HoldoutGate(max_dd=0.01), None, 6240.0)
    assert res.metrics.max_drawdown > 0.01
    assert res.passed is False
    assert any(x.startswith("max drawdown") and "> dd_cap 1%" in x for x in res.reasons)
    loose = judge_finalist(f, _bars(130), COSTS, HoldoutGate(max_dd=0.99), None, 6240.0)
    assert loose.passed is True and loose.reasons == []
