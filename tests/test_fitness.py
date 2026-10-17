import pytest

from alglory.backtest.metrics import Metrics
from alglory.evolve.fitness import (
    FitnessConfig,
    HoldoutGate,
    OOSGate,
    fitness,
    holdout_pass,
    oos_pass,
)


def m(net=0.2, dd=0.1, pf=1.8, sharpe=1.2, wr=0.6, trades=60, ayr=0.2):
    return Metrics(
        net_profit=net,
        max_drawdown=dd,
        profit_factor=pf,
        sharpe=sharpe,
        win_rate=wr,
        trades=trades,
        avg_yearly_return=ayr,
    )


CFG = FitnessConfig()


def test_losing_strategy_scores_zero():
    assert fitness(m(net=-0.1), CFG) == 0.0
    assert fitness(m(net=0.0), CFG) == 0.0


def test_fitness_monotonic_in_profit_factor():
    lo = fitness(m(pf=1.2), CFG)
    hi = fitness(m(pf=2.5), CFG)
    assert hi > lo > 0


def test_fitness_monotonic_in_sharpe():
    assert fitness(m(sharpe=2.0), CFG) > fitness(m(sharpe=0.5), CFG)


def test_undertrading_penalized():
    full = fitness(m(trades=CFG.min_trades), CFG)
    half = fitness(m(trades=CFG.min_trades // 2), CFG)
    assert half < full
    assert half == pytest.approx(full * (CFG.min_trades // 2) / CFG.min_trades)


def test_drawdown_penalty_goes_to_zero_at_cap():
    assert fitness(m(dd=CFG.dd_cap), CFG) == pytest.approx(0.0)
    assert fitness(m(dd=CFG.dd_cap * 0.5), CFG) < fitness(m(dd=0.02), CFG)


def test_fitness_bounded():
    best = m(pf=100.0, sharpe=10.0, dd=0.0, trades=500)
    assert 0.0 < fitness(best, CFG) <= 1.0


GATE = OOSGate()


def test_oos_pass_truth_table():
    good = m(net=0.1, pf=1.5, dd=0.1, trades=20)
    assert oos_pass(good, GATE) is True
    assert oos_pass(m(net=-0.05, pf=1.5, dd=0.1, trades=20), GATE) is False
    assert oos_pass(m(net=0.1, pf=1.05, dd=0.1, trades=20), GATE) is False
    assert oos_pass(m(net=0.1, pf=1.5, dd=0.5, trades=20), GATE) is False
    assert oos_pass(m(net=0.1, pf=1.5, dd=0.1, trades=GATE.min_trades - 1), GATE) is False


def test_fitness_rewards_win_rate_when_profitable():
    assert fitness(m(wr=0.68), CFG) > fitness(m(wr=0.50), CFG)


def test_fitness_zero_when_losing_even_with_high_win_rate():
    assert fitness(m(net=-0.05, wr=0.80), CFG) == 0.0


def test_wr_score_saturates_at_70pct():
    assert fitness(m(wr=0.70), CFG) == pytest.approx(fitness(m(wr=0.90), CFG))


def test_select_gate_requires_55pct_win_rate():
    assert oos_pass(m(pf=1.5, trades=40, wr=0.55), GATE) is False
    assert oos_pass(m(pf=1.5, trades=40, wr=0.56), GATE) is True


HOLD = HoldoutGate()


def test_holdout_pass_bar():
    ok = dict(wr=0.61, pf=1.2, trades=50, net=0.02)
    assert holdout_pass(m(**ok), HOLD) is True
    assert holdout_pass(m(**{**ok, "wr": 0.60}), HOLD) is False
    assert holdout_pass(m(**{**ok, "pf": 1.09}), HOLD) is False
    assert holdout_pass(m(**{**ok, "trades": 49}), HOLD) is False
    assert holdout_pass(m(**{**ok, "net": -0.01}), HOLD) is False


def test_wr_score_is_zero_at_or_below_45pct():
    assert fitness(m(wr=0.45), CFG) == pytest.approx(fitness(m(wr=0.40), CFG))


def test_wr_score_linear_between_45_and_70pct():
    lo = fitness(m(wr=0.45), CFG)
    hi = fitness(m(wr=0.70), CFG)
    assert fitness(m(wr=0.575), CFG) == pytest.approx((lo + hi) / 2)


def test_holdout_pass_applies_max_dd():
    ok = dict(wr=0.61, pf=1.2, trades=50, net=0.02)
    assert holdout_pass(m(**{**ok, "dd": 0.40}), HoldoutGate(max_dd=0.35)) is False
    assert holdout_pass(m(**{**ok, "dd": 0.35}), HoldoutGate(max_dd=0.35)) is True
    assert HoldoutGate().max_dd == 0.35  # FitnessConfig's default dd_cap
