import pytest

from alglory.backtest.metrics import Metrics
from alglory.evolve.fitness import FitnessConfig, OOSGate, fitness, oos_pass


def m(net=0.2, dd=0.1, pf=1.8, sharpe=1.2, wr=0.5, trades=60, ayr=0.2):
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
