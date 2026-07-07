import numpy as np
import pytest

from alglory.backtest.engine import BacktestResult, Trade
from alglory.backtest.metrics import Metrics, compute_metrics


def _trade(pnl, reason="tp"):
    return Trade(
        entry_i=0, exit_i=1, direction=1, entry=1.0, exit=1.1, pnl_pct=pnl, reason=reason
    )


def test_metrics_basic_hand_computed():
    equity = np.array([1.0, 1.02, 1.02, 1.0098, 1.0098])
    trades = [_trade(0.02), _trade(-0.01, "sl")]
    m = compute_metrics(BacktestResult(trades=trades, equity=equity), bars_per_year=6240.0)
    assert m.net_profit == pytest.approx(0.0098)
    assert m.trades == 2
    assert m.win_rate == pytest.approx(0.5)
    # gross wins 0.02, gross losses 0.01 -> PF 2
    assert m.profit_factor == pytest.approx(2.0)
    # max drawdown: peak 1.02 -> trough 1.0098
    assert m.max_drawdown == pytest.approx((1.02 - 1.0098) / 1.02)


def test_metrics_no_trades():
    m = compute_metrics(
        BacktestResult(trades=[], equity=np.ones(100)), bars_per_year=6240.0
    )
    assert m.trades == 0
    assert m.net_profit == 0.0
    assert m.profit_factor == 0.0
    assert m.sharpe == 0.0
    assert m.win_rate == 0.0


def test_metrics_all_wins_pf_capped():
    equity = np.array([1.0, 1.02, 1.05])
    m = compute_metrics(
        BacktestResult(trades=[_trade(0.02), _trade(0.03)], equity=equity),
        bars_per_year=6240.0,
    )
    assert m.profit_factor == 100.0
    assert m.win_rate == 1.0
    assert m.max_drawdown == 0.0


def test_avg_yearly_return_annualizes():
    # equity doubles over exactly one year of bars
    n = 1000
    equity = np.linspace(1.0, 2.0, n)
    m = compute_metrics(
        BacktestResult(trades=[_trade(1.0)], equity=equity), bars_per_year=float(n)
    )
    assert m.avg_yearly_return == pytest.approx(1.0, rel=0.01)


def test_sharpe_positive_for_steady_gains():
    rng = np.random.default_rng(1)
    rets = 0.0005 + rng.normal(0, 0.001, 2000)
    equity = np.cumprod(1 + rets)
    m = compute_metrics(
        BacktestResult(trades=[_trade(0.1)], equity=equity), bars_per_year=6240.0
    )
    assert m.sharpe > 5


def test_metrics_dict_round_trip():
    equity = np.array([1.0, 1.02])
    m = compute_metrics(
        BacktestResult(trades=[_trade(0.02)], equity=equity), bars_per_year=6240.0
    )
    m2 = Metrics.from_dict(m.to_dict())
    assert m2 == m
