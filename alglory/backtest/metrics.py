"""Performance metrics computed from a BacktestResult."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .engine import BacktestResult

PF_CAP = 100.0


@dataclass(frozen=True)
class Metrics:
    net_profit: float
    max_drawdown: float
    profit_factor: float
    sharpe: float
    win_rate: float
    trades: int
    avg_yearly_return: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Metrics":
        return cls(**d)


def compute_metrics(result: BacktestResult, bars_per_year: float) -> Metrics:
    equity = np.asarray(result.equity, dtype=float)
    trades = result.trades
    n_trades = len(trades)

    net_profit = float(equity[-1] - 1.0) if len(equity) else 0.0

    peaks = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = (peaks - equity) / peaks
    max_drawdown = float(np.nanmax(dd)) if len(equity) else 0.0

    if n_trades == 0:
        return Metrics(0.0, max_drawdown, 0.0, 0.0, 0.0, 0, 0.0)

    pnls = np.array([t.pnl_pct for t in trades])
    gross_win = pnls[pnls > 0].sum()
    gross_loss = -pnls[pnls < 0].sum()
    if gross_loss == 0.0:
        profit_factor = PF_CAP if gross_win > 0 else 0.0
    else:
        profit_factor = float(min(gross_win / gross_loss, PF_CAP))

    win_rate = float((pnls > 0).sum() / n_trades)

    rets = np.diff(equity) / equity[:-1]
    std = rets.std()
    sharpe = float(rets.mean() / std * np.sqrt(bars_per_year)) if std > 0 else 0.0

    if equity[-1] > 0 and len(equity) > 1:
        avg_yearly_return = float(equity[-1] ** (bars_per_year / len(equity)) - 1.0)
    else:
        avg_yearly_return = -1.0

    return Metrics(
        net_profit=net_profit,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        sharpe=sharpe,
        win_rate=win_rate,
        trades=n_trades,
        avg_yearly_return=avg_yearly_return,
    )
