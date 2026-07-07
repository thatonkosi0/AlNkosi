"""Fitness scoring (in-sample) and the out-of-sample survival gate."""

from __future__ import annotations

from dataclasses import dataclass

from alglory.backtest.metrics import Metrics


@dataclass(frozen=True)
class FitnessConfig:
    min_trades: int = 30
    dd_cap: float = 0.35
    pf_weight: float = 0.6
    sharpe_weight: float = 0.4


def fitness(m: Metrics, cfg: FitnessConfig) -> float:
    if m.net_profit <= 0.0 or m.trades == 0:
        return 0.0
    pf_score = min(m.profit_factor, 3.0) / 3.0
    sharpe_score = min(max(m.sharpe, 0.0), 3.0) / 3.0
    score = cfg.pf_weight * pf_score + cfg.sharpe_weight * sharpe_score
    if m.trades < cfg.min_trades:
        score *= m.trades / cfg.min_trades
    dd_penalty = max(0.0, 1.0 - m.max_drawdown / cfg.dd_cap)
    return float(score * dd_penalty)


@dataclass(frozen=True)
class OOSGate:
    min_pf: float = 1.1
    max_dd: float = 0.30
    min_trades: int = 8


def oos_pass(m: Metrics, gate: OOSGate) -> bool:
    return (
        m.net_profit > 0.0
        and m.profit_factor >= gate.min_pf
        and m.max_drawdown <= gate.max_dd
        and m.trades >= gate.min_trades
    )
