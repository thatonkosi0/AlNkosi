"""Finalists and the once-per-market holdout judgement.

A finalist is a genome that passed the select gate. After a market's breeding
is finished, each finalist is backtested exactly once on that timeframe's
holdout slice - bars nothing else in the campaign has seen. Holdout results
are never fed back into breeding or selection; keep_best only trims the
passers to a per-market cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from alglory.analysis.prop import breaches, prop_metrics, trade_stream
from alglory.backtest.engine import Costs, run_backtest
from alglory.backtest.metrics import Metrics, compute_metrics
from alglory.deploy.mql5 import Guardrails
from alglory.evolve.fitness import HoldoutGate, holdout_pass
from alglory.genome import Genome


@dataclass
class Finalist:
    genome: Genome
    tribe: str
    timeframe: str
    is_metrics: Metrics
    select_metrics: Metrics

    @property
    def key(self) -> str:
        return self.genome.to_json() + "|" + self.timeframe

    def to_json(self) -> dict:
        return {
            "genome": self.genome.to_json(),
            "tribe": self.tribe,
            "timeframe": self.timeframe,
            "is_metrics": self.is_metrics.to_dict(),
            "select_metrics": self.select_metrics.to_dict(),
        }

    @classmethod
    def from_json(cls, d: dict | str) -> "Finalist":
        if isinstance(d, str):
            d = json.loads(d)
        return cls(
            genome=Genome.from_json(d["genome"]),
            tribe=d["tribe"],
            timeframe=d["timeframe"],
            is_metrics=Metrics.from_dict(d["is_metrics"]),
            select_metrics=Metrics.from_dict(d["select_metrics"]),
        )


@dataclass
class HoldoutResult:
    finalist: Finalist
    metrics: Metrics
    records: list[dict] = field(default_factory=list)
    passed: bool = False
    reasons: list[str] = field(default_factory=list)


def _naive_iso(t) -> str:
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.isoformat()


def _gate_reasons(m: Metrics, gate: HoldoutGate) -> list[str]:
    out = []
    if not m.net_profit > 0.0:
        out.append(f"net profit {m.net_profit:.2%} not positive")
    if not m.win_rate > gate.min_win_rate:
        out.append(f"win rate {m.win_rate:.1%} not above {gate.min_win_rate:.0%}")
    if not m.profit_factor >= gate.min_pf:
        out.append(f"profit factor {m.profit_factor:.2f} below {gate.min_pf}")
    if not m.trades >= gate.min_trades:
        out.append(f"{m.trades} trades below {gate.min_trades}")
    if not m.max_drawdown <= gate.max_dd:
        out.append(f"max drawdown {m.max_drawdown:.0%} > dd_cap {gate.max_dd:.0%}")
    return out


def judge_finalist(
    f: Finalist,
    holdout_bars: pd.DataFrame,
    costs: Costs,
    gate: HoldoutGate,
    preset: Guardrails | None,
    bars_per_year: float,
) -> HoldoutResult:
    """One backtest on the holdout slice; the verdict and why."""
    res = run_backtest(f.genome, holdout_bars, costs)
    m = compute_metrics(res, bars_per_year)
    times = holdout_bars["time"]
    risk = f.genome.risk.risk_pct
    records = [
        {"exit_time": _naive_iso(times.iloc[t.exit_i]), "r": t.pnl_pct / risk}
        for t in res.trades
    ]
    reasons: list[str] = []
    gate_ok = holdout_pass(m, gate)
    if not gate_ok:
        reasons += _gate_reasons(m, gate) or ["holdout gate failed"]
    if preset is not None:
        stream = trade_stream(res.trades, times, risk, preset.risk_pct)
        reasons += breaches(prop_metrics(stream), preset)
    return HoldoutResult(
        finalist=f, metrics=m, records=records, passed=gate_ok and not reasons, reasons=reasons
    )


def keep_best(results: list[HoldoutResult], k: int) -> list[HoldoutResult]:
    """Top-k holdout passers by win_rate x profit_factor."""
    passers = [r for r in results if r.passed]
    passers.sort(key=lambda r: r.metrics.win_rate * r.metrics.profit_factor, reverse=True)
    return passers[:k]
