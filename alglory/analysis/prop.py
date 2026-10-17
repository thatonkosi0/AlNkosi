"""Prop-firm limits (worst daily loss, max drawdown) measured from trade streams."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alglory.backtest.engine import Trade
from alglory.deploy.mql5 import Guardrails


@dataclass(frozen=True)
class PropMetrics:
    worst_day_loss: float  # positive fraction of balance
    max_drawdown: float  # positive fraction of balance


def trade_stream(trades: list[Trade], times: pd.Series, genome_risk: float, risk_pct: float) -> pd.DataFrame:
    """Trades as (exit_time, ret), rescaled from the genome's risk to `risk_pct`."""
    return pd.DataFrame({
        "exit_time": [times.iloc[t.exit_i] for t in trades],
        "ret": [t.pnl_pct / genome_risk * risk_pct for t in trades],
    })


def stream_from_records(records: list[dict], risk_pct: float) -> pd.DataFrame:
    """Stored {"exit_time": iso, "r": R-multiple} records as (exit_time, ret)."""
    return pd.DataFrame({
        "exit_time": pd.to_datetime([r["exit_time"] for r in records]),
        "ret": [r["r"] * risk_pct for r in records],
    })


def prop_metrics(stream: pd.DataFrame) -> PropMetrics:
    if stream.empty:
        return PropMetrics(0.0, 0.0)
    s = stream.sort_values("exit_time", kind="stable")
    bal = np.cumprod(1.0 + s["ret"].to_numpy(dtype=float))
    peak = np.maximum.accumulate(np.concatenate(([1.0], bal)))[1:]
    max_dd = float(((peak - bal) / peak).max())

    opening = np.concatenate(([1.0], bal[:-1]))
    day = s["exit_time"].dt.normalize().to_numpy()
    # first trade's opening and last trade's close per calendar day
    frame = pd.DataFrame({"day": day, "open": opening, "close": bal})
    g = frame.groupby("day", sort=False).agg(open=("open", "first"), close=("close", "last"))
    worst = float(((g["open"] - g["close"]) / g["open"]).max())
    return PropMetrics(worst_day_loss=max(worst, 0.0), max_drawdown=max(max_dd, 0.0))


def combined_prop(streams: list[pd.DataFrame]) -> PropMetrics:
    return prop_metrics(pd.concat(streams, ignore_index=True).sort_values("exit_time", kind="stable"))


def breaches(m: PropMetrics, g: Guardrails) -> list[str]:
    out = []
    if m.worst_day_loss > g.daily_loss_pct:
        out.append(f"daily loss {m.worst_day_loss:.2%} exceeds {g.daily_loss_pct:.2%}")
    if m.max_drawdown > g.max_dd_pct:
        out.append(f"drawdown {m.max_drawdown:.2%} exceeds {g.max_dd_pct:.2%}")
    return out
