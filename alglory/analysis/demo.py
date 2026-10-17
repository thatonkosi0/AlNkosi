"""Demo-account results: per-strategy stats, account-level prop check, stage rules."""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from alglory.analysis.prop import PropMetrics

MAGIC_LO, MAGIC_HI = 990_000, 999_999
MIN_DAYS = 21
MIN_TRADES = 20
MIN_WIN_RATE = 0.60  # must be exceeded


def _net(deals: pd.DataFrame) -> pd.Series:
    return deals["profit"] + deals["commission"] + deals["swap"]


def demo_stats(deals: pd.DataFrame, magic: int) -> dict:
    """A trade's result is its out-deal's profit+commission+swap plus the
    commission+swap of the in-deal that opened it (most recent unmatched in)."""
    mine = deals[deals["magic"] == magic].sort_values("time", kind="stable")
    results: list[float] = []
    open_cost: float | None = None
    for row in mine.itertuples():
        cost = float(row.commission) + float(row.swap)
        if row.entry == "in":
            open_cost = cost
        elif row.entry == "out":
            results.append(float(row.profit) + cost + (open_cost or 0.0))
            open_cost = None
    if not results:
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "net": 0.0}
    trades, wins = len(results), sum(r > 0 for r in results)
    return {"trades": trades, "wins": wins, "win_rate": wins / trades, "net": float(sum(results))}


def account_prop(deals: pd.DataFrame, current_balance: float) -> tuple[PropMetrics, dict[date, float]]:
    """Rebuild the balance path backwards from ``current_balance`` using every
    deal, then measure the Alglory-magic P&L against it."""
    if deals.empty:
        return PropMetrics(0.0, 0.0), {}
    d = deals.sort_values("time", kind="stable").reset_index(drop=True)
    net = _net(d)
    after = current_balance - (net.sum() - net.cumsum())  # balance after each deal
    before = after - net
    d = d.assign(net=net, before=before, day=pd.to_datetime(d["time"]).dt.date)
    alg = d["magic"].between(MAGIC_LO, MAGIC_HI)

    day_loss: dict[date, float] = {}
    for day, g in d.groupby("day", sort=True):
        pnl = float(g.loc[alg.loc[g.index], "net"].sum())
        opening = float(g["before"].iloc[0])
        day_loss[day] = max(-pnl / opening, 0.0) if opening > 0 else 0.0

    equity = float(d["before"].iloc[0]) + d["net"].where(alg, 0.0).cumsum()
    peak = equity.cummax().clip(lower=float(d["before"].iloc[0]))
    dd = float(((peak - equity) / peak).max())
    return PropMetrics(max(day_loss.values()), max(dd, 0.0)), day_loss


def decide_stage(stats: dict, started_at: datetime, now: datetime, breached: bool) -> tuple[str, str | None]:
    if breached:
        return "demo_failed", "prop limit breached"
    if (now - started_at).days < MIN_DAYS:
        return "on_demo", None
    if stats["trades"] < MIN_TRADES:
        return "on_demo", "needs more trades"
    failed = []
    if not stats["win_rate"] > MIN_WIN_RATE:
        failed.append(f"win rate {stats['win_rate']:.0%} not above {MIN_WIN_RATE:.0%}")
    if not stats["net"] > 0:
        failed.append(f"net {stats['net']:.2f} not positive")
    if failed:
        return "demo_failed", "; ".join(failed)
    return "demo_passed", None
