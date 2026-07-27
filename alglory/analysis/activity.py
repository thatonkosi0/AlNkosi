"""Trade-frequency diagnostics: is "no trades" a bug or just a quiet strategy?

A vaulted genome is frozen and replayed over its own history; this reports how
OFTEN it actually opens a position, translated into human units (trades per
week/month and the typical calendar gap between entries). Many bred strategies
fire only a handful of times a year — an MA/MACD *cross* that must also agree
with a trend filter is a rare event — so seeing zero fills over a short live
session is expected behaviour, not a defect. This module lets you tell the two
apart with a number instead of a guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alglory.backtest.engine import BacktestResult, Costs, run_backtest
from alglory.genome import Genome

# how many bars each timeframe advances in a ~5-day trading week
_BARS_PER_WEEK = {"M15": 480.0, "H1": 120.0, "H4": 30.0, "D1": 5.0}
_WEEKS_PER_MONTH = 4.345


def trade_activity(result: BacktestResult, bars: pd.DataFrame, timeframe: str) -> dict:
    """Summarise how frequently ``result`` opened positions over ``bars``."""
    if timeframe not in _BARS_PER_WEEK:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; expected one of {sorted(_BARS_PER_WEEK)}"
        )
    per_week = _BARS_PER_WEEK[timeframe]
    bars_per_day = per_week / 5.0
    n_bars = len(bars)
    trades = result.trades
    n_trades = len(trades)

    weeks = n_bars / per_week
    trades_per_week = n_trades / weeks if weeks > 0 else 0.0

    entries = [t.entry_i for t in trades]
    gaps = np.diff(entries) if len(entries) > 1 else np.array([])
    median_gap_bars = float(np.median(gaps)) if gaps.size else float("inf")

    return {
        "trades": n_trades,
        "bars": n_bars,
        "trades_per_week": trades_per_week,
        "trades_per_month": trades_per_week * _WEEKS_PER_MONTH,
        "bars_per_trade": (n_bars / n_trades) if n_trades else float("inf"),
        "median_gap_bars": median_gap_bars,
        "median_gap_days": (median_gap_bars / bars_per_day) if gaps.size else float("inf"),
        "verdict": _verdict(n_trades, trades_per_week),
    }


def _verdict(n_trades: int, trades_per_week: float) -> str:
    """Plain-language read on whether a live no-fill is alarming."""
    if n_trades == 0:
        return (
            "NEVER TRADES: the strategy took no positions even across its full "
            "history — this is a real problem (over-tight filters or a broken "
            "signal), not just low activity."
        )
    if trades_per_week < 0.25:
        return (
            "VERY LOW ACTIVITY: fewer than ~1 trade a month; zero fills over days "
            "or a couple of weeks live is completely expected."
        )
    if trades_per_week < 1.0:
        return (
            "LOW ACTIVITY: roughly weekly-to-monthly; a quiet live session with no "
            "fills is normal."
        )
    return "ACTIVE: at least ~1 trade/week; a totally silent EA here is worth investigating."


def activity_for_genome(
    genome: Genome, bars: pd.DataFrame, costs: Costs, timeframe: str
) -> dict:
    """Convenience wrapper: backtest ``genome`` then summarise its activity."""
    return trade_activity(run_backtest(genome, bars, costs), bars, timeframe)
