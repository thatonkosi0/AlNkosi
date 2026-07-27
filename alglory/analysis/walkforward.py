"""Walk-forward matrix: how a fixed genome holds up across time segments.

The vaulted genome is frozen (no re-optimization), so this is a robustness
matrix: history is cut into N contiguous segments for several values of N,
the strategy is backtested on each segment independently, and the fraction
of profitable segments drives a letter grade. A strategy that only made its
money in one lucky stretch fails quickly here.
"""

from __future__ import annotations

import pandas as pd

from alglory.backtest.engine import Costs, run_backtest
from alglory.backtest.metrics import compute_metrics
from alglory.data.cache import InsufficientDataError
from alglory.genome import Genome

MIN_SEGMENT_BARS = 300  # indicators need warmup; shorter segments are noise
DEFAULT_SEGMENT_COUNTS = (4, 6, 8)
_GRADE_BANDS = ((0.80, "A"), (0.65, "B"), (0.50, "C"), (0.35, "D"))


def _grade(pass_rate: float) -> str:
    for threshold, grade in _GRADE_BANDS:
        if pass_rate >= threshold:
            return grade
    return "F"


def walk_forward_matrix(
    genome: Genome,
    bars: pd.DataFrame,
    costs: Costs,
    bars_per_year: float,
    segment_counts: tuple[int, ...] = DEFAULT_SEGMENT_COUNTS,
) -> dict:
    max_segments = max(segment_counts)
    if len(bars) // max_segments < MIN_SEGMENT_BARS:
        raise InsufficientDataError(
            f"Walk-forward with {max_segments} segments needs at least "
            f"{max_segments * MIN_SEGMENT_BARS} bars; only {len(bars)} available."
        )

    rows = []
    total = passed = 0
    for n in segment_counts:
        seg_len = len(bars) // n
        cells = []
        for i in range(n):
            seg = bars.iloc[i * seg_len : (i + 1) * seg_len].reset_index(drop=True)
            m = compute_metrics(run_backtest(genome, seg, costs), bars_per_year)
            ok = m.net_profit > 0.0
            cells.append(
                {
                    "net_profit": m.net_profit,
                    "profit_factor": m.profit_factor,
                    "max_drawdown": m.max_drawdown,
                    "trades": m.trades,
                    "pass": ok,
                }
            )
            total += 1
            passed += ok
        rows.append(
            {"segments": n, "cells": cells, "profitable": sum(c["pass"] for c in cells)}
        )

    pass_rate = passed / total if total else 0.0
    return {"rows": rows, "pass_rate": pass_rate, "grade": _grade(pass_rate)}
