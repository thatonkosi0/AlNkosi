"""Chronological three-way split: breed / select / holdout."""

from __future__ import annotations

import pandas as pd


def split_bars(
    bars: pd.DataFrame, select_fraction: float = 0.2, holdout_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split bars in time order into (breed, select, holdout)."""
    if select_fraction <= 0 or holdout_fraction <= 0:
        raise ValueError("select_fraction and holdout_fraction must be positive")
    if select_fraction + holdout_fraction >= 0.9:
        raise ValueError("select_fraction + holdout_fraction must be below 0.9")
    n = len(bars)
    holdout_start = n - int(round(n * holdout_fraction))
    select_start = holdout_start - int(round(n * select_fraction))
    return (
        bars.iloc[:select_start].reset_index(drop=True),
        bars.iloc[select_start:holdout_start].reset_index(drop=True),
        bars.iloc[holdout_start:].reset_index(drop=True),
    )
