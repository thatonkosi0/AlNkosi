"""Portfolio analytics: pairwise equity-curve correlation and diversity grades.

Correlation is computed on per-bar equity returns, aligned on the common
tail (curves may have different lengths across symbols/timeframes). A
strategy's diversity grade reflects its average absolute correlation with
every other strategy in the selection — an "A" adds genuinely independent
behavior to the portfolio, an "F" duplicates exposure it already has.
"""

from __future__ import annotations

import numpy as np

MIN_OVERLAP = 10  # fewer overlapping return observations than this is noise
_GRADE_BANDS = ((0.20, "A"), (0.35, "B"), (0.50, "C"), (0.70, "D"))


def _grade(avg_abs_corr: float) -> str:
    for threshold, grade in _GRADE_BANDS:
        if avg_abs_corr <= threshold:
            return grade
    return "F"


def _returns(curve: list[float]) -> np.ndarray:
    eq = np.asarray(curve, dtype=np.float64)
    if len(eq) < 2:
        return np.empty(0)
    prev = np.where(eq[:-1] == 0.0, 1.0, eq[:-1])  # equity never touches zero
    return np.diff(eq) / prev


def correlation_matrix(curves: list[list[float]]) -> np.ndarray | None:
    rets = [_returns(c) for c in curves]
    overlap = min(len(r) for r in rets)
    if overlap < MIN_OVERLAP:
        return None
    aligned = np.vstack([r[-overlap:] for r in rets])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.corrcoef(aligned)
    m = np.nan_to_num(m, nan=0.0)  # zero-variance curves correlate with nothing
    np.fill_diagonal(m, 1.0)
    return m


def portfolio_report(rows: list[dict], curves: list[list[float]]) -> dict:
    """Correlation heatmap + per-strategy diversity grades.

    rows[i] describes the strategy whose OOS equity curve is curves[i];
    each needs at least id/name/symbol/tribe keys.
    """
    if len(rows) < 2:
        return {
            "strategies": [dict(r, avg_correlation=None, grade=None) for r in rows],
            "matrix": [],
            "note": "At least two vaulted strategies are needed for portfolio analysis.",
        }
    matrix = correlation_matrix(curves)
    if matrix is None:
        return {
            "strategies": [dict(r, avg_correlation=None, grade=None) for r in rows],
            "matrix": [],
            "note": "Equity curves overlap on too few bars to correlate.",
        }

    strategies = []
    n = len(rows)
    for i, row in enumerate(rows):
        others = np.abs(np.delete(matrix[i], i))
        avg = float(others.mean()) if n > 1 else 0.0
        strategies.append(dict(row, avg_correlation=avg, grade=_grade(avg)))
    return {"strategies": strategies, "matrix": matrix.tolist(), "note": None}
