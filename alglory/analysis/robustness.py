"""Robustness scoring — surface strategies that hold up, not lucky outliers.

The campaign selects by an OOS slice across many candidates, which cherry-picks
overfit outliers: the top-by-OOS-net-profit are frequently full-history losers
(verified — see the vault-ranking-overfit finding). This re-checks a frozen
genome on the whole cached history and across walk-forward segments, then
collapses the result into ONE sortable score so the vault can rank trustworthy
strategies first.

score = walk_forward_pass_rate * (1 + full_history_net), halved when the
strategy is degenerately hyperactive (bleeds on spread). Higher is better;
unscored strategies sort last.
"""

from __future__ import annotations

import pandas as pd

from alglory.analysis.activity import trade_activity
from alglory.backtest.engine import Costs, run_backtest
from alglory.backtest.metrics import compute_metrics
from alglory.analysis.walkforward import walk_forward_matrix
from alglory.data.cache import InsufficientDataError
from alglory.genome import Genome

HYPERACTIVE_TRADES_PER_WEEK = 20.0


def _score(pass_rate: float, net: float, penalise: bool) -> float:
    score = pass_rate * (1.0 + max(net, -0.99))
    if penalise:
        score *= 0.5
    return round(float(score), 4)


def robustness(
    genome: Genome,
    bars: pd.DataFrame,
    costs: Costs,
    bars_per_year: float,
    *,
    timeframe: str | None = None,
) -> dict:
    """Return {grade, score, pass_rate, full_hist_net, trades, walk_forward}.

    Falls back to a half-weight, full-history-only score when there aren't
    enough bars for a walk-forward matrix (so short-history symbols still rank).
    """
    result = run_backtest(genome, bars, costs)
    m = compute_metrics(result, bars_per_year)
    net = m.net_profit

    tpw = None
    if timeframe is not None:
        try:
            tpw = trade_activity(result, bars, timeframe)["trades_per_week"]
        except (ValueError, KeyError):
            tpw = None
    hyperactive = tpw is not None and tpw > HYPERACTIVE_TRADES_PER_WEEK

    try:
        wf = walk_forward_matrix(genome, bars, costs, bars_per_year)
        return {
            "grade": wf["grade"],
            "score": _score(wf["pass_rate"], net, hyperactive),
            "pass_rate": wf["pass_rate"],
            "full_hist_net": net,
            "trades": m.trades,
            "walk_forward": True,
        }
    except InsufficientDataError:
        # not enough history to prove time-robustness — score at half weight
        provisional = "C" if net > 0 else "F"
        return {
            "grade": provisional,
            "score": _score(0.5, net, hyperactive),
            "pass_rate": None,
            "full_hist_net": net,
            "trades": m.trades,
            "walk_forward": False,
        }


def rescore_vault(vault, cache, symbol: str, timeframe: str, bars_per_year: float) -> dict:
    """Compute + persist robustness for every vaulted strategy of one symbol/tf.

    Loads the cached bars once and scores each genome against them. Returns
    {scored, skipped} — skipped are strategies with no cached bars to score on.
    """
    bars = cache.load(symbol, timeframe)
    if bars is None or len(bars) < 50:
        return {"scored": 0, "skipped": 0, "reason": "no cached bars for this symbol/timeframe"}
    costs = Costs(spread=0.0001)
    rows = vault.list_strategies(symbol=symbol, timeframe=timeframe, limit=100_000)
    scored = 0
    for row in rows:
        full = vault.get_strategy(row["id"])
        genome = Genome.from_json(full["genome_json"])
        r = robustness(genome, bars, costs, bars_per_year, timeframe=timeframe)
        vault.set_robustness(row["id"], r["grade"], r["score"])
        scored += 1
    return {"scored": scored, "skipped": 0}
