"""Robustness scoring + persisted vault ranking."""

from __future__ import annotations

import numpy as np

from alglory.analysis.robustness import _score, rescore_vault, robustness
from alglory.backtest.engine import Costs
from alglory.data.sample import generate_bars
from alglory.genome import (
    FilterGene,
    Genome,
    ManagementGene,
    RiskGene,
    SignalGene,
    random_genome,
)
from alglory.vault.db import Vault

BPY = 6240.0
COSTS = Costs(spread=0.0001)


class _Cache:
    def __init__(self, bars):
        self._bars = bars

    def load(self, symbol, timeframe):
        return self._bars


def _genome():
    return Genome(
        "trend", "both",
        SignalGene("ma_cross", {"fast": 10, "slow": 40}),
        FilterGene(trend_ma=None, session=None, atr_regime=None),
        ManagementGene(sl_atr=1.5, tp_atr=3.0, trail_atr=None, max_bars=None),
        RiskGene(0.01),
    )


# ---- score function --------------------------------------------------


def test_score_rewards_winners_and_robustness():
    # higher pass rate and higher net -> higher score
    assert _score(1.0, 0.2, False) > _score(0.5, 0.2, False)
    assert _score(1.0, 0.2, False) > _score(1.0, -0.1, False)


def test_hyperactivity_halves_the_score():
    assert _score(1.0, 0.2, True) == _score(1.0, 0.2, False) / 2


# ---- robustness() ----------------------------------------------------


def test_robustness_full_walk_forward():
    bars = generate_bars("EURUSD", "H1", 3000, seed=7)
    r = robustness(_genome(), bars, COSTS, BPY, timeframe="H1")
    assert r["walk_forward"] is True
    assert r["grade"] in {"A", "B", "C", "D", "F"}
    assert isinstance(r["score"], float)
    assert "full_hist_net" in r and "trades" in r


def test_robustness_falls_back_when_history_too_short():
    bars = generate_bars("EURUSD", "H1", 400, seed=7)  # < 8*300 needed for WF
    r = robustness(_genome(), bars, COSTS, BPY, timeframe="H1")
    assert r["walk_forward"] is False
    assert r["pass_rate"] is None
    assert r["grade"] in {"C", "F"}


# ---- rescore_vault + persisted ranking -------------------------------


def _seed(vault, genome, symbol="EURUSD", tf="H1"):
    m = {"net_profit": 0.2, "max_drawdown": 0.1, "profit_factor": 1.5,
         "sharpe": 1.0, "win_rate": 0.5, "trades": 40, "avg_yearly_return": 0.2}
    return vault.add_strategy(
        name=None, tribe="trend", symbol=symbol, timeframe=tf,
        genome_json=genome.to_json(), is_metrics=m, oos_metrics=m,
        equity=np.linspace(1, 1.2, 30), campaign_id=None,
    )


def test_rescore_persists_and_sorts(tmp_path):
    vault = Vault(tmp_path / "v.db")
    bars = generate_bars("EURUSD", "H1", 3000, seed=7)
    ids = [_seed(vault, random_genome("trend", np.random.default_rng(s))) for s in range(3)]

    out = rescore_vault(vault, _Cache(bars), "EURUSD", "H1", BPY)
    assert out["scored"] == 3

    ranked = vault.list_strategies(symbol="EURUSD", timeframe="H1", sort="robustness_score")
    scores = [r["robustness_score"] for r in ranked]
    assert all(s is not None for s in scores)
    assert scores == sorted(scores, reverse=True)  # descending by robustness
    assert all(r["robustness_grade"] in {"A", "B", "C", "D", "F"} for r in ranked)


def test_unscored_strategies_sort_last(tmp_path):
    vault = Vault(tmp_path / "v.db")
    bars = generate_bars("EURUSD", "H1", 3000, seed=7)
    scored_id = _seed(vault, _genome())
    rescore_vault(vault, _Cache(bars), "EURUSD", "H1", BPY)
    unscored_id = _seed(vault, _genome())  # added after rescore -> NULL score

    ranked = vault.list_strategies(symbol="EURUSD", timeframe="H1", sort="robustness_score")
    assert ranked[0]["id"] == scored_id
    assert ranked[-1]["id"] == unscored_id
    assert ranked[-1]["robustness_score"] is None
