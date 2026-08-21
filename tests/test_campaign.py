import json

import numpy as np
import pytest
from pydantic import ValidationError

from alglory.data.cache import BarCache
from alglory.evolve.campaign import CampaignConfig, run_campaign
from alglory.genome import random_genome
from alglory.vault.db import Vault


@pytest.fixture
def vault(tmp_path):
    return Vault(tmp_path / "vault.db")


@pytest.fixture
def cache(tmp_path):
    return BarCache(tmp_path)


TINY = dict(
    symbols=["EURUSD"],
    timeframe="H1",
    tribes=["trend", "momentum"],
    population=8,
    generations=3,
    seed=42,
    source="sample",
    min_trades=5,
)


def _run(cfg_kwargs, vault, cache, stop_after=None):
    events = []
    stopped = {"n": 0}

    def emit(e):
        events.append(e)

    def should_stop():
        stopped["n"] += 1
        return stop_after is not None and stopped["n"] > stop_after

    run_campaign(CampaignConfig(**cfg_kwargs), vault, cache, emit, should_stop)
    return events


def test_campaign_event_protocol_order(vault, cache):
    events = _run(TINY, vault, cache)
    types = [e["type"] for e in events]
    assert types[0] == "campaign_started"
    assert types[-1] == "campaign_finished"
    assert "log" in types
    gens = [e for e in events if e["type"] == "generation"]
    # 2 tribes x 3 generations
    assert len(gens) == 6
    for g in gens:
        assert set(g) >= {
            "type", "campaign_id", "symbol", "tribe", "gen", "of",
            "top_fitness", "survivors", "culled", "population",
            "top_equity", "vaulted_count_total",
        }
        assert g["population"] == 8
        for curve in g["top_equity"]:
            assert len(curve["points"]) <= 200
    finished = events[-1]
    assert finished["status"] == "done"
    assert finished["error"] is None


def test_campaign_persists_and_vaults(vault, cache):
    events = _run(TINY, vault, cache)
    started = events[0]
    row = vault.get_campaign(started["campaign_id"])
    assert row["status"] == "done"
    vaulted_events = [e for e in events if e["type"] == "strategy_vaulted"]
    assert vault.count() == len(vaulted_events)
    if vault.count():
        top = vault.list_strategies()[0]
        full = vault.get_strategy(top["id"])
        assert full["oos_metrics"]["net_profit"] > 0
        assert full["symbol"] == "EURUSD"


def test_campaign_scores_robustness_at_insert(vault, cache):
    events = _run(TINY, vault, cache)
    vaulted = [e for e in events if e["type"] == "strategy_vaulted"]
    for e in vaulted:
        assert "robustness_grade" in e  # event always carries the grade slot
    if vault.count():
        for row in vault.list_strategies():
            assert row["robustness_score"] is not None  # scored at insert, no manual rescore
            assert row["robustness_grade"] in {"A", "B", "C", "D", "F"}


def test_campaign_cancellation(vault, cache):
    events = _run(TINY, vault, cache, stop_after=1)
    finished = events[-1]
    assert finished["type"] == "campaign_finished"
    assert finished["status"] == "cancelled"
    row = vault.get_campaign(events[0]["campaign_id"])
    assert row["status"] == "cancelled"


def test_campaign_mt5_unavailable_fails_clearly(vault, cache, monkeypatch):
    # Hermetic: force the unavailable path even on machines where the
    # MetaTrader5 package and a live terminal are actually present.
    from alglory.data.mt5source import ConnStatus, MT5Source

    monkeypatch.setattr(
        MT5Source,
        "connect",
        lambda self: ConnStatus(ok=False, message="MetaTrader5 unavailable (test)"),
    )
    cfg = dict(TINY, source="mt5")
    events = _run(cfg, vault, cache)
    finished = events[-1]
    assert finished["status"] == "failed"
    assert "MetaTrader5" in finished["error"] or "MT5" in finished["error"]
    row = vault.get_campaign(events[0]["campaign_id"])
    assert row["status"] == "failed"


def test_campaign_pause_blocks_then_resumes(vault, cache):
    events = []
    # report "paused" for the first few polls, then let the campaign run on
    pause_polls = {"n": 0}

    def should_pause():
        pause_polls["n"] += 1
        return pause_polls["n"] <= 3

    run_campaign(
        CampaignConfig(**TINY),
        vault,
        cache,
        emit=events.append,
        should_stop=lambda: False,
        should_pause=should_pause,
    )
    lines = [e["line"] for e in events if e["type"] == "log"]
    assert any("paused" in line for line in lines)
    assert any("resumed" in line for line in lines)
    assert events[-1]["status"] == "done"


def test_campaign_cancel_while_paused(vault, cache):
    events = []
    stop = {"now": False}

    def should_pause():
        stop["now"] = True  # cancel arrives while paused
        return True

    run_campaign(
        CampaignConfig(**TINY),
        vault,
        cache,
        emit=events.append,
        should_stop=lambda: stop["now"],
        should_pause=should_pause,
    )
    assert events[-1]["status"] == "cancelled"


def test_campaign_resumes_from_checkpoint(vault, cache):
    events = _run(TINY, vault, cache, stop_after=1)
    assert events[-1]["status"] == "cancelled"
    cid = events[0]["campaign_id"]
    ckpt = json.loads(vault.get_campaign(cid)["progress_json"])
    assert ckpt["population"], "checkpoint must carry the next generation's population"
    units_before = ckpt["units_done"]
    assert units_before >= 1

    events2 = []
    run_campaign(
        CampaignConfig(**TINY),
        vault,
        cache,
        emit=events2.append,
        should_stop=lambda: False,
        resume_campaign_id=cid,
    )
    started = events2[0]
    assert started["type"] == "campaign_started"
    assert started["resumed"] is True
    assert started["campaign_id"] == cid
    assert events2[-1]["status"] == "done"
    assert vault.get_campaign(cid)["status"] == "done"

    gens = [e for e in events2 if e["type"] == "generation"]
    total = started["total_units"]
    # picks up exactly where the checkpoint left off — no unit is redone
    assert len(gens) == total - units_before
    assert gens[0]["units_done"] == units_before + 1
    assert gens[-1]["units_done"] == total

    # the resumed run never re-vaults a strategy the first run already stored
    jsons = [
        vault.get_strategy(r["id"])["genome_json"] for r in vault.list_strategies(limit=500)
    ]
    assert len(jsons) == len(set(jsons))


def test_campaign_resume_requires_checkpoint(vault, cache):
    cid = vault.create_campaign(CampaignConfig(**TINY).model_dump_json())
    with pytest.raises(ValueError, match="checkpoint"):
        run_campaign(
            CampaignConfig(**TINY),
            vault,
            cache,
            emit=lambda e: None,
            should_stop=lambda: False,
            resume_campaign_id=cid,
        )


def test_campaign_config_validation():
    with pytest.raises(Exception):
        CampaignConfig(symbols=[], timeframe="H1", tribes=["trend"])
    with pytest.raises(Exception):
        CampaignConfig(symbols=["EURUSD"], timeframe="H1", tribes=["not_a_tribe"])
    with pytest.raises(Exception):
        CampaignConfig(symbols=["EURUSD"], timeframe="M3", tribes=["trend"])


def test_campaign_rejects_invalid_symbol_characters():
    with pytest.raises(ValidationError):
        CampaignConfig(symbols=["../../etc"], timeframe="H1", tribes=["trend"])
    with pytest.raises(ValidationError):
        CampaignConfig(symbols=['EUR"USD'], timeframe="H1", tribes=["trend"])


def test_robustness_scoring_failure_is_logged(monkeypatch, vault, cache):
    """A robustness scorer that raises must be logged, not abort the campaign.

    Reaching that try/except needs at least one strategy to vault. Whether a
    real GA run vaults anything is non-deterministic (OOS gating on evolved
    genomes), so this used to be forced with a 40-population/15-generation
    two-tribe campaign — ~40s, and still only *probably* vaulting. Instead,
    inject exactly one vaulted genome into the first generation's result: the
    vaulting path then runs deterministically and the GA stays tiny.
    """
    import alglory.evolve.campaign as campaign_mod

    def _boom(*a, **kw):
        raise RuntimeError("scorer exploded")

    # Get the actual module by importing with __import__ to avoid name collision
    robustness_module = __import__("alglory.analysis.robustness", fromlist=["robustness"])

    monkeypatch.setattr(robustness_module, "robustness", _boom)

    real_evolve_tribe = campaign_mod.evolve_tribe
    injected = []

    def _evolve_tribe_with_one_vault(*args, **kwargs):
        for result in real_evolve_tribe(*args, **kwargs):
            if not injected:
                injected.append(result.top.genome)
                # (genome, is_metrics, oos_metrics) — the campaign only reads
                # .to_dict()/.net_profit off these, so the top candidate's own
                # in-sample metrics stand in for both.
                result.vaulted = [
                    (result.top.genome, result.top.is_metrics, result.top.is_metrics)
                ]
            yield result

    monkeypatch.setattr(campaign_mod, "evolve_tribe", _evolve_tribe_with_one_vault)

    events = []
    cfg = CampaignConfig(
        symbols=["EURUSD"],
        tribes=["trend"],
        population=6,
        generations=2,
        seed=42,
        timeframe="H1",
        source="sample",
        min_trades=5,
    )
    run_campaign(cfg, vault, cache, events.append, should_stop=lambda: False)

    # the premise: the vaulting path ran. The injection guarantees it; the real
    # GA may or may not additionally vault one of its own (that non-determinism
    # is exactly what this test no longer depends on), so assert >= 1.
    assert len(injected) == 1
    assert len([e for e in events if e["type"] == "strategy_vaulted"]) >= 1

    logs = [e["line"] for e in events if e["type"] == "log"]
    assert any("robustness scoring failed" in line for line in logs)
    # and the failure did not abort the campaign
    finished = next(e for e in events if e["type"] == "campaign_finished")
    assert finished["status"] == "done"


def test_optimize_fails_loudly_when_seed_tribe_not_selected(vault, cache):
    g = random_genome("trend", np.random.default_rng(0))
    m = {"net_profit": 0.2, "max_drawdown": 0.1, "profit_factor": 1.5,
         "sharpe": 1.0, "win_rate": 0.5, "trades": 40, "avg_yearly_return": 0.2}
    sid = vault.add_strategy(
        name=None, tribe="trend", symbol="EURUSD", timeframe="H1",
        genome_json=g.to_json(), is_metrics=m, oos_metrics=m,
        equity=np.linspace(1, 1.2, 30), campaign_id=None,
    )
    cfg = CampaignConfig(
        symbols=["EURUSD"], tribes=["momentum"], population=8, generations=2,
        seed_strategy_id=sid,
    )
    events = []
    run_campaign(cfg, vault, cache, events.append, should_stop=lambda: False)
    finished = next(e for e in events if e["type"] == "campaign_finished")
    assert finished["status"] == "failed"
    assert "not in the selected tribes" in (finished["error"] or "")
