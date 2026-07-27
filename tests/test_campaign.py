import json

import pytest

from alglory.data.cache import BarCache
from alglory.evolve.campaign import CampaignConfig, run_campaign
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
