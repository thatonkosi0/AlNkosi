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


def test_campaign_config_validation():
    with pytest.raises(Exception):
        CampaignConfig(symbols=[], timeframe="H1", tribes=["trend"])
    with pytest.raises(Exception):
        CampaignConfig(symbols=["EURUSD"], timeframe="H1", tribes=["not_a_tribe"])
    with pytest.raises(Exception):
        CampaignConfig(symbols=["EURUSD"], timeframe="M3", tribes=["trend"])
