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
    timeframes=["H1"],
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
    # picks up exactly where the checkpoint left off â€” no unit is redone
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
    two-tribe campaign â€” ~40s, and still only *probably* vaulting. Instead,
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
                # (genome, is_metrics, oos_metrics) â€” the campaign only reads
                # .to_dict()/.net_profit off these, so the top candidate's own
                # in-sample metrics stand in for both.
                result.vaulted = [
                    (result.top.genome, result.top.is_metrics, result.top.is_metrics)
                ]
            yield result

    monkeypatch.setattr(campaign_mod, "evolve_tribe", _evolve_tribe_with_one_vault)
    # vaulting now happens only after the holdout stage: pass it deterministically
    _fake_judge(monkeypatch)

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


def test_legacy_config_still_parses():
    cfg = CampaignConfig(
        symbols=["EURUSD"], timeframe="H1", tribes=["trend"], oos_fraction=0.25
    )
    assert cfg.timeframes == ["H1"]
    assert cfg.select_fraction == 0.25
    assert cfg.holdout_fraction == 0.2
    assert cfg.target_preset == "personal"


def test_config_rejects_bad_preset_and_timeframes():
    with pytest.raises(ValidationError):
        CampaignConfig(symbols=["EURUSD"], tribes=["trend"], target_preset="nope")
    with pytest.raises(ValidationError):
        CampaignConfig(symbols=["EURUSD"], tribes=["trend"], timeframes=["M3"])
    with pytest.raises(ValidationError):
        CampaignConfig(symbols=["EURUSD"], tribes=["trend"], timeframes=[])


def test_campaign_runs_every_timeframe(vault, cache):
    one = _run(TINY, vault, cache)[0]["total_units"]
    events = _run(dict(TINY, timeframes=["H1", "H4"]), vault, cache)
    assert events[0]["total_units"] == one * 2
    gens = [e for e in events if e["type"] == "generation"]
    assert len(gens) == one * 2
    tfs = [g["timeframe"] for g in gens]
    assert set(tfs) == {"H1", "H4"}
    assert tfs == sorted(tfs, key=["H1", "H4"].index)  # H1 units before H4
    for e in events:
        if e["type"] == "strategy_vaulted":
            assert e["timeframe"] in {"H1", "H4"}
    assert events[-1]["status"] == "done"


def test_resume_state_includes_timeframe():
    from alglory.evolve.campaign import _resume_state

    cfg = CampaignConfig(
        **dict(TINY, timeframes=["H1", "H4"], symbols=["EURUSD", "GBPUSD"])
    )
    ckpt = {"symbol": "EURUSD", "timeframe": "H4", "tribe": "trend", "gen": 1,
            "population": None}
    assert _resume_state(cfg, ckpt) == (0, 1, 0, 0, None)
    # last gen of last tribe at last timeframe rolls to the next symbol
    ckpt = {"symbol": "EURUSD", "timeframe": "H4", "tribe": "momentum", "gen": 3,
            "population": None}
    assert _resume_state(cfg, ckpt) == (1, 0, 0, 0, None)
    # old checkpoints without a timeframe resume at the first timeframe
    ckpt = {"symbol": "EURUSD", "tribe": "trend", "gen": 1, "population": None}
    assert _resume_state(cfg, ckpt)[1] == 0


# -- Task 6: finalists, once-per-market holdout, keep-best, skips, optimise --

import alglory.evolve.campaign as campaign_mod  # noqa: E402
import alglory.evolve.holdout as holdout_mod  # noqa: E402
from alglory.backtest.metrics import Metrics  # noqa: E402
from alglory.data.cache import InsufficientDataError  # noqa: E402
from alglory.evolve.holdout import Finalist, HoldoutResult  # noqa: E402


def _force_finalists(monkeypatch, spy=None):
    """Make each generation's top genome a select-passer, so campaign tests
    have finalists without depending on GA luck (Ruling B)."""
    real = campaign_mod.evolve_tribe

    def wrapped(*args, **kwargs):
        if spy is not None:
            spy.append((args, kwargs))
        for result in real(*args, **kwargs):
            result.vaulted = [(result.top.genome, result.top.is_metrics, result.top.is_metrics)]
            yield result

    monkeypatch.setattr(campaign_mod, "evolve_tribe", wrapped)


def _fake_judge(monkeypatch, passed=True):
    """Deterministic holdout verdicts; each call scores higher than the last."""
    calls = []

    def judge(f, holdout_bars, costs, gate, preset, bars_per_year):
        calls.append(f)
        win = 0.61 + 0.001 * len(calls)
        m = Metrics(net_profit=0.1, max_drawdown=0.05, profit_factor=1.5, sharpe=1.0,
                    win_rate=win, trades=60, avg_yearly_return=0.1)
        return HoldoutResult(finalist=f, metrics=m,
                             records=[{"exit_time": "2024-01-01T00:00:00", "r": 1.0}],
                             passed=passed, reasons=[] if passed else ["test says no"])

    monkeypatch.setattr(campaign_mod, "judge_finalist", judge)
    return calls


def _logs(events):
    return [e["line"] for e in events if e["type"] == "log"]


def _verdicts(events):
    return [line for line in _logs(events)
            if ": holdout pass" in line or ": holdout fail" in line]


def test_holdout_runs_once_per_market_after_all_units(vault, cache, monkeypatch):
    events = []
    evolve_calls = []
    _force_finalists(monkeypatch, spy=evolve_calls)
    # real judge_finalist; the pass decision only checks the trade floor
    monkeypatch.setattr(holdout_mod, "holdout_pass", lambda m, g: m.trades >= g.min_trades)
    real_judge = campaign_mod.judge_finalist
    judged = []

    def spy_judge(f, holdout_bars, *a, **kw):
        judged.append((len(events), f.key, f.timeframe, holdout_bars["time"].min()))
        return real_judge(f, holdout_bars, *a, **kw)

    monkeypatch.setattr(campaign_mod, "judge_finalist", spy_judge)
    cfg = CampaignConfig(**dict(TINY, timeframes=["H1", "H4"], min_holdout_trades=5))
    run_campaign(cfg, vault, cache, events.append, should_stop=lambda: False)
    assert events[-1]["status"] == "done"

    last_gen = max(i for i, e in enumerate(events) if e["type"] == "generation")
    vaulted_at = [i for i, e in enumerate(events) if e["type"] == "strategy_vaulted"]
    assert all(i > last_gen for i in vaulted_at)

    # every finalist judged exactly once, all after the market's last unit
    assert judged, "forced finalists must reach the holdout stage"
    keys = [k for _, k, _, _ in judged]
    assert len(keys) == len(set(keys))
    assert all(n > last_gen for n, _, _, _ in judged)
    assert len(_verdicts(events)) == len(judged)

    # the GA never saw a holdout bar
    holdout_start = {tf: t for _, _, tf, t in judged}
    for args, kwargs in evolve_calls:
        breed, select = args[1], args[2]
        tf = next(t for t, bpy in campaign_mod.BARS_PER_YEAR.items()
                  if bpy == kwargs["bars_per_year"])
        if tf in holdout_start:
            assert breed["time"].max() < holdout_start[tf]
            assert select["time"].max() < holdout_start[tf]

    for row in vault.list_strategies(limit=500):
        assert row["stage"] == "holdout_passed"
        assert row["ho_trades"] >= 5


def test_keep_per_market_respected(vault, cache, monkeypatch):
    _force_finalists(monkeypatch)
    calls = _fake_judge(monkeypatch)
    events = _run(dict(TINY, symbols=["EURUSD", "GBPUSD"], keep_per_market=1), vault, cache)
    assert events[-1]["status"] == "done"
    assert calls
    n_eur = sum(1 for line in _verdicts(events) if line.startswith("EURUSD"))
    best = {"EURUSD": 0.61 + 0.001 * n_eur, "GBPUSD": 0.61 + 0.001 * len(calls)}
    for sym in ("EURUSD", "GBPUSD"):
        rows = vault.list_strategies(symbol=sym, limit=500)
        assert len(rows) == 1  # forced passers exist, so exactly the cap
        assert rows[0]["stage"] == "holdout_passed"
        assert rows[0]["ho_win_rate"] == pytest.approx(best[sym])  # the best one kept
    # EURUSD is vaulted before GBPUSD starts breeding
    eur_vault = next(i for i, e in enumerate(events)
                     if e["type"] == "strategy_vaulted" and e["symbol"] == "EURUSD")
    gbp_gen = next(i for i, e in enumerate(events)
                   if e["type"] == "generation" and e["symbol"] == "GBPUSD")
    assert eur_vault < gbp_gen


def test_failed_holdout_vaults_nothing_and_logs_reasons(vault, cache, monkeypatch):
    _force_finalists(monkeypatch)
    _fake_judge(monkeypatch, passed=False)
    events = _run(TINY, vault, cache)
    assert events[-1]["status"] == "done"
    assert vault.count() == 0
    assert any("holdout fail" in line and "test says no" in line for line in _logs(events))


def test_insufficient_history_skips_unit_not_campaign(vault, cache, monkeypatch):
    real = campaign_mod._load_bars

    def load(cfg, symbol, timeframe, cache, emit):
        if timeframe == "H4":
            raise InsufficientDataError("Only 10 bars of history available")
        return real(cfg, symbol, timeframe, cache, emit)

    monkeypatch.setattr(campaign_mod, "_load_bars", load)
    events = _run(dict(TINY, timeframes=["H1", "H4"]), vault, cache)
    assert events[-1]["status"] == "done"
    gens = [e for e in events if e["type"] == "generation"]
    assert gens and {g["timeframe"] for g in gens} == {"H1"}
    assert any("EURUSD H4: skipped" in line for line in _logs(events))


def test_unknown_symbol_skips_market(vault, cache, monkeypatch):
    real = campaign_mod._load_bars

    def load(cfg, symbol, timeframe, cache, emit):
        if symbol == "NOPE":
            raise RuntimeError("MT5 returned no NOPE H1 bars")
        return real(cfg, symbol, timeframe, cache, emit)

    monkeypatch.setattr(campaign_mod, "_load_bars", load)
    events = _run(dict(TINY, symbols=["EURUSD", "NOPE"]), vault, cache)
    assert events[-1]["status"] == "done"
    assert any(e["symbol"] == "EURUSD" for e in events if e["type"] == "generation")
    assert any("NOPE: skipped" in line for line in _logs(events))


def _seed_strategy(vault):
    g = random_genome("trend", np.random.default_rng(0))
    m = {"net_profit": 0.2, "max_drawdown": 0.1, "profit_factor": 1.5,
         "sharpe": 1.0, "win_rate": 0.6, "trades": 40, "avg_yearly_return": 0.2}
    sid = vault.add_strategy(
        name=None, tribe="trend", symbol="EURUSD", timeframe="H1",
        genome_json=g.to_json(), is_metrics=m, oos_metrics=m,
        equity=np.linspace(1, 1.2, 30), campaign_id=None,
    )
    return sid, g


def test_optimise_seeds_population(vault, cache, monkeypatch):
    sid, g = _seed_strategy(vault)
    spy = []
    _force_finalists(monkeypatch, spy=spy)
    _run(dict(TINY, tribes=["trend"], seed_strategy_id=sid), vault, cache)
    initial = spy[0][1]["initial"]
    assert initial is not None
    assert g.to_json() in [x.to_json() for x in initial]


def test_optimise_results_flagged_reused(vault, cache, monkeypatch):
    sid, _ = _seed_strategy(vault)
    _force_finalists(monkeypatch)
    _fake_judge(monkeypatch)
    events = _run(dict(TINY, tribes=["trend"], seed_strategy_id=sid), vault, cache)
    assert events[-1]["status"] == "done"
    rows = [r for r in vault.list_strategies(limit=500) if r["id"] != sid]
    assert rows
    assert all(r["holdout_reused"] == 1 for r in rows)
    assert any("holdout reused" in line for line in _logs(events))


def test_second_campaign_on_same_window_flagged_reused(vault, cache, monkeypatch):
    _force_finalists(monkeypatch)
    _fake_judge(monkeypatch)
    first = _run(TINY, vault, cache)
    windows = json.loads(vault.get_campaign(first[0]["campaign_id"])["holdout_windows_json"])
    assert [(w["symbol"], w["timeframe"]) for w in windows] == [("EURUSD", "H1")]
    first_rows = vault.list_strategies(limit=500)
    assert first_rows and all(r["holdout_reused"] == 0 for r in first_rows)
    assert not any("holdout reused" in line for line in _logs(first))

    second = _run(TINY, vault, cache)
    first_ids = {r["id"] for r in first_rows}
    new = [r for r in vault.list_strategies(limit=500) if r["id"] not in first_ids]
    assert new and all(r["holdout_reused"] == 1 for r in new)
    assert any("holdout reused" in line for line in _logs(second))


def test_checkpoint_carries_finalists_and_holdout_done(vault, cache, monkeypatch):
    _force_finalists(monkeypatch)
    _fake_judge(monkeypatch)
    events = _run(dict(TINY, symbols=["EURUSD", "GBPUSD"]), vault, cache)
    ckpt = json.loads(vault.get_campaign(events[0]["campaign_id"])["progress_json"])
    assert ckpt["holdout_done"] == ["EURUSD", "GBPUSD"]
    # judged markets are pruned from the checkpoint (Ruling F.3)
    assert ckpt["finalists"] == {}


def test_checkpoint_carries_finalists_mid_market(vault, cache, monkeypatch):
    _force_finalists(monkeypatch)
    _fake_judge(monkeypatch)
    # cancel after EURUSD's first tribe: its finalists must be in the checkpoint
    events = _run(dict(TINY, symbols=["EURUSD", "GBPUSD"]), vault, cache, stop_after=4)
    assert events[-1]["status"] == "cancelled"
    ckpt = json.loads(vault.get_campaign(events[0]["campaign_id"])["progress_json"])
    assert ckpt["holdout_done"] == []
    assert set(ckpt["finalists"]) == {"EURUSD"}
    assert ckpt["finalists"]["EURUSD"]
    for d in ckpt["finalists"]["EURUSD"]:
        assert Finalist.from_json(d).timeframe == "H1"


def _resume(cfg_kwargs, vault, cache, cid):
    events = []
    run_campaign(CampaignConfig(**cfg_kwargs), vault, cache, events.append,
                 should_stop=lambda: False, resume_campaign_id=cid)
    return events


def _ckpt(vault, cid):
    return json.loads(vault.get_campaign(cid)["progress_json"])


def _assert_judged_once(calls):
    keys = [f.key for f in calls]
    assert keys and len(keys) == len(set(keys))


def test_cancel_resume_keeps_finalists_and_judges_once(vault, cache, monkeypatch):
    _force_finalists(monkeypatch)
    calls = _fake_judge(monkeypatch)
    cfg = dict(TINY, timeframes=["H1", "H4"])
    # 2 tribes x (1 pre-tribe + 3 per-gen) should_stop calls = 8 -> cancel after H1
    first = _run(cfg, vault, cache, stop_after=8)
    assert first[-1]["status"] == "cancelled"
    cid = first[0]["campaign_id"]
    assert not any(e["type"] == "strategy_vaulted" for e in first)
    assert not _verdicts(first)
    assert not any("finalists passed the holdout" in line for line in _logs(first))
    assert not calls
    ckpt = _ckpt(vault, cid)
    assert ckpt["timeframe"] == "H1" and ckpt["finalists"]["EURUSD"]
    h1_keys = {Finalist.from_json(d).key for d in ckpt["finalists"]["EURUSD"]}

    second = _resume(cfg, vault, cache, cid)
    assert second[-1]["status"] == "done"
    _assert_judged_once(calls)
    assert len(_verdicts(second)) == len(calls)
    judged = {f.key for f in calls}
    assert h1_keys <= judged  # finalists bred before the cancel are not lost
    assert {f.timeframe for f in calls} == {"H1", "H4"}
    ids = [e["strategy_id"] for e in second if e["type"] == "strategy_vaulted"]
    assert ids and len(ids) == len(set(ids))
    assert len(ids) <= CampaignConfig(**cfg).keep_per_market
    assert vault.count() == len(ids)


def test_cancel_after_market_last_generation_runs_its_holdout_on_resume(
    vault, cache, monkeypatch
):
    _force_finalists(monkeypatch)
    calls = _fake_judge(monkeypatch)
    cfg = dict(TINY, symbols=["EURUSD", "GBPUSD"])
    # the 8th should_stop call follows EURUSD's last persisted generation
    first = _run(cfg, vault, cache, stop_after=7)
    assert first[-1]["status"] == "cancelled"
    cid = first[0]["campaign_id"]
    ckpt = _ckpt(vault, cid)
    assert (ckpt["symbol"], ckpt["tribe"], ckpt["gen"]) == ("EURUSD", "momentum", 3)
    assert ckpt["holdout_done"] == []
    eur_keys = {Finalist.from_json(d).key for d in ckpt["finalists"]["EURUSD"]}
    assert eur_keys and not calls

    second = _resume(cfg, vault, cache, cid)
    assert second[-1]["status"] == "done"
    _assert_judged_once(calls)
    assert eur_keys <= {f.key for f in calls}
    eur_verdict = next(i for i, e in enumerate(second) if e["type"] == "log"
                       and e["line"].startswith("EURUSD H1") and "holdout" in e["line"])
    gbp_gen = next(i for i, e in enumerate(second)
                   if e["type"] == "generation" and e["symbol"] == "GBPUSD")
    assert eur_verdict < gbp_gen
    # EURUSD is not bred again
    assert not any(e["type"] == "generation" and e["symbol"] == "EURUSD" for e in second)
    final = _ckpt(vault, cid)
    assert final["holdout_done"] == ["EURUSD", "GBPUSD"]
    assert final["units_done"] == final["total_units"]


def test_crash_before_holdout_done_runs_holdout_on_resume(vault, cache, monkeypatch):
    _force_finalists(monkeypatch)
    calls = _fake_judge(monkeypatch)
    real_stage = campaign_mod._holdout_stage
    crashed = {"n": 0}

    def stage(*a, **kw):
        if not crashed["n"]:
            crashed["n"] += 1
            raise RuntimeError("simulated crash before the holdout stage")
        return real_stage(*a, **kw)

    monkeypatch.setattr(campaign_mod, "_holdout_stage", stage)
    first = _run(TINY, vault, cache)
    assert first[-1]["status"] == "failed"
    cid = first[0]["campaign_id"]
    assert _ckpt(vault, cid)["holdout_done"] == []
    assert not calls

    second = _resume(TINY, vault, cache, cid)
    assert second[-1]["status"] == "done"
    _assert_judged_once(calls)
    assert len(_verdicts(second)) == len(calls)
    assert _ckpt(vault, cid)["holdout_done"] == ["EURUSD"]
    assert vault.count() >= 1


def test_crash_mid_holdout_stage_no_duplicate_windows_or_strategies(
    vault, cache, monkeypatch
):
    _force_finalists(monkeypatch)
    _fake_judge(monkeypatch)
    real_add = vault.add_strategy
    adds = {"n": 0}

    def add(*a, **kw):
        adds["n"] += 1
        if adds["n"] == 2:
            raise RuntimeError("simulated crash while vaulting")
        return real_add(*a, **kw)

    monkeypatch.setattr(vault, "add_strategy", add)
    first = _run(TINY, vault, cache)
    assert first[-1]["status"] == "failed"
    cid = first[0]["campaign_id"]
    assert vault.count() == 1
    windows_before = json.loads(vault.get_campaign(cid)["holdout_windows_json"])
    assert len(windows_before) == 1

    second = _resume(TINY, vault, cache, cid)
    assert second[-1]["status"] == "done"
    windows = json.loads(vault.get_campaign(cid)["holdout_windows_json"])
    assert windows == windows_before  # no duplicate window on resume (Ruling F.4)
    rows = vault.list_strategies(limit=500)
    jsons = [vault.get_strategy(r["id"])["genome_json"] for r in rows]
    assert len(jsons) == len(set(jsons))  # nothing vaulted twice
    assert 1 < len(rows) <= 3
    assert second[-1]["vaulted_total"] == len(rows)


def test_resume_when_bred_timeframe_no_longer_loads(vault, cache, monkeypatch):
    _force_finalists(monkeypatch)
    calls = _fake_judge(monkeypatch)
    cfg = dict(TINY, timeframes=["H1", "H4"])
    first = _run(cfg, vault, cache, stop_after=8)  # cancel after H1
    cid = first[0]["campaign_id"]
    assert _ckpt(vault, cid)["finalists"]["EURUSD"]

    real = campaign_mod._load_bars

    def load(cfg_, symbol, timeframe, cache_, emit):
        if timeframe == "H1":
            raise InsufficientDataError("Only 10 bars of history available")
        return real(cfg_, symbol, timeframe, cache_, emit)

    monkeypatch.setattr(campaign_mod, "_load_bars", load)
    second = _resume(cfg, vault, cache, cid)
    assert second[-1]["status"] == "done"
    assert "EURUSD H1: holdout skipped — no data" in _logs(second)
    assert calls and {f.timeframe for f in calls} == {"H4"}
    _assert_judged_once(calls)
    final = _ckpt(vault, cid)
    assert final["units_done"] == final["total_units"]  # H1 not counted twice
    gens = [e for e in second if e["type"] == "generation"]
    assert gens[-1]["units_done"] == final["total_units"]


def test_resume_when_market_skips_counts_units_once_and_names_error(
    vault, cache, monkeypatch
):
    _force_finalists(monkeypatch)
    calls = _fake_judge(monkeypatch)
    cfg = dict(TINY, timeframes=["H1", "H4"])
    first = _run(cfg, vault, cache, stop_after=8)  # cancel after H1
    cid = first[0]["campaign_id"]

    def load(*a, **kw):
        raise KeyError("boom")  # a programming error, not a broker answer

    monkeypatch.setattr(campaign_mod, "_load_bars", load)
    second = _resume(cfg, vault, cache, cid)
    assert second[-1]["status"] == "done"
    logs = _logs(second)
    assert any(line.startswith("EURUSD: skipped") and "KeyError" in line for line in logs)
    assert "EURUSD H1: holdout skipped — no data" in logs
    assert not calls
    final = _ckpt(vault, cid)
    assert final["units_done"] == final["total_units"]
    assert final["holdout_done"] == ["EURUSD"]
    assert final["finalists"] == {}
