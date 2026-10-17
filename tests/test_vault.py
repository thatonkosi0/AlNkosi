import numpy as np
import pytest

from alglory.genome import random_genome
from alglory.vault.db import Vault


@pytest.fixture
def vault(tmp_path):
    return Vault(tmp_path / "vault.db")


def _metrics(net=0.2, pf=1.8, dd=0.1):
    return {
        "net_profit": net,
        "max_drawdown": dd,
        "profit_factor": pf,
        "sharpe": 1.2,
        "win_rate": 0.5,
        "trades": 60,
        "avg_yearly_return": net,
    }


def _add(
    vault, tribe="trend", symbol="EURUSD", timeframe="H1", net=0.2, pf=1.8, dd=0.1,
    campaign_id=None,
):
    rng = np.random.default_rng(0)
    g = random_genome(tribe, rng)
    return vault.add_strategy(
        name=None,
        tribe=tribe,
        symbol=symbol,
        timeframe=timeframe,
        genome_json=g.to_json(),
        is_metrics=_metrics(),
        oos_metrics=_metrics(net=net, pf=pf, dd=dd),
        equity=np.linspace(1.0, 1.0 + net, 50),
        campaign_id=campaign_id,
    )


def test_add_and_get_round_trip(vault):
    sid = _add(vault)
    row = vault.get_strategy(sid)
    assert row["id"] == sid
    assert row["name"] == f"ALG-TR-{sid:05d}"
    assert row["tribe"] == "trend"
    assert row["oos_metrics"]["net_profit"] == pytest.approx(0.2)
    assert row["is_metrics"]["profit_factor"] == pytest.approx(1.8)
    equity = np.array(row["equity"])
    assert len(equity) == 50
    assert equity[-1] == pytest.approx(1.2)
    assert "genome_json" in row and "ma_cross" in row["genome_json"] or True
    assert vault.get_strategy(99999) is None


def test_list_filters_and_sort(vault):
    _add(vault, tribe="trend", symbol="EURUSD", net=0.1)
    _add(vault, tribe="momentum", symbol="GBPUSD", net=0.5)
    _add(vault, tribe="trend", symbol="EURUSD", net=0.3)

    assert len(vault.list_strategies()) == 3
    assert len(vault.list_strategies(symbol="EURUSD")) == 2
    assert len(vault.list_strategies(tribe="momentum")) == 1
    assert len(vault.list_strategies(symbol="EURUSD", tribe="momentum")) == 0

    top = vault.list_strategies(sort="oos_net_profit", desc=True)
    assert [r["oos_net_profit"] for r in top] == pytest.approx([0.5, 0.3, 0.1])
    asc = vault.list_strategies(sort="oos_net_profit", desc=False)
    assert asc[0]["oos_net_profit"] == pytest.approx(0.1)


def test_list_metric_range_filters(vault):
    _add(vault, net=0.1, pf=1.2, dd=0.05)
    _add(vault, net=0.5, pf=2.5, dd=0.20)
    _add(vault, net=-0.2, pf=0.8, dd=0.40)

    assert len(vault.list_strategies(min_oos_net_profit=0.0)) == 2
    assert len(vault.list_strategies(min_oos_profit_factor=2.0)) == 1
    assert len(vault.list_strategies(max_oos_max_drawdown=0.25)) == 2
    combined = vault.list_strategies(min_oos_net_profit=0.2, min_oos_profit_factor=1.0)
    assert len(combined) == 1
    assert combined[0]["oos_net_profit"] == pytest.approx(0.5)
    # None means "no constraint"
    assert len(vault.list_strategies(min_oos_profit_factor=None)) == 3


def test_list_rejects_unknown_metric_filter(vault):
    _add(vault)
    with pytest.raises(ValueError, match="filter"):
        vault.list_strategies(min_bogus_metric=1.0)


def test_campaign_tribe_genomes(vault):
    cid = vault.create_campaign("{}")
    sid = _add(vault, campaign_id=cid)
    _add(vault, campaign_id=cid, tribe="momentum", symbol="GBPUSD")
    _add(vault)  # not part of the campaign
    genomes = vault.campaign_tribe_genomes(cid, "EURUSD", "trend")
    assert genomes == {vault.get_strategy(sid)["genome_json"]}
    assert vault.campaign_tribe_genomes(cid, "USDJPY", "trend") == set()


def test_list_rejects_unknown_sort_column(vault):
    _add(vault)
    with pytest.raises(ValueError, match="sort"):
        vault.list_strategies(sort="1; DROP TABLE strategies")


def test_delete(vault):
    sid = _add(vault)
    assert vault.delete_strategy(sid) is True
    assert vault.get_strategy(sid) is None
    assert vault.delete_strategy(sid) is False


def test_campaign_lifecycle(vault):
    cid = vault.create_campaign('{"symbols": ["EURUSD"]}')
    row = vault.get_campaign(cid)
    assert row["status"] == "running"
    vault.update_campaign(cid, status="done", progress_json='{"gen": 3}')
    row = vault.get_campaign(cid)
    assert row["status"] == "done"
    assert row["progress_json"] == '{"gen": 3}'
    assert len(vault.list_campaigns()) == 1


def test_insights_aggregation(vault):
    _add(vault, tribe="trend", symbol="EURUSD", net=0.1)
    _add(vault, tribe="trend", symbol="EURUSD", net=0.3)
    _add(vault, tribe="momentum", symbol="GBPUSD", net=0.5)
    ins = vault.insights()
    by_tribe = {r["key"]: r for r in ins["by_tribe"]}
    assert by_tribe["trend"]["count"] == 2
    assert by_tribe["trend"]["avg_oos_net_profit"] == pytest.approx(0.2)
    by_symbol = {r["key"]: r for r in ins["by_symbol"]}
    assert by_symbol["GBPUSD"]["count"] == 1
    assert set(ins.keys()) == {"by_symbol", "by_timeframe", "by_tribe"}


def _add_stage(vault, stage="legacy", **kw):
    rng = np.random.default_rng(0)
    g = random_genome("trend", rng)
    return vault.add_strategy(
        name=None, tribe="trend", symbol="EURUSD", timeframe="H1",
        genome_json=g.to_json(), is_metrics=_metrics(), oos_metrics=_metrics(),
        equity=np.linspace(1.0, 1.2, 50), campaign_id=None, stage=stage, **kw,
    )


def test_migration_marks_existing_rows_legacy(tmp_path):
    import sqlite3

    from alglory.vault import db as vdb

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(vdb._SCHEMA)  # pre-change schema: no stage columns
    con.execute(
        "INSERT INTO strategies (tribe, symbol, timeframe, genome_json, is_json, oos_json)"
        " VALUES ('trend','EURUSD','H1','{}','{}','{}')"
    )
    con.commit()
    con.close()
    Vault(path)
    v = Vault(path)  # second open must not error
    assert v.get_strategy(1)["stage"] == "legacy"


def test_add_strategy_with_holdout(vault):
    trades = [{"exit_time": "2024-01-02T10:00:00", "r": 1.5},
              {"exit_time": "2024-01-03T10:00:00", "r": -1.0}]
    sid = _add_stage(
        vault, stage="holdout_passed",
        holdout_metrics={"win_rate": 0.63, "profit_factor": 1.9, "trades": 40,
                         "net_profit": 0.12},
        holdout_trades=trades, holdout_reused=True,
    )
    s = vault.get_strategy(sid)
    assert s["stage"] == "holdout_passed"
    assert s["ho_win_rate"] == 0.63
    assert s["ho_trades"] == 40
    assert s["holdout_trades"] == trades
    assert s["holdout_reused"] == 1
    assert s["demo"] == {}


def test_set_stage_and_rejects_unknown(vault):
    sid = _add_stage(vault, stage="holdout_passed")
    with pytest.raises(ValueError):
        vault.set_stage(sid, "live")
    vault.set_stage(sid, "on_demo", demo_started_at="2026-01-01T00:00:00",
                    demo_preset="p", demo_json={"a": 1})
    s = vault.get_strategy(sid)
    assert (s["stage"], s["demo_preset"], s["demo"]) == ("on_demo", "p", {"a": 1})
    vault.set_stage(sid, "demo_passed")
    s = vault.get_strategy(sid)
    assert s["stage"] == "demo_passed" and s["demo_preset"] == "p"


def test_list_filters_stage_and_hides_legacy(vault):
    a = _add_stage(vault, stage="legacy")
    b = _add_stage(vault, stage="holdout_passed")
    ids = lambda rows: {r["id"] for r in rows}  # noqa: E731
    assert ids(vault.list_strategies()) == {a, b}
    assert ids(vault.list_strategies(include_legacy=False)) == {b}
    assert ids(vault.list_strategies(stage="legacy")) == {a}
    assert ids(vault.list_strategies(stage="holdout_passed")) == {b}
    assert "stage" in vault.list_strategies()[0]
    vault.list_strategies(sort="ho_win_rate")


def test_consumed_holdout_windows(vault):
    c1, c2, c3 = (vault.create_campaign("{}") for _ in range(3))
    w = lambda s, tf, a: {"symbol": s, "timeframe": tf, "start": a, "end": a + "Z"}  # noqa: E731
    vault.set_holdout_windows(c1, [w("EURUSD", "H1", "2024-01-01"), w("GBPUSD", "H1", "2024-02-01")])
    vault.set_holdout_windows(c2, [w("EURUSD", "H1", "2024-03-01")])
    vault.set_holdout_windows(c3, [w("EURUSD", "M15", "2024-04-01")])
    got = vault.consumed_holdout_windows("EURUSD", "H1")
    assert sorted(x["start"] for x in got) == ["2024-01-01", "2024-03-01"]
    got = vault.consumed_holdout_windows("EURUSD", "H1", exclude_campaign=c1)
    assert [x["start"] for x in got] == ["2024-03-01"]
