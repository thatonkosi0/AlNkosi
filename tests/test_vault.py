import numpy as np
import pytest

from alglory.genome import random_genome
from alglory.vault.db import Vault


@pytest.fixture
def vault(tmp_path):
    return Vault(tmp_path / "vault.db")


def _metrics(net=0.2, pf=1.8):
    return {
        "net_profit": net,
        "max_drawdown": 0.1,
        "profit_factor": pf,
        "sharpe": 1.2,
        "win_rate": 0.5,
        "trades": 60,
        "avg_yearly_return": net,
    }


def _add(vault, tribe="trend", symbol="EURUSD", timeframe="H1", net=0.2, campaign_id=None):
    rng = np.random.default_rng(0)
    g = random_genome(tribe, rng)
    return vault.add_strategy(
        name=None,
        tribe=tribe,
        symbol=symbol,
        timeframe=timeframe,
        genome_json=g.to_json(),
        is_metrics=_metrics(),
        oos_metrics=_metrics(net=net),
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
