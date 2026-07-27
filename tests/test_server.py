import numpy as np
import pytest
from fastapi.testclient import TestClient

from alglory.config import AppConfig
from alglory.evolve.campaign import CampaignConfig
from alglory.genome import random_genome
from alglory.server.app import create_app
from alglory.server.worker import BusyError
from alglory.vault.db import Vault


class FakeManager:
    def __init__(self):
        self.running = False
        self.started_with = None
        self.stopped = False
        self.paused = False
        self.resume_id = None
        self.current_campaign_id = None
        self.last_events = []

    def is_running(self):
        return self.running

    def start(self, cfg: CampaignConfig, resume_campaign_id=None):
        if self.running:
            raise BusyError("busy")
        self.started_with = cfg
        self.resume_id = resume_campaign_id
        self.running = True

    def stop(self):
        self.stopped = True

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def is_paused(self):
        return self.paused

    def drain(self):
        return []


@pytest.fixture
def app_client(tmp_path):
    (tmp_path / "ui").mkdir()
    (tmp_path / "ui" / "index.html").write_text("<html>ALGLORY</html>")
    cfg = AppConfig(data_dir=tmp_path, db_path=tmp_path / "vault.db", ui_dir=tmp_path / "ui")
    app = create_app(cfg)
    fake = FakeManager()
    app.state.manager = fake
    return TestClient(app), fake, cfg


def _seed_strategy(cfg, tribe="trend", net=0.3):
    vault = Vault(cfg.db_path)
    g = random_genome(tribe, np.random.default_rng(0))
    metrics = {
        "net_profit": net, "max_drawdown": 0.1, "profit_factor": 1.5,
        "sharpe": 1.0, "win_rate": 0.5, "trades": 40, "avg_yearly_return": net,
    }
    return vault.add_strategy(
        name=None, tribe=tribe, symbol="EURUSD", timeframe="H1",
        genome_json=g.to_json(), is_metrics=metrics, oos_metrics=metrics,
        equity=np.linspace(1, 1 + net, 30), campaign_id=None,
    )


def test_status_shape(app_client):
    client, fake, cfg = app_client
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"mt5", "vault_count", "campaign"}
    assert body["campaign"]["running"] is False
    assert isinstance(body["mt5"]["available"], bool)


def test_campaign_start_and_busy(app_client):
    client, fake, cfg = app_client
    payload = {"symbols": ["EURUSD"], "tribes": ["trend"], "population": 8, "generations": 2}
    r = client.post("/api/campaigns", json=payload)
    assert r.status_code == 202
    assert fake.started_with.symbols == ["EURUSD"]
    fake.running = True
    r2 = client.post("/api/campaigns", json=payload)
    assert r2.status_code == 409


def test_campaign_validation_error(app_client):
    client, fake, cfg = app_client
    r = client.post("/api/campaigns", json={"symbols": [], "tribes": ["trend"]})
    assert r.status_code == 422


def test_campaign_cancel(app_client):
    client, fake, cfg = app_client
    fake.running = True
    r = client.post("/api/campaigns/cancel")
    assert r.status_code == 200
    assert fake.stopped is True


def test_campaign_pause_and_resume(app_client):
    client, fake, cfg = app_client
    # not running -> 409
    assert client.post("/api/campaigns/pause").status_code == 409
    assert client.post("/api/campaigns/resume").status_code == 409
    fake.running = True
    assert client.post("/api/campaigns/pause").status_code == 200
    assert fake.paused is True
    status = client.get("/api/status").json()
    assert status["campaign"]["paused"] is True
    assert client.post("/api/campaigns/resume").status_code == 200
    assert fake.paused is False


def test_campaign_restart(app_client):
    client, fake, cfg = app_client
    vault = Vault(cfg.db_path)
    assert client.post("/api/campaigns/424242/restart").status_code == 404

    cid = vault.create_campaign('{"symbols": ["EURUSD"], "tribes": ["trend"]}')
    # still marked running -> not resumable
    assert client.post(f"/api/campaigns/{cid}/restart").status_code == 422
    vault.update_campaign(cid, status="failed")
    # failed but never checkpointed -> not resumable
    assert client.post(f"/api/campaigns/{cid}/restart").status_code == 422

    vault.update_campaign(
        cid,
        progress_json='{"symbol": "EURUSD", "tribe": "trend", "gen": 1, '
        '"units_done": 1, "population": null}',
    )
    r = client.post(f"/api/campaigns/{cid}/restart")
    assert r.status_code == 202
    assert fake.resume_id == cid
    assert fake.started_with.symbols == ["EURUSD"]

    fake.running = True
    assert client.post(f"/api/campaigns/{cid}/restart").status_code == 409


def test_vault_metric_filter_params(app_client):
    client, fake, cfg = app_client
    _seed_strategy(cfg, net=0.1)
    _seed_strategy(cfg, net=0.6)
    rows = client.get("/api/vault?min_oos_net_profit=0.3").json()
    assert len(rows) == 1
    assert rows[0]["oos_net_profit"] == pytest.approx(0.6)
    # seeded strategies carry 10% drawdown, so a 5% cap filters everything
    assert client.get("/api/vault?max_oos_max_drawdown=0.05").json() == []
    assert len(client.get("/api/vault?min_oos_profit_factor=1.0").json()) == 2


def test_campaign_get_by_id(app_client):
    client, fake, cfg = app_client
    cid = Vault(cfg.db_path).create_campaign('{"symbols": ["EURUSD"]}')
    body = client.get(f"/api/campaigns/{cid}").json()
    assert body["id"] == cid
    assert client.get("/api/campaigns/424242").status_code == 404


def test_campaign_accepts_commission(app_client):
    client, fake, cfg = app_client
    payload = {
        "symbols": ["EURUSD"], "tribes": ["trend"], "population": 8,
        "generations": 2, "commission_per_lot": 0.00007,
    }
    assert client.post("/api/campaigns", json=payload).status_code == 202
    assert fake.started_with.commission_per_lot == pytest.approx(0.00007)


def test_vault_list_get_delete(app_client):
    client, fake, cfg = app_client
    sid = _seed_strategy(cfg)
    rows = client.get("/api/vault").json()
    assert len(rows) == 1
    assert rows[0]["name"].startswith("ALG-TR-")

    detail = client.get(f"/api/vault/{sid}").json()
    assert detail["oos_metrics"]["net_profit"] == pytest.approx(0.3)
    assert len(detail["equity"]) == 30

    assert client.get("/api/vault/424242").status_code == 404
    assert client.delete(f"/api/vault/{sid}").status_code == 200
    assert client.get(f"/api/vault/{sid}").status_code == 404


def test_vault_filters(app_client):
    client, fake, cfg = app_client
    _seed_strategy(cfg, tribe="trend")
    _seed_strategy(cfg, tribe="momentum")
    rows = client.get("/api/vault", params={"tribe": "momentum"}).json()
    assert len(rows) == 1
    r = client.get("/api/vault", params={"sort": "evil; DROP"})
    assert r.status_code == 422


def test_insights(app_client):
    client, fake, cfg = app_client
    _seed_strategy(cfg)
    body = client.get("/api/insights").json()
    assert body["by_tribe"][0]["key"] == "trend"


def test_deploy_writes_file(app_client):
    client, fake, cfg = app_client
    sid = _seed_strategy(cfg)
    r = client.post(f"/api/deploy/{sid}", json={"preset": "prop_conservative"})
    assert r.status_code == 200
    body = r.json()
    assert body["path"].endswith(".mq5")
    from pathlib import Path

    code = Path(body["path"]).read_text(encoding="utf-8")
    assert "OnTick" in code
    assert "instructions" in body
    assert client.post("/api/deploy/999999", json={"preset": "personal"}).status_code == 404
    assert client.post(f"/api/deploy/{sid}", json={"preset": "nope"}).status_code == 422


def test_deploy_reports_mt5_fallback(app_client, monkeypatch):
    # When MT5 can't be located, deploy must fall back to the exports folder —
    # and SAY SO instead of implying an MT5 success. Patched so the result does
    # not depend on whether the test machine happens to have MT5 installed.
    import alglory.server.app as appmod

    class _NoMT5:
        def available(self):
            return False

    monkeypatch.setattr(appmod, "MT5Source", _NoMT5)
    client, fake, cfg = app_client
    sid = _seed_strategy(cfg)
    body = client.post(f"/api/deploy/{sid}", json={"preset": "personal"}).json()
    assert body["target"] == "fallback"
    assert body["mt5_detected"] is False
    assert "exports" in body["path"].replace("\\", "/")
    assert "MetaTrader5" in body["reason"] or "MT5" in body["reason"]
    assert "not detected" in body["instructions"].lower()


def test_deploy_targets_mt5_when_connected(app_client, monkeypatch, tmp_path):
    import alglory.server.app as appmod
    from alglory.data.mt5source import ConnStatus

    experts = tmp_path / "Terminal" / "MQL5" / "Experts" / "Alglory"
    experts.mkdir(parents=True)

    class _LiveMT5:
        def available(self):
            return True

        def connect(self):
            return ConnStatus(ok=True, message="Connected to MetaTrader 5.")

        def experts_dir(self):
            return experts

    monkeypatch.setattr(appmod, "MT5Source", _LiveMT5)
    client, fake, cfg = app_client
    sid = _seed_strategy(cfg)
    body = client.post(f"/api/deploy/{sid}", json={"preset": "personal"}).json()
    assert body["target"] == "mt5"
    assert body["mt5_detected"] is True
    assert body["reason"] is None
    assert body["path"].replace("\\", "/").startswith(str(experts).replace("\\", "/"))


def test_deploy_surfaces_min_lot_fallback_and_account(app_client):
    client, fake, cfg = app_client
    sid = _seed_strategy(cfg)
    body = client.post(f"/api/deploy/{sid}", json={"preset": "personal"}).json()
    assert body["min_lot_fallback"] is True
    assert "fallback" in body["sizing_note"].lower()
    # the generated EA carries the fallback so a small account still trades
    from pathlib import Path

    code = Path(body["path"]).read_text(encoding="utf-8")
    assert "InpMinLotFallback = true" in code


def test_deploy_estimates_min_account_when_bars_cached(app_client):
    import numpy as np

    from alglory.data.cache import BarCache
    from alglory.data.sample import generate_bars

    client, fake, cfg = app_client
    BarCache(cfg.data_dir).save("EURUSD", "H1", generate_bars("EURUSD", "H1", 2000, seed=7))
    sid = _seed_strategy(cfg)
    body = client.post(f"/api/deploy/{sid}", json={"preset": "personal"}).json()
    assert body["min_account_estimate"] is not None
    assert body["min_account_estimate"] > 0
    assert np.isfinite(body["min_account_estimate"])


def test_deploy_custom_out_dir_marks_target(app_client, tmp_path):
    client, fake, cfg = app_client
    sid = _seed_strategy(cfg)
    dest = tmp_path / "custom_out"
    body = client.post(
        f"/api/deploy/{sid}", json={"preset": "personal", "out_dir": str(dest)}
    ).json()
    assert body["target"] == "custom"
    assert body["path"].replace("\\", "/").startswith(str(dest).replace("\\", "/"))


class FakeExecutorManager:
    def __init__(self):
        self.armed = None
        self.stopped = False

    def arm_strategies(self, runtimes, mode, confirm_live, *, poll_seconds=5.0, max_open_positions=None):
        self.armed = {
            "runtimes": runtimes, "mode": mode, "confirm_live": confirm_live,
            "max_open_positions": max_open_positions,
        }

    def stop(self):
        self.stopped = True

    def status(self):
        return {"running": False, "armed": self.armed is not None,
                "mode": self.armed["mode"] if self.armed else None, "strategies": []}

    def drain(self):
        return []

    def is_running(self):
        return False


def _install_fake_executor(client):
    fake = FakeExecutorManager()
    client.app.state.executor = fake
    return fake


def test_executor_arm_live_requires_confirm(app_client):
    client, _, cfg = app_client
    _install_fake_executor(client)
    sid = _seed_strategy(cfg)
    r = client.post("/api/executor/arm", json={"strategy_ids": [sid], "mode": "live"})
    assert r.status_code == 400
    # with confirm the validation passes (fake manager arms without MT5)
    ok = client.post(
        "/api/executor/arm",
        json={"strategy_ids": [sid], "mode": "live", "confirm_live": True},
    )
    assert ok.status_code == 200


def test_executor_arm_validation_errors(app_client):
    client, _, cfg = app_client
    _install_fake_executor(client)
    sid = _seed_strategy(cfg)
    assert client.post("/api/executor/arm", json={"strategy_ids": [sid], "mode": "bogus"}).status_code == 422
    assert client.post("/api/executor/arm", json={"strategy_ids": [999999], "mode": "dry_run"}).status_code == 404
    assert client.post("/api/executor/arm", json={"strategy_ids": [], "mode": "dry_run"}).status_code == 422


def test_executor_arm_dry_run_builds_runtimes(app_client):
    client, _, cfg = app_client
    fake = _install_fake_executor(client)
    sid = _seed_strategy(cfg)
    r = client.post(
        "/api/executor/arm",
        json={"strategy_ids": [sid], "mode": "dry_run", "preset": "prop_conservative",
              "max_open_positions": 3},
    )
    assert r.status_code == 200
    assert fake.armed is not None
    assert fake.armed["mode"] == "dry_run"
    assert fake.armed["max_open_positions"] == 3
    assert len(fake.armed["runtimes"]) == 1
    assert fake.armed["runtimes"][0].magic == 990_000 + sid


def test_executor_stop_and_status(app_client):
    client, _, cfg = app_client
    fake = _install_fake_executor(client)
    assert client.post("/api/executor/stop").json() == {"status": "stopped"}
    assert fake.stopped is True
    assert client.get("/api/executor/status").json()["armed"] is False


def test_websocket_hello(app_client):
    client, fake, cfg = app_client
    with client.websocket_connect("/ws/events") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "hello"
        assert msg["campaign"] is None


def test_static_ui_served(app_client):
    client, fake, cfg = app_client
    r = client.get("/")
    assert r.status_code == 200
    assert "ALGLORY" in r.text
