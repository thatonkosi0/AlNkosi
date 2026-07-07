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
        self.current_campaign_id = None
        self.last_events = []

    def is_running(self):
        return self.running

    def start(self, cfg: CampaignConfig):
        if self.running:
            raise BusyError("busy")
        self.started_with = cfg
        self.running = True

    def stop(self):
        self.stopped = True

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
