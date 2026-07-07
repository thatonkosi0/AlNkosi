"""End-to-end smoke test: real app factory, real worker process, tiny campaign."""

import time

import pytest
from fastapi.testclient import TestClient

from alglory.config import AppConfig, UI_DIR
from alglory.server.app import create_app

TIMEOUT_S = 120


@pytest.fixture
def client(tmp_path):
    cfg = AppConfig(data_dir=tmp_path, db_path=tmp_path / "vault.db", ui_dir=UI_DIR)
    app = create_app(cfg)
    with TestClient(app) as c:  # context manager runs the lifespan pump
        yield c


def test_full_campaign_via_api(client):
    payload = {
        "symbols": ["EURUSD"],
        "timeframe": "H1",
        "tribes": ["trend", "breakout"],
        "population": 12,
        "generations": 4,
        "seed": 42,
        "source": "sample",
        "min_trades": 5,
    }
    r = client.post("/api/campaigns", json=payload)
    assert r.status_code == 202

    deadline = time.time() + TIMEOUT_S
    while time.time() < deadline:
        status = client.get("/api/status").json()
        if not status["campaign"]["running"]:
            break
        time.sleep(1.0)
    else:
        pytest.fail(f"campaign still running after {TIMEOUT_S}s")

    campaigns = client.get("/api/campaigns").json()
    assert campaigns[0]["status"] == "done"

    rows = client.get("/api/vault").json()
    assert len(rows) > 0, "expected at least one vaulted strategy from seed 42"

    insights = client.get("/api/insights").json()
    assert insights["by_symbol"][0]["key"] == "EURUSD"

    top = rows[0]
    deploy = client.post(f"/api/deploy/{top['id']}", json={"preset": "personal"}).json()
    from pathlib import Path

    code = Path(deploy["path"]).read_text(encoding="utf-8")
    assert "OnTick" in code
    assert "ALGLORY" in code
