"""End-to-end smoke test: real app factory, real worker process, tiny campaign."""

import time

import pytest
from fastapi.testclient import TestClient

from alglory.config import AppConfig, UI_DIR
from alglory.server.app import create_app

TIMEOUT_S = 240  # generous headroom: a real spawned worker on a loaded CI box
POLL_S = 0.25


@pytest.fixture
def client(tmp_path):
    cfg = AppConfig(data_dir=tmp_path, db_path=tmp_path / "vault.db", ui_dir=UI_DIR)
    app = create_app(cfg)
    with TestClient(app) as c:  # context manager runs the lifespan pump
        yield c


def _wait_for_terminal_campaign(client, timeout_s=TIMEOUT_S):
    """Poll until the campaign reaches a terminal state.

    Keys off the worker's own authoritative status writes (done/failed/
    cancelled) rather than a fixed sleep or the derived process-alive flag, so a
    slow-but-progressing worker under CPU contention doesn't false-fail. Uses a
    monotonic deadline so a system clock change can't skew the wait.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        campaigns = client.get("/api/campaigns").json()
        if campaigns and campaigns[0]["status"] in ("done", "failed", "cancelled"):
            return campaigns[0]
        time.sleep(POLL_S)
    pytest.fail(f"campaign did not reach a terminal state within {timeout_s}s")


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

    campaign = _wait_for_terminal_campaign(client)
    assert campaign["status"] == "done", f"campaign ended in state {campaign['status']!r}"

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
