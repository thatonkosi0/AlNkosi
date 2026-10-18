import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

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
    app = create_app(cfg, token="test-token")
    fake = FakeManager()
    app.state.manager = fake
    client = TestClient(app)
    client.headers.update({"X-Alglory-Token": "test-token"})
    return client, fake, cfg


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


def test_campaign_accepts_seed_strategy_id(app_client):
    # the OPTIMIZE trigger posts a normal campaign config plus seed_strategy_id;
    # it must flow through to the manager so the population seeds from that genome
    client, fake, cfg = app_client
    payload = {
        "symbols": ["EURUSD"], "tribes": ["trend"],
        "population": 8, "generations": 2, "seed_strategy_id": 42,
    }
    r = client.post("/api/campaigns", json=payload)
    assert r.status_code == 202
    assert fake.started_with.seed_strategy_id == 42


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


def test_list_strategies_stage_filter_and_legacy_toggle(app_client):
    client, fake, cfg = app_client
    legacy = _seed_strategy(cfg)
    passed = _seed_strategy(cfg)
    Vault(cfg.db_path).set_stage(passed, "holdout_passed")
    ids = lambda **p: {r["id"] for r in client.get("/api/vault", params=p).json()}
    assert ids() == {legacy, passed}
    assert ids(include_legacy="false") == {passed}
    assert ids(stage="legacy") == {legacy}
    assert ids(stage="holdout_passed") == {passed}
    assert ids(stage="on_demo") == set()
    row = client.get("/api/vault", params={"stage": "holdout_passed"}).json()[0]
    assert row["stage"] == "holdout_passed"


def test_campaign_get_by_id(app_client):
    client, fake, cfg = app_client
    cid = Vault(cfg.db_path).create_campaign('{"symbols": ["EURUSD"]}')
    body = client.get(f"/api/campaigns/{cid}").json()
    assert body["id"] == cid
    assert client.get("/api/campaigns/424242").status_code == 404


def test_campaign_accepts_cost_overrides(app_client):
    client, fake, cfg = app_client
    payload = {
        "symbols": ["EURUSD"], "tribes": ["trend"], "population": 8,
        "generations": 2, "cost_overrides": {"EURUSD": {"commission_per_lot": 3.0}},
    }
    assert client.post("/api/campaigns", json=payload).status_code == 202
    assert fake.started_with.cost_overrides == {"EURUSD": {"commission_per_lot": 3.0}}


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


def test_vault_rescore_and_robustness_sort(app_client):
    from alglory.data.cache import BarCache
    from alglory.data.sample import generate_bars

    client, fake, cfg = app_client
    BarCache(cfg.data_dir).save("EURUSD", "H1", generate_bars("EURUSD", "H1", 3000, seed=7))
    sid = _seed_strategy(cfg)

    out = client.post("/api/vault/rescore?symbol=EURUSD&timeframe=H1").json()
    assert out["scored"] >= 1

    ranked = client.get("/api/vault?sort=robustness_score").json()
    scored = next(r for r in ranked if r["id"] == sid)
    assert scored["robustness_score"] is not None
    assert scored["robustness_grade"] in {"A", "B", "C", "D", "F"}
    # unknown timeframe is rejected
    assert client.post("/api/vault/rescore?symbol=EURUSD&timeframe=ZZ").status_code == 422


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
    with client.websocket_connect("/ws/events?token=test-token") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "hello"
        assert msg["campaign"] is None


def test_static_ui_served(app_client):
    client, fake, cfg = app_client
    r = client.get("/")
    assert r.status_code == 200
    assert "ALGLORY" in r.text


def test_api_rejects_missing_or_wrong_token(app_client):
    client, fake, cfg = app_client
    bare = TestClient(client.app)
    assert bare.get("/api/status").status_code == 401
    bare.headers.update({"X-Alglory-Token": "wrong"})
    assert bare.get("/api/status").status_code == 401
    bare.headers.update({"X-Alglory-Token": "test-token"})
    assert bare.get("/api/status").status_code == 200


def test_static_ui_unprotected_without_token(app_client):
    client, fake, cfg = app_client
    bare = TestClient(client.app)
    assert bare.get("/").status_code == 200  # the UI itself needs no token to load


def test_websocket_requires_token(app_client):
    client, fake, cfg = app_client
    bare = TestClient(client.app)
    with pytest.raises(WebSocketDisconnect):
        with bare.websocket_connect("/ws/events"):
            pass


# A malformed token must fail closed with a clean 401 — never an unhandled
# 500. secrets.compare_digest raises TypeError on non-ASCII str, and decoding
# raw invalid-UTF-8 header bytes raises UnicodeDecodeError; both are reachable
# from a browser address bar in the one module that gatekeeps live trading.
@pytest.mark.parametrize(
    "query_token",
    [
        "%C3%A9",        # valid UTF-8, non-ASCII -> compare_digest TypeError
        "%FF",           # invalid UTF-8 byte    -> UnicodeDecodeError
        "%FF%FE%00",     # invalid UTF-8 + NUL
        "caf%C3%A9",     # non-ASCII mixed with ASCII
        "%E2%82%AC" * 8,  # longer non-ASCII payload
    ],
)
def test_malformed_query_token_is_a_clean_401(app_client, query_token):
    client, fake, cfg = app_client
    bare = TestClient(client.app)
    r = bare.get(f"/api/status?token={query_token}")
    assert r.status_code == 401


@pytest.mark.parametrize("header_bytes", [b"\xc3\xa9", b"\xff", b"\xff\xfe"])
def test_malformed_header_token_is_a_clean_401(app_client, header_bytes):
    client, fake, cfg = app_client
    bare = TestClient(client.app)
    r = bare.get("/api/status", headers={b"X-Alglory-Token": header_bytes})
    assert r.status_code == 401


def test_malformed_token_does_not_break_the_websocket_gate(app_client):
    client, fake, cfg = app_client
    bare = TestClient(client.app)
    with pytest.raises(WebSocketDisconnect):
        with bare.websocket_connect("/ws/events?token=%FF"):
            pass


def test_correct_token_still_accepted_after_hardening(app_client):
    client, fake, cfg = app_client
    bare = TestClient(client.app)
    assert bare.get("/api/status?token=test-token").status_code == 200


def test_deploy_uses_strategy_magic(app_client):
    from pathlib import Path

    client, fake, cfg = app_client
    sid = _seed_strategy(cfg)
    body = client.post(f"/api/deploy/{sid}", json={"preset": "personal"}).json()
    code = Path(body["path"]).read_text(encoding="utf-8")
    assert f"InpMagic = {990000 + sid};" in code


# ---- demo-stage API (arm rules, precheck, refresh) -------------------------

import pandas as pd
from datetime import UTC, datetime, timedelta

from alglory.data.mt5source import AccountInfo, ConnStatus


def _seed_staged(cfg, stage="holdout_passed", records=None):
    vault = Vault(cfg.db_path)
    g = random_genome("trend", np.random.default_rng(0))
    metrics = {"net_profit": 0.3, "max_drawdown": 0.1, "profit_factor": 1.5,
               "sharpe": 1.0, "win_rate": 0.5, "trades": 40, "avg_yearly_return": 0.3}
    return vault.add_strategy(
        name=None, tribe="trend", symbol="EURUSD", timeframe="H1",
        genome_json=g.to_json(), is_metrics=metrics, oos_metrics=metrics,
        equity=np.linspace(1, 1.3, 30), campaign_id=None, stage=stage,
        holdout_metrics={"net_profit": 0.1, "trades": 5} if records is not None else None,
        holdout_trades=records,
    )


def _losing_records(day="2026-01-05", n=4):
    return [{"exit_time": f"{day}T{9 + i:02d}:00:00", "r": -1.0} for i in range(n)]


def test_arm_refuses_demo_failed(app_client):
    client, _, cfg = app_client
    fake = _install_fake_executor(client)
    sid = _seed_staged(cfg, stage="demo_failed", records=_losing_records())
    r = client.post("/api/executor/arm", json={"strategy_ids": [sid], "mode": "demo"})
    assert r.status_code == 409
    assert r.json()["detail"] == f"Strategy {sid} failed its demo; it cannot be re-armed."
    assert fake.armed is None


def test_arm_demo_sets_on_demo_and_start_time(app_client):
    client, _, cfg = app_client
    _install_fake_executor(client)
    sid = _seed_staged(cfg, records=_losing_records())
    r = client.post("/api/executor/arm",
                    json={"strategy_ids": [sid], "mode": "demo", "preset": "prop_conservative"})
    assert r.status_code == 200
    row = Vault(cfg.db_path).get_strategy(sid)
    assert row["stage"] == "on_demo"
    assert row["demo_preset"] == "prop_conservative"
    assert datetime.fromisoformat(row["demo_started_at"]).tzinfo is not None
    # re-arming keeps the original start time
    client.post("/api/executor/arm", json={"strategy_ids": [sid], "mode": "demo"})
    assert Vault(cfg.db_path).get_strategy(sid)["demo_started_at"] == row["demo_started_at"]


def test_arm_allows_legacy(app_client):
    client, _, cfg = app_client
    fake = _install_fake_executor(client)
    sid = _seed_strategy(cfg)
    r = client.post("/api/executor/arm", json={"strategy_ids": [sid], "mode": "dry_run"})
    assert r.status_code == 200
    assert fake.armed is not None
    assert Vault(cfg.db_path).get_strategy(sid)["stage"] == "legacy"


def test_precheck_flags_joint_breach_and_excludes_legacy(app_client):
    client, _, cfg = app_client
    a = _seed_staged(cfg, records=_losing_records())
    b = _seed_staged(cfg, records=_losing_records())
    legacy = _seed_strategy(cfg)
    # 8 x -1R at 0.5% on one day is ~3.9%, over the 3% daily limit; one alone is ~2%
    r = client.post("/api/executor/precheck",
                    json={"strategy_ids": [a, b, legacy], "preset": "prop_conservative"})
    assert r.status_code == 200
    body = r.json()
    assert body["worst_day_loss"] > 0.03
    assert any("daily loss" in x for x in body["breaches"])
    assert body["excluded"] == [{"id": legacy, "reason": "legacy — no holdout"}]
    solo = client.post("/api/executor/precheck",
                       json={"strategy_ids": [a], "preset": "prop_conservative"}).json()
    assert solo["breaches"] == []


def test_precheck_all_excluded_returns_zeros(app_client):
    client, _, cfg = app_client
    legacy = _seed_strategy(cfg)
    r = client.post("/api/executor/precheck", json={"strategy_ids": [legacy], "preset": "personal"})
    assert r.status_code == 200
    assert r.json() == {
        "worst_day_loss": 0.0, "max_drawdown": 0.0, "breaches": [],
        "excluded": [{"id": legacy, "reason": "legacy — no holdout"}],
    }


class _DealsSource:
    def __init__(self, deals, balance=10_000.0):
        self._deals, self._balance = deals, balance

    def connect(self):
        return ConnStatus(ok=True, message="ok")

    def account(self):
        return AccountInfo(1, self._balance, self._balance, "USD", True)

    def deals_since(self, start):
        return self._deals[self._deals["time"] >= start].reset_index(drop=True)


class _DownSource:
    def connect(self):
        return ConnStatus(ok=False, message="terminal not running")


def _deals_for(magic, wins, losses, now):
    rows = []
    n = wins + losses
    for i in range(n):
        t = now - timedelta(hours=(n - i) * 3)
        profit = 10.0 if i < wins else -5.0
        rows.append({"time": t, "magic": magic, "entry": "in", "profit": 0.0,
                     "commission": 0.0, "swap": 0.0})
        rows.append({"time": t + timedelta(minutes=30), "magic": magic, "entry": "out",
                     "profit": profit, "commission": 0.0, "swap": 0.0})
    return rows


def _set_on_demo(cfg, sid, days_ago, preset="personal"):
    started = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    Vault(cfg.db_path).set_stage(sid, "on_demo", demo_started_at=started, demo_preset=preset)


def test_demo_refresh_promotes_and_keeps_waiting(app_client):
    client, _, cfg = app_client
    a = _seed_staged(cfg, records=_losing_records())
    b = _seed_staged(cfg, records=_losing_records())
    _set_on_demo(cfg, a, 22)
    _set_on_demo(cfg, b, 22)
    now = datetime.now(UTC)
    rows = _deals_for(990_000 + a, 18, 7, now - timedelta(days=1)) + \
        _deals_for(990_000 + b, 6, 4, now - timedelta(days=1))
    deals = pd.DataFrame(rows)
    client.app.state.demo_source_factory = lambda: _DealsSource(deals)

    r = client.post("/api/demo/refresh")
    assert r.status_code == 200
    body = r.json()
    by_id = {s["id"]: s for s in body["strategies"]}
    assert by_id[a]["stage"] == "demo_passed"
    assert by_id[a]["trades"] == 25
    assert by_id[a]["win_rate"] == pytest.approx(0.72)
    assert by_id[a]["legacy"] is False
    assert by_id[b]["stage"] == "on_demo"
    assert by_id[b]["note"] == "needs more trades"
    assert set(body["account"]) == {"worst_day_loss", "max_drawdown", "breaches"}
    row = Vault(cfg.db_path).get_strategy(a)
    assert row["stage"] == "demo_passed"
    assert row["demo"]["trades"] == 25


def test_demo_refresh_503_when_mt5_down_changes_nothing(app_client):
    client, _, cfg = app_client
    a = _seed_staged(cfg, records=_losing_records())
    _set_on_demo(cfg, a, 30)
    before = Vault(cfg.db_path).get_strategy(a)
    client.app.state.demo_source_factory = lambda: _DownSource()
    r = client.post("/api/demo/refresh")
    assert r.status_code == 503
    assert "terminal not running" in r.json()["detail"]
    after = Vault(cfg.db_path).get_strategy(a)
    assert after["stage"] == before["stage"] == "on_demo"
    assert after["demo_json"] == before["demo_json"]


def test_demo_refresh_breach_fails_strategy(app_client):
    client, _, cfg = app_client
    a = _seed_staged(cfg, records=_losing_records())
    _set_on_demo(cfg, a, 2, preset="prop_conservative")
    t = datetime.now(UTC) - timedelta(hours=2)
    deals = pd.DataFrame([
        {"time": t, "magic": 990_000 + a, "entry": "in", "profit": 0.0,
         "commission": 0.0, "swap": 0.0},
        {"time": t + timedelta(minutes=10), "magic": 990_000 + a, "entry": "out",
         "profit": -500.0, "commission": 0.0, "swap": 0.0},
    ])
    client.app.state.demo_source_factory = lambda: _DealsSource(deals, balance=9_500.0)
    body = client.post("/api/demo/refresh").json()
    s = body["strategies"][0]
    assert s["stage"] == "demo_failed"
    assert s["note"] == "prop limit breached"
    assert body["account"]["breaches"]


def _deal(t, magic, entry, profit):
    return {"time": t, "magic": magic, "entry": entry, "profit": profit,
            "commission": 0.0, "swap": 0.0}


def test_demo_refresh_drawdown_fails_only_strategies_that_traded_in_it(app_client):
    client, _, cfg = app_client
    loser, late, idle, same_day, later = (
        _seed_staged(cfg, records=_losing_records()) for _ in range(5))
    for sid in (loser, idle, same_day, later):
        _set_on_demo(cfg, sid, 5, preset="prop_conservative")
    _set_on_demo(cfg, late, 2, preset="prop_conservative")  # armed after the drawdown
    now = datetime.now(UTC)
    hit = now - timedelta(days=4)
    m = lambda sid: 990_000 + sid  # noqa: E731
    deals = pd.DataFrame([
        # a 9% account drawdown 4 days ago, from loser (over 8% dd and 3% daily)
        _deal(hit, m(loser), "in", 0.0),
        _deal(hit + timedelta(minutes=5), m(loser), "out", -900.0),
        # same_day closed a small winner on that breach day
        _deal(hit + timedelta(minutes=10), m(same_day), "in", 0.0),
        _deal(hit + timedelta(minutes=20), m(same_day), "out", 5.0),
        # later traded only after the breach day, inside its own drawdown window
        _deal(now - timedelta(days=3), m(later), "in", 0.0),
        _deal(now - timedelta(days=3, minutes=-10), m(later), "out", 5.0),
        # late only traded after its own start: no drawdown since then
        *[d for i in range(3) for d in (
            _deal(now - timedelta(hours=10 - i), m(late), "in", 0.0),
            _deal(now - timedelta(hours=10 - i, minutes=-30), m(late), "out", 10.0))],
    ])
    balance = 10_000.0 - 900.0 + 5.0 + 5.0 + 30.0
    client.app.state.demo_source_factory = lambda: _DealsSource(deals, balance=balance)
    body = client.post("/api/demo/refresh").json()
    by_id = {s["id"]: s for s in body["strategies"]}
    assert by_id[loser]["stage"] == "demo_failed"
    assert by_id[same_day]["stage"] == "demo_failed"  # traded on a breach day
    assert by_id[later]["stage"] == "demo_failed"  # traded inside its own 9% drawdown
    assert by_id[late]["stage"] == "on_demo"  # armed after the drawdown
    assert by_id[idle]["stage"] == "on_demo"  # no closed deals: nothing to blame
    v = Vault(cfg.db_path)
    assert v.get_strategy(late)["stage"] == "on_demo"
    assert v.get_strategy(idle)["stage"] == "on_demo"


@pytest.mark.parametrize("preset", [None, "bogus"])
def test_demo_refresh_unknown_preset_stays_on_demo(app_client, preset):
    client, _, cfg = app_client
    a = _seed_staged(cfg, records=_losing_records())
    started = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    v = Vault(cfg.db_path)
    v.set_stage(a, "on_demo", demo_started_at=started, demo_preset=preset)
    now = datetime.now(UTC)
    # good enough to pass, and a 50% loss day: neither decides an unknown preset
    deals = pd.DataFrame(_deals_for(990_000 + a, 25, 0, now - timedelta(days=2)) + [
        _deal(now - timedelta(hours=1), 990_000 + a, "in", 0.0),
        _deal(now - timedelta(minutes=30), 990_000 + a, "out", -5_000.0)])
    client.app.state.demo_source_factory = lambda: _DealsSource(deals, balance=5_250.0)
    s = client.post("/api/demo/refresh").json()["strategies"][0]
    assert s["stage"] == "on_demo"
    assert s["note"] == "unknown preset — re-arm"
    row = v.get_strategy(a)
    assert row["stage"] == "on_demo"
    assert row["demo"]["note"] == "unknown preset — re-arm"
