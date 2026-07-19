"""FastAPI application: REST API, WebSocket event stream, static UI."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from alglory.config import AppConfig
from alglory.data.mt5source import MT5Source
from alglory.deploy.mql5 import GUARDRAIL_PRESETS, Guardrails, generate_ea, write_ea
from alglory.evolve.campaign import CampaignConfig
from alglory.genome import Genome
from alglory.server.worker import BusyError, CampaignManager
from alglory.vault.db import Vault

POLL_INTERVAL = 0.25


class DeployRequest(BaseModel):
    preset: str = "personal"
    custom: dict | None = None
    out_dir: str | None = None


def create_app(cfg: AppConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Single drainer: keeps the worker's event queue flowing even with no
        # browser attached (a full pipe would block the worker's exit), and
        # broadcasts every event to all connected sockets.
        async def pump():
            while True:
                for event in app.state.manager.drain():
                    dead = []
                    for sock in app.state.sockets:
                        try:
                            await sock.send_json(event)
                        except Exception:
                            dead.append(sock)
                    for sock in dead:
                        app.state.sockets.discard(sock)
                await asyncio.sleep(POLL_INTERVAL)

        task = asyncio.create_task(pump())
        yield
        task.cancel()

    app = FastAPI(title="Alglory", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.manager = CampaignManager(cfg.db_path, cfg.data_dir)
    app.state.sockets = set()

    def vault() -> Vault:
        return Vault(cfg.db_path)

    @app.get("/api/status")
    def status():
        src = MT5Source()
        available = src.available()
        connected, message = False, "MetaTrader5 package not installed."
        if available:
            conn = src.connect()
            connected, message = conn.ok, conn.message
        manager = app.state.manager
        return {
            "mt5": {"available": available, "connected": connected, "message": message},
            "vault_count": vault().count(),
            "campaign": {
                "running": manager.is_running(),
                "paused": manager.is_paused(),
                "campaign_id": manager.current_campaign_id,
            },
        }

    @app.post("/api/campaigns", status_code=202)
    def start_campaign(config: CampaignConfig):
        manager = app.state.manager
        if manager.is_running():
            raise HTTPException(409, "A campaign is already running; cancel it first.")
        try:
            manager.start(config)
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "started"}

    @app.post("/api/campaigns/cancel")
    def cancel_campaign():
        manager = app.state.manager
        if not manager.is_running():
            raise HTTPException(409, "No campaign is running.")
        manager.stop()
        return {"status": "cancelling"}

    @app.post("/api/campaigns/pause")
    def pause_campaign():
        manager = app.state.manager
        if not manager.is_running():
            raise HTTPException(409, "No campaign is running.")
        manager.pause()
        return {"status": "pausing"}

    @app.post("/api/campaigns/resume")
    def resume_campaign():
        manager = app.state.manager
        if not manager.is_running():
            raise HTTPException(409, "No campaign is running.")
        manager.resume()
        return {"status": "resuming"}

    @app.get("/api/campaigns")
    def list_campaigns():
        return vault().list_campaigns()

    @app.get("/api/campaigns/{cid}")
    def get_campaign(cid: int):
        row = vault().get_campaign(cid)
        if row is None:
            raise HTTPException(404, f"Campaign {cid} not found.")
        return row

    @app.get("/api/vault")
    def vault_list(
        symbol: str | None = None,
        timeframe: str | None = None,
        tribe: str | None = None,
        sort: str = "oos_net_profit",
        desc: bool = True,
    ):
        try:
            return vault().list_strategies(
                symbol=symbol, timeframe=timeframe, tribe=tribe, sort=sort, desc=desc
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/vault/{sid}")
    def vault_get(sid: int):
        row = vault().get_strategy(sid)
        if row is None:
            raise HTTPException(404, f"Strategy {sid} not found.")
        return row

    @app.delete("/api/vault/{sid}")
    def vault_delete(sid: int):
        if not vault().delete_strategy(sid):
            raise HTTPException(404, f"Strategy {sid} not found.")
        return {"status": "deleted"}

    @app.get("/api/insights")
    def insights():
        return vault().insights()

    @app.post("/api/deploy/{sid}")
    def deploy(sid: int, req: DeployRequest):
        row = vault().get_strategy(sid)
        if row is None:
            raise HTTPException(404, f"Strategy {sid} not found.")
        if req.custom is not None:
            try:
                guard = Guardrails(**req.custom)
            except TypeError as exc:
                raise HTTPException(422, f"Invalid custom guardrails: {exc}") from exc
        elif req.preset in GUARDRAIL_PRESETS:
            guard = GUARDRAIL_PRESETS[req.preset]
        else:
            raise HTTPException(
                422, f"Unknown preset {req.preset!r}; valid: {sorted(GUARDRAIL_PRESETS)}"
            )

        genome = Genome.from_json(row["genome_json"])
        name = row["name"].replace("-", "_")
        code = generate_ea(
            genome,
            name=name,
            symbol=row["symbol"],
            timeframe=row["timeframe"],
            guardrails=guard,
        )
        if req.out_dir:
            out_dir = Path(req.out_dir)
        else:
            src = MT5Source()
            experts = None
            if src.available() and src.connect().ok:
                experts = src.experts_dir()
            out_dir = experts if experts else cfg.data_dir / "exports"
        path = write_ea(code, name, out_dir)
        return {
            "path": str(path),
            "instructions": (
                f"1. Open MetaEditor (F4 in MT5) and open {path.name}. "
                "2. Compile with F7. "
                f"3. In MT5, open a {row['symbol']} {row['timeframe']} chart and drag "
                f"{name} from Navigator > Expert Advisors onto it. "
                "4. Enable Algo Trading. The EA enforces its guardrails automatically."
            ),
        }

    @app.websocket("/ws/events")
    async def ws_events(ws: WebSocket):
        await ws.accept()
        manager = app.state.manager
        campaign = None
        if manager.is_running() and manager.current_campaign_id is not None:
            row = Vault(cfg.db_path).get_campaign(manager.current_campaign_id)
            if row:
                campaign = {
                    "campaign_id": row["id"],
                    "status": row["status"],
                    "progress": json.loads(row["progress_json"]) if row["progress_json"] else None,
                    "recent_events": manager.last_events,
                }
        await ws.send_json({"type": "hello", "campaign": campaign})
        app.state.sockets.add(ws)
        try:
            while True:
                # broadcasting happens in the lifespan pump; this loop only
                # keeps the connection open and notices client closes
                await ws.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            app.state.sockets.discard(ws)

    app.mount("/", StaticFiles(directory=cfg.ui_dir, html=True), name="ui")
    return app
