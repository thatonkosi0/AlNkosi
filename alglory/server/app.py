"""FastAPI application: REST API, WebSocket event stream, static UI."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

from alglory.analysis.sizing import (
    DEFAULT_MAX_RISK_PCT,
    EURUSD_STD,
    min_account_for_risk_sizing,
    min_account_to_trade,
)
from alglory.config import AppConfig
from alglory.data.cache import BarCache
from alglory.data.mt5source import MT5Source
from alglory.deploy.mql5 import GUARDRAIL_PRESETS, Guardrails, generate_ea, write_ea
from alglory.indicators import atr
from alglory.evolve.campaign import CampaignConfig
from alglory.genome import Genome
from alglory.live.executor import ExecutorError, StrategyRuntime
from alglory.server.executor_worker import ExecutorManager
from alglory.server.worker import BusyError, CampaignManager
from alglory.vault.db import Vault

_EXECUTOR_MAGIC_BASE = 990_000

POLL_INTERVAL = 0.25


class DeployRequest(BaseModel):
    preset: str = "personal"
    custom: dict | None = None
    out_dir: str | None = None


class ExecutorArmRequest(BaseModel):
    strategy_ids: list[int]
    mode: str = "dry_run"
    preset: str = "personal"
    confirm_live: bool = False
    poll_seconds: float = 5.0
    max_open_positions: int | None = None


def _resolve_deploy_dir(cfg: AppConfig) -> tuple[Path, str, str | None]:
    """Pick where the EA is written and report how that decision was made.

    Returns (out_dir, target, reason). ``target`` is ``"mt5"`` when the file
    lands in the live terminal's Experts folder, or ``"fallback"`` when MT5
    could not be located and we wrote to ``exports`` instead. ``reason`` carries
    the actionable explanation for a fallback (None on success).
    """
    src = MT5Source()
    if not src.available():
        reason = (
            "The MetaTrader5 Python package is not installed, so the MT5 Experts "
            "folder can't be located. Install it with: pip install alglory[mt5] "
            "(Windows only), then start MetaTrader 5 and deploy again."
        )
        return cfg.data_dir / "exports", "fallback", reason

    conn = src.connect()
    if not conn.ok:
        return cfg.data_dir / "exports", "fallback", conn.message

    experts = src.experts_dir()
    if experts is None:
        reason = (
            "MetaTrader 5 is connected but its terminal data path could not be "
            "read, so the Experts folder is unknown."
        )
        return cfg.data_dir / "exports", "fallback", reason

    return experts, "mt5", None


def _sizing_note(cfg: AppConfig, row: dict, genome: Genome, guard: Guardrails) -> tuple[float | None, str]:
    """Estimate the account size needed for this strategy to actually place trades.

    Uses the cached history's median ATR to turn the genome's ATR-multiple stop
    into a currency distance, then mirrors the EA's sizing. Best-effort: if no
    bars are cached we return a generic pointer instead of a number.
    """
    generic = (
        "Position size is risk-based off your live balance, and the min-lot "
        "fallback (InpMinLotFallback, ON by default) trades the broker minimum "
        "lot when risk-sizing would round to zero — so the strategy still fires "
        "on a modest account. If you still see no trades, open MT5 Toolbox > "
        "Experts/Journal; the EA logs exactly why."
    )
    try:
        bars = BarCache(cfg.data_dir).load(row["symbol"], row["timeframe"])
    except Exception:
        bars = None
    if bars is None or len(bars) < 50:
        return None, generic

    med_atr = float(
        np.nanmedian(
            atr(bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy(), 14)
        )
    )
    sl_dist = genome.management.sl_atr * med_atr
    if sl_dist <= 0:
        return None, generic
    floor = min_account_to_trade(sl_dist, EURUSD_STD, max_risk_pct=DEFAULT_MAX_RISK_PCT)
    risk_based = min_account_for_risk_sizing(guard.risk_pct, sl_dist, EURUSD_STD)
    note = (
        f"Estimated sizing ({row['symbol']} {row['timeframe']}, standard-lot): the "
        f"strategy trades the broker minimum lot from about {floor:,.0f}, and sizes "
        f"fully by its {guard.risk_pct:.2%} risk from about {risk_based:,.0f} (account "
        f"currency). Below ~{floor:,.0f} the EA refuses and logs the balance needed. "
        f"Min-lot fallback is ON so a modest account still trades."
    )
    return floor, note


def _deploy_instructions(path: Path, name: str, row: dict, target: str, reason: str | None) -> str:
    if target == "fallback":
        return (
            f"MT5 was not detected, so the EA was NOT placed in your MetaTrader "
            f"Experts folder. {reason} It was saved to {path} instead. To use it: "
            f"either start MetaTrader 5 (logged in) and click DEPLOY again, or "
            f"manually copy {path.name} into your terminal's MQL5/Experts folder, "
            f"then open it in MetaEditor (F4) and compile with F7."
        )
    where = (
        "your MT5 Experts folder"
        if target == "mt5"
        else f"the folder you chose ({path.parent})"
    )
    return (
        f"EA written into {where}. "
        f"1. Open MetaEditor (F4 in MT5) and open {path.name}. "
        "2. Compile with F7. "
        f"3. In MT5, open a {row['symbol']} {row['timeframe']} chart and drag "
        f"{name} from Navigator > Expert Advisors onto it. "
        "4. Enable Algo Trading. The EA enforces its guardrails automatically."
    )


def create_app(cfg: AppConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Single drainer: keeps the worker's event queue flowing even with no
        # browser attached (a full pipe would block the worker's exit), and
        # broadcasts every event to all connected sockets.
        async def broadcast(event):
            dead = []
            for sock in app.state.sockets:
                try:
                    await sock.send_json(event)
                except Exception:
                    dead.append(sock)
            for sock in dead:
                app.state.sockets.discard(sock)

        async def pump():
            while True:
                for event in app.state.manager.drain():
                    await broadcast(event)
                for event in app.state.executor.drain():
                    await broadcast(event)
                await asyncio.sleep(POLL_INTERVAL)

        task = asyncio.create_task(pump())
        yield
        task.cancel()

    app = FastAPI(title="Alglory", version="0.1.0", lifespan=lifespan)
    app.state.cfg = cfg
    app.state.manager = CampaignManager(cfg.db_path, cfg.data_dir)
    app.state.executor = ExecutorManager()
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

    @app.post("/api/campaigns/{cid}/restart", status_code=202)
    def restart_campaign(cid: int):
        manager = app.state.manager
        if manager.is_running():
            raise HTTPException(409, "A campaign is already running; cancel it first.")
        row = vault().get_campaign(cid)
        if row is None:
            raise HTTPException(404, f"Campaign {cid} not found.")
        if row["status"] not in ("failed", "cancelled"):
            raise HTTPException(
                422, f"Campaign {cid} is {row['status']!r}; only failed or "
                "cancelled campaigns can be resumed."
            )
        if not row["progress_json"]:
            raise HTTPException(422, f"Campaign {cid} has no checkpoint to resume from.")
        cfg_c = CampaignConfig.model_validate_json(row["config_json"])
        try:
            manager.start(cfg_c, resume_campaign_id=cid)
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "resuming", "campaign_id": cid}

    @app.post("/api/executor/arm")
    def executor_arm(req: ExecutorArmRequest):
        mode = req.mode.lower()
        if mode not in ("dry_run", "demo", "live"):
            raise HTTPException(422, f"Unknown mode {req.mode!r}; valid: dry_run, demo, live.")
        if mode == "live" and not req.confirm_live:
            raise HTTPException(400, "Live mode requires confirm_live=true.")
        if req.preset not in GUARDRAIL_PRESETS:
            raise HTTPException(
                422, f"Unknown preset {req.preset!r}; valid: {sorted(GUARDRAIL_PRESETS)}"
            )
        if not req.strategy_ids:
            raise HTTPException(422, "No strategy_ids given.")
        runtimes = []
        for sid in req.strategy_ids:
            row = vault().get_strategy(sid)
            if row is None:
                raise HTTPException(404, f"Strategy {sid} not found.")
            runtimes.append(
                StrategyRuntime(
                    genome=Genome.from_json(row["genome_json"]),
                    symbol=row["symbol"],
                    timeframe=row["timeframe"],
                    guard=GUARDRAIL_PRESETS[req.preset],
                    magic=_EXECUTOR_MAGIC_BASE + sid,
                )
            )
        try:
            app.state.executor.arm_strategies(
                runtimes, mode, req.confirm_live,
                poll_seconds=req.poll_seconds,
                max_open_positions=req.max_open_positions,
            )
        except BusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ExecutorError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:  # MT5 not installed / not connected
            raise HTTPException(503, str(exc)) from exc
        return app.state.executor.status()

    @app.post("/api/executor/stop")
    def executor_stop():
        app.state.executor.stop()
        return {"status": "stopped"}

    @app.get("/api/executor/status")
    def executor_status():
        return app.state.executor.status()

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
        min_oos_net_profit: float | None = None,
        min_oos_profit_factor: float | None = None,
        min_oos_sharpe: float | None = None,
        min_oos_trades: float | None = None,
        max_oos_max_drawdown: float | None = None,
    ):
        try:
            return vault().list_strategies(
                symbol=symbol,
                timeframe=timeframe,
                tribe=tribe,
                sort=sort,
                desc=desc,
                min_oos_net_profit=min_oos_net_profit,
                min_oos_profit_factor=min_oos_profit_factor,
                min_oos_sharpe=min_oos_sharpe,
                min_oos_trades=min_oos_trades,
                max_oos_max_drawdown=max_oos_max_drawdown,
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
            out_dir, target, reason = Path(req.out_dir), "custom", None
        else:
            out_dir, target, reason = _resolve_deploy_dir(cfg)
        path = write_ea(code, name, out_dir)
        min_account, sizing_note = _sizing_note(cfg, row, genome, guard)
        return {
            "path": str(path),
            "target": target,
            "mt5_detected": target == "mt5",
            "reason": reason,
            "min_account_estimate": min_account,
            "min_lot_fallback": True,
            "sizing_note": sizing_note,
            "instructions": _deploy_instructions(path, name, row, target, reason)
            + " "
            + sizing_note,
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
