"""Campaign worker process management.

The engine runs in a separate process so a crash can never take down the
server, and so breeding keeps running while the browser is closed. Events
flow back through a multiprocessing queue that the server drains and
broadcasts to WebSocket clients.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
from pathlib import Path

from alglory.data.cache import BarCache
from alglory.evolve.campaign import CampaignConfig, run_campaign
from alglory.vault.db import Vault


def _worker_main(
    cfg_json: str,
    db_path: str,
    data_dir: str,
    q,
    stop_event,
    pause_event,
    resume_campaign_id: int | None = None,
) -> None:
    cfg = CampaignConfig.model_validate_json(cfg_json)
    vault = Vault(Path(db_path))
    cache = BarCache(Path(data_dir))
    run_campaign(
        cfg,
        vault,
        cache,
        emit=q.put,
        should_stop=stop_event.is_set,
        should_pause=pause_event.is_set,
        resume_campaign_id=resume_campaign_id,
    )


class BusyError(Exception):
    pass


class CampaignManager:
    def __init__(self, db_path: Path, data_dir: Path):
        self._db_path = str(db_path)
        self._data_dir = str(data_dir)
        self._ctx = mp.get_context("spawn")
        self._proc: mp.Process | None = None
        self._queue = None
        self._stop_event = None
        self._pause_event = None
        self.current_campaign_id: int | None = None
        self.last_events: list[dict] = []

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self, cfg: CampaignConfig, resume_campaign_id: int | None = None) -> None:
        if self.is_running():
            raise BusyError("A campaign is already running; cancel it first.")
        self._queue = self._ctx.Queue()
        self._stop_event = self._ctx.Event()
        self._pause_event = self._ctx.Event()
        self.current_campaign_id = resume_campaign_id
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(
                cfg.model_dump_json(),
                self._db_path,
                self._data_dir,
                self._queue,
                self._stop_event,
                self._pause_event,
                resume_campaign_id,
            ),
            daemon=True,
        )
        self._proc.start()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._pause_event is not None:
            self._pause_event.clear()  # a paused campaign must still cancel

    def pause(self) -> None:
        if self._pause_event is not None:
            self._pause_event.set()

    def resume(self) -> None:
        if self._pause_event is not None:
            self._pause_event.clear()

    def is_paused(self) -> bool:
        return self._pause_event is not None and self._pause_event.is_set()

    def drain(self) -> list[dict]:
        """Pull all pending events from the worker (non-blocking)."""
        events: list[dict] = []
        if self._queue is None:
            return events
        while True:
            try:
                event = self._queue.get_nowait()
            except (queue.Empty, ValueError):
                break
            if event.get("type") == "campaign_started":
                self.current_campaign_id = event.get("campaign_id")
            events.append(event)
        # keep a small tail so late-joining sockets can catch up
        if events:
            self.last_events = (self.last_events + events)[-50:]
        return events
