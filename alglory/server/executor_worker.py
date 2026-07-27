"""Executor worker: runs the LiveExecutor on a background thread.

Unlike the campaign worker (a separate process), the executor runs in a thread
so it can hold the single MT5 terminal connection. On each poll it asks the bar
clock for the latest closed bars per strategy, and only acts when a new closed
bar has appeared — emitting the executor's intent as an event the server
broadcasts to WebSocket clients. `stop()` is the kill switch: it flattens every
strategy and halts.
"""

from __future__ import annotations

import queue
import threading

from alglory.live.clock import MT5BarClock
from alglory.live.executor import LiveExecutor, Mode
from alglory.server.worker import BusyError


class ExecutorManager:
    def __init__(self, poll_seconds: float = 5.0) -> None:
        self._executor: LiveExecutor | None = None
        self._clock = None
        self._runtimes: list = []
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._queue: queue.Queue = queue.Queue()
        self._last_bar: dict[int, str] = {}
        self._poll = poll_seconds
        self._armed = False

    # ---- lifecycle --------------------------------------------------------

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def arm(self, executor: LiveExecutor, clock, *, poll_seconds: float | None = None, start: bool = True) -> None:
        if self.is_running():
            raise BusyError("An executor is already running; stop it first.")
        self._executor = executor
        self._clock = clock
        self._runtimes = executor.runtimes
        self._last_bar = {}
        self._armed = True
        if poll_seconds is not None:
            self._poll = poll_seconds
        self._stop.clear()
        if start:
            self._thread = threading.Thread(target=self._loop, name="alglory-executor", daemon=True)
            self._thread.start()

    def arm_strategies(self, runtimes, mode: str, confirm_live: bool, *, poll_seconds: float = 5.0) -> None:
        """Build the MT5-backed executor for the given runtimes and start it."""
        source = self._build_source()
        executor = LiveExecutor(source, runtimes, mode=Mode(mode), confirm_live=confirm_live)
        self.arm(executor, MT5BarClock(source), poll_seconds=poll_seconds)

    def _build_source(self):
        from alglory.data.mt5source import MT5Source

        src = MT5Source()
        if not src.available():
            raise RuntimeError(
                "MetaTrader5 package not installed — the live executor needs a running terminal."
            )
        conn = src.connect()
        if not conn.ok:
            raise RuntimeError(conn.message)
        return src

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self._executor is not None:
            self._executor.stop()  # flatten + halt every runtime
        self._armed = False

    # ---- loop -------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # never let the thread die silently
                self._queue.put({"type": "executor_error", "error": str(exc)})
            self._stop.wait(self._poll)

    def tick(self) -> None:
        """One poll: act on any strategy that has a newly closed bar."""
        if self._executor is None or self._clock is None:
            return
        for rt in self._runtimes:
            bars = self._clock.latest_closed_bars(rt.symbol, rt.timeframe)
            if bars is None or len(bars) == 0:
                continue
            stamp = str(bars["time"].iloc[-1])
            if self._last_bar.get(rt.magic) == stamp:
                continue  # no new closed bar since last poll
            self._last_bar[rt.magic] = stamp
            intent = self._executor.on_bar(rt, bars)
            self._queue.put(self._event(rt, intent))

    # ---- introspection ----------------------------------------------------

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "armed": self._armed,
            "mode": self._executor.mode.value if self._executor else None,
            "strategies": self._executor.status() if self._executor else [],
        }

    def drain(self) -> list[dict]:
        events: list[dict] = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events

    @staticmethod
    def _event(rt, intent) -> dict:
        return {
            "type": "executor_intent",
            "magic": rt.magic,
            "symbol": rt.symbol,
            "action": intent.action.value,
            "lots": intent.lots,
            "direction": intent.direction,
            "reason": intent.reason,
        }
