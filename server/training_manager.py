"""Owns the training subprocess lifecycle and bridges its multiprocessing.Queue
messages into the asyncio event loop running the web server."""

import asyncio
import multiprocessing as mp
import threading
from typing import Callable, Optional

from server.model_interface import ArchitectureSpec
from server import train_worker

_ctx = mp.get_context("spawn")


class TrainingManager:
    def __init__(self, loop: asyncio.AbstractEventLoop, on_message: Callable[[dict], "asyncio.Future"]):
        """on_message: async callable, invoked (from the event loop) with each
        message dict coming out of the training subprocess."""
        self._loop = loop
        self._on_message = on_message
        self._process: Optional[mp.process.BaseProcess] = None
        self._stop_event: Optional[mp.synchronize.Event] = None
        self._queue: Optional[mp.queues.Queue] = None
        self._bridge_thread: Optional[threading.Thread] = None

    def start(self, spec: ArchitectureSpec, checkpoint_path: str) -> None:
        self.stop()

        self._stop_event = _ctx.Event()
        self._queue = _ctx.Queue()
        self._process = _ctx.Process(
            target=train_worker.run,
            args=(spec.to_dict(), self._queue, self._stop_event, checkpoint_path),
            daemon=True,
        )
        self._process.start()

        self._bridge_thread = threading.Thread(target=self._bridge_loop, daemon=True)
        self._bridge_thread.start()

    def stop(self) -> None:
        if self._process is not None and self._process.is_alive():
            if self._stop_event is not None:
                self._stop_event.set()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
        self._process = None

    def _bridge_loop(self) -> None:
        queue = self._queue
        assert queue is not None
        while True:
            try:
                msg = queue.get(timeout=1)
            except Exception:
                if self._process is not None and not self._process.is_alive():
                    return
                continue
            future = asyncio.run_coroutine_threadsafe(self._on_message(msg), self._loop)
            try:
                future.result()
            except Exception:
                pass
            if msg.get("type") == "training_status" and msg.get("state") in (
                "idle",
                "stopped",
                "error",
            ):
                return
