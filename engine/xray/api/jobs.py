"""Job manager for the API server: one job at a time, the rest wait in a queue."""

from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from xray.core import paths
from xray.core.jobs import JOBS, JobCancelled, JobError, new_run_id, run_job
from xray.core.storage import Storage

Event = dict[str, Any]


class JobManager:
    """Queues jobs and runs them one by one on a background thread.

    Only one model fits in memory comfortably, so jobs never run in parallel.
    Every state change is passed to `publish` (called from the worker thread).
    """

    def __init__(self, home: Path, publish: Callable[[Event], None]) -> None:
        self.home = home
        self.publish = publish
        self._queue: queue.Queue[tuple[str, str, dict] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._active: dict[str, Event] = {}  # run id -> latest event
        self._cancelled: set[str] = set()
        self._proc: tuple[str, subprocess.Popen] | None = None
        self._thread = threading.Thread(target=self._loop, name="xray-jobs", daemon=True)
        self._thread.start()

    def submit(self, job_type: str, params: dict[str, Any]) -> str:
        if job_type not in JOBS or job_type.startswith("_"):
            raise ValueError(f"Unknown job type '{job_type}'.")
        run_id = new_run_id()
        with Storage(paths.db_path(self.home)) as db:
            db.create_run(run_id, job_type, params, status="queued")
        self._emit(run_id, job_type, "queued")
        self._queue.put((run_id, job_type, params))
        return run_id

    def cancel(self, run_id: str) -> bool:
        """Cancel a queued or running job. False if it is not active."""
        with self._lock:
            if run_id not in self._active:
                return False
            self._cancelled.add(run_id)
            if self._proc and self._proc[0] == run_id:
                self._proc[1].terminate()
        return True

    def active(self) -> list[Event]:
        with self._lock:
            return list(self._active.values())

    def shutdown(self) -> None:
        with self._lock:
            self._cancelled.update(self._active)
            if self._proc:
                self._proc[1].terminate()
        self._queue.put(None)
        self._thread.join(timeout=10)

    def _emit(self, run_id: str, job_type: str, event: str, **extra: Any) -> None:
        payload = {"run_id": run_id, "type": job_type, "event": event, **extra}
        with self._lock:
            if event in {"done", "failed", "cancelled"}:
                self._active.pop(run_id, None)
                self._cancelled.discard(run_id)
            else:
                self._active[run_id] = payload
        self.publish(payload)

    def _loop(self) -> None:
        while (item := self._queue.get()) is not None:
            run_id, job_type, params = item
            if run_id in self._cancelled:  # cancelled while still queued
                with Storage(paths.db_path(self.home)) as db:
                    db.finish_run(run_id, "cancelled")
                self._emit(run_id, job_type, "cancelled")
                continue

            def on_process(proc: subprocess.Popen, run_id: str = run_id) -> None:
                with self._lock:
                    self._proc = (run_id, proc)

            def on_event(e: Event, run_id: str = run_id, job_type: str = job_type) -> None:
                self._emit(
                    run_id, job_type, "progress",
                    stage=e.get("stage"), done=e.get("done"), total=e.get("total"),
                )  # fmt: skip

            self._emit(run_id, job_type, "started")
            try:
                run_job(
                    job_type,
                    params,
                    home=self.home,
                    run_id=run_id,
                    on_event=on_event,
                    on_process=on_process,
                    cancelled=lambda run_id=run_id: run_id in self._cancelled,
                )
                self._emit(run_id, job_type, "done")
            except JobCancelled:
                self._emit(run_id, job_type, "cancelled")
            except JobError as e:
                self._emit(run_id, job_type, "failed", error=str(e), kind=e.kind)
            except Exception as e:  # noqa: BLE001 - never let the job thread die
                with Storage(paths.db_path(self.home)) as db:
                    db.finish_run(run_id, "failed", error=f"{type(e).__name__}: {e}")
                self._emit(run_id, job_type, "failed", error=str(e), kind="error")
            finally:
                with self._lock:
                    self._proc = None
