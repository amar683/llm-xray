"""Job runner: every heavy analysis runs in its own worker process.

If a model run crashes or the OS kills it for using too much memory, only the
worker dies. The caller (CLI today, the API server later) gets a clean error
and the run is recorded as failed.

Protocol: the parent writes one JSON job spec to the worker's stdin. The worker
prints one event per line on stdout, prefixed with MARKER so stray library
output cannot be mistaken for an event. Worker stderr goes to `worker.log`.
"""

from __future__ import annotations

import json
import secrets
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from xray.core import paths
from xray.core.storage import Storage

MARKER = "@@XRAY "

# Job type -> "module:function". Resolved inside the worker, so the parent
# process never imports torch.
JOBS: dict[str, str] = {
    "timing": "xray.analyzers.timing:timing_job",
    "profile": "xray.analyzers.profile:profile_job",
    "quant": "xray.analyzers.quant:quant_job",
    "lens": "xray.analyzers.interp:lens_job",
    "attribution": "xray.analyzers.interp:attribution_job",
    "loops": "xray.analyzers.loops:loops_job",
    "edit": "xray.analyzers.edit:edit_job",
    "bench": "xray.backends.runtimes:bench_job",
    "_selftest": "xray.core.worker:selftest_job",
}

Event = dict[str, Any]


class JobError(RuntimeError):
    """The job failed. `kind` is 'error', 'memory' or 'crash'."""

    def __init__(self, message: str, kind: str = "error", run_id: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.run_id = run_id


class JobCancelled(RuntimeError):
    def __init__(self, run_id: str) -> None:
        super().__init__(f"Run {run_id} was cancelled.")
        self.run_id = run_id


def new_run_id() -> str:
    """Sortable by start time, unique enough for one machine."""
    return f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def run_job(
    job_type: str,
    params: dict[str, Any],
    *,
    home: str | Path | None = None,
    on_event: Callable[[Event], None] | None = None,
    run_id: str | None = None,
    on_process: Callable[[subprocess.Popen], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Run a job in a worker process, record it, and return its output.

    Returns {"run_id", "manifest", "result"}. Raises JobError or JobCancelled.

    A caller that queues jobs passes the `run_id` of a run it already created
    with status "queued". To cancel from another thread, terminate the process
    handed to `on_process` and make `cancelled()` return True.
    """
    if job_type not in JOBS:
        raise ValueError(f"Unknown job type '{job_type}'. Known: {', '.join(sorted(JOBS))}")

    home_dir = paths.home(home)
    queued = run_id is not None
    run_id = run_id or new_run_id()
    out_dir = paths.run_dir(home_dir, run_id)
    result_path = out_dir / "result.json"
    spec = {"type": job_type, "params": params, "result_path": str(result_path)}

    with Storage(paths.db_path(home_dir)) as db:
        if queued:
            db.set_status(run_id, "running")
        else:
            db.create_run(run_id, job_type, params)
        failure: Event | None = None
        finished = False

        with (out_dir / "worker.log").open("w") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "xray.core.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
            )
            db.set_pid(run_id, proc.pid)
            if on_process:
                on_process(proc)
            try:
                proc.stdin.write(json.dumps(spec))
                proc.stdin.close()
                for line in proc.stdout:
                    if not line.startswith(MARKER):
                        continue
                    event = json.loads(line[len(MARKER) :])
                    if event["event"] == "error":
                        failure = event
                    elif event["event"] == "result":
                        finished = True
                    elif on_event:
                        on_event(event)
                code = proc.wait()
                proc.stdout.close()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait()
                proc.stdout.close()
                db.finish_run(run_id, "cancelled")
                raise JobCancelled(run_id) from None

        if cancelled():
            db.finish_run(run_id, "cancelled")
            raise JobCancelled(run_id)
        if failure:
            db.finish_run(run_id, "failed", error=failure["message"])
            raise JobError(failure["message"], failure.get("kind", "error"), run_id)
        if code != 0 or not finished:
            # No error event: the process died without a chance to report.
            reason = (
                f"The worker was killed by signal {-code}, most likely for using too much memory."
                if code < 0
                else f"The worker exited unexpectedly (code {code}). See {out_dir / 'worker.log'}."
            )
            db.finish_run(run_id, "failed", error=reason)
            raise JobError(reason, "crash", run_id)

        output = json.loads(result_path.read_text())
        model_id = db.upsert_model(output["model"]) if output.get("model") else None
        db.add_artifact(run_id, "result", result_path)
        for kind, filename in output.get("artifacts", {}).items():
            db.add_artifact(run_id, kind, out_dir / filename)
        if output.get("layer_stats"):
            db.add_layer_stats(run_id, output["layer_stats"])
        db.finish_run(run_id, "done", manifest=output["manifest"], model_id=model_id)

    return {"run_id": run_id, "manifest": output["manifest"], "result": output["result"]}
