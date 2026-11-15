"""Worker process entry point: `python -m xray.core.worker`.

Reads one job spec from stdin, runs it, writes the output to the spec's
`result_path`, and reports progress as events on stdout.
"""

from __future__ import annotations

import importlib
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xray.core.jobs import JOBS, MARKER
from xray.core.memory import MemoryGuardError


def emit(event: dict[str, Any]) -> None:
    sys.stdout.write(MARKER + json.dumps(event) + "\n")
    sys.stdout.flush()


@dataclass
class JobContext:
    """What a job function gets besides its parameters."""

    run_dir: Path  # where the job may write extra output files

    def progress(self, stage: str, done: int | None = None, total: int | None = None) -> None:
        emit({"event": "progress", "stage": stage, "done": done, "total": total})


def selftest_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    """A job with no model, for testing the runner: succeed, fail or die on request."""
    action = params.get("action", "ok")
    if action == "crash":
        import os

        os._exit(3)
    if action == "memory":
        from xray.core.memory import guard_memory

        guard_memory(10**12, 2, {"ram_available_gb": 1.0, "ram_total_gb": 1.0}, "cpu")
    ctx.progress("working", 1, 2)
    backend = {"name": "none", "device": "cpu", "dtype": "none"}
    return {"manifest": {"backend": backend}, "result": {"ok": True}}


def main() -> int:
    spec = json.loads(sys.stdin.read())
    try:
        module_name, func_name = JOBS[spec["type"]].split(":")
        job = getattr(importlib.import_module(module_name), func_name)
        output = job(spec["params"], JobContext(Path(spec["result_path"]).parent))
        with open(spec["result_path"], "w") as f:
            json.dump(output, f, indent=2)
        emit({"event": "result"})
        return 0
    except MemoryGuardError as e:
        emit({"event": "error", "kind": "memory", "message": str(e)})
    except MemoryError:
        emit({"event": "error", "kind": "memory", "message": "The job ran out of memory."})
    except Exception as e:  # noqa: BLE001 - report every failure to the parent
        traceback.print_exc()
        kind = "memory" if "out of memory" in str(e).lower() else "error"
        emit({"event": "error", "kind": kind, "message": f"{type(e).__name__}: {e}"})
    return 1


if __name__ == "__main__":
    sys.exit(main())
