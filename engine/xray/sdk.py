"""The public Python API.

    import xray

    m = xray.Model("Qwen/Qwen2.5-1.5B")      # a local folder, or a repo id in the HF cache
    m.inspect()                               # anatomy, read from file headers
    run = m.timing(contexts=[128, 512])       # a recorded run
    run.result["contexts"][0]["decode"]["tokens_per_s"]
    print(m.report()["text"])                 # compact report of everything measured

Analyses that load the model (`timing`, `profile`, `quant`, `lens`, `attribute`,
`loops`, `edit`, `bench`) run in a separate worker process and are recorded, so
they also appear in `xray runs` and in the desktop app. Each returns a `Run`.
Everything else reads files and returns plain dictionaries.

Data lives under `$XRAY_HOME` (default `.xray/` in the current folder); pass
`home=` to use another folder.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xray.core import paths
from xray.core.jobs import run_job
from xray.core.storage import Storage
from xray.core.summary import compare_runs, summarize


@dataclass
class Run:
    """One recorded analysis."""

    id: str
    type: str
    result: dict[str, Any]
    manifest: dict[str, Any] = field(repr=False, default_factory=dict)

    def summary(self) -> list[dict[str, Any]]:
        """Headline numbers: a list of {label, value, format, better}."""
        return summarize(self.type, self.result)

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, type={self.type!r})"


def load_run(run_id: str, home: str | Path | None = None) -> Run:
    """Load a finished run by id."""
    with Storage(paths.db_path(paths.home(home))) as db:
        row = db.get_run(run_id)
        artifacts = db.artifacts(run_id)
    if row is None:
        raise KeyError(f"No run '{run_id}'.")
    if row["status"] != "done" or "result" not in artifacts:
        raise ValueError(f"Run {run_id} is {row['status']}: {row['error'] or 'it has no result'}.")
    data = json.loads(Path(artifacts["result"]).read_text())
    return Run(run_id, row["type"], data["result"], data["manifest"])


def runs(home: str | Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """Recorded runs, newest first: id, type, status, model_path, started_at, params."""
    with Storage(paths.db_path(paths.home(home))) as db:
        rows = db.list_runs(limit)
    for row in rows:
        row["params"] = json.loads(row.pop("params_json"))
    return rows


def compare(a: Run | str, b: Run | str, home: str | Path | None = None) -> list[dict[str, Any]]:
    """Two runs of the same kind, metric by metric: label, a, b, change, verdict."""
    ra = a if isinstance(a, Run) else load_run(a, home)
    rb = b if isinstance(b, Run) else load_run(b, home)
    return compare_runs(ra.type, ra.result, rb.type, rb.result)


class Model:
    """A local model and everything X-Ray can measure about it."""

    def __init__(self, model: str, home: str | Path | None = None, device: str = "auto") -> None:
        from xray.models.inspect import resolve_model_dir

        self.path = str(resolve_model_dir(model))  # raises FileNotFoundError; never downloads
        self.home = home
        self.device = device

    def __repr__(self) -> str:
        from xray.report import model_label

        return f"Model({model_label(self.path)!r})"

    # -- instant: read from files, nothing is loaded ----------------------------

    def inspect(self) -> dict[str, Any]:
        """Architecture, parameter breakdown and memory estimates."""
        from xray.models.inspect import inspect_model

        return inspect_model(self.path)

    def fit(self, context: int = 4096) -> dict[str, Any]:
        """Whether the model fits this machine, for analysis and for running."""
        from xray.core.device import system_info
        from xray.core.memory import fit_report

        return fit_report(self.inspect(), system_info(), context)

    def report(self, budget_tokens: int = 3000) -> dict[str, Any]:
        """Compact report of the latest runs. The text is in `["text"]`."""
        from xray.report import build_report

        return build_report(self.path, budget_tokens, self.home)

    def query(self, text: str) -> str:
        """One detail from saved runs, e.g. `layer 12` or `neuron 12 6553`."""
        from xray.report.query import run_query

        return run_query(self.path, text, self.home)

    def runs(self, run_type: str | None = None) -> list[dict[str, Any]]:
        """This model's recorded runs, newest first."""
        return [
            r for r in runs(self.home, 1000)
            if r["model_path"] == self.path and (run_type is None or r["type"] == run_type)
        ]  # fmt: skip

    def latest(self, run_type: str) -> Run | None:
        """The most recent finished run of one kind, or None."""
        for row in self.runs(run_type):
            if row["status"] == "done":
                return load_run(row["id"], self.home)
        return None

    def runtimes(self) -> list[dict[str, Any]]:
        """PyTorch speed next to llama.cpp and MLX, from the latest timing and bench runs."""
        from xray.core.summary import runtime_rows

        timing, bench = self.latest("timing"), self.latest("bench")
        return runtime_rows(timing.result if timing else None, bench.result if bench else None)

    # -- recorded analyses: run in a worker process -----------------------------

    def _run(self, job: str, on_progress: Callable[[dict], None] | None, **params: Any) -> Run:
        params = {k: v for k, v in params.items() if v is not None}
        out = run_job(
            job, {"model": self.path, "device": self.device, **params},
            home=self.home, on_event=on_progress,
        )  # fmt: skip
        return Run(out["run_id"], job, out["result"], out["manifest"])

    def timing(self, contexts: list[int] | None = None, on_progress=None, **options: Any) -> Run:
        """Time per layer, attention vs MLP, prefill vs decode."""
        return self._run("timing", on_progress, contexts=contexts, **options)

    def profile(
        self, sets: list[str] | None = None, limit: int | None = None, new_tokens: int = 16,
        ablation_prompts: int = 16, on_progress=None, **options: Any,
    ) -> Run:  # fmt: skip
        """Which neurons fire for which prompt set, and which layers are redundant.

        `sets` are built-in names (coding, chat, math, instruction) or prompt files.
        """
        sets = sets or ["coding", "chat", "math", "instruction"]
        return self._run(
            "profile", on_progress, sets=sets, limit=limit, new_tokens=new_tokens,
            ablation_prompts=ablation_prompts, **options,
        )  # fmt: skip

    def quant(self, target_bits: list[int] | None = None, on_progress=None, **options: Any) -> Run:
        """Quantization sensitivity per weight matrix, and a mixed-precision plan."""
        return self._run("quant", on_progress, target_bits=target_bits, **options)

    def lens(self, prompt: str, new_tokens: int = 12, chat: bool = True, on_progress=None) -> Run:
        """Logit lens: at which layer each generated token is decided."""
        return self._run("lens", on_progress, prompt=prompt, new_tokens=new_tokens, chat=chat)

    def attribute(self, prompt: str, chat: bool = True, on_progress=None) -> Run:
        """Which blocks, heads and neurons produced the next token."""
        return self._run("attribution", on_progress, prompt=prompt, chat=chat)

    def loops(
        self, sets: list[str] | None = None, limit: int = 10, new_tokens: int = 120,
        chat: bool = True, on_progress=None,
    ) -> Run:  # fmt: skip
        """Find prompts that make the model repeat itself, and what drives the loop."""
        return self._run(
            "loops", on_progress, sets=sets, limit=limit, new_tokens=new_tokens, chat=chat
        )

    def edit(
        self, commands: list[str], heal: bool = False, save: bool = False,
        run_code: bool = False, on_progress=None,
    ) -> Run:  # fmt: skip
        """Apply edit commands to a copy and measure it against the untouched base."""
        from xray.patch.commands import parse_commands

        parse_commands(commands)  # fail on a typo before starting a worker
        return self._run(
            "edit", on_progress, commands=list(commands), heal={"steps": 150} if heal else None,
            save=save, run_code=run_code,
        )  # fmt: skip

    def bench(
        self, llama_cpp: str | None = None, types: list[str] | None = None, mlx: bool = True,
        on_progress=None,
    ) -> Run:  # fmt: skip
        """Speed in llama.cpp (needs a built checkout) and MLX (Apple Silicon)."""
        params = {
            k: v
            for k, v in {"llama_cpp": llama_cpp, "types": types, "mlx": mlx}.items()
            if v is not None
        }
        out = run_job("bench", {"model": self.path, **params}, home=self.home, on_event=on_progress)
        return Run(out["run_id"], "bench", out["result"], out["manifest"])
