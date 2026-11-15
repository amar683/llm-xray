"""Local HTTP + WebSocket API for the desktop app.

Binds to 127.0.0.1 only. Every request must carry the per-launch token
(`Authorization: Bearer <token>`; `?token=` for the WebSocket, which cannot set
headers from a browser).
"""

from __future__ import annotations

import asyncio
import json
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from xray import __version__
from xray.api.jobs import JobManager
from xray.core import paths
from xray.core.storage import Storage
from xray.data import builtin_sets, load_prompt_set

# The Vite dev server and the Tauri webview.
ALLOWED_ORIGINS = [
    "http://localhost:1420",
    "http://127.0.0.1:1420",
    "tauri://localhost",
    "http://tauri.localhost",
]
SUPPORTED_MODEL_TYPES = {"llama", "qwen2", "qwen3"}


class ModelRequest(BaseModel):
    model: str
    context: int = 4096


class PatchRequest(BaseModel):
    model: str
    name: str = ""
    commands: list[str]
    parent_id: str | None = None


class ReportRequest(BaseModel):
    model: str
    budget: int = 3000


class QueryRequest(BaseModel):
    model: str
    query: str


class AdvisorRequest(BaseModel):
    model: str
    goal: str
    budget: int = 3000


class JobRequest(BaseModel):
    type: str
    params: dict[str, Any]


def _local_models() -> list[dict[str, Any]]:
    """Supported models already in the local Hugging Face cache. Never downloads."""
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.errors import CacheNotFound

    try:
        cache = scan_cache_dir()
    except CacheNotFound:
        return []
    found = []
    for repo in cache.repos:
        if repo.repo_type != "model" or not repo.revisions:
            continue
        snapshot = max(repo.revisions, key=lambda r: r.last_modified).snapshot_path
        config = snapshot / "config.json"
        if not config.exists() or not any(snapshot.glob("*.safetensors")):
            continue
        cfg = json.loads(config.read_text())
        found.append(
            {
                "id": repo.repo_id,
                "path": str(snapshot),
                "model_type": cfg.get("model_type"),
                "n_layers": cfg.get("num_hidden_layers"),
                "size_bytes": repo.size_on_disk,
                "supported": cfg.get("model_type") in SUPPORTED_MODEL_TYPES,
            }
        )
    return sorted(found, key=lambda m: (not m["supported"], m["id"].lower()))


def create_app(home: str | Path | None = None, token: str | None = None) -> FastAPI:
    home_dir = paths.home(home)
    token = token or secrets.token_urlsafe(24)
    sockets: set[WebSocket] = set()

    def storage() -> Storage:
        return Storage(paths.db_path(home_dir))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        loop = asyncio.get_running_loop()

        async def broadcast(event: dict) -> None:
            for ws in list(sockets):
                try:
                    await ws.send_json(event)
                except Exception:  # noqa: BLE001 - a dead socket must not stop the others
                    sockets.discard(ws)

        def publish(event: dict) -> None:  # called from the job thread
            asyncio.run_coroutine_threadsafe(broadcast(event), loop)

        with storage() as db:
            db.fail_unfinished("The engine stopped before this run finished.")
        app.state.jobs = JobManager(home_dir, publish)
        yield
        app.state.jobs.shutdown()

    app = FastAPI(title="LLM X-Ray engine", version=__version__, lifespan=lifespan)
    app.state.token = token
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type"],
    )

    def auth(request: Request) -> None:
        header = request.headers.get("authorization", "")
        if not secrets.compare_digest(header, f"Bearer {token}"):
            raise HTTPException(401, "Missing or wrong engine token.")

    guarded = [Depends(auth)]

    def get_run(run_id: str) -> tuple[dict, dict[str, str]]:
        with storage() as db:
            run = db.get_run(run_id)
            artifacts = db.artifacts(run_id)
        if run is None:
            raise HTTPException(404, f"No run '{run_id}'.")
        return run, artifacts

    def get_arrays(run_id: str):
        from xray.analyzers.profile import load_arrays

        _, artifacts = get_run(run_id)
        if "activations" not in artifacts:
            raise HTTPException(404, "This run has no per-neuron data.")
        return load_arrays(Path(artifacts["activations"]))

    def anatomy_or_404(model: str) -> dict:
        from xray.models.inspect import inspect_model

        try:
            return inspect_model(model)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    # -- basics -----------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/system", dependencies=guarded)
    def system() -> dict:
        from xray.core.device import system_info

        return system_info()

    @app.get("/prompt-sets", dependencies=guarded)
    def prompt_sets() -> list[dict]:
        return [
            {"name": n, "prompts": len(ps.prompts), "example": ps.prompts[0]}
            for n in builtin_sets()
            for ps in [load_prompt_set(n)]
        ]

    # -- models -----------------------------------------------------------------

    @app.get("/models", dependencies=guarded)
    def models() -> list[dict]:
        return _local_models()

    @app.post("/models/inspect", dependencies=guarded)
    def inspect(req: ModelRequest) -> dict:
        return anatomy_or_404(req.model)

    @app.post("/models/fit", dependencies=guarded)
    def fit(req: ModelRequest) -> dict:
        from xray.core.device import system_info
        from xray.core.memory import fit_report

        return fit_report(anatomy_or_404(req.model), system_info(), req.context)

    @app.post("/models/runtimes", dependencies=guarded)
    def runtimes(req: ModelRequest) -> dict:
        """PyTorch speed next to llama.cpp and MLX, from the latest timing and bench runs."""
        import os

        from xray.core.summary import runtime_rows
        from xray.report import collect

        try:
            latest = collect(req.model, home_dir)["latest"]
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        timing, bench = latest.get("timing"), latest.get("bench")
        return {
            "rows": runtime_rows(
                timing["result"] if timing else None, bench["result"] if bench else None
            ),
            "bench": {
                "id": bench["id"],
                "started_at": bench["started_at"],
                "n_prompt": bench["result"]["n_prompt"],
                "n_gen": bench["result"]["n_gen"],
                "skipped": bench["result"]["skipped"],
            }
            if bench
            else None,
            "has_timing": timing is not None,
            # Whether the engine already knows a llama.cpp folder (the path itself is not returned).
            "llama_cpp_configured": bool(os.environ.get("XRAY_LLAMA_CPP")),
        }

    @app.post("/models/cost-value", dependencies=guarded)
    def cost_value(req: ModelRequest) -> dict:
        """Per layer: what it costs (time, size) next to what it is worth
        (damage when skipped, damage when quantized), from this model's most
        recent timing, profile and quant runs."""
        anatomy = anatomy_or_404(req.model)
        latest: dict[str, dict] = {}
        with storage() as db:
            for row in db.list_runs(500):
                if (
                    row["status"] == "done"
                    and row["model_path"] == anatomy["path"]
                    and row["type"] not in latest
                ):
                    result = Path(db.artifacts(row["id"])["result"])
                    latest[row["type"]] = {
                        "id": row["id"],
                        "started_at": row["started_at"],
                        "result": json.loads(result.read_text())["result"],
                    }

        layers = [
            {"layer": layer["index"], "params": layer["params"], "bytes": layer["bytes"]}
            for layer in anatomy["layers"]
        ]
        if "timing" in latest:
            ctx = latest["timing"]["result"]["contexts"][0]
            for row, t in zip(layers, ctx["decode"]["layers"], strict=True):
                row["decode_ms"] = t["est_ms"]
                row["decode_share"] = t["share"]
        redundancy = (latest.get("profile") or {}).get("result", {}).get("redundancy")
        if redundancy:
            for row, r in zip(layers, redundancy["layers"], strict=True):
                row["skip_kl"] = r["drop_kl"]
                row["skip_top1"] = r["drop_top1_agreement"]
                row["in_out_cosine"] = r["in_out_cosine"]
        if "quant" in latest:
            q = latest["quant"]["result"]
            for row, r in zip(layers, q["layers"], strict=True):
                row["quant_kl"] = r["kl"]
        return {
            "layers": layers,
            "sources": {
                k: {"id": v["id"], "started_at": v["started_at"]} for k, v in latest.items()
            },
            "missing": [k for k in ("timing", "profile", "quant") if k not in latest],
            "quant_probe_bits": latest["quant"]["result"]["settings"]["probe_bits"]
            if "quant" in latest
            else None,
        }

    # -- jobs and runs ----------------------------------------------------------

    @app.post("/jobs", dependencies=guarded)
    def submit(req: JobRequest, request: Request) -> dict:
        try:
            return {"run_id": request.app.state.jobs.submit(req.type, req.params)}
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.get("/jobs", dependencies=guarded)
    def active_jobs(request: Request) -> list[dict]:
        return request.app.state.jobs.active()

    @app.delete("/jobs/{run_id}", dependencies=guarded)
    def cancel(run_id: str, request: Request) -> dict:
        if not request.app.state.jobs.cancel(run_id):
            raise HTTPException(404, f"Run '{run_id}' is not queued or running.")
        return {"cancelling": run_id}

    @app.get("/runs", dependencies=guarded)
    def runs(limit: int = Query(50, ge=1, le=500)) -> list[dict]:
        with storage() as db:
            rows = db.list_runs(limit)
        for row in rows:
            row["params"] = json.loads(row.pop("params_json"))
        return rows

    @app.get("/runs/compare", dependencies=guarded)
    def compare(a: str, b: str) -> dict:
        from xray.core.summary import compare_runs

        loaded = []
        for run_id in (a, b):
            run, artifacts = get_run(run_id)
            if run["status"] != "done" or "result" not in artifacts:
                raise HTTPException(400, f"Run '{run_id}' has no result to compare.")
            loaded.append(
                (run["type"], json.loads(Path(artifacts["result"]).read_text())["result"])
            )
        try:
            rows = compare_runs(loaded[0][0], loaded[0][1], loaded[1][0], loaded[1][1])
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {"a": a, "b": b, "type": loaded[0][0], "rows": rows}

    @app.get("/runs/{run_id}", dependencies=guarded)
    def run_detail(run_id: str) -> dict:
        run, artifacts = get_run(run_id)
        run["params"] = json.loads(run.pop("params_json"))
        manifest = run.pop("manifest_json")
        run["manifest"] = json.loads(manifest) if manifest else None
        result = None
        if run["status"] == "done" and "result" in artifacts:
            result = json.loads(Path(artifacts["result"]).read_text())["result"]
        return {"run": run, "result": result}

    # -- report, queries, advisor -----------------------------------------------

    @app.post("/report", dependencies=guarded)
    def report(req: ReportRequest) -> dict:
        from xray.report import build_report

        try:
            r = build_report(req.model, req.budget, home_dir)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        r.pop("aliases")  # local file paths: not needed by the UI
        return r

    @app.post("/query", dependencies=guarded)
    def query(req: QueryRequest) -> dict:
        from xray.report.query import QueryError, run_query

        try:
            return {"query": req.query, "answer": run_query(req.model, req.query, home_dir)}
        except QueryError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.get("/advisor/status", dependencies=guarded)
    def advisor_status() -> dict:
        """Whether the optional advisor could run. Never reveals the credential."""
        import importlib.util
        import os

        return {
            "sdk_installed": importlib.util.find_spec("anthropic") is not None,
            "credential_in_environment": bool(
                os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            ),
        }

    @app.post("/advisor/preview", dependencies=guarded)
    def advisor_preview(req: AdvisorRequest) -> dict:
        """Exactly what `/advisor/send` would send. Sends nothing."""
        from xray.report.advisor import AdvisorError, build_request

        try:
            request = build_request(req.model, req.goal, req.budget, home_dir)
        except AdvisorError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        request.pop("aliases")
        return request

    @app.post("/advisor/send", dependencies=guarded)
    def advisor_send(req: AdvisorRequest, request: Request) -> dict:
        from xray.report.advisor import AdvisorError, run_advisor

        try:
            return run_advisor(
                req.model, req.goal, req.budget, home_dir,
                client=getattr(request.app.state, "advisor_client", None),
            )  # fmt: skip
        except AdvisorError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        except Exception as e:  # noqa: BLE001 - API, auth and network errors
            raise HTTPException(
                502, f"The Claude API request failed: {type(e).__name__}: {e}"
            ) from e

    # -- patches ----------------------------------------------------------------

    @app.get("/patches", dependencies=guarded)
    def patches(model: str | None = None) -> list[dict]:
        from xray.patch.history import list_patches

        try:
            return list_patches(model, home=home_dir)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e

    @app.post("/patches", dependencies=guarded)
    def add_patch(req: PatchRequest) -> dict:
        from xray.patch.commands import PatchError
        from xray.patch.history import create_patch

        try:
            return create_patch(req.model, req.name, req.commands, req.parent_id, home=home_dir)
        except (PatchError, ValueError) as e:
            raise HTTPException(400, str(e)) from e
        except (KeyError, FileNotFoundError) as e:
            raise HTTPException(404, str(e).strip("'\"")) from e

    @app.delete("/patches/{patch_id}", dependencies=guarded)
    def remove_patch(patch_id: str) -> dict:
        with storage() as db:
            if db.get_patch(patch_id) is None:
                raise HTTPException(404, f"No patch '{patch_id}'.")
            try:
                db.delete_patch(patch_id)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
        return {"deleted": patch_id}

    # -- per-neuron data (profile runs) -----------------------------------------

    @app.get("/runs/{run_id}/heatmap", dependencies=guarded)
    def heatmap(
        run_id: str,
        set_name: str = Query(alias="set"),
        mode: str = Query("rate", pattern="^(rate|specific)$"),
        bins: int = Query(200, ge=10, le=1000),
    ) -> dict:
        """Layers x bins grid. Each layer's neurons are sorted from highest to
        lowest value, then averaged into `bins` columns.

        rate: how often each neuron fires in this set.
        specific: firing rate in this set minus the rate in all other sets.
        """
        from xray.analyzers.profile import firing_rate, pool

        by_set = get_arrays(run_id)
        if set_name not in by_set:
            raise HTTPException(404, f"Run has no set '{set_name}'.")
        values = firing_rate(by_set[set_name])
        if mode == "specific":
            others = [a for n, a in by_set.items() if n != set_name]
            if not others:
                raise HTTPException(400, "Need at least two sets to compare.")
            values = values - firing_rate(pool(others))
        values = -np.sort(-values, axis=1)
        n_layers, n_neurons = values.shape
        bins = min(bins, n_neurons)
        edges = np.linspace(0, n_neurons, bins + 1).astype(int)
        grid = np.stack(
            [values[:, a:b].mean(axis=1) for a, b in zip(edges, edges[1:], strict=False)], axis=1
        )
        return {
            "set": set_name,
            "mode": mode,
            "layers": n_layers,
            "neurons": n_neurons,
            "bins": bins,
            "min": float(grid.min()),
            "max": float(grid.max()),
            "values": np.round(grid, 4).tolist(),
        }

    @app.get("/runs/{run_id}/neurons", dependencies=guarded)
    def neurons(
        run_id: str,
        layer: int,
        set_name: str = Query(alias="set"),
        top: int = Query(10, ge=1, le=200),
    ) -> list[dict]:
        from xray.analyzers.profile import top_neurons

        by_set = get_arrays(run_id)
        n_layers = next(iter(by_set.values()))["sum"].shape[0]
        if set_name not in by_set or not 0 <= layer < n_layers:
            raise HTTPException(404, "Unknown set or layer.")
        return top_neurons(by_set, set_name, layer, top)

    @app.get("/runs/{run_id}/neuron", dependencies=guarded)
    def neuron(run_id: str, layer: int, neuron: int) -> dict:
        from xray.analyzers.profile import neuron_detail

        by_set = get_arrays(run_id)
        n_layers, n_neurons = next(iter(by_set.values()))["sum"].shape
        if not (0 <= layer < n_layers and 0 <= neuron < n_neurons):
            raise HTTPException(404, "Unknown layer or neuron.")
        return neuron_detail(by_set, layer, neuron)

    # -- live events ------------------------------------------------------------

    @app.websocket("/events")
    async def events(ws: WebSocket, token_param: str = Query("", alias="token")) -> None:
        if not secrets.compare_digest(token_param, token):
            await ws.close(code=4401)
            return
        await ws.accept()
        sockets.add(ws)
        try:
            for event in ws.app.state.jobs.active():
                await ws.send_json(event)
            while True:
                await ws.receive_text()  # clients only listen; this detects disconnects
        except WebSocketDisconnect:
            pass
        finally:
            sockets.discard(ws)

    return app
