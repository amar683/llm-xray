"""Real-world runtime speed: llama.cpp and MLX next to the PyTorch numbers.

X-Ray analyses models in PyTorch because PyTorch exposes everything. People
*run* models in llama.cpp or MLX, which are faster and behave differently. This
module measures those runtimes on the same model so every speed shown in
PyTorch can be read against what a user would actually get.

Both runtimes are optional. llama.cpp needs a built checkout (only its
`llama-bench`, `llama-quantize` and converter are run); MLX needs the `mlx`
extra and Apple Silicon.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xray import __version__
from xray.backends.llamacpp import LlamaCppError, Tools, convert_to_f16, find_tools, quantize
from xray.core.fingerprint import model_fingerprint
from xray.models.inspect import inspect_model

DEFAULT_TYPES = ("F16", "Q8_0", "Q4_K_M")
FILLER = (
    "Explain what a hash map is and when you would use one. "
    "What is the difference between a process and a thread? "
)


def bench_llamacpp(tools: Tools, gguf: Path, n_prompt: int, n_gen: int, repeats: int) -> dict:
    """Prefill and decode speed of one GGUF file, from `llama-bench`."""
    if tools.bench is None:
        raise LlamaCppError("This llama.cpp build has no llama-bench.")
    proc = subprocess.run(
        [str(tools.bench), "-m", str(gguf), "-p", str(n_prompt), "-n", str(n_gen),
         "-r", str(repeats), "-o", "json"],
        capture_output=True, text=True,
    )  # fmt: skip
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-6:])
        raise LlamaCppError(f"llama-bench failed (exit {proc.returncode}):\n{tail}")
    rows = json.loads(proc.stdout[proc.stdout.index("[") :])
    prefill = next(r for r in rows if r["n_prompt"] > 0 and r["n_gen"] == 0)
    decode = next(r for r in rows if r["n_gen"] > 0 and r["n_prompt"] == 0)
    return {
        "file_bytes": gguf.stat().st_size,
        "prefill_tps": prefill["avg_ts"],
        "prefill_tps_std": prefill["stddev_ts"],
        "decode_tps": decode["avg_ts"],
        "decode_tps_std": decode["stddev_ts"],
        "backends": prefill.get("backends"),
        "build": prefill.get("build_commit"),
    }


def bench_mlx(model_dir: Path, n_prompt: int, n_gen: int) -> dict[str, Any]:
    """Prefill and decode speed in MLX, loading the model folder as it is."""
    try:
        import mlx_lm
    except ImportError as e:
        raise RuntimeError(
            "MLX is not installed: run `uv sync --extra mlx` (Apple Silicon)."
        ) from e
    model, tokenizer = mlx_lm.load(str(model_dir))
    ids = tokenizer.encode(FILLER)
    prompt = (ids * (n_prompt // len(ids) + 1))[:n_prompt]

    def run(max_tokens: int):
        final = None
        for chunk in mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
            final = chunk  # the last chunk carries the speed figures
        return final

    run(8)  # warm-up: the first pass compiles kernels
    r = run(n_gen)
    return {
        "prefill_tps": r.prompt_tps,
        "decode_tps": r.generation_tps,
        "prompt_tokens": r.prompt_tokens,
        "generated_tokens": r.generation_tokens,
        "peak_memory_gb": r.peak_memory,
        "version": getattr(mlx_lm, "__version__", None),
    }


def run_bench(
    model: str,
    out_dir: Path,
    llama_cpp: str | None = None,
    types: tuple[str, ...] = DEFAULT_TYPES,
    use_mlx: bool = True,
    n_prompt: int = 512,
    n_gen: int = 128,
    repeats: int = 3,
    progress=lambda *a, **k: None,
) -> dict[str, Any]:
    """Measure every available runtime. Skipped runtimes are listed with the reason."""
    anatomy = inspect_model(model)
    model_dir = Path(anatomy["path"])
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    try:
        tools = find_tools(llama_cpp)
        out_dir.mkdir(parents=True, exist_ok=True)
        progress("converting to GGUF")
        f16 = convert_to_f16(tools, model_dir, out_dir / "model-f16.gguf")
        files: list[tuple[str, Path]] = []
        for gguf_type in types:
            if gguf_type.upper() == "F16":
                files.append(("F16", f16))
                continue
            path = out_dir / f"bench-{gguf_type.lower()}.gguf"
            if not path.exists():
                progress(f"quantizing to {gguf_type}")
                quantize(tools, f16, path, gguf_type, pure=False)
            files.append((gguf_type, path))
        # Mixed-precision plans exported earlier by `xray export` are measured too.
        for path in sorted(out_dir.glob("mixed-at-*.gguf")):
            size = path.stem.removeprefix("mixed-at-").upper()
            files.append((f"mixed plan ({size} size)", path))
        for n, (label, path) in enumerate(files):
            progress(f"llama.cpp {label}", n, len(files))
            rows.append({"runtime": "llama.cpp", "variant": label,
                         **bench_llamacpp(tools, path, n_prompt, n_gen, repeats)})  # fmt: skip
    except LlamaCppError as e:
        skipped.append({"runtime": "llama.cpp", "reason": str(e)})

    if use_mlx and platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            progress("MLX")
            rows.append({"runtime": "MLX", "variant": "as stored (16-bit)",
                         "file_bytes": anatomy["totals"]["bytes_on_disk"],
                         **bench_mlx(model_dir, n_prompt, n_gen)})  # fmt: skip
        except Exception as e:  # noqa: BLE001 - an optional runtime must not fail the run
            skipped.append({"runtime": "MLX", "reason": f"{type(e).__name__}: {e}"})
    elif use_mlx:
        skipped.append({"runtime": "MLX", "reason": "MLX runs on Apple Silicon only."})

    if not rows:
        reasons = "; ".join(f"{s['runtime']}: {s['reason']}" for s in skipped)
        raise RuntimeError(f"No runtime could be measured. {reasons}")
    return {
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "repeats": repeats,
        "rows": rows,
        "skipped": skipped,
    }


def bench_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    """Worker entry point for the `bench` job. Does not load the model in PyTorch."""
    from xray.core.device import system_info

    anatomy = inspect_model(params["model"])
    model_dir = Path(anatomy["path"])
    fingerprint = model_fingerprint(model_dir)
    # runs/<id>/ -> the data folder two levels up holds exports/.
    out_dir = (
        Path(params["out"])
        if params.get("out")
        else ctx.run_dir.parent.parent / "exports" / fingerprint
    )
    result = run_bench(
        params["model"],
        out_dir,
        llama_cpp=params.get("llama_cpp"),
        types=tuple(params.get("types") or DEFAULT_TYPES),
        use_mlx=params.get("mlx", True),
        n_prompt=params.get("n_prompt", 512),
        n_gen=params.get("n_gen", 128),
        repeats=params.get("repeats", 3),
        progress=ctx.progress,
    )
    arch = anatomy["architecture"]
    # The llama.cpp folder is a local path: keep it out of the recorded parameters.
    recorded = {k: v for k, v in params.items() if k not in ("llama_cpp", "out")}
    return {
        "manifest": {
            "run_type": "bench",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "engine_version": __version__,
            "model": {"path": str(model_dir), "fingerprint": fingerprint},
            "backend": {"name": "runtimes", "device": "native", "dtype": "various"},
            "seed": 0,
            "params": recorded,
            "hardware": system_info(),
        },
        "model": {
            "fingerprint": fingerprint,
            "path": str(model_dir),
            "arch": arch["model_type"],
            "n_layers": arch["n_layers"],
            "params": anatomy["totals"]["params"],
        },
        "result": result,
    }
