"""Memory estimates: the hardware-fit advisor and the pre-load guard.

Pure Python (no torch), so the CLI can answer "will it fit?" instantly.
"""

from __future__ import annotations

from typing import Any

from xray.models.inspect import QUANT_BITS, kv_cache_bytes

GB = 2**30

# Runtime, tokenizer, framework buffers.
RUNTIME_OVERHEAD_BYTES = int(0.75 * GB)
# Analysis keeps hidden states, hook statistics and temporaries alive.
ANALYSIS_HEADROOM = 1.2
# The OS and other apps need the rest of a shared memory pool.
USABLE_FRACTION = 0.75

ANALYSIS_DTYPE_BYTES = {"float32": 4, "float16": 2, "bfloat16": 2}


class MemoryGuardError(RuntimeError):
    """Raised before loading a model that would not fit in available memory."""


def analysis_bytes(total_params: int, dtype_bytes: int) -> int:
    """Estimated memory to load and analyse a model in PyTorch."""
    return int(total_params * dtype_bytes * ANALYSIS_HEADROOM) + RUNTIME_OVERHEAD_BYTES


def memory_budget(system: dict[str, Any], device_type: str) -> dict[str, float]:
    """Bytes available now, and usable at most, on the device that holds the model."""
    if device_type == "cuda":
        total = system.get("vram_total_gb", 0.0) * GB
        return {"available": system.get("vram_free_gb", total / GB) * GB, "usable": total}
    return {
        "available": system["ram_available_gb"] * GB,
        "usable": system["ram_total_gb"] * GB * USABLE_FRACTION,
    }


def guard_memory(
    total_params: int,
    dtype_bytes: int,
    system: dict[str, Any],
    device_type: str,
    force: bool = False,
) -> None:
    """Refuse to load a model that will not fit, instead of crashing or swapping."""
    need = analysis_bytes(total_params, dtype_bytes)
    available = memory_budget(system, device_type)["available"]
    if need > available and not force:
        raise MemoryGuardError(
            f"This model needs about {need / GB:.1f} GB to analyse, but only "
            f"{available / GB:.1f} GB is available. Close other apps, pick a smaller "
            "model, or pass --force to try anyway."
        )


def _verdict(need: float, budget: dict[str, float]) -> str:
    if need <= budget["available"]:
        return "fits"
    if need <= budget["usable"]:
        return "fits if you free memory"
    return "too large"


def fit_report(anatomy: dict[str, Any], system: dict[str, Any], context: int = 4096) -> dict:
    """Will this model fit on this machine, for analysis and for everyday running?"""
    params = anatomy["totals"]["params"]
    arch = anatomy["architecture"]
    device_type = system["default_device"]
    budget = memory_budget(system, device_type)
    kv = kv_cache_bytes(arch, context) or 0

    analysis = []
    for dtype, nbytes in (("float16 / bfloat16", 2), ("float32", 4)):
        need = analysis_bytes(params, nbytes)
        analysis.append({"option": dtype, "need_bytes": need, "verdict": _verdict(need, budget)})

    running = []
    for quant, bits in QUANT_BITS.items():
        need = int(params * bits / 8) + kv + RUNTIME_OVERHEAD_BYTES
        running.append({"option": quant, "need_bytes": need, "verdict": _verdict(need, budget)})

    def best(rows: list[dict]) -> str | None:
        for wanted in ("fits", "fits if you free memory"):
            for r in rows:
                if r["verdict"] == wanted:
                    return r["option"]
        return None

    return {
        "device": device_type,
        "context": context,
        "available_bytes": budget["available"],
        "usable_bytes": budget["usable"],
        "analysis": analysis,
        "running": running,
        "recommended": {"analysis": best(analysis), "running": best(running)},
    }
