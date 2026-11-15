"""Device selection, synchronisation and hardware facts."""

from __future__ import annotations

import platform
import time
from typing import Any

import psutil
import torch


def pick_device(preference: str = "auto") -> torch.device:
    """Return the device to run on. `auto` prefers CUDA, then MPS, then CPU."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """Half precision on GPUs, full precision on CPU."""
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def sync(device: torch.device) -> None:
    """Block until queued GPU work has finished.

    GPU backends run asynchronously, so a timer read without this measures
    how long it took to *queue* the work, not to do it.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def now(device: torch.device) -> float:
    """Synchronised high-resolution timestamp in seconds."""
    sync(device)
    return time.perf_counter()


def system_info() -> dict[str, Any]:
    """Hardware and backend facts for the run manifest and hardware-fit advice."""
    vm = psutil.virtual_memory()
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_version": platform.release(),
        "machine": platform.machine(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / 2**30, 1),
        "ram_available_gb": round(vm.available / 2**30, 1),
        "backends": {
            "cuda": torch.cuda.is_available(),
            "mps": torch.backends.mps.is_available(),
            "cpu": True,
        },
        "default_device": str(pick_device()),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["vram_total_gb"] = round(props.total_memory / 2**30, 1)
    elif torch.backends.mps.is_available():
        # Apple Silicon shares one memory pool between CPU and GPU.
        info["gpu"] = "Apple GPU (unified memory)"
        info["vram_total_gb"] = info["ram_total_gb"]
    return info
