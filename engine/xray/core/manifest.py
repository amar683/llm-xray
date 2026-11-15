"""Run manifest: everything needed to reproduce or compare a run."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from xray import __version__
from xray.core.device import system_info
from xray.core.fingerprint import model_fingerprint  # noqa: F401 - re-exported


def build_manifest(
    *,
    model_dir: Path,
    run_type: str,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_type": run_type,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "engine_version": __version__,
        "model": {"path": str(model_dir), "fingerprint": model_fingerprint(model_dir)},
        "backend": {
            "name": "pytorch",
            "device": str(device),
            "dtype": str(dtype).removeprefix("torch."),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "seed": seed,
        "params": params,
        "hardware": system_info(),
    }
