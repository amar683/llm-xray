"""A cheap, stable identity for a model folder (no torch needed)."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors",)


def model_fingerprint(model_dir: Path) -> str:
    """Cheap, stable identity for a model folder.

    Hashes config.json plus each weight file's name, size and safetensors
    header (tensor names, shapes, offsets). Avoids reading gigabytes of weights.
    """
    h = hashlib.sha256()
    config = model_dir / "config.json"
    if config.exists():
        h.update(config.read_bytes())
    for f in sorted(model_dir.iterdir()):
        if f.suffix not in WEIGHT_SUFFIXES:
            continue
        h.update(f.name.encode())
        h.update(str(f.stat().st_size).encode())
        with f.open("rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            h.update(fh.read(n))
    return h.hexdigest()[:16]
