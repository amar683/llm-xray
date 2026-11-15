"""Where the engine keeps its database and run outputs."""

from __future__ import annotations

import os
from pathlib import Path


def home(override: str | Path | None = None) -> Path:
    """Data folder: explicit override, then $XRAY_HOME, then `.xray` in the current folder.

    The desktop app will pass its own application-data folder here.
    """
    p = Path(override or os.environ.get("XRAY_HOME") or ".xray").expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def db_path(home_dir: Path) -> Path:
    return home_dir / "xray.db"


def run_dir(home_dir: Path, run_id: str) -> Path:
    p = home_dir / "runs" / run_id
    p.mkdir(parents=True, exist_ok=True)
    return p
