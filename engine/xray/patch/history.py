"""Patch history: named, stackable lists of edit commands, kept per model.

A patch stores only its own commands and a link to the patch it builds on.
Applying a patch means applying its ancestors' commands first, then its own.
The base model is the root of every chain and is never modified, so
"reverting" is choosing an earlier patch, or none.
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from xray.core import paths
from xray.core.fingerprint import model_fingerprint
from xray.core.storage import Storage
from xray.models.inspect import resolve_model_dir
from xray.patch.commands import parse_commands


def new_patch_id() -> str:
    return f"p{secrets.token_hex(3)}"


def create_patch(
    model: str,
    name: str,
    commands: list[str],
    parent_id: str | None = None,
    home: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and record a patch. Returns it with its `effective_commands`."""
    parse_commands(commands)  # reject typos before recording anything
    if not commands:
        raise ValueError("A patch needs at least one command.")
    model_dir = resolve_model_dir(model)
    patch_id = new_patch_id()
    with Storage(paths.db_path(paths.home(home))) as db:
        db.create_patch(
            patch_id, model_fingerprint(model_dir), str(model_dir), name.strip() or patch_id,
            commands, parent_id,
        )  # fmt: skip
        return describe_patch(db, patch_id)


def describe_patch(db: Storage, patch_id: str) -> dict[str, Any]:
    chain = db.patch_chain(patch_id)
    patch = chain[-1]
    patch["effective_commands"] = [c for p in chain for c in p["commands"]]
    patch["chain"] = [{"id": p["id"], "name": p["name"]} for p in chain]
    return patch


def list_patches(model: str | None = None, home: str | Path | None = None) -> list[dict[str, Any]]:
    fingerprint = model_fingerprint(resolve_model_dir(model)) if model else None
    with Storage(paths.db_path(paths.home(home))) as db:
        return [describe_patch(db, p["id"]) for p in db.list_patches(fingerprint)]
