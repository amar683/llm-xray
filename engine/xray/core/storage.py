"""SQLite storage for models, runs and per-layer statistics."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id          INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL UNIQUE,
    path        TEXT NOT NULL,
    arch        TEXT,
    n_layers    INTEGER,
    params      INTEGER,
    added_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    status        TEXT NOT NULL,          -- queued | running | done | failed | cancelled
    model_id      INTEGER REFERENCES models(id),
    backend       TEXT,
    params_json   TEXT NOT NULL,
    manifest_json TEXT,
    error         TEXT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT
);
CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    kind   TEXT NOT NULL,
    path   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS layer_stats (
    run_id    TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    context   INTEGER,
    phase     TEXT,
    layer     INTEGER,
    component TEXT NOT NULL,
    metric    TEXT NOT NULL,
    value     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS layer_stats_run ON layer_stats(run_id);
CREATE TABLE IF NOT EXISTS patches (
    id                TEXT PRIMARY KEY,
    model_fingerprint TEXT NOT NULL,
    model_path        TEXT NOT NULL,
    parent_id         TEXT REFERENCES patches(id),
    name              TEXT NOT NULL,
    commands_json     TEXT NOT NULL,     -- this patch's own commands, not its parents'
    created_at        TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        # Added in schema 3: the worker's process id, so a restarted server can tell
        # a run that is still alive in another process from one that died.
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(runs)")}
        if "pid" not in columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN pid INTEGER")
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- models ---------------------------------------------------------------

    def upsert_model(self, model: dict[str, Any]) -> int:
        self.conn.execute(
            """INSERT INTO models (fingerprint, path, arch, n_layers, params, added_at)
               VALUES (:fingerprint, :path, :arch, :n_layers, :params, :added_at)
               ON CONFLICT(fingerprint) DO UPDATE SET path = excluded.path""",
            {**model, "added_at": _now()},
        )
        row = self.conn.execute(
            "SELECT id FROM models WHERE fingerprint = ?", (model["fingerprint"],)
        ).fetchone()
        self.conn.commit()
        return row["id"]

    # -- runs -----------------------------------------------------------------

    def create_run(
        self, run_id: str, run_type: str, params: dict[str, Any], status: str = "running"
    ) -> None:
        self.conn.execute(
            "INSERT INTO runs (id, type, status, params_json, started_at) VALUES (?, ?, ?, ?, ?)",
            (run_id, run_type, status, json.dumps(params), _now()),
        )
        self.conn.commit()

    def set_status(self, run_id: str, status: str) -> None:
        self.conn.execute("UPDATE runs SET status = ? WHERE id = ?", (status, run_id))
        self.conn.commit()

    def set_pid(self, run_id: str, pid: int) -> None:
        self.conn.execute("UPDATE runs SET pid = ? WHERE id = ?", (pid, run_id))
        self.conn.commit()

    def fail_unfinished(self, reason: str) -> int:
        """Mark runs whose process is gone as failed.

        A run still marked queued or running is dead unless its worker process
        is alive. That leaves alone a run another process (a CLI command, say)
        is working on right now.
        """
        import psutil

        rows = self.conn.execute(
            "SELECT id, pid FROM runs WHERE status IN ('queued', 'running')"
        ).fetchall()
        dead = [r["id"] for r in rows if r["pid"] is None or not psutil.pid_exists(r["pid"])]
        self.conn.executemany(
            "UPDATE runs SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
            [(reason, _now(), run_id) for run_id in dead],
        )
        self.conn.commit()
        return len(dead)

    def finish_run(
        self,
        run_id: str,
        status: str,
        *,
        manifest: dict[str, Any] | None = None,
        model_id: int | None = None,
        error: str | None = None,
    ) -> None:
        backend = None
        if manifest:
            b = manifest["backend"]
            backend = f"{b['name']}-{b['device']}-{b['dtype']}"
        self.conn.execute(
            """UPDATE runs SET status = ?, manifest_json = ?, model_id = ?, backend = ?,
                               error = ?, finished_at = ? WHERE id = ?""",
            (
                status,
                json.dumps(manifest) if manifest else None,
                model_id,
                backend,
                error,
                _now(),
                run_id,
            ),
        )
        self.conn.commit()

    def add_artifact(self, run_id: str, kind: str, path: Path) -> None:
        self.conn.execute(
            "INSERT INTO run_artifacts (run_id, kind, path) VALUES (?, ?, ?)",
            (run_id, kind, str(path)),
        )
        self.conn.commit()

    def add_layer_stats(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        self.conn.executemany(
            """INSERT INTO layer_stats (run_id, context, phase, layer, component, metric, value)
               VALUES (:run_id, :context, :phase, :layer, :component, :metric, :value)""",
            [{"run_id": run_id, **r} for r in rows],
        )
        self.conn.commit()

    # -- patches ---------------------------------------------------------------

    def create_patch(
        self,
        patch_id: str,
        model_fingerprint: str,
        model_path: str,
        name: str,
        commands: list[str],
        parent_id: str | None = None,
    ) -> None:
        if parent_id is not None:
            parent = self.get_patch(parent_id)
            if parent is None:
                raise KeyError(f"No patch '{parent_id}' to build on.")
            if parent["model_fingerprint"] != model_fingerprint:
                raise ValueError(f"Patch '{parent_id}' belongs to a different model.")
        self.conn.execute(
            """INSERT INTO patches (id, model_fingerprint, model_path, parent_id, name,
                                    commands_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                patch_id,
                model_fingerprint,
                model_path,
                parent_id,
                name,
                json.dumps(commands),
                _now(),
            ),
        )
        self.conn.commit()

    def get_patch(self, patch_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM patches WHERE id = ?", (patch_id,)).fetchone()
        if row is None:
            return None
        patch = dict(row)
        patch["commands"] = json.loads(patch.pop("commands_json"))
        return patch

    def patch_chain(self, patch_id: str) -> list[dict[str, Any]]:
        """The patch and its ancestors, oldest first."""
        chain = []
        current: str | None = patch_id
        while current is not None:
            patch = self.get_patch(current)
            if patch is None:
                raise KeyError(f"No patch '{current}'.")
            chain.append(patch)
            current = patch["parent_id"]
        return chain[::-1]

    def effective_commands(self, patch_id: str) -> list[str]:
        """Everything to apply for this patch: its ancestors' commands, then its own."""
        return [c for patch in self.patch_chain(patch_id) for c in patch["commands"]]

    def list_patches(self, model_fingerprint: str | None = None) -> list[dict[str, Any]]:
        where = "WHERE model_fingerprint = ?" if model_fingerprint else ""
        rows = self.conn.execute(
            f"SELECT id FROM patches {where} ORDER BY rowid",  # creation order
            (model_fingerprint,) if model_fingerprint else (),
        ).fetchall()
        return [self.get_patch(r["id"]) for r in rows]

    def delete_patch(self, patch_id: str) -> None:
        """Remove a patch. Refused while other patches are built on it."""
        children = self.conn.execute(
            "SELECT id FROM patches WHERE parent_id = ?", (patch_id,)
        ).fetchall()
        if children:
            names = ", ".join(r["id"] for r in children)
            raise ValueError(f"Patch '{patch_id}' has patches built on it ({names}).")
        self.conn.execute("DELETE FROM patches WHERE id = ?", (patch_id,))
        self.conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT r.id, r.type, r.status, r.backend, r.started_at, r.finished_at, r.error,
                      r.params_json, m.path AS model_path, m.fingerprint AS model_fingerprint
               FROM runs r LEFT JOIN models m ON m.id = r.model_id
               ORDER BY r.started_at DESC, r.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def artifacts(self, run_id: str) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT kind, path FROM run_artifacts WHERE run_id = ?", (run_id,)
        ).fetchall()
        return {r["kind"]: r["path"] for r in rows}

    def layer_stats(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT context, phase, layer, component, metric, value FROM layer_stats"
            " WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]
