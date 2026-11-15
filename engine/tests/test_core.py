import json
from pathlib import Path

import pytest

from xray.core import paths
from xray.core.jobs import JobError, new_run_id, run_job
from xray.core.memory import GB, MemoryGuardError, analysis_bytes, fit_report, guard_memory
from xray.core.storage import Storage
from xray.models.inspect import inspect_model

SYSTEM = {"default_device": "mps", "ram_total_gb": 16.0, "ram_available_gb": 8.0}


def _anatomy(params: int) -> dict:
    arch = {"n_layers": 28, "n_kv_heads": 2, "head_dim": 128, "max_context": 32768}
    return {"totals": {"params": params}, "architecture": arch}


# -- memory ---------------------------------------------------------------------


def test_guard_allows_small_and_refuses_large():
    guard_memory(1_000_000_000, 2, SYSTEM, "mps")  # ~3.2 GB needed, 8 available
    with pytest.raises(MemoryGuardError, match="needs about"):
        guard_memory(8_000_000_000, 2, SYSTEM, "mps")  # ~20 GB needed
    guard_memory(8_000_000_000, 2, SYSTEM, "mps", force=True)


def test_analysis_bytes_includes_headroom_and_overhead():
    assert analysis_bytes(1_000_000_000, 2) == int(2e9 * 1.2) + int(0.75 * GB)


def test_fit_report_verdicts_and_recommendation():
    # 3B params: fp16 analysis ~7.5 GB fits in 8; fp32 ~14.2 GB exceeds 12 usable.
    r = fit_report(_anatomy(3_000_000_000), SYSTEM)
    verdicts = {row["option"]: row["verdict"] for row in r["analysis"]}
    assert verdicts == {"float16 / bfloat16": "fits", "float32": "too large"}
    assert r["recommended"] == {"analysis": "float16 / bfloat16", "running": "F16"}

    # 5.5B params: F16 (~11.1 GB) only fits after freeing memory; Q8 (~6.3 GB) fits now.
    r = fit_report(_anatomy(5_500_000_000), SYSTEM)
    running = {row["option"]: row["verdict"] for row in r["running"]}
    assert running["F16"] == "fits if you free memory"
    assert running["Q8_0"] == "fits"
    assert r["recommended"]["running"] == "Q8_0"

    r = fit_report(_anatomy(70_000_000_000), SYSTEM)
    assert r["recommended"] == {"analysis": None, "running": None}


def test_fit_report_on_real_anatomy(tiny_model_dir):
    r = fit_report(inspect_model(str(tiny_model_dir)), SYSTEM, context=32)
    assert all(row["verdict"] == "fits" for row in r["analysis"] + r["running"])


# -- storage --------------------------------------------------------------------


def test_storage_round_trip(tmp_path):
    manifest = {"backend": {"name": "pytorch", "device": "cpu", "dtype": "float32"}}
    model = {"fingerprint": "abc", "path": "/m", "arch": "llama", "n_layers": 3, "params": 10}
    with Storage(tmp_path / "x.db") as db:
        db.create_run("r1", "timing", {"contexts": [8]})
        assert db.get_run("r1")["status"] == "running"

        model_id = db.upsert_model(model)
        assert db.upsert_model({**model, "path": "/moved"}) == model_id  # same fingerprint

        row = {"context": 8, "phase": "decode", "component": "block", "metric": "est_ms"}
        db.add_layer_stats(
            "r1", [{**row, "layer": 0, "value": 1.5}, {**row, "layer": 1, "value": 2}]
        )
        db.add_artifact("r1", "result", tmp_path / "result.json")
        db.finish_run("r1", "done", manifest=manifest, model_id=model_id)

        run = db.get_run("r1")
        assert (run["status"], run["backend"]) == ("done", "pytorch-cpu-float32")
        assert json.loads(run["params_json"]) == {"contexts": [8]}
        assert db.artifacts("r1") == {"result": str(tmp_path / "result.json")}
        assert [s["value"] for s in db.layer_stats("r1")] == [1.5, 2.0]
        assert db.list_runs()[0]["model_path"] == "/moved"

    with Storage(tmp_path / "x.db") as db:  # reopening keeps the data
        assert db.get_run("r1")["status"] == "done"


def test_fail_unfinished_spares_runs_whose_worker_is_alive(tmp_path):
    import os

    with Storage(tmp_path / "x.db") as db:
        for run_id in ("alive", "dead", "never-started", "finished"):
            db.create_run(run_id, "timing", {})
        db.set_pid("alive", os.getpid())  # this very process
        db.set_pid("dead", 2**22 + 12345)  # no such process
        db.finish_run("finished", "done")
        assert db.fail_unfinished("The engine stopped.") == 2
        status = {
            r: db.get_run(r)["status"] for r in ("alive", "dead", "never-started", "finished")
        }
        assert status == {
            "alive": "running",
            "dead": "failed",
            "never-started": "failed",
            "finished": "done",
        }
        assert db.get_run("dead")["error"] == "The engine stopped."


def test_old_databases_gain_the_pid_column(tmp_path):
    import sqlite3

    conn = sqlite3.connect(tmp_path / "old.db")
    conn.execute(
        "CREATE TABLE runs (id TEXT PRIMARY KEY, type TEXT NOT NULL, status TEXT NOT NULL, "
        "model_id INTEGER, backend TEXT, params_json TEXT NOT NULL, manifest_json TEXT, "
        "error TEXT, started_at TEXT NOT NULL, finished_at TEXT)"
    )
    conn.execute(
        "INSERT INTO runs VALUES ('r', 'timing', 'done', NULL, NULL, '{}', NULL, NULL, 'now', NULL)"
    )
    conn.commit()
    conn.close()
    with Storage(tmp_path / "old.db") as db:
        assert db.get_run("r")["pid"] is None
        db.set_pid("r", 7)
        assert db.get_run("r")["pid"] == 7


def test_run_ids_are_unique_and_sortable():
    ids = [new_run_id() for _ in range(50)]
    assert len(set(ids)) == 50 and all(len(i) == 20 for i in ids)


# -- jobs (real worker subprocess) ----------------------------------------------


def test_timing_job_runs_in_worker_and_is_recorded(tiny_model_dir, tmp_path):
    events = []
    out = run_job(
        "timing",
        {
            "model": str(tiny_model_dir),
            "contexts": [8],
            "device": "cpu",
            "prefill_repeats": 2,
            "decode_repeats": 2,
        },
        home=tmp_path,
        on_event=events.append,
    )
    assert out["result"]["contexts"][0]["context"] == 8
    assert out["manifest"]["backend"]["device"] == "cpu"
    assert "loading model" in [e["stage"] for e in events]

    with Storage(paths.db_path(tmp_path)) as db:
        run = db.get_run(out["run_id"])
        assert run["status"] == "done" and run["model_id"] is not None
        assert len(db.layer_stats(out["run_id"])) == 2 * (3 * 3 + 3)
        result_file = db.artifacts(out["run_id"])["result"]
    assert json.loads(Path(result_file).read_text())["result"] == out["result"]


def test_failed_job_is_recorded_with_its_error(tmp_path):
    with pytest.raises(JobError, match="not a local folder") as e:
        run_job("timing", {"model": str(tmp_path / "missing")}, home=tmp_path)
    assert e.value.kind == "error"
    with Storage(paths.db_path(tmp_path)) as db:
        run = db.get_run(e.value.run_id)
    assert run["status"] == "failed" and "not a local folder" in run["error"]


def test_worker_crash_is_reported(tmp_path):
    with pytest.raises(JobError, match="exited unexpectedly") as e:
        run_job("_selftest", {"action": "crash"}, home=tmp_path)
    assert e.value.kind == "crash"
    with Storage(paths.db_path(tmp_path)) as db:
        assert db.get_run(e.value.run_id)["status"] == "failed"
    assert (tmp_path / "runs" / e.value.run_id / "worker.log").exists()


def test_memory_guard_error_is_reported_as_memory(tmp_path):
    with pytest.raises(JobError, match="needs about") as e:
        run_job("_selftest", {"action": "memory"}, home=tmp_path)
    assert e.value.kind == "memory"


def test_progress_events_reach_the_caller(tmp_path):
    events = []
    out = run_job("_selftest", {"action": "ok"}, home=tmp_path, on_event=events.append)
    assert out["result"] == {"ok": True}
    assert events == [{"event": "progress", "stage": "working", "done": 1, "total": 2}]


def test_unknown_job_type(tmp_path):
    with pytest.raises(ValueError, match="Unknown job type"):
        run_job("nope", {}, home=tmp_path)
