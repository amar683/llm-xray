import json
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from xray.api.app import create_app
from xray.core import paths
from xray.core.jobs import run_job
from xray.core.storage import Storage
from xray.core.summary import compare_runs, summarize
from xray.eval import load_task
from xray.eval.checkers import check, check_json, check_python, function_body, isolation
from xray.patch.commands import PatchError
from xray.patch.history import create_patch, list_patches

AUTH = {"Authorization": "Bearer t"}


# -- checkers -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "ok"),
    [
        (' {"name": "x", "category": "y", "price": 3}\nItem: next', True),
        ('{"name": "x", "category": "y", "price": 3, "extra": 1}', True),
        (' {"name": "x", "price": 3}', False),  # a key is missing
        (' {"name": "x", "category": "y", "price": }', False),  # not valid JSON
        (' ["name", "category", "price"]', False),  # not an object
        ("name: x, category: y, price: 3", False),
    ],
)
def test_check_json(output, ok):
    assert check_json(output, ["name", "category", "price"]) is ok


def test_function_body_stops_where_the_model_moves_on():
    out = "    if n > 0:\n        return True\n\n    return False\n\ndef next_one(x):\n    pass"
    assert function_body(out) == "    if n > 0:\n        return True\n\n    return False"


@pytest.mark.skipif(sys.platform == "win32", reason="process limits are POSIX only")
def test_check_python_runs_tests_and_contains_bad_code():
    item = load_task("code", "tune")[0]  # is_even
    assert item["entry"] == "is_even"
    assert check_python("    return n % 2 == 0\n\ndef other():\n    pass", item)
    assert check_python("    return is_empty([]) and n % 2 == 0\n", item)  # may use the examples
    assert not check_python("    return n % 2 == 1\n", item)  # wrong answer
    assert not check_python("    return n %% 2\n", item)  # syntax error
    assert not check_python("", item)
    assert not check_python("    import os\n    return True\n", item)  # rejected unrun
    assert not check_python("    return open('/etc/passwd').read()\n", item)
    assert not check_python("    while True:\n        pass\n", item, timeout=3)  # never ends
    assert "limits" in isolation()


def test_check_dispatches_on_the_item_kind():
    assert check(" Paris", {"answer": "Paris"})
    assert check(' {"a": 1}', {"check": "json", "keys": ["a"]})
    assert not check(" London", {"answer": "Paris"})


# -- patch history --------------------------------------------------------------


def test_patches_stack_on_their_parents(tiny_model_dir, tmp_path):
    model = str(tiny_model_dir)
    first = create_patch(model, "no layer 1", ["drop_layers layers=1"], home=tmp_path)
    second = create_patch(model, "lighter", ["prune_neurons keep=0.5"], first["id"], home=tmp_path)
    third = create_patch(model, "", ["zero_head layer=0 head=1"], second["id"], home=tmp_path)

    assert first["effective_commands"] == ["drop_layers layers=1"]
    assert third["effective_commands"] == [
        "drop_layers layers=1", "prune_neurons keep=0.5", "zero_head layer=0 head=1"
    ]  # fmt: skip
    assert [p["name"] for p in third["chain"]] == ["no layer 1", "lighter", third["id"]]
    assert third["commands"] == ["zero_head layer=0 head=1"]  # only its own are stored
    assert [p["id"] for p in list_patches(model, home=tmp_path)] == [
        first["id"],
        second["id"],
        third["id"],
    ]

    with Storage(paths.db_path(tmp_path)) as db:
        with pytest.raises(ValueError, match="built on it"):
            db.delete_patch(first["id"])
        db.delete_patch(third["id"])
        assert db.get_patch(third["id"]) is None
        assert db.effective_commands(second["id"]) == third["effective_commands"][:2]


def test_bad_patches_are_not_recorded(tiny_model_dir, tmp_path):
    model = str(tiny_model_dir)
    with pytest.raises(PatchError):
        create_patch(model, "typo", ["explode layer=1"], home=tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        create_patch(model, "empty", [], home=tmp_path)
    with pytest.raises(KeyError):
        create_patch(model, "orphan", ["drop_layers layers=1"], "p000000", home=tmp_path)
    assert list_patches(model, home=tmp_path) == []


# -- comparison -----------------------------------------------------------------


def _edit_result(params: float, capitals: float, kl: float) -> dict:
    def stage(p, acc, k):
        return {
            "size": {"params": p, "layers": 3, "mlp_width": 64},
            "tasks": {"capitals": {"accuracy": acc}},
            "drift": {"kl": k, "top1_agreement": 1 - k},
            "health": {"prompts": 8, "looped": 2, "ended": 4},
        }

    return {
        "tasks": ["capitals"],
        "stages": {"base": stage(1e9, 0.9, 0.0), "edited": stage(params, capitals, kl)},
    }


def test_compare_runs_marks_better_and_worse():
    a, b = _edit_result(9e8, 0.8, 0.2), _edit_result(8e8, 0.7, 0.1)
    rows = {r["label"]: r for r in compare_runs("edit", a, "edit", b)}
    assert rows["Parameters removed"]["verdict"] == "better"  # more removed
    assert rows["Capitals accuracy"]["verdict"] == "worse" and rows["Capitals accuracy"][
        "change"
    ] == pytest.approx(-0.1)
    assert rows["Drift from base (KL)"]["verdict"] == "better"
    assert rows["Layers"]["verdict"] == "same"
    assert all(r["verdict"] == "same" for r in compare_runs("edit", a, "edit", a))
    with pytest.raises(ValueError, match="Cannot compare"):
        compare_runs("edit", a, "timing", {})
    assert {m["label"] for m in summarize("edit", a)} == set(rows)


# -- end to end: jobs and API ---------------------------------------------------


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    return tmp_path_factory.mktemp("history-home")


@pytest.fixture(scope="module")
def edit_runs(tiny_model_dir, home):
    """Two edit runs on the tiny model: a patch, and a second patch stacked on it."""
    model = str(tiny_model_dir)
    first = create_patch(model, "one", ["drop_layers layers=1"], home=home)
    second = create_patch(model, "two", ["prune_neurons keep=0.5"], first["id"], home=home)
    runs = []
    for patch in (first, second):
        params = {
            "model": model, "commands": patch["effective_commands"], "patch_id": patch["id"],
            "device": "cpu", "run_code": patch is second,
        }  # fmt: skip
        runs.append(run_job("edit", params, home=home))
    return runs


def test_stacked_patch_run_and_code_task(edit_runs):
    first, second = edit_runs
    assert first["result"]["patch_id"] != second["result"]["patch_id"]
    assert len(second["result"]["applied"]) == 2  # parent's command, then its own
    assert "code" not in first["result"]["tasks"] and first["result"]["code_isolation"] is None
    assert "code" in second["result"]["tasks"] and "limits" in second["result"]["code_isolation"]
    assert second["result"]["stages"]["edited"]["tasks"]["code"]["n"] == 10


def test_api_patches_and_compare(edit_runs, tiny_model_dir, home):
    first, second = edit_runs
    with TestClient(create_app(home=home, token="t")) as client:
        listed = client.get("/patches", params={"model": str(tiny_model_dir)}, headers=AUTH).json()
        assert [p["name"] for p in listed] == ["one", "two"]
        assert listed[1]["chain"][0]["name"] == "one"

        body = {
            "model": str(tiny_model_dir),
            "name": "three",
            "commands": ["zero_head layer=0 head=0"],
            "parent_id": listed[1]["id"],
        }
        made = client.post("/patches", json=body, headers=AUTH).json()
        assert len(made["effective_commands"]) == 3
        bad = client.post("/patches", json={**body, "commands": ["nope x=1"]}, headers=AUTH)
        assert bad.status_code == 400
        assert (
            client.delete(f"/patches/{listed[0]['id']}", headers=AUTH).status_code == 400
        )  # has children
        assert client.delete(f"/patches/{made['id']}", headers=AUTH).status_code == 200
        assert client.delete("/patches/p000000", headers=AUTH).status_code == 404

        cmp = client.get(
            "/runs/compare", params={"a": first["run_id"], "b": second["run_id"]}, headers=AUTH
        )
        rows = {r["label"]: r for r in cmp.json()["rows"]}
        assert cmp.json()["type"] == "edit"
        assert rows["MLP width"]["a"] == 64 and rows["MLP width"]["b"] == 32
        assert rows["Parameters removed"]["verdict"] == "better"
        assert rows["Code accuracy"]["a"] is None  # only the second run scored the code task
        missing = client.get(
            "/runs/compare", params={"a": first["run_id"], "b": "nope"}, headers=AUTH
        )
        assert missing.status_code == 404


def test_loops_job_saves_its_direction(tiny_model_dir, tmp_path):
    out = run_job(
        "loops",
        {
            "model": str(tiny_model_dir),
            "sets": ["coding"],
            "limit": 4,
            "new_tokens": 40,
            "device": "cpu",
        },
        home=tmp_path,
    )
    direction = (out["result"]["analysis"] or {}).get("direction")
    with Storage(paths.db_path(tmp_path)) as db:
        artifacts = db.artifacts(out["run_id"])
    if direction:  # needs a loop with a normal stretch before it
        vector = np.load(direction["file"])
        assert vector.shape == (32,) and artifacts["loop_direction"] == direction["file"]
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-3)
        json.dumps(out["result"])  # the vector itself is not left in the result
    else:
        assert "loop_direction" not in artifacts
