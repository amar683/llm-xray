import time

import pytest
from fastapi.testclient import TestClient

from xray.api.app import create_app

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    app = create_app(home=tmp_path_factory.mktemp("home"), token=TOKEN)
    with TestClient(app) as c:
        yield c


def wait_for(client, run_id, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}", headers=AUTH).json()
        if body["run"]["status"] not in {"queued", "running"}:
            return body
        time.sleep(0.2)
    raise TimeoutError(run_id)


@pytest.fixture(scope="module")
def profile_run(client, tiny_model_dir):
    params = {
        "model": str(tiny_model_dir), "sets": ["coding", "math"], "limit": 3,
        "new_tokens": 2, "ablation_prompts": 2, "device": "cpu",
    }  # fmt: skip
    r = client.post("/jobs", json={"type": "profile", "params": params}, headers=AUTH)
    assert r.status_code == 200
    return wait_for(client, r.json()["run_id"])


def test_health_is_open_but_everything_else_needs_the_token(client):
    assert client.get("/health").json()["ok"] is True
    assert client.get("/system").status_code == 401
    assert client.get("/system", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/system", headers=AUTH).json()["backends"]["cpu"] is True


def test_prompt_sets(client):
    sets = client.get("/prompt-sets", headers=AUTH).json()
    assert [s["name"] for s in sets] == ["chat", "coding", "instruction", "math"]
    assert all(s["prompts"] == 50 and s["example"] for s in sets)


def test_inspect_and_fit(client, tiny_model_dir):
    body = {"model": str(tiny_model_dir)}
    anatomy = client.post("/models/inspect", json=body, headers=AUTH).json()
    assert anatomy["architecture"]["n_layers"] == 3
    fit = client.post("/models/fit", json=body, headers=AUTH).json()
    assert fit["recommended"]["analysis"] == "float16 / bfloat16"
    missing = client.post("/models/inspect", json={"model": "/no/such/model"}, headers=AUTH)
    assert missing.status_code == 404


def test_models_list_shape(client):
    for m in client.get("/models", headers=AUTH).json():
        assert {
            "id",
            "path",
            "model_type",
            "supported",
            "size_bytes",
            "problems",
            "warnings",
        } <= set(m)
        assert m["supported"] == (m["problems"] == [])


def test_job_validation(client):
    bad = client.post("/jobs", json={"type": "nope", "params": {}}, headers=AUTH)
    assert bad.status_code == 400
    hidden = client.post("/jobs", json={"type": "_selftest", "params": {}}, headers=AUTH)
    assert hidden.status_code == 400
    assert client.delete("/jobs/not-a-run", headers=AUTH).status_code == 404
    assert client.get("/runs/not-a-run", headers=AUTH).status_code == 404


def test_profile_job_result_and_listing(client, profile_run):
    assert profile_run["run"]["status"] == "done"
    assert set(profile_run["result"]["sets"]) == {"coding", "math"}
    assert profile_run["run"]["manifest"]["backend"]["device"] == "cpu"
    listed = client.get("/runs", headers=AUTH).json()
    row = next(r for r in listed if r["id"] == profile_run["run"]["id"])
    assert row["type"] == "profile" and row["params"]["limit"] == 3 and row["model_path"]


def test_heatmap(client, profile_run):
    url = f"/runs/{profile_run['run']['id']}/heatmap"
    h = client.get(url, params={"set": "coding", "bins": 16}, headers=AUTH).json()
    assert (h["layers"], h["bins"], h["neurons"]) == (3, 16, 64)
    assert len(h["values"]) == 3 and all(len(row) == 16 for row in h["values"])
    for row in h["values"]:
        assert row == sorted(row, reverse=True)  # neurons sorted from most to least active
        assert all(0 <= v <= 1 for v in row)
    s = client.get(url, params={"set": "coding", "mode": "specific", "bins": 16}, headers=AUTH)
    assert s.json()["min"] <= 0 <= s.json()["max"]
    assert client.get(url, params={"set": "nope"}, headers=AUTH).status_code == 404


def test_neuron_queries(client, profile_run):
    base = f"/runs/{profile_run['run']['id']}"
    top = client.get(f"{base}/neurons", params={"layer": 1, "set": "math", "top": 5}, headers=AUTH)
    assert len(top.json()) == 5
    n = top.json()[0]["neuron"]
    detail = client.get(f"{base}/neuron", params={"layer": 1, "neuron": n}, headers=AUTH).json()
    assert set(detail) == {"coding", "math"}
    out_of_range = client.get(f"{base}/neuron", params={"layer": 9, "neuron": 0}, headers=AUTH)
    assert out_of_range.status_code == 404


def test_events_and_cancelling_a_queued_job(client, tiny_model_dir):
    params = {
        "model": str(tiny_model_dir), "contexts": [8], "device": "cpu",
        "prefill_repeats": 2, "decode_repeats": 2,
    }  # fmt: skip
    with client.websocket_connect(f"/events?token={TOKEN}") as ws:
        first = client.post("/jobs", json={"type": "timing", "params": params}, headers=AUTH)
        second = client.post("/jobs", json={"type": "timing", "params": params}, headers=AUTH)
        first_id, second_id = first.json()["run_id"], second.json()["run_id"]
        assert client.delete(f"/jobs/{second_id}", headers=AUTH).status_code == 200

        seen = []
        while not any(e["run_id"] == second_id and e["event"] == "cancelled" for e in seen):
            seen.append(ws.receive_json())

    first_events = [e["event"] for e in seen if e["run_id"] == first_id]
    assert first_events[:2] == ["queued", "started"] and "progress" in first_events
    assert first_events[-1] == "done"
    assert wait_for(client, first_id)["run"]["status"] == "done"
    assert wait_for(client, second_id)["run"]["status"] == "cancelled"
    assert client.get("/jobs", headers=AUTH).json() == []


def test_websocket_rejects_a_wrong_token(client):
    with pytest.raises(Exception):  # noqa: B017 - starlette raises on a refused handshake
        with client.websocket_connect("/events?token=wrong"):
            pass


def test_cost_value_combines_latest_runs(client, tiny_model_dir, profile_run):
    params = {
        "model": str(tiny_model_dir), "device": "cpu", "calibration_prompts": 4,
        "heldout_prompts": 4, "target_bits": [4],
    }  # fmt: skip
    r = client.post("/jobs", json={"type": "quant", "params": params}, headers=AUTH)
    quant = wait_for(client, r.json()["run_id"])
    assert quant["run"]["status"] == "done", quant["run"]["error"]
    assert len(quant["result"]["tensors"]) == 3 * 7 + 2

    cv = client.post("/models/cost-value", json={"model": str(tiny_model_dir)}, headers=AUTH).json()
    assert [row["layer"] for row in cv["layers"]] == [0, 1, 2]
    assert cv["sources"]["quant"]["id"] == quant["run"]["id"]
    assert cv["sources"]["profile"]["id"] == profile_run["run"]["id"]
    for row in cv["layers"]:
        assert row["skip_kl"] > 0 and row["quant_kl"] >= 0 and row["decode_ms"] > 0
    assert cv["missing"] == []
