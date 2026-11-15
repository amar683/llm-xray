import re
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from xray.api.app import create_app
from xray.core.jobs import run_job
from xray.core.summary import compare_runs
from xray.report import (
    LOOP_DIRECTION_ALIAS,
    build_report,
    estimate_tokens,
    model_label,
    resolve_aliases,
)
from xray.report.advisor import SYSTEM, TOOLS, AdvisorError, build_request, run_advisor
from xray.report.query import QueryError, run_query

AUTH = {"Authorization": "Bearer t"}


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    return tmp_path_factory.mktemp("report-home")


@pytest.fixture(scope="module")
def model(tiny_model_dir, home):
    """The tiny model with one run of every kind recorded."""
    m = str(tiny_model_dir)
    common = {"model": m, "device": "cpu"}
    run_job(
        "timing", {**common, "contexts": [8], "prefill_repeats": 2, "decode_repeats": 2}, home=home
    )
    run_job(
        "profile",
        {**common, "sets": ["coding", "math"], "limit": 3, "new_tokens": 2, "ablation_prompts": 2},
        home=home,
    )
    run_job(
        "quant",
        {**common, "calibration_prompts": 4, "heldout_prompts": 4, "target_bits": [4]},
        home=home,
    )
    run_job("loops", {**common, "sets": ["coding"], "limit": 4, "new_tokens": 40}, home=home)
    run_job("attribution", {**common, "prompt": "a private prompt"}, home=home)
    run_job("edit", {**common, "commands": ["zero_head layer=1 head=2"]}, home=home)
    return m


# -- report ---------------------------------------------------------------------


def test_report_covers_every_kind_of_run_within_budget(model, home):
    r = build_report(model, 3000, home)
    assert r["fits"] and r["tokens_estimate"] <= 3000 and r["missing"] == []
    assert r["tokens_estimate"] == estimate_tokens(r["text"])
    for heading in ("# Model report", "## Key findings", "## Speed", "## Layers", "## Activations",
                    "## Quantization", "## Loops", "## Edits already tried", "## What can be done"):  # fmt: skip
        assert heading in r["text"], heading
    assert "zero_head layer=1 head=2" in r["text"]  # the edit that was tried
    assert all(s["state"] == "full" for s in r["sections"])
    assert set(r["sources"]) >= {"timing", "profile", "quant", "loops", "edit"}


def test_report_never_contains_paths_or_prompts(model, home, tiny_model_dir):
    r = build_report(model, 3000, home)
    assert str(tiny_model_dir) not in r["text"] and str(home) not in r["text"]
    assert not re.search(r"/(Users|home|private|tmp|var)/", r["text"])
    assert "a private prompt" not in r["text"]
    for path in r["aliases"].values():  # aliases are returned separately, for local use only
        assert path not in r["text"]


def test_small_budget_shortens_then_drops_low_priority_sections(model, home):
    full = build_report(model, 3000, home)
    tight = build_report(model, 700, home)
    assert tight["tokens_estimate"] < full["tokens_estimate"]
    states = {s["name"]: s["state"] for s in tight["sections"]}
    # The essentials always stay; the extras give way.
    assert states["model"] == states["findings"] == states["reference"] == "full"
    assert any(v in ("short", "dropped") for v in states.values())
    assert "## What can be done" in tight["text"]
    tiny = build_report(model, 10, home)  # an impossible budget is reported, not hidden
    assert not tiny["fits"]


def test_model_label_hides_the_folder():
    assert (
        model_label("/Users/me/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/abc")
        == "Qwen/Qwen2.5-1.5B"
    )
    assert model_label("/some/where/my-model") == "my-model"


def test_resolve_aliases():
    aliases = {LOOP_DIRECTION_ALIAS: "/data/dir.npy"}
    cmd = f"ablate_direction vector={LOOP_DIRECTION_ALIAS} from_layer=1"
    assert resolve_aliases([cmd, "zero_head layer=0 head=0"], aliases) == [
        'ablate_direction vector="/data/dir.npy" from_layer=1', "zero_head layer=0 head=0"
    ]  # fmt: skip
    with pytest.raises(ValueError, match="Unknown alias"):
        resolve_aliases(["ablate_direction vector=@other from_layer=1"], aliases)


# -- queries --------------------------------------------------------------------


def test_queries(model, home):
    layer = run_query(model, "layer 1", home)
    assert layer.startswith("Layer 1:") and "Skipped:" in layer and "Quantized to 4 bits" in layer
    assert "fires on" in run_query(model, "neuron 1 5", home)
    assert "most math-specific" in run_query(model, "top_neurons 1 math", home)
    head = run_query(model, "head 1 2", home)
    assert "direct push" in head and "a private prompt" not in head
    assert "4 bits KL" in run_query(model, "tensor 0 v_proj", home)
    assert run_query(model, "help", home).startswith("Queries:")
    assert all(
        estimate_tokens(run_query(model, q, home)) < 500 for q in ("layer 0", "tensor 2 down_proj")
    )


@pytest.mark.parametrize(
    ("text", "message"),
    [("layer 99", "between 0 and 2"), ("layer x", "must be a number"), ("nonsense", "Could not understand"),
     ("top_neurons 1 nope", "No prompt set"), ("tensor 0 bogus", "KIND must be")],
)  # fmt: skip
def test_bad_queries_explain_themselves(model, home, text, message):
    with pytest.raises(QueryError, match=message):
        run_query(model, text, home)


# -- advisor (scripted stand-in for the API: nothing leaves the machine) ---------


class FakeClaude:
    """Replays scripted replies and records every request it is given."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.replies.pop(0)


def reply(*blocks, stop_reason="tool_use"):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
    )  # fmt: skip


def tool(name, call_id, **inputs):
    return SimpleNamespace(type="tool_use", name=name, id=call_id, input=inputs)


def text(value):
    return SimpleNamespace(type="text", text=value)


def test_preview_is_exactly_what_gets_sent(model, home):
    preview = build_request(model, "Make it loop less.", 3000, home)
    payload = preview["payload"]
    assert payload["model"] == "claude-opus-5-5" and payload["system"] == SYSTEM
    assert payload["messages"][0]["content"].endswith("## Goal\nMake it loop less.")
    assert preview["report_fits"] and preview["tokens_estimate"] > preview["report_tokens"]

    fake = FakeClaude([reply(text("No edit is worth trying."), stop_reason="end_turn")])
    result = run_advisor(model, "Make it loop less.", 3000, home, client=fake)
    sent = fake.requests[0]
    assert sent["messages"] == payload["messages"] and sent["tools"] == TOOLS == payload["tools"]
    assert sent["system"][0]["text"] == SYSTEM and sent["model"] == payload["model"]
    assert sent["fallbacks"] == "default" and sent["output_config"] == {"effort": "medium"}
    assert "tool_choice" not in sent and "thinking" not in sent
    # A reply in words means "no edit": nothing to apply.
    assert result["commands"] == [] and result["message"] == "No edit is worth trying."
    with pytest.raises(AdvisorError, match="what you want"):
        build_request(model, "  ", 3000, home)


def test_advisor_round_trip_runs_locally_end_to_end(model, home):
    """Advisor asks a query, proposes a bad command, corrects it; the edit then runs locally."""
    fake = FakeClaude([
        reply(text("Let me look at layer 1."), tool("query", "q1", query="layer 1"), tool("query", "q2", query="layer 99")),
        reply(tool("propose_edit", "p1", commands=["zero_head layer=1"], rationale="r", expected_effect="e")),
        reply(tool("propose_edit", "p2", commands=["zero_head layer=1 head=3"],
                   rationale="Head 3 in layer 1 is a copy head.", expected_effect="Fewer loops.")),
    ])  # fmt: skip
    steps = []
    result = run_advisor(model, "Reduce looping.", 3000, home, client=fake, on_step=steps.append)

    assert result["commands"] == ["zero_head layer=1 head=3"] and result["rounds"] == 3
    assert result["rationale"].startswith("Head 3") and result["usage"] == {
        "input_tokens": 300,
        "output_tokens": 60,
    }
    assert [q["query"] for q in result["queries"]] == ["layer 1", "layer 99"]

    # Round 2 carried both query answers back, in one user message, locally computed.
    answers = fake.requests[1]["messages"][-1]["content"]
    assert [a["tool_use_id"] for a in answers] == ["q1", "q2"]
    assert answers[0]["content"] == run_query(model, "layer 1", home) and not answers[0]["is_error"]
    assert answers[1]["is_error"] and "between 0 and 2" in answers[1]["content"]
    # Round 3 told the advisor its first proposal was invalid instead of failing.
    correction = fake.requests[2]["messages"][-1]["content"][0]
    assert correction["is_error"] and "missing: head" in correction["content"]
    assert any("locally" in s for s in steps)
    # Nothing sent ever contained a local path.
    assert str(home) not in str(fake.requests)

    # The proposal is then applied and measured by the ordinary local edit job.
    edited = run_job(
        "edit", {"model": model, "commands": result["commands"], "device": "cpu"}, home=home
    )
    assert edited["result"]["applied"] == ["Switched off head 3 in layer 1."]
    assert edited["result"]["stages"]["edited"]["drift"]["kl"] > 0
    rows = compare_runs("edit", edited["result"], "edit", edited["result"])
    assert rows and all(r["verdict"] == "same" for r in rows)
    assert (
        "zero_head layer=1 head=3" in build_report(model, 3000, home)["text"]
    )  # now in the report


def test_advisor_alias_is_resolved_only_locally(model, home, tmp_path, monkeypatch):
    import numpy as np

    import xray.report.advisor as advisor

    np.save(tmp_path / "d.npy", np.ones(32, dtype=np.float32))
    real = advisor.build_request

    def with_alias(*args, **kwargs):
        request = real(*args, **kwargs)
        request["aliases"] = {LOOP_DIRECTION_ALIAS: str(tmp_path / "d.npy")}
        return request

    monkeypatch.setattr(advisor, "build_request", with_alias)
    command = f"ablate_direction vector={LOOP_DIRECTION_ALIAS} from_layer=1"
    fake = FakeClaude(
        [reply(tool("propose_edit", "p", commands=[command], rationale="r", expected_effect="e"))]
    )
    result = run_advisor(model, "Stop the loop.", 3000, home, client=fake)
    assert result["commands_as_proposed"] == [command]
    assert result["commands"] == [f'ablate_direction vector="{tmp_path / "d.npy"}" from_layer=1']
    assert str(tmp_path) not in str(fake.requests)


def test_advisor_failures(model, home):
    with pytest.raises(AdvisorError, match="declined"):
        run_advisor(model, "g", 3000, home, client=FakeClaude([reply(stop_reason="refusal")]))
    with pytest.raises(AdvisorError, match="cut off"):
        run_advisor(model, "g", 3000, home, client=FakeClaude([reply(stop_reason="max_tokens")]))
    endless = FakeClaude([reply(tool("query", f"q{i}", query="layer 0")) for i in range(20)])
    with pytest.raises(AdvisorError, match="within 8 rounds"):
        run_advisor(model, "g", 3000, home, client=endless)
    assert len(endless.requests) == 8

    def no_credential(**_kwargs):
        raise TypeError("Could not resolve authentication method.")

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=no_credential)))
    with pytest.raises(AdvisorError, match="nothing was sent"):
        run_advisor(model, "g", 3000, home, client=client)


# -- API ------------------------------------------------------------------------


def test_api_report_query_and_advisor(model, home, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    app = create_app(home=home, token="t")
    with TestClient(app) as client:
        r = client.post("/report", json={"model": model, "budget": 3000}, headers=AUTH).json()
        assert r["fits"] and "aliases" not in r and "## Layers" in r["text"]
        q = client.post("/query", json={"model": model, "query": "layer 0"}, headers=AUTH).json()
        assert q["answer"].startswith("Layer 0:")
        assert (
            client.post("/query", json={"model": model, "query": "zzz"}, headers=AUTH).status_code
            == 400
        )

        status = client.get("/advisor/status", headers=AUTH).json()
        assert status == {
            "sdk_installed": status["sdk_installed"],
            "credential_in_environment": False,
        }
        body = {"model": model, "goal": "Reduce looping."}
        preview = client.post("/advisor/preview", json=body, headers=AUTH).json()
        assert "aliases" not in preview and preview["payload"]["messages"][0]["content"].endswith(
            "Reduce looping."
        )
        assert (
            client.post("/advisor/preview", json={**body, "goal": ""}, headers=AUTH).status_code
            == 400
        )

        app.state.advisor_client = FakeClaude(
            [
                reply(
                    tool(
                        "propose_edit",
                        "p",
                        commands=["zero_head layer=0 head=1"],
                        rationale="r",
                        expected_effect="e",
                    )
                )
            ]
        )
        sent = client.post("/advisor/send", json=body, headers=AUTH).json()
        assert sent["commands"] == ["zero_head layer=0 head=1"] and sent["rounds"] == 1
        assert client.post("/report", json={"model": "/no/such"}, headers=AUTH).status_code == 404
