import pytest
import torch

from xray.analyzers.common import greedy_generate, norm_scale, residual_states
from xray.analyzers.interp import run_attribution, run_lens
from xray.analyzers.loops import Ablation, detect_loop, run_loops
from xray.core.jobs import run_job


def _hook_count(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@pytest.fixture(scope="module")
def ids():
    return torch.randint(0, 128, (1, 10), generator=torch.Generator().manual_seed(3))


# -- shared helpers -------------------------------------------------------------


def test_residual_states_end_in_the_final_logits(tiny, ids):
    states, logits = residual_states(tiny, ids)
    assert len(states) == 3
    refs = tiny.refs
    with torch.inference_mode():
        rebuilt = refs.lm_head(refs.final_norm(states[-1]))
    assert torch.allclose(rebuilt, logits, atol=1e-5)
    assert _hook_count(tiny.model) == 0


def test_norm_scale_reproduces_the_norm(tiny, ids):
    states, _ = residual_states(tiny, ids)
    h = states[-1][-1]
    with torch.inference_mode():
        assert torch.allclose(
            h * norm_scale(tiny.refs.final_norm, h), tiny.refs.final_norm(h), atol=1e-5
        )


# -- logit lens -----------------------------------------------------------------


def test_lens_last_layer_is_the_model_output(tiny, ids):
    r = run_lens(tiny, ids, new_tokens=4)
    assert len(r["prob"]) == 3 and len(r["prob"][0]) == len(r["positions"]) == 4
    # At the last layer the lens *is* the model: its top token is the prediction.
    assert r["top"][-1] == r["predicted_tokens"]
    assert all(0 <= a <= 2 for a in r["answer_layer"])
    full = torch.cat([ids, torch.tensor([[int(t) for t in r["predicted_tokens"]]])], dim=1)
    _, logits = residual_states(tiny, full)
    expected = torch.softmax(logits[r["positions"]].float(), -1).max(-1).values
    assert torch.allclose(torch.tensor(r["prob"][-1]), expected, atol=1e-5)


# -- attribution ----------------------------------------------------------------


@pytest.fixture(scope="module")
def attribution(tiny, ids):
    return run_attribution(tiny, ids, top_heads=4, top_neurons=5)


def test_direct_contributions_add_up_to_the_logit_gap(attribution):
    assert attribution["direct_total"] == pytest.approx(attribution["logit_diff"], abs=1e-3)
    assert attribution["logit_diff"] > 0  # the target is the top prediction


def test_head_contributions_add_up_to_the_attention_block(attribution):
    # The attention output is the sum of its heads (this model has no output bias).
    for row, heads in zip(attribution["components"], attribution["head_direct"], strict=True):
        assert sum(heads) == pytest.approx(row["attn_direct"], abs=1e-3)


def test_attribution_shapes_and_cleanup(attribution, tiny):
    assert len(attribution["components"]) == 3 and len(attribution["head_direct"][0]) == 4
    assert len(attribution["heads"]) == 4 and len(attribution["neurons"]) == 5
    assert all({"attn_ablate", "mlp_ablate"} <= set(c) for c in attribution["components"])
    assert any(abs(c["mlp_ablate"]) > 0 for c in attribution["components"])
    assert _hook_count(tiny.model) == 0
    assert all(p.grad is None for p in tiny.model.parameters())


# -- loops ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([1, 2, 3, 4, 5, 6] + [7, 8, 9] * 6, {"period": 3, "onset": 6}),
        ([5, 6] + [9] * 20, {"period": 1, "onset": 2}),
        (list(range(40)), None),
        ([1, 2, 3] * 2 + list(range(10, 30)), None),  # repeated early, then moved on
        ([7, 8, 9] * 3, None),  # too short to call a loop
    ],
)
def test_detect_loop(tokens, expected):
    found = detect_loop(tokens)
    if expected is None:
        assert found is None
    else:
        assert (found["period"], found["onset"]) == (expected["period"], expected["onset"])


def test_ablation_hooks(tiny, ids):
    with torch.inference_mode():
        base = tiny.model(input_ids=ids).logits
        ablation = Ablation(tiny)
        try:
            assert torch.equal(tiny.model(input_ids=ids).logits, base)  # inactive by default
            ablation.heads = {(1, 2)}
            assert not torch.allclose(tiny.model(input_ids=ids).logits, base)
            ablation.clear()
            d = torch.zeros(32)
            d[0] = 1.0
            ablation.direction, ablation.from_layer = d, 2
            states, _ = residual_states(tiny, ids)
            assert states[-1][:, 0].abs().max() < 1e-6  # removed at the last block
            assert states[0][:, 0].abs().max() > 1e-6  # untouched before `from_layer`
        finally:
            ablation.remove()
    assert _hook_count(tiny.model) == 0


def test_run_loops_on_a_tiny_model(tiny_model_dir):
    from xray.models.adapters import load_model

    lm = load_model(str(tiny_model_dir), device="cpu", with_tokenizer=False, eager_attention=True)
    with torch.inference_mode():
        # transformers installs its own hooks the first time attentions are requested.
        lm.model(input_ids=torch.zeros((1, 2), dtype=torch.long), output_attentions=True)
    hooks_before = _hook_count(lm.model)
    g = torch.Generator().manual_seed(5)
    encoded = [torch.randint(0, 128, (1, 6), generator=g) for _ in range(6)]
    prompts = [("a", f"p{i}") for i in range(6)]
    r = run_loops(lm, prompts, encoded, new_tokens=40, test_prompts=3)
    assert r["summary"]["prompts"] == 6
    assert r["summary"]["looped"] == sum(bool(c["loop"]) for c in r["cases"])
    assert "tokens" not in r["cases"][0]
    if r["analysis"]:  # a random tiny model usually falls into a loop
        a = r["analysis"]
        assert len(a["recurrence"]) == 3 and len(a["copy_attention"]) == 3
        assert all(-1.001 <= v <= 1.001 for row in a["recurrence"] for v in row)
        assert a["direction"] is None or "vector" not in a["direction"]
        assert r["causal"]["tests"][0]["intervention"] == "copy heads off"
    assert _hook_count(lm.model) == hooks_before


def test_greedy_generate_refuses_invalid_numbers(tiny, ids, monkeypatch):
    original = tiny.model.forward

    def broken(*args, **kwargs):
        out = original(*args, **kwargs)
        out.logits = out.logits * float("nan")
        return out

    monkeypatch.setattr(tiny.model, "forward", broken)
    with pytest.raises(FloatingPointError, match="invalid numbers"):
        greedy_generate(tiny, ids, 3)


# -- jobs -----------------------------------------------------------------------


def test_lens_and_attribution_jobs(tiny_model_dir, tmp_path):
    base = {"model": str(tiny_model_dir), "prompt": "hello there", "device": "cpu"}
    lens = run_job("lens", {**base, "new_tokens": 3}, home=tmp_path)
    assert lens["result"]["prompt"] == "hello there" and len(lens["result"]["prob"]) == 3
    attribution = run_job("attribution", base, home=tmp_path)
    assert attribution["result"]["direct_total"] == pytest.approx(
        attribution["result"]["logit_diff"], abs=1e-3
    )
