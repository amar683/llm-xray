import numpy as np
import pytest
import torch

from xray.analyzers.profile import (
    compare_sets,
    drop_layer_ablation,
    encode_prompts,
    firing_rate,
    layer_stats,
    load_arrays,
    neuron_detail,
    pool,
    run_profile,
    save_arrays,
    summarise_layers,
    top_neurons,
)
from xray.core import paths
from xray.core.jobs import run_job
from xray.core.storage import Storage
from xray.data import builtin_sets, load_prompt_set
from xray.probes.hooks import EDGES


def _hook_count(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@pytest.fixture(scope="module")
def sets(tiny):
    g = torch.Generator().manual_seed(1)
    return {
        "a": [torch.randint(0, 64, (1, n), generator=g) for n in (6, 9, 12, 7)],
        "b": [torch.randint(64, 128, (1, n), generator=g) for n in (8, 10, 5, 11)],
    }


@pytest.fixture(scope="module")
def profiled(tiny, sets):
    return run_profile(tiny, sets, new_tokens=3, ablation_prompts=4)


# -- prompt sets ----------------------------------------------------------------


def test_builtin_prompt_sets():
    assert builtin_sets() == ["chat", "coding", "instruction", "math"]
    for name in builtin_sets():
        ps = load_prompt_set(name)
        assert len(ps.prompts) == len(set(ps.prompts)) == 50 and ps.source == "builtin"
    assert load_prompt_set("coding", limit=3).prompts == load_prompt_set("coding").prompts[:3]
    assert load_prompt_set("coding").digest != load_prompt_set("chat").digest


def test_user_prompt_files(tmp_path):
    txt = tmp_path / "mine.txt"
    txt.write_text("first prompt\n\nsecond prompt\n")
    ps = load_prompt_set(str(txt))
    assert (ps.name, ps.prompts, ps.source) == ("mine", ["first prompt", "second prompt"], "user")
    with pytest.raises(FileNotFoundError, match="not a built-in"):
        load_prompt_set("no-such-set")


def test_encode_prompts_fit_the_context(tiny):
    encoded = encode_prompts(tiny, ["x" * 500, "short"], reserve=10)
    assert encoded[0].shape == (1, 64 - 10 - 1)  # truncated to leave room to generate
    assert encoded[1].shape == (1, 5)


# -- profile run ----------------------------------------------------------------


def test_profile_result_shape(profiled, sets, tiny):
    result, by_set = profiled
    assert set(result["sets"]) == set(by_set) == {"a", "b"}
    for name, encoded in sets.items():
        prompt_tokens = sum(ids.shape[1] - 1 for ids in encoded)  # first token skipped
        tokens = result["sets"][name]["tokens"]
        assert prompt_tokens <= tokens <= prompt_tokens + 3 * len(encoded)
        assert len(result["sets"][name]["layers"]) == 3
    assert result["all"]["tokens"] == sum(s["tokens"] for s in result["sets"].values())
    assert result["run"]["prompts"] == 8
    assert len(result["memory"]["samples"]) >= 1
    assert _hook_count(tiny.model) == 0


def test_pooled_summary_matches_sets(profiled):
    result, by_set = profiled
    pooled = pool(list(by_set.values()))
    assert int(pooled["tokens"]) == result["all"]["tokens"]
    assert np.array_equal(pooled["ge"], by_set["a"]["ge"] + by_set["b"]["ge"])
    assert np.array_equal(
        pooled["abs_max"], np.maximum(by_set["a"]["abs_max"], by_set["b"]["abs_max"])
    )
    for row in summarise_layers(pooled):
        assert 0 <= row["cold_fraction"] <= 1 and 0 <= row["hot_fraction"] <= 1
        assert -1 <= row["in_out_cosine"] <= 1 and row["residual_norm"] > 0


def test_compare_sets_on_constructed_data():
    """Two sets, one layer, four neurons with known firing rates."""

    def arrays(rates):
        ge = np.zeros((1, len(EDGES), 4))
        ge[0, EDGES.index(0.1)] = np.array(rates) * 100
        zeros = np.zeros((1, 4))
        return {
            "tokens": np.array(100), "ge": ge, "sum": zeros, "abs_sum": zeros, "sq_sum": zeros,
            "abs_max": zeros, "block_cos_sum": np.zeros(1), "block_norm_sum": np.zeros(1),
            "block_tokens": np.array([100]),
        }  # fmt: skip

    #              specific to x   unused in x   shared   dead
    by_set = {"x": arrays([0.9, 0.0, 0.5, 0.0]), "y": arrays([0.1, 0.6, 0.5, 0.0])}
    assert np.allclose(firing_rate(by_set["x"])[0], [0.9, 0.0, 0.5, 0.0])

    cmp = compare_sets(by_set, top=2)
    assert cmp["x"]["specific_fraction"] == 0.25 and cmp["x"]["unused_fraction"] == 0.25
    assert cmp["y"]["specific_fraction"] == 0.25 and cmp["y"]["unused_fraction"] == 0.0
    assert cmp["x"]["top_specific"][0] == {
        "layer": 0, "neuron": 0, "rate": 0.9, "rate_elsewhere": 0.1
    }  # fmt: skip
    assert top_neurons(by_set, "y", 0, top=1) == [{"neuron": 1, "rate": 0.6, "rate_elsewhere": 0.0}]
    assert compare_sets({"x": by_set["x"]}) == {}  # nothing to compare a single set with


def test_drop_layer_ablation(tiny, sets):
    r = drop_layer_ablation(tiny, sets["a"])
    assert r["positions"] == sum(ids.shape[1] - 1 for ids in sets["a"])
    assert len(r["layers"]) == 3 and r["baseline_nll"] > 0
    for row in r["layers"]:
        assert row["drop_kl"] > 0  # KL divergence is non-negative; a real change is positive
        assert 0 <= row["drop_top1_agreement"] <= 1
    assert _hook_count(tiny.model) == 0


def test_redundancy_in_profile_result(profiled):
    red = profiled[0]["redundancy"]
    kls = {row["layer"]: row["drop_kl"] for row in red["layers"]}
    assert red["most_removable"] == sorted(kls, key=kls.get)
    assert all("in_out_cosine" in row for row in red["layers"])


def test_ablation_can_be_turned_off(tiny, sets):
    result, _ = run_profile(tiny, {"a": sets["a"][:2]}, new_tokens=0, ablation_prompts=0)
    assert result["redundancy"] is None and result["comparison"] == {}
    assert result["run"]["generated_tokens"] == 0


# -- disk -----------------------------------------------------------------------


def test_arrays_round_trip(profiled, tmp_path):
    _, by_set = profiled
    path = tmp_path / "activations.npz"
    save_arrays(path, by_set)
    loaded = load_arrays(path)
    assert set(loaded) == {"a", "b"}
    for name in by_set:
        for key, value in by_set[name].items():
            assert np.array_equal(loaded[name][key], value), (name, key)
    # Summaries computed from disk equal the ones computed live.
    assert summarise_layers(loaded["a"]) == summarise_layers(by_set["a"])


def test_neuron_detail(profiled):
    _, by_set = profiled
    d = neuron_detail(by_set, layer=1, neuron=7)
    assert set(d) == {"a", "b"}
    a = d["a"]
    assert a["max_abs"] >= a["mean_abs"] >= abs(a["mean"]) and a["std"] >= 0
    rates = list(a["reach_rate"].values())
    assert rates == sorted(rates, reverse=True)  # higher thresholds are reached less often


def test_layer_stats_rows(profiled):
    rows = layer_stats(profiled[0])
    # (2 sets + pooled) x 3 layers x 5 metrics, plus 3 layers x 3 ablation metrics
    assert len(rows) == 3 * 3 * 5 + 3 * 3


# -- job ------------------------------------------------------------------------


def test_profile_job_end_to_end(tiny_model_dir, tmp_path):
    out = run_job(
        "profile",
        {
            "model": str(tiny_model_dir),
            "sets": ["coding", "math"],
            "limit": 3,
            "new_tokens": 2,
            "ablation_prompts": 2,
            "device": "cpu",
        },
        home=tmp_path,
    )
    assert set(out["result"]["sets"]) == {"coding", "math"}
    assert [s["name"] for s in out["manifest"]["prompt_sets"]] == ["coding", "math"]

    with Storage(paths.db_path(tmp_path)) as db:
        artifacts = db.artifacts(out["run_id"])
        assert len(db.layer_stats(out["run_id"])) == 3 * 3 * 5 + 3 * 3
    loaded = load_arrays(artifacts["activations"])
    assert int(loaded["coding"]["tokens"]) == out["result"]["sets"]["coding"]["tokens"]
    assert summarise_layers(loaded["math"]) == out["result"]["sets"]["math"]["layers"]
