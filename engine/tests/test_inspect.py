import pytest

from xray.core.manifest import model_fingerprint
from xray.models.inspect import classify_tensor, inspect_model, resolve_model_dir


def test_param_count_matches_loaded_model(tiny_model_dir, tiny):
    report = inspect_model(str(tiny_model_dir))
    assert report["totals"]["params"] == sum(p.numel() for p in tiny.model.parameters())


def test_architecture_and_layers(tiny_model_dir):
    report = inspect_model(str(tiny_model_dir))
    arch = report["architecture"]
    assert (arch["model_type"], arch["n_layers"], arch["hidden_size"]) == ("llama", 3, 32)
    assert arch["head_dim"] == 8
    assert [layer["index"] for layer in report["layers"]] == [0, 1, 2]
    assert set(report["layers"][0]["components"]) == {"attention", "mlp", "norm"}
    assert sum(report["by_component"].values()) == report["totals"]["params"]


def test_memory_estimates_shrink_with_quantisation(tiny_model_dir):
    w = inspect_model(str(tiny_model_dir))["memory_estimates"]["weights_bytes"]
    assert w["F16"] > w["Q8_0"] > w["Q4_K_M"] > w["Q3_K_M"] > 0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.7.self_attn.q_proj.weight", (7, "attention")),
        ("model.layers.12.mlp.down_proj.weight", (12, "mlp")),
        ("model.layers.0.input_layernorm.weight", (0, "norm")),
        ("model.embed_tokens.weight", (None, "embeddings")),
        ("model.norm.weight", (None, "final_norm")),
        ("lm_head.weight", (None, "lm_head")),
    ],
)
def test_classify_tensor(name, expected):
    assert classify_tensor(name) == expected


def test_missing_model_never_downloads():
    with pytest.raises(FileNotFoundError):
        resolve_model_dir("xray-tests/definitely-not-a-cached-model")


def test_fingerprint_is_stable(tiny_model_dir):
    assert model_fingerprint(tiny_model_dir) == model_fingerprint(tiny_model_dir)
    assert len(model_fingerprint(tiny_model_dir)) == 16


# -- compatibility --------------------------------------------------------------

import json
import shutil

from safetensors.torch import load_file, save_file

from xray.models.adapters import load_model
from xray.models.inspect import UnsupportedModel, compatibility_of


def _copy_model(src, dst, config_changes=None, drop=()):
    """A copy of the tiny model with its config edited and some weights removed."""
    dst.mkdir()
    config = json.loads((src / "config.json").read_text())
    config.update(config_changes or {})
    (dst / "config.json").write_text(json.dumps(config))
    weights = load_file(src / "model.safetensors")
    save_file(
        {k: v for k, v in weights.items() if not any(d in k for d in drop)},
        dst / "model.safetensors",
    )
    return dst


def test_the_tiny_model_is_compatible(tiny_model_dir):
    r = compatibility_of(tiny_model_dir)
    assert r["ok"] and r["problems"] == []
    assert any("tokenizer" in w for w in r["warnings"])  # the test model has none
    assert inspect_model(str(tiny_model_dir))["compatibility"]["ok"]


def test_a_foreign_architecture_is_refused(tiny_model_dir, tmp_path):
    m = _copy_model(tiny_model_dir, tmp_path / "draft", {"architectures": ["DSpineDraftModel"]})
    r = compatibility_of(m)
    assert not r["ok"] and "DSpineDraftModel" in r["problems"][0]


def test_missing_weights_are_refused_before_loading(tiny_model_dir, tmp_path):
    m = _copy_model(tiny_model_dir, tmp_path / "headless", drop=("embed_tokens", "lm_head"))
    r = compatibility_of(m)
    assert not r["ok"]
    assert "the embeddings, the output head" in r["problems"][0] and "random" in r["problems"][0]
    with pytest.raises(UnsupportedModel, match="cannot be analysed"):
        load_model(str(m), device="cpu", with_tokenizer=False)


def test_weights_without_the_model_prefix_are_accepted(tiny_model_dir, tmp_path):
    m = tmp_path / "unprefixed"
    m.mkdir()
    shutil.copy(tiny_model_dir / "config.json", m / "config.json")
    weights = load_file(tiny_model_dir / "model.safetensors")
    save_file({k.removeprefix("model."): v for k, v in weights.items()}, m / "model.safetensors")
    assert compatibility_of(m)["ok"]


def test_an_unknown_model_type_is_refused(tiny_model_dir, tmp_path):
    m = _copy_model(
        tiny_model_dir,
        tmp_path / "other",
        {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]},
    )
    problems = compatibility_of(m)["problems"]
    assert any("gpt2" in p for p in problems) and any("GPT2LMHeadModel" in p for p in problems)


def test_jobs_refuse_an_incompatible_model_with_its_reason(tiny_model_dir, tmp_path):
    from xray.core.jobs import JobError, run_job

    m = _copy_model(tiny_model_dir, tmp_path / "draft2", {"architectures": ["DSpineDraftModel"]})
    with pytest.raises(JobError, match="DSpineDraftModel"):
        run_job("quant", {"model": str(m), "device": "cpu"}, home=tmp_path / "home")
