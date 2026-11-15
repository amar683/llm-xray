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
