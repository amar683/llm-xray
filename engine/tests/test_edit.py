import numpy as np
import pytest
import torch

from xray.analyzers.common import greedy_generate
from xray.analyzers.edit import save_model
from xray.analyzers.loops import Ablation
from xray.core.jobs import JobError, run_job
from xray.eval import Evaluator, builtin_tasks, is_correct, load_task
from xray.models.adapters import load_model
from xray.patch import PatchError, apply_commands, neuron_importance, parse_command, parse_commands
from xray.patch.heal import LoRALinear, heal
from xray.probes.hooks import SkipProbe


@pytest.fixture
def fresh(tiny_model_dir):
    """A private float32 copy of the tiny model: these tests change its weights."""
    return load_model(str(tiny_model_dir), device="cpu", with_tokenizer=False)


@pytest.fixture(scope="module")
def ids():
    return torch.randint(0, 128, (1, 12), generator=torch.Generator().manual_seed(7))


def logits(lm, ids):
    with torch.inference_mode():
        return lm.model(input_ids=ids).logits


# -- commands -------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        r"C:\Users\runner\AppData\Local\Temp\d.npy",
        r"C:\Program Files\x\d.npy",
        "/home/me/a b/d.npy",
    ],
)
def test_paths_survive_parsing_on_every_system(path):
    quoted = parse_command(f'ablate_direction vector="{path}" from_layer=1')
    assert quoted["vector"] == path
    if " " not in path:  # unquoted works too, as long as there are no spaces
        assert parse_command(f"ablate_direction vector={path} from_layer=1")["vector"] == path


def test_parse_commands():
    assert parse_command("zero_head layer=13 head=4") == {"op": "zero_head", "layer": 13, "head": 4}
    assert parse_command("drop_layers layers=16,10,10")["layers"] == [10, 16]
    assert parse_command("scale_neuron layer=1 neuron=5 factor=0.5")["factor"] == 0.5
    assert len(parse_commands(["# a comment", "", "prune_neurons keep=0.9"])) == 1


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("explode layer=1", "Unknown command"),
        ("zero_head layer=1", "missing: head"),
        ("zero_head layer=one head=2", "must be int"),
        ("zero_head layer=1 head=2 extra=3", "does not take"),
        ("prune_neurons keep=1.5", "at most 1"),
        ("drop_layers layers=a,b", "numbers separated by commas"),
    ],
)
def test_bad_commands_are_rejected_with_a_reason(text, message):
    with pytest.raises(PatchError, match=message):
        parse_command(text)


def test_out_of_range_targets_are_rejected(fresh):
    for text in (
        "zero_head layer=9 head=0",
        "zero_head layer=0 head=9",
        "drop_layers layers=0,1,2",
    ):
        with pytest.raises(PatchError):
            apply_commands(fresh, parse_commands([text]))


def test_zero_head_matches_switching_the_head_off_at_run_time(fresh, ids):
    ablation = Ablation(fresh)
    ablation.heads = {(1, 2)}
    expected = logits(fresh, ids)
    ablation.remove()
    apply_commands(fresh, parse_commands(["zero_head layer=1 head=2"]))
    assert torch.allclose(logits(fresh, ids), expected, atol=1e-5)


def test_scale_neuron(fresh, ids):
    before = logits(fresh, ids)
    apply_commands(fresh, parse_commands(["scale_neuron layer=0 neuron=3 factor=0"]))
    assert fresh.refs.layers[0].mlp_out.weight[:, 3].abs().max() == 0
    assert not torch.allclose(logits(fresh, ids), before)


def test_drop_layers_matches_skipping_and_still_generates(fresh, ids):
    with SkipProbe(fresh.refs) as skip:
        skip.layers = frozenset({1})
        expected = logits(fresh, ids)
    fresh.model.config.layer_types = ["full_attention"] * 3  # a per-layer list, as Qwen has
    log = apply_commands(fresh, parse_commands(["drop_layers layers=1"]))
    assert fresh.model.config.layer_types == ["full_attention"] * 2
    assert torch.allclose(logits(fresh, ids), expected, atol=1e-5)
    assert len(fresh.refs.layers) == 2 and fresh.model.config.num_hidden_layers == 2
    assert "2 remain" in log[0]
    tokens, _ = greedy_generate(fresh, ids, 5)  # exercises the key/value cache
    assert len(tokens) <= 5


def test_prune_neurons(fresh, ids):
    calibration = [ids]
    before = logits(fresh, ids)
    apply_commands(fresh, parse_commands(["prune_neurons keep=1.0"]), calibration)
    assert torch.allclose(
        logits(fresh, ids), before, atol=1e-5
    )  # keeping everything changes nothing

    importance = neuron_importance(fresh, calibration)
    apply_commands(fresh, parse_commands(["prune_neurons keep=0.5"]), calibration)
    mlp = fresh.refs.layers[0].mlp
    assert mlp.gate_proj.out_features == mlp.up_proj.out_features == mlp.down_proj.in_features == 32
    assert fresh.model.config.intermediate_size == 32
    assert importance.shape == (3, 64)
    assert logits(fresh, ids).shape == before.shape
    with pytest.raises(PatchError, match="calibration"):
        apply_commands(fresh, parse_commands(["prune_neurons keep=0.5"]))


def test_ablate_direction(fresh, ids, tmp_path):
    d = torch.randn(32, generator=torch.Generator().manual_seed(1))
    np.save(tmp_path / "d.npy", d.numpy())
    apply_commands(
        fresh, parse_commands([f"ablate_direction vector={tmp_path / 'd.npy'} from_layer=1"])
    )
    unit = d / d.norm()
    for layer in fresh.refs.layers[1:]:
        assert (unit @ layer.attn_out.weight).abs().max() < 1e-5
        assert (unit @ layer.mlp_out.weight).abs().max() < 1e-5
    assert (
        unit @ fresh.refs.layers[0].mlp_out.weight
    ).abs().max() > 1e-3  # earlier layer untouched
    np.save(tmp_path / "bad.npy", np.zeros(7))
    with pytest.raises(PatchError, match="shape"):
        apply_commands(
            fresh, parse_commands([f"ablate_direction vector={tmp_path / 'bad.npy'} from_layer=0"])
        )


def test_edited_model_saves_and_reloads(fresh, ids, tmp_path):
    apply_commands(fresh, parse_commands(["drop_layers layers=0", "prune_neurons keep=0.5"]), [ids])
    expected = logits(fresh, ids)
    saved = save_model(fresh, tmp_path / "edited")
    assert saved["weights_bytes"] > 0
    reloaded = load_model(str(tmp_path / "edited"), device="cpu", with_tokenizer=False)
    assert len(reloaded.refs.layers) == 2
    assert torch.allclose(logits(reloaded, ids), expected, atol=1e-2)  # saved in half precision


# -- healing --------------------------------------------------------------------


def test_healing_reduces_the_distance_to_the_base_model(fresh):
    g = torch.Generator().manual_seed(2)
    train = [torch.randint(0, 128, (1, 10), generator=g) for _ in range(6)]
    with torch.inference_mode():
        teacher = [fresh.model(input_ids=x).logits[0].to(torch.float16) for x in train]
    apply_commands(fresh, parse_commands(["prune_neurons keep=0.5"]), train)
    report = heal(fresh, train, teacher, steps=60, rank=4, lr=1e-2)
    assert report["loss_end"] < report["loss_start"]
    assert report["steps"] == 60 and report["trained_params"] > 0
    # Adapters are merged away: plain linear layers, no stray gradients.
    assert not any(isinstance(m, LoRALinear) for m in fresh.model.modules())
    assert all(p.grad is None for p in fresh.model.parameters())
    assert fresh.refs.layers[0].mlp_out.in_features == 32


# -- evals ----------------------------------------------------------------------


def test_tasks_split_into_disjoint_halves():
    assert builtin_tasks() == ["arithmetic", "capitals", "json"]  # code is opt-in
    assert builtin_tasks(run_code=True) == ["arithmetic", "capitals", "code", "json"]
    for name in builtin_tasks(run_code=True):
        tune, test = load_task(name, "tune"), load_task(name, "test")
        assert len(tune) == len(test) and not {i["prompt"] for i in tune} & {
            i["prompt"] for i in test
        }


@pytest.mark.parametrize(
    ("output", "answer", "ok"),
    [
        (" Paris.", "Paris", True), (" paris, of course", "Paris", True), (" 19\n", "19", True),
        (" 195", "19", False), (" Parisian", "Paris", False), (" London", "Paris", False),
        (" Bogotá", "Bogot", True), ("", "Paris", False),
    ],
)  # fmt: skip
def test_is_correct(output, answer, ok):
    assert is_correct(output, answer) is ok


def test_evaluator_reports_no_drift_for_an_unchanged_model(tiny):
    evaluator = Evaluator(tiny, health_tokens=10)
    r = evaluator.evaluate(tiny)
    assert r["drift"]["kl"] == pytest.approx(0.0, abs=1e-6) and r["drift"]["top1_agreement"] == 1.0
    assert set(r["tasks"]) == {"arithmetic", "capitals", "json"}
    with pytest.raises(ValueError, match="run_code"):
        Evaluator(tiny, tasks=["code"])
    assert r["tasks"]["capitals"]["n"] == 25 and r["size"]["layers"] == 3


# -- job ------------------------------------------------------------------------


def test_edit_job_end_to_end(tiny_model_dir, tmp_path):
    out = run_job(
        "edit",
        {
            "model": str(tiny_model_dir),
            "commands": ["drop_layers layers=1", "prune_neurons keep=0.5"],
            "heal": {"steps": 20, "rank": 2, "lr": 1e-2},
            "save": True,
            "device": "cpu",
        },
        home=tmp_path,
    )
    r = out["result"]
    assert list(r["stages"]) == ["base", "edited", "healed"]
    assert r["stages"]["base"]["drift"]["kl"] == pytest.approx(0.0, abs=1e-6)
    assert r["stages"]["edited"]["drift"]["kl"] > 0
    assert r["stages"]["edited"]["size"]["params"] < r["stages"]["base"]["size"]["params"]
    assert r["stages"]["healed"]["size"] == r["stages"]["edited"]["size"]
    assert len(r["applied"]) == 2 and r["healing"]["steps"] == 20
    reloaded = load_model(r["saved"]["path"], device="cpu", with_tokenizer=False)
    assert len(reloaded.refs.layers) == 2


def test_edit_job_rejects_a_bad_patch_before_loading(tiny_model_dir, tmp_path):
    with pytest.raises(JobError, match="Unknown command"):
        run_job(
            "edit", {"model": str(tiny_model_dir), "commands": ["explode now=1"]}, home=tmp_path
        )
