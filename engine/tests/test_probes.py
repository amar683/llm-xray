from collections import defaultdict

import numpy as np
import torch

from xray.analyzers.timing import module_sets
from xray.probes.hooks import EDGES, ActivationProbe, SkipProbe, TimingProbe


def _hook_count(model) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


def test_probes_do_not_change_outputs_and_clean_up(tiny, prompts):
    ids = prompts[1]
    modules, partition, _ = module_sets(tiny)
    sink = defaultdict(list)
    with torch.inference_mode():
        before = tiny.model(input_ids=ids).logits
        with TimingProbe(modules, tiny.device) as timing, ActivationProbe(tiny.refs):
            timing.measure(partition, synced=True, sink=sink)
            assert _hook_count(tiny.model) > 0
            during = tiny.model(input_ids=ids).logits
    assert torch.equal(before, during)
    assert _hook_count(tiny.model) == 0


def test_timing_probe_only_times_selected_modules(tiny, prompts):
    modules, partition, components = module_sets(tiny)
    assert partition == ["embed", "layer.0", "layer.1", "layer.2", "final_norm", "lm_head"]
    assert len(components) == 6

    blocks, comps = defaultdict(list), defaultdict(list)
    with TimingProbe(modules, tiny.device) as probe, torch.inference_mode():
        tiny.model(input_ids=prompts[0])  # nothing selected: nothing recorded
        probe.measure(partition, synced=True, sink=blocks)
        for ids in prompts:
            tiny.model(input_ids=ids)
        probe.measure(components, synced=False, sink=comps)
        tiny.model(input_ids=prompts[0])
        probe.off()
        tiny.model(input_ids=prompts[0])

    assert set(blocks) == set(partition) and set(comps) == set(components)
    assert all(len(v) == len(prompts) and min(v) > 0 for v in blocks.values())
    assert all(len(v) == 1 for v in comps.values())


def test_activation_probe_matches_direct_computation(tiny, prompts):
    ids = prompts[2]
    captured = {}
    handle = tiny.refs.layers[1].mlp_out.register_forward_pre_hook(
        lambda _m, args: captured.setdefault("x", args[0].detach().clone())
    )
    with ActivationProbe(tiny.refs) as probe, torch.inference_mode():
        tiny.model(input_ids=ids)
        arr = probe.arrays("all")
    handle.remove()

    acts = captured["x"][0, 1:]  # position 0 is skipped by design
    tokens = acts.shape[0]
    assert int(arr["tokens"]) == tokens == 11
    assert arr["sum"].shape == (3, 64) and arr["ge"].shape == (3, len(EDGES), 64)
    assert np.allclose(arr["sum"][1], acts.sum(0).numpy(), atol=1e-5)
    assert np.allclose(arr["abs_sum"][1], acts.abs().sum(0).numpy(), atol=1e-5)
    assert np.allclose(arr["sq_sum"][1], (acts**2).sum(0).numpy(), atol=1e-5)
    assert np.allclose(arr["abs_max"][1], acts.abs().max(0).values.numpy())
    for k, edge in enumerate(EDGES):
        assert np.array_equal(arr["ge"][1, k], (acts.abs() >= edge).sum(0).numpy())
    assert (arr["block_tokens"] == tokens).all()
    assert (np.abs(arr["block_cos_sum"] / tokens) <= 1.0).all()


def test_activation_probe_groups_and_decode_tokens(tiny, prompts):
    with ActivationProbe(tiny.refs) as probe, torch.inference_mode():
        probe.begin("a")
        probe.skip_first = True
        out = tiny.model(input_ids=prompts[0], use_cache=True)  # 5 tokens -> 4 counted
        probe.skip_first = False
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        tiny.model(input_ids=nxt, past_key_values=out.past_key_values)  # +1 decode token
        probe.begin("b")
        probe.skip_first = True
        tiny.model(input_ids=prompts[1])  # 9 tokens -> 8 counted
        a, b = probe.arrays("a"), probe.arrays("b")
    assert (int(a["tokens"]), int(b["tokens"])) == (5, 8)
    assert not np.allclose(a["abs_sum"], b["abs_sum"])


def test_skip_probe_removes_a_block(tiny, prompts):
    ids = prompts[1]
    with torch.inference_mode():
        base = tiny.model(input_ids=ids).logits
        with SkipProbe(tiny.refs) as skip:
            untouched = tiny.model(input_ids=ids).logits
            skip.layers = frozenset({1})
            skipped = tiny.model(input_ids=ids).logits
            skip.layers = frozenset({0, 1, 2})
            all_skipped = tiny.model(input_ids=ids).logits
        # With every block skipped the model is just embed -> norm -> lm_head.
        refs = tiny.refs
        expected = refs.lm_head(refs.final_norm(refs.embed(ids)))
    assert torch.equal(base, untouched)
    assert not torch.allclose(base, skipped)
    assert torch.allclose(all_skipped, expected, atol=1e-5)
    assert _hook_count(tiny.model) == 0
