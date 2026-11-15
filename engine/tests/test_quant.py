import pytest
import torch

from xray.analyzers.quant import BITS, fake_quant, make_plan, plan_bytes, quant_targets, run_quant


def test_fake_quant_error_shrinks_with_more_bits():
    w = torch.randn(8, 64, generator=torch.Generator().manual_seed(0))
    errors = [((w - fake_quant(w, b)) ** 2).mean().item() for b in BITS]
    assert errors == sorted(errors, reverse=True) and errors[-1] < 1e-4
    assert fake_quant(w, 4).shape == w.shape
    assert fake_quant(torch.randn(3, 50), 4).shape == (3, 50)  # width not a multiple of 32


def test_slicing_big_matrices_changes_nothing(monkeypatch):
    import xray.analyzers.quant as quant

    g = torch.Generator().manual_seed(1)
    w = (torch.randn(300, 70, generator=g) * 0.05).to(torch.float16)
    w[7, 3] = 1.5  # an outlier
    imp = torch.rand(70, generator=g)
    whole_q, whole_stats = quant.fake_quant(w, 4), quant.weight_stats(w, imp)
    monkeypatch.setattr(quant, "CHUNK_VALUES", 1000)  # forces 15+ slices of a few rows
    assert len(list(quant._row_slices(w))) > 10
    assert torch.equal(quant.fake_quant(w, 4), whole_q)
    sliced = quant.weight_stats(w, imp)
    for key in ("std", "max_abs", "outlier_ratio", "kurtosis"):
        assert sliced[key] == pytest.approx(whole_stats[key], rel=1e-6), key
    for b in BITS:
        assert sliced["rel_error"][str(b)] == pytest.approx(
            whole_stats["rel_error"][str(b)], rel=1e-5
        )
        assert sliced["weighted_error"][str(b)] == pytest.approx(
            whole_stats["weighted_error"][str(b)], rel=1e-5
        )


def test_weight_stats_match_direct_formulas():
    from xray.analyzers.quant import weight_stats

    w = torch.randn(64, 64, generator=torch.Generator().manual_seed(2))
    s = weight_stats(w, None)
    x = w.double()
    assert s["std"] == pytest.approx(x.std(unbiased=False).item(), rel=1e-6)
    kurt = (((x - x.mean()) ** 4).mean() / x.var(unbiased=False) ** 2).item()
    assert s["kurtosis"] == pytest.approx(kurt, rel=1e-6)
    assert s["max_abs"] == pytest.approx(x.abs().max().item())


def test_plan_respects_budget_and_spends_bits_on_sensitive_tensors():
    def tensor(name, kl):
        return {"name": name, "params": 1000, "kl": kl,
                "weighted_error": {str(b): 4.0 ** -b for b in BITS}}  # fmt: skip

    tensors = [tensor("fragile", 1.0), tensor("robust", 0.001)]
    budget = plan_bytes(tensors, {"fragile": 4, "robust": 4})
    plan = make_plan(tensors, budget, probe_bits=4)
    assert plan_bytes(tensors, plan) <= budget
    assert plan["fragile"] > plan["robust"]


def test_plan_is_not_blocked_by_a_noisy_measurement():
    # Measured damage at 4 bits is (wrongly) higher than at 3; 5 bits is far better.
    noisy = {"name": "noisy", "params": 1000, "kl": 0.2, "kl_by_bits": {"3": 0.1, "4": 0.2},
             "weighted_error": {str(b): 4.0 ** -b for b in BITS}}  # fmt: skip
    plan = make_plan([noisy], budget_bytes=1000 * 8.5 / 8, probe_bits=4)
    assert plan["noisy"] >= 5


def test_kl_is_never_negative_even_with_rounding_noise(tiny, prompts):
    from xray.analyzers.quant import compare, log_probs

    base = log_probs(tiny, prompts)
    # Comparing the model with itself: the true KL is exactly zero.
    assert compare(tiny, prompts, base)["kl"] == 0.0
    nudged = [b + 1e-7 for b in base]  # a slightly different "base" gives a tiny negative sum
    assert compare(tiny, prompts, nudged)["kl"] >= 0.0


def test_run_quant_end_to_end_and_restores_weights(tiny, prompts):
    before = {n: p.data.clone() for n, p in quant_targets(tiny).items()}
    r = run_quant(tiny, prompts[:2], prompts[2:], probe_bits=4, targets_bits=(4,))
    assert len(r["tensors"]) == len(before) == 3 * 7 + 2
    assert all(t["kl"] >= 0 for t in r["tensors"]) and len(r["layers"]) == 3
    plan = r["plans"][0]
    assert plan["mixed"]["bytes"] <= plan["budget_bytes"]
    assert plan["uniform"]["kl"] > 0 and plan["mixed"]["kl"] > 0
    for n, p in quant_targets(tiny).items():
        assert torch.equal(p.data, before[n]), n
