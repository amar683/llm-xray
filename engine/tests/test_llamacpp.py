import os
import time

import pytest

from xray.analyzers.quant import BITS, make_plan, plan_bytes
from xray.backends.llamacpp import (
    GGUF_BPW,
    GGUF_TYPES,
    LlamaCppError,
    find_tools,
    gguf_name,
    heldout_text,
    parse_kld,
)
from xray.data import load_prompt_set

KLD_LOG = """
Mean PPL(Q)                   :   2.097630 ±   0.101127
Mean PPL(base)                :   2.014500 ±   0.093630
====== KL divergence statistics ======
Mean    KLD:   0.043750 ±   0.002748
Maximum KLD:   1.863891
RMS Δp    :  6.698 ± 0.311 %
Same top p: 92.222 ± 0.755 %
"""


@pytest.mark.parametrize(
    ("hf", "gguf"),
    [
        ("model.layers.3.self_attn.q_proj.weight", "blk.3.attn_q.weight"),
        ("model.layers.27.self_attn.o_proj.weight", "blk.27.attn_output.weight"),
        ("model.layers.0.mlp.down_proj.weight", "blk.0.ffn_down.weight"),
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("lm_head.weight", "output.weight"),
    ],
)
def test_gguf_name(hf, gguf):
    assert gguf_name(hf) == gguf


def test_gguf_name_rejects_unknown_tensors():
    with pytest.raises(LlamaCppError):
        gguf_name("model.layers.0.something_new.weight")


def test_every_plan_width_has_a_gguf_type():
    assert set(GGUF_TYPES) == set(GGUF_BPW) == set(BITS)


def test_parse_kld():
    r = parse_kld(KLD_LOG)
    assert r["kl"] == pytest.approx(0.04375) and r["kl_err"] == pytest.approx(0.002748)
    assert r["top1_agreement"] == pytest.approx(0.92222)
    assert r["ppl"] == pytest.approx(2.09763) and r["ppl_base"] == pytest.approx(2.0145)
    with pytest.raises(LlamaCppError):
        parse_kld("no statistics here")


def test_find_tools_picks_the_newest_build(tmp_path, monkeypatch):
    monkeypatch.delenv("XRAY_LLAMA_CPP", raising=False)
    with pytest.raises(LlamaCppError, match="Pass --llama-cpp"):
        find_tools(None)
    with pytest.raises(LlamaCppError, match="does not look like"):
        find_tools(tmp_path)

    (tmp_path / "convert_hf_to_gguf.py").touch()
    for i, build in enumerate(("build-old", "build")):
        bin_dir = tmp_path / build / "bin"
        bin_dir.mkdir(parents=True)
        for tool in ("llama-quantize", "llama-perplexity"):
            (bin_dir / tool).touch()
            os.utime(bin_dir / tool, (time.time() + i * 100,) * 2)
    tools = find_tools(tmp_path)
    assert tools.quantize == tmp_path / "build" / "bin" / "llama-quantize"
    assert tools.perplexity.parent == tools.quantize.parent

    monkeypatch.setenv("XRAY_LLAMA_CPP", str(tmp_path))
    assert find_tools(None).root == tmp_path


def test_heldout_text_excludes_calibration_prompts():
    text = heldout_text(["coding", "math"], calibration_prompts=8)
    lines = text.strip().splitlines()
    assert len(lines) == 2 * (50 - 4)
    assert not set(lines) & set(load_prompt_set("coding").prompts[:4])


def test_plan_fits_a_budget_in_real_gguf_bits():
    def tensor(name, kl):
        return {"name": name, "params": 10_000, "kl": kl,
                "weighted_error": {str(b): 4.0 ** -b for b in BITS}}  # fmt: skip

    tensors = [tensor("a", 1.0), tensor("b", 0.01), tensor("c", 0.0001)]
    budget = plan_bytes(tensors, {t["name"]: 4 for t in tensors}, GGUF_BPW.get)
    plan = make_plan(tensors, budget, 4, GGUF_BPW.get)
    assert plan_bytes(tensors, plan, GGUF_BPW.get) <= budget
    assert plan["a"] > plan["c"]
