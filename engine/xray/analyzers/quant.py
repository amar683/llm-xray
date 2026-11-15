"""Quantization sensitivity and mixed-precision planning.

Everything here is *simulated* in PyTorch: weights are rounded to a few bits
with one scale and offset per block of 32 values, which approximates GGUF's
block quantizers but is not identical to them. A plan found here is a
candidate; the real check is an exported GGUF measured with llama.cpp.

Steps:
  1. importance   - mean squared input of every linear layer over the
                    calibration prompts (what llama.cpp calls an imatrix)
  2. weight stats - per tensor: spread, outliers, rounding error at each bit
                    width, plain and importance-weighted
  3. sensitivity  - quantize one tensor at a time and measure how far the
                    model's predictions move (KL divergence, loss, top-1)
  4. planning     - choose a bit width per tensor to fit a size budget,
                    spending bits where they reduce predicted damage most
  5. verification - quantize the whole model to the plan, and to a uniform
                    width of the same size, and compare both on held-out prompts
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from xray.analyzers.profile import encode_prompts
from xray.core.device import default_dtype, pick_device, system_info
from xray.core.manifest import build_manifest, model_fingerprint
from xray.core.memory import guard_memory
from xray.data import load_prompt_set
from xray.models.adapters import LoadedModel, load_model
from xray.models.inspect import classify_tensor, inspect_model

BITS = (3, 4, 5, 6, 8)
BLOCK = 32
# Storage for each block's scale and offset, spread over its 32 weights.
OVERHEAD_BITS = 0.5
FULL_BITS = 16.0

_KIND_RE = re.compile(r"\.(\w+)\.weight$")


def bits_per_weight(bits: int) -> float:
    return bits + OVERHEAD_BITS


def fake_quant(w: torch.Tensor, bits: int, block: int = BLOCK) -> torch.Tensor:
    """Round `w` to `bits` bits, with one scale and offset per block along each row."""
    rows, cols = w.shape
    pad = (-cols) % block
    x = w.float()
    if pad:
        x = torch.nn.functional.pad(x, (0, pad), mode="replicate")
    x = x.reshape(rows, -1, block)
    lo = x.amin(-1, keepdim=True)
    hi = x.amax(-1, keepdim=True)
    levels = 2**bits - 1
    scale = (hi - lo) / levels
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = ((x - lo) / scale).round().clamp_(0, levels) * scale + lo
    return q.reshape(rows, -1)[:, :cols].to(w.dtype)


def quant_targets(lm: LoadedModel) -> dict[str, torch.nn.Parameter]:
    """The 2-D weight matrices worth quantizing, by parameter name.

    Norms and biases are tiny and stay at full precision. A tied output head
    shares its matrix with the embeddings and appears once.
    """
    seen: set[int] = set()
    targets = {}
    for name, p in lm.model.named_parameters():
        if p.ndim != 2 or id(p) in seen:
            continue
        seen.add(id(p))
        targets[name] = p
    return targets


def describe(name: str) -> dict[str, Any]:
    layer, component = classify_tensor(name)
    m = _KIND_RE.search(name)
    return {"name": name, "layer": layer, "component": component, "kind": m.group(1) if m else name}


# -- measuring predictions --------------------------------------------------------


@torch.inference_mode()
def log_probs(lm: LoadedModel, encoded: list[torch.Tensor]) -> list[torch.Tensor]:
    """Next-token log-probabilities at every position of every prompt."""
    return [
        torch.log_softmax(lm.model(input_ids=ids.to(lm.device)).logits[0].float(), dim=-1)
        for ids in encoded
    ]


def _nll(logp: torch.Tensor, ids: torch.Tensor) -> float:
    return -logp[:-1].gather(1, ids[0, 1:, None].to(logp.device)).sum().item()


@torch.inference_mode()
def compare(lm: LoadedModel, encoded: list[torch.Tensor], base: list[torch.Tensor]) -> dict:
    """How far the model's current predictions are from `base` on `encoded`."""
    kl = agree = nll = base_nll = 0.0
    positions = 0
    for ids, b in zip(encoded, base, strict=True):
        logp = torch.log_softmax(lm.model(input_ids=ids.to(lm.device)).logits[0].float(), dim=-1)
        kl += (b[1:].exp() * (b[1:] - logp[1:])).sum().item()
        agree += (logp[1:].argmax(-1) == b[1:].argmax(-1)).sum().item()
        nll += _nll(logp, ids)
        base_nll += _nll(b, ids)
        positions += ids.shape[1] - 1
    positions = max(positions, 1)
    return {
        "kl": kl / positions,
        "top1_agreement": agree / positions,
        "nll_increase": (nll - base_nll) / positions,
    }


# -- importance and weight statistics ---------------------------------------------


@torch.inference_mode()
def input_importance(
    lm: LoadedModel, encoded: list[torch.Tensor], targets: dict[str, torch.nn.Parameter]
) -> dict[str, torch.Tensor]:
    """Mean squared input per input channel for every targeted linear layer."""
    by_param = {id(p): name for name, p in targets.items()}
    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make(name: str):
        def hook(_module, args):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            sums[name] = sums.get(name, 0) + (x * x).sum(0)
            counts[name] = counts.get(name, 0) + x.shape[0]

        return hook

    for module in lm.model.modules():
        if isinstance(module, torch.nn.Linear) and id(module.weight) in by_param:
            handles.append(module.register_forward_pre_hook(make(by_param[id(module.weight)])))
    try:
        for ids in encoded:
            lm.model(input_ids=ids.to(lm.device))
    finally:
        for h in handles:
            h.remove()
    return {name: sums[name] / counts[name] for name in sums}


@torch.inference_mode()
def weight_stats(w: torch.Tensor, importance: torch.Tensor | None) -> dict[str, Any]:
    """Shape of the weight distribution and rounding error at each bit width.

    `rel_error` is ||w - q||^2 / ||w||^2. `weighted_error` weights each input
    column by how strongly the model drives it, so errors on busy inputs count
    for more; it tracks real damage better than the plain error.
    """
    x = w.float()
    std = x.std().item()
    centred = x - x.mean()
    imp = importance.float() if importance is not None else None
    if imp is not None and imp.shape[0] != x.shape[1]:
        imp = None  # e.g. a tied embedding matrix: its hooked input is the hidden state
    denom = (x * x).sum().item() or 1.0
    w_denom = ((x * x) * imp).sum().item() if imp is not None else denom
    rel, weighted = {}, {}
    for bits in BITS:
        err = (x - fake_quant(w, bits).float()) ** 2
        rel[str(bits)] = err.sum().item() / denom
        weighted[str(bits)] = (
            (err * imp).sum().item() / (w_denom or 1.0) if imp is not None else rel[str(bits)]
        )
    return {
        "params": x.numel(),
        "std": std,
        "max_abs": x.abs().max().item(),
        "outlier_ratio": x.abs().max().item() / std if std else 0.0,
        "kurtosis": ((centred**4).mean() / (std**4)).item() if std else 0.0,
        "rel_error": rel,
        "weighted_error": weighted,
    }


# -- planning ---------------------------------------------------------------------


def predicted_damage(tensor: dict[str, Any], bits: int, probe_bits: int) -> float:
    """Expected KL from quantizing this tensor to `bits`.

    Uses the measured KL when this width was tested. Otherwise scales the KL
    from the nearest tested width by how the importance-weighted rounding error
    changes with bit width.
    """
    measured = {int(k): v for k, v in tensor.get("kl_by_bits", {}).items()}
    measured.setdefault(probe_bits, tensor["kl"])
    if bits in measured:
        return measured[bits]
    anchor = min(measured, key=lambda m: abs(m - bits))
    anchor_err = tensor["weighted_error"][str(anchor)]
    if anchor_err <= 0:
        return 0.0
    return measured[anchor] * tensor["weighted_error"][str(bits)] / anchor_err


def plan_bytes(
    tensors: list[dict[str, Any]],
    assignment: dict[str, int],
    bpw: Callable[[int], float] = bits_per_weight,
) -> float:
    """Size of the planned weights. `bpw` maps a bit width to stored bits per weight."""
    return sum(t["params"] * bpw(assignment[t["name"]]) / 8 for t in tensors)


def make_plan(
    tensors: list[dict[str, Any]],
    budget_bytes: float,
    probe_bits: int,
    bpw: Callable[[int], float] = bits_per_weight,
) -> dict[str, int]:
    """Choose a bit width per tensor within `budget_bytes`.

    Greedy: start everything at the lowest width, then repeatedly make the one
    upgrade (of any tensor, to any higher width) that removes the most
    predicted damage per byte. Considering every higher width, not just the
    next one, keeps a single noisy measurement from blocking a tensor.
    """
    assignment = {t["name"]: BITS[0] for t in tensors}
    spent = plan_bytes(tensors, assignment, bpw)
    by_name = {t["name"]: t for t in tensors}
    while True:
        best, best_gain = None, 0.0
        for name, bits in assignment.items():
            t = by_name[name]
            now = predicted_damage(t, bits, probe_bits)
            for nxt in BITS[BITS.index(bits) + 1 :]:
                cost = t["params"] * (bpw(nxt) - bpw(bits)) / 8
                if spent + cost > budget_bytes:
                    break
                gain = (now - predicted_damage(t, nxt, probe_bits)) / cost
                if gain > best_gain:
                    best, best_gain = (name, nxt, cost), gain
        if best is None:
            return assignment
        assignment[best[0]] = best[1]
        spent += best[2]


# -- applying and restoring -------------------------------------------------------


@torch.inference_mode()
def apply_assignment(targets: dict[str, torch.nn.Parameter], assignment: dict[str, int]) -> None:
    for name, bits in assignment.items():
        targets[name].data.copy_(fake_quant(targets[name].data, bits))


@torch.inference_mode()
def restore(lm: LoadedModel, targets: dict[str, torch.nn.Parameter]) -> None:
    """Put the original weights back by re-reading them from disk (no spare copy in memory)."""
    remaining = set(targets)
    for f in sorted(Path(lm.path).glob("*.safetensors")):
        with safe_open(f, framework="pt") as st:
            for name in remaining & set(st.keys()):
                p = targets[name]
                p.data.copy_(st.get_tensor(name).to(device=p.device, dtype=p.dtype))
                remaining.discard(name)
    if remaining:
        raise RuntimeError(
            f"Could not restore {len(remaining)} tensors from disk: {sorted(remaining)[:3]}"
        )


# -- the run ----------------------------------------------------------------------


def run_quant(
    lm: LoadedModel,
    calibration: list[torch.Tensor],
    heldout: list[torch.Tensor],
    probe_bits: int = 4,
    targets_bits: tuple[int, ...] = (3, 4),
    progress=lambda *a, **k: None,
    extra_probe_bits: tuple[int, ...] = (3,),
) -> dict[str, Any]:
    """`probe_bits` is the headline one-tensor test. `extra_probe_bits` are also
    measured, so the planner does not have to extrapolate to those widths."""
    measured_bits = sorted({probe_bits, *extra_probe_bits})
    if any(b not in BITS for b in measured_bits):
        raise ValueError(f"probe bits must be among {BITS}")
    targets = quant_targets(lm)
    names = list(targets)

    progress("measuring importance")
    importance = input_importance(lm, calibration, targets)
    base = log_probs(lm, calibration)

    tensors = []
    for n, name in enumerate(names):
        if n % 10 == 0:
            progress("testing tensors", n, len(names))
        p = targets[name]
        row = {**describe(name), **weight_stats(p.data, importance.get(name))}
        original = p.data.clone()
        row["kl_by_bits"] = {}
        try:
            for bits in measured_bits:
                p.data.copy_(fake_quant(original, bits))
                measured = compare(lm, calibration, base)
                row["kl_by_bits"][str(bits)] = measured["kl"]
                if bits == probe_bits:
                    row.update(measured)
        finally:
            p.data.copy_(original)
        del original
        tensors.append(row)
    del base

    # Per-layer roll-up of the single-tensor sensitivities.
    layers: dict[int, dict[str, Any]] = {}
    for t in tensors:
        if t["layer"] is None:
            continue
        row = layers.setdefault(
            t["layer"], {"layer": t["layer"], "kl": 0.0, "attention_kl": 0.0, "mlp_kl": 0.0}
        )
        row["kl"] += t["kl"]
        row[f"{t['component']}_kl"] = row.get(f"{t['component']}_kl", 0.0) + t["kl"]

    # Plans, each checked against a uniform width of the same size on held-out prompts.
    progress("checking plans")
    held_base = log_probs(lm, heldout)
    plans = []
    try:
        for n, bits in enumerate(targets_bits):
            progress(f"checking plan for {bits}-bit size", n, len(targets_bits))
            uniform = {name: bits for name in names}
            budget = plan_bytes(tensors, uniform)
            mixed = make_plan(tensors, budget, probe_bits)

            apply_assignment(targets, uniform)
            uniform_result = compare(lm, heldout, held_base)
            restore(lm, targets)
            apply_assignment(targets, mixed)
            mixed_result = compare(lm, heldout, held_base)
            restore(lm, targets)

            total = sum(t["params"] for t in tensors)
            mixed_bytes = plan_bytes(tensors, mixed)
            histogram = {str(b): 0 for b in BITS}
            for t in tensors:
                histogram[str(mixed[t["name"]])] += t["params"]
            plans.append(
                {
                    "target_bits": bits,
                    "budget_bytes": budget,
                    "uniform": {
                        "bits": bits,
                        "bytes": budget,
                        "avg_bpw": bits_per_weight(bits),
                        **uniform_result,
                    },
                    "mixed": {
                        "assignment": mixed,
                        "bytes": mixed_bytes,
                        "avg_bpw": mixed_bytes * 8 / total,
                        "predicted_kl": sum(
                            predicted_damage(t, mixed[t["name"]], probe_bits) for t in tensors
                        ),
                        "params_by_bits": histogram,
                        **mixed_result,
                    },
                }
            )
    finally:
        restore(lm, targets)

    return {
        "settings": {
            "probe_bits": probe_bits,
            "measured_bits": measured_bits,
            "bits_options": list(BITS),
            "block": BLOCK,
            "overhead_bits": OVERHEAD_BITS,
            "calibration_prompts": len(calibration),
            "heldout_prompts": len(heldout),
            "simulated": True,
        },
        "full_precision_bytes": sum(t["params"] for t in tensors) * FULL_BITS / 8,
        "tensors": tensors,
        "layers": [layers[i] for i in sorted(layers)],
        "plans": plans,
    }


def layer_stats(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for layer in result["layers"]:
        for component, key in (("block", "kl"), ("attn", "attention_kl"), ("mlp", "mlp_kl")):
            rows.append(
                {"context": None, "phase": f"quant{result['settings']['probe_bits']}",
                 "layer": layer["layer"], "component": component, "metric": "quant_kl",
                 "value": layer[key]}
            )  # fmt: skip
    return rows


def split_prompts(sets: list[str], calibration: int, heldout: int) -> tuple[list[str], list[str]]:
    """Calibration prompts from the front of each set, held-out ones from the back."""
    loaded = [load_prompt_set(s).prompts for s in sets]
    per_c = max(1, calibration // len(loaded))
    per_h = max(1, heldout // len(loaded))
    calib = [p for prompts in loaded for p in prompts[:per_c]]
    held = [p for prompts in loaded for p in prompts[::-1][:per_h]]
    overlap = set(calib) & set(held)
    if overlap:
        raise ValueError(
            "Prompt sets are too small to keep calibration and held-out prompts apart."
        )
    return calib, held


def quant_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    """Worker entry point for the `quant` job."""
    seed = params.get("seed", 0)
    torch.manual_seed(seed)
    sets = params.get("sets") or ["coding", "chat", "math", "instruction"]
    calib_text, held_text = split_prompts(
        sets, params.get("calibration_prompts", 16), params.get("heldout_prompts", 32)
    )

    ctx.progress("checking memory")
    anatomy = inspect_model(params["model"])
    device = pick_device(params.get("device", "auto"))
    dtype = getattr(torch, params["dtype"]) if params.get("dtype") else default_dtype(device)
    guard_memory(
        anatomy["totals"]["params"],
        torch.empty((), dtype=dtype).element_size(),
        system_info(),
        device.type,
        force=params.get("force", False),
    )

    ctx.progress("loading model")
    has_tokenizer = (Path(anatomy["path"]) / "tokenizer_config.json").exists()
    lm = load_model(anatomy["path"], str(device), dtype, with_tokenizer=has_tokenizer)
    result = run_quant(
        lm,
        encode_prompts(lm, calib_text),
        encode_prompts(lm, held_text),
        probe_bits=params.get("probe_bits", 4),
        extra_probe_bits=tuple(params.get("extra_probe_bits", (3,))),
        targets_bits=tuple(params.get("target_bits", (4, 5))),
        progress=ctx.progress,
    )
    arch = anatomy["architecture"]
    return {
        "manifest": build_manifest(
            model_dir=lm.path,
            run_type="quant",
            device=lm.device,
            dtype=lm.dtype,
            seed=seed,
            params=params,
        ),  # fmt: skip
        "model": {
            "fingerprint": model_fingerprint(lm.path),
            "path": str(lm.path),
            "arch": arch["model_type"],
            "n_layers": arch["n_layers"],
            "params": anatomy["totals"]["params"],
        },
        "result": result,
        "layer_stats": layer_stats(result),
    }
