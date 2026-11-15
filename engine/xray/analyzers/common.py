"""Helpers shared by the interpretability analyzers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from xray.core.device import default_dtype, pick_device, system_info
from xray.core.manifest import build_manifest, model_fingerprint
from xray.core.memory import guard_memory
from xray.models.adapters import LoadedModel, load_model
from xray.models.inspect import inspect_model


def eos_ids(lm: LoadedModel) -> set[int]:
    eos = getattr(lm.model.config, "eos_token_id", None)
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, list | tuple) else {eos}


@torch.inference_mode()
def greedy_generate(lm: LoadedModel, ids: torch.Tensor, new_tokens: int) -> tuple[list[int], bool]:
    """Greedy decoding. Returns (generated token ids, whether the model ended by itself)."""
    eos = eos_ids(lm)
    out = lm.model(input_ids=ids.to(lm.device), use_cache=True)
    tokens: list[int] = []
    for _ in range(new_tokens):
        if not torch.isfinite(out.logits[:, -1]).all():
            raise FloatingPointError(
                f"The model produced invalid numbers as {lm.dtype}. Try another dtype."
            )
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        if nxt.item() in eos:
            return tokens, True
        tokens.append(nxt.item())
        out = lm.model(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
    return tokens, False


def token_text(lm: LoadedModel, ids: list[int]) -> list[str]:
    """One readable string per token id."""
    if lm.tokenizer is None:
        return [str(i) for i in ids]
    return [lm.tokenizer.decode([i]) for i in ids]


def norm_scale(norm: torch.nn.Module, h: torch.Tensor) -> torch.Tensor:
    """The per-dimension factor an RMS norm applies to `h` (weight / rms).

    Freezing this factor makes the final norm linear, so a logit can be split
    exactly into one contribution per component of the residual stream.
    """
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6))
    rms = (h.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return norm.weight.float() / rms


@torch.inference_mode()
def residual_states(lm: LoadedModel, ids: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
    """The residual stream after every block, and the final logits.

    Read with hooks rather than `output_hidden_states`, whose last entry has
    already been through the final norm.
    """
    states: list[torch.Tensor] = []
    handles = [
        layer.block.register_forward_hook(
            lambda _m, _a, out: states.append((out[0] if isinstance(out, tuple) else out)[0])
        )
        for layer in lm.refs.layers
    ]
    try:
        logits = lm.model(input_ids=ids.to(lm.device)).logits[0]
    finally:
        for h in handles:
            h.remove()
    return states, logits


def load_for_job(
    params: dict[str, Any], ctx, eager_attention: bool = False
) -> tuple[LoadedModel, dict[str, Any]]:
    """Memory check, then load the model a job asked for. Returns (model, anatomy)."""
    torch.manual_seed(params.get("seed", 0))
    ctx.progress("checking memory")
    anatomy = inspect_model(params["model"])
    device = pick_device(params.get("device", "auto"))
    dtype = getattr(torch, params["dtype"]) if params.get("dtype") else default_dtype(device)
    if eager_attention and not params.get("dtype") and dtype == torch.float16:
        # Eager attention overflows to NaN in float16 on Apple GPUs; bfloat16 does not.
        dtype = torch.bfloat16
    guard_memory(
        anatomy["totals"]["params"],
        torch.empty((), dtype=dtype).element_size(),
        system_info(),
        device.type,
        force=params.get("force", False),
    )
    ctx.progress("loading model")
    has_tokenizer = (Path(anatomy["path"]) / "tokenizer_config.json").exists()
    lm = load_model(
        anatomy["path"], str(device), dtype,
        with_tokenizer=has_tokenizer, eager_attention=eager_attention,
    )  # fmt: skip
    return lm, anatomy


def job_output(
    lm: LoadedModel, anatomy: dict[str, Any], run_type: str, params: dict[str, Any], result: dict
) -> dict[str, Any]:
    arch = anatomy["architecture"]
    return {
        "manifest": build_manifest(
            model_dir=lm.path,
            run_type=run_type,
            device=lm.device,
            dtype=lm.dtype,
            seed=params.get("seed", 0),
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
    }
