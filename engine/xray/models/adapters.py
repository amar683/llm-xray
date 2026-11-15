"""Model adapters: map an architecture's module names to a common shape.

Analyzers never touch `model.model.layers[i].mlp.down_proj` directly; they ask
the adapter. Supporting a new model family means adding one adapter here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from xray.core.device import default_dtype, pick_device
from xray.models.inspect import UnsupportedModel, require_compatible, resolve_model_dir


@dataclass(frozen=True)
class LayerRefs:
    """The parts of one transformer block that analyzers hook into."""

    index: int
    block: nn.Module  # whole decoder layer
    attn: nn.Module  # self-attention sub-block
    mlp: nn.Module  # feed-forward sub-block
    mlp_out: nn.Module  # final linear of the MLP; its *input* is the neuron activations
    attn_out: nn.Module  # final linear of attention; its *input* is the per-head outputs


@dataclass(frozen=True)
class ModelRefs:
    embed: nn.Module
    layers: list[LayerRefs]
    final_norm: nn.Module
    lm_head: nn.Module


class LlamaLikeAdapter:
    """Llama-style decoder: `model.layers[i].{self_attn, mlp.{gate,up,down}_proj}`.

    Llama 3.x and Qwen 2.x/3.x share this layout in Hugging Face transformers.
    """

    model_types = frozenset({"llama", "qwen2", "qwen3"})

    def refs(self, model: nn.Module) -> ModelRefs:
        base = model.model
        layers = [
            LayerRefs(
                index=i,
                block=block,
                attn=block.self_attn,
                mlp=block.mlp,
                mlp_out=block.mlp.down_proj,
                attn_out=block.self_attn.o_proj,
            )
            for i, block in enumerate(base.layers)
        ]
        return ModelRefs(
            embed=base.embed_tokens,
            layers=layers,
            final_norm=base.norm,
            lm_head=model.lm_head,
        )


ADAPTERS = [LlamaLikeAdapter()]


def get_adapter(model_type: str) -> LlamaLikeAdapter:
    for adapter in ADAPTERS:
        if model_type in adapter.model_types:
            return adapter
    supported = sorted(t for a in ADAPTERS for t in a.model_types)
    raise ValueError(f"Unsupported model type '{model_type}'. Supported: {', '.join(supported)}")


@dataclass
class LoadedModel:
    model: nn.Module
    tokenizer: object | None
    refs: ModelRefs
    device: torch.device
    dtype: torch.dtype
    path: Path


def load_model(
    model: str,
    device: str = "auto",
    dtype: torch.dtype | None = None,
    with_tokenizer: bool = True,
    eager_attention: bool = False,
) -> LoadedModel:
    """Load a local model for analysis. Never downloads.

    Pass `eager_attention=True` when the analysis needs attention weights.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_dir(model)
    require_compatible(path)  # header check: refuse before spending time and memory
    dev = pick_device(device)
    dt = dtype or default_dtype(dev)
    # Attention weights are only returned by the slower "eager" implementation.
    extra = {"attn_implementation": "eager"} if eager_attention else {}
    hf_model, info = AutoModelForCausalLM.from_pretrained(
        path, dtype=dt, local_files_only=True, output_loading_info=True, **extra
    )
    if info["missing_keys"]:
        # Belt and braces: the header check should already have caught this.
        missing = sorted(info["missing_keys"])
        raise UnsupportedModel(
            f"{path.name} cannot be analysed: {len(missing)} weights were not in its files "
            f"and would be random ({', '.join(missing[:3])})."
        )
    hf_model.to(dev).eval()
    tokenizer = (
        AutoTokenizer.from_pretrained(path, local_files_only=True) if with_tokenizer else None
    )
    refs = get_adapter(hf_model.config.model_type).refs(hf_model)
    return LoadedModel(hf_model, tokenizer, refs, dev, dt, path)
