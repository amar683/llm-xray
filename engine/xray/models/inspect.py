"""Model anatomy without loading weights.

Reads config.json and the safetensors headers only, so inspecting a
multi-gigabyte model takes milliseconds and no memory.
"""

from __future__ import annotations

import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}  # fmt: skip

# Approximate bits per weight for common GGUF quantisation types.
QUANT_BITS = {
    "F16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.69,
    "Q4_K_M": 4.85,
    "Q3_K_M": 3.91,
}

_LAYER_RE = re.compile(r"layers\.(\d+)\.(.+)")


def resolve_model_dir(model: str) -> Path:
    """Accept a local folder or a Hugging Face repo id already in the local cache.

    Never downloads: a repo id that is not cached raises FileNotFoundError.
    """
    p = Path(model).expanduser()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(model, local_files_only=True))
    except (LocalEntryNotFoundError, OSError, ValueError) as e:
        raise FileNotFoundError(
            f"'{model}' is not a local folder and is not in the local Hugging Face cache."
        ) from e


def read_safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    """Tensor name -> {dtype, shape, data_offsets} for one .safetensors file."""
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def classify_tensor(name: str) -> tuple[int | None, str]:
    """Map a tensor name to (layer index or None, component)."""
    m = _LAYER_RE.search(name)
    if m:
        rest = m.group(2)
        if rest.startswith("self_attn"):
            return int(m.group(1)), "attention"
        if rest.startswith("mlp"):
            return int(m.group(1)), "mlp"
        if "norm" in rest:
            return int(m.group(1)), "norm"
        return int(m.group(1)), "other"
    if "embed_tokens" in name:
        return None, "embeddings"
    if "lm_head" in name:
        return None, "lm_head"
    if "norm" in name:
        return None, "final_norm"
    return None, "other"


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def inspect_model(model: str) -> dict[str, Any]:
    """Anatomy report: architecture, parameter/byte breakdown, memory estimates."""
    model_dir = resolve_model_dir(model)
    config = json.loads((model_dir / "config.json").read_text())

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors weight files in {model_dir}")

    per_layer: dict[int, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"params": 0, "bytes": 0})
    )
    globals_: dict[str, dict[str, int]] = defaultdict(lambda: {"params": 0, "bytes": 0})
    dtypes: dict[str, int] = defaultdict(int)
    n_tensors = 0

    for f in files:
        for name, meta in read_safetensors_header(f).items():
            params = _numel(meta["shape"])
            nbytes = params * DTYPE_BYTES.get(meta["dtype"], 0)
            layer, comp = classify_tensor(name)
            bucket = per_layer[layer][comp] if layer is not None else globals_[comp]
            bucket["params"] += params
            bucket["bytes"] += nbytes
            dtypes[meta["dtype"]] += params
            n_tensors += 1

    layers = [
        {
            "index": i,
            "components": dict(per_layer[i]),
            "params": sum(c["params"] for c in per_layer[i].values()),
            "bytes": sum(c["bytes"] for c in per_layer[i].values()),
        }
        for i in sorted(per_layer)
    ]
    total_params = sum(layer["params"] for layer in layers) + sum(
        g["params"] for g in globals_.values()
    )
    total_bytes = sum(layer["bytes"] for layer in layers) + sum(
        g["bytes"] for g in globals_.values()
    )

    by_component: dict[str, int] = defaultdict(int)
    for layer in layers:
        for comp, v in layer["components"].items():
            by_component[comp] += v["params"]
    for comp, v in globals_.items():
        by_component[comp] += v["params"]

    hidden = config.get("hidden_size")
    n_heads = config.get("num_attention_heads")
    n_kv_heads = config.get("num_key_value_heads", n_heads)
    head_dim = config.get("head_dim") or (hidden // n_heads if hidden and n_heads else None)

    arch = {
        "model_type": config.get("model_type"),
        "n_layers": config.get("num_hidden_layers", len(layers)),
        "hidden_size": hidden,
        "mlp_size": config.get("intermediate_size"),
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "vocab_size": config.get("vocab_size"),
        "max_context": config.get("max_position_embeddings"),
        "activation": config.get("hidden_act"),
        "tied_embeddings": bool(config.get("tie_word_embeddings", False)),
    }

    return {
        "path": str(model_dir),
        "architecture": arch,
        "totals": {
            "params": total_params,
            "bytes_on_disk": total_bytes,
            "tensors": n_tensors,
            "weight_files": len(files),
            "stored_dtypes": dict(dtypes),
        },
        "by_component": dict(by_component),
        "globals": dict(globals_),
        "layers": layers,
        "memory_estimates": memory_estimates(total_params, arch),
    }


def kv_cache_bytes(arch: dict[str, Any], context: int, bytes_per_value: int = 2) -> int | None:
    """Keys + values for every layer at a given context length (fp16 by default)."""
    if not all(arch.get(k) for k in ("n_layers", "n_kv_heads", "head_dim")):
        return None
    return 2 * arch["n_layers"] * arch["n_kv_heads"] * arch["head_dim"] * context * bytes_per_value


def memory_estimates(total_params: int, arch: dict[str, Any]) -> dict[str, Any]:
    """Approximate weight size per quantisation level and KV cache per context length."""
    weights = {q: int(total_params * bits / 8) for q, bits in QUANT_BITS.items()}
    kv = {}
    for ctx in (2048, 4096, 8192, 32768):
        if arch.get("max_context") and ctx > arch["max_context"]:
            continue
        b = kv_cache_bytes(arch, ctx)
        if b is not None:
            kv[str(ctx)] = b
    return {"weights_bytes": weights, "kv_cache_bytes_fp16": kv}
