"""Edit commands: small, reviewable changes to a model's weights.

A patch is an ordered list of text commands. The base model on disk is never
modified: a patch is applied to a loaded copy, and "reverting" is simply
loading the base again. Commands:

  zero_head layer=13 head=4
      Switch off one attention head.
  scale_neuron layer=12 neuron=2031 factor=0
      Scale one MLP neuron's output (0 switches it off).
  ablate_direction vector=/path/to/direction.npy from_layer=2
      Stop every block from `from_layer` on from writing along a direction.
  drop_layers layers=10,16
      Remove whole blocks. The model gets smaller and faster.
  prune_neurons keep=0.9
      Keep the most important share of MLP neurons in every layer (the same
      number per layer, so the result is still a standard model). Importance
      is a neuron's average activation size times the size of its output
      weights, measured on calibration prompts.

Everything is done on the weights themselves, so an edited model can be saved
and converted like any other.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from xray.models.adapters import LoadedModel, get_adapter
from xray.patch.commands import COMMANDS, PatchError, parse_command, parse_commands  # noqa: F401
from xray.probes.hooks import ActivationProbe


def _layer(lm: LoadedModel, index: int, op: str):
    if not 0 <= index < len(lm.refs.layers):
        raise PatchError(f"'{op}': layer {index} does not exist (0 to {len(lm.refs.layers) - 1}).")
    return lm.refs.layers[index]


def _refresh(lm: LoadedModel) -> None:
    lm.refs = get_adapter(lm.model.config.model_type).refs(lm.model)


@torch.no_grad()
def neuron_importance(lm: LoadedModel, encoded: list[torch.Tensor]) -> torch.Tensor:
    """[layers, neurons]: average |activation| times the norm of the neuron's output weights."""
    with ActivationProbe(lm.refs) as probe:
        for ids in encoded:
            lm.model(input_ids=ids.to(lm.device))
        arr = probe.arrays("all")
    mean_abs = torch.from_numpy(arr["abs_sum"] / max(int(arr["tokens"]), 1))
    norms = torch.stack(
        [layer.mlp_out.weight.float().norm(dim=0).cpu() for layer in lm.refs.layers]
    )
    return mean_abs * norms


def _resize(linear: nn.Linear, rows: torch.Tensor | None, cols: torch.Tensor | None) -> nn.Linear:
    """A smaller copy of `linear` keeping the given output rows and/or input columns."""
    w = linear.weight.data
    if rows is not None:
        w = w[rows]
    if cols is not None:
        w = w[:, cols]
    new = nn.Linear(
        w.shape[1], w.shape[0], bias=linear.bias is not None, device=w.device, dtype=w.dtype
    )
    new.weight.data.copy_(w)
    if linear.bias is not None:
        new.bias.data.copy_(linear.bias.data[rows] if rows is not None else linear.bias.data)
    return new


@torch.no_grad()
def apply_commands(
    lm: LoadedModel, commands: list[dict[str, Any]], calibration: list[torch.Tensor] | None = None
) -> list[str]:
    """Apply a patch to a loaded model in place.

    Returns one line per command saying what changed.
    """
    log = []
    n_heads = lm.model.config.num_attention_heads
    for c in commands:
        op = c["op"]
        if op == "zero_head":
            layer = _layer(lm, c["layer"], op)
            if not 0 <= c["head"] < n_heads:
                raise PatchError(f"'{op}': head {c['head']} does not exist (0 to {n_heads - 1}).")
            width = layer.attn_out.in_features // n_heads
            # The head's output enters through these input columns of the output projection.
            layer.attn_out.weight.data[:, c["head"] * width : (c["head"] + 1) * width] = 0
            log.append(f"Switched off head {c['head']} in layer {c['layer']}.")

        elif op == "scale_neuron":
            layer = _layer(lm, c["layer"], op)
            if not 0 <= c["neuron"] < layer.mlp_out.in_features:
                raise PatchError(
                    f"'{op}': neuron {c['neuron']} does not exist in layer {c['layer']}."
                )
            layer.mlp_out.weight.data[:, c["neuron"]] *= c["factor"]
            log.append(f"Scaled neuron {c['neuron']} in layer {c['layer']} by {c['factor']:g}.")

        elif op == "ablate_direction":
            _layer(lm, c["from_layer"], op)
            d = torch.from_numpy(np.load(c["vector"])).float()
            width = lm.refs.layers[0].mlp_out.out_features
            if d.shape != (width,):
                raise PatchError(
                    f"'{op}': the vector has shape {tuple(d.shape)}, expected ({width},)."
                )
            d = (d / d.norm()).to(lm.device)
            for layer in lm.refs.layers[c["from_layer"] :]:
                for linear in (layer.attn_out, layer.mlp_out):
                    w = linear.weight.data.float()
                    linear.weight.data.copy_((w - torch.outer(d, d @ w)).to(linear.weight.dtype))
                    if linear.bias is not None:
                        b = linear.bias.data.float()
                        linear.bias.data.copy_((b - d * (d @ b)).to(linear.bias.dtype))
            log.append(f"Removed one direction from what layers {c['from_layer']} and up write.")

        elif op == "drop_layers":
            for i in c["layers"]:
                _layer(lm, i, op)
            if len(c["layers"]) >= len(lm.refs.layers):
                raise PatchError(f"'{op}': cannot remove every layer.")
            base = lm.model.model
            kept = [block for i, block in enumerate(base.layers) if i not in c["layers"]]
            for i, block in enumerate(kept):
                # The key/value cache is indexed by this number.
                if hasattr(block.self_attn, "layer_idx"):
                    block.self_attn.layer_idx = i
            base.layers = nn.ModuleList(kept)
            config = lm.model.config
            n_before = len(kept) + len(c["layers"])
            # Some configs carry one entry per layer (e.g. Qwen's `layer_types`).
            for key, value in list(vars(config).items()):
                if isinstance(value, list) and len(value) == n_before and key != "eos_token_id":
                    setattr(config, key, [v for i, v in enumerate(value) if i not in c["layers"]])
            config.num_hidden_layers = len(kept)
            _refresh(lm)
            log.append(f"Removed layers {', '.join(map(str, c['layers']))}; {len(kept)} remain.")

        elif op == "prune_neurons":
            if calibration is None:
                raise PatchError(f"'{op}' needs calibration prompts to rank neurons.")
            width = lm.refs.layers[0].mlp_out.in_features
            keep = max(1, round(width * c["keep"]))
            importance = neuron_importance(lm, calibration)
            for layer, scores in zip(lm.refs.layers, importance, strict=True):
                idx = scores.topk(keep).indices.sort().values.to(lm.device)
                mlp = layer.mlp
                mlp.gate_proj = _resize(mlp.gate_proj, idx, None)
                mlp.up_proj = _resize(mlp.up_proj, idx, None)
                mlp.down_proj = _resize(mlp.down_proj, None, idx)
            lm.model.config.intermediate_size = keep
            _refresh(lm)
            log.append(f"Kept {keep} of {width} MLP neurons in every layer ({keep / width:.0%}).")
    return log
