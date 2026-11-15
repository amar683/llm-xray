"""Probes: PyTorch hooks that measure a model while it runs.

Each probe attaches hooks, accumulates *running* statistics on the device
(never storing raw activations), and removes its hooks on exit.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from xray.core.device import sync
from xray.models.adapters import ModelRefs


class Probe:
    """Base class: context manager that guarantees hooks are removed."""

    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


class TimingProbe(Probe):
    """Wall-clock time per module, with a switchable measuring mode.

    Hooks are attached once to every named module. Before each forward pass the
    caller picks which modules to time and how:

    - `synced=True`: the device is synchronised just before and just after the
      module, so the sample is the module's full cost (CPU dispatch plus device
      execution). Only time modules that do not contain each other in the same
      pass: a sync inside a module inflates the module around it.
    - `synced=False`: no synchronisation. The sample is how long the call took
      to return, i.e. CPU-side dispatch only. Nesting is fine.

    With nothing selected the hooks do nothing, which gives the reference speed.
    """

    def __init__(self, modules: dict[str, nn.Module], device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.active: frozenset[str] = frozenset()
        self.synced = False
        self.sink: dict[str, list[float]] | None = None
        self._t0: dict[str, float] = {}
        for name, module in modules.items():
            self._attach(name, module)

    def measure(self, names, synced: bool, sink: dict[str, list[float]]) -> None:
        """Time `names` during the next passes, appending seconds to `sink[name]`."""
        self.active, self.synced, self.sink = frozenset(names), synced, sink

    def off(self) -> None:
        self.active, self.sink = frozenset(), None

    def _attach(self, name: str, module: nn.Module) -> None:
        def pre(_module, _args):
            if name in self.active:
                if self.synced:
                    sync(self.device)
                self._t0[name] = time.perf_counter()

        def post(_module, _args, _output):
            if name in self.active:
                if self.synced:
                    sync(self.device)
                self.sink[name].append(time.perf_counter() - self._t0[name])

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))


# |activation| thresholds for the per-neuron histogram. For every neuron the
# probe counts how many tokens reached each threshold, so firing rates at any of
# these levels, and a coarse distribution, can be read back later.
EDGES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


class ActivationProbe(Probe):
    """Streaming statistics for MLP neurons and the residual stream.

    MLP neurons: the input to each block's final MLP linear is the per-neuron
    activation. For every neuron this keeps running sums (value, |value|,
    value squared), the maximum |value|, and how many tokens reached each
    threshold in `EDGES`.

    Residual stream: for each block, the cosine similarity between its input
    and output (near 1.0 means the block barely changes anything, a sign of
    redundancy) and the mean output norm.

    Memory is fixed: nothing grows with the number of prompts. Statistics are
    kept per *group* (for example one per prompt set); call `begin(name)` to
    choose which group the following passes count towards.

    Set `skip_first` for passes that start a sequence: the first token carries
    an outsized norm in most models and would dominate every average.
    """

    def __init__(self, refs: ModelRefs, edges: tuple[float, ...] = EDGES) -> None:
        super().__init__()
        self.edges = edges
        self.skip_first = True
        self.n_layers = len(refs.layers)
        self.groups: dict[str, dict[str, Any]] = {}
        self._group: dict[str, Any] | None = None
        self._block_in: dict[int, torch.Tensor] = {}
        self.begin("all")

        for layer in refs.layers:
            i = layer.index
            self._handles.append(layer.mlp_out.register_forward_pre_hook(self._neuron_hook(i)))
            self._handles.append(
                layer.block.register_forward_pre_hook(self._block_pre(i), with_kwargs=True)
            )
            self._handles.append(layer.block.register_forward_hook(self._block_post(i)))

    def begin(self, group: str) -> None:
        """Count the following forward passes towards `group`."""
        self._group = self.groups.setdefault(
            group,
            {
                "neuron": {},
                "block": {
                    i: {"cos_sum": 0.0, "norm_sum": 0.0, "tokens": 0} for i in range(self.n_layers)
                },
            },
        )

    def _neuron_hook(self, i: int):
        def hook(_module, args):
            x = args[0].detach()
            if self.skip_first:
                x = x[:, 1:]
            if x.shape[1] == 0:
                return
            v = x.float().reshape(-1, x.shape[-1])  # (tokens, neurons)
            a = v.abs()
            s = self._group["neuron"].get(i)
            if s is None:
                n, dev = v.shape[-1], v.device
                s = self._group["neuron"][i] = {
                    "sum": torch.zeros(n, device=dev),
                    "abs_sum": torch.zeros(n, device=dev),
                    "sq_sum": torch.zeros(n, device=dev),
                    "abs_max": torch.zeros(n, device=dev),
                    "ge": torch.zeros(len(self.edges), n, device=dev),
                    "tokens": 0,
                }
            s["sum"] += v.sum(0)
            s["abs_sum"] += a.sum(0)
            s["sq_sum"] += (v * v).sum(0)
            s["abs_max"] = torch.maximum(s["abs_max"], a.max(0).values)
            s["ge"] += torch.stack([(a >= e).sum(0) for e in self.edges])
            s["tokens"] += v.shape[0]

        return hook

    def _block_pre(self, i: int):
        def hook(_module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            self._block_in[i] = h.detach()

        return hook

    def _block_post(self, i: int):
        def hook(_module, _args, output):
            h_out = (output[0] if isinstance(output, tuple) else output).detach()
            h_in = self._block_in.pop(i)
            if self.skip_first:
                h_in, h_out = h_in[:, 1:], h_out[:, 1:]
            if h_out.shape[1] == 0:
                return
            a, b = h_in.float(), h_out.float()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
            s = self._group["block"][i]
            s["cos_sum"] += cos.sum().item()
            s["norm_sum"] += b.norm(dim=-1).sum().item()
            s["tokens"] += cos.numel()

        return hook

    def arrays(self, group: str) -> dict[str, np.ndarray]:
        """This group's statistics as numpy arrays: [layers, neurons] unless noted."""
        g = self.groups[group]
        layers = sorted(g["neuron"])
        if not layers:
            raise ValueError(f"No activations were recorded for group '{group}'.")

        def stack(key: str) -> np.ndarray:
            return torch.stack([g["neuron"][i][key] for i in layers]).cpu().numpy()

        block = [g["block"][i] for i in layers]
        return {
            "tokens": np.array(g["neuron"][layers[0]]["tokens"], dtype=np.int64),
            "sum": stack("sum"),
            "abs_sum": stack("abs_sum"),
            "sq_sum": stack("sq_sum"),
            "abs_max": stack("abs_max"),
            "ge": stack("ge"),  # [layers, edges, neurons]
            "block_cos_sum": np.array([b["cos_sum"] for b in block]),
            "block_norm_sum": np.array([b["norm_sum"] for b in block]),
            "block_tokens": np.array([b["tokens"] for b in block], dtype=np.int64),
        }


class SkipProbe(Probe):
    """Makes chosen blocks pass their input straight through, as if removed.

    Set `layers` to the block indices to skip before each forward pass.
    """

    def __init__(self, refs: ModelRefs) -> None:
        super().__init__()
        self.layers: frozenset[int] = frozenset()
        self._inputs: dict[int, torch.Tensor] = {}
        for layer in refs.layers:
            i = layer.index
            self._handles.append(
                layer.block.register_forward_pre_hook(self._pre(i), with_kwargs=True)
            )
            self._handles.append(layer.block.register_forward_hook(self._post(i)))

    def _pre(self, i: int):
        def hook(_module, args, kwargs):
            if i in self.layers:
                self._inputs[i] = args[0] if args else kwargs["hidden_states"]

        return hook

    def _post(self, i: int):
        def hook(_module, _args, output):
            if i not in self.layers:
                return None
            h = self._inputs.pop(i)
            return (h, *output[1:]) if isinstance(output, tuple) else h

        return hook
