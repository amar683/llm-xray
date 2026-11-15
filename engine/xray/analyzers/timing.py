"""Per-layer timing profiler.

Per-layer time on a GPU is not a single number. GPU backends are asynchronous:
the CPU queues work for layer k+1 while the device is still executing layer k.
So this profiler runs four kinds of forward pass and reports each honestly:

  wall        no timers; the real speed, and the total everything must add up to
  dispatch    unsynchronised timers; CPU-side cost of calling each module
  blocks      synchronised timers on embed, every block, final norm and lm_head
  components  synchronised timers on every block's attention and MLP

`blocks` and `components` are separate passes because a synchronisation inside
a block would inflate the block's own measurement.

A module's `share` is its synchronised time divided by the sum over all
top-level modules, and `est_ms` is that share of the real wall time. Shares are
therefore comparable between layers and add up to the true total.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from xray.core.bench import memory_bandwidth_gbps
from xray.core.device import default_dtype, now, pick_device, system_info
from xray.core.manifest import build_manifest, model_fingerprint
from xray.core.memory import guard_memory
from xray.models.adapters import LoadedModel, load_model
from xray.models.inspect import inspect_model
from xray.probes.hooks import TimingProbe

# Filler text for building prompts of an exact length. Content does not affect timing.
FILLER = (
    "Write a Python function that checks whether a string is a palindrome. "
    "Explain what a hash map is and when you would use one. "
    "What is the difference between a process and a thread? "
    "A train travels 120 km in 1.5 hours. What is its average speed? "
)

KINDS = ("wall", "dispatch", "blocks", "components")
TOP_OTHER = ("embed", "final_norm", "lm_head")
MIN_PREFILL_SECONDS = 1.0
MAX_PREFILL_REPEATS = 30


def _median_ms(xs: list[float]) -> float:
    return statistics.median(xs) * 1e3 if xs else 0.0


def _p95_ms(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, round(0.95 * (len(s) - 1)))] * 1e3


def module_sets(lm: LoadedModel) -> tuple[dict[str, torch.nn.Module], list[str], list[str]]:
    """All timed modules, the top-level partition, and the per-block components."""
    refs = lm.refs
    modules = {"embed": refs.embed, "final_norm": refs.final_norm, "lm_head": refs.lm_head}
    partition, components = ["embed"], []
    for layer in refs.layers:
        i = layer.index
        modules[f"layer.{i}"] = layer.block
        modules[f"layer.{i}.attn"] = layer.attn
        modules[f"layer.{i}.mlp"] = layer.mlp
        partition.append(f"layer.{i}")
        components += [f"layer.{i}.attn", f"layer.{i}.mlp"]
    partition += ["final_norm", "lm_head"]
    return modules, partition, components


def context_ids(lm: LoadedModel, length: int, seed: int = 0) -> torch.Tensor:
    """A prompt of exactly `length` tokens. Content does not affect timing."""
    if lm.tokenizer is None:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, lm.model.config.vocab_size, (1, length), generator=g)
    ids = lm.tokenizer(FILLER, add_special_tokens=False).input_ids
    ids = (ids * (length // len(ids) + 1))[:length]
    return torch.tensor([ids])


class _Phase:
    """Samples collected for one phase (prefill or decode) at one context length."""

    def __init__(self) -> None:
        self.wall: dict[str, list[float]] = defaultdict(list)
        self.dispatch: dict[str, list[float]] = defaultdict(list)
        self.isolated: dict[str, list[float]] = defaultdict(list)


def _forward(lm, probe, sets, kind, phase: _Phase, ids, past=None):
    _, partition, components = sets
    if kind == "wall":
        probe.off()
    elif kind == "dispatch":
        probe.measure(partition + components, synced=False, sink=phase.dispatch)
    elif kind == "blocks":
        probe.measure(partition, synced=True, sink=phase.isolated)
    else:
        probe.measure(components, synced=True, sink=phase.isolated)
    t0 = now(lm.device)
    out = lm.model(input_ids=ids, past_key_values=past, use_cache=True)
    phase.wall[kind].append(now(lm.device) - t0)
    probe.off()
    return out


def _summarise(phase: _Phase, lm: LoadedModel, sets, tokens_per_pass: int) -> dict[str, Any]:
    _, partition, _ = sets
    wall_ms = _median_ms(phase.wall["wall"])
    iso = {n: _median_ms(v) for n, v in phase.isolated.items()}
    disp = {n: _median_ms(v) for n, v in phase.dispatch.items()}
    iso_total = sum(iso[n] for n in partition) or 1.0

    def top(name: str) -> dict[str, float]:
        share = iso[name] / iso_total
        return {
            "share": share,
            "est_ms": share * wall_ms,
            "isolated_ms": iso[name],
            "isolated_p95_ms": _p95_ms(phase.isolated[name]),
            "dispatch_ms": disp[name],
        }

    layers = []
    totals = defaultdict(float)
    for layer in lm.refs.layers:
        name = f"layer.{layer.index}"
        row = {"layer": layer.index, **top(name)}
        attn, mlp = iso[f"{name}.attn"], iso[f"{name}.mlp"]
        # Attention and MLP were timed in a different pass from the block, so
        # scale them down if together they exceed it.
        scale = min(1.0, iso[name] / (attn + mlp)) if attn + mlp else 0.0
        per_ms = row["est_ms"] / iso[name] if iso[name] else 0.0
        row["attn_ms"] = attn * scale * per_ms
        row["mlp_ms"] = mlp * scale * per_ms
        row["other_ms"] = max(0.0, row["est_ms"] - row["attn_ms"] - row["mlp_ms"])
        layers.append(row)
        totals["attention"] += row["attn_ms"]
        totals["mlp"] += row["mlp_ms"]
        totals["layer_other"] += row["other_ms"]

    other = {name: top(name) for name in TOP_OTHER}
    for name in TOP_OTHER:
        totals[name] = other[name]["est_ms"]

    wall_s = wall_ms / 1e3
    return {
        "wall_ms": wall_ms,
        "wall_p95_ms": _p95_ms(phase.wall["wall"]),
        "tokens_per_s": tokens_per_pass / wall_s if wall_s else 0.0,
        # How much of the real time is the CPU calling into the framework.
        "dispatch_fraction": min(1.0, sum(disp[n] for n in partition) / wall_ms) if wall_ms else 0,
        # Extra wall time each instrumented pass costs compared with the plain one.
        "probe_overhead_pct": {
            k: (_median_ms(phase.wall[k]) / wall_ms - 1) * 100 if wall_ms else 0.0
            for k in KINDS[1:]
        },
        "shares": {k: v / wall_ms if wall_ms else 0.0 for k, v in totals.items()},
        "layers": layers,
        "other": other,
    }


@torch.inference_mode()
def run_timing(
    lm: LoadedModel,
    contexts: list[int],
    prefill_repeats: int = 5,
    decode_repeats: int = 24,
    progress=lambda stage, done=None, total=None: None,
) -> dict[str, Any]:
    """Time every block at each context length. Returns a JSON-serialisable result."""
    sets = module_sets(lm)
    max_ctx = getattr(lm.model.config, "max_position_embeddings", None)

    def decode_room(context: int) -> int:
        """Decode repeats that fit after the prompt (one step per pass kind each)."""
        if not max_ctx:
            return decode_repeats
        return min(decode_repeats, (max_ctx - context - 2) // len(KINDS))

    # A context is skipped only when not even two decode repeats fit after it.
    usable = [c for c in contexts if decode_room(c) >= 2]
    skipped = [c for c in contexts if c not in usable]
    results = []

    with TimingProbe(sets[0], lm.device) as probe:
        for n, length in enumerate(usable):
            progress(f"context {length}", n, len(usable))
            ids = context_ids(lm, length).to(lm.device)
            prefill, decode = _Phase(), _Phase()

            # Warm up at this exact shape: GPU backends compile kernels per shape.
            _forward(lm, probe, sets, "wall", _Phase(), ids)
            trial = _Phase()
            out = _forward(lm, probe, sets, "wall", trial, ids)
            # Short prefills are cheap and noisy, so repeat them more: aim for
            # about a second of measurement per pass kind, within sane limits.
            wanted = round(MIN_PREFILL_SECONDS / max(trial.wall["wall"][0], 1e-6))
            n_prefill = max(prefill_repeats, min(MAX_PREFILL_REPEATS, wanted))
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            _forward(lm, probe, sets, "wall", _Phase(), nxt, out.past_key_values)

            # Interleave the pass kinds so slow drift affects all of them equally.
            for _ in range(n_prefill):
                for kind in KINDS:
                    out = _forward(lm, probe, sets, kind, prefill, ids)
            if not torch.isfinite(out.logits[:, -1]).all():
                raise FloatingPointError(
                    f"Model produced non-finite logits with dtype {lm.dtype}. Try float32."
                )

            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            n_decode = decode_room(length)
            for _ in range(n_decode):
                for kind in KINDS:
                    out = _forward(lm, probe, sets, kind, decode, nxt, past)
                    past = out.past_key_values
                    nxt = out.logits[:, -1].argmax(-1, keepdim=True)

            results.append(
                {
                    "context": length,
                    "prefill_repeats": n_prefill,
                    "decode_repeats": n_decode,
                    "prefill": _summarise(prefill, lm, sets, length),
                    "decode": _summarise(decode, lm, sets, 1),
                }
            )
            del out, past
            if lm.device.type == "mps":
                torch.mps.empty_cache()

    progress("bandwidth", len(usable), len(usable))
    return {
        "contexts": results,
        "skipped_contexts": skipped,
        "bandwidth": _bandwidth(lm, results),
        "layer_devices": _layer_devices(lm),
        "settings": {"prefill_repeats": prefill_repeats, "decode_repeats": decode_repeats},
    }


def _weight_bytes(module: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _bandwidth(lm: LoadedModel, results: list[dict]) -> dict[str, Any]:
    """How far decoding is from the hardware's memory-bandwidth limit.

    Each generated token reads every block's weights and the output head once.
    """
    peak = memory_bandwidth_gbps(lm.device)
    read = sum(_weight_bytes(layer.block) for layer in lm.refs.layers)
    read += _weight_bytes(lm.refs.lm_head)
    floor_ms = read / (peak * 1e9) * 1e3
    actual = results[0]["decode"]["wall_ms"] if results else 0.0
    return {
        "peak_gb_per_s": peak,
        "weights_read_per_token_gb": read / 1e9,
        "decode_floor_ms": floor_ms,
        "decode_actual_ms": actual,
        "times_above_floor": actual / floor_ms if floor_ms else 0.0,
    }


def _layer_devices(lm: LoadedModel) -> dict[str, Any]:
    """Which device each block's weights live on. Blocks off the main device are slow."""
    devices = [str(next(layer.block.parameters()).device) for layer in lm.refs.layers]
    main = max(set(devices), key=devices.count)
    return {
        "main": main,
        "offloaded_layers": [i for i, d in enumerate(devices) if d != main],
    }


def layer_stats(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a timing result into rows for the `layer_stats` table."""
    rows = []
    for ctx in result["contexts"]:
        for phase in ("prefill", "decode"):
            base = {"context": ctx["context"], "phase": phase}
            for layer in ctx[phase]["layers"]:
                for component, key in (("block", "est_ms"), ("attn", "attn_ms"), ("mlp", "mlp_ms")):
                    rows.append(
                        {**base, "layer": layer["layer"], "component": component,
                         "metric": "est_ms", "value": layer[key]}
                    )  # fmt: skip
            for name, v in ctx[phase]["other"].items():
                rows.append(
                    {**base, "layer": None, "component": name,
                     "metric": "est_ms", "value": v["est_ms"]}
                )  # fmt: skip
    return rows


def timing_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    """Worker entry point for the `timing` job."""
    seed = params.get("seed", 0)
    torch.manual_seed(seed)

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
    result = run_timing(
        lm,
        contexts=params.get("contexts", [128, 512, 2048]),
        prefill_repeats=params.get("prefill_repeats", 5),
        decode_repeats=params.get("decode_repeats", 24),
        progress=ctx.progress,
    )
    arch = anatomy["architecture"]
    return {
        "manifest": build_manifest(
            model_dir=lm.path,
            run_type="timing",
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
