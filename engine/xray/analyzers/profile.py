"""Activation profiler: which neurons fire for which kind of prompt, and which
layers the model could do without.

For each prompt set the model reads the prompt and generates a few tokens while
an ActivationProbe keeps running per-neuron statistics. Nothing grows with the
number of prompts. Afterwards:

  - per-set summaries: firing rates, hot and cold neurons, residual stream
  - comparison: neurons specific to one set, and neurons a set never uses
  - redundancy: how much each block changes its input, and what happens to the
    model's predictions when that block is skipped

The per-neuron arrays are saved to `activations.npz` so later queries (and the
desktop app) can read them without re-running the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from xray.core.device import default_dtype, now, pick_device, system_info
from xray.core.manifest import build_manifest, model_fingerprint
from xray.core.memory import guard_memory
from xray.data import PromptSet, load_prompt_set
from xray.models.adapters import LoadedModel, load_model
from xray.models.inspect import inspect_model
from xray.probes.hooks import EDGES, ActivationProbe, SkipProbe

ARRAYS_FILE = "activations.npz"
FIRE_THRESHOLD = 0.1  # must be one of EDGES
HOT_RATE = 0.5  # fires on at least half of all tokens
COLD_RATE = 0.01  # fires on at most 1% of tokens
SPECIFIC_GAP = 0.2  # fires this much more often in one set than in the others
USED_ELSEWHERE_RATE = 0.05

Arrays = dict[str, np.ndarray]


# -- encoding -------------------------------------------------------------------


def encode_prompts(
    lm: LoadedModel, prompts: list[str], reserve: int = 0, chat: bool = True
) -> list[torch.Tensor]:
    """Tokenise prompts, using the model's chat template when it has one.

    Prompts are truncated so that prompt plus `reserve` generated tokens fits
    in the model's context. Pass `chat=False` to feed the text as it is, which
    suits base (non-chat) models.
    """
    max_ctx = getattr(lm.model.config, "max_position_embeddings", None)
    limit = max(2, max_ctx - reserve - 1) if max_ctx else None
    tok = lm.tokenizer
    encoded = []
    for p in prompts:
        if tok is None:
            # No tokenizer (test models): map bytes onto the vocabulary.
            ids = torch.tensor([[b % lm.model.config.vocab_size for b in p.encode()]])
        elif chat and getattr(tok, "chat_template", None):
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        else:
            ids = tok(p, return_tensors="pt").input_ids
        encoded.append(ids[:, :limit] if limit else ids)
    return encoded


def _eos_ids(lm: LoadedModel) -> set[int]:
    eos = getattr(lm.model.config, "eos_token_id", None)
    if eos is None:
        return set()
    return set(eos) if isinstance(eos, list | tuple) else {eos}


def allocated_bytes(device: torch.device) -> int:
    """Memory currently held for tensors on `device` (process memory on CPU)."""
    if device.type == "mps":
        return torch.mps.current_allocated_memory()
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    return psutil.Process().memory_info().rss


# -- summaries (pure numpy: work the same on live and reloaded arrays) ----------


def firing_rate(arr: Arrays, threshold: float = FIRE_THRESHOLD) -> np.ndarray:
    """Share of tokens on which each neuron's |activation| reached `threshold`."""
    return arr["ge"][:, EDGES.index(threshold), :] / max(int(arr["tokens"]), 1)


def pool(arrays: list[Arrays]) -> Arrays:
    """Combine several groups' statistics into one."""
    out = {k: sum(a[k] for a in arrays) for k in arrays[0] if k != "abs_max"}
    out["abs_max"] = np.maximum.reduce([a["abs_max"] for a in arrays])
    return out


def summarise_layers(arr: Arrays) -> list[dict[str, Any]]:
    tokens = max(int(arr["tokens"]), 1)
    rate = firing_rate(arr)
    block_tokens = np.maximum(arr["block_tokens"], 1)
    cos = arr["block_cos_sum"] / block_tokens
    norm = arr["block_norm_sum"] / block_tokens
    return [
        {
            "layer": i,
            "neurons": int(rate.shape[1]),
            "mean_abs_activation": float(arr["abs_sum"][i].mean() / tokens),
            "max_abs_activation": float(arr["abs_max"][i].max()),
            "mean_firing_rate": float(rate[i].mean()),
            "hot_fraction": float((rate[i] >= HOT_RATE).mean()),
            "cold_fraction": float((rate[i] <= COLD_RATE).mean()),
            "in_out_cosine": float(cos[i]),
            "residual_norm": float(norm[i]),
        }
        for i in range(rate.shape[0])
    ]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation; 0.0 when either side is constant (undefined)."""
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compare_sets(by_set: dict[str, Arrays], top: int = 10) -> dict[str, Any]:
    """Each set against all the others pooled.

    specific: fires at least SPECIFIC_GAP more often here than elsewhere.
    unused:   cold here but in use elsewhere, so a model serving only this
              kind of prompt might not need it.
    """
    out = {}
    for name, arr in by_set.items():
        others = [a for n, a in by_set.items() if n != name]
        if not others:
            continue
        here, rest = firing_rate(arr), firing_rate(pool(others))
        gap = here - rest
        specific = gap >= SPECIFIC_GAP
        unused = (here <= COLD_RATE) & (rest >= USED_ELSEWHERE_RATE)

        flat = np.argsort(gap, axis=None)[::-1][:top]
        layers_idx, neurons_idx = np.unravel_index(flat, gap.shape)
        out[name] = {
            "specific_fraction": float(specific.mean()),
            "unused_fraction": float(unused.mean()),
            "layers": [
                {
                    "layer": i,
                    "specific_fraction": float(specific[i].mean()),
                    "unused_fraction": float(unused[i].mean()),
                    "rate_correlation": _correlation(here[i], rest[i]),
                }
                for i in range(gap.shape[0])
            ],
            "top_specific": [
                {
                    "layer": int(li),
                    "neuron": int(ni),
                    "rate": float(here[li, ni]),
                    "rate_elsewhere": float(rest[li, ni]),
                }
                for li, ni in zip(layers_idx, neurons_idx, strict=True)
            ],
        }
    return out


# -- redundancy -----------------------------------------------------------------


@torch.inference_mode()
def drop_layer_ablation(
    lm: LoadedModel, encoded: list[torch.Tensor], progress=lambda *a, **k: None
) -> dict[str, Any]:
    """Skip one block at a time and measure how far the predictions move.

    For each block: KL divergence from the intact model's next-token
    distribution, how often the top prediction stays the same, and the change
    in the model's loss on the prompt's own tokens.
    """
    n_layers = len(lm.refs.layers)
    kl = torch.zeros(n_layers, dtype=torch.float64)
    agree = torch.zeros(n_layers, dtype=torch.float64)
    nll = torch.zeros(n_layers, dtype=torch.float64)
    base_nll, positions, targets_n = 0.0, 0, 0

    def loss(logp: torch.Tensor, ids: torch.Tensor) -> float:
        return -logp[:-1].gather(1, ids[0, 1:, None]).sum().item()

    with SkipProbe(lm.refs) as skip:
        for n, ids in enumerate(encoded):
            progress("layer ablation", n, len(encoded))
            ids = ids.to(lm.device)
            if ids.shape[1] < 3:
                continue
            skip.layers = frozenset()
            base = torch.log_softmax(lm.model(input_ids=ids).logits[0].float(), dim=-1)
            base_top = base[1:].argmax(-1)
            base_nll += loss(base, ids)
            positions += ids.shape[1] - 1
            targets_n += ids.shape[1] - 1
            for i in range(n_layers):
                skip.layers = frozenset({i})
                logp = torch.log_softmax(lm.model(input_ids=ids).logits[0].float(), dim=-1)
                kl[i] += (base[1:].exp() * (base[1:] - logp[1:])).sum().item()
                agree[i] += (logp[1:].argmax(-1) == base_top).sum().item()
                nll[i] += loss(logp, ids)
            skip.layers = frozenset()

    positions, targets_n = max(positions, 1), max(targets_n, 1)
    return {
        "prompts": len(encoded),
        "positions": positions,
        "baseline_nll": base_nll / targets_n,
        "layers": [
            {
                "layer": i,
                "drop_kl": kl[i].item() / positions,
                "drop_top1_agreement": agree[i].item() / positions,
                "drop_nll_increase": nll[i].item() / targets_n - base_nll / targets_n,
            }
            for i in range(n_layers)
        ],
    }


# -- the profile run ------------------------------------------------------------


@torch.inference_mode()
def run_profile(
    lm: LoadedModel,
    sets: dict[str, list[torch.Tensor]],
    new_tokens: int = 16,
    ablation_prompts: int = 16,
    progress=lambda *a, **k: None,
) -> tuple[dict[str, Any], dict[str, Arrays]]:
    """Profile every prompt set. Returns (JSON-serialisable result, arrays by set)."""
    eos = _eos_ids(lm)
    total = sum(len(v) for v in sets.values())
    sample_every = max(1, total // 20)
    memory, done, generated = [], 0, 0
    t_start = now(lm.device)

    with ActivationProbe(lm.refs) as probe:
        for name, encoded in sets.items():
            probe.begin(name)
            for ids in encoded:
                if done % sample_every == 0:
                    progress(f"profiling {name}", done, total)
                probe.skip_first = True
                out = lm.model(input_ids=ids.to(lm.device), use_cache=new_tokens > 0)
                probe.skip_first = False
                for _ in range(new_tokens):
                    nxt = out.logits[:, -1].argmax(-1, keepdim=True)
                    if nxt.item() in eos:
                        break
                    out = lm.model(
                        input_ids=nxt, past_key_values=out.past_key_values, use_cache=True
                    )
                    generated += 1
                del out
                done += 1
                if done % sample_every == 0 or done == total:
                    memory.append(
                        {"prompts_done": done, "allocated_gb": allocated_bytes(lm.device) / 1e9}
                    )
        by_set = {name: probe.arrays(name) for name in sets}
    seconds = now(lm.device) - t_start

    # A balanced sample across sets for the (much more expensive) ablation.
    per_set = max(1, ablation_prompts // max(len(sets), 1))
    sample = [ids for encoded in sets.values() for ids in encoded[:per_set]][:ablation_prompts]
    ablation = drop_layer_ablation(lm, sample, progress) if ablation_prompts > 0 else None

    everything = pool(list(by_set.values()))
    all_layers = summarise_layers(everything)
    redundancy = None
    if ablation:
        for row, summary in zip(ablation["layers"], all_layers, strict=True):
            row["in_out_cosine"] = summary["in_out_cosine"]
        ranked = sorted(ablation["layers"], key=lambda r: r["drop_kl"])
        redundancy = {**ablation, "most_removable": [r["layer"] for r in ranked[:5]]}

    # Memory should be flat once every kernel shape has been seen; compare the
    # end with a point a quarter of the way in.
    early = memory[len(memory) // 4]["allocated_gb"] if memory else 0.0
    result = {
        "sets": {
            name: {
                "prompts": len(sets[name]),
                "tokens": int(arr["tokens"]),
                "layers": summarise_layers(arr),
            }
            for name, arr in by_set.items()
        },
        "all": {"tokens": int(everything["tokens"]), "layers": all_layers},
        "comparison": compare_sets(by_set),
        "redundancy": redundancy,
        "memory": {
            "samples": memory,
            "growth_gb": (memory[-1]["allocated_gb"] - early) if memory else 0.0,
            "peak_gb": max((m["allocated_gb"] for m in memory), default=0.0),
        },
        "run": {
            "prompts": total,
            "generated_tokens": generated,
            "new_tokens_per_prompt": new_tokens,
            "seconds": seconds,
        },
        "thresholds": {
            "fire": FIRE_THRESHOLD,
            "hot_rate": HOT_RATE,
            "cold_rate": COLD_RATE,
            "specific_gap": SPECIFIC_GAP,
            "used_elsewhere_rate": USED_ELSEWHERE_RATE,
        },
        "arrays_file": ARRAYS_FILE,
    }
    return result, by_set


# -- disk -----------------------------------------------------------------------


def save_arrays(path: Path, by_set: dict[str, Arrays]) -> None:
    flat = {f"{name}/{key}": value for name, arr in by_set.items() for key, value in arr.items()}
    np.savez_compressed(path, edges=np.array(EDGES), **flat)


def load_arrays(path: Path) -> dict[str, Arrays]:
    """Read back what `save_arrays` wrote: {set name: arrays}."""
    by_set: dict[str, Arrays] = {}
    with np.load(path) as data:
        if tuple(data["edges"]) != EDGES:
            raise ValueError(f"{path} was written with different histogram edges.")
        for key in data.files:
            if "/" in key:
                name, field = key.split("/", 1)
                by_set.setdefault(name, {})[field] = data[key]
    return by_set


def neuron_detail(by_set: dict[str, Arrays], layer: int, neuron: int) -> dict[str, Any]:
    """Everything recorded about one neuron, per prompt set."""
    out = {}
    for name, arr in by_set.items():
        tokens = max(int(arr["tokens"]), 1)
        mean = arr["sum"][layer, neuron] / tokens
        var = max(arr["sq_sum"][layer, neuron] / tokens - mean**2, 0.0)
        out[name] = {
            "tokens": tokens,
            "mean": float(mean),
            "std": float(var**0.5),
            "mean_abs": float(arr["abs_sum"][layer, neuron] / tokens),
            "max_abs": float(arr["abs_max"][layer, neuron]),
            "reach_rate": {
                str(e): float(arr["ge"][layer, k, neuron] / tokens) for k, e in enumerate(EDGES)
            },
        }
    return out


def top_neurons(
    by_set: dict[str, Arrays], set_name: str, layer: int, top: int = 10
) -> list[dict[str, Any]]:
    """Neurons in `layer` that fire most specifically for `set_name`."""
    here = firing_rate(by_set[set_name])[layer]
    others = [a for n, a in by_set.items() if n != set_name]
    rest = firing_rate(pool(others))[layer] if others else np.zeros_like(here)
    order = np.argsort(here - rest)[::-1][:top]
    return [
        {"neuron": int(n), "rate": float(here[n]), "rate_elsewhere": float(rest[n])} for n in order
    ]


# -- job ------------------------------------------------------------------------


def layer_stats(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a profile result into rows for the `layer_stats` table."""
    rows = []
    metrics = (
        "mean_firing_rate",
        "hot_fraction",
        "cold_fraction",
        "in_out_cosine",
        "residual_norm",
    )
    groups = {**{n: s["layers"] for n, s in result["sets"].items()}, "all": result["all"]["layers"]}
    for name, layers in groups.items():
        for layer in layers:
            for metric in metrics:
                rows.append(
                    {"context": None, "phase": name, "layer": layer["layer"],
                     "component": "mlp" if "fraction" in metric or "firing" in metric else "block",
                     "metric": metric, "value": layer[metric]}
                )  # fmt: skip
    for layer in (result["redundancy"] or {}).get("layers", []):
        for metric in ("drop_kl", "drop_top1_agreement", "drop_nll_increase"):
            rows.append(
                {"context": None, "phase": "all", "layer": layer["layer"],
                 "component": "block", "metric": metric, "value": layer[metric]}
            )  # fmt: skip
    return rows


def profile_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    """Worker entry point for the `profile` job."""
    seed = params.get("seed", 0)
    torch.manual_seed(seed)
    new_tokens = params.get("new_tokens", 16)

    prompt_sets: list[PromptSet] = [load_prompt_set(s, params.get("limit")) for s in params["sets"]]

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

    encoded = {ps.name: encode_prompts(lm, ps.prompts, reserve=new_tokens) for ps in prompt_sets}
    result, by_set = run_profile(
        lm,
        encoded,
        new_tokens=new_tokens,
        ablation_prompts=params.get("ablation_prompts", 16),
        progress=ctx.progress,
    )
    ctx.progress("saving")
    save_arrays(ctx.run_dir / ARRAYS_FILE, by_set)

    arch = anatomy["architecture"]
    manifest = build_manifest(
        model_dir=lm.path, run_type="profile", device=lm.device, dtype=lm.dtype,
        seed=seed, params=params,
    )  # fmt: skip
    manifest["prompt_sets"] = [
        {"name": ps.name, "source": ps.source, "digest": ps.digest, "prompts": len(ps.prompts)}
        for ps in prompt_sets
    ]
    return {
        "manifest": manifest,
        "model": {
            "fingerprint": model_fingerprint(lm.path),
            "path": str(lm.path),
            "arch": arch["model_type"],
            "n_layers": arch["n_layers"],
            "params": anatomy["totals"]["params"],
        },
        "result": result,
        "layer_stats": layer_stats(result),
        "artifacts": {"activations": ARRAYS_FILE},
    }
