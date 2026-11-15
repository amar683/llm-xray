"""Loop hunter and loop analyzer.

A transformer has no loops between layers; a model "loops" when its output
starts repeating itself token for token. This module:

  1. hunts   - generates greedily from many prompts and flags the ones that
               end up repeating (and other failures: never stopping)
  2. looks inside one looping case:
       recurrence  how similar each layer's state is to the state one period
                   earlier, per token (the loop shows up as values near 1)
       copy heads  attention heads that look back exactly to the token that
                   followed the previous copy
       writers     which attention/MLP blocks push the repeated token
       direction   the average state while looping minus the average before,
                   and whether it rises before the loop starts
  3. tests the cause - switches off the top copy heads, or removes the loop
     direction, regenerates, and counts how many prompts still loop and how
     much the non-looping prompts change
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from xray.analyzers.common import (
    greedy_generate,
    job_output,
    load_for_job,
    norm_scale,
    token_text,
)
from xray.analyzers.profile import encode_prompts
from xray.data import load_prompt_set
from xray.models.adapters import LoadedModel

MIN_LOOP_TOKENS = 12
MAX_PERIOD = 64
MAX_RECURRENCE_COLUMNS = 240
DIRECTION_FILE = "loop_direction.npy"


def detect_loop(tokens: list[int]) -> dict[str, int] | None:
    """Find a repeating tail: tokens[i] == tokens[i - period] from `onset + period` on.

    Returns the smallest period whose repetition covers at least two further
    periods and MIN_LOOP_TOKENS tokens, with `onset` the index where the first
    cycle starts. None when the output does not end in a loop.
    """
    n = len(tokens)
    for period in range(1, min(MAX_PERIOD, n // 3) + 1):
        run = 0
        while run < n - period and tokens[n - 1 - run] == tokens[n - 1 - run - period]:
            run += 1
        if run >= 2 * period and run >= MIN_LOOP_TOKENS:
            return {"period": period, "onset": n - run - period, "repeated_tokens": run + period}
    return None


# -- ablations used by the causal test --------------------------------------------


class Ablation:
    """Hooks that switch off attention heads and/or remove a direction from the
    residual stream from one layer onwards. Inactive until `heads`/`direction` is set."""

    def __init__(self, lm: LoadedModel) -> None:
        self.heads: set[tuple[int, int]] = set()
        self.direction: torch.Tensor | None = None
        self.from_layer = 0
        self.n_heads = lm.model.config.num_attention_heads
        self._handles = []
        for layer in lm.refs.layers:
            self._handles.append(layer.attn_out.register_forward_pre_hook(self._heads(layer.index)))
            self._handles.append(layer.block.register_forward_hook(self._direction(layer.index)))

    def clear(self) -> None:
        self.heads, self.direction = set(), None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def _heads(self, i: int):
        def hook(_m, args):
            mine = [h for layer, h in self.heads if layer == i]
            if not mine:
                return None
            x = args[0].clone()
            width = x.shape[-1] // self.n_heads
            for h in mine:
                x[..., h * width : (h + 1) * width] = 0
            return (x,)

        return hook

    def _direction(self, i: int):
        def hook(_m, _args, output):
            if self.direction is None or i < self.from_layer:
                return None
            h = output[0] if isinstance(output, tuple) else output
            d = self.direction.to(h.dtype)
            h = h - (h @ d)[..., None] * d
            return (h, *output[1:]) if isinstance(output, tuple) else h

        return hook


# -- hunting ----------------------------------------------------------------------


def hunt(
    lm: LoadedModel,
    prompts: list[tuple[str, str]],
    encoded: list[torch.Tensor],
    new_tokens: int,
    progress=lambda *a, **k: None,
    label: str = "generating",
) -> list[dict[str, Any]]:
    """Generate from every prompt and record what went wrong, if anything."""
    cases = []
    for n, ((set_name, prompt), ids) in enumerate(zip(prompts, encoded, strict=True)):
        progress(label, n, len(prompts))
        tokens, ended = greedy_generate(lm, ids, new_tokens)
        loop = detect_loop(tokens)
        cases.append(
            {
                "index": n,
                "set": set_name,
                "prompt": prompt,
                "tokens": tokens,
                "n_tokens": len(tokens),
                "ended": ended,
                "hit_limit": not ended,
                "loop": loop,
                "tail": "".join(token_text(lm, tokens[-40:])),
            }
        )
    return cases


def _summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(cases), 1)
    looped = [c for c in cases if c["loop"]]
    return {
        "prompts": len(cases),
        "looped": len(looped),
        "loop_rate": len(looped) / n,
        "ended_normally": sum(c["ended"] for c in cases),
        "hit_limit_without_loop": sum(c["hit_limit"] and not c["loop"] for c in cases),
    }


# -- looking inside one loop ------------------------------------------------------


@torch.inference_mode()
def analyse_loop(lm: LoadedModel, ids: torch.Tensor, case: dict[str, Any]) -> dict[str, Any]:
    """Recurrence, copy heads, writers and loop direction for one looping generation.

    Needs a model loaded with eager attention (for attention weights).
    """
    refs, model = lm.refs, lm.model
    period, onset = case["loop"]["period"], case["loop"]["onset"]
    full = torch.cat([ids, torch.tensor([case["tokens"]], dtype=ids.dtype)], dim=1).to(lm.device)
    n_prompt, total = ids.shape[1], full.shape[1]
    onset_abs = n_prompt + onset

    attn_out: dict[int, torch.Tensor] = {}
    mlp_out: dict[int, torch.Tensor] = {}
    states: dict[int, torch.Tensor] = {}
    final_in: dict[str, torch.Tensor] = {}
    handles = [
        refs.final_norm.register_forward_pre_hook(lambda _m, a: final_in.__setitem__("h", a[0][0]))
    ]
    for layer in refs.layers:
        i = layer.index
        handles += [
            layer.attn.register_forward_hook(
                lambda _m, _a, out, i=i: attn_out.__setitem__(
                    i, (out[0] if isinstance(out, tuple) else out)[0]
                )
            ),
            layer.mlp.register_forward_hook(
                lambda _m, _a, out, i=i: mlp_out.__setitem__(i, out[0])
            ),
            layer.block.register_forward_hook(
                lambda _m, _a, out, i=i: states.__setitem__(
                    i, (out[0] if isinstance(out, tuple) else out)[0]
                )
            ),
        ]
    try:
        out = model(input_ids=full, output_attentions=True)
    finally:
        for h in handles:
            h.remove()
    if not out.attentions or out.attentions[0] is None:
        raise RuntimeError(
            "The model did not return attention weights; load it with eager attention."
        )

    n_layers = len(refs.layers)
    loop_pos = torch.arange(onset_abs + period, total, device=lm.device)  # inside the loop
    pre_pos = torch.arange(n_prompt, onset_abs, device=lm.device)  # generated, before the loop
    gen_pos = torch.arange(n_prompt, total, device=lm.device)

    # Recurrence: similarity to the state one period earlier, for every generated token.
    recurrence = torch.zeros(n_layers, total - n_prompt)
    for i in range(n_layers):
        h = states[i].float()
        valid = gen_pos[gen_pos - period >= 0]
        sim = torch.nn.functional.cosine_similarity(h[valid], h[valid - period], dim=-1)
        recurrence[i, (valid - n_prompt).cpu()] = sim.cpu()
    in_loop = recurrence[:, (loop_pos - n_prompt).cpu()].mean(1)
    before = (
        recurrence[:, (pre_pos - n_prompt).cpu()].mean(1) if len(pre_pos) else torch.zeros(n_layers)
    )
    locked = next((i for i in range(n_layers) if in_loop[i] >= 0.9), None)
    step = max(1, -(-recurrence.shape[1] // MAX_RECURRENCE_COLUMNS))
    cols = list(range(0, recurrence.shape[1], step))

    # Copy heads: attention from each loop token to the token after its previous copy.
    # With a period of one the previous copy is simply the previous token.
    back = period - 1 if period > 1 else 1
    copy = torch.stack(
        [a[0][:, loop_pos, loop_pos - back].float().mean(-1) for a in out.attentions]
    ).cpu()
    flat = copy.flatten().topk(min(8, copy.numel()))
    n_heads = copy.shape[1]
    copy_heads = [
        {"layer": int(k // n_heads), "head": int(k % n_heads), "attention": float(v)}
        for v, k in zip(flat.values.tolist(), flat.indices.tolist(), strict=True)
    ]

    # Writers: each block's direct push on the token that actually comes next.
    w_u = refs.lm_head.weight.float()
    centre = w_u.mean(0)

    def push(component: torch.Tensor, pos: torch.Tensor) -> float:
        if len(pos) == 0:
            return 0.0
        pos = pos[pos + 1 < total][-48:]
        direction = (w_u[full[0, pos + 1]] - centre) * norm_scale(
            refs.final_norm, final_in["h"][pos]
        )
        return float((component[pos].float() * direction).sum(-1).mean())

    writers = []
    for i in range(n_layers):
        for kind, comp in (("attn", attn_out[i]), ("mlp", mlp_out[i])):
            writers.append(
                {
                    "layer": i,
                    "kind": kind,
                    "in_loop": push(comp, loop_pos),
                    "before": push(comp, pre_pos),
                }
            )
    top_writers = sorted(writers, key=lambda w: -w["in_loop"])[:8]

    # Loop direction, at the layer where looping and normal states separate most.
    direction = None
    if len(pre_pos) >= 4:
        best = None
        for i in range(n_layers):
            h = states[i].float()
            d = h[loop_pos].mean(0) - h[pre_pos].mean(0)
            d = d / d.norm().clamp(min=1e-6)
            proj = h[gen_pos] @ d
            pre = proj[: len(pre_pos)]
            gap = (proj[len(pre_pos) + period :].mean() - pre.mean()) / pre.std().clamp(min=1e-6)
            if best is None or gap > best[0]:
                best = (float(gap), i, d, proj.cpu())
        gap, layer_i, d, proj = best
        # Early warning: first generated token whose projection clears the level
        # seen in the first half of the normal stretch by two standard deviations.
        calm = proj[: max(2, len(pre_pos) // 2)]
        threshold = float(calm.mean() + 2 * calm.std())
        crossing = next((t for t in range(len(calm), len(proj)) if proj[t] > threshold), None)
        direction = {
            "layer": layer_i,
            "separation_sigma": gap,
            "projection": [round(float(v), 3) for v in proj[cols]],
            "threshold": threshold,
            "first_crossing": crossing,
            "warning_tokens": (onset - crossing) if crossing is not None else None,
            "vector": d,  # removed before the result is saved
        }

    return {
        "prompt": case["prompt"],
        "set": case["set"],
        "period": period,
        "onset": onset,
        "generated_tokens": case["n_tokens"],
        "loop_text": "".join(token_text(lm, case["tokens"][onset : onset + period])),
        "tokens": [token_text(lm, [case["tokens"][c]])[0] for c in cols],
        "columns": cols,
        "recurrence": [[round(float(v), 3) for v in row[cols]] for row in recurrence],
        "recurrence_in_loop": in_loop.tolist(),
        "recurrence_before": before.tolist(),
        "locked_from_layer": locked,
        "copy_attention": copy.tolist(),  # [layers][heads]
        "copy_heads": copy_heads,
        "writers": top_writers,
        "direction": direction,
    }


# -- the run ----------------------------------------------------------------------


def _changed_share(before: list[dict], after: list[dict]) -> float:
    """Share of generated tokens that differ, position by position."""
    same = total = 0
    for b, a in zip(before, after, strict=True):
        n = max(len(b["tokens"]), len(a["tokens"]))
        total += n
        same += sum(x == y for x, y in zip(b["tokens"], a["tokens"], strict=False))
    return 1 - same / total if total else 0.0


def run_loops(
    lm: LoadedModel,
    prompts: list[tuple[str, str]],
    encoded: list[torch.Tensor],
    new_tokens: int = 120,
    ablate_heads: int = 4,
    test_prompts: int = 8,
    progress=lambda *a, **k: None,
    save_direction: Path | None = None,
) -> dict[str, Any]:
    cases = hunt(lm, prompts, encoded, new_tokens, progress)
    looping = [c for c in cases if c["loop"]]
    healthy = [c for c in cases if c["ended"]]
    result: dict[str, Any] = {
        "new_tokens": new_tokens,
        "summary": _summary(cases),
        "by_set": {
            s: _summary([c for c in cases if c["set"] == s])
            for s in dict.fromkeys(c["set"] for c in cases)
        },
        "cases": [{k: v for k, v in c.items() if k != "tokens"} for c in cases],
        "analysis": None,
        "causal": None,
    }
    if not looping:
        return result

    # Analyse the case with the longest normal stretch before the loop: it gives
    # the clearest before/after contrast.
    progress("looking inside a loop")
    case = max(looping, key=lambda c: c["loop"]["onset"])
    analysis = analyse_loop(lm, encoded[case["index"]], case)
    vector = analysis["direction"].pop("vector") if analysis["direction"] else None
    if vector is not None and save_direction is not None:
        # Saved so `ablate_direction vector=<file>` can use it as a weight edit.
        np.save(save_direction, vector.float().cpu().numpy())
        analysis["direction"]["file"] = str(save_direction)
    result["analysis"] = analysis

    # Causal test on a sample of looping prompts, with side effects on healthy ones.
    loop_sample, healthy_sample = looping[:test_prompts], healthy[:test_prompts]
    sample = loop_sample + healthy_sample
    sample_prompts = [(c["set"], c["prompt"]) for c in sample]
    sample_ids = [encoded[c["index"]] for c in sample]
    interventions = [
        (
            "copy heads off",
            {(h["layer"], h["head"]) for h in analysis["copy_heads"][:ablate_heads]},
            None,
        )
    ]
    if vector is not None:
        interventions.append(("loop direction removed", set(), vector))

    ablation = Ablation(lm)
    tests = []
    try:
        for label, heads, direction in interventions:
            ablation.clear()
            ablation.heads, ablation.direction = heads, direction
            ablation.from_layer = analysis["direction"]["layer"] if direction is not None else 0
            after = hunt(lm, sample_prompts, sample_ids, new_tokens, progress, f"testing: {label}")
            loops_after, healthy_after = after[: len(loop_sample)], after[len(loop_sample) :]
            tests.append(
                {
                    "intervention": label,
                    "heads": sorted(heads),
                    "from_layer": ablation.from_layer if direction is not None else None,
                    "still_looping": sum(bool(c["loop"]) for c in loops_after),
                    "now_end_normally": sum(c["ended"] for c in loops_after),
                    "healthy_still_end": sum(c["ended"] for c in healthy_after),
                    "healthy_now_loop": sum(bool(c["loop"]) for c in healthy_after),
                    "healthy_tokens_changed": _changed_share(healthy_sample, healthy_after),
                    "example_tail": loops_after[0]["tail"] if loops_after else "",
                }
            )
    finally:
        ablation.remove()
    result["causal"] = {
        "looping_prompts_tested": len(loop_sample),
        "healthy_prompts_tested": len(healthy_sample),
        "tests": tests,
    }
    return result


def loops_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    new_tokens = params.get("new_tokens", 120)
    sets = params.get("sets") or ["coding", "chat", "math", "instruction"]
    prompts = [(s, p) for s in sets for p in load_prompt_set(s, params.get("limit", 10)).prompts]
    if params.get("prompt"):
        prompts = [("custom", params["prompt"])]
    lm, anatomy = load_for_job(params, ctx, eager_attention=True)
    encoded = encode_prompts(
        lm, [p for _, p in prompts], reserve=new_tokens, chat=params.get("chat", True)
    )
    result = run_loops(
        lm,
        prompts,
        encoded,
        new_tokens=new_tokens,
        ablate_heads=params.get("ablate_heads", 4),
        test_prompts=params.get("test_prompts", 8),
        progress=ctx.progress,
        save_direction=ctx.run_dir / DIRECTION_FILE,
    )
    output = job_output(lm, anatomy, "loops", params, result)
    if (ctx.run_dir / DIRECTION_FILE).exists():
        output["artifacts"] = {"loop_direction": DIRECTION_FILE}
    return output
