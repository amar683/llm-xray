"""Logit lens and attribution: where an answer forms, and what produced it."""

from __future__ import annotations

from typing import Any

import torch

from xray.analyzers.common import (
    greedy_generate,
    job_output,
    load_for_job,
    norm_scale,
    residual_states,
    token_text,
)
from xray.analyzers.profile import encode_prompts
from xray.models.adapters import LoadedModel

MAX_LENS_POSITIONS = 64


# -- logit lens -------------------------------------------------------------------


@torch.inference_mode()
def run_lens(lm: LoadedModel, ids: torch.Tensor, new_tokens: int = 24) -> dict[str, Any]:
    """Decode every layer's hidden state as if it were the last.

    The model first generates `new_tokens` greedily. Then, for each position
    that predicted a generated token, every layer's hidden state is passed
    through the final norm and output head. This shows at which layer the
    final prediction first appears and how its probability builds up.
    """
    generated, _ = greedy_generate(lm, ids, new_tokens)
    full = torch.cat([ids, torch.tensor([generated], dtype=ids.dtype)], dim=1).to(lm.device)
    states, final_logits = residual_states(lm, full)

    n_prompt = ids.shape[1]
    # Position t predicts token t+1; keep the positions that predict generated tokens.
    first = max(n_prompt - 1, full.shape[1] - 1 - MAX_LENS_POSITIONS)
    positions = list(range(first, full.shape[1] - 1)) or [full.shape[1] - 1]
    pos = torch.tensor(positions, device=lm.device)
    target = final_logits[pos].argmax(-1)  # the model's own final prediction

    probs, tops, top_ids = [], [], []
    for hidden in states:
        logits = lm.refs.lm_head(lm.refs.final_norm(hidden[pos])).float()
        p = torch.softmax(logits, dim=-1)
        probs.append(p.gather(1, target[:, None])[:, 0].tolist())
        best = logits.argmax(-1)
        top_ids.append(best)
        tops.append(token_text(lm, best.tolist()))

    # The layer from which the top prediction equals the final one and stays so.
    agree = torch.stack(top_ids) == target[None, :]  # [layers, positions]
    n_layers = agree.shape[0]
    answer_layer = []
    for j in range(agree.shape[1]):
        layer = n_layers - 1
        while layer > 0 and agree[layer - 1, j]:
            layer -= 1
        answer_layer.append(layer if agree[layer, j] else n_layers - 1)

    all_ids = full[0].tolist()
    return {
        "prompt_tokens": n_prompt,
        "generated_text": "".join(token_text(lm, generated)),
        "positions": positions,
        "input_tokens": token_text(lm, [all_ids[p] for p in positions]),
        "predicted_tokens": token_text(lm, target.tolist()),
        "prob": probs,  # [layers][positions]: probability of the final prediction
        "top": tops,  # [layers][positions]: each layer's own top token
        "answer_layer": answer_layer,
        "mean_answer_layer": sum(answer_layer) / len(answer_layer),
    }


# -- attribution ------------------------------------------------------------------


class _Capture:
    """Hooks that record (and optionally zero) each block's attention and MLP output."""

    def __init__(self, lm: LoadedModel) -> None:
        self.attn: dict[int, torch.Tensor] = {}
        self.mlp: dict[int, torch.Tensor] = {}
        self.heads: dict[int, torch.Tensor] = {}
        self.neurons: dict[int, torch.Tensor] = {}
        self.zero: tuple[str, int] | None = None  # ("attn" | "mlp", layer)
        self.zero_head: tuple[int, int] | None = None  # (layer, head)
        self.keep_grad = False
        self.n_heads = lm.model.config.num_attention_heads
        self._handles = []
        for layer in lm.refs.layers:
            i = layer.index
            self._handles += [
                layer.attn.register_forward_hook(self._attn(i)),
                layer.mlp.register_forward_hook(self._mlp(i)),
                layer.attn_out.register_forward_pre_hook(self._heads(i)),
                layer.mlp_out.register_forward_pre_hook(self._neurons(i)),
            ]

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def _attn(self, i: int):
        def hook(_m, _args, output):
            out = output[0] if isinstance(output, tuple) else output
            if self.zero == ("attn", i):
                out = torch.zeros_like(out)
                return (out, *output[1:]) if isinstance(output, tuple) else out
            self.attn[i] = out
            return None

        return hook

    def _mlp(self, i: int):
        def hook(_m, _args, output):
            if self.zero == ("mlp", i):
                return torch.zeros_like(output)
            self.mlp[i] = output
            return None

        return hook

    def _heads(self, i: int):
        def hook(_m, args):
            x = args[0]
            if self.zero_head and self.zero_head[0] == i:
                width = x.shape[-1] // self.n_heads
                h = self.zero_head[1]
                x = x.clone()
                x[..., h * width : (h + 1) * width] = 0
                return (x,)
            self.heads[i] = x
            return None

        return hook

    def _neurons(self, i: int):
        def hook(_m, args):
            if self.keep_grad:
                args[0].retain_grad()
            self.neurons[i] = args[0]

        return hook


def run_attribution(
    lm: LoadedModel,
    ids: torch.Tensor,
    target_id: int | None = None,
    top_heads: int = 12,
    top_neurons: int = 20,
    progress=lambda *a, **k: None,
) -> dict[str, Any]:
    """What produced the model's next-token prediction at the end of the prompt.

    Everything is measured on the *logit difference* between the predicted
    token and the runner-up, which removes effects common to all tokens.

    direct:  each component's own push on that difference (its output, read
             through the frozen final norm and the output head).
    ablate:  how the difference changes when the component's output is zeroed,
             which also counts its effect on later layers.
    neurons: gradient x activation, summed over positions.
    """
    ids = ids.to(lm.device)
    refs, model = lm.refs, lm.model
    cap = _Capture(lm)
    try:
        # One pass with gradients for the neuron attribution.
        cap.keep_grad = True
        ends: dict[str, torch.Tensor] = {}
        extra = [
            refs.final_norm.register_forward_pre_hook(
                lambda _m, args: ends.__setitem__("final", args[0].detach())
            ),
            refs.embed.register_forward_hook(
                lambda _m, _a, out: ends.__setitem__("embed", out.detach())
            ),
        ]
        out = model(input_ids=ids)
        for h in extra:
            h.remove()
        logits = out.logits[0, -1].float()
        order = logits.argsort(descending=True)
        target = int(target_id) if target_id is not None else int(order[0])
        runner = int(order[0]) if target != int(order[0]) else int(order[1])
        diff = logits[target] - logits[runner]
        diff.backward()
        base_diff = diff.item()
        neuron_attr = {
            i: (x.detach().float() * x.grad.float()).sum((0, 1)) for i, x in cap.neurons.items()
        }
        neuron_last = {i: x.detach()[0, -1].float() for i, x in cap.neurons.items()}
        model.zero_grad(set_to_none=True)
        cap.keep_grad = False

        with torch.inference_mode():
            w_u = refs.lm_head.weight
            scale = norm_scale(refs.final_norm, ends["final"][0, -1])
            # Direction in residual space that the logit difference reads.
            direction = (w_u[target].float() - w_u[runner].float()) * scale

            def direct(x: torch.Tensor) -> float:
                return float(x.detach()[0, -1].float() @ direction)

            n_heads = cap.n_heads
            components, head_grid = [], []
            for layer in refs.layers:
                i = layer.index
                z = cap.heads[i].detach()[0, -1].float()
                width = z.shape[0] // n_heads
                w_o = layer.attn_out.weight.float()  # [d_model, heads * width]
                per_head = [
                    float(
                        (w_o[:, h * width : (h + 1) * width] @ z[h * width : (h + 1) * width])
                        @ direction
                    )
                    for h in range(n_heads)
                ]
                head_grid.append(per_head)
                components.append(
                    {
                        "layer": i,
                        "attn_direct": direct(cap.attn[i]),
                        "mlp_direct": direct(cap.mlp[i]),
                    }
                )
            embed_direct = float(ends["embed"][0, -1].float() @ direction)

            def diff_now() -> float:
                lg = model(input_ids=ids).logits[0, -1].float()
                return (lg[target] - lg[runner]).item()

            for n, row in enumerate(components):
                progress("ablating components", n, len(components))
                for kind in ("attn", "mlp"):
                    cap.zero = (kind, row["layer"])
                    row[f"{kind}_ablate"] = diff_now() - base_diff
            cap.zero = None

            flat = sorted(
                ((abs(v), li, h) for li, row in enumerate(head_grid) for h, v in enumerate(row)),
                reverse=True,
            )[:top_heads]
            heads = []
            for n, (_, li, h) in enumerate(flat):
                progress("ablating heads", n, len(flat))
                cap.zero_head = (li, h)
                heads.append(
                    {
                        "layer": li,
                        "head": h,
                        "direct": head_grid[li][h],
                        "ablate": diff_now() - base_diff,
                    }
                )
            cap.zero_head = None

            stacked = torch.stack(
                [neuron_attr[i] for i in sorted(neuron_attr)]
            )  # [layers, neurons]
            best = stacked.abs().flatten().topk(min(top_neurons, stacked.numel())).indices
            neurons = [
                {
                    "layer": int(k // stacked.shape[1]),
                    "neuron": int(k % stacked.shape[1]),
                    "attribution": float(stacked.flatten()[k]),
                    "activation": float(
                        neuron_last[int(k // stacked.shape[1])][int(k % stacked.shape[1])]
                    ),
                }
                for k in best.tolist()
            ]
    finally:
        cap.remove()

    explained = embed_direct + sum(c["attn_direct"] + c["mlp_direct"] for c in components)
    return {
        "prompt_tokens": token_text(lm, ids[0].tolist()),
        "target": token_text(lm, [target])[0],
        "runner_up": token_text(lm, [runner])[0],
        "logit_diff": base_diff,
        "direct_total": explained,  # should be close to logit_diff
        "embed_direct": embed_direct,
        "components": components,
        "head_direct": head_grid,  # [layers][heads]
        "heads": heads,
        "neurons": neurons,
    }


# -- jobs -------------------------------------------------------------------------


def lens_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    lm, anatomy = load_for_job(params, ctx)
    ids = encode_prompts(
        lm, [params["prompt"]], reserve=params.get("new_tokens", 24), chat=params.get("chat", True)
    )[0]
    ctx.progress("reading layers")
    result = run_lens(lm, ids, new_tokens=params.get("new_tokens", 24))
    result["prompt"] = params["prompt"]
    return job_output(lm, anatomy, "lens", params, result)


def attribution_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    lm, anatomy = load_for_job(params, ctx)
    ids = encode_prompts(lm, [params["prompt"]], chat=params.get("chat", True))[0]
    result = run_attribution(lm, ids, progress=ctx.progress)
    result["prompt"] = params["prompt"]
    return job_output(lm, anatomy, "attribution", params, result)
