"""Drill-down queries: small, specific answers from the latest runs on disk.

    layer N              everything measured for one layer
    neuron L N           one neuron's firing by prompt set
    top_neurons L SET    the neurons in layer L most specific to a prompt set
    head L H             one attention head's copy attention and direct push
    tensor L KIND        one weight matrix's quantization sensitivity

Answers are plain text of a few hundred tokens at most, for a person or an AI
advisor that wants more detail than the report without the raw data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xray.report import collect

HELP = "Queries: layer N | neuron L N | top_neurons L SET | head L H | tensor L KIND"
KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class QueryError(ValueError):
    pass


def _int(value: str, what: str, limit: int) -> int:
    try:
        n = int(value)
    except ValueError as e:
        raise QueryError(f"{what} must be a number, got '{value}'. {HELP}") from e
    if not 0 <= n < limit:
        raise QueryError(f"{what} must be between 0 and {limit - 1}.")
    return n


def _arrays(data: dict[str, Any]):
    from xray.analyzers.profile import load_arrays

    run = data["latest"].get("profile")
    if not run or "activations" not in run["artifacts"]:
        raise QueryError("No activation profile for this model yet (run `xray profile`).")
    return load_arrays(Path(run["artifacts"]["activations"]))


def _layer(data: dict[str, Any], i: int) -> str:
    latest = data["latest"]
    out = [f"Layer {i}: {data['anatomy']['layers'][i]['params'] / 1e6:.1f}M params."]
    if "timing" in latest:
        d = latest["timing"]["result"]["contexts"][0]["decode"]["layers"][i]
        out.append(
            f"Decode time {d['est_ms']:.2f} ms (attention {d['attn_ms']:.2f}, MLP {d['mlp_ms']:.2f})."
        )
    if "profile" in latest:
        r = latest["profile"]["result"]
        a = r["all"]["layers"][i]
        out.append(
            f"MLP neurons firing {a['mean_firing_rate']:.1%}, always-on {a['hot_fraction']:.1%}, "
            f"cold {a['cold_fraction']:.1%}; output-to-input similarity {a['in_out_cosine']:.3f}."
        )
        if r["redundancy"]:
            s = r["redundancy"]["layers"][i]
            out.append(
                f"Skipped: prediction shift KL {s['drop_kl']:.3f}, top prediction unchanged "
                f"{s['drop_top1_agreement']:.1%}, loss {s['drop_nll_increase']:+.3f}."
            )
    if "quant" in latest:
        q = latest["quant"]["result"]
        rows = [t for t in q["tensors"] if t["layer"] == i]
        out.append(
            f"Quantized to {q['settings']['probe_bits']} bits, one matrix at a time (KL): "
            + ", ".join(f"{t['kind']} {t['kl']:.4f}" for t in rows)
            + "."
        )
    loops = (latest.get("loops") or {}).get("result", {}).get("analysis")
    if loops:
        heads = loops["copy_attention"][i]
        best = max(range(len(heads)), key=lambda h: heads[h])
        out.append(
            f"In the examined loop: similarity to one period earlier {loops['recurrence_in_loop'][i]:.2f} "
            f"(before the loop {loops['recurrence_before'][i]:.2f}); strongest copy head {best} "
            f"({heads[best]:.0%})."
        )
    return " ".join(out)


def run_query(model: str, text: str, home: str | Path | None = None) -> str:
    parts = text.split()
    if not parts or parts[0] == "help":
        return HELP
    data = collect(model, home)
    n_layers = len(data["anatomy"]["layers"])
    arch = data["anatomy"]["architecture"]
    op, args = parts[0], parts[1:]

    if op == "layer" and len(args) == 1:
        return _layer(data, _int(args[0], "layer", n_layers))

    if op == "neuron" and len(args) == 2:
        from xray.analyzers.profile import neuron_detail

        layer = _int(args[0], "layer", n_layers)
        neuron = _int(args[1], "neuron", arch["mlp_size"])
        detail = neuron_detail(_arrays(data), layer, neuron)
        return f"Layer {layer} neuron {neuron}. " + " ".join(
            f"{name}: fires on {d['reach_rate']['0.1']:.1%} of tokens, mean {d['mean']:+.3f}, "
            f"max {d['max_abs']:.2f}."
            for name, d in detail.items()
        )

    if op == "top_neurons" and len(args) in (2, 3):
        from xray.analyzers.profile import top_neurons

        by_set = _arrays(data)
        layer = _int(args[0], "layer", n_layers)
        if args[1] not in by_set:
            raise QueryError(f"No prompt set '{args[1]}'. Sets: {', '.join(by_set)}.")
        k = _int(args[2], "count", 21) if len(args) == 3 else 8
        rows = top_neurons(by_set, args[1], layer, k)
        return (
            f"Layer {layer}, most {args[1]}-specific neurons (fires here vs elsewhere): "
            + ", ".join(
                f"{r['neuron']} ({r['rate']:.0%} vs {r['rate_elsewhere']:.0%})" for r in rows
            )
        )

    if op == "head" and len(args) == 2:
        layer = _int(args[0], "layer", n_layers)
        head = _int(args[1], "head", arch["n_heads"])
        out = [f"Layer {layer} head {head}."]
        loops = (data["latest"].get("loops") or {}).get("result", {}).get("analysis")
        if loops:
            out.append(
                f"In the examined loop it sends {loops['copy_attention'][layer][head]:.0%} of its attention to the token after the previous copy."
            )
        attribution = data["latest"].get("attribution")
        if attribution:
            r = attribution["result"]
            # The prompt and tokens are the user's own text, so they stay out of the answer.
            out.append(
                "In the latest attribution run its direct push on the gap between the chosen "
                f"token and the runner-up is {r['head_direct'][layer][head]:+.2f}."
            )
        if len(out) == 1:
            out.append("No loop or attribution run covers this head yet.")
        return " ".join(out)

    if op == "tensor" and len(args) == 2:
        layer = _int(args[0], "layer", n_layers)
        if args[1] not in KINDS:
            raise QueryError(f"KIND must be one of: {', '.join(KINDS)}.")
        run = data["latest"].get("quant")
        if not run:
            raise QueryError("No quantization run for this model yet (run `xray quant`).")
        q = run["result"]
        t = next(t for t in q["tensors"] if t["layer"] == layer and t["kind"] == args[1])
        plan = "; ".join(
            f"{p['target_bits']}-bit-size plan gives it {p['mixed']['assignment'][t['name']]} bits"
            for p in q["plans"]
        )
        return (
            f"Layer {layer} {args[1]}: {t['params'] / 1e6:.1f}M weights. Prediction shift by bit width: "
            + ", ".join(f"{b} bits KL {kl:.4f}" for b, kl in t["kl_by_bits"].items())
            + f". Largest weight is {t['outlier_ratio']:.0f}x typical. {plan}."
        )

    raise QueryError(f"Could not understand '{text}'. {HELP}")
