"""Headline numbers for a run, and a side-by-side comparison of two runs.

Each run type contributes a short list of metrics. `better` says which way is
good ("higher", "lower" or None when neither is), so a comparison can mark a
change as an improvement or a regression.
"""

from __future__ import annotations

from typing import Any

Metric = dict[str, Any]


def _m(label: str, value: float | None, fmt: str, better: str | None = None) -> Metric:
    return {"label": label, "value": value, "format": fmt, "better": better}


def _timing(r: dict) -> list[Metric]:
    first, last = r["contexts"][0], r["contexts"][-1]
    d = first["decode"]
    return [
        _m("Decode speed (tokens/s)", d["tokens_per_s"], "num1", "higher"),
        _m(
            f"Prefill speed at {last['context']} tokens (tokens/s)",
            last["prefill"]["tokens_per_s"],
            "num0",
            "higher",
        ),
        _m("Decode time per token (ms)", d["wall_ms"], "num2", "lower"),
        _m("Attention share of decode", d["shares"]["attention"], "pct"),
        _m("MLP share of decode", d["shares"]["mlp"], "pct"),
        _m("Output head share of decode", d["shares"]["lm_head"], "pct"),
        _m("Decode vs. memory limit (x)", r["bandwidth"]["times_above_floor"], "num2", "lower"),
    ]


def _profile(r: dict) -> list[Metric]:
    layers = r["all"]["layers"]
    out = [
        _m("Prompts", r["run"]["prompts"], "int"),
        _m("Tokens measured", r["all"]["tokens"], "int"),
        _m(
            "Neurons firing (average)",
            sum(x["mean_firing_rate"] for x in layers) / len(layers),
            "pct",
        ),
        _m("Cold neurons (average)", sum(x["cold_fraction"] for x in layers) / len(layers), "pct2"),
        _m("Peak memory (GB)", r["memory"]["peak_gb"], "num2", "lower"),
    ]
    for name, cmp in r["comparison"].items():
        out.append(_m(f"Neurons specific to {name}", cmp["specific_fraction"], "pct2"))
    if r["redundancy"]:
        kls = [x["drop_kl"] for x in r["redundancy"]["layers"]]
        out.append(_m("Smallest damage from skipping a layer (KL)", min(kls), "num3"))
    return out


def _quant(r: dict) -> list[Metric]:
    out = [_m("Matrices tested", len(r["tensors"]), "int")]
    for plan in r["plans"]:
        size = f"{plan['target_bits']}-bit size"
        out += [
            _m(f"Uniform KL at {size}", plan["uniform"]["kl"], "num4", "lower"),
            _m(f"Mixed plan KL at {size}", plan["mixed"]["kl"], "num4", "lower"),
            _m(
                f"Mixed plan top prediction unchanged at {size}",
                plan["mixed"]["top1_agreement"],
                "pct",
                "higher",
            ),
        ]
    return out


def _lens(r: dict) -> list[Metric]:
    return [
        _m("Tokens inspected", len(r["answer_layer"]), "int"),
        _m("Average layer where the token is decided", r["mean_answer_layer"], "num1"),
    ]


def _attribution(r: dict) -> list[Metric]:
    return [
        _m("Logit gap to the runner-up", r["logit_diff"], "num2"),
        _m("Largest gap change from switching a block off", max(
            abs(c[k]) for c in r["components"] for k in ("attn_ablate", "mlp_ablate")
        ), "num2"),
    ]  # fmt: skip


def _loops(r: dict) -> list[Metric]:
    s = r["summary"]
    out = [
        _m("Prompts", s["prompts"], "int"),
        _m("Looped", s["loop_rate"], "pct", "lower"),
        _m("Ended normally", s["ended_normally"] / max(s["prompts"], 1), "pct", "higher"),
    ]
    for test in (r["causal"] or {}).get("tests", []):
        n = r["causal"]["looping_prompts_tested"]
        out.append(
            _m(
                f"Still looping with {test['intervention']}",
                test["still_looping"] / max(n, 1),
                "pct",
                "lower",
            )
        )
    return out


def _edit(r: dict) -> list[Metric]:
    stages = r["stages"]
    last = list(stages.values())[-1]
    base = stages["base"]
    out = [
        _m("Parameters (billions)", last["size"]["params"] / 1e9, "num3", "lower"),
        _m(
            "Parameters removed",
            1 - last["size"]["params"] / base["size"]["params"],
            "pct",
            "higher",
        ),
        _m("Layers", last["size"]["layers"], "int"),
        _m("MLP width", last["size"]["mlp_width"], "int"),
    ]
    for task in r["tasks"]:
        out.append(
            _m(f"{task.capitalize()} accuracy", last["tasks"][task]["accuracy"], "pct", "higher")
        )
    h = last["health"]
    out += [
        _m("Drift from base (KL)", last["drift"]["kl"], "num4", "lower"),
        _m("Top prediction unchanged", last["drift"]["top1_agreement"], "pct", "higher"),
        _m("Health prompts that loop", h["looped"] / max(h["prompts"], 1), "pct", "lower"),
        _m("Health prompts that end normally", h["ended"] / max(h["prompts"], 1), "pct", "higher"),
    ]
    return out


def _bench(r: dict) -> list[Metric]:
    out = []
    for row in r["rows"]:
        name = f"{row['runtime']} {row['variant']}"
        out += [
            _m(f"{name}: decode (tokens/s)", row["decode_tps"], "num1", "higher"),
            _m(f"{name}: prefill (tokens/s)", row["prefill_tps"], "num0", "higher"),
        ]
    return out


def runtime_rows(timing: dict | None, bench: dict | None) -> list[dict[str, Any]]:
    """PyTorch (the analysis backend) followed by every measured real runtime.

    Each row: runtime, variant, file_bytes (None for PyTorch), prefill_tps,
    decode_tps, and `vs_pytorch` (decode speed relative to PyTorch) when both exist.
    """
    rows: list[dict[str, Any]] = []
    if timing:
        n_prompt = (bench or {}).get("n_prompt", 512)
        ctx = min(timing["contexts"], key=lambda c: abs(c["context"] - n_prompt))
        rows.append(
            {
                "runtime": "PyTorch",
                "variant": "analysis backend",
                "prefill_context": ctx["context"],
                "file_bytes": None,
                "prefill_tps": ctx["prefill"]["tokens_per_s"],
                "decode_tps": timing["contexts"][0]["decode"]["tokens_per_s"],
            }
        )
    for row in (bench or {}).get("rows", []):
        rows.append(
            {
                k: row.get(k)
                for k in ("runtime", "variant", "file_bytes", "prefill_tps", "decode_tps")
            }
        )
    base = rows[0]["decode_tps"] if timing and rows else None
    for row in rows:
        row["vs_pytorch"] = row["decode_tps"] / base if base else None
    return rows


SUMMARIES = {
    "timing": _timing, "profile": _profile, "quant": _quant, "lens": _lens,
    "attribution": _attribution, "loops": _loops, "edit": _edit,
    "bench": _bench,
}  # fmt: skip


def summarize(run_type: str, result: dict[str, Any]) -> list[Metric]:
    if run_type not in SUMMARIES:
        raise ValueError(f"No summary for '{run_type}' runs.")
    return SUMMARIES[run_type](result)


def compare_runs(type_a: str, result_a: dict, type_b: str, result_b: dict) -> list[dict[str, Any]]:
    """Rows for every metric either run has: both values, the change, and its verdict."""
    if type_a != type_b:
        raise ValueError(f"Cannot compare a {type_a} run with a {type_b} run.")
    a = {m["label"]: m for m in summarize(type_a, result_a)}
    b = {m["label"]: m for m in summarize(type_b, result_b)}
    rows = []
    for label in dict.fromkeys([*a, *b]):
        ma, mb = a.get(label), b.get(label)
        meta = ma or mb
        va, vb = (ma or {}).get("value"), (mb or {}).get("value")
        change = verdict = None
        if va is not None and vb is not None:
            change = vb - va
            if meta["better"] and abs(change) > 1e-12:
                good = change > 0 if meta["better"] == "higher" else change < 0
                verdict = "better" if good else "worse"
            elif abs(change) <= 1e-12:
                verdict = "same"
        rows.append(
            {
                "label": label,
                "format": meta["format"],
                "a": va,
                "b": vb,
                "change": change,
                "verdict": verdict,
            }
        )
    return rows
