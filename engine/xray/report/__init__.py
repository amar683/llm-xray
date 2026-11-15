"""Compact report: everything measured about a model, in a few thousand tokens.

The report is what a developer (or, optionally, an AI advisor) reads instead of
gigabytes of weights. It is built from the most recent run of each kind for a
model, entirely from files on disk.

Two rules shape it:

  budget   Sections are added in priority order and shortened or dropped until
           the whole text fits a token budget. Nothing is silently cut: the
           result lists which sections were shortened or left out.
  privacy  The text never contains a file path, user name or host name. Files
           are referred to by alias (`@loop_direction`), which only the local
           engine can resolve.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from xray.core import paths
from xray.core.storage import Storage
from xray.models.inspect import inspect_model

DEFAULT_BUDGET = 3000
# Number-heavy Markdown tokenizes densely: the real report measured 2.24
# characters per token with a model tokenizer. 2.2 keeps the estimate at or
# above the true count, so a report that "fits" really does.
CHARS_PER_TOKEN = 2.2
LOOP_DIRECTION_ALIAS = "@loop_direction"

COMMAND_HELP = """\
Edit commands (applied to a copy; the base model is never changed):
- zero_head layer=N head=N — switch off one attention head
- scale_neuron layer=N neuron=N factor=X — scale one MLP neuron (0 = off)
- drop_layers layers=A,B — remove whole layers
- prune_neurons keep=0.9 — keep the top share of MLP neurons per layer
- ablate_direction vector=@loop_direction from_layer=N — stop layers writing along the loop direction

Drill-down queries (answered locally, a few hundred tokens each):
- layer N — everything measured for one layer
- neuron L N — one neuron's firing by prompt set
- top_neurons L SET — the neurons in layer L most specific to a prompt set
- head L H — one attention head's copy attention and direct push
- tensor L KIND — one weight matrix's quantization sensitivity (KIND: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)"""


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def model_label(model_dir: str) -> str:
    """A short public name for a model folder, never the path itself."""
    m = re.search(r"models--([^/]+)--([^/]+)", model_dir)
    return f"{m.group(1)}/{m.group(2)}" if m else Path(model_dir).name


def collect(model: str, home: str | Path | None = None) -> dict[str, Any]:
    """The latest finished run of each kind for this model, plus every edit run."""
    anatomy = inspect_model(model)
    latest: dict[str, dict[str, Any]] = {}
    edits: list[dict[str, Any]] = []
    with Storage(paths.db_path(paths.home(home))) as db:
        for row in db.list_runs(1000):
            if row["status"] != "done" or row["model_path"] != anatomy["path"]:
                continue
            artifacts = db.artifacts(row["id"])
            if "result" not in artifacts:
                continue
            run_dir = Path(artifacts["result"]).parent
            entry = {
                "id": row["id"],
                "started_at": row["started_at"],
                "params": json.loads(row["params_json"]),
                "result": json.loads(Path(artifacts["result"]).read_text())["result"],
                "artifacts": artifacts,
                "extras": {
                    f.stem: json.loads(f.read_text())
                    for f in sorted(run_dir.glob("*.json"))
                    if f.name.startswith(("export-", "gguf-"))
                },
            }
            if row["type"] == "edit":
                edits.append(entry)
            latest.setdefault(row["type"], entry)
    return {"anatomy": anatomy, "latest": latest, "edits": edits}


# -- sections ---------------------------------------------------------------------
# Each returns (full text, short text). Either may be "" when there is no data.


def _pct(x: float, digits: int = 0) -> str:
    return f"{x * 100:.{digits}f}%"


def _model(data: dict) -> tuple[str, str]:
    a = data["anatomy"]
    arch, tot = a["architecture"], a["totals"]
    comp = a["by_component"]
    text = (
        f"# Model report: {model_label(a['path'])}\n"
        f"{arch['model_type']}, {arch['n_layers']} layers, hidden {arch['hidden_size']}, "
        f"MLP width {arch['mlp_size']}, {arch['n_heads']} heads ({arch['n_kv_heads']} KV), "
        f"vocab {arch['vocab_size']}. {tot['params'] / 1e9:.2f}B params, "
        f"{tot['bytes_on_disk'] / 2**30:.2f} GB at 16-bit. "
        f"MLP {_pct(comp.get('mlp', 0) / tot['params'])}, "
        f"embeddings {_pct(comp.get('embeddings', 0) / tot['params'])}, "
        f"attention {_pct(comp.get('attention', 0) / tot['params'])} of params."
    )
    return text, text


def _runtimes(data: dict) -> str:
    """One line on measured llama.cpp / MLX speed, when a bench run exists."""
    bench = data["latest"].get("bench")
    if not bench:
        return ""
    return (
        " Real runtimes (decode tok/s): "
        + ", ".join(
            f"{r['runtime']} {r['variant']} {r['decode_tps']:.0f}"
            + (f" ({r['file_bytes'] / 2**30:.2f} GB)" if r.get("file_bytes") else "")
            for r in bench["result"]["rows"]
        )
        + "."
    )


def _timing(data: dict) -> tuple[str, str]:
    run = data["latest"].get("timing")
    if not run:
        text = _runtimes(data).strip()
        return (f"## Speed\n{text}", f"## Speed\n{text}") if text else ("", "")
    r = run["result"]
    d, bw = r["contexts"][0]["decode"], r["bandwidth"]
    last = r["contexts"][-1]
    s = d["shares"]
    short = (
        f"## Speed (PyTorch)\nDecode {d['tokens_per_s']:.1f} tok/s; MLP {_pct(s['mlp'])}, "
        f"attention {_pct(s['attention'])}, output head {_pct(s['lm_head'])} of decode time. "
        f"Decode is {bw['times_above_floor']:.2f}x the memory-bandwidth floor."
    )
    times = [x["est_ms"] for x in d["layers"]]
    full = (
        short
        + (
            f" Prefill {last['prefill']['tokens_per_s']:.0f} tok/s at {last['context']} tokens. "
            f"Per-layer decode time {min(times):.2f}-{max(times):.2f} ms (near uniform). "
            f"Each token reads {bw['weights_read_per_token_gb']:.2f} GB of weights."
        )
        + _runtimes(data)
    )
    return full, short


def _layers(data: dict) -> tuple[str, str]:
    """One row per layer joining redundancy, quantization sensitivity and activity."""
    profile, quant = data["latest"].get("profile"), data["latest"].get("quant")
    red = (profile or {}).get("result", {}).get("redundancy")
    if not red and not quant:
        return "", ""
    n = len(data["anatomy"]["layers"])
    skip = {x["layer"]: x for x in (red or {}).get("layers", [])}
    q = {x["layer"]: x["kl"] for x in (quant or {}).get("result", {}).get("layers", [])}
    act = {
        x["layer"]: x for x in (profile or {}).get("result", {}).get("all", {}).get("layers", [])
    }

    def row(i: int) -> str:
        s, a = skip.get(i), act.get(i)
        return (
            (f"{i}|{s['drop_kl']:.2f}|{_pct(s['drop_top1_agreement'])}" if s else f"{i}|-|-")
            + (f"|{q[i]:.3f}" if i in q else "|-")
            + (f"|{_pct(a['mean_firing_rate'])}|{a['in_out_cosine']:.2f}" if a else "|-|-")
        )

    header = (
        "## Layers\nskipKL = prediction shift when the layer is skipped; top1 = top prediction "
        "unchanged when skipped; quantKL = shift when its weights go to "
        f"{(quant or {}).get('result', {}).get('settings', {}).get('probe_bits', 4)} bits; "
        "fire = MLP neurons firing; cos = output-to-input similarity.\n"
        "layer|skipKL|top1|quantKL|fire|cos\n"
    )
    full = header + "\n".join(row(i) for i in range(n))
    picks: list[int] = []
    if skip:
        by = sorted(skip, key=lambda i: skip[i]["drop_kl"])
        picks += by[:4] + by[-3:]
    if q:
        picks += sorted(q, key=lambda i: -q[i])[:3]
    short = header + "\n".join(row(i) for i in sorted(set(picks))) + "\n(other layers omitted)"
    return full, short


def _activations(data: dict) -> tuple[str, str]:
    run = data["latest"].get("profile")
    if not run:
        return "", ""
    r = run["result"]
    lines = [f"## Activations ({r['run']['prompts']} prompts, {r['all']['tokens']} tokens)"]
    for name, cmp in r["comparison"].items():
        tops = ", ".join(f"L{t['layer']}.{t['neuron']}" for t in cmp["top_specific"][:3])
        lines.append(
            f"- {name}: {_pct(cmp['specific_fraction'], 2)} of neurons specific, "
            f"{_pct(cmp['unused_fraction'], 2)} idle here but used elsewhere; most specific {tops}"
        )
    short = (
        lines[0]
        + "\n"
        + "; ".join(
            f"{n}: {_pct(c['unused_fraction'], 2)} idle-only" for n, c in r["comparison"].items()
        )
    )
    return "\n".join(lines), short


def _quant(data: dict) -> tuple[str, str]:
    run = data["latest"].get("quant")
    if not run:
        return "", ""
    r = run["result"]
    lines = ["## Quantization (simulated in PyTorch unless marked GGUF)"]
    for plan in r["plans"]:
        u, m = plan["uniform"], plan["mixed"]
        lines.append(
            f"- {plan['target_bits']}-bit size: uniform KL {u['kl']:.4f}, mixed-precision plan KL "
            f"{m['kl']:.4f} (top1 {_pct(u['top1_agreement'])} vs {_pct(m['top1_agreement'])})"
        )
    for extra in run["extras"].values():
        for c in extra.get("comparisons", []):
            b, m = c["baseline"], c["mixed"]
            lines.append(
                f"- GGUF {c['type']} ({b['file_bytes'] / 2**30:.2f} GB): KL {b['kl']:.4f}; mixed plan "
                f"at the same size: KL {m['kl']:.4f}"
            )
    short = "\n".join(lines)
    ranked = sorted(r["tensors"], key=lambda t: -t["kl"])[:6]
    lines.append(
        "Most sensitive matrices: "
        + ", ".join(
            (f"L{t['layer']}.{t['kind']}" if t["layer"] is not None else "embeddings")
            + f" ({t['kl']:.3f})"
            for t in ranked
        )
    )
    return "\n".join(lines), short


def _loops(data: dict) -> tuple[str, str]:
    run = data["latest"].get("loops")
    if not run:
        return "", ""
    r = run["result"]
    s = r["summary"]
    short = (
        f"## Loops (greedy decoding, {s['prompts']} prompts)\n"
        f"{s['looped']} looped ({_pct(s['loop_rate'])}), {s['ended_normally']} ended normally."
    )
    lines = [short]
    a = r["analysis"]
    if a:
        lines.append(
            f"One loop examined: starts at generated token {a['onset']}, period {a['period']}. "
            "Copy heads: "
            + ", ".join(
                f"L{h['layer']}.{h['head']} ({_pct(h['attention'])})" for h in a["copy_heads"][:5]
            )
            + ". Writers of the repeated token: "
            + ", ".join(f"L{w['layer']} {w['kind']}" for w in a["writers"][:4])
            + "."
        )
        d = a["direction"]
        if d:
            lines.append(
                f"Loop direction clearest at layer {d['layer']} "
                f"({d['separation_sigma']:.0f} sigma); available as {LOOP_DIRECTION_ALIAS}."
            )
    for t in (r["causal"] or {}).get("tests", []):
        n = r["causal"]["looping_prompts_tested"]
        lines.append(
            f"- Run-time test, {t['intervention']}: {t['still_looping']}/{n} still loop; "
            f"{_pct(t['healthy_tokens_changed'])} of healthy output changed"
        )
    return "\n".join(lines), short


def _edits(data: dict) -> tuple[str, str]:
    if not data["edits"]:
        return "", ""
    base = data["edits"][0]["result"]["stages"]["base"]
    tasks = data["edits"][0]["result"]["tasks"]

    def scores(stage: dict) -> str:
        acc = ", ".join(
            f"{t} {_pct(stage['tasks'][t]['accuracy'])}" for t in tasks if t in stage["tasks"]
        )
        h = stage["health"]
        return (
            f"{acc}; top1 vs base {_pct(stage['drift']['top1_agreement'])}; "
            f"loops {h['looped']}/{h['prompts']}"
        )

    lines = ["## Edits already tried (accuracy on held-out questions)", f"- base: {scores(base)}"]
    for run in data["edits"][:6]:
        r = run["result"]
        last = list(r["stages"].values())[-1]
        cmds = "; ".join(
            re.sub(r'vector="[^"]*"|vector=\S+', f"vector={LOOP_DIRECTION_ALIAS}", c)
            for c in r["commands"]
        )
        healed = " + heal" if r["healing"] else ""
        removed = 1 - last["size"]["params"] / base["size"]["params"]
        lines.append(f"- {cmds}{healed}: {scores(last)}; params -{_pct(removed)}")
    short = "\n".join(lines[:4])
    return "\n".join(lines), short


def _findings(data: dict) -> tuple[str, str]:
    """The few things most worth acting on, each derived from a measurement."""
    out = []
    latest = data["latest"]
    if "profile" in latest:
        r = latest["profile"]["result"]
        worst = max((c["unused_fraction"] for c in r["comparison"].values()), default=0.0)
        out.append(
            f"At most {_pct(worst, 2)} of MLP neurons are idle for one prompt set only, so "
            "task-specific neuron pruning has almost nothing to remove."
        )
        if r["redundancy"]:
            layers = r["redundancy"]["layers"]
            best = min(layers, key=lambda x: x["drop_kl"])
            out.append(
                f"No layer is free to drop: the least harmful (layer {best['layer']}) still "
                f"changes the top prediction on {_pct(1 - best['drop_top1_agreement'])} of positions. "
                f"Least harmful: {', '.join(str(i) for i in r['redundancy']['most_removable'])}."
            )
    if "timing" in latest:
        bw = latest["timing"]["result"]["bandwidth"]
        if bw["times_above_floor"] < 1.5:
            out.append(
                "Decode speed is close to the memory-bandwidth limit, so a smaller "
                "(more quantized) model should decode faster roughly in proportion to its size."
            )
    if "quant" in latest:
        for extra in latest["quant"]["extras"].values():
            for c in extra.get("comparisons", []):
                b, m = c["baseline"], c["mixed"]
                if m["kl"] < b["kl"]:
                    out.append(
                        f"Real GGUF: the mixed-precision plan beats {c['type']} at the same size "
                        f"(KL {m['kl']:.4f} vs {b['kl']:.4f})."
                    )
    if "loops" in latest:
        s = latest["loops"]["result"]["summary"]
        if s["loop_rate"] > 0.3:
            out.append(
                f"{_pct(s['loop_rate'])} of prompts end in a repetition loop under greedy decoding."
            )
    if not out:
        return "", ""
    text = "## Key findings\n" + "\n".join(f"- {line}" for line in out)
    return text, text


def _reference(_data: dict) -> tuple[str, str]:
    text = "## What can be done\n" + COMMAND_HELP
    return text, text


# (name, builder, must keep). Order is priority: later sections give way first.
SECTIONS = [
    ("model", _model, True),
    ("findings", _findings, True),
    ("reference", _reference, True),
    ("edits", _edits, False),
    ("loops", _loops, False),
    ("quantization", _quant, False),
    ("layers", _layers, False),
    ("timing", _timing, False),
    ("activations", _activations, False),
]
ORDER = [
    "model",
    "findings",
    "timing",
    "layers",
    "activations",
    "quantization",
    "loops",
    "edits",
    "reference",
]


def build_report(
    model: str, budget_tokens: int = DEFAULT_BUDGET, home: str | Path | None = None
) -> dict[str, Any]:
    """Build the report text within `budget_tokens`. Returns the text and what was cut."""
    data = collect(model, home)
    built = {name: fn(data) for name, fn, _ in SECTIONS}
    state = {name: "full" if built[name][0] else "empty" for name, _, _ in SECTIONS}

    def render() -> str:
        parts = []
        for name in ORDER:
            if state[name] == "full":
                parts.append(built[name][0])
            elif state[name] == "short":
                parts.append(built[name][1])
        return "\n\n".join(parts)

    # Give way from the lowest priority up: shorten everything first, then drop.
    optional = [name for name, _, must in reversed(SECTIONS) if not must]
    for step in ("short", "dropped"):
        for name in optional:
            if estimate_tokens(render()) <= budget_tokens:
                break
            if state[name] in ("full", "short") and state[name] != step:
                if step == "short" and built[name][1] == built[name][0]:
                    continue
                state[name] = step
    text = render()

    aliases = {}
    direction = ((data["latest"].get("loops") or {}).get("result", {}).get("analysis") or {}).get(
        "direction"
    )
    if direction and direction.get("file"):
        aliases[LOOP_DIRECTION_ALIAS] = direction["file"]
    return {
        "model": model_label(data["anatomy"]["path"]),
        "text": text,
        "tokens_estimate": estimate_tokens(text),
        "budget_tokens": budget_tokens,
        "fits": estimate_tokens(text) <= budget_tokens,
        "sections": [
            {
                "name": name,
                "state": state[name],
                "tokens": estimate_tokens(
                    built[name][0]
                    if state[name] == "full"
                    else built[name][1]
                    if state[name] == "short"
                    else ""
                ),
            }
            for name in ORDER
        ],
        "sources": {
            t: {"id": e["id"], "started_at": e["started_at"]} for t, e in data["latest"].items()
        },
        "missing": [t for t in ("timing", "profile", "quant", "loops") if t not in data["latest"]],
        "aliases": aliases,  # resolved locally; never part of `text`
    }


def resolve_aliases(commands: list[str], aliases: dict[str, str]) -> list[str]:
    """Replace `@alias` tokens with the local file they stand for."""
    out = []
    for command in commands:
        for alias, target in aliases.items():
            command = command.replace(f"vector={alias}", f'vector="{target}"')
        if "@" in command:
            raise ValueError(
                f"Unknown alias in '{command}'. Known: {', '.join(aliases) or 'none'}."
            )
        out.append(command)
    return out
