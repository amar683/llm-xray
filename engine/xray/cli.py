"""`xray` command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="LLM X-Ray: measure what a local model does inside.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()

ModelArg = Annotated[str, typer.Argument(help="Model folder, or a repo id in the local HF cache.")]
JsonOpt = Annotated[Path | None, typer.Option("--json", help="Also write the result to this file.")]


def _gb(n: float) -> str:
    return f"{n / 2**30:.2f} GB"


def _write_json(path: Path | None, data: dict) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
        console.print(f"\nWrote [bold]{path}[/bold]")


def _fail(message: str) -> None:
    console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(1)


@app.command()
def system(as_json: Annotated[bool, typer.Option("--json-out", help="Print JSON.")] = False):
    """Show hardware and available backends."""
    from xray.core.device import system_info

    info = system_info()
    if as_json:
        console.print_json(json.dumps(info))
        return
    t = Table(title="System", show_header=False)
    for k, v in info.items():
        if isinstance(v, dict):
            v = ", ".join(k2 for k2, ok in v.items() if ok)
        t.add_row(k, str(v))
    console.print(t)


@app.command()
def inspect(model: ModelArg, json_path: JsonOpt = None):
    """Model anatomy from config and weight headers. Does not load the weights."""
    from xray.models.inspect import inspect_model

    try:
        r = inspect_model(model)
    except FileNotFoundError as e:
        _fail(str(e))

    a, tot = r["architecture"], r["totals"]
    t = Table(
        title=f"Anatomy: {Path(r['path']).parent.parent.name or r['path']}", show_header=False
    )
    t.add_row("architecture", str(a["model_type"]))
    t.add_row("layers", str(a["n_layers"]))
    t.add_row("hidden / MLP size", f"{a['hidden_size']} / {a['mlp_size']}")
    t.add_row("heads (KV heads)", f"{a['n_heads']} ({a['n_kv_heads']})")
    t.add_row("vocabulary", f"{a['vocab_size']:,}")
    t.add_row("max context", f"{a['max_context']:,}")
    t.add_row("tied embeddings", str(a["tied_embeddings"]))
    t.add_row("parameters", f"{tot['params']:,}")
    t.add_row("size on disk", _gb(tot["bytes_on_disk"]))
    t.add_row("stored dtypes", ", ".join(tot["stored_dtypes"]))
    console.print(t)

    c = Table(title="Parameters by component")
    c.add_column("component")
    c.add_column("params", justify="right")
    c.add_column("share", justify="right")
    for comp, n in sorted(r["by_component"].items(), key=lambda kv: -kv[1]):
        c.add_row(comp, f"{n:,}", f"{n / tot['params']:.1%}")
    console.print(c)

    layer0 = r["layers"][0]
    console.print(
        f"Each of the {len(r['layers'])} layers: {layer0['params']:,} params "
        f"(attention {layer0['components']['attention']['params']:,}, "
        f"MLP {layer0['components']['mlp']['params']:,})"
    )

    m = Table(title="Estimated memory")
    m.add_column("weights at")
    m.add_column("size", justify="right")
    for q, b in r["memory_estimates"]["weights_bytes"].items():
        m.add_row(q, _gb(b))
    for ctx, b in r["memory_estimates"]["kv_cache_bytes_fp16"].items():
        m.add_row(f"+ KV cache @ {int(ctx):,} ctx", _gb(b))
    console.print(m)
    _write_json(json_path, r)


@app.command()
def fit(
    model: ModelArg,
    context: Annotated[int, typer.Option(help="Context length to budget for.")] = 4096,
    json_path: JsonOpt = None,
):
    """Will this model fit on this machine? Does not load the weights."""
    from xray.core.device import system_info
    from xray.core.memory import fit_report
    from xray.models.inspect import inspect_model

    try:
        anatomy = inspect_model(model)
    except FileNotFoundError as e:
        _fail(str(e))
    r = fit_report(anatomy, system_info(), context)

    console.print(
        f"Device [bold]{r['device']}[/bold]: {_gb(r['available_bytes'])} available now, "
        f"{_gb(r['usable_bytes'])} usable at most."
    )
    colour = {"fits": "green", "fits if you free memory": "yellow", "too large": "red"}
    for title, key in (
        ("Analysing in X-Ray (PyTorch)", "analysis"),
        (f"Running day to day (GGUF, {context:,} context)", "running"),
    ):
        t = Table(title=title)
        t.add_column("option")
        t.add_column("needs", justify="right")
        t.add_column("verdict")
        for row in r[key]:
            c = colour[row["verdict"]]
            t.add_row(row["option"], _gb(row["need_bytes"]), f"[{c}]{row['verdict']}[/{c}]")
        console.print(t)
    rec = r["recommended"]
    console.print(
        f"Recommended: analyse as [bold]{rec['analysis'] or 'nothing fits'}[/bold], "
        f"run as [bold]{rec['running'] or 'nothing fits'}[/bold]. Estimates, not measurements."
    )
    _write_json(json_path, r)


@app.command()
def timing(
    model: ModelArg,
    contexts: Annotated[
        str, typer.Option(help="Context lengths, comma separated.")
    ] = "128,512,2048",
    prefill_repeats: Annotated[
        int, typer.Option(help="Minimum timed prefill passes per kind (short contexts get more).")
    ] = 5,
    decode_repeats: Annotated[int, typer.Option(help="Timed decode steps per kind.")] = 24,
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    dtype: Annotated[str, typer.Option(help="auto, float16, bfloat16 or float32.")] = "auto",
    force: Annotated[bool, typer.Option(help="Skip the memory check.")] = False,
    json_path: JsonOpt = None,
):
    """Where the time goes: per layer, attention vs MLP, prefill vs decode."""
    from xray.core.jobs import JobCancelled, JobError, run_job

    params = {
        "model": model,
        "contexts": [int(c) for c in contexts.split(",") if c.strip()],
        "prefill_repeats": prefill_repeats,
        "decode_repeats": decode_repeats,
        "device": device,
        "force": force,
    }
    if dtype != "auto":
        params["dtype"] = dtype

    try:
        with console.status("Starting...") as status:
            out = run_job("timing", params, on_event=lambda e: status.update(f"{e['stage']}..."))
    except JobCancelled as e:
        _fail(str(e))
    except JobError as e:
        _fail(f"{e} (run {e.run_id})")

    _render_timing(out)
    _write_json(json_path, out)


@app.command()
def runs(limit: Annotated[int, typer.Option(help="How many to show.")] = 20):
    """List recorded runs, newest first."""
    from xray.core import paths
    from xray.core.storage import Storage

    with Storage(paths.db_path(paths.home())) as db:
        rows = db.list_runs(limit)
    if not rows:
        console.print("No runs recorded yet.")
        return
    t = Table(title="Runs")
    t.add_column("id", no_wrap=True)
    for col in ("type", "status", "backend", "model"):
        t.add_column(col)
    colour = {"done": "green", "failed": "red", "cancelled": "yellow", "running": "cyan"}
    for row in rows:
        model_name = Path(row["model_path"]).parent.parent.name if row["model_path"] else ""
        c = colour.get(row["status"], "white")
        t.add_row(
            row["id"], row["type"], f"[{c}]{row['status']}[/{c}]",
            (row["backend"] or "").removeprefix("pytorch-"), model_name.removeprefix("models--"),
        )  # fmt: skip
    console.print(t)


def _render_timing(out: dict) -> None:
    r, b = out["result"], out["manifest"]["backend"]
    console.print(
        f"Run [bold]{out['run_id']}[/bold] on {b['device']} as {b['dtype']} "
        f"(PyTorch {b['torch']}). Times are medians."
    )
    if r["skipped_contexts"]:
        console.print(f"Skipped contexts beyond the model's limit: {r['skipped_contexts']}")
    if r["layer_devices"]["offloaded_layers"]:
        console.print(
            f"[yellow]Layers not on {r['layer_devices']['main']} (slow):[/yellow] "
            f"{r['layer_devices']['offloaded_layers']}"
        )

    sweep = Table(title="By context length (share of time)")
    for col in ("context", "phase", "ms", "tok/s", "attention", "MLP", "lm_head", "other"):
        sweep.add_column(col, justify="right", no_wrap=True)
    for ctx in r["contexts"]:
        for phase in ("prefill", "decode"):
            p, s = ctx[phase], ctx[phase]["shares"]
            other = s["layer_other"] + s["embed"] + s["final_norm"]
            sweep.add_row(
                f"{ctx['context']:,}" if phase == "prefill" else "",
                phase,
                f"{p['wall_ms']:.1f}",
                f"{p['tokens_per_s']:,.1f}",
                f"{s['attention']:.1%}",
                f"{s['mlp']:.1%}",
                f"{s['lm_head']:.1%}",
                f"{other:.1%}",
            )
    console.print(sweep)

    last = r["contexts"][-1]
    t = Table(title=f"Per layer at {last['context']:,} tokens of context (estimated ms)")
    for col in ("layer", "prefill", "attn", "mlp", "share", "decode", "attn", "mlp", "share"):
        t.add_column(col, justify="right", no_wrap=True)
    for p, d in zip(last["prefill"]["layers"], last["decode"]["layers"], strict=True):
        t.add_row(
            str(p["layer"]),
            f"{p['est_ms']:.2f}", f"{p['attn_ms']:.2f}", f"{p['mlp_ms']:.2f}", f"{p['share']:.1%}",
            f"{d['est_ms']:.2f}", f"{d['attn_ms']:.2f}", f"{d['mlp_ms']:.2f}", f"{d['share']:.1%}",
        )  # fmt: skip
    for name in ("embed", "final_norm", "lm_head"):
        p, d = last["prefill"]["other"][name], last["decode"]["other"][name]
        t.add_row(
            name, f"{p['est_ms']:.2f}", "", "", f"{p['share']:.1%}",
            f"{d['est_ms']:.2f}", "", "", f"{d['share']:.1%}",
        )  # fmt: skip
    console.print(t)

    bw = r["bandwidth"]
    console.print(
        f"Decode reads {bw['weights_read_per_token_gb']:.2f} GB of weights per token. "
        f"At this machine's measured {bw['peak_gb_per_s']:.0f} GB/s that takes at least "
        f"{bw['decode_floor_ms']:.1f} ms; actual is {bw['decode_actual_ms']:.1f} ms "
        f"({bw['times_above_floor']:.2f}x the floor)."
    )
    disp = r["contexts"][0]["decode"]["dispatch_fraction"]
    console.print(f"CPU-side dispatch accounts for {disp:.0%} of decode time.")
    _runtime_table(out["manifest"]["model"]["path"])


def _render_profile(out: dict) -> None:
    r, b = out["result"], out["manifest"]["backend"]
    run = r["run"]
    console.print(
        f"Run [bold]{out['run_id']}[/bold] on {b['device']} as {b['dtype']}: "
        f"{run['prompts']} prompts, {r['all']['tokens']:,} tokens "
        f"({run['generated_tokens']:,} generated) in {run['seconds']:.0f} s."
    )
    th = r["thresholds"]

    s = Table(title="By prompt set")
    for col in ("set", "prompts", "tokens", "firing", "specific", "unused here"):
        s.add_column(col, justify="right", no_wrap=True)
    for name, data in r["sets"].items():
        rate = sum(row["mean_firing_rate"] for row in data["layers"]) / len(data["layers"])
        cmp = r["comparison"].get(name)
        s.add_row(
            name, str(data["prompts"]), f"{data['tokens']:,}", f"{rate:.1%}",
            f"{cmp['specific_fraction']:.2%}" if cmp else "-",
            f"{cmp['unused_fraction']:.2%}" if cmp else "-",
        )  # fmt: skip
    console.print(s)
    console.print(
        f"firing: |activation| > {th['fire']}. specific: fires at least "
        f"{th['specific_gap']:.0%} more often than in the other sets. unused here: fires on "
        f"at most {th['cold_rate']:.0%} of this set's tokens but at least "
        f"{th['used_elsewhere_rate']:.0%} elsewhere."
    )

    red = {row["layer"]: row for row in (r["redundancy"] or {}).get("layers", [])}
    t = Table(title="Per layer (all sets pooled)")
    cols = ["layer", "cos", "norm", "firing", "hot", "cold"]
    if red:
        cols += ["skip KL", "top-1", "loss+"]
    for col in cols:
        t.add_column(col, justify="right", no_wrap=True)
    for row in r["all"]["layers"]:
        cells = [
            str(row["layer"]), f"{row['in_out_cosine']:.3f}", f"{row['residual_norm']:.1f}",
            f"{row['mean_firing_rate']:.1%}", f"{row['hot_fraction']:.1%}",
            f"{row['cold_fraction']:.1%}",
        ]  # fmt: skip
        if red:
            d = red[row["layer"]]
            cells += [
                f"{d['drop_kl']:.3f}", f"{d['drop_top1_agreement']:.1%}",
                f"{d['drop_nll_increase']:+.3f}",
            ]  # fmt: skip
        t.add_row(*cells)
    console.print(t)
    console.print(
        "cos: similarity of a layer's output to its input (1.0 = changes nothing). "
        "norm: size of the layer's output. skip KL / top-1 / loss+: with that one layer "
        "skipped, how far predictions move, how often the top prediction is unchanged, "
        "and the rise in loss."
    )
    if red:
        rd = r["redundancy"]
        console.print(
            f"Skipping one layer at a time on {rd['prompts']} prompts ({rd['positions']:,} "
            f"positions). Least harmful to skip: layers "
            f"{', '.join(str(i) for i in rd['most_removable'])}."
        )

    for name, cmp in r["comparison"].items():
        tops = ", ".join(
            f"L{n['layer']}.{n['neuron']} ({n['rate']:.0%} vs {n['rate_elsewhere']:.0%})"
            for n in cmp["top_specific"][:4]
        )
        console.print(f"Most [bold]{name}[/bold]-specific neurons: {tops}")

    m = r["memory"]
    console.print(
        f"Memory: peak {m['peak_gb']:.2f} GB, change over the last three quarters of the run "
        f"{m['growth_gb'] * 1000:+.0f} MB."
    )


def _load_run(run_id: str) -> tuple[dict, dict]:
    """A recorded run's database row and its result file."""
    from xray.core import paths
    from xray.core.storage import Storage

    with Storage(paths.db_path(paths.home())) as db:
        run = db.get_run(run_id)
        if run is None:
            _fail(f"No run with id '{run_id}'. See `xray runs`.")
        artifacts = db.artifacts(run_id)
    if run["status"] != "done":
        _fail(f"Run {run_id} is {run['status']}: {run['error'] or 'no result'}")
    return run, artifacts


@app.command()
def profile(
    model: ModelArg,
    sets: Annotated[
        str, typer.Option(help="Prompt sets, comma separated: built-in names or files.")
    ] = "coding,chat,math,instruction",
    limit: Annotated[int | None, typer.Option(help="Use only the first N prompts per set.")] = None,
    new_tokens: Annotated[int, typer.Option(help="Tokens to generate per prompt.")] = 16,
    ablation_prompts: Annotated[
        int, typer.Option(help="Prompts for the skip-a-layer test (0 to turn it off).")
    ] = 16,
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    dtype: Annotated[str, typer.Option(help="auto, float16, bfloat16 or float32.")] = "auto",
    force: Annotated[bool, typer.Option(help="Skip the memory check.")] = False,
    json_path: JsonOpt = None,
):
    """Which neurons fire for which prompts, and which layers are redundant."""
    from xray.core.jobs import JobCancelled, JobError, run_job

    params = {
        "model": model,
        "sets": [s.strip() for s in sets.split(",") if s.strip()],
        "limit": limit,
        "new_tokens": new_tokens,
        "ablation_prompts": ablation_prompts,
        "device": device,
        "force": force,
    }
    if dtype != "auto":
        params["dtype"] = dtype

    def show(e: dict) -> None:
        count = f" {e['done']}/{e['total']}" if e.get("total") else ""
        status.update(f"{e['stage']}{count}...")

    try:
        with console.status("Starting...") as status:
            out = run_job("profile", params, on_event=show)
    except JobCancelled as e:
        _fail(str(e))
    except JobError as e:
        _fail(f"{e} (run {e.run_id})")
    _render_profile(out)
    _write_json(json_path, out)


@app.command()
def show(run_id: Annotated[str, typer.Argument(help="Run id from `xray runs`.")]):
    """Show a recorded run again, read from disk."""
    run, artifacts = _load_run(run_id)
    data = json.loads(Path(artifacts["result"]).read_text())
    out = {"run_id": run_id, "manifest": data["manifest"], "result": data["result"]}
    renderers = {
        "timing": _render_timing, "profile": _render_profile, "quant": _render_quant,
        "lens": _render_lens, "attribution": _render_attribution, "loops": _render_loops,
        "edit": _render_edit, "bench": _render_bench,
    }  # fmt: skip
    if run["type"] not in renderers:
        _fail(f"Don't know how to show a '{run['type']}' run.")
    renderers[run["type"]](out)


@app.command()
def neurons(
    run_id: Annotated[str, typer.Argument(help="A profile run id from `xray runs`.")],
    layer: Annotated[int, typer.Option(help="Layer index.")],
    set_name: Annotated[
        str | None, typer.Option("--set", help="Rank neurons by how specific they are to this set.")
    ] = None,
    neuron: Annotated[int | None, typer.Option(help="Show one neuron in detail.")] = None,
    top: Annotated[int, typer.Option(help="How many neurons to list.")] = 10,
):
    """Drill into a profile run's per-neuron data, read from disk."""
    from xray.analyzers.profile import load_arrays, neuron_detail, top_neurons

    _, artifacts = _load_run(run_id)
    if "activations" not in artifacts:
        _fail(f"Run {run_id} has no per-neuron data (is it a profile run?).")
    by_set = load_arrays(Path(artifacts["activations"]))
    n_layers, n_neurons = next(iter(by_set.values()))["sum"].shape
    if not 0 <= layer < n_layers:
        _fail(f"Layer must be between 0 and {n_layers - 1}.")

    if neuron is not None:
        if not 0 <= neuron < n_neurons:
            _fail(f"Neuron must be between 0 and {n_neurons - 1}.")
        detail = neuron_detail(by_set, layer, neuron)
        t = Table(title=f"Layer {layer}, neuron {neuron}")
        for col in ("set", "tokens", "mean", "std", "max |act|", "fires >0.1", ">1.0"):
            t.add_column(col, justify="right", no_wrap=True)
        for name, d in detail.items():
            t.add_row(
                name, f"{d['tokens']:,}", f"{d['mean']:+.3f}", f"{d['std']:.3f}",
                f"{d['max_abs']:.2f}", f"{d['reach_rate']['0.1']:.1%}",
                f"{d['reach_rate']['1.0']:.1%}",
            )  # fmt: skip
        console.print(t)
        return

    set_name = set_name or next(iter(by_set))
    if set_name not in by_set:
        _fail(f"Run has no set '{set_name}'. Sets: {', '.join(by_set)}")
    t = Table(title=f"Layer {layer}: neurons most specific to '{set_name}'")
    for col in ("neuron", f"fires in {set_name}", "fires elsewhere"):
        t.add_column(col, justify="right", no_wrap=True)
    for n in top_neurons(by_set, set_name, layer, top):
        t.add_row(str(n["neuron"]), f"{n['rate']:.1%}", f"{n['rate_elsewhere']:.1%}")
    console.print(t)


def _render_quant(out: dict) -> None:
    r, b = out["result"], out["manifest"]["backend"]
    st = r["settings"]
    console.print(
        f"Run [bold]{out['run_id']}[/bold] on {b['device']} as {b['dtype']}. Simulated "
        f"quantization ({st['block']}-value blocks), {st['calibration_prompts']} calibration "
        f"and {st['heldout_prompts']} held-out prompts."
    )

    p = Table(title="Mixed precision vs. uniform at the same size (held-out prompts)")
    for col in ("size of", "plan", "GB", "bits/weight", "KL", "top-1 same", "loss+"):
        p.add_column(col, justify="right", no_wrap=True)
    for plan in r["plans"]:
        for label in ("uniform", "mixed"):
            v = plan[label]
            p.add_row(
                f"{plan['target_bits']}-bit" if label == "uniform" else "",
                label, f"{v['bytes'] / 2**30:.2f}", f"{v['avg_bpw']:.2f}", f"{v['kl']:.4f}",
                f"{v['top1_agreement']:.1%}", f"{v['nll_increase']:+.4f}",
            )  # fmt: skip
    console.print(p)
    for plan in r["plans"]:
        u, m = plan["uniform"], plan["mixed"]
        change = (m["kl"] / u["kl"] - 1) if u["kl"] else 0.0
        mix = ", ".join(
            f"{bits}-bit {n / sum(m['params_by_bits'].values()):.0%}"
            for bits, n in m["params_by_bits"].items() if n
        )  # fmt: skip
        if set(m["assignment"].values()) == {plan["target_bits"]}:
            console.print(
                f"At {plan['target_bits']}-bit size the plan is the same as uniform: "
                f"{plan['target_bits']} bits is the lowest width on offer, so no trade is possible."
            )
            continue
        verdict = "[green]better[/green]" if change < 0 else "[red]worse[/red]"
        console.print(
            f"At {plan['target_bits']}-bit size the mixed plan is {verdict} than uniform: "
            f"KL {change:+.0%}. Weights: {mix}."
        )

    t = Table(title=f"Most sensitive tensors (each quantized alone to {st['probe_bits']} bits)")
    for col in ("tensor", "params", "KL", "top-1 same", "outliers", "rounding err"):
        t.add_column(col, justify="right", no_wrap=True)
    ranked = sorted(r["tensors"], key=lambda x: -x["kl"])
    for row in ranked[:10]:
        name = f"L{row['layer']}.{row['kind']}" if row["layer"] is not None else row["kind"]
        t.add_row(
            name, f"{row['params'] / 1e6:.1f}M", f"{row['kl']:.5f}",
            f"{row['top1_agreement']:.1%}", f"{row['outlier_ratio']:.0f}x",
            f"{row['weighted_error'][str(st['probe_bits'])]:.2%}",
        )  # fmt: skip
    console.print(t)
    console.print(
        "outliers: largest weight divided by the typical weight. rounding err: share of the "
        "tensor's signal lost to rounding, weighted by how busy each input is."
    )
    by_layer = sorted(r["layers"], key=lambda x: -x["kl"])
    console.print(
        "Most sensitive layers: "
        + ", ".join(f"{row['layer']} ({row['kl']:.4f})" for row in by_layer[:5])
        + ". Least: "
        + ", ".join(f"{row['layer']} ({row['kl']:.4f})" for row in by_layer[-5:])
        + "."
    )


@app.command()
def quant(
    model: ModelArg,
    probe_bits: Annotated[
        int, typer.Option(help="Bit width for the one-tensor-at-a-time test.")
    ] = 4,
    target_bits: Annotated[
        str, typer.Option(help="Plan sizes to build, as uniform bit widths, comma separated.")
    ] = "4,5",
    calibration_prompts: Annotated[
        int, typer.Option(help="Prompts for measuring sensitivity.")
    ] = 16,
    heldout_prompts: Annotated[int, typer.Option(help="Separate prompts for checking plans.")] = 32,
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    dtype: Annotated[str, typer.Option(help="auto, float16, bfloat16 or float32.")] = "auto",
    force: Annotated[bool, typer.Option(help="Skip the memory check.")] = False,
    json_path: JsonOpt = None,
):
    """Which tensors are sensitive to quantization, and a mixed-precision plan."""
    from xray.core.jobs import JobCancelled, JobError, run_job

    params = {
        "model": model,
        "probe_bits": probe_bits,
        "target_bits": [int(b) for b in target_bits.split(",") if b.strip()],
        "calibration_prompts": calibration_prompts,
        "heldout_prompts": heldout_prompts,
        "device": device,
        "force": force,
    }
    if dtype != "auto":
        params["dtype"] = dtype

    def show(e: dict) -> None:
        count = f" {e['done']}/{e['total']}" if e.get("total") else ""
        status.update(f"{e['stage']}{count}...")

    try:
        with console.status("Starting...") as status:
            out = run_job("quant", params, on_event=show)
    except JobCancelled as e:
        _fail(str(e))
    except JobError as e:
        _fail(f"{e} (run {e.run_id})")
    _render_quant(out)
    _write_json(json_path, out)


@app.command()
def export(
    run_id: Annotated[str, typer.Argument(help="A quant run id from `xray runs`.")],
    llama_cpp: Annotated[
        Path | None, typer.Option(envvar="XRAY_LLAMA_CPP", help="A built llama.cpp folder.")
    ] = None,
    target_bits: Annotated[int, typer.Option(help="Size to match: 3, 4, 5, 6 or 8 bits.")] = 4,
    context: Annotated[int, typer.Option(help="Context length for the measurement.")] = 128,
    out: Annotated[Path | None, typer.Option(help="Where to write the GGUF files.")] = None,
    discard: Annotated[bool, typer.Option(help="Delete the GGUF files after measuring.")] = False,
    json_path: JsonOpt = None,
):
    """Export a quant run's plan as a real GGUF and check it with llama.cpp."""
    from xray.backends.llamacpp import LlamaCppError, export_and_check, find_tools, has_space
    from xray.core import paths

    run, artifacts = _load_run(run_id)
    if run["type"] != "quant":
        _fail(f"Run {run_id} is a {run['type']} run; export needs a quant run.")
    data = json.loads(Path(artifacts["result"]).read_text())
    model_dir = Path(data["manifest"]["model"]["path"])
    out_dir = out or paths.home() / "exports" / data["manifest"]["model"]["fingerprint"]
    model_bytes = int(data["result"]["full_precision_bytes"])

    try:
        tools = find_tools(llama_cpp)
        if not has_space(out_dir, model_bytes):
            _fail(f"Not enough free disk space in {out_dir} for the export.")
        with console.status("Starting...") as status:
            result = export_and_check(
                data["result"], data["manifest"]["params"], model_dir, tools, out_dir,
                target_bits=target_bits, context=context, keep_files=not discard,
                progress=lambda stage: status.update(f"{stage}..."),
            )  # fmt: skip
    except LlamaCppError as e:
        _fail(str(e))

    t = Table(title="Real GGUF files, measured by llama.cpp against full precision")
    for col in ("same size as", "file", "GB", "bpw", "KL", "±", "top-1 same"):
        t.add_column(col, justify="right", no_wrap=True)
    for c in result["comparisons"]:
        for side, name in (("baseline", c["type"]), ("mixed", "mixed plan")):
            v = c[side]
            t.add_row(
                c["type"] if side == "baseline" else "", name,
                f"{v['file_bytes'] / 2**30:.3f}", f"{v['bpw']:.2f}",
                f"{v['kl']:.4f}", f"{v['kl_err']:.4f}", f"{v['top1_agreement']:.1%}",
            )  # fmt: skip
    console.print(t)
    for c in result["comparisons"]:
        b, m = c["baseline"], c["mixed"]
        change = m["kl"] / b["kl"] - 1
        clear = abs(m["kl"] - b["kl"]) > 2 * (m["kl_err"] ** 2 + b["kl_err"] ** 2) ** 0.5
        verdict = "[green]better[/green]" if change < 0 else "[red]worse[/red]"
        note = "" if clear else " (within measurement noise)"
        size = "smaller" if m["file_bytes"] <= b["file_bytes"] else "LARGER"
        console.print(
            f"Against {c['label']}: the mixed plan is {verdict}, KL {change:+.0%}{note}; "
            f"its file is {abs(m['file_bytes'] - b['file_bytes']) / 2**20:.1f} MB {size}."
        )
    if result["files_kept"]:
        console.print(f"Files are in {result['out_dir']}")
    export_file = Path(artifacts["result"]).parent / f"export-{target_bits}bit.json"
    export_file.write_text(json.dumps(result, indent=2))
    _write_json(json_path, result)


def _run_and_render(job_type: str, params: dict, render, json_path: Path | None) -> None:
    """Run a job with a progress line, show its result, optionally save it."""
    from xray.core.jobs import JobCancelled, JobError, run_job

    def show(e: dict) -> None:
        count = f" {e['done']}/{e['total']}" if e.get("total") else ""
        status.update(f"{e['stage']}{count}...")

    try:
        with console.status("Starting...") as status:
            out = run_job(job_type, params, on_event=show)
    except JobCancelled as e:
        _fail(str(e))
    except JobError as e:
        _fail(f"{e} (run {e.run_id})")
    render(out)
    _write_json(json_path, out)


def _shade(p: float) -> str:
    """A probability as a block character plus two digits, for the lens grid."""
    blocks = " ░▒▓█"
    return f"{blocks[min(4, int(p * 5))]}{min(99, round(p * 100)):02d}"


def _render_lens(out: dict) -> None:
    r = out["result"]
    console.print(f"Run [bold]{out['run_id']}[/bold]. Generated: {r['generated_text']!r}")
    n = min(len(r["positions"]), 12)
    t = Table(title="Probability (%) of the final prediction, by layer")
    t.add_column("layer", justify="right")
    for token in r["predicted_tokens"][:n]:
        t.add_column(repr(token)[1:-1][:8], justify="right", no_wrap=True)
    for layer, row in enumerate(r["prob"]):
        t.add_row(str(layer), *[_shade(p) for p in row[:n]])
    t.add_row("answer at", *[str(a) for a in r["answer_layer"][:n]])
    console.print(t)
    console.print(
        "Each column is one generated token. A cell shows how sure the model would be of that "
        "token if it stopped at that layer. 'answer at' is the layer from which the top "
        f"prediction no longer changes (average {r['mean_answer_layer']:.1f})."
    )


def _render_attribution(out: dict) -> None:
    r = out["result"]
    console.print(
        f"Run [bold]{out['run_id']}[/bold]. Next token {r['target']!r} over runner-up "
        f"{r['runner_up']!r}: logit gap {r['logit_diff']:.2f} "
        f"(direct contributions add up to {r['direct_total']:.2f})."
    )
    c = Table(title="Blocks that matter most (change in the gap when switched off)")
    for col in ("block", "direct push", "gap change if off"):
        c.add_column(col, justify="right", no_wrap=True)
    blocks = [
        (f"L{row['layer']} {kind}", row[f"{kind}_direct"], row[f"{kind}_ablate"])
        for row in r["components"] for kind in ("attn", "mlp")
    ]  # fmt: skip
    for name, direct, ablate in sorted(blocks, key=lambda b: -abs(b[2]))[:10]:
        c.add_row(name, f"{direct:+.2f}", f"{ablate:+.2f}")
    console.print(c)

    h = Table(title="Attention heads with the largest direct push")
    for col in ("head", "direct push", "gap change if off"):
        h.add_column(col, justify="right", no_wrap=True)
    for row in r["heads"][:8]:
        h.add_row(
            f"L{row['layer']}.{row['head']}", f"{row['direct']:+.2f}", f"{row['ablate']:+.2f}"
        )
    console.print(h)

    n = Table(title="MLP neurons (gradient x activation)")
    for col in ("neuron", "attribution", "activation at last token"):
        n.add_column(col, justify="right", no_wrap=True)
    for row in r["neurons"][:10]:
        n.add_row(
            f"L{row['layer']}.{row['neuron']}",
            f"{row['attribution']:+.2f}",
            f"{row['activation']:+.2f}",
        )
    console.print(n)
    console.print(
        "direct push: the part's own contribution to the gap. gap change if off: what happens "
        "to the gap when the part's output is zeroed, including its effect on later layers. "
        "A negative change means the part was helping the predicted token."
    )


def _render_loops(out: dict) -> None:
    r = out["result"]
    s = r["summary"]
    console.print(
        f"Run [bold]{out['run_id']}[/bold]. {s['prompts']} prompts, greedy decoding, up to "
        f"{r['new_tokens']} tokens: [bold]{s['looped']} looped[/bold] ({s['loop_rate']:.0%}), "
        f"{s['ended_normally']} ended normally, {s['hit_limit_without_loop']} hit the limit "
        "without looping."
    )
    b = Table(title="By prompt set")
    for col in ("set", "prompts", "looped", "ended normally"):
        b.add_column(col, justify="right")
    for name, v in r["by_set"].items():
        b.add_row(
            name,
            str(v["prompts"]),
            f"{v['looped']} ({v['loop_rate']:.0%})",
            str(v["ended_normally"]),
        )
    console.print(b)
    a = r["analysis"]
    if not a:
        console.print("No loops found, so there is nothing to look inside.")
        return

    console.print(
        f"\nInside one loop ({a['set']} prompt): starts at generated token {a['onset']}, "
        f"repeats every {a['period']} token(s): {a['loop_text']!r}"
    )
    rec = Table(title="Similarity to the state one period earlier")
    for col in ("layer", "before loop", "in loop"):
        rec.add_column(col, justify="right")
    layers = range(0, len(a["recurrence_in_loop"]), max(1, len(a["recurrence_in_loop"]) // 10))
    for i in layers:
        rec.add_row(str(i), f"{a['recurrence_before'][i]:.2f}", f"{a['recurrence_in_loop'][i]:.2f}")
    console.print(rec)
    locked = a["locked_from_layer"]
    console.print(
        f"The loop is locked in (similarity 0.9 or more) from layer {locked}."
        if locked is not None
        else "No layer reaches 0.9 similarity."
    )
    console.print(
        "Copy heads (attention to the token after the previous copy): "
        + ", ".join(
            f"L{h['layer']}.{h['head']} ({h['attention']:.0%})" for h in a["copy_heads"][:6]
        )
    )
    console.print(
        "Blocks pushing the repeated token (in loop vs. before): "
        + ", ".join(
            f"L{w['layer']} {w['kind']} ({w['in_loop']:+.1f} vs. {w['before']:+.1f})"
            for w in a["writers"][:5]
        )
    )
    d = a["direction"]
    if d and d.get("file"):
        console.print(
            "To remove this direction as a weight edit, use the command\n"
            f'  ablate_direction vector="{d["file"]}" from_layer={d["layer"]}'
        )
    if d:
        warn = d["warning_tokens"]
        early = (
            f"it rises {warn} token(s) before the loop starts"
            if warn and warn > 0
            else "it gives no early warning here"
        )
        console.print(
            f"Loop direction: clearest at layer {d['layer']} "
            f"({d['separation_sigma']:.0f} standard deviations from normal); {early}."
        )

    c = r["causal"]
    t = Table(
        title=f"Does switching parts off stop the loops? "
        f"({c['looping_prompts_tested']} looping, {c['healthy_prompts_tested']} healthy prompts)"
    )
    for col in ("intervention", "still loop", "now end", "healthy changed", "healthy now loop"):
        t.add_column(col, justify="right")
    for test in c["tests"]:
        t.add_row(
            test["intervention"],
            f"{test['still_looping']}/{c['looping_prompts_tested']}",
            str(test["now_end_normally"]),
            f"{test['healthy_tokens_changed']:.0%} of tokens",
            f"{test['healthy_now_loop']}/{c['healthy_prompts_tested']}",
        )
    console.print(t)


PromptArg = Annotated[str, typer.Argument(help="The prompt to analyse.")]
RawOpt = Annotated[
    bool, typer.Option("--raw", help="Feed the text as is, without the chat template.")
]


@app.command()
def lens(
    model: ModelArg,
    prompt: PromptArg,
    new_tokens: Annotated[int, typer.Option(help="Tokens to generate and inspect.")] = 12,
    raw: RawOpt = False,
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    json_path: JsonOpt = None,
):
    """Logit lens: at which layer each generated token is decided."""
    params = {
        "model": model,
        "prompt": prompt,
        "new_tokens": new_tokens,
        "chat": not raw,
        "device": device,
    }
    _run_and_render("lens", params, _render_lens, json_path)


@app.command()
def attribute(
    model: ModelArg,
    prompt: PromptArg,
    raw: RawOpt = False,
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    json_path: JsonOpt = None,
):
    """Which blocks, heads and neurons produced the next token."""
    params = {"model": model, "prompt": prompt, "chat": not raw, "device": device}
    _run_and_render("attribution", params, _render_attribution, json_path)


@app.command()
def loops(
    model: ModelArg,
    sets: Annotated[
        str, typer.Option(help="Prompt sets, comma separated.")
    ] = "coding,chat,math,instruction",
    limit: Annotated[int, typer.Option(help="Prompts per set.")] = 10,
    new_tokens: Annotated[int, typer.Option(help="Tokens to generate per prompt.")] = 120,
    raw: RawOpt = False,
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    json_path: JsonOpt = None,
):
    """Find prompts that make the model repeat itself, and what drives the loop."""
    params = {
        "model": model, "sets": [s.strip() for s in sets.split(",") if s.strip()],
        "limit": limit, "new_tokens": new_tokens, "chat": not raw, "device": device,
    }  # fmt: skip
    _run_and_render("loops", params, _render_loops, json_path)


def _render_edit(out: dict) -> None:
    r, b = out["result"], out["manifest"]["backend"]
    console.print(f"Run [bold]{out['run_id']}[/bold] on {b['device']} as {b['dtype']}.")
    for line in r["applied"]:
        console.print(f"  - {line}")
    stages = r["stages"]
    base = stages["base"]

    t = Table(title="Each stage against the base model")
    t.add_column("")
    for name in stages:
        t.add_column(name, justify="right", no_wrap=True)

    def row(label: str, fn) -> None:
        t.add_row(label, *[fn(s) for s in stages.values()])

    row("parameters", lambda s: f"{s['size']['params'] / 1e9:.3f} B")
    row("size (fp16)", lambda s: _gb(s["size"]["fp16_bytes"]))
    row("layers / MLP width", lambda s: f"{s['size']['layers']} / {s['size']['mlp_width']}")
    for task in r["tasks"]:
        row(
            f"{task} accuracy",
            lambda s, task=task: (
                f"{s['tasks'][task]['accuracy']:.0%} ± {s['tasks'][task]['stderr']:.0%}"
            ),
        )
    row("drift from base (KL)", lambda s: f"{s['drift']['kl']:.4f}")
    row("top prediction unchanged", lambda s: f"{s['drift']['top1_agreement']:.1%}")
    row("loss change", lambda s: f"{s['drift']['nll_increase']:+.4f}")
    row(
        "looped / ended",
        lambda s: f"{s['health']['looped']} / {s['health']['ended']} of {s['health']['prompts']}",
    )
    console.print(t)

    last = list(stages.values())[-1]
    saved = 1 - last["size"]["params"] / base["size"]["params"]
    console.print(
        f"The final model has {saved:.1%} fewer parameters than the base. Accuracy is on the "
        f"held-out half of each task ({', '.join(str(base['tasks'][t]['n']) for t in r['tasks'])} "
        "questions), ± one standard error."
    )
    h = r["healing"]
    if h:
        console.print(
            f"Healing: {h['steps']} steps on {h['train_prompts']} prompts, rank {h['rank']} "
            f"({h['trained_params'] / 1e6:.1f}M trained weights); distance to the base on the "
            f"training prompts went from {h['loss_start']:.3f} to {h['loss_end']:.3f}."
        )
    if r.get("code_isolation"):
        console.print(f"Code task isolation: {r['code_isolation']}.")
    if r["saved"]:
        console.print(
            f"Saved to {r['saved']['path']} ({_gb(r['saved']['weights_bytes'])} of weights)."
        )


@app.command()
def edit(
    model: ModelArg,
    command: Annotated[
        list[str] | None,
        typer.Option("--command", "-c", help="An edit command. Repeat for several."),
    ] = None,
    on: Annotated[str | None, typer.Option(help="Build on this saved patch (its id).")] = None,
    name: Annotated[str | None, typer.Option(help="Save these commands as a named patch.")] = None,
    heal: Annotated[bool, typer.Option(help="Heal the edited model by short fine-tuning.")] = False,
    heal_steps: Annotated[int, typer.Option(help="Healing steps.")] = 150,
    save: Annotated[bool, typer.Option(help="Save the edited model as a model folder.")] = False,
    run_code: Annotated[
        bool,
        typer.Option(help="Also score the code task, which runs model-written code in a sandbox."),
    ] = False,
    device: Annotated[str, typer.Option(help="auto, mps, cuda or cpu.")] = "auto",
    force: Annotated[bool, typer.Option(help="Skip the memory check.")] = False,
    json_path: JsonOpt = None,
):
    """Apply edits to a model and measure the result against the untouched base.

    Commands: zero_head layer=N head=N | scale_neuron layer=N neuron=N factor=X |
    ablate_direction vector=FILE from_layer=N | drop_layers layers=A,B |
    prune_neurons keep=0.9

    With --name the commands are saved as a patch; --on stacks them on an earlier
    patch. `xray edit MODEL --on ID` alone re-runs a saved patch.
    """
    from xray.core import paths
    from xray.core.storage import Storage
    from xray.patch.commands import PatchError, parse_commands
    from xray.patch.history import create_patch

    commands = list(command or [])
    if not commands and not on:
        _fail("Give at least one --command, or --on PATCH_ID to re-run a saved patch.")
    try:
        parse_commands(commands)
        patch_id = on
        if name or (on and commands):
            patch = create_patch(model, name or "", commands, parent_id=on)
            patch_id, effective = patch["id"], patch["effective_commands"]
            console.print(
                f"Saved patch [bold]{patch_id}[/bold] ({patch['name']})"
                + (f", built on {on}." if on else ".")
            )
        elif on:
            with Storage(paths.db_path(paths.home())) as db:
                effective = db.effective_commands(on)
        else:
            effective = commands
    except (PatchError, KeyError, ValueError, FileNotFoundError) as e:
        _fail(str(e).strip("'\""))
    params = {
        "model": model, "commands": effective, "patch_id": patch_id, "save": save,
        "run_code": run_code, "device": device, "force": force,
        "heal": {"steps": heal_steps} if heal else None,
    }  # fmt: skip
    _run_and_render("edit", params, _render_edit, json_path)


@app.command()
def patches(
    model: Annotated[str | None, typer.Argument(help="Only this model's patches.")] = None,
    delete: Annotated[str | None, typer.Option(help="Delete the patch with this id.")] = None,
):
    """List saved patches and what each one builds on."""
    from xray.core import paths
    from xray.core.storage import Storage
    from xray.patch.history import list_patches

    if delete:
        try:
            with Storage(paths.db_path(paths.home())) as db:
                if db.get_patch(delete) is None:
                    _fail(f"No patch '{delete}'.")
                db.delete_patch(delete)
        except ValueError as e:
            _fail(str(e))
        console.print(f"Deleted patch {delete}.")
        return
    try:
        rows = list_patches(model)
    except FileNotFoundError as e:
        _fail(str(e))
    if not rows:
        console.print("No patches saved yet. Use `xray edit MODEL -c ... --name NAME`.")
        return
    t = Table(title="Patches")
    for col in ("id", "name", "builds on", "own commands", "total"):
        t.add_column(col)
    for row in rows:
        chain = " > ".join(p["name"] for p in row["chain"][:-1]) or "base model"
        t.add_row(
            row["id"], row["name"], chain, "\n".join(row["commands"]),
            str(len(row["effective_commands"])),
        )  # fmt: skip
    console.print(t)


def _fmt_metric(value: float | None, fmt: str, signed: bool = False) -> str:
    if value is None:
        return "-"
    sign = "+" if signed and value > 0 else ""
    if fmt.startswith("pct"):
        digits = 2 if fmt == "pct2" else 1
        return f"{sign}{value * 100:.{digits}f}{' pts' if signed else '%'}"
    if fmt == "int":
        return f"{sign}{value:,.0f}"
    return f"{sign}{value:,.{int(fmt[3:])}f}"


@app.command()
def compare(
    run_a: Annotated[str, typer.Argument(help="First run id.")],
    run_b: Annotated[str, typer.Argument(help="Second run id.")],
    json_path: JsonOpt = None,
):
    """Compare two runs of the same kind, metric by metric."""
    from xray.core.summary import compare_runs

    loaded = []
    for run_id in (run_a, run_b):
        run, artifacts = _load_run(run_id)
        loaded.append((run, json.loads(Path(artifacts["result"]).read_text())["result"]))
    (ra, a), (rb, b) = loaded
    try:
        rows = compare_runs(ra["type"], a, rb["type"], b)
    except ValueError as e:
        _fail(str(e))
    t = Table(title=f"{ra['type']} runs: A = {run_a}, B = {run_b}")
    for col in ("metric", "A", "B", "change", ""):
        t.add_column(col, justify="right" if col != "metric" else "left")
    colour = {"better": "green", "worse": "red"}
    for row in rows:
        verdict = row["verdict"] or ""
        mark = f"[{colour[verdict]}]{verdict}[/{colour[verdict]}]" if verdict in colour else verdict
        t.add_row(
            row["label"],
            _fmt_metric(row["a"], row["format"]),
            _fmt_metric(row["b"], row["format"]),
            _fmt_metric(row["change"], row["format"], signed=True),
            mark,
        )  # fmt: skip
    console.print(t)
    _write_json(json_path, {"a": run_a, "b": run_b, "type": ra["type"], "rows": rows})


@app.command()
def gguf(
    run_id: Annotated[str, typer.Argument(help="An edit run that was saved (`xray edit --save`).")],
    llama_cpp: Annotated[
        Path | None, typer.Option(envvar="XRAY_LLAMA_CPP", help="A built llama.cpp folder.")
    ] = None,
    gguf_type: Annotated[str, typer.Option("--type", help="GGUF quantization type.")] = "Q4_K_M",
    context: Annotated[int, typer.Option(help="Context length for the measurement.")] = 128,
    out: Annotated[Path | None, typer.Option(help="Where to write the GGUF files.")] = None,
    json_path: JsonOpt = None,
):
    """Convert an edited model to GGUF and compare it with the base in llama.cpp."""
    from xray.backends.llamacpp import LlamaCppError, export_edited, find_tools
    from xray.core import paths

    run, artifacts = _load_run(run_id)
    data = json.loads(Path(artifacts["result"]).read_text())
    saved = data["result"].get("saved") if run["type"] == "edit" else None
    if not saved:
        _fail(f"Run {run_id} has no saved model. Run `xray edit ... --save` first.")
    base_dir = Path(data["manifest"]["model"]["path"])
    out_dir = out or paths.home() / "exports" / data["manifest"]["model"]["fingerprint"]
    try:
        tools = find_tools(llama_cpp)
        with console.status("Starting...") as status:
            r = export_edited(
                tools, base_dir, Path(saved["path"]), out_dir, gguf_type, context,
                progress=lambda stage: status.update(f"{stage}..."),
            )  # fmt: skip
    except LlamaCppError as e:
        _fail(str(e))

    t = Table(title="GGUF files, measured by llama.cpp against the full-precision base")
    for col in ("model", "file", "GB", "KL", "±", "top-1 same"):
        t.add_column(col, justify="right", no_wrap=True)
    t.add_row("base", "F16", f"{r['base_f16_bytes'] / 2**30:.3f}", "0", "", "100%")
    for label, key, kind in (
        ("base", "base", r["type"]),
        ("edited", "edited_f16", "F16"),
        ("edited", "edited", r["type"]),
    ):  # fmt: skip
        v = r[key]
        t.add_row(
            label, kind, f"{v['file_bytes'] / 2**30:.3f}", f"{v['kl']:.4f}", f"{v['kl_err']:.4f}",
            f"{v['top1_agreement']:.1%}",
        )  # fmt: skip
    console.print(t)
    smaller = 1 - r["edited"]["file_bytes"] / r["base"]["file_bytes"]
    console.print(
        f"At {r['type']} the edited file is {smaller:.1%} smaller than the base file. "
        f"Files are in {r['out_dir']}"
    )
    (Path(artifacts["result"]).parent / f"gguf-{gguf_type.lower()}.json").write_text(
        json.dumps(r, indent=2)
    )
    _write_json(json_path, r)


@app.command()
def report(
    model: ModelArg,
    budget: Annotated[int, typer.Option(help="Token budget for the report.")] = 3000,
    json_path: JsonOpt = None,
):
    """A compact report of everything measured for a model, within a token budget."""
    from xray.report import build_report

    try:
        r = build_report(model, budget)
    except FileNotFoundError as e:
        _fail(str(e))
    console.print(r["text"], markup=False, highlight=False)
    cut = [
        f"{s['name']} ({s['state']})" for s in r["sections"] if s["state"] in ("short", "dropped")
    ]
    console.print(
        f"\n[bold]About {r['tokens_estimate']:,} tokens[/bold] of a {budget:,} budget "
        f"(a cautious estimate: 2.2 characters per token)."
        + (f" Shortened or left out: {', '.join(cut)}." if cut else " Nothing was cut.")
        + ("" if r["fits"] else " [red]Still over budget.[/red]")
    )
    if r["missing"]:
        console.print(f"No run yet for: {', '.join(r['missing'])}.")
    _write_json(json_path, {k: v for k, v in r.items() if k != "aliases"})


@app.command()
def query(
    model: ModelArg,
    text: Annotated[str, typer.Argument(help="e.g. 'layer 12' or 'neuron 12 6553'.")],
):
    """Ask for one specific detail from the saved measurements."""
    from xray.report.query import QueryError, run_query

    try:
        console.print(run_query(model, text), markup=False, highlight=False)
    except (QueryError, FileNotFoundError) as e:
        _fail(str(e))


@app.command()
def advise(
    model: ModelArg,
    goal: Annotated[str, typer.Option(help="What you want help with, in your own words.")],
    budget: Annotated[int, typer.Option(help="Token budget for the report that is sent.")] = 3000,
    send: Annotated[
        bool,
        typer.Option(
            help="Actually send the request to the Claude API. Without this, only a preview."
        ),
    ] = False,
    apply: Annotated[
        bool,
        typer.Option(help="After a proposal comes back, run it locally as an edit and measure it."),
    ] = False,
    json_path: JsonOpt = None,
):
    """Optional: ask Claude what edit to try next, using only the compact report.

    Without --send this prints exactly what would be sent and sends nothing.
    Credentials are read from the environment (ANTHROPIC_API_KEY).
    """
    from xray.report.advisor import AdvisorError, build_request, run_advisor

    try:
        request = build_request(model, goal, budget)
    except (AdvisorError, FileNotFoundError) as e:
        _fail(str(e))
    payload = request["payload"]
    if not send:
        console.print("[bold]Nothing has been sent.[/bold] This is what --send would send:\n")
        console.print(
            f"[bold]To:[/bold] the Claude API, model {payload['model']}, effort {payload['effort']}"
        )
        console.print("\n[bold]System prompt[/bold]")
        console.print(payload["system"], markup=False, highlight=False)
        console.print("\n[bold]Tools the advisor may call[/bold]")
        for tool in payload["tools"]:
            console.print(f"- {tool['name']}: {tool['description']}", markup=False, highlight=False)
        console.print("\n[bold]Message[/bold]")
        console.print(payload["messages"][0]["content"], markup=False, highlight=False)
        console.print(
            f"\nAbout {request['tokens_estimate']:,} tokens in total. No weights, prompts, file "
            "paths or host details are included. If the advisor asks drill-down queries, each "
            "answer (a few hundred tokens) is sent too. Add --send to send it."
        )
        return

    try:
        with console.status("Asking the advisor...") as status:
            result = run_advisor(model, goal, budget, on_step=lambda s: status.update(f"{s}..."))
    except AdvisorError as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001 - surface API and network errors plainly
        _fail(f"The request to the Claude API failed: {type(e).__name__}: {e}")

    for q in result["queries"]:
        console.print(f"[dim]Advisor asked:[/dim] {q['query']}", highlight=False)
    u = result["usage"]
    console.print(
        f"[dim]{result['rounds']} round(s), {u['input_tokens']:,} tokens in, "
        f"{u['output_tokens']:,} out.[/dim]"
    )
    if not result["commands"]:
        console.print("\n[bold]The advisor recommends no edit:[/bold]")
        console.print(result["message"], markup=False, highlight=False)
        _write_json(json_path, result)
        return
    console.print("\n[bold]Proposed edit[/bold]")
    for c in result["commands_as_proposed"]:
        console.print(f"  {c}", markup=False, highlight=False)
    console.print(f"\n[bold]Why:[/bold] {result['rationale']}", highlight=False)
    console.print(f"[bold]Expected:[/bold] {result['expected_effect']}", highlight=False)
    _write_json(json_path, result)
    if not apply:
        console.print("\nNot applied. Re-run with --apply, or pass these commands to `xray edit`.")
        return
    console.print("\nApplying locally and measuring...")
    params = {"model": model, "commands": result["commands"], "device": "auto"}
    _run_and_render("edit", params, _render_edit, None)


def _runtime_table(model_path: str, bench_result: dict | None = None) -> None:
    """PyTorch next to llama.cpp and MLX, from the latest timing and bench runs."""
    from xray.core.summary import runtime_rows
    from xray.report import collect

    latest = collect(model_path)["latest"]
    timing = latest["timing"]["result"] if "timing" in latest else None
    bench = bench_result or (latest["bench"]["result"] if "bench" in latest else None)
    if not bench:
        console.print(
            "No real-runtime numbers yet: `xray bench MODEL --llama-cpp FOLDER` measures "
            "llama.cpp and MLX."
        )
        return
    t = Table(title="PyTorch vs. real runtimes")
    t.add_column("runtime", no_wrap=True)
    t.add_column("variant")
    for col in ("GB", "prefill", "decode", "vs PyTorch"):
        t.add_column(col, justify="right", no_wrap=True)
    for row in runtime_rows(timing, bench):
        t.add_row(
            row["runtime"], row["variant"],
            f"{row['file_bytes'] / 2**30:.2f}" if row["file_bytes"] else "-",
            f"{row['prefill_tps']:,.0f}", f"{row['decode_tps']:.1f}",
            f"{row['vs_pytorch']:.2f}x" if row["vs_pytorch"] else "-",
        )  # fmt: skip
    console.print(t)
    console.print("prefill and decode are tokens per second; GB is the model file size.")
    if not timing:
        console.print("No PyTorch timing run yet for comparison (`xray timing MODEL`).")
    for s in bench.get("skipped", []):
        console.print(f"[yellow]{s['runtime']} was not measured:[/yellow] {s['reason']}")


def _render_bench(out: dict) -> None:
    r = out["result"]
    console.print(
        f"Run [bold]{out['run_id']}[/bold]. Prefill of {r['n_prompt']} tokens, then "
        f"{r['n_gen']} generated tokens, {r['repeats']} repeats."
    )
    _runtime_table(out["manifest"]["model"]["path"], r)


@app.command()
def bench(
    model: ModelArg,
    llama_cpp: Annotated[
        Path | None, typer.Option(envvar="XRAY_LLAMA_CPP", help="A built llama.cpp folder.")
    ] = None,
    types: Annotated[
        str, typer.Option(help="GGUF types to measure, comma separated.")
    ] = "F16,Q8_0,Q4_K_M",
    mlx: Annotated[bool, typer.Option(help="Also measure MLX (Apple Silicon).")] = True,
    out: Annotated[Path | None, typer.Option(help="Where GGUF files are kept.")] = None,
    json_path: JsonOpt = None,
):
    """Measure real-world speed in llama.cpp and MLX, next to the PyTorch numbers."""
    params = {
        "model": model, "types": [t.strip() for t in types.split(",") if t.strip()], "mlx": mlx,
        "llama_cpp": str(llama_cpp) if llama_cpp else None,
        "out": str(out.resolve()) if out else None,
    }  # fmt: skip
    _run_and_render("bench", params, _render_bench, json_path)


def _exit_with_parent(parent_pid: str | None) -> None:
    """Stop this server when the process that started it is gone.

    The desktop shell stops the engine on a normal quit. If the shell crashes
    or is force-quit it cannot, and the engine would otherwise run on unseen.
    """
    if not parent_pid or not parent_pid.isdigit():
        return
    import os
    import threading
    import time

    import psutil

    def watch() -> None:
        while psutil.pid_exists(int(parent_pid)):
            time.sleep(2)
        os._exit(0)

    threading.Thread(target=watch, name="xray-parent-watch", daemon=True).start()


@app.command()
def serve(
    port: Annotated[int, typer.Option(help="Port to listen on. 0 picks a free one.")] = 0,
    token: Annotated[
        str | None, typer.Option(envvar="XRAY_TOKEN", help="Access token. Default: random.")
    ] = None,
):
    """Start the local API server used by the desktop app."""
    import os
    import secrets
    import socket

    import uvicorn

    from xray.api.app import create_app

    # Open the port *before* announcing it. Whoever reads the "ready" line may
    # connect at once: connections then wait in the socket's queue until the
    # server accepts them, instead of being refused.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name != "nt":
        # Lets a port closed a moment ago be reused. (On Windows this option
        # would instead let two servers share a port, so it is left off there.)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(128)
    except OSError:
        sock.close()
        _fail(f"Port {port} is already in use. Pick another with --port, or use 0.")
    port = sock.getsockname()[1]
    token = token or secrets.token_urlsafe(24)
    # One machine-readable line for the desktop shell, which starts this process.
    print(json.dumps({"event": "ready", "port": port, "token": token}), flush=True)
    _exit_with_parent(os.environ.get("XRAY_PARENT_PID"))
    # XRAY_ACCESS_LOG=1 logs each request (to stderr), for diagnosing the desktop app.
    level = "info" if os.environ.get("XRAY_ACCESS_LOG") else "warning"
    config = uvicorn.Config(create_app(token=token), log_level=level)
    uvicorn.Server(config).run(sockets=[sock])


if __name__ == "__main__":
    app()
