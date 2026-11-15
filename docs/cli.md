# `xray` command reference

Generated from the commands by `engine/scripts/make_cli_docs.py`. Do not edit by hand.

| Command | What it does |
|---|---|
| [`xray advise`](#xray-advise) | Optional: ask Claude what edit to try next, using only the compact report. |
| [`xray attribute`](#xray-attribute) | Which blocks, heads and neurons produced the next token. |
| [`xray bench`](#xray-bench) | Measure real-world speed in llama.cpp and MLX, next to the PyTorch numbers. |
| [`xray compare`](#xray-compare) | Compare two runs of the same kind, metric by metric. |
| [`xray edit`](#xray-edit) | Apply edits to a model and measure the result against the untouched base. |
| [`xray export`](#xray-export) | Export a quant run's plan as a real GGUF and check it with llama.cpp. |
| [`xray fit`](#xray-fit) | Will this model fit on this machine? Does not load the weights. |
| [`xray gguf`](#xray-gguf) | Convert an edited model to GGUF and compare it with the base in llama.cpp. |
| [`xray inspect`](#xray-inspect) | Model anatomy from config and weight headers. Does not load the weights. |
| [`xray lens`](#xray-lens) | Logit lens: at which layer each generated token is decided. |
| [`xray loops`](#xray-loops) | Find prompts that make the model repeat itself, and what drives the loop. |
| [`xray neurons`](#xray-neurons) | Drill into a profile run's per-neuron data, read from disk. |
| [`xray patches`](#xray-patches) | List saved patches and what each one builds on. |
| [`xray profile`](#xray-profile) | Which neurons fire for which prompts, and which layers are redundant. |
| [`xray quant`](#xray-quant) | Which tensors are sensitive to quantization, and a mixed-precision plan. |
| [`xray query`](#xray-query) | Ask for one specific detail from the saved measurements. |
| [`xray report`](#xray-report) | A compact report of everything measured for a model, within a token budget. |
| [`xray runs`](#xray-runs) | List recorded runs, newest first. |
| [`xray serve`](#xray-serve) | Start the local API server used by the desktop app. |
| [`xray show`](#xray-show) | Show a recorded run again, read from disk. |
| [`xray system`](#xray-system) | Show hardware and available backends. |
| [`xray timing`](#xray-timing) | Where the time goes: per layer, attention vs MLP, prefill vs decode. |

## `xray advise`

Optional: ask Claude what edit to try next, using only the compact report.

Without --send this prints exactly what would be sent and sends nothing.
Credentials are read from the environment (ANTHROPIC_API_KEY).

```
xray advise
```

## `xray attribute`

Which blocks, heads and neurons produced the next token.

```
xray attribute
```

## `xray bench`

Measure real-world speed in llama.cpp and MLX, next to the PyTorch numbers.

```
xray bench
```

## `xray compare`

Compare two runs of the same kind, metric by metric.

```
xray compare
```

## `xray edit`

Apply edits to a model and measure the result against the untouched base.

Commands: zero_head layer=N head=N | scale_neuron layer=N neuron=N factor=X |
ablate_direction vector=FILE from_layer=N | drop_layers layers=A,B |
prune_neurons keep=0.9

With --name the commands are saved as a patch; --on stacks them on an earlier
patch. `xray edit MODEL --on ID` alone re-runs a saved patch.

```
xray edit
```

## `xray export`

Export a quant run's plan as a real GGUF and check it with llama.cpp.

```
xray export
```

## `xray fit`

Will this model fit on this machine? Does not load the weights.

```
xray fit
```

## `xray gguf`

Convert an edited model to GGUF and compare it with the base in llama.cpp.

```
xray gguf
```

## `xray inspect`

Model anatomy from config and weight headers. Does not load the weights.

```
xray inspect
```

## `xray lens`

Logit lens: at which layer each generated token is decided.

```
xray lens
```

## `xray loops`

Find prompts that make the model repeat itself, and what drives the loop.

```
xray loops
```

## `xray neurons`

Drill into a profile run's per-neuron data, read from disk.

```
xray neurons
```

## `xray patches`

List saved patches and what each one builds on.

```
xray patches
```

## `xray profile`

Which neurons fire for which prompts, and which layers are redundant.

```
xray profile
```

## `xray quant`

Which tensors are sensitive to quantization, and a mixed-precision plan.

```
xray quant
```

## `xray query`

Ask for one specific detail from the saved measurements.

```
xray query
```

## `xray report`

A compact report of everything measured for a model, within a token budget.

```
xray report
```

## `xray runs`

List recorded runs, newest first.

```
xray runs
```

## `xray serve`

Start the local API server used by the desktop app.

```
xray serve
```

## `xray show`

Show a recorded run again, read from disk.

```
xray show
```

## `xray system`

Show hardware and available backends.

```
xray system
```

## `xray timing`

Where the time goes: per layer, attention vs MLP, prefill vs decode.

```
xray timing
```
