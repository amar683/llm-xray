# LLM X-Ray user guide

LLM X-Ray measures what a local language model does inside: which layers are
slow, which neurons matter for your prompts, where it loops, and what can be
quantized or removed. Everything runs on your machine. Nothing is uploaded.

## First run

1. Open the app. The first launch sets up its own Python environment, which
   downloads Python and PyTorch (about 2 GB) and can take several minutes. The
   window shows what it is doing. Later launches take a few seconds.
2. Choose a model on the **Models** view. The app lists supported models already
   in your Hugging Face cache, or you can paste the path of a model folder (one
   with `config.json` and `.safetensors` files). It never downloads a model.
3. Follow **Start here**. It lists the analyses in a sensible order and ticks
   off the ones already done for the chosen model.

Supported model families: Llama 3.x and Qwen 2.x / 3.x, up to about 8B
parameters. As a guide, 16 GB of memory handles models up to about 3B.

## The views

| View | What it answers | Time for a small model |
|---|---|---|
| Anatomy | What the model is made of, and whether it fits this machine | instant |
| Timing | Where the time goes per layer; PyTorch next to llama.cpp and MLX | under a minute |
| Activations | Which neurons fire for which prompts; which layers are redundant | a few minutes |
| Quantization | Which weights tolerate fewer bits; a mixed-precision plan | about 5 minutes |
| Cost vs Value | Each layer's cost next to how much the model depends on it | instant |
| Logit Lens | At which layer each generated token is decided | seconds |
| Attribution | Which blocks, heads and neurons produced a token | under a minute |
| Loops | Which prompts make it repeat itself, and what drives the loop | several minutes |
| Edits | What a change to the weights costs, measured against the base | a minute or more |
| Report | Everything measured, in a few thousand tokens | instant |
| Runs | Every saved run; tick two of a kind to compare them | instant |

Analyses that load the model run one at a time. A second one waits in a queue,
and any of them can be cancelled.

## Things worth knowing

- **Numbers are PyTorch numbers unless marked otherwise.** PyTorch is used for
  analysis because it exposes everything. The Timing view's "PyTorch vs. real
  runtimes" panel measures llama.cpp and MLX so you can see real-world speed.
- **Quantization is simulated first.** The Quantization view rounds weights in
  PyTorch, which is cruder than GGUF. Use `xray export` to produce a real GGUF
  and measure it with llama.cpp before trusting a plan.
- **Edits never change the model on disk.** They are applied to a loaded copy.
  Going back is choosing an earlier patch, or none.
- **Loops are hunted with greedy decoding**, which makes small models repeat
  themselves far more than normal sampling does.
- **The AI advisor is optional and off by default.** It previews exactly what
  it would send, and sends only the report and your goal: no weights, prompts
  or file paths. It needs `ANTHROPIC_API_KEY` set where the engine runs.

## Where things are kept

The app keeps its data in the operating system's application-data folder for
`dev.llmxray.desktop`:

| Item | What it is |
|---|---|
| `venv/` | The Python environment (about 2 GB) |
| `xray/xray.db` | The list of runs, models and patches |
| `xray/runs/<id>/` | Each run's full result |
| `xray/exports/` | GGUF files, if you exported any (several GB) |
| `engine.log` | The engine's log, for troubleshooting |

Deleting the folder resets the app. Your models are never stored there.

## Command line and Python

Everything in the app is also available as the `xray` command
([reference](cli.md)) and as a Python package
([example notebook](../examples/quickstart.ipynb)).

## Troubleshooting

| Symptom | What to do |
|---|---|
| "The engine could not start" | The message shows the cause. `engine.log` in the data folder has more. |
| A run fails with "needs about N GB" | The model does not fit in free memory. Close other apps or use a smaller model. |
| A model is listed as "Not supported yet" | Only Llama and Qwen text models are supported in this version. |
| Timing differs between runs | Run-to-run drift of a few percent is normal; single layers can vary more. |
