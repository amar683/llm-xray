# How it works

What each measurement means, how it is taken, and where it can mislead.

## The engine

The engine is a Python package built on PyTorch and Hugging Face transformers.
Each analysis attaches hooks to the model, runs prompts through it, and removes
the hooks afterwards. It keeps running statistics (sums, maxima, counts) rather
than raw activations, so memory stays flat however many prompts you use.

Heavy jobs run in a separate worker process. A crash or an out-of-memory kill is
then reported as a failed run instead of taking the caller down. Before loading
a model, the engine estimates the memory it needs and refuses if it will not
fit.

Every run is saved with a manifest: the model's fingerprint, the device and
precision, the library versions, the hardware and the settings. Runs live in a
SQLite database under `$XRAY_HOME` (default `.xray/` in the current folder),
with each run's full output in `runs/<run id>/result.json`.

## Timing

GPUs run asynchronously: the processor queues work for the next layer while the
GPU is still busy with the previous one, so a timer around a layer measures
little unless the two are synchronised, and synchronising slows everything
down. X-Ray therefore measures in separate passes:

| Pass | What it gives |
|---|---|
| Plain | The real speed, with nothing attached |
| Dispatch | How long the processor spends issuing work, unsynchronised |
| Blocks | Each layer, synchronised |
| Components | Each layer's attention and MLP, synchronised |

Each layer's reported time is its share of the synchronised total applied to
the real, plain speed, so the layers add up to the true total.

These are PyTorch timings. People run models in llama.cpp or MLX, which are
faster and behave differently; `xray bench` measures those on the same model so
the two can be read side by side.

**Limits.** Run-to-run drift is a few percent for totals and can reach 10–15%
for a single layer.

## Activations and redundancy

For every MLP neuron X-Ray records how often its activation exceeds a threshold
("fires"), separately for each prompt set. The four built-in sets (`coding`,
`chat`, `math`, `instruction`) have 50 prompts each; pass your own with
`--sets mine.jsonl,coding`, a `.jsonl` file with a `prompt` key per line or a
`.txt` file with one prompt per line.

A neuron is *specific* to a set when it fires at least 20 percentage points more
often there than in the other sets, and *idle* in a set when it fires on at most
1% of that set's tokens while firing on at least 5% elsewhere.

Redundancy is measured two ways: how similar each layer's output is to its input
(close to 1 means the layer changes little), and what happens to the model's
predictions when the layer is skipped entirely.

**Limits.** The built-in prompts come from templates, so some "specific"
neurons may react to repeated wording rather than the task. The skip test uses
16 prompts by default, so small differences between layers are noise.

## Quantization

`xray quant` rounds each weight matrix in PyTorch, one at a time, to 3 and 4
bits (one scale and offset per block of 32 values) and measures how far the
model's predictions move. A planner then gives each matrix a bit width to fit a
size budget, spending bits where they reduce the predicted damage most, and
checks the plan against uniform quantization of the same size on prompts it was
not tuned on.

That simulation is cruder than GGUF's quantizers: the absolute numbers are much
larger than real GGUF files show. Its *ranking* of sensitive matrices is what
transfers. `xray export` turns a plan into a real GGUF with llama.cpp and
measures it against full precision, next to llama.cpp's own uniform and `_K_M`
files of the same size:

```bash
uv sync --extra export
uv run xray export <quant run id> --llama-cpp /path/to/llama.cpp
```

It needs a llama.cpp checkout that is already built, which it only runs and
never modifies. The files go under `$XRAY_HOME/exports/` and take several GB;
`--discard` deletes them after measuring.

## Logit lens and attribution

The logit lens passes each layer's internal state straight to the output layer,
showing how sure the model would be of its final answer if it stopped there.

Attribution measures what produced the model's next token, using the gap
between its top choice and the runner-up. *Direct push* is each block's own
contribution to that gap; these add up to the gap exactly, which is a built-in
check. *Gap change if switched off* zeroes a block's output and re-runs the
model, so it also counts effects on later layers. Individual neurons are ranked
by gradient times activation.

Prompts are wrapped in the model's chat template by default. `--raw` feeds the
text as it is, which suits base models not trained on chat.

## Loops

`xray loops` generates from many prompts with greedy decoding and flags outputs
that end in a repeating cycle. For one looping case it then measures, per layer,
how similar each state is to the state one cycle earlier; which attention heads
attend to the token after the previous copy; which blocks push the repeated
token; and a "loop direction" (the average state while looping minus the
average before). Finally it switches off the top copy heads, or removes the
direction, regenerates, and counts how many prompts still loop and how much
normal output changes.

The loop direction is saved, and `ablate_direction` can remove it as a
permanent weight edit.

**Limits.** Greedy decoding makes small models loop far more often than normal
sampling does. The inside view comes from a single looping case.

## Edits

Edits are small text commands applied to a loaded copy of the model; the file on
disk is never modified.

| Command | What it does |
|---|---|
| `zero_head layer=N head=N` | Switch off one attention head |
| `scale_neuron layer=N neuron=N factor=X` | Scale one MLP neuron's output (0 switches it off) |
| `drop_layers layers=A,B` | Remove whole layers |
| `prune_neurons keep=0.9` | Keep the most important share of MLP neurons in every layer |
| `ablate_direction vector=FILE from_layer=N` | Stop layers writing along a saved direction |

Each edit is scored against the untouched model in three ways: accuracy on tasks
with exact answers (capitals, arithmetic, JSON output; only the held-out half
of each), drift in next-token predictions on ordinary prompts, and whether
generation loops. `--run-code` adds a small coding task whose answers are run
in a separate interpreter with no network, no file writes, CPU and time limits;
code with imports or other escape hatches is rejected without being run.

`--heal` runs a short low-rank fine-tune that teaches the edited model to
reproduce the original model's predictions. It needs full precision, so about
twice the memory. `--save` writes the result as a normal model folder, and
`xray gguf` converts that to GGUF.

`--name` saves the commands as a patch and `--on ID` stacks new commands on an
earlier patch. A patch stores only its own commands and a link to its parent, so
going back is choosing an earlier patch.

## Report and advisor

`xray report` condenses the latest run of each kind into a few thousand tokens.
Sections are shortened or dropped to fit `--budget`, and the output says which.
The text never contains file paths or your prompts. The token count is a
deliberately cautious estimate (2.2 characters per token).

The advisor is optional. `xray advise MODEL --goal "..."` prints exactly what
would be sent and sends nothing. With `--send` it sends the report and your goal
to the Claude API (`uv sync --extra advisor`, with `ANTHROPIC_API_KEY` set). The
advisor may ask for details, which are answered locally from saved runs, and
ends by proposing edit commands. Nothing is applied until you run them.
