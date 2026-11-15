# LLM X-Ray

**See what a local language model does inside, on your own machine.**

LLM X-Ray loads a model you already have (Llama or Qwen, up to about 8B
parameters) and measures it: which layers take the time, which neurons fire for
which kind of prompt, which layers it could do without, which weights tolerate
fewer bits, and why it sometimes repeats itself. You can then change the model's
weights and see exactly what each change costs, measured against the original.

Everything runs locally. Nothing is uploaded, and no model is ever downloaded.

> **Status: 0.9.0, release candidate.** The engine is tested on macOS, Windows
> and Linux. The desktop app has been installed and run on macOS (Apple
> Silicon) only, and installers are not yet signed. See
> [What works where](#what-works-where).

## What it tells you

A few real findings from Qwen2.5-1.5B on a MacBook (Apple M5, 24 GB):

- **Decode speed is limited by memory, not compute.** Each generated token
  reads 3.1 GB of weights, and the model runs within 15% of the machine's
  memory-bandwidth limit. So a smaller file decodes faster: 40 tokens/s at
  16-bit, 107 at 4-bit in llama.cpp.
- **Task-specific pruning has almost nothing to remove.** At most 0.06% of MLP
  neurons are idle for one kind of prompt (coding, chat, math, instructions)
  but used for another.
- **No layer is free to drop.** Even the least important one changes the
  model's top prediction on about a quarter of positions when skipped.
- **Smarter quantization beats uniform at the same size.** Giving sensitive
  weight matrices more bits and others fewer produced a real GGUF file with 23%
  less distortion than llama.cpp's own Q4_K_M, at the same size and speed.
- **Repetition loops have a visible signature.** Inside a loop every layer's
  state is 98–99% identical to one cycle earlier, and specific attention heads
  copy the previous cycle.

A full example report is in
[docs/samples/report-qwen2.5-1.5b.md](docs/samples/report-qwen2.5-1.5b.md).

## Features

| Area | What you get |
|---|---|
| **Anatomy** | Layers, parameters, and memory at each quantization level, read from the file headers in an instant. Whether the model fits your machine. |
| **Timing** | Time per layer, split into attention and MLP, for reading a prompt and for generating. Side by side with the speed you would get in llama.cpp and MLX. |
| **Activations** | How often every MLP neuron fires for each kind of prompt; neurons specific to one kind; what happens to predictions when each layer is skipped. |
| **Quantization** | How much each weight matrix suffers at 3 to 8 bits; a mixed-precision plan for a size budget, exported as a real GGUF and measured with llama.cpp. |
| **Logit lens** | At which layer each generated token is decided. |
| **Attribution** | Which layers, attention heads and neurons produced a given token. |
| **Loops** | Which prompts make the model repeat itself, what the loop looks like inside, and whether switching parts off stops it. |
| **Edits** | Switch off heads or neurons, drop layers, prune neurons, optionally "heal" with a short fine-tune, and score each step against the original on held-out questions. Edits never touch the file on disk. |
| **Report** | Everything measured, condensed into a few thousand tokens. An optional AI advisor can suggest what to try next from that report alone. |

## Install

The short version:

- **Desktop app:** download the installer for your system from the
  [Releases](../../releases) page and open it. The first launch sets itself up
  (about a 2 GB download).
- **Command line or Python:** with [uv](https://docs.astral.sh/uv/) installed,

  ```bash
  git clone https://github.com/amar683/llm-xray.git
  cd llm-xray/engine
  uv sync
  uv run xray system
  ```

**[INSTALL.md](INSTALL.md)** has the full instructions for each system,
the security prompts you will see, GPU support, optional extras and
troubleshooting.

## Quick start

You need a supported model on disk. If you use Hugging Face, any model you
have downloaded before is already in its cache; otherwise point X-Ray at a model
folder.

**Desktop app.** Open it, pick a model on the *Models* view, then follow
*Start here*, which lists the analyses in a sensible order.

**Command line.**

```bash
uv run xray inspect Qwen/Qwen2.5-1.5B           # anatomy, instant
uv run xray timing Qwen/Qwen2.5-1.5B            # where the time goes
uv run xray profile Qwen/Qwen2.5-1.5B           # neurons and layer redundancy
uv run xray loops Qwen/Qwen2.5-1.5B             # repetition loops
uv run xray report Qwen/Qwen2.5-1.5B            # everything, condensed
```

Every command is listed in [docs/cli.md](docs/cli.md).

**Python.**

```python
import xray

m = xray.Model("Qwen/Qwen2.5-1.5B")   # a local folder, or a repo id in the HF cache
m.inspect()                            # instant: read from file headers
run = m.timing(contexts=[128, 512])    # recorded, like every analysis
print(m.report()["text"])              # everything measured so far
```

A worked example is in [examples/quickstart.ipynb](examples/quickstart.ipynb).

## What works where

| | macOS (Apple Silicon) | Linux | Windows |
|---|---|---|---|
| Command line and Python | Tested | Tested (CI) | Tested (CI) |
| Desktop app | Installed and run | Builds; not yet run | Builds; not yet run |
| GPU acceleration | Apple GPU | NVIDIA (CUDA), untested | **None yet: CPU only** |
| llama.cpp benchmarks | Yes | Yes, with your own llama.cpp build | Yes, with your own llama.cpp build |
| MLX benchmarks | Yes | – | – |

On Windows the standard PyTorch package is CPU-only, so analyses run on the
processor. Small models (1–2B parameters) work but take several times longer.

Supported models: Llama 3.x and Qwen 2.x / 3.x text models. As a guide, 16 GB
of memory handles models up to about 3B parameters for analysis.

## Privacy

- Models are read from where they already are. They are never copied,
  uploaded or downloaded.
- The engine listens on 127.0.0.1 only and needs a fresh random token for every
  request.
- There is no telemetry. The only network use is the first-run setup and, if
  you choose to use it, the AI advisor.
- The AI advisor is off by default. It shows exactly what it would send before
  sending anything, and sends only the condensed report and your goal: no
  weights, prompts or file paths.

## How it works

The analysis engine is a Python package built on PyTorch and Hugging Face
transformers. It attaches hooks to the model to measure it while it runs, and
keeps running statistics rather than raw activations, so memory stays flat
however many prompts you use. Heavy jobs run in a separate worker process, so a
crash or an out-of-memory kill is reported as a failed run instead of taking
everything down. Every run is saved with a record of the model, hardware and
settings it used.

The desktop app (Tauri and React) carries the engine. On first launch it
installs Python and PyTorch into its own folder with
[uv](https://docs.astral.sh/uv/), then starts the engine as a local server and
talks to it over 127.0.0.1.

[docs/how-it-works.md](docs/how-it-works.md) explains the measurements and
their limits in more detail.

## Documentation

| Document | What it covers |
|---|---|
| [INSTALL.md](INSTALL.md) | Installing on each system, GPU support, extras, troubleshooting |
| [docs/user-guide.md](docs/user-guide.md) | Using the desktop app |
| [docs/cli.md](docs/cli.md) | Every command and option (generated from the code) |
| [docs/how-it-works.md](docs/how-it-works.md) | What each measurement means, and its limits |
| [examples/quickstart.ipynb](examples/quickstart.ipynb) | The Python API, step by step |
| [docs/release.md](docs/release.md) | What is verified where, and how releases are built |

## Development

```bash
cd engine
uv sync
uv run pytest                     # unit and integration tests
uv run python scripts/smoke.py    # every command, end to end, on a tiny model
```

To work on the interface in a browser, start the engine and the dev server in
two terminals:

```bash
cd engine && uv run xray serve --port 8791
```

```bash
cd app && npm install && npm run dev
```

Open http://localhost:1420 and enter the address and token that `xray serve`
printed. Building the desktop app itself needs Rust; see
[INSTALL.md](INSTALL.md#build-the-desktop-app-from-source).

| Folder | What it is |
|---|---|
| `engine/` | The Python engine, `xray` command and API server |
| `app/` | The desktop app: Tauri shell (Rust) and interface (React) |
| `docs/` | Documentation |
| `examples/` | Notebook examples |

## License

LLM X-Ray is licensed under the [Apache License 2.0](LICENSE).
