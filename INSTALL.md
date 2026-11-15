# Installing LLM X-Ray

There are three ways to use LLM X-Ray. Pick one:

| Way | Best for | You need |
|---|---|---|
| [Desktop app](#desktop-app) | Exploring a model visually | Nothing preinstalled |
| [Command line and Python](#command-line-and-python) | Scripting, notebooks, servers | [uv](https://docs.astral.sh/uv/) |
| [Build the app from source](#build-the-desktop-app-from-source) | Changing the app | uv, Node.js and Rust |

Whichever you choose, you also need [a model on disk](#getting-a-model).

## Before you start

**Operating systems**

| | Status |
|---|---|
| macOS 12.3 or later on Apple Silicon (M1 or newer) | Fully tested, including the desktop app |
| Linux (x86-64) | Command line and Python tested; desktop app builds but has not been run |
| Windows 10 or 11 (x86-64) | Command line and Python tested; desktop app builds but has not been run |
| macOS on Intel | Not supported |

**Memory.** Analysis loads the model at 16-bit precision. As a guide:

| Model size | Memory needed (about) |
|---|---|
| 0.5–1.5B parameters | 8 GB |
| 3B parameters | 16 GB |
| 7–8B parameters | 32 GB |

The `xray fit` command, or the Anatomy view in the app, tells you whether a
particular model fits your machine before anything is loaded. Healing an edited
model needs about twice as much.

**Disk.** About 3 GB for the program and its Python packages, plus room for
your models. Exported GGUF files (optional) take several GB more.

**GPU.**

| System | What is used |
|---|---|
| macOS (Apple Silicon) | The Apple GPU, automatically |
| Linux with an NVIDIA GPU | CUDA, automatically (the standard PyTorch package for Linux includes it). Not yet tested on real hardware. |
| Windows | **The CPU only, for now.** The standard PyTorch package for Windows has no CUDA support. Small models (1–2B parameters) work but analyses take several times longer. |
| No GPU | The CPU, on any system |

## Desktop app

1. Open the project's [Releases](../../releases) page and download the file
   for your system from the latest release:

   | System | File |
   |---|---|
   | macOS | `LLM.X-Ray_<version>_aarch64.dmg` |
   | Windows | `LLM.X-Ray_<version>_x64-setup.exe` or `.msi` |
   | Linux | `.AppImage`, `.deb` or `.rpm` |

   If there is no release yet, [build the app from source](#build-the-desktop-app-from-source)
   or use the [command line](#command-line-and-python).

2. Install it the usual way for your system (see below for the security
   prompts you will see).

3. Open it. **The first launch takes several minutes**: the app downloads
   Python and PyTorch (about 2 GB) into its own folder and shows its progress.
   This happens once; later launches take a few seconds. It needs an internet
   connection only for this step.

4. Choose a model on the *Models* view, then follow *Start here*. The
   [user guide](docs/user-guide.md) explains each view.

### Security prompts

The installers are not yet signed with a developer certificate, so your system
will warn you the first time. This is expected.

**macOS.** Open the `.dmg` and drag *LLM X-Ray* to *Applications*. The first
time you open it, macOS says it "cannot be opened because the developer cannot
be verified". To allow it:

1. Open *System Settings > Privacy & Security*.
2. Scroll to the message about LLM X-Ray and click **Open Anyway**.
3. Confirm with **Open**.

If macOS instead says the app "is damaged and can't be opened", it was
quarantined during download. Remove the quarantine flag, then open it normally:

```bash
xattr -dr com.apple.quarantine "/Applications/LLM X-Ray.app"
```

**Windows.** SmartScreen shows "Windows protected your PC". Click **More
info**, then **Run anyway**.

**Linux.** Make the AppImage executable before running it:

```bash
chmod +x LLM.X-Ray_*.AppImage
```

The `.deb` and `.rpm` packages install with your package manager as usual.

### Where the app keeps its data

| System | Folder |
|---|---|
| macOS | `~/Library/Application Support/dev.llmxray.desktop/` |
| Windows | `%APPDATA%\dev.llmxray.desktop\` |
| Linux | `~/.local/share/dev.llmxray.desktop/` |

It holds the Python environment, the saved runs and `engine.log`. Your models
are never copied there.

### Uninstalling

Remove the app the usual way, then delete the data folder above to reclaim the
space (about 2 GB, more if you exported GGUF files).

## Command line and Python

This gives you the `xray` command and the `xray` Python package. It is tested
on macOS, Linux and Windows.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), which
   installs the right Python version and all packages for you:

   macOS and Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Windows (PowerShell):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Get the code and install the engine (about 2 GB of packages, mostly PyTorch):

   ```bash
   git clone https://github.com/amar683/llm-xray.git
   cd llm-xray/engine
   uv sync
   ```

3. Check that it works:

   ```bash
   uv run xray system
   ```

   This lists your hardware and the backends it found (`mps` on a Mac, `cuda`
   with an NVIDIA GPU on Linux, otherwise `cpu`).

4. Run the commands from the `engine` folder with `uv run xray ...`. To use
   `xray` directly from anywhere, activate the environment first:

   macOS and Linux:

   ```bash
   source .venv/bin/activate
   ```

   Windows (PowerShell):

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

**Where runs are saved.** In a `.xray` folder inside the directory you run the
command from. Set `XRAY_HOME` to keep them in one place instead.

**Using it from Python.** After `uv sync`, `import xray` works in the engine's
environment. For Jupyter, run `uv run --with jupyter jupyter lab` from the
`engine` folder and open `../examples/quickstart.ipynb`.

### Optional extras

Install any of these with `uv sync --extra NAME` (repeat `--extra` for
several):

| Extra | What it adds | Also needs |
|---|---|---|
| `export` | `xray export` and `xray gguf`: real GGUF files from a quantization plan or an edited model | A built [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout |
| `mlx` | MLX speed measurements in `xray bench` | Apple Silicon |
| `advisor` | `xray advise --send`: the optional AI advisor | An `ANTHROPIC_API_KEY` in your environment |

For the llama.cpp features, pass the folder with `--llama-cpp /path/to/llama.cpp`
or set `XRAY_LLAMA_CPP`. X-Ray only runs its tools (`llama-quantize`,
`llama-perplexity`, `llama-bench`) and never modifies the folder.

### Checking your installation

The smoke suite runs every command end to end on a tiny made-up model. It needs
no model of your own and no internet, and takes about a minute:

```bash
uv run python scripts/smoke.py
```

It ends with `23 checks passed` when everything works.

## Getting a model

X-Ray never downloads a model; it reads one you already have. Two ways to
point it at one:

- **A model in your Hugging Face cache**, by its name, for example
  `Qwen/Qwen2.5-1.5B`. Anything you have downloaded before with Hugging Face
  tools is already there. To download one now:

  ```bash
  uv run hf download Qwen/Qwen2.5-1.5B
  ```

  (Some models, such as Llama, require you to accept their license on the
  Hugging Face website and sign in with `uv run hf auth login` first.)

- **A model folder** anywhere on disk, by its path. It must contain
  `config.json` and `.safetensors` weight files.

Supported: Llama 3.x and Qwen 2.x / 3.x text models. GGUF files cannot be
analysed directly; use the original model folder.

## Build the desktop app from source

You need, in addition to [uv](#command-line-and-python):

| Tool | Get it from |
|---|---|
| Node.js 22 or later | [nodejs.org](https://nodejs.org) |
| Rust (stable) | [rustup.rs](https://rustup.rs) |
| Linux only: system libraries | `sudo apt install libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf` (Debian/Ubuntu) |
| Windows only: build tools | [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and WebView2 (included in Windows 11) |

Then:

```bash
cd llm-xray/engine
uv sync
cd ../app
npm install
npm run tauri build
```

The installer appears under `app/src-tauri/target/release/bundle/`. The build
copies your `uv` into the app, so whoever installs it needs nothing else.

To run the app without building an installer, use `npm run tauri dev`.

## Troubleshooting

| Problem | What to do |
|---|---|
| `xray: command not found` | Run it as `uv run xray ...` from the `engine` folder, or activate the environment (step 4 above). |
| "is not a local folder and is not in the local Hugging Face cache" | The model is not downloaded. See [Getting a model](#getting-a-model). |
| "This model needs about N GB to analyse" | It does not fit in free memory. Close other apps, or use a smaller model. `xray fit MODEL` shows the numbers. |
| "Unsupported model type" | Only Llama and Qwen text models are supported in this version. |
| Everything is slow on Windows | Expected for now: Windows runs on the CPU. See [GPU](#before-you-start). |
| `xray system` shows only `cpu` on Linux with an NVIDIA GPU | Check that the NVIDIA driver works (`nvidia-smi`). |
| The desktop app says "The engine could not start" | The message shows the cause; `engine.log` in the [data folder](#where-the-app-keeps-its-data) has more. Quit and reopen to retry. |
| The first launch fails partway | Usually a network interruption during the 2 GB download. Reopen the app to try again; packages already downloaded are reused. |
| Port 8791 is already in use (development) | Use another port, `xray serve --port 8792`, or `--port 0` for any free one. |

If something else goes wrong, please open an issue on GitHub with the output of
`uv run xray system` and the error message.
