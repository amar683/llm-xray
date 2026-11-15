# Releasing

Version 0.9.0 is a release candidate. It has been built and run on one machine
(macOS, Apple Silicon). This page lists what a 1.0 release still needs and how
to do each part.

## What is verified, and where

| Item | macOS (Apple Silicon) | Windows | Linux |
|---|---|---|---|
| Engine tests | passed locally and in CI | passed in CI | passed in CI |
| Smoke suite (every command, API server) | passed locally and in CI | passed in CI | passed in CI |
| Desktop shell compiles | built locally, checked in CI | checked in CI | checked in CI |
| Desktop app installed and run | yes, locally | no | no |
| Signed / notarized installer | no | no | no |
| CUDA | not applicable | not tested | not tested |

Pushes and pull requests run CI on Linux only, to save Actions minutes. macOS
and Windows are tested by the release workflow on every version tag (full test
suite, then the smoke suite), or on demand from the Actions tab: **CI > Run
workflow** runs all three systems.

CI runs on GitHub's standard runners, which have no GPU, so every engine test
there runs on the CPU. "Checked in CI" for the desktop shell means it compiles;
no Windows or Linux installer has been built or launched.

## Steps to 1.0

1. **Build and launch the app on Windows and Linux.** CI proves the engine and
   the shell's code work there, but not the installer or first-run setup.
2. **Test on an NVIDIA machine.** Nothing has run on CUDA. The risky parts are
   half-precision choices and memory checks, which were tuned on Apple hardware.
3. **Tag a release** (`git tag v1.0.0 && git push --tags`). `release.yml` runs
   the smoke suite on each system, then builds installers and attaches them to
   a draft GitHub release.
4. **Install each installer on a clean machine** and run through
   [the first-run steps](user-guide.md#first-run).

## The smoke suite

```bash
cd engine
uv run python scripts/smoke.py
```

It builds a tiny random model and drives every `xray` command and the API
server against it, with no downloads, in about a minute on a CPU. It is the
gate for a release on each system.

## Signing

Unsigned installers work, but macOS and Windows warn users on first launch.

| System | What you need | Secrets for `release.yml` |
|---|---|---|
| macOS | An Apple Developer account and a "Developer ID Application" certificate | `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD` (an app-specific password), `APPLE_TEAM_ID` |
| Windows | A code-signing certificate | Configure `bundle.windows` in `tauri.conf.json`; see the Tauri signing guide |
| Linux | Nothing | – |

With the Apple secrets set, the release workflow signs and notarizes the macOS
app automatically.

## Not built yet

- **Auto-update.** Tauri's updater plugin needs a signing key pair and a place
  to host update manifests (GitHub Releases works). It is not configured.
- **Offline bundle.** The app downloads Python and PyTorch on first run. A
  bundle that carries them (about 2–3 GB) for machines without internet is not
  built.
- **CUDA build selection.** On Windows and Linux the engine installs whatever
  PyTorch build the lock file resolves to. Choosing the CUDA or CPU build per
  machine at first run is not implemented.
- **UI tests.** The React code has no automated tests; it was checked by hand.

## How the app is put together

- The installer contains the desktop shell, the UI, the engine's source and a
  copy of `uv` (placed by `app/scripts/prepare-uv.mjs` at build time).
- On launch the shell runs `uv sync` into `venv/` in the app's data folder,
  then starts the engine's API server on a free local port with a fresh token.
- The UI talks to the engine directly on 127.0.0.1. The shell stops the engine
  on quit, and the engine stops itself if the shell disappears.
