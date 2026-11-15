"""Smoke test: every command, end to end, on a tiny model. No downloads.

    uv run python scripts/smoke.py

Builds a tiny random model in a temporary folder and drives the real `xray`
command line and API server against it, the way a user would. It is the gate
for a release build on each operating system: it needs nothing but the
installed engine, runs in about a minute on a CPU, and exits non-zero on the
first failure.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def make_model(path: Path) -> None:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )  # fmt: skip
    LlamaForCausalLM(config).save_pretrained(path)


class Smoke:
    def __init__(self, home: Path, model: Path) -> None:
        # PYTHONUTF8: on Windows, child processes otherwise print in the console's
        # code page (cp1252), which a UTF-8 reader cannot decode.
        self.env = {**os.environ, "XRAY_HOME": str(home), "COLUMNS": "200", "PYTHONUTF8": "1"}
        self.model = str(model)
        self.passed = 0

    def xray(self, *args: str, expect: str | None = None, fail: bool = False) -> str:
        cmd = [sys.executable, "-m", "xray.cli", *args]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=self.env, encoding="utf-8", errors="replace"
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        label = "xray " + " ".join(a if len(a) < 40 else "..." for a in args)
        ok = (proc.returncode != 0) if fail else (proc.returncode == 0)
        if ok and expect is not None:
            ok = expect in out
        if not ok:
            print(f"FAIL  {label}\n{out[-1500:]}")
            raise SystemExit(1)
        self.passed += 1
        print(f"ok    {label}")
        return out

    def run_id(self, run_type: str) -> str:
        from xray.core import paths
        from xray.core.storage import Storage

        with Storage(paths.db_path(Path(self.env["XRAY_HOME"]))) as db:
            return next(
                r["id"]
                for r in db.list_runs(100)
                if r["type"] == run_type and r["status"] == "done"
            )

    def server(self) -> None:
        cmd = [sys.executable, "-m", "xray.cli", "serve", "--port", "0", "--token", "smoke"]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, text=True, env=self.env, encoding="utf-8", errors="replace"
        )
        try:
            ready = json.loads(proc.stdout.readline())
            base = f"http://127.0.0.1:{ready['port']}"

            def get(path: str, token: str | None) -> tuple[int, dict]:
                headers = {"Authorization": f"Bearer {token}"} if token else {}
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(base + path, headers=headers), timeout=30
                    ) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, {}

            for _ in range(100):  # the port is announced just before the server listens
                try:
                    if get("/health", None)[0] == 200:
                        break
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.1)
            assert get("/health", None)[1]["ok"] is True
            assert get("/system", None)[0] == 401, "the API must refuse requests without the token"
            assert get("/system", "smoke")[1]["backends"]["cpu"] is True
            assert len(get("/runs", "smoke")[1]) >= 8
        finally:
            proc.terminate()
            proc.wait(timeout=20)
        self.passed += 1
        print("ok    xray serve (health, token check, system, runs)")


def main() -> int:
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "tiny-model"
        make_model(model)
        s = Smoke(root / "home", model)
        m, cpu = s.model, ("--device", "cpu")

        s.xray("system", expect="default_device")
        s.xray("inspect", m, expect="llama")
        s.xray("fit", m, expect="Recommended")
        s.xray("inspect", str(root / "missing"), fail=True)
        s.xray("timing", m, "--contexts", "8,16", *cpu, expect="By context length")
        s.xray("profile", m, "--sets", "coding,math", "--limit", "3", "--new-tokens", "2",
               "--ablation-prompts", "2", *cpu, expect="Per layer")  # fmt: skip
        s.xray(
            "neurons", s.run_id("profile"), "--layer", "1", "--set", "math", expect="most specific"
        )
        s.xray("quant", m, "--calibration-prompts", "4", "--heldout-prompts", "4", "--target-bits", "4",
               *cpu, expect="Most sensitive tensors")  # fmt: skip
        s.xray("lens", m, "hello there", "--new-tokens", "3", *cpu, expect="answer at")
        s.xray("attribute", m, "hello there", *cpu, expect="logit gap")
        s.xray(
            "loops",
            m,
            "--sets",
            "coding",
            "--limit",
            "3",
            "--new-tokens",
            "40",
            *cpu,
            expect="looped",
        )
        s.xray(
            "edit",
            m,
            "-c",
            "zero_head layer=0 head=1",
            "--name",
            "first",
            *cpu,
            expect="Saved patch",
        )
        s.xray(
            "edit",
            m,
            "-c",
            "drop_layers layers=1",
            "--run-code",
            *cpu,
            expect="Code task isolation",
        )
        s.xray("edit", m, "-c", "explode now=1", fail=True)
        s.xray("patches", m, expect="first")
        edits = []
        from xray.core import paths
        from xray.core.storage import Storage

        with Storage(paths.db_path(root / "home")) as db:
            edits = [
                r["id"] for r in db.list_runs(100) if r["type"] == "edit" and r["status"] == "done"
            ]
        s.xray("compare", edits[1], edits[0], expect="Drift from base")
        s.xray("show", s.run_id("timing"), expect="By context length")
        s.xray("report", m, expect="# Model report")
        s.xray("query", m, "layer 1", expect="Layer 1:")
        s.xray("advise", m, "--goal", "Make it loop less", expect="Nothing has been sent")
        s.xray("bench", m, "--no-mlx", fail=True)  # no llama.cpp folder given: a clear refusal
        s.xray("runs", expect="timing")
        s.server()
        print(f"\n{s.passed} checks passed in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
