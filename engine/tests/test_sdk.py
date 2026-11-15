import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import pytest

import xray
from xray.backends.llamacpp import LlamaCppError, Tools
from xray.backends.runtimes import bench_llamacpp, run_bench
from xray.core.summary import runtime_rows, summarize

ENGINE = Path(__file__).parent.parent
ROOT = ENGINE.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ENGINE / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- public API -----------------------------------------------------------------


def test_import_is_light_and_lazy():
    code = "import sys, xray; assert 'torch' not in sys.modules; xray.Model; print('ok')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr
    assert set(xray.__all__) >= {"Model", "Run", "compare", "load_run", "runs"}
    with pytest.raises(AttributeError):
        xray.no_such_thing  # noqa: B018


def test_model_reads_without_loading(tiny_model_dir, tmp_path):
    m = xray.Model(str(tiny_model_dir), home=tmp_path, device="cpu")
    assert m.inspect()["architecture"]["n_layers"] == 3
    assert m.fit()["recommended"]["analysis"]
    assert m.runs() == [] and m.latest("timing") is None and m.runtimes() == []
    assert repr(m).startswith("Model(")
    with pytest.raises(FileNotFoundError):
        xray.Model("/no/such/model")


def test_runs_are_recorded_reloadable_and_comparable(tiny_model_dir, tmp_path):
    m = xray.Model(str(tiny_model_dir), home=tmp_path, device="cpu")
    events = []
    timing = m.timing(contexts=[8], prefill_repeats=2, decode_repeats=2, on_progress=events.append)
    assert timing.type == "timing" and timing.result["contexts"][0]["context"] == 8
    assert any(e["stage"] == "loading model" for e in events)
    assert timing.summary()[0]["label"].startswith("Decode speed")

    again = xray.load_run(timing.id, home=tmp_path)
    assert again.result == timing.result and again.manifest["backend"]["device"] == "cpu"
    assert m.latest("timing").id == timing.id
    assert [r["id"] for r in xray.runs(home=tmp_path)] == [timing.id]

    a = m.edit(["zero_head layer=0 head=1"])
    b = m.edit(["zero_head layer=1 head=2"])
    rows = xray.compare(a, b)
    assert rows == xray.compare(a.id, b.id, home=tmp_path)
    assert {r["label"] for r in rows} >= {"Layers", "Drift from base (KL)"}
    assert [r["type"] for r in m.runs()] == ["edit", "edit", "timing"]
    assert len(m.runs("edit")) == 2

    assert m.runtimes()[0]["runtime"] == "PyTorch"  # no bench yet: PyTorch alone
    assert "Layer 0:" in m.query("layer 0") and "# Model report" in m.report()["text"]
    with pytest.raises(ValueError, match="Unknown command"):
        m.edit(["explode"])
    with pytest.raises(KeyError):
        xray.load_run("nope", home=tmp_path)


# -- runtimes -------------------------------------------------------------------

BENCH_JSON = json.dumps([
    {"n_prompt": 512, "n_gen": 0, "avg_ts": 3000.0, "stddev_ts": 5.0, "backends": "MTL", "build_commit": "abc"},
    {"n_prompt": 0, "n_gen": 128, "avg_ts": 100.0, "stddev_ts": 1.0, "backends": "MTL", "build_commit": "abc"},
])  # fmt: skip


def fake_tools(tmp_path: Path, script: str) -> Tools:
    bench = tmp_path / "llama-bench"
    bench.write_text(f"#!/bin/sh\n{script}\n")
    bench.chmod(0o755)
    return Tools(tmp_path, tmp_path / "convert.py", tmp_path / "q", tmp_path / "p", bench)


@pytest.mark.skipif(platform.system() == "Windows", reason="uses a shell script stand-in")
def test_bench_llamacpp_parses_llama_bench_output(tmp_path):
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"x" * 1000)
    tools = fake_tools(tmp_path, f"echo 'loading...'\ncat <<'JSON'\n{BENCH_JSON}\nJSON")
    r = bench_llamacpp(tools, gguf, 512, 128, 3)
    assert (r["prefill_tps"], r["decode_tps"], r["file_bytes"], r["build"]) == (
        3000.0,
        100.0,
        1000,
        "abc",
    )
    with pytest.raises(LlamaCppError, match="exit 2"):
        bench_llamacpp(fake_tools(tmp_path, "echo boom >&2; exit 2"), gguf, 512, 128, 3)
    with pytest.raises(LlamaCppError, match="no llama-bench"):
        bench_llamacpp(Tools(tmp_path, tmp_path, tmp_path, tmp_path, None), gguf, 512, 128, 3)


def test_run_bench_reports_what_it_could_not_measure(tiny_model_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("XRAY_LLAMA_CPP", raising=False)
    with pytest.raises(RuntimeError, match="No runtime could be measured"):
        run_bench(str(tiny_model_dir), tmp_path / "out", llama_cpp=None, use_mlx=False)
    with pytest.raises(RuntimeError, match="llama.cpp: Pass --llama-cpp"):
        run_bench(str(tiny_model_dir), tmp_path / "out", llama_cpp=None, use_mlx=False)


def test_runtime_rows_put_pytorch_first_and_compare_decode_speed():
    timing = {"contexts": [
        {"context": 128, "prefill": {"tokens_per_s": 2000.0}, "decode": {"tokens_per_s": 30.0}},
        {"context": 512, "prefill": {"tokens_per_s": 3000.0}, "decode": {"tokens_per_s": 29.0}},
    ]}  # fmt: skip
    bench = {"n_prompt": 512, "rows": [
        {"runtime": "llama.cpp", "variant": "Q4_K_M", "file_bytes": 900, "prefill_tps": 3100.0, "decode_tps": 90.0},
    ]}  # fmt: skip
    rows = runtime_rows(timing, bench)
    assert [r["runtime"] for r in rows] == ["PyTorch", "llama.cpp"]
    assert rows[0]["prefill_tps"] == 3000.0 and rows[0]["prefill_context"] == 512  # closest context
    assert rows[0]["vs_pytorch"] == 1.0 and rows[1]["vs_pytorch"] == pytest.approx(3.0)
    assert runtime_rows(None, bench)[0]["vs_pytorch"] is None  # nothing to compare against
    assert runtime_rows(None, None) == []
    labels = [m["label"] for m in summarize("bench", bench)]
    assert labels == ["llama.cpp Q4_K_M: decode (tokens/s)", "llama.cpp Q4_K_M: prefill (tokens/s)"]


# -- examples and docs ----------------------------------------------------------


def test_quickstart_notebook_runs(tiny_model_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("XRAY_EXAMPLE_MODEL", str(tiny_model_dir))
    monkeypatch.setenv("XRAY_EXAMPLE_CONTEXT", "8")
    monkeypatch.setenv("XRAY_HOME", str(tmp_path))
    namespace = load_script("run_notebook").run(ROOT / "examples" / "quickstart.ipynb", quiet=True)
    out = capsys.readouterr().out
    assert "decode:" in out and "least harmful layers to skip" in out and "Layer 1:" in out
    assert namespace["report"]["fits"] and namespace["again"].id == namespace["timing"].id
    assert {r["type"] for r in namespace["m"].runs()} == {"timing", "profile", "edit"}


def test_cli_docs_are_up_to_date():
    docs = load_script("make_cli_docs")
    assert (ROOT / "docs" / "cli.md").read_text() == docs.render(), (
        "docs/cli.md is stale: run `uv run python scripts/make_cli_docs.py`"
    )
