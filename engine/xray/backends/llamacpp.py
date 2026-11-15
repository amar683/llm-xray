"""Real GGUF export and measurement through an existing llama.cpp build.

The engine never modifies the llama.cpp folder: it only runs its converter
script and its `llama-quantize` / `llama-perplexity` binaries, and writes every
output file into the export folder it is given.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xray.analyzers.quant import make_plan, plan_bytes, predicted_damage
from xray.data import load_prompt_set

# Bit width -> GGUF tensor type, and the bits per weight each really stores.
GGUF_TYPES = {3: "Q3_K", 4: "Q4_K", 5: "Q5_K", 6: "Q6_K", 8: "Q8_0"}
GGUF_BPW = {3: 3.4375, 4: 4.5, 5: 5.5, 6: 6.5625, 8: 8.5}

_HF_TO_GGUF = {
    "self_attn.q_proj": "attn_q",
    "self_attn.k_proj": "attn_k",
    "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate",
    "mlp.up_proj": "ffn_up",
    "mlp.down_proj": "ffn_down",
}
_LAYER_RE = re.compile(r"layers\.(\d+)\.(.+)\.weight$")


class LlamaCppError(RuntimeError):
    pass


@dataclass(frozen=True)
class Tools:
    root: Path
    convert: Path
    quantize: Path
    perplexity: Path
    bench: Path | None = None


def find_tools(root: str | Path | None = None) -> Tools:
    """Locate the converter and binaries in a llama.cpp checkout.

    `root` defaults to $XRAY_LLAMA_CPP. With several build folders, the most
    recently built `llama-quantize` wins.
    """
    root = root or os.environ.get("XRAY_LLAMA_CPP")
    if not root:
        raise LlamaCppError("Pass --llama-cpp or set XRAY_LLAMA_CPP to a llama.cpp folder.")
    root = Path(root).expanduser()
    convert = root / "convert_hf_to_gguf.py"
    candidates = sorted(
        root.glob("*/bin/llama-quantize"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not convert.exists() or not candidates:
        raise LlamaCppError(
            f"{root} does not look like a built llama.cpp checkout "
            "(need convert_hf_to_gguf.py and a */bin/llama-quantize)."
        )
    bin_dir = candidates[0].parent
    perplexity = bin_dir / "llama-perplexity"
    if not perplexity.exists():
        raise LlamaCppError(f"{bin_dir} has no llama-perplexity.")
    bench = bin_dir / "llama-bench"
    return Tools(root, convert, candidates[0], perplexity, bench if bench.exists() else None)


def gguf_name(hf_name: str) -> str:
    """Hugging Face parameter name -> GGUF tensor name."""
    m = _LAYER_RE.search(hf_name)
    if m and m.group(2) in _HF_TO_GGUF:
        return f"blk.{m.group(1)}.{_HF_TO_GGUF[m.group(2)]}.weight"
    if "embed_tokens" in hf_name:
        return "token_embd.weight"
    if "lm_head" in hf_name:
        return "output.weight"
    raise LlamaCppError(f"No GGUF name known for tensor '{hf_name}'.")


def _run(cmd: list[str | Path], env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = "\n".join(out.strip().splitlines()[-8:])
        raise LlamaCppError(f"{Path(str(cmd[0])).name} failed (exit {proc.returncode}):\n{tail}")
    return out


def convert_to_f16(tools: Tools, model_dir: Path, out: Path) -> Path:
    """Hugging Face folder -> F16 GGUF. Skipped when `out` already exists."""
    if out.exists():
        return out
    try:
        import sentencepiece  # noqa: F401 - the converter imports it even for BPE models
    except ImportError as e:
        raise LlamaCppError(
            "The llama.cpp converter needs the 'sentencepiece' package: "
            "run `uv sync --extra export`."
        ) from e
    tmp = out.with_suffix(".partial")
    _run(
        [sys.executable, tools.convert, model_dir, "--outfile", tmp, "--outtype", "f16"],
        env={"PYTHONPATH": str(tools.root / "gguf-py")},
    )
    tmp.rename(out)
    return out


def quantize(
    tools: Tools,
    f16: Path,
    out: Path,
    base_type: str,
    tensor_types: dict[str, str] | None = None,
    pure: bool = True,
) -> dict[str, Any]:
    """Run llama-quantize. `tensor_types` maps GGUF tensor names to types."""
    cmd: list[str | Path] = [tools.quantize]
    if pure:
        cmd.append("--pure")
    if tensor_types:
        listing = out.with_suffix(".tensor-types.txt")
        # Patterns are regular expressions searched in the name, so anchor and escape.
        listing.write_text(
            "".join(f"^{re.escape(name)}$={t.lower()}\n" for name, t in tensor_types.items())
        )
        cmd += ["--tensor-type-file", listing]
    log = _run([*cmd, f16, out, base_type])
    m = re.search(r"quant size\s*=\s*([\d.]+) MiB \(([\d.]+) BPW\)", log)
    return {
        "path": str(out),
        "file_bytes": out.stat().st_size,
        "weights_mib": float(m.group(1)) if m else None,
        "bpw": float(m.group(2)) if m else None,
    }


def save_base_logits(tools: Tools, f16: Path, text: Path, out: Path, context: int) -> None:
    _run(
        [tools.perplexity, "-m", f16, "-f", text, "-c", str(context), "-b", str(context),
         "--save-all-logits", out]
    )  # fmt: skip


_KLD_FIELDS = {
    "kl": r"Mean\s+KLD:\s+([-\d.]+)\s+±\s+([\d.]+)",
    "top1_agreement": r"Same top p:\s+([-\d.]+)\s+±\s+([\d.]+)\s*%",
    "ppl": r"Mean PPL\(Q\)\s*:\s+([-\d.]+)\s+±\s+([\d.]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s+([-\d.]+)\s+±\s+([\d.]+)",
}


def parse_kld(log: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, pattern in _KLD_FIELDS.items():
        m = re.search(pattern, log)
        if m:
            scale = 0.01 if key == "top1_agreement" else 1.0
            out[key] = float(m.group(1)) * scale
            out[f"{key}_err"] = float(m.group(2)) * scale
    if "kl" not in out:
        raise LlamaCppError("Could not find KL divergence in llama-perplexity's output.")
    return out


def kl_divergence(tools: Tools, model: Path, base_logits: Path, context: int) -> dict[str, float]:
    log = _run(
        [tools.perplexity, "-m", model, "-c", str(context), "-b", str(context),
         "--kl-divergence", "--kl-divergence-base", base_logits]
    )  # fmt: skip
    return parse_kld(log)


def heldout_text(sets: list[str], calibration_prompts: int) -> str:
    """Every prompt the quant run did *not* use for calibration."""
    per_set = max(1, calibration_prompts // len(sets))
    return "\n".join(p for s in sets for p in load_prompt_set(s).prompts[per_set:]) + "\n"


def export_and_check(
    quant_result: dict[str, Any],
    quant_params: dict[str, Any],
    model_dir: Path,
    tools: Tools,
    out_dir: Path,
    target_bits: int = 4,
    context: int = 128,
    keep_files: bool = True,
    progress=lambda stage: None,
) -> dict[str, Any]:
    """Export real GGUFs and measure them with llama.cpp.

    For each baseline (uniform Q{n}_K, and llama.cpp's own Q{n}_K_M mix when it
    exists) a mixed plan is built for the same weight budget from the quant
    run's sensitivities, exported, and compared on held-out text.
    """
    if target_bits not in GGUF_TYPES:
        raise LlamaCppError(f"target_bits must be one of {sorted(GGUF_TYPES)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = quant_result["tensors"]
    probe = quant_result["settings"]["probe_bits"]
    sets = quant_params.get("sets") or ["coding", "chat", "math", "instruction"]

    progress("converting to GGUF")
    f16 = convert_to_f16(tools, model_dir, out_dir / "model-f16.gguf")
    text = out_dir / "heldout.txt"
    text.write_text(heldout_text(sets, quant_result["settings"]["calibration_prompts"]))
    progress("measuring full precision")
    base = out_dir / "base-logits.bin"
    save_base_logits(tools, f16, text, base, context)

    uniform_type = GGUF_TYPES[target_bits]
    baselines = [(f"uniform {uniform_type}", uniform_type, True)]
    if target_bits in (3, 4, 5):
        baselines.append((f"llama.cpp {uniform_type}_M", f"{uniform_type}_M", False))

    def bpw(bits: int) -> float:
        return GGUF_BPW[bits]

    uniform_budget = plan_bytes(tensors, {t["name"]: target_bits for t in tensors}, bpw)
    uniform_file = None
    comparisons = []
    for label, gguf_type, pure in baselines:
        slug = gguf_type.lower()
        progress(f"exporting {label}")
        baseline = quantize(tools, f16, out_dir / f"baseline-{slug}.gguf", gguf_type, pure=pure)
        uniform_file = uniform_file or baseline["file_bytes"]
        # Whatever the baseline spends beyond pure uniform is extra budget for the plan.
        budget = uniform_budget + (baseline["file_bytes"] - uniform_file)
        assignment = make_plan(tensors, budget, probe, bpw)
        progress(f"exporting mixed plan at the size of {label}")
        mixed = quantize(
            tools,
            f16,
            out_dir / f"mixed-at-{slug}.gguf",
            uniform_type,
            tensor_types={gguf_name(n): GGUF_TYPES[b] for n, b in assignment.items()},
        )
        progress(f"measuring {label} and its mixed plan")
        baseline.update(kl_divergence(tools, Path(baseline["path"]), base, context))
        mixed.update(kl_divergence(tools, Path(mixed["path"]), base, context))
        mixed["assignment"] = assignment
        mixed["predicted_kl_simulated"] = sum(
            predicted_damage(t, assignment[t["name"]], probe) for t in tensors
        )
        comparisons.append(
            {"label": label, "type": gguf_type, "baseline": baseline, "mixed": mixed}
        )

    if not keep_files:
        for c in comparisons:
            for side in ("baseline", "mixed"):
                Path(c[side]["path"]).unlink(missing_ok=True)
        f16.unlink(missing_ok=True)
    base.unlink(missing_ok=True)
    return {
        "llama_cpp": str(tools.root),
        "quantize_binary": str(tools.quantize),
        "context": context,
        "target_bits": target_bits,
        "out_dir": str(out_dir),
        "files_kept": keep_files,
        "comparisons": comparisons,
    }


def export_edited(
    tools: Tools,
    base_dir: Path,
    edited_dir: Path,
    out_dir: Path,
    gguf_type: str = "Q4_K_M",
    context: int = 128,
    progress=lambda stage: None,
) -> dict[str, Any]:
    """Convert an edited model to GGUF and measure it against the base model.

    Both the base and the edited model are quantized to `gguf_type`; each is
    then compared with the *full-precision base* on the same held-out text, so
    the two KL figures are directly comparable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    progress("converting the base model")
    base_f16 = convert_to_f16(tools, base_dir, out_dir / "model-f16.gguf")
    progress("converting the edited model")
    edited_f16 = out_dir / "edited-f16.gguf"
    edited_f16.unlink(missing_ok=True)
    convert_to_f16(tools, edited_dir, edited_f16)

    text = out_dir / "heldout.txt"
    text.write_text(heldout_text(["coding", "chat", "math", "instruction"], 16))
    progress("measuring the full-precision base")
    logits = out_dir / "base-logits.bin"
    save_base_logits(tools, base_f16, text, logits, context)

    slug = gguf_type.lower()
    rows = {}
    for label, f16, name in (
        ("base", base_f16, f"baseline-{slug}.gguf"),
        ("edited", edited_f16, f"edited-{slug}.gguf"),
    ):
        progress(f"quantizing and measuring the {label} model")
        info = quantize(tools, f16, out_dir / name, gguf_type, pure=False)
        info.update(kl_divergence(tools, Path(info["path"]), logits, context))
        rows[label] = info
    progress("measuring the edited model at full precision")
    edited_full = {"path": str(edited_f16), "file_bytes": edited_f16.stat().st_size}
    edited_full.update(kl_divergence(tools, edited_f16, logits, context))
    logits.unlink(missing_ok=True)
    return {
        "type": gguf_type,
        "context": context,
        "out_dir": str(out_dir),
        "base": rows["base"],
        "edited": rows["edited"],
        "edited_f16": edited_full,
        "base_f16_bytes": base_f16.stat().st_size,
    }


def disk_needed(model_bytes: int) -> int:
    """Rough space an export needs: the F16 file plus four quantized ones."""
    return int(model_bytes * 2.5)


def has_space(out_dir: Path, model_bytes: int) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(out_dir).free > disk_needed(model_bytes)
