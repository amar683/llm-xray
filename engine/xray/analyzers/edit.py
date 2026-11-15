"""The edit job: apply a patch, optionally heal it, and measure every stage
against the untouched base model."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch

from xray.analyzers.common import job_output, load_for_job
from xray.analyzers.profile import encode_prompts
from xray.data import load_prompt_set
from xray.eval import SETS, Evaluator, load_task, teacher_logits
from xray.patch import apply_commands, parse_commands
from xray.patch.heal import heal

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "special_tokens_map.json", "added_tokens.json", "generation_config.json",
                   "tokenizer.model", "chat_template.jinja")  # fmt: skip


def training_prompts(lm, tasks: list[str], per_set: int = 8) -> list[torch.Tensor]:
    """Prompts for calibration and healing: the front of each built-in set plus
    the `tune` half of each task. Never the prompts used for scoring."""
    chat = [p for s in SETS for p in load_prompt_set(s).prompts[:per_set]]
    plain = [item["prompt"] for t in tasks for item in load_task(t, "tune")[:8]]
    return encode_prompts(lm, chat) + encode_prompts(lm, plain, chat=False)


def save_model(lm, out_dir: Path) -> dict[str, Any]:
    """Write the edited model as a standard Hugging Face folder, in half precision."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lm.model.to(torch.float16).save_pretrained(out_dir)
    for name in TOKENIZER_FILES:
        src = Path(lm.path) / name
        if src.exists():
            shutil.copy2(src, out_dir / name)
    size = sum(f.stat().st_size for f in out_dir.glob("*.safetensors"))
    return {"path": str(out_dir), "weights_bytes": size}


def edit_job(params: dict[str, Any], ctx) -> dict[str, Any]:
    commands = parse_commands(params["commands"])  # fail on a typo before loading anything
    heal_cfg = params.get("heal")
    if heal_cfg:
        # Training needs full precision.
        params = {**params, "dtype": "float32"}
    lm, anatomy = load_for_job(params, ctx)

    evaluator = Evaluator(lm, params.get("tasks"), run_code=params.get("run_code", False))
    train = training_prompts(lm, evaluator.tasks)
    stages = {"base": evaluator.evaluate(lm, ctx.progress, "base model")}
    teacher = teacher_logits(lm, train) if heal_cfg else None

    ctx.progress("applying edits")
    log = apply_commands(lm, commands, calibration=train)
    stages["edited"] = evaluator.evaluate(lm, ctx.progress, "edited model")

    healing = None
    if heal_cfg:
        healing = heal(
            lm, train, teacher,
            steps=heal_cfg.get("steps", 150), rank=heal_cfg.get("rank", 8),
            lr=heal_cfg.get("lr", 3e-4), seed=params.get("seed", 0), progress=ctx.progress,
        )  # fmt: skip
        stages["healed"] = evaluator.evaluate(lm, ctx.progress, "healed model")

    saved = None
    if params.get("save"):
        ctx.progress("saving")
        saved = save_model(lm, ctx.run_dir / "model")

    result = {
        "commands": params["commands"],
        "applied": log,
        "stages": stages,
        "healing": healing,
        "saved": saved,
        "tasks": evaluator.tasks,
        "code_isolation": evaluator.code_isolation,
        "patch_id": params.get("patch_id"),
    }
    return job_output(lm, anatomy, "edit", params, result)
