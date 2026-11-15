"""Eval runner: is a model still right, and how far has it moved from the base?

Three kinds of evidence, because no single number is enough:

  tasks   questions with one exact answer (capitals, arithmetic). Plain-text
          completion, so they work on base and chat models alike. Each task is
          split in two: the `tune` half may be used while fitting an edit, the
          `test` half is only ever used for the reported score.
  drift   how far next-token predictions have moved from the base model on
          ordinary prompts (KL divergence, share of unchanged top predictions).
  health  on a few chat prompts: does generation loop, and does it end?

Accuracy comes with its binomial standard error so a 2-point change on 25
questions is not mistaken for a result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from xray.analyzers.common import greedy_generate, token_text
from xray.analyzers.loops import detect_loop
from xray.analyzers.profile import encode_prompts
from xray.analyzers.quant import compare, log_probs
from xray.data import load_prompt_set
from xray.eval.checkers import NEW_TOKENS, check, check_prefix, isolation
from xray.models.adapters import LoadedModel

TASK_DIR = Path(__file__).parent.parent / "data" / "evals"
CODE_TASK = "code"
SETS = ("coding", "chat", "math", "instruction")


def builtin_tasks(run_code: bool = False) -> list[str]:
    """Task names. The `code` task runs model-written code, so it is opt-in."""
    names = sorted(p.stem for p in TASK_DIR.glob("*.jsonl"))
    return names if run_code else [n for n in names if n != CODE_TASK]


def load_task(name: str, split: str = "test") -> list[dict[str, Any]]:
    """Even-numbered items are the `tune` half, odd-numbered the `test` half."""
    items = [json.loads(line) for line in (TASK_DIR / f"{name}.jsonl").read_text().splitlines()]
    return items[(0 if split == "tune" else 1) :: 2]


def is_correct(output: str, answer: str) -> bool:
    return check_prefix(output, answer)


def run_task(lm: LoadedModel, name: str, split: str = "test") -> dict[str, Any]:
    items = load_task(name, split)
    kind = items[0].get("check", "prefix")
    new_tokens = NEW_TOKENS[kind]
    encoded = encode_prompts(lm, [i["prompt"] for i in items], reserve=new_tokens, chat=False)
    correct, wrong = 0, []
    for item, ids in zip(items, encoded, strict=True):
        tokens, _ = greedy_generate(lm, ids, new_tokens)
        output = "".join(token_text(lm, tokens))
        if check(output, item):
            correct += 1
        elif len(wrong) < 3:
            wrong.append(
                {
                    "prompt": item["prompt"].splitlines()[-1],
                    "answer": item.get("answer", kind),
                    "output": output.strip()[:60],
                }
            )
    n = len(items)
    p = correct / n
    return {
        "n": n,
        "correct": correct,
        "accuracy": p,
        "stderr": math.sqrt(p * (1 - p) / n),
        "check": kind,
        "examples_wrong": wrong,
    }


def drift_prompts(per_set: int = 8) -> list[str]:
    """The last prompts of each built-in set: never used for calibration or healing."""
    return [p for s in SETS for p in load_prompt_set(s).prompts[-per_set:]]


def health_prompts(per_set: int = 2) -> list[str]:
    return [p for s in SETS for p in load_prompt_set(s).prompts[-per_set:]]


class Evaluator:
    """Holds the base model's predictions so later versions can be measured against them."""

    def __init__(
        self,
        lm: LoadedModel,
        tasks: list[str] | None = None,
        health_tokens: int = 80,
        run_code: bool = False,
    ) -> None:
        self.tasks = tasks or builtin_tasks(run_code)
        if CODE_TASK in self.tasks and not run_code:
            raise ValueError("The code task runs model-written code; pass run_code=True.")
        self.code_isolation = isolation() if CODE_TASK in self.tasks else None
        self.health_tokens = health_tokens
        self.drift_ids = encode_prompts(lm, drift_prompts())
        self.health_ids = encode_prompts(lm, health_prompts(), reserve=health_tokens)
        self.base_logp = log_probs(lm, self.drift_ids)

    def evaluate(
        self, lm: LoadedModel, progress=lambda *a, **k: None, label: str = ""
    ) -> dict[str, Any]:
        tasks = {}
        for name in self.tasks:
            progress(f"evaluating {label}: {name}")
            tasks[name] = run_task(lm, name)
        progress(f"evaluating {label}: drift from base")
        drift = compare(lm, self.drift_ids, self.base_logp)
        progress(f"evaluating {label}: loops")
        loops = ended = 0
        for ids in self.health_ids:
            tokens, done = greedy_generate(lm, ids, self.health_tokens)
            loops += bool(detect_loop(tokens))
            ended += done
        n = len(self.health_ids)
        params = sum(p.numel() for p in lm.model.parameters())
        return {
            "tasks": tasks,
            "drift": drift,
            "health": {"prompts": n, "looped": loops, "ended": ended},
            "size": {
                "params": params,
                "fp16_bytes": params * 2,
                "layers": len(lm.refs.layers),
                "mlp_width": lm.refs.layers[0].mlp_out.in_features,
            },
        }


@torch.inference_mode()
def teacher_logits(lm: LoadedModel, encoded: list[torch.Tensor]) -> list[torch.Tensor]:
    """Base-model logits kept on the CPU in half precision, for distillation."""
    return [
        lm.model(input_ids=ids.to(lm.device)).logits[0].to("cpu", torch.float16) for ids in encoded
    ]
