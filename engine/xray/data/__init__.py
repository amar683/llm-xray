"""Built-in prompt sets and prompt-set loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

PROMPT_DIR = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class PromptSet:
    name: str
    prompts: list[str]
    source: str  # "builtin" or "user"
    digest: str  # identifies the exact prompts used, for the run manifest


def builtin_sets() -> list[str]:
    return sorted(p.stem for p in PROMPT_DIR.glob("*.jsonl"))


def load_prompt_set(name_or_path: str, limit: int | None = None) -> PromptSet:
    """Load a built-in set by name, or a user file (.jsonl with a `prompt` key, or .txt)."""
    builtin = PROMPT_DIR / f"{name_or_path}.jsonl"
    path, source = (builtin, "builtin") if builtin.exists() else (Path(name_or_path), "user")
    if not path.is_file():
        raise FileNotFoundError(
            f"'{name_or_path}' is not a built-in prompt set ({', '.join(builtin_sets())}) "
            "and is not a file."
        )
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    prompts = [json.loads(line)["prompt"] for line in lines] if path.suffix == ".jsonl" else lines
    prompts = prompts[:limit] if limit else prompts
    if not prompts:
        raise ValueError(f"Prompt set '{name_or_path}' is empty.")
    digest = hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]
    return PromptSet(path.stem, prompts, source, digest)
