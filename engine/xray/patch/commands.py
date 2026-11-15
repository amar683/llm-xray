"""Parsing and validating edit commands (no torch needed)."""

from __future__ import annotations

import shlex
from typing import Any

COMMANDS: dict[str, dict[str, type]] = {
    "zero_head": {"layer": int, "head": int},
    "scale_neuron": {"layer": int, "neuron": int, "factor": float},
    "ablate_direction": {"vector": str, "from_layer": int},
    "drop_layers": {"layers": str},
    "prune_neurons": {"keep": float},
}


class PatchError(ValueError):
    pass


def _split(text: str) -> list[str]:
    """Split on spaces, honouring quotes, with backslashes kept as they are.

    Plain `shlex.split` treats a backslash as an escape, which silently turns
    a Windows path like C:\\Users\\me\\d.npy into C:Usersmed.npy.
    """
    lexer = shlex.shlex(text, posix=True)
    lexer.whitespace_split = True
    lexer.escape = ""
    return list(lexer)


def parse_command(text: str) -> dict[str, Any]:
    """`name key=value ...` -> {"op": name, key: typed value, ...}, validated."""
    parts = _split(text)
    if not parts:
        raise PatchError("Empty command.")
    op, spec = parts[0], COMMANDS.get(parts[0])
    if spec is None:
        raise PatchError(f"Unknown command '{op}'. Known: {', '.join(COMMANDS)}.")
    args: dict[str, Any] = {"op": op}
    for part in parts[1:]:
        key, sep, value = part.partition("=")
        if not sep or key not in spec:
            raise PatchError(f"'{op}' does not take '{part}'. Expected: {', '.join(spec)}.")
        try:
            args[key] = spec[key](value)
        except ValueError as e:
            raise PatchError(f"'{op}': {key} must be {spec[key].__name__}, got '{value}'.") from e
    missing = [k for k in spec if k not in args]
    if missing:
        raise PatchError(f"'{op}' is missing: {', '.join(missing)}.")
    if op == "drop_layers":
        try:
            args["layers"] = sorted({int(x) for x in args["layers"].split(",") if x.strip()})
        except ValueError as e:
            raise PatchError("'drop_layers': layers must be numbers separated by commas.") from e
    if op == "prune_neurons" and not 0 < args["keep"] <= 1:
        raise PatchError("'prune_neurons': keep must be above 0 and at most 1.")
    return args


def parse_commands(lines: list[str]) -> list[dict[str, Any]]:
    return [
        parse_command(line) for line in lines if line.strip() and not line.strip().startswith("#")
    ]
