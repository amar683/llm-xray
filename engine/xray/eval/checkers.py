"""Answer checkers for eval tasks.

  prefix  the output starts with the expected answer
  json    the output starts with a JSON object that has the required keys
  python  the output completes a function that passes its tests when run

The `python` checker executes model-written code, so it is opt-in and
isolated: a separate interpreter with no site packages, an empty environment,
an empty temporary working folder, CPU and file-size limits, a wall-clock
timeout, and (on macOS) a sandbox profile that denies network access and file
writes. Code containing imports or other escape hatches is rejected unrun.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
from functools import cache
from typing import Any

# How many tokens each kind of answer needs.
NEW_TOKENS = {"prefix": 6, "json": 40, "python": 64}

FORBIDDEN = ("import", "open(", "exec(", "eval(", "compile(", "__", "input(", "globals(",
             "locals(", "getattr(", "setattr(", "os.", "sys.", "subprocess", "socket")  # fmt: skip
SANDBOX_PROFILE = "(version 1)(allow default)(deny network*)(deny file-write*)"
OK_MARK = "XRAY_TESTS_PASSED"


def check_prefix(output: str, answer: str) -> bool:
    """The output must start with the answer and not run on into a longer
    number ("195" for "19") or a longer plain word ("Parisian" for "Paris")."""
    text = output.strip().lower()
    answer = answer.lower()
    if not text.startswith(answer):
        return False
    rest = text[len(answer) :]
    if not rest:
        return True
    if answer[-1].isdigit():
        return not rest[0].isdigit()
    return not ("a" <= rest[0] <= "z")


def check_json(output: str, keys: list[str]) -> bool:
    """The output must begin with one valid JSON object containing every key."""
    text = output.lstrip()
    if not text.startswith("{"):
        return False
    try:
        value, _ = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return False
    return isinstance(value, dict) and all(k in value for k in keys)


def function_body(output: str) -> str:
    """The indented lines the model wrote before it moved on to something else."""
    body = []
    for line in output.splitlines():
        if line.strip() and not line.startswith((" ", "\t")):
            break
        body.append(line)
    return "\n".join(body).rstrip()


@cache
def isolation() -> str:
    """How model-written code is isolated on this machine."""
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        probe = subprocess.run(
            ["sandbox-exec", "-p", SANDBOX_PROFILE, sys.executable, "-I", "-S", "-c", "print(1)"],
            capture_output=True, timeout=20,
        )  # fmt: skip
        if probe.returncode == 0:
            return "sandbox (no network, no file writes) + process limits"
    return "process limits only"


def _limits() -> None:  # runs in the child process just before the code starts
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))  # cannot write any file


def check_python(output: str, item: dict[str, Any], timeout: float = 8.0) -> bool:
    """Run the completed function against its tests in an isolated interpreter."""
    body = function_body(output)
    if not body.strip() or any(word in body for word in FORBIDDEN):
        return False
    script = (
        # The whole prompt is valid Python: the example functions, then the stub.
        f"{item['prompt']}{body}\n\n"
        f"_tests = {item['tests']!r}\n"
        f"_ok = all({item['entry']}(*args) == expected for args, expected in _tests)\n"
        f"print({OK_MARK!r} if _ok else 'failed')\n"
    )
    cmd = [sys.executable, "-I", "-S", "-c", script]
    if isolation().startswith("sandbox"):
        cmd = ["sandbox-exec", "-p", SANDBOX_PROFILE, *cmd]
    with tempfile.TemporaryDirectory() as cwd:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env={},
                preexec_fn=_limits if platform.system() != "Windows" else None,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            return False
    return proc.returncode == 0 and proc.stdout.strip() == OK_MARK


def check(output: str, item: dict[str, Any]) -> bool:
    kind = item.get("check", "prefix")
    if kind == "json":
        return check_json(output, item["keys"])
    if kind == "python":
        return check_python(output, item)
    return check_prefix(output, item["answer"])
