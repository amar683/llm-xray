"""Optional AI advisor: ask Claude what to try next, using only the compact report.

Off unless the user asks for it, and nothing leaves the machine without a
preview first:

  1. `build_request` returns exactly what would be sent: the system prompt, the
     tool definitions and one user message holding the report and the goal.
     No weights, activations, prompts, file paths or host details are in it.
  2. `run_advisor` sends it. The advisor may ask drill-down queries, which are
     answered locally from files on disk (a few hundred tokens each), and ends
     by proposing edit commands.
  3. The proposal is validated locally. Running it is a separate, local step:
     the normal edit job applies the commands to a copy and measures them.

Credentials come from the environment (`ANTHROPIC_API_KEY`, or an `ant auth
login` profile); the engine never stores or asks for a key.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from xray.patch.commands import PatchError, parse_commands
from xray.report import DEFAULT_BUDGET, build_report, estimate_tokens, resolve_aliases
from xray.report.query import QueryError, run_query

MODEL = "claude-opus-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000
MAX_ROUNDS = 8
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM = """\
You are advising a developer who is studying and editing a small local language model with a \
tool called LLM X-Ray. You will be given a compact measurement report for that model and the \
developer's goal.

The report is all the evidence there is. Every number in it was measured on the developer's \
machine; treat it as ground truth and do not assume facts about the model that it does not \
state. If you need more detail, call the `query` tool. Each query is answered from files on the \
developer's machine and costs them nothing but a moment, so use it when a decision depends on a \
detail the report leaves out, and skip it when the report already settles the matter.

When you are ready, call `propose_edit` once with the edit commands to try. A proposal is not \
applied automatically: the developer reviews it, then the tool applies it to a copy of the model \
and measures accuracy, drift from the original and looping against the untouched base. So \
propose the smallest edit that tests your idea, and say what you expect it to change so the \
measurement can confirm or refute it.

The report's "Edits already tried" section shows what earlier edits cost. Use it: do not repeat \
an edit that is listed there, and weigh a new proposal against those results. If the evidence \
says the goal cannot be reached with the available commands, or that the best move is not an \
edit at all (for example, quantizing instead of pruning), say so plainly in your reply and do \
not call `propose_edit`.

Write for a developer who may not know interpretability jargon. Keep the rationale to a few \
sentences."""

TOOLS: list[dict[str, Any]] = [
    {
        "name": "query",
        "description": (
            "Ask for more detail about the model than the report gives. Answered locally from the "
            "developer's saved measurements. Forms: `layer N`, `neuron L N`, `top_neurons L SET`, "
            "`head L H`, `tensor L KIND`."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "One query, e.g. `layer 12`."}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "propose_edit",
        "description": (
            "Propose edit commands for the developer to review and measure. Call this once, when "
            "you have decided what to try. Use only the commands listed in the report."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Edit commands, one per item, exactly as they should be run.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why these commands, citing the measurements that led to them.",
                },
                "expected_effect": {
                    "type": "string",
                    "description": "What the measurement should show if the idea is right.",
                },
            },
            "required": ["commands", "rationale", "expected_effect"],
            "additionalProperties": False,
        },
    },
]


class AdvisorError(RuntimeError):
    pass


def build_request(
    model: str, goal: str, budget_tokens: int = DEFAULT_BUDGET, home: str | Path | None = None
) -> dict[str, Any]:
    """Exactly what the first request would contain, plus the local-only alias table."""
    if not goal.strip():
        raise AdvisorError("Say what you want the advisor to help with.")
    report = build_report(model, budget_tokens, home)
    user = f"{report['text']}\n\n## Goal\n{goal.strip()}"
    payload = {
        "model": MODEL,
        "effort": EFFORT,
        "system": SYSTEM,
        "tools": TOOLS,
        "messages": [{"role": "user", "content": user}],
    }
    sent = SYSTEM + user + "".join(t["description"] for t in TOOLS)
    return {
        "payload": payload,
        "tokens_estimate": estimate_tokens(sent),
        "report_tokens": report["tokens_estimate"],
        "report_fits": report["fits"],
        "aliases": report["aliases"],  # never sent
    }


def default_client():
    try:
        import anthropic
    except ImportError as e:
        raise AdvisorError(
            "The advisor needs the Anthropic SDK: run `uv sync --extra advisor`."
        ) from e
    return anthropic.Anthropic()


def _text(content: list[Any]) -> str:
    return "\n".join(b.text for b in content if b.type == "text").strip()


def run_advisor(
    model: str,
    goal: str,
    budget_tokens: int = DEFAULT_BUDGET,
    home: str | Path | None = None,
    client: Any = None,
    on_step: Callable[[str], None] = lambda _s: None,
) -> dict[str, Any]:
    """Send the report and goal, answer the advisor's queries locally, return its proposal.

    Returns {"commands" (aliases resolved to local files), "commands_as_proposed",
    "rationale", "expected_effect", "message", "queries", "usage", "rounds"}.
    `commands` is empty when the advisor recommends no edit; `message` then says why.
    """
    request = build_request(model, goal, budget_tokens, home)
    payload = request["payload"]
    client = client or default_client()
    messages = list(payload["messages"])
    queries: list[dict[str, str]] = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    for round_number in range(1, MAX_ROUNDS + 1):
        on_step("asking the advisor" if round_number == 1 else "sending answers back")
        # Refusal fallbacks are on: if a safety classifier declines, the API retries the
        # same request on a fallback model inside this call.
        try:
            response = client.beta.messages.create(
                model=payload["model"],
                max_tokens=MAX_TOKENS,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                output_config={"effort": payload["effort"]},
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                messages=messages,
            )
        except TypeError as e:
            # The SDK raises this before any request when it finds no credential.
            if "authentication" in str(e).lower():
                raise AdvisorError(
                    "No Claude API credential was found, so nothing was sent. Set "
                    "ANTHROPIC_API_KEY in the environment the engine runs in."
                ) from e
            raise
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens

        if response.stop_reason == "refusal":
            raise AdvisorError("The advisor declined this request.")
        if response.stop_reason == "max_tokens":
            raise AdvisorError("The advisor's reply was cut off before it finished.")

        calls = [b for b in response.content if b.type == "tool_use"]
        if not calls:  # it answered in words: no edit recommended
            return {
                "commands": [], "commands_as_proposed": [], "rationale": "", "expected_effect": "",
                "message": _text(response.content), "queries": queries, "usage": usage,
                "rounds": round_number,
            }  # fmt: skip

        messages.append({"role": "assistant", "content": response.content})
        results = []
        proposal = None
        for call in calls:
            if call.name == "query":
                text = str(call.input.get("query", ""))
                on_step(f"answering '{text}' locally")
                try:
                    answer, failed = run_query(model, text, home), False
                except QueryError as e:
                    answer, failed = str(e), True
                queries.append({"query": text, "answer": answer})
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": answer,
                        "is_error": failed,
                    }
                )
            elif call.name == "propose_edit":
                commands = [str(c) for c in call.input.get("commands", [])]
                try:
                    if not commands:
                        raise PatchError("No commands were given.")
                    resolved = resolve_aliases(commands, request["aliases"])
                    parse_commands(resolved)
                except (PatchError, ValueError) as e:
                    # Let the advisor correct a malformed command instead of failing the run.
                    results.append(
                        {"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                         "content": f"Those commands are not valid: {e} Please correct them."}
                    )  # fmt: skip
                    continue
                proposal = {
                    "commands": resolved,
                    "commands_as_proposed": commands,
                    "rationale": str(call.input.get("rationale", "")),
                    "expected_effect": str(call.input.get("expected_effect", "")),
                }
                results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "content": "Recorded."}
                )
            else:
                results.append(
                    {"type": "tool_result", "tool_use_id": call.id, "is_error": True,
                     "content": f"Unknown tool '{call.name}'."}
                )  # fmt: skip
        if proposal:
            return {
                **proposal, "message": _text(response.content), "queries": queries,
                "usage": usage, "rounds": round_number,
            }  # fmt: skip
        messages.append({"role": "user", "content": results})

    raise AdvisorError(f"The advisor did not reach a proposal within {MAX_ROUNDS} rounds.")
