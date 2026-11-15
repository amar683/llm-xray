"""Generate docs/cli.md from the commands themselves, so the two cannot drift.

Run from the engine folder:  uv run python scripts/make_cli_docs.py
"""

from pathlib import Path

import click
import typer

from xray.cli import app

OUT = Path(__file__).parent.parent.parent / "docs" / "cli.md"


def render() -> str:
    root = typer.main.get_command(app)
    lines = [
        "# `xray` command reference",
        "",
        "Generated from the commands by `engine/scripts/make_cli_docs.py`. Do not edit by hand.",
        "",
        "| Command | What it does |",
        "|---|---|",
    ]
    commands = sorted(root.commands.items())
    for name, cmd in commands:
        first = (cmd.help or "").strip().splitlines()[0] if cmd.help else ""
        lines.append(f"| [`xray {name}`](#xray-{name}) | {first} |")
    for name, cmd in commands:
        args = [p for p in cmd.params if isinstance(p, click.Argument)]
        opts = [p for p in cmd.params if isinstance(p, click.Option) and p.name != "help"]
        usage = " ".join(
            ["xray", name, *[a.name.upper() for a in args], "[OPTIONS]" if opts else ""]
        )
        lines += [
            "",
            f"## `xray {name}`",
            "",
            (cmd.help or "").strip(),
            "",
            "```",
            usage.strip(),
            "```",
        ]
        if args or opts:
            lines += ["", "| Parameter | Description | Default |", "|---|---|---|"]
        for a in args:
            lines.append(f"| `{a.name.upper()}` | {getattr(a, 'help', None) or ''} | required |")
        for o in opts:
            flags = ", ".join(f"`{f}`" for f in [*o.opts, *o.secondary_opts])
            default = (
                "" if o.default in (None, False, ()) or callable(o.default) else f"`{o.default}`"
            )
            if o.required:
                default = "required"
            env = f" Env: `{o.envvar}`." if o.envvar else ""
            lines.append(f"| {flags} | {(o.help or '').strip()}{env} | {default} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")
