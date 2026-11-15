"""Run a notebook's code cells in order, in one namespace, without Jupyter.

    uv run python scripts/run_notebook.py ../examples/quickstart.ipynb

Used by the tests to prove the examples still work. It executes the code the
same way a notebook would, top to bottom, but does not start a Jupyter kernel.
"""

import json
import sys
from pathlib import Path


def run(path: Path, quiet: bool = False) -> dict:
    notebook = json.loads(path.read_text())
    namespace: dict = {"__name__": "__main__"}
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    for n, cell in enumerate(cells, 1):
        source = "".join(cell["source"])
        if not quiet:
            print(f"--- cell {n}/{len(cells)} ---")
        exec(compile(source, f"{path.name}[cell {n}]", "exec"), namespace)  # noqa: S102
    return namespace


if __name__ == "__main__":
    run(Path(sys.argv[1]))
