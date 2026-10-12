"""Execute a project notebook's plain Python cells and save visible outputs.

This runner also works when a Jupyter kernel cannot start in a restricted shell.
"""

import argparse
import contextlib
import io
import os
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def display(value):
    if hasattr(value, "to_string"):
        print(value.to_string(index=False))
    else:
        print(value)


def run(path: Path):
    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__", "display": display}
    os.chdir(ROOT)
    for number, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(cell.source, f"{path.name}:cell-{number}", "exec"), namespace)
        except Exception:
            print(output.getvalue())
            raise
        cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=output.getvalue())]
        cell.execution_count = number + 1
        print(f"{path.name}: cell {number + 1} passed", flush=True)
    nbformat.write(notebook, path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", help="Notebook paths; all three by default")
    args = parser.parse_args()
    paths = [ROOT / name for name in args.notebooks] if args.notebooks else sorted((ROOT / "notebooks").glob("*.ipynb"))
    for item in paths:
        run(item)
