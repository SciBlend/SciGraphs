"""Turn the percent-format sources in `_src/` into `.ipynb` files, which keeps
them runnable head-first inside `blender -b` under `verify_notebooks.py`.

    python3 notebooks/tools/build_notebooks.py [04]

At column zero `# %%` opens a code cell and `# %% [markdown]` a markdown cell,
whose "# " prefixes are stripped and whose marker text becomes its first line.
"""

import json
import pathlib
import sys

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
NOTEBOOKS_DIR = TOOLS_DIR.parent
SRC_DIR = NOTEBOOKS_DIR / "_src"

KERNEL = {
    "display_name": "Blender",
    "language": "python",
    "name": "blender",
}


def split_cells(text):
    cells = []
    kind = "code"
    buffer = []

    def flush():
        if not buffer:
            return
        source = "\n".join(buffer).strip("\n")
        if source.strip():
            cells.append((kind, source))
        buffer.clear()

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            marker = line[len("# %%"):].strip()
            if marker.startswith("[markdown]"):
                kind = "markdown"
                rest = marker[len("[markdown]"):].strip()
                if rest:
                    buffer.append(rest)
            else:
                kind = "code"
            continue
        buffer.append(line)
    flush()

    out = []
    for kind, source in cells:
        if kind == "markdown":
            lines = []
            for line in source.splitlines():
                if line.startswith("# "):
                    lines.append(line[2:])
                elif line == "#":
                    lines.append("")
                else:
                    lines.append(line)
            source = "\n".join(lines).strip("\n")
        out.append((kind, source))
    return out


def to_notebook(cells):
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": KERNEL,
            "language_info": {"name": "python", "file_extension": ".py"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for kind, source in cells:
        lines = source.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb["cells"].append(cell)
    return nb


def build(path):
    cells = split_cells(path.read_text(encoding="utf-8"))
    target = NOTEBOOKS_DIR / (path.stem + ".ipynb")
    target.write_text(
        json.dumps(to_notebook(cells), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")
    code = sum(1 for k, _ in cells if k == "code")
    print(f"{target.name:<36} {len(cells):>3} cells ({code} code)")
    return target


def main(argv):
    sources = sorted(SRC_DIR.glob("*.py"))
    if argv:
        wanted = tuple(argv)
        sources = [p for p in sources if p.stem.startswith(wanted)]
    if not sources:
        print("no matching sources in", SRC_DIR)
        return 1
    for path in sources:
        build(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
