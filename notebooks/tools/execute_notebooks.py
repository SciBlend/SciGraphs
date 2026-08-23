"""Execute the notebooks against the live Blender kernel and store the outputs.

    ./notebooks/tools/launch.sh --opengl   # another terminal, first
    python3 notebooks/tools/execute_notebooks.py [04 06]

`jupyter nbconvert --execute` starts its own kernel, where `import bpy` fails on
cell one, so this drives the one `launch.sh` started; restarting an embedded
kernel restarts Blender. Outputs land in the `.ipynb` in place, so run
build -> verify -> execute.
"""

import argparse
import base64
import json
import os
import pathlib
import sys
import time

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
NOTEBOOKS_DIR = TOOLS_DIR.parent

# One connection file per Blender version, so glob the version out.
CONNECTION_GLOB = "*/extensions/.user/*/jupyter_blender/connection_cache/*.json"

# Long enough for notebook 07, which waits on Overpass rate limiting.
CELL_TIMEOUT = 900


def find_connection_file():
    base = pathlib.Path.home() / ".config" / "blender"
    candidates = sorted(base.glob(CONNECTION_GLOB), key=lambda p: p.stat().st_mtime)
    if not candidates:
        sys.exit(
            "No kernel connection file found.\n"
            "Start Blender with the notebook server first:\n"
            "    ./notebooks/tools/launch.sh --opengl")
    return candidates[-1]


def load_client(connection_file):
    try:
        from jupyter_client import BlockingKernelClient
    except ImportError:
        sys.exit(
            "jupyter_client is not importable from this Python.\n"
            "Use the environment that runs JupyterLab, e.g.\n"
            "    ~/.local/share/blender-jupyter/venv/bin/python "
            "notebooks/tools/execute_notebooks.py")

    client = BlockingKernelClient(connection_file=str(connection_file))
    client.load_connection_file()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
    except Exception as exc:  # noqa: BLE001 - report whatever went wrong
        sys.exit(f"The kernel did not answer ({exc}). Is Blender still open?")
    return client


def run_cell(client, source, timeout=CELL_TIMEOUT):
    """Execute one cell and collect its outputs in nbformat shape."""
    msg_id = client.execute(source, allow_stdin=False)
    outputs, streams = [], {}
    deadline = time.time() + timeout
    failed = None

    while time.time() < deadline:
        try:
            msg = client.get_iopub_msg(timeout=5)
        except Exception:  # noqa: BLE001 - an empty queue is not an error
            continue
        if msg["parent_header"].get("msg_id") != msg_id:
            continue

        kind, content = msg["msg_type"], msg["content"]

        if kind == "stream":
            # Merge consecutive writes, or a print loop grows one per line.
            name = content["name"]
            if name in streams:
                streams[name]["text"] += content["text"]
            else:
                out = {"output_type": "stream", "name": name,
                       "text": content["text"]}
                streams[name] = out
                outputs.append(out)

        elif kind in ("display_data", "update_display_data"):
            outputs.append({"output_type": "display_data",
                            "data": content["data"],
                            "metadata": content.get("metadata", {})})
            streams.clear()

        elif kind == "execute_result":
            outputs.append({"output_type": "execute_result",
                            "data": content["data"],
                            "metadata": content.get("metadata", {}),
                            "execution_count": content.get("execution_count")})
            streams.clear()

        elif kind == "error":
            outputs.append({"output_type": "error",
                            "ename": content["ename"],
                            "evalue": content["evalue"],
                            "traceback": content["traceback"]})
            failed = f"{content['ename']}: {content['evalue']}"
            streams.clear()

        elif kind == "status" and content["execution_state"] == "idle":
            break
    else:
        failed = f"timed out after {timeout}s"

    return outputs, failed


def split_lines(notebook):
    """Store every multiline string as a list of lines, as nbformat writes it.

    The kernel sends `text` and `data[mime]` as single strings and dumping them
    verbatim is still valid nbformat, but Quarto reads them as arrays: a string
    `text/plain` next to an image crashes `quarto render` with
    `textPlain.some is not a function`, and a string `text/html` would hit
    `.join` the same way. `application/json` is an object, not text.
    """
    for cell in notebook.get("cells", ()):
        source = cell.get("source")
        if isinstance(source, str):
            cell["source"] = source.splitlines(True)
        for out in cell.get("outputs", ()) or ():
            text = out.get("text")
            if isinstance(text, str):
                out["text"] = text.splitlines(True)
            data = out.get("data") or {}
            for mime, payload in data.items():
                if mime != "application/json" and isinstance(payload, str):
                    data[mime] = payload.splitlines(True)
    return notebook


# The kernel outlives every edit: after `import nb`, Python's cache serves
# that copy for the rest of Blender's life and outputs come from the old code.
# Registered classes are kept, since re-registering duplicates every panel.
RELOAD_HELPERS = """
import sys

# The two wheels are purged along with the add-on. They were left out while
# they were dependencies that only ever arrived installed; now they are built
# from `engine/` and `core/` in this same tree, so an edit there has to reach a
# kernel that has already imported them. Renaming one engine function and not
# purging it cost two notebooks a run.
_HELPERS = ('nb',)
_stale = [n for n in sys.modules
          if n in _HELPERS
          or any(n.startswith(h + '.') for h in _HELPERS)
          or n == 'SciGraphs.api' or n.startswith('SciGraphs.api.')
          or n.startswith('SciGraphs.core.') or n.startswith('SciGraphs.ui.')
          or n == 'scigraphs_engine' or n.startswith('scigraphs_engine.')
          or n == 'scigraphs_core' or n.startswith('scigraphs_core.')]
for _name in _stale:
    del sys.modules[_name]
del _name, _stale, _HELPERS

# Empty the importer's in-memory graph cache between notebooks.
#
# `verify_notebooks.py` gives every notebook a fresh Blender, so it never sees
# this; the live kernel keeps one process for all twenty, and the cache is a
# module-level dict keyed by **object name** (`core/osmnx/graph_cache.py:22`).
# Notebooks reuse names, several build a `Street_Network` - so a notebook that
# imports a graph and asks for it back can be handed the previous notebook's
# graph instead, with different parameters and different attributes. It surfaces
# far downstream as a missing attribute rather than as a wrong graph.
#
# It is order-dependent by construction, so it stayed hidden until the notebooks
# were renumbered and the running order changed.
try:
    import importlib
    _bl = next(k for k in __import__('bpy').context.preferences.addons.keys()
               if k.rsplit('.', 1)[-1].lower() == 'scigraphs')
    _imp = importlib.import_module(_bl + '.core.data_io.importer')
    if hasattr(_imp, '_osmnx_graph_cache'):
        _imp._osmnx_graph_cache.clear()
    del _bl, _imp, importlib
except Exception as _exc:
    print('could not clear the importer graph cache (%s)' % _exc)
"""


def execute(client, path, verbose=True):
    run_cell(client, RELOAD_HELPERS, timeout=60)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    count = 0
    errors = []
    started = time.time()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if not source.strip():
            continue
        count += 1
        outputs, failed = run_cell(client, source)
        cell["outputs"] = outputs
        cell["execution_count"] = count
        if failed:
            errors.append(f"cell {index + 1}: {failed}")
        if verbose:
            mark = "!" if failed else "."
            sys.stdout.write(mark)
            sys.stdout.flush()

    path.write_text(
        json.dumps(split_lines(notebook), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8")

    images = sum(
        1 for cell in notebook["cells"]
        for out in cell.get("outputs", [])
        if "image/png" in out.get("data", {}))

    return {"notebook": path.name, "cells": count, "images": images,
            "errors": errors, "seconds": round(time.time() - started, 1)}


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("which", nargs="*",
                        help="prefixes to run, e.g. 04 06 (default: all)")
    args = parser.parse_args(argv)

    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if args.which:
        notebooks = [p for p in notebooks if p.stem.startswith(tuple(args.which))]
    if not notebooks:
        sys.exit("no matching notebooks")

    connection_file = find_connection_file()
    print(f"kernel: {connection_file}")
    client = load_client(connection_file)
    print(f"executing {len(notebooks)} notebook(s) against the live Blender\n")

    results = []
    for path in notebooks:
        print(f"  {path.name:<30}", end="", flush=True)
        result = execute(client, path)
        results.append(result)
        state = "ok" if not result["errors"] else f"{len(result['errors'])} FAILED"
        print(f"  {result['cells']:>3} cells, {result['images']:>2} images, "
              f"{result['seconds']:>6}s  [{state}]")
        for error in result["errors"][:3]:
            print(f"      {error}")

    total_images = sum(r["images"] for r in results)
    failed = [r for r in results if r["errors"]]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} notebooks executed clean, "
          f"{total_images} images stored")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
