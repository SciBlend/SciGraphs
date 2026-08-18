"""Refresh docs/notebooks/_ipynb/ from the executed notebooks in notebooks/.
Quarto's `embed` cannot reach outside the project directory and an empty copy
fails the build with `Unable to embed content from notebook`. This pre-render
hook copies rather than symlinks: `quarto render` refuses those too.
"""

import json
import pathlib
import shutil
import sys

DOCS = pathlib.Path(__file__).resolve().parent
SOURCE = DOCS.parent / "notebooks"
TARGET = DOCS / "notebooks" / "_ipynb"


def _has_outputs(path) -> bool:
    try:
        with open(path, encoding="utf-8") as handle:
            notebook = json.load(handle)
    except (OSError, ValueError):
        # An unreadable destination must not veto a refresh.
        return False
    return any(cell.get("outputs")
               for cell in notebook.get("cells", ())
               if cell.get("cell_type") == "code")


def main() -> int:
    if not SOURCE.is_dir():
        # A docs-only checkout is legitimate, so warn without failing.
        print(f"[sync-notebooks] no {SOURCE}, leaving _ipynb/ as it is",
              file=sys.stderr)
        return 0

    TARGET.mkdir(parents=True, exist_ok=True)

    # Numbered notebooks only: a bare `0*.ipynb` silently skips 10-19.
    sources = sorted(SOURCE.glob("[0-9][0-9]_*.ipynb"))
    if not sources:
        print(f"[sync-notebooks] no notebooks matched in {SOURCE}",
              file=sys.stderr)
        return 0

    copied = 0
    refused = []
    for src in sources:
        dst = TARGET / src.name
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        # NEVER replace a notebook that has outputs with one that has none.
        # `build_notebooks.py` wipes outputs, `execute_notebooks.py` restores
        # them from a live Blender, and newer-wins in between would overwrite
        # 126 figures with empty cells. Not in git; ~30 minutes to regenerate.
        if dst.exists() and _has_outputs(dst) and not _has_outputs(src):
            refused.append(src.name)
            continue
        shutil.copy2(src, dst)
        copied += 1

    if refused:
        print(f"[sync-notebooks] REFUSED to overwrite {len(refused)} executed "
              f"notebook(s) with output-stripped sources: "
              f"{', '.join(refused[:4])}"
              f"{' ...' if len(refused) > 4 else ''}\n"
              f"[sync-notebooks] run notebooks/tools/execute_notebooks.py "
              f"against a live Blender to refresh them, then render again.",
              file=sys.stderr)

    # A renamed notebook otherwise leaves a stale page behind.
    names = {p.name for p in sources}
    removed = 0
    for stale in TARGET.glob("*.ipynb"):
        if stale.name not in names:
            stale.unlink()
            removed += 1

    print(f"[sync-notebooks] {len(sources)} notebooks, {copied} refreshed, "
          f"{removed} stale removed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
