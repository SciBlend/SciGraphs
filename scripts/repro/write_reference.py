"""Regenerate docs/reference/pipeline-options.md from the live add-on.

    env -u LD_LIBRARY_PATH blender -b --python scripts/repro/write_reference.py

Half the document comes from RNA that moves, so a hand-maintained reference
goes stale silently. Registers the working-tree property groups first, for the
same reason `_run_one.py` does: Blender auto-loads the installed extension and
registered classes do not follow sys.path, so an un-nudged run documents the
snapshot in ~/.config/blender rather than the working tree.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.join(ROOT, "scripts", "repro"))

DEFAULT_OUT = os.path.join(ROOT, "docs", "reference", "pipeline-options.md")


def main(argv):
    out_path = argv[0] if argv else DEFAULT_OUT

    from _run_one import reregister_worktree_properties
    reregistered = reregister_worktree_properties()
    if reregistered:
        print("using working-tree groups: %s" % ", ".join(reregistered))

    from SciGraphs.core.repro.reference import generate_reference_markdown

    text = generate_reference_markdown()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)

    print("wrote %s (%d lines)" % (out_path, text.count("\n") + 1))
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    code = main(argv)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
