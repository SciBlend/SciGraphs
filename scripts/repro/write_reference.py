"""Regenerate docs/reference/pipeline-options.md from the live add-on.

    env -u LD_LIBRARY_PATH blender -b --python scripts/repro/write_reference.py

Half the document comes from RNA that moves, so a hand-maintained reference goes
stale silently. Registers the working-tree property groups first, like
`_run_one.py`, or it documents the installed snapshot instead."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.insert(0, os.path.join(ROOT, "scripts", "repro"))

DEFAULT_OUT = os.path.join(ROOT, "docs", "reference", "pipeline-options.qmd")


def main(argv):
    out_path = argv[0] if argv else DEFAULT_OUT

    from _run_one import reregister_worktree_properties
    reregistered = reregister_worktree_properties()
    if reregistered:
        print("using working-tree groups: %s" % ", ".join(reregistered))

    from SciGraphs.core.repro.reference import generate_reference_markdown

    text = generate_reference_markdown()

    # Quarto front matter, so the list is a page in the sidebar rather than a
    # raw file the guide links out to.
    if out_path.endswith(".qmd"):
        text = (
            '---\ntitle: "Pipeline options reference"\n---\n\n'
            '::: {.lead}\n'
            'Every option a pipeline specification can set. Generated from the '
            'add-on itself -- the schema for the declarative stages, the live '
            'RNA for the scene property groups -- so it cannot drift from the '
            'code the way a hand-maintained list does.\n'
            ':::\n\n'
            + text.split("\n", 1)[1].lstrip("\n")
        )
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
