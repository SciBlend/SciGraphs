"""Verify the add-on works from a public-only checkout, by blocking every
private module through a sys.meta_path finder and running a pipeline. Exit 0
means the public tree is self-sufficient.

    env -u LD_LIBRARY_PATH blender -b --python engine/check_without_engine.py -- \
        --spec examples/pipelines/figures/fig1_flatfile_lesmiserables.json
"""

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Made absent, to prove the pipeline runs without the render engine.
PRIVATE_MODULES = (
    "SciGraphs.ui.gpu_render",
    "SciGraphs.core.render",
    "scigraphs_engine",
)


class _BlockPrivate:
    """Raises from find_spec, so no later finder can resolve a blocked name."""

    def __init__(self, names):
        self.names = tuple(names)
        self.blocked = []

    def find_spec(self, fullname, path=None, target=None):
        if self._matches(fullname):
            self.blocked.append(fullname)
            raise ModuleNotFoundError(
                "%s is absent from a public checkout" % fullname,
                name=fullname,
            )
        return None

    def _matches(self, fullname):
        return any(
            fullname == name or fullname.startswith(name + ".")
            for name in self.names
        )


def purge_already_imported(names):
    removed = []
    for key in list(sys.modules):
        if any(key == n or key.startswith(n + ".") for n in names):
            del sys.modules[key]
            removed.append(key)
    return removed


def parse_args(argv):
    args = {"spec": None, "base_dir": None}
    for i, token in enumerate(argv):
        if token == "--spec":
            args["spec"] = argv[i + 1]
        elif token == "--base-dir":
            args["base_dir"] = argv[i + 1]
    return args


def main(argv):
    args = parse_args(argv)

    removed = purge_already_imported(PRIVATE_MODULES)
    print("purged %d already-imported private module(s)" % len(removed))

    blocker = _BlockPrivate(PRIVATE_MODULES)
    sys.meta_path.insert(0, blocker)

    failures = []

    # The public packages must import with the engine gone.
    for name in (
        "SciGraphs",
        "SciGraphs.core.repro.schema",
        "SciGraphs.core.repro.executor",
        "SciGraphs.core.mesh.geometry",
        "SciGraphs.core.coloring.colormaps",
        "SciGraphs.core.visualization.text_overlay",
        "SciGraphs.ui.coloring.functions",
        "SciGraphs.properties.viz_properties",
    ):
        try:
            importlib.import_module(name)
            print("  import ok    %s" % name)
        except Exception as exc:
            failures.append("import %s: %s" % (name, exc))
            print("  IMPORT FAIL  %s: %s" % (name, exc))

    # The guarded call sites must degrade rather than raise.
    for name in (
        "SciGraphs.properties.edge_style_properties",
        "SciGraphs.ui.operators.scigraphs.layout_operators",
        "SciGraphs.ui.operators.scigraphs.data_operators",
        "SciGraphs.ui",
    ):
        try:
            importlib.import_module(name)
            print("  guarded ok   %s" % name)
        except Exception as exc:
            failures.append("guarded %s: %s" % (name, exc))
            print("  GUARD FAIL   %s: %s" % (name, exc))

    if args["spec"] and not failures:
        print("running %s" % os.path.basename(args["spec"]))
        try:
            from SciGraphs.core.repro.executor import run_pipeline
            result = run_pipeline(
                args["spec"], base_dir=args["base_dir"],
                stop_on_error=False, verbose=False,
            )
            print("  pipeline success=%s artifacts=%d errors=%d"
                  % (result.success, len(result.artifacts), len(result.errors)))
            for err in result.errors:
                print("    %s" % err)
            if not result.success:
                failures.append("pipeline did not complete")
        except Exception as exc:
            failures.append("pipeline raised: %s" % exc)
            print("  PIPELINE RAISED %s" % exc)

    print()
    if blocker.blocked:
        print("blocked imports attempted: %s"
              % ", ".join(sorted(set(blocker.blocked))))
    if failures:
        print("PUBLIC CHECKOUT IS BROKEN -- %d problem(s)" % len(failures))
        for problem in failures:
            print("  - %s" % problem)
        return 1

    print("PUBLIC CHECKOUT IS SELF-SUFFICIENT")
    return 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    code = main(argv)
    # Blender segfaults during interpreter teardown with this add-on, which
    # would replace the exit code with 134 after a clean run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
