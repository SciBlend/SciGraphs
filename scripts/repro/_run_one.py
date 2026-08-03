"""Execute one pipeline specification inside Blender and report the outcome.

One Blender process per spec, so a crash cannot affect the next one:

    env -u LD_LIBRARY_PATH blender -b --python scripts/repro/_run_one.py -- \
        --spec examples/pipelines/figures/fig1_flatfile_lesmiserables.json \
        --report /tmp/result.json

The outcome goes to a JSON report rather than stdout, which Blender fills with
log lines that look plausible even when the pipeline failed halfway.
"""

import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_args(argv):
    args = {"spec": None, "base_dir": None, "report": None}
    for i, token in enumerate(argv):
        if token == "--spec":
            args["spec"] = argv[i + 1]
        elif token == "--base-dir":
            args["base_dir"] = argv[i + 1]
        elif token == "--report":
            args["report"] = argv[i + 1]
    return args


def reregister_worktree_properties():
    """Force the working tree's property groups over the installed add-on's.

    Blender auto-loads the installed extension and registered classes do not
    follow sys.path, so without this a spec's new fields are silently dropped.
    The guard checks that EVERY expected property is present, not just one: a
    background sync copies them across gradually, so a single-name check goes
    green while the rest are still missing.

    Returns a list of the groups that were re-registered.
    """
    import bpy

    reregistered = []

    def _ensure(module, pointer, expected):
        group = getattr(bpy.context.scene, pointer, None)
        if group is not None and all(hasattr(group, n) for n in expected):
            return
        try:
            module.register()
            reregistered.append(pointer)
        except Exception as exc:
            print("could not re-register %s: %s" % (pointer, exc))

    try:
        from SciGraphs.ui.coloring import properties as coloring_properties
        _ensure(
            coloring_properties,
            "scigraphs_coloring",
            ("color_norm", "color_gamma", "clip_low_pct", "clip_high_pct",
             "vmin", "vmax", "auto_range", "reverse"),
        )
    except Exception as exc:
        print("coloring properties unavailable: %s" % exc)

    try:
        from SciGraphs.properties import viz_properties
        _ensure(
            viz_properties,
            "scigraphs_viz",
            ("node_scale", "node_resolution", "edge_thickness", "edge_resolution"),
        )
    except Exception as exc:
        print("viz properties unavailable: %s" % exc)

    if reregistered:
        print("re-registered from working tree: %s" % ", ".join(reregistered))
    return reregistered


def main(argv):
    args = parse_args(argv)
    report = {
        "spec": args["spec"],
        "success": False,
        "pipeline_hash": None,
        "output_dir": None,
        "manifest_path": None,
        "artifacts": [],
        "errors": [],
        "warnings": [],
        "duration_ms": 0,
    }

    try:
        # Scene clearing and camera framing are pipeline stages
        # (meta.clear_scene, render.frame_camera); this runner only executes.
        report["reregistered"] = reregister_worktree_properties()

        # Not re-exported by the package __init__, so import the module.
        from SciGraphs.core.repro.executor import run_pipeline

        result = run_pipeline(
            args["spec"],
            base_dir=args["base_dir"],
            stop_on_error=False,
            verbose=True,
        )
        report.update({
            "success": bool(result.success),
            "pipeline_hash": result.pipeline_hash,
            "output_dir": result.output_dir,
            "manifest_path": result.manifest_path,
            "artifacts": list(result.artifacts),
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "duration_ms": int(result.duration_ms),
        })
    except Exception:
        report["errors"].append(traceback.format_exc())

    if args["report"]:
        os.makedirs(os.path.dirname(os.path.abspath(args["report"])), exist_ok=True)
        with open(args["report"], "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

    print("RESULT success=%s artifacts=%d errors=%d"
          % (report["success"], len(report["artifacts"]), len(report["errors"])))
    for err in report["errors"]:
        print("  ERROR %s" % err.strip().splitlines()[-1])
    return 0 if report["success"] else 1


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    code = main(argv)
    # Blender segfaults during interpreter teardown with this add-on, which
    # would replace the exit code with 134 after a clean run.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
