"""Extract the live SciGraphs vocabulary and validate pipeline specs against it.

Run inside Blender:

    env -u LD_LIBRARY_PATH blender -b --python scripts/repro/spec_doctor.py -- \
        --out /tmp/vocab.json examples/pipelines/*.json

`schema.py` hardcodes enum lists that drift from the operator RNA, so a spec can
pass `validate_pipeline` and still be rejected at call time; this script catches
that by checking specs against the registered RNA. It only reads RNA -- nothing
here writes to the scene or executes a pipeline.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import bpy  # noqa: E402

# Property groups a spec may address through `ops[].scene_props`. The spec-side
# name and the scene attribute differ ("coloring" -> scene.scigraphs_coloring).
from SciGraphs.core.repro.registry import SCENE_PROPERTY_GROUPS  # noqa: E402

GROUPS = tuple(SCENE_PROPERTY_GROUPS)

# Where a `visual` field ends up. The names differ on the two sides (spec
# `edge_style` -> operator `preset`), so the binding cannot be inferred by
# matching identifiers. Taken from executor.py:629-696.
VISUAL_BINDINGS = {
    "edge_style": ("scigraphs.apply_edge_style_preset", "preset"),
    "rendering_preset": ("scigraphs.apply_rendering_preset", "preset"),
}


def _enum_items(prop):
    try:
        return [item.identifier for item in prop.enum_items]
    except AttributeError:
        return None


def collect_vocabulary():
    """Read the registered RNA into a plain dict.

    Re-registers the working-tree property groups first; otherwise the report
    describes the installed extension in ~/.config/blender.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from _run_one import reregister_worktree_properties
        reregister_worktree_properties()
    except Exception as exc:
        print("could not refresh property groups: %s" % exc)

    scene = bpy.context.scene
    groups = {}
    for name in GROUPS:
        group = getattr(scene, SCENE_PROPERTY_GROUPS[name], None)
        if group is None:
            groups[name] = None  # not registered in this session
            continue
        props = {}
        for prop in group.bl_rna.properties:
            if prop.identifier == "rna_type":
                continue
            props[prop.identifier] = {
                "type": prop.type,
                "enum": _enum_items(prop),
            }
        groups[name] = props

    # Operator enums are the ones that actually reject a value at call time.
    operators = {}
    for attr in dir(bpy.ops.scigraphs):
        if attr.startswith("_"):
            continue
        try:
            rna = getattr(bpy.ops.scigraphs, attr).get_rna_type()
        except Exception:
            continue
        params = {}
        for prop in rna.properties:
            if prop.identifier == "rna_type":
                continue
            params[prop.identifier] = {
                "type": prop.type,
                "enum": _enum_items(prop),
            }
        operators["scigraphs.%s" % attr] = params

    return {"groups": groups, "operators": operators}


def _walk_scene_props(spec):
    """Yield (group, property_name) for every scene_props entry in a spec."""
    for op in spec.get("ops", []) or []:
        scene_props = op.get("scene_props") or {}
        for key, value in scene_props.items():
            if isinstance(value, dict) and key in GROUPS:
                for inner in value:
                    yield key, inner
            else:
                # Flat form: applied to scene.scigraphs.
                yield "scigraphs", key


def validate(spec, vocab):
    """Return a list of human-readable problems for one spec."""
    problems = []
    groups = vocab["groups"]

    for group, name in _walk_scene_props(spec):
        known = groups.get(group)
        if known is None:
            problems.append(
                "scene_props group '%s' is not registered in this session" % group
            )
        elif name not in known:
            near = [k for k in known if name.split("_")[-1] in k][:4]
            hint = (" -- nearest: %s" % ", ".join(near)) if near else ""
            problems.append(
                "scene_props %s.%s does not exist%s" % (group, name, hint)
            )

    # The schema validates these against a hardcoded list, but applies them
    # through an operator whose enum may have moved. The operator is authority.
    visual = spec.get("visual") or {}
    for field, (op_name, param) in VISUAL_BINDINGS.items():
        value = visual.get(field)
        if not value:
            continue
        params = vocab["operators"].get(op_name)
        if params is None:
            problems.append(
                "visual.%s needs operator %s, which is not registered"
                % (field, op_name)
            )
            continue
        allowed = (params.get(param) or {}).get("enum")
        if allowed and value not in allowed:
            problems.append(
                "visual.%s '%s' rejected by %s.%s -- allowed: %s"
                % (field, value, op_name, param, ", ".join(allowed))
            )

    # Operator ids referenced directly must exist, and their enum props must
    # hold values the operator still accepts.
    for op in spec.get("ops", []) or []:
        op_id = op.get("id", "")
        if not op_id.startswith("scigraphs."):
            continue
        params = vocab["operators"].get(op_id)
        if params is None:
            problems.append("ops id '%s' is not a registered operator" % op_id)
            continue
        for key, value in (op.get("props") or {}).items():
            if key not in params:
                problems.append("ops %s: no property '%s'" % (op_id, key))
                continue
            allowed = params[key]["enum"]
            if allowed and isinstance(value, str) and value not in allowed:
                problems.append(
                    "ops %s.%s '%s' invalid -- allowed: %s"
                    % (op_id, key, value, ", ".join(allowed))
                )

    return problems


def schema_drift(vocab):
    """Report enum lists that schema.py hardcodes and the RNA no longer matches."""
    from SciGraphs.core.repro.schema import SCHEMA

    drift = []
    for field, (op_name, param) in VISUAL_BINDINGS.items():
        declared = ((SCHEMA.get("visual", {}).get("properties", {})
                     .get(field, {})).get("enum"))
        params = vocab["operators"].get(op_name) or {}
        live = (params.get(param) or {}).get("enum")
        if not declared or not live:
            continue
        dead = [v for v in declared if v not in live]
        missing = [v for v in live if v not in declared]
        if dead or missing:
            drift.append({
                "field": "visual.%s" % field,
                "operator": "%s.%s" % (op_name, param),
                "declared_but_dead": dead,
                "live_but_undeclared": missing,
            })
    return drift


def main(argv):
    out_path = None
    if "--out" in argv:
        i = argv.index("--out")
        out_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    specs = [a for a in argv if a.endswith((".json", ".yaml", ".yml"))]

    vocab = collect_vocabulary()
    if out_path:
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(vocab, handle, indent=2, sort_keys=True)
        print("vocabulary -> %s" % out_path)

    registered = [g for g, v in vocab["groups"].items() if v is not None]
    print("groups registered: %s" % ", ".join(registered) or "(none)")
    print("operators found:   %d" % len(vocab["operators"]))

    drift = schema_drift(vocab)
    if drift:
        print()
        print("SCHEMA DRIFT -- schema.py hardcodes enums the RNA no longer has:")
        for entry in drift:
            print("  %s -> %s" % (entry["field"], entry["operator"]))
            if entry["declared_but_dead"]:
                print("      accepted by schema, rejected by operator: %s"
                      % ", ".join(entry["declared_but_dead"]))
            if entry["live_but_undeclared"]:
                print("      valid but the schema refuses them: %s"
                      % ", ".join(entry["live_but_undeclared"]))
    print()

    total_bad = 0
    for path in specs:
        try:
            with open(path, encoding="utf-8") as handle:
                if path.endswith((".yaml", ".yml")):
                    import yaml
                    spec = yaml.safe_load(handle)
                else:
                    spec = json.load(handle)
        except Exception as exc:
            print("%-44s READ FAILED  %s" % (os.path.basename(path), exc))
            total_bad += 1
            continue

        problems = validate(spec, vocab)
        name = os.path.basename(path)
        if problems:
            total_bad += 1
            print("%-44s %d problem(s)" % (name, len(problems)))
            for problem in problems:
                print("    - %s" % problem)
        else:
            print("%-44s ok" % name)

    print()
    print("%d of %d specs have problems" % (total_bad, len(specs)))
    return 1 if total_bad else 0


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    code = main(argv)
    # Blender segfaults during interpreter teardown with this add-on, which
    # would replace the verdict with 134.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
