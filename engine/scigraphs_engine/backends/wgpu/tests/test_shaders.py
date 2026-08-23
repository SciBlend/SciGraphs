# The shader loader: the schema, what it refuses, and what wgsl/ actually holds.
# Half the file is rejections, because a mistyped format reads one column
# shifted and draws something that looks nearly right instead of failing.

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from harness import Report, finish   # noqa: E402

from scigraphs_engine.mesh import ShaderRef                      # noqa: E402
from scigraphs_engine.backends.wgpu import device, shaders       # noqa: E402

GOOD = {
    "vertex_entry": "vs_main",
    "fragment_entry": "fs_main",
    "attributes": [
        {"name": "pos", "location": 0, "format": "float32x3"},
        {"name": "color", "location": 1, "format": "float32x4"},
    ],
    "bindings": [
        {"name": "camera", "group": 0, "binding": 0, "type": "uniform",
         "visibility": ["vertex", "fragment"]},
    ],
}


def check_variant_names(r):
    cases = {
        ShaderRef("sphere"): "sphere",
        ShaderRef("sphere", filtered=True): "sphere_f",
        ShaderRef("sphere", animated=True): "sphere_d",
        ShaderRef("sphere", filtered=True, animated=True): "sphere_fd",
        ShaderRef("round_point"): "round_point",
    }
    for ref, want in cases.items():
        r.check(f"variant_name({ref.base}, {ref.filtered}, {ref.animated})",
                shaders.variant_name(ref) == want, shaders.variant_name(ref))
    r.check("a plain string passes through",
            shaders.variant_name("ribbon_id") == "ribbon_id")


def check_schema_accepts(r):
    decl = shaders.parse_declaration(copy.deepcopy(GOOD), "good")
    r.check("entry points parsed",
            (decl["vertex_entry"], decl["fragment_entry"])
            == ("vs_main", "fs_main"))
    r.check("attributes parsed in order",
            [a.name for a in decl["attributes"]] == ["pos", "color"])
    r.check("columns derived from format",
            [a.columns for a in decl["attributes"]] == [3, 4])
    r.check("camera binding parsed",
            decl["bindings"][0].name == "camera"
            and decl["bindings"][0].binding == 0)
    r.check("expand defaults to absent", decl["expand"] is None)
    r.check("cull defaults to none", decl["cull"] == "none")
    r.check("depth_compare defaults to less-equal",
            decl["depth_compare"] == "less-equal")

    with_expand = copy.deepcopy(GOOD)
    with_expand["expand"] = {"mode": "instanced_quad", "vertex_count": 4,
                             "topology": "triangle-strip"}
    decl = shaders.parse_declaration(with_expand, "expander")
    r.check("expand parsed", decl["expand"] is not None
            and decl["expand"].vertex_count == 4
            and decl["expand"].topology == "triangle-strip")

    dyn = copy.deepcopy(GOOD)
    dyn["bindings"].append({"name": "draw", "group": 0, "binding": 1,
                            "type": "uniform", "dynamic": True, "size": 32})
    decl = shaders.parse_declaration(dyn, "dyn")
    r.check("dynamic binding flagged", decl["bindings"][1].dynamic is True)
    r.check("declared binding size kept", decl["bindings"][1].size == 32)


def check_schema_rejects(r):
    def bad(**changes):
        decl = copy.deepcopy(GOOD)
        for key, value in changes.items():
            if value is _DROP:
                decl.pop(key, None)
            else:
                decl[key] = value
        return decl

    cases = [
        ("missing vertex_entry", bad(vertex_entry=_DROP)),
        ("missing fragment_entry", bad(fragment_entry=_DROP)),
        ("non-string entry point", bad(vertex_entry=7)),
        ("missing attributes", bad(attributes=_DROP)),
        ("empty attributes", bad(attributes=[])),
        ("attribute missing a name", bad(attributes=[{"location": 0,
                                                      "format": "float32x3"}])),
        ("unknown vertex format", bad(attributes=[
            {"name": "pos", "location": 0, "format": "float16x4"}])),
        ("duplicate attribute name", bad(attributes=[
            {"name": "pos", "location": 0, "format": "float32x3"},
            {"name": "pos", "location": 1, "format": "float32x3"}])),
        ("duplicate location", bad(attributes=[
            {"name": "pos", "location": 0, "format": "float32x3"},
            {"name": "col", "location": 0, "format": "float32x4"}])),
        ("two bindings on one slot", bad(bindings=[
            {"name": "a", "group": 0, "binding": 0},
            {"name": "b", "group": 0, "binding": 0}])),
        ("unknown binding type", bad(bindings=[
            {"name": "a", "group": 0, "binding": 0, "type": "sampler"}])),
        ("unknown visibility", bad(bindings=[
            {"name": "a", "group": 0, "binding": 0, "visibility": ["geometry"]}])),
        ("unknown expand mode", bad(expand={"mode": "geometry_shader"})),
        ("non-object expand", bad(expand="yes")),
        ("unknown cull", bad(cull="sideways")),
        ("unknown depth_compare", bad(depth_compare="roughly")),
        ("top level is not an object", ["vs_main"]),
    ]
    for name, decl in cases:
        r.raises(f"rejects: {name}", shaders.ShaderError,
                 shaders.parse_declaration, decl, "bad")


class _DROP:
    pass


def check_schema1_normalization(r):
    decl = shaders.parse_declaration({
        "schema": 1,
        "name": "sphere",
        "entry_points": {"vertex": "vs_main", "fragment": "fs_main"},
        "topology": "triangle-list",
        "indexed": True,
        "attributes": [
            {"name": "pos", "location": 0, "format": "float32x3",
             "array_stride": 12},
            {"name": "color", "location": 1, "format": "float32x4",
             "array_stride": 16},
        ],
        "bind_groups": [{
            "group": 0,
            "bindings": [
                {"binding": 0, "name": "u_camera", "type": "uniform",
                 "size": 208, "visibility": ["vertex", "fragment"]},
                {"binding": 1, "name": "u_light", "type": "uniform",
                 "size": 80, "visibility": ["fragment"]},
            ],
        }],
        "writes_frag_depth": True,
        "state": {"blend": "none", "depth_test": True, "depth_write": True,
                  "depth_compare": "less", "cull": "none"},
    }, "schema1")
    r.check("schema-1 entry points", decl["vertex_entry"] == "vs_main")
    r.check("schema-1 attributes",
            [a.name for a in decl["attributes"]] == ["pos", "color"])
    r.check("schema-1 u_camera aliases to camera",
            decl["bindings"][0].name == "camera")
    r.check("schema-1 keeps the declared 208-byte camera size",
            decl["bindings"][0].size == 208)
    r.check("schema-1 leaves a caller block's name alone",
            decl["bindings"][1].name == "u_light")
    r.check("schema-1 depth_compare honored", decl["depth_compare"] == "less")
    r.check("schema-1 declared topology carried",
            decl["declared_topology"] == "triangle-list")
    r.check("schema-1 state becomes a DrawState",
            decl["default_state"] is not None
            and decl["default_state"].blend == "NONE"
            and decl["default_state"].depth_test is True)
    r.raises("schema-1 with half an entry_points is rejected",
             shaders.ShaderError, shaders.parse_declaration,
             {"entry_points": {"vertex": "vs_main"}, "attributes": []},
             "half")


def check_library(r, ctx):
    lib = shaders.ShaderLibrary(ctx)
    available = lib.available()
    r.check("wgsl/ has at least round_point", "round_point" in available,
            ", ".join(available))
    r.note(f"shaders present: {', '.join(available) or '(none)'}")

    rp = lib.get(ShaderRef("round_point"))
    r.check("round_point loads", rp is not None)
    r.check("round_point declares pos and color",
            [a.name for a in rp.attributes] == ["pos", "color"])
    r.check("round_point expands to instances", rp.expand is not None
            and rp.expand.mode == "instanced_quad")
    r.check("round_point wants the camera block", rp.wants_camera)
    r.check("round_point wants the per-draw block", rp.wants_draw)
    r.check("round_point compiled to a module", rp.module is not None)

    r.check("a shader that does not exist is None, not an error",
            lib.get(ShaderRef("no_such_shader")) is None)
    r.check("the None is cached too",
            lib.get(ShaderRef("no_such_shader")) is None)
    r.check("get() is memoized", lib.get(ShaderRef("round_point")) is rp)

    # Every shader in the directory must load, whoever added it.
    for name in available:
        shader = lib.get(name)
        r.check(f"{name} loads and compiles", shader is not None
                and shader.module is not None,
                f"{len(shader.attributes)} attrs, "
                f"{len(shader.bindings)} bindings" if shader else "")
        if shader is None:
            continue
        locs = [a.location for a in shader.attributes]
        r.check(f"{name} locations are unique", len(set(locs)) == len(locs))
        r.check(f"{name} binds camera at group 0",
                all(b.group == 0 for b in shader.bindings),
                str([f"{b.name}@{b.group}.{b.binding}" for b in shader.bindings]))
    return lib


def check_broken_pair_raises(r, ctx, tmpdir):
    os.makedirs(tmpdir, exist_ok=True)
    with open(os.path.join(tmpdir, "orphan.wgsl"), "w") as fh:
        fh.write("@vertex fn vs_main() -> @builtin(position) vec4<f32> "
                 "{ return vec4<f32>(0.0); }")
    lib = shaders.ShaderLibrary(ctx, tmpdir)
    r.raises("a .wgsl with no .json raises", shaders.ShaderError,
             lib.get, "orphan")
    r.check("and it is not listed as available",
            "orphan" not in lib.available(), str(lib.available()))

    with open(os.path.join(tmpdir, "liar.wgsl"), "w") as fh:
        fh.write("@vertex fn vs_main() -> @builtin(position) vec4<f32> "
                 "{ return vec4<f32>(0.0); }\n"
                 "@fragment fn fs_main() -> @location(0) vec4<f32> "
                 "{ return vec4<f32>(1.0); }")
    decl = copy.deepcopy(GOOD)
    decl["vertex_entry"] = "not_there"
    with open(os.path.join(tmpdir, "liar.json"), "w") as fh:
        json.dump(decl, fh)
    lib2 = shaders.ShaderLibrary(ctx, tmpdir)
    r.raises("a JSON naming a missing entry point raises", shaders.ShaderError,
             lib2.get, "liar")

    with open(os.path.join(tmpdir, "broken.wgsl"), "w") as fh:
        fh.write("// nothing")
    with open(os.path.join(tmpdir, "broken.json"), "w") as fh:
        fh.write("{not json")
    r.raises("unparseable JSON raises", shaders.ShaderError,
             shaders.ShaderLibrary(ctx, tmpdir).get, "broken")

    with open(os.path.join(tmpdir, "nocompile.wgsl"), "w") as fh:
        fh.write("@vertex fn vs_main() -> @builtin(position) vec4<f32> "
                 "{ return this_is_not_wgsl; }\n"
                 "@fragment fn fs_main() -> @location(0) vec4<f32> "
                 "{ return vec4<f32>(1.0); }")
    with open(os.path.join(tmpdir, "nocompile.json"), "w") as fh:
        json.dump(GOOD, fh)
    r.raises("WGSL that does not compile raises", shaders.ShaderError,
             shaders.ShaderLibrary(ctx, tmpdir).get, "nocompile")


def main():
    r = Report("test_shaders", minimum=45)
    ctx = device.acquire()
    r.section("Variant names (shared with the add-on's shaders._variant)")
    check_variant_names(r)
    r.section("Schema: what it accepts")
    check_schema_accepts(r)
    r.section("Schema: what it refuses")
    check_schema_rejects(r)
    r.section("Schema 1 (the shader port's shape), normalized")
    check_schema1_normalization(r)
    r.section("The real wgsl/ directory")
    check_library(r, ctx)
    r.section("Broken pairs raise rather than degrade")
    scratch = os.path.join(HERE, "_scratch_shaders")
    try:
        check_broken_pair_raises(r, ctx, scratch)
    finally:
        # Leftover broken shaders are one `cp` from being loaded.
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
    r.check("the scratch shader directory was cleaned up",
            not os.path.exists(scratch), scratch)
    finish(r)


if __name__ == "__main__":
    main()
