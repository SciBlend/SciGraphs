# MeshSpec -> buffers and draws: the three plain forms, and range groups. Range
# groups are why this file exists: per-class vertex buffers turned a parameter
# change on 2.8M bundled segments from 123 ms into 1.44 s. The Blender side
# materializes an index buffer where this backend does not.

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from harness import Report, finish   # noqa: E402

from scigraphs_engine import mesh                            # noqa: E402
from scigraphs_engine.mesh import (                          # noqa: E402
    DrawState, LINES, MeshGroup, MeshSpec, ShaderRef, TRIS,
)
from scigraphs_engine.backends.wgpu import (                 # noqa: E402
    ShaderLibrary, buffers, device,
)


def line_shader_stub():
    """Shader-shaped stub for the line specs, whose WGSL is not ported yet."""
    class Attr:
        def __init__(self, name, loc, fmt, cols):
            self.name, self.location, self.format, self.columns = \
                name, loc, fmt, cols

    class Stub:
        name = "line_stub"
        expand = None
        cull = "none"
        writes_depth = True
        depth_compare = "less-equal"
        default_state = None
        attributes = (Attr("pos", 0, "float32x3", 3),)
        bindings = ()

    return Stub()


def check_plain_indexed(r, ctx, lib):
    n = 7
    coords = np.random.default_rng(1).normal(size=(n, 3)).astype(np.float32)
    colors = np.ones((n, 4), np.float32)
    radii = np.full(n, 0.1, np.float32)
    spec = mesh.sphere_spec(coords, colors, radii)
    r.check("sphere_spec is indexed", spec.indices is not None
            and spec.groups is None)
    r.check("sphere_spec has 4 vertices per node", spec.vertex_count == n * 4)

    shader = lib.get(spec.shader)
    if shader is None:
        r.check("sphere shader available for the indexed form", False,
                "wgsl/sphere.wgsl not ported; the indexed path went untested")
        return
    up = buffers.upload(ctx, spec, shader)
    r.check("one vertex buffer per declared attribute",
            len(up.vertex_buffers) == len(shader.attributes),
            f"{len(up.vertex_buffers)} buffers for "
            f"{[a.name for a in shader.attributes]}")
    r.check("an index buffer was built", up.index_buffer is not None)
    r.check("index count is 6 per node", up.index_count == n * 6, up.index_count)
    r.check("exactly one draw", len(up.draws) == 1)
    d = up.draws[0]
    r.check("that draw is indexed", d.indexed is True)
    r.check("with the whole index buffer", d.count == n * 6, d.count)
    r.check("from offset 0", d.first == 0)
    r.check("one instance", d.instances == 1)
    r.check("topology is triangle-list", up.topology == "triangle-list",
            up.topology)
    up.destroy()


def check_plain_unindexed(r, ctx, lib):
    n = 11
    coords = np.random.default_rng(2).normal(size=(n, 3)).astype(np.float32)
    colors = np.ones((n, 4), np.float32)
    spec, weight = mesh.point_specs(coords, colors)[0]
    r.check("point_specs gives one unbucketed spec",
            spec.indices is None and spec.groups is None and weight is None)

    shader = lib.get(spec.shader)
    r.check("round_point shader available", shader is not None)
    up = buffers.upload(ctx, spec, shader)
    r.check("no index buffer", up.index_buffer is None)
    r.check("one draw", len(up.draws) == 1)
    d = up.draws[0]
    r.check("not indexed", d.indexed is False)
    r.check("expand turns n points into n instances", d.instances == n,
            d.instances)
    r.check("of 4 vertices each", d.count == 4, d.count)
    r.check("topology comes from the shader's expand, not the spec",
            up.topology == "triangle-strip", up.topology)
    r.note("a spec of POINTS drawn as a triangle-strip is the point-size "
           "workaround (README 3.2), not a mistake")
    up.destroy()


def check_index_groups(r, ctx):
    coords = np.random.default_rng(3).normal(size=(20, 3)).astype(np.float32)
    edges = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
                     dtype=np.int32)
    weights = np.array([0.05, 0.1, 0.5, 0.55, 0.95, 0.99], dtype=np.float32)
    spec = mesh.bucketed_line_spec(coords, edges, weights, buckets=4)
    r.check("bucketed_line_spec produced groups", spec is not None
            and spec.groups is not None)
    r.check("groups carry explicit indices",
            all(not g.is_range for g in spec.groups))
    r.check("and one shared position attribute",
            list(spec.attrs) == ["pos"]
            and spec.attrs["pos"].shape[0] == 20)

    shader = line_shader_stub()
    up = buffers.upload(ctx, spec, shader)
    r.check("one vertex buffer, shared by every group",
            len(up.vertex_buffers) == 1)
    r.check("one index buffer holding every group",
            up.index_buffer is not None)
    r.check("as many draws as groups", len(up.draws) == len(spec.groups),
            f"{len(up.draws)} draws, {len(spec.groups)} groups")

    # The offsets have to tile the concatenated index buffer exactly.
    offset = 0
    ok = True
    for g, d in zip(spec.groups, up.draws):
        ok &= d.indexed and d.first == offset and d.count == g.indices.size
        offset += g.indices.size
    r.check("each group draws its own slice at the right first_index", ok,
            str([(d.first, d.count) for d in up.draws]))
    r.check("the slices exactly fill the index buffer",
            offset == up.index_count, f"{offset} vs {up.index_count}")
    r.check("group weights survive to the draw",
            [d.weight for d in up.draws] == [g.weight for g in spec.groups])
    up.destroy()


def check_range_groups_are_offsets(r, ctx):
    n = 240
    rng = np.random.default_rng(4)
    seg_a = rng.normal(size=(n, 3)).astype(np.float32)
    seg_b = rng.normal(size=(n, 3)).astype(np.float32)
    widths = rng.random(n).astype(np.float32)
    spec = mesh.bucketed_segment_spec(seg_a, seg_b, widths, buckets=6)
    r.check("bucketed_segment_spec produced range groups", spec is not None
            and spec.groups and all(g.is_range for g in spec.groups))
    r.check("it produced more than one class, so there is something to offset",
            len(spec.groups) > 1, f"{len(spec.groups)} classes")
    r.check("the spec itself carries no indices", spec.indices is None)

    shader = line_shader_stub()
    up = buffers.upload(ctx, spec, shader)

    r.check("NO index buffer was built for a range-grouped spec",
            up.index_buffer is None and up.index_count == 0,
            f"index_buffer={up.index_buffer!r}")
    would_have_been = sum(int(g.count) for g in spec.groups)
    r.note(f"the Blender path would have materialized {would_have_been} int32 "
           f"({would_have_been * 4} bytes) that say nothing but 'a to b'")

    r.check("one draw per class", len(up.draws) == len(spec.groups))
    ok_offsets = all(
        (not d.indexed) and d.first == int(g.start) and d.count == int(g.count)
        for g, d in zip(spec.groups, up.draws))
    r.check("every draw is a non-indexed draw at first_vertex=group.start",
            ok_offsets, str([(d.first, d.count) for d in up.draws]))
    r.check("the ranges are contiguous and cover every vertex",
            up.draws[0].first == 0
            and all(a.first + a.count == b.first
                    for a, b in zip(up.draws, up.draws[1:]))
            and up.draws[-1].first + up.draws[-1].count == spec.vertex_count,
            f"last ends at {up.draws[-1].first + up.draws[-1].count}, "
            f"{spec.vertex_count} vertices")
    r.check("weights survive", [d.weight for d in up.draws]
            == [g.weight for g in spec.groups])

    draws, index_arrays = buffers.plan(spec, shader)
    r.check("plan() agrees with upload() without a device",
            index_arrays == []
            and [(d.first, d.count) for d in draws]
            == [(d.first, d.count) for d in up.draws])
    up.destroy()


def check_range_groups_under_expand(r, ctx, lib):
    n = 30
    coords = np.random.default_rng(5).normal(size=(n, 3)).astype(np.float32)
    colors = np.ones((n, 4), np.float32)
    spec = MeshSpec(
        topology=mesh.POINTS, shader=ShaderRef("round_point"),
        attrs={"pos": coords, "color": colors},
        groups=(MeshGroup(start=0, count=10, weight=0.1),
                MeshGroup(start=10, count=20, weight=0.9)))
    shader = lib.get(spec.shader)
    up = buffers.upload(ctx, spec, shader)
    r.check("two draws", len(up.draws) == 2)
    r.check("first_instance carries the range start",
            [d.first_instance for d in up.draws] == [0, 10],
            str([d.first_instance for d in up.draws]))
    r.check("instance count carries the range count",
            [d.instances for d in up.draws] == [10, 20])
    r.check("each still draws 4 corner vertices",
            [d.count for d in up.draws] == [4, 4])
    r.check("and still no index buffer", up.index_buffer is None)
    up.destroy()

    bad = MeshSpec(topology=mesh.POINTS, shader=ShaderRef("round_point"),
                   attrs={"pos": coords, "color": colors},
                   groups=(MeshGroup(indices=np.arange(5, dtype=np.int32)),))
    r.raises("an expanding shader refuses explicit-index groups",
             buffers.UploadError, buffers.upload, ctx, bad, shader)


def check_state_inheritance(r, ctx, lib):
    coords = np.zeros((3, 3), np.float32)
    colors = np.ones((3, 4), np.float32)
    spec = MeshSpec(
        topology=mesh.POINTS, shader=ShaderRef("round_point"),
        attrs={"pos": coords, "color": colors},
        state=DrawState(blend="ALPHA", point_size=9.0),
        groups=(MeshGroup(start=0, count=3, state=DrawState(point_size=3.0)),))
    shader = lib.get(spec.shader)
    up = buffers.upload(ctx, spec, shader,
                        DrawState(depth_test=False, line_width=7.0))
    s = up.draws[0].state
    r.check("group overrides spec", s.point_size == 3.0, s.point_size)
    r.check("spec fills what the group left unset", s.blend == "ALPHA", s.blend)
    r.check("the renderer default fills what neither set",
            s.depth_test is False and s.line_width == 7.0,
            f"depth_test={s.depth_test} line_width={s.line_width}")
    up.destroy()


def check_rejections(r, ctx, lib):
    coords = np.zeros((5, 3), np.float32)
    colors = np.ones((5, 4), np.float32)
    shader = lib.get(ShaderRef("round_point"))

    missing = MeshSpec(topology=mesh.POINTS, shader=ShaderRef("round_point"),
                       attrs={"pos": coords})
    r.raises("a spec missing a declared attribute is refused",
             buffers.UploadError, buffers.upload, ctx, missing, shader)

    wrong = MeshSpec(topology=mesh.POINTS, shader=ShaderRef("round_point"),
                     attrs={"pos": coords, "color": np.ones((5, 3), np.float32)})
    r.raises("a float32x4 declaration over an (N,3) array is refused",
             buffers.UploadError, buffers.upload, ctx, wrong, shader)

    ragged = MeshSpec(topology=mesh.POINTS, shader=ShaderRef("round_point"),
                      attrs={"pos": coords, "color": np.ones((3, 4), np.float32)})
    r.raises("MeshSpec.validate's ragged-attribute failure is surfaced",
             buffers.UploadError, buffers.upload, ctx, ragged, shader)

    oob = MeshSpec(topology=TRIS, shader=ShaderRef("sphere"),
                   attrs={"pos": coords}, indices=np.array([[0, 1, 99]], np.int32))
    r.raises("an out-of-range index is refused",
             buffers.UploadError, buffers.plan, oob, line_shader_stub())

    empty = MeshSpec(topology=mesh.POINTS, shader=ShaderRef("round_point"),
                     attrs={"pos": np.zeros((0, 3), np.float32),
                            "color": np.zeros((0, 4), np.float32)})
    r.check("an empty spec uploads to None, not an exception",
            buffers.upload(ctx, empty, shader) is None)
    r.check("a None spec uploads to None",
            buffers.upload(ctx, None, shader) is None)
    r.check("a None shader uploads to None",
            buffers.upload(ctx, missing, None) is None)


def check_scalar_attribute(r, ctx, lib):
    """mesh.py emits 1-D arrays for scalars; they upload as (N, 1)."""
    n = 6
    rng = np.random.default_rng(6)
    spec = mesh.ribbon_spec(rng.normal(size=(n, 3)).astype(np.float32),
                            rng.normal(size=(n, 3)).astype(np.float32),
                            np.array([1.0, 0.5, 0.2, 1.0], np.float32),
                            np.full(n, 0.05, np.float32),
                            np.full(n, 0.03, np.float32))
    r.check("ribbon_spec emits 1-D 'side' and 'radius'",
            spec.attrs["side"].ndim == 1 and spec.attrs["radius"].ndim == 1)
    shader = lib.get(spec.shader)
    if shader is None:
        r.check("ribbon shader available for the scalar-attribute check", False,
                "wgsl/ribbon.wgsl not ported")
        return
    up = buffers.upload(ctx, spec, shader)
    r.check("a 1-D attribute uploads without complaint",
            len(up.vertex_buffers) == len(shader.attributes))
    r.check("index count is 6 per segment", up.index_count == n * 6,
            up.index_count)
    r.note(f"{up.bytes_uploaded} bytes for {n} ribbon segments")
    up.destroy()


def main():
    r = Report("test_upload", minimum=50)
    ctx = device.acquire()
    lib = ShaderLibrary(ctx)
    r.section("Form 1: a plain indexed spec")
    check_plain_indexed(r, ctx, lib)
    r.section("Form 2: no indices, no groups")
    check_plain_unindexed(r, ctx, lib)
    r.section("Form 3: groups with explicit indices")
    check_index_groups(r, ctx)
    r.section("Form 4: range groups become draw offsets (the whole point)")
    check_range_groups_are_offsets(r, ctx)
    r.section("Range groups on an expanding shader")
    check_range_groups_under_expand(r, ctx, lib)
    r.section("DrawState inheritance")
    check_state_inheritance(r, ctx, lib)
    r.section("Scalar (1-D) attributes")
    check_scalar_attribute(r, ctx, lib)
    r.section("What upload refuses")
    check_rejections(r, ctx, lib)
    finish(r)


if __name__ == "__main__":
    main()
