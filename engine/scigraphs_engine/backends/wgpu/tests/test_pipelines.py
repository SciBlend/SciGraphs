# The pipeline cache: does it hit, and for the right reasons. Too few hits
# means something is in the key that should not be and the twelve-pipeline
# problem is back; too many means two blend modes share a pipeline and the
# second draws with the first's state. So every field is checked both ways.

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from harness import Report, finish   # noqa: E402

from scigraphs_engine import mesh                          # noqa: E402
from scigraphs_engine.mesh import DrawState, ShaderRef     # noqa: E402
from scigraphs_engine.backends.wgpu import (               # noqa: E402
    PipelineCache, PipelineError, ShaderLibrary, Target, device, pipelines,
)


def check_state_key(r):
    """The Nones must be resolved before hashing, or the cache is wrong."""
    r.check("None DrawState resolves to the documented defaults",
            pipelines.state_key(None) == ("NONE", True, True),
            str(pipelines.state_key(None)))
    r.check("an all-None DrawState resolves the same way",
            pipelines.state_key(DrawState()) == pipelines.state_key(None))
    r.check("two states differing only in an unset field share a key",
            pipelines.state_key(DrawState(blend="ALPHA"))
            == pipelines.state_key(DrawState(blend="ALPHA", depth_test=None)))
    r.check("point_size is NOT in the key",
            pipelines.state_key(DrawState(point_size=2.0))
            == pipelines.state_key(DrawState(point_size=40.0)),
            "twelve size buckets must not be twelve pipelines")
    r.check("line_width is NOT in the key",
            pipelines.state_key(DrawState(line_width=1.0))
            == pipelines.state_key(DrawState(line_width=9.0)))
    for field, a, b in (("blend", DrawState(blend="ALPHA"),
                         DrawState(blend="ADDITIVE")),
                        ("depth_test", DrawState(depth_test=True),
                         DrawState(depth_test=False)),
                        ("depth_write", DrawState(depth_write=True),
                         DrawState(depth_write=False))):
        r.check(f"{field} IS in the key",
                pipelines.state_key(a) != pipelines.state_key(b))
    r.raises("an unknown blend mode is a PipelineError, not a silent default",
             PipelineError, pipelines.state_key, DrawState(blend="SCREEN"))
    r.check("every blend name maps to a WebGPU blend state or None",
            set(pipelines.BLEND) == {"NONE", "ALPHA", "ALPHA_PREMULT",
                                     "ADDITIVE"},
            str(sorted(pipelines.BLEND)))


def check_cache(r, ctx, lib):
    shader = lib.get(ShaderRef("round_point"))
    cache = PipelineCache(ctx)
    r.check("a fresh cache is empty", len(cache) == 0
            and cache.hits == 0 and cache.misses == 0)

    p1 = cache.get(shader, "triangle-strip", DrawState())
    r.check("the first get is a miss", cache.misses == 1 and cache.hits == 0)
    r.check("and returned a pipeline", p1 is not None)

    p2 = cache.get(shader, "triangle-strip", DrawState())
    r.check("an identical request is a hit", cache.hits == 1
            and cache.misses == 1)
    r.check("and returns the same object", p2 is p1)

    p3 = cache.get(shader, "triangle-strip",
                   DrawState(point_size=99.0, line_width=42.0))
    r.check("a different point size still hits", p3 is p1 and cache.hits == 2,
            f"{cache.hits} hits, {cache.misses} misses")

    p4 = cache.get(shader, "triangle-strip", DrawState(blend="ALPHA"))
    r.check("a different blend misses", p4 is not p1 and cache.misses == 2)

    p5 = cache.get(shader, "triangle-strip", DrawState(depth_write=False))
    r.check("a different depth_write misses", p5 is not p1
            and p5 is not p4 and cache.misses == 3)

    p6 = cache.get(shader, "point-list", DrawState())
    r.check("a different topology misses", p6 is not p1 and cache.misses == 4)

    r.check("the cache holds exactly the four distinct pipelines",
            len(cache) == 4, len(cache))

    layouts_a = cache.bind_group_layouts(shader)
    layouts_b = cache.bind_group_layouts(shader)
    r.check("bind group layouts are cached too", layouts_a is layouts_b)
    r.note(f"{cache.hits} hits / {cache.misses} misses over 7 requests")


def check_cache_over_a_real_render(r, ctx, lib):
    from scigraphs_engine.backends.wgpu import Renderer

    n = 600
    rng = np.random.default_rng(7)
    coords = rng.normal(size=(n, 3)).astype(np.float32)
    colors = np.ones((n, 4), np.float32)
    norm = rng.random(n).astype(np.float32)
    specs = mesh.point_specs(coords, colors, norm, size_by_attr=True,
                            buckets=12)
    r.check("point_specs produced twelve size classes", len(specs) == 12,
            f"{len(specs)} classes")

    target = Target(ctx, 128, 96)
    renderer = Renderer(ctx, target, library=lib)
    sized = [mesh.MeshSpec(topology=s.topology, shader=s.shader,
                           attrs=s.attrs,
                           state=DrawState(point_size=2.0 + 20.0 * w))
             for s, w in specs]
    meshes = [renderer.upload(s) for s in sized]
    meshes = [m for m in meshes if m is not None]
    r.check("all twelve uploaded", len(meshes) == 12, len(meshes))

    stats = renderer.render(meshes, _camera(coords, target))
    r.check("twelve classes are twelve draws", stats["draws"] == 12,
            str(stats))
    r.check("...and ONE pipeline", stats["pipelines_cached"] == 1,
            f"{stats['pipelines_cached']} pipelines created for 12 draws")
    r.note("on a backend that keyed pipelines by point size this would be 12")

    # A second frame must create nothing new.
    before = renderer.pipelines.misses
    renderer.render(meshes, _camera(coords, target))
    r.check("a second frame creates no pipelines",
            renderer.pipelines.misses == before)
    for m in meshes:
        m.destroy()
    target.destroy()


def _camera(coords, target):
    from scigraphs_engine.backends.wgpu import camera
    return camera.fit_view(coords, target.size)


def main():
    r = Report("test_pipelines", minimum=25)
    ctx = device.acquire()
    lib = ShaderLibrary(ctx)
    r.section("The cache key")
    check_state_key(r)
    r.section("Hits and misses")
    check_cache(r, ctx, lib)
    r.section("Over a real twelve-bucket render")
    check_cache_over_a_real_render(r, ctx, lib)
    finish(r)


if __name__ == "__main__":
    main()
