# Frame times, measured rather than asserted. A threshold invented here would
# fail on the next machine, so the numbers print with the adapter string.

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from harness import Report, finish   # noqa: E402

from scigraphs_engine import mesh                          # noqa: E402
from scigraphs_engine.mesh import DrawState, ShaderRef     # noqa: E402
from scigraphs_engine.backends.wgpu import (               # noqa: E402
    Renderer, ShaderLibrary, Target, camera, device,
)

SIZES = (1_000, 100_000, 1_000_000)
W, H = 1920, 1080
REPEATS = 20


def cloud(n, seed=0x5C16):
    rng = np.random.default_rng(seed)
    coords = rng.normal(0.0, 1.0, size=(n, 3)).astype(np.float32)
    colors = np.ones((n, 4), np.float32)
    colors[:, 0] = rng.random(n)
    colors[:, 2] = rng.random(n)
    return coords, colors


def check_sync_actually_waits(r, ctx, lib):
    """Every number below depends on sync() being a real wait."""
    coords, colors = cloud(1_000_000)
    target = Target(ctx, W, H)
    renderer = Renderer(ctx, target, library=lib)
    cam = camera.fit_view(coords, target.size)
    spec = mesh.MeshSpec(topology=mesh.POINTS, shader=ShaderRef("round_point"),
                         attrs={"pos": coords, "color": colors},
                         state=DrawState(point_size=6.0))
    up = renderer.upload(spec)
    renderer.render(up, cam)
    kind = renderer.sync()
    r.note(f"sync() path in use: {kind!r}")

    t0 = time.perf_counter()
    renderer.render(up, cam)
    t_submit = time.perf_counter()
    renderer.sync()
    t_sync = time.perf_counter()
    submit_ms = (t_submit - t0) * 1e3
    wait_ms = (t_sync - t_submit) * 1e3
    r.note(f"1M points: submit returned in {submit_ms:.2f} ms, "
           f"sync waited a further {wait_ms:.2f} ms")

    # A no-op returns in constant time, a real wait scales with the submit.
    # Timing readbacks instead drowns it: ~23 ms readback, ~1.3 ms 1M render.
    def sync_after(n_frames):
        renderer.sync()
        t0 = time.perf_counter()
        for _ in range(n_frames):
            renderer.render(up, cam)
        t_submit = time.perf_counter()
        renderer.sync()
        return ((t_submit - t0) * 1e3, (time.perf_counter() - t_submit) * 1e3)

    sync_after(4)   # warm
    samples = {n: [sync_after(n) for _ in range(5)] for n in (1, 4, 16)}
    waits = {n: float(np.median([s[1] for s in v])) for n, v in samples.items()}
    submits = {n: float(np.median([s[0] for s in v])) for n, v in samples.items()}
    for n in (1, 4, 16):
        r.note(f"{n:>2} queued 1M-point frames: submit {submits[n]:.2f} ms, "
               f"then sync waited {waits[n]:.2f} ms")
    r.check("sync() waits: its cost grows with the queued work",
            waits[16] > waits[1] * 4,
            f"16 frames waited {waits[16]:.2f} ms vs {waits[1]:.2f} ms for 1")
    r.check("and the wait, not the submit, is where the GPU time is",
            waits[16] > submits[16],
            f"wait {waits[16]:.2f} ms vs submit {submits[16]:.2f} ms")
    up.destroy()
    target.destroy()


def bench_points(r, ctx, lib):
    target = Target(ctx, W, H)
    renderer = Renderer(ctx, target, library=lib)
    rows = []
    for n in SIZES:
        coords, colors = cloud(n)
        cam = camera.fit_view(coords, target.size)
        spec = mesh.MeshSpec(
            topology=mesh.POINTS, shader=ShaderRef("round_point"),
            attrs={"pos": coords, "color": colors},
            state=DrawState(point_size=6.0))

        t0 = time.perf_counter()
        up = renderer.upload(spec)
        renderer.sync()
        upload_ms = (time.perf_counter() - t0) * 1e3

        renderer.render(up, cam)         # warm the pipeline cache
        renderer.sync()

        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            renderer.render(up, cam)
            renderer.sync()
            times.append((time.perf_counter() - t0) * 1e3)
        times = np.array(times)

        # First read allocates the staging buffer (88 ms vs 20); spend it now.
        target.read()
        reads = []
        for _ in range(3):
            t0 = time.perf_counter()
            renderer.render(up, cam)
            img = target.read()
            reads.append((time.perf_counter() - t0) * 1e3)
        read_ms = float(np.median(reads))

        bg = img[0, 0, :3]
        ink = float((np.abs(img[..., :3] - bg).max(axis=-1) > 8 / 255).mean())
        rows.append((n, upload_ms, float(times.mean()), float(times.min()),
                     float(np.median(times)), read_ms,
                     up.bytes_uploaded / 1e6, ink))
        r.check(f"{n:,} points actually drew something", ink > 0.001,
                f"ink {ink:.4f}")
        up.destroy()

    print()
    print(f"  {W}x{H}, round_point, point_size 6, {REPEATS} frames each")
    print(f"  {'points':>10} {'upload':>9} {'frame':>9} {'best':>9} "
          f"{'median':>9} {'+readback':>10} {'VRAM':>8} {'ink':>7}")
    for n, up_ms, mean, best, med, read_ms, mb, ink in rows:
        print(f"  {n:>10,} {up_ms:>8.2f}m {mean:>8.2f}m {best:>8.2f}m "
              f"{med:>8.2f}m {read_ms:>9.2f}m {mb:>7.1f}M {ink:>6.3f}")
    print("  (m = milliseconds; frame = encode + submit + sync, no readback)")

    for n, up_ms, mean, best, med, read_ms, mb, ink in rows:
        r.note(f"{n:,} pts: upload {up_ms:.2f} ms, frame {mean:.2f} ms "
               f"(best {best:.2f}), +readback {read_ms:.2f} ms, {mb:.1f} MB")
    target.destroy()
    return rows


def bench_draw_count(r, ctx, lib):
    n = 120_000
    coords, colors = cloud(n)
    target = Target(ctx, W, H)
    renderer = Renderer(ctx, target, library=lib)
    cam = camera.fit_view(coords, target.size)

    one = renderer.upload(mesh.MeshSpec(
        topology=mesh.POINTS, shader=ShaderRef("round_point"),
        attrs={"pos": coords, "color": colors},
        state=DrawState(point_size=6.0)))
    from scigraphs_engine.mesh import MeshGroup
    step = n // 12
    twelve = renderer.upload(mesh.MeshSpec(
        topology=mesh.POINTS, shader=ShaderRef("round_point"),
        attrs={"pos": coords, "color": colors},
        state=DrawState(point_size=6.0),
        groups=tuple(MeshGroup(start=i * step,
                               count=step if i < 11 else n - 11 * step,
                               weight=(i + 0.5) / 12) for i in range(12))))
    r.check("the grouped upload is twelve draws over the same buffers",
            len(twelve.draws) == 12
            and twelve.vertex_buffers[0].size == one.vertex_buffers[0].size,
            f"{len(twelve.draws)} draws")

    out = {}
    for name, up in (("1 draw", one), ("12 draws", twelve)):
        renderer.render(up, cam)
        renderer.sync()
        times = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            renderer.render(up, cam)
            renderer.sync()
            times.append((time.perf_counter() - t0) * 1e3)
        out[name] = float(np.median(times))
    r.note(f"{n:,} points as 1 draw: {out['1 draw']:.2f} ms; "
           f"as 12 range groups: {out['12 draws']:.2f} ms "
           f"(+{out['12 draws'] - out['1 draw']:.2f} ms for 11 extra draws)")
    one.destroy()
    twelve.destroy()
    target.destroy()


def bench_pipeline_creation(r, ctx, lib):
    from scigraphs_engine.backends.wgpu import PipelineCache
    shader = lib.get(ShaderRef("round_point"))
    times = []
    for i in range(8):
        cache = PipelineCache(ctx)
        t0 = time.perf_counter()
        cache.get(shader, "triangle-strip", DrawState())
        times.append((time.perf_counter() - t0) * 1e3)
    cache = PipelineCache(ctx)
    cache.get(shader, "triangle-strip", DrawState())
    t0 = time.perf_counter()
    for _ in range(1000):
        cache.get(shader, "triangle-strip", DrawState())
    hit_us = (time.perf_counter() - t0) * 1e3
    r.note(f"pipeline creation: {np.median(times):.2f} ms median of 8; "
           f"a cache hit: {hit_us:.2f} us")
    r.check("a cache hit is at least 100x cheaper than a creation",
            hit_us < np.median(times) * 1000 / 100,
            f"{hit_us:.2f} us vs {np.median(times) * 1000:.0f} us")


def main():
    r = Report("bench", minimum=6)
    ctx = device.acquire()
    lib = ShaderLibrary(ctx)
    print(f"  adapter: {ctx.describe()}")
    r.section("Is sync() real?")
    check_sync_actually_waits(r, ctx, lib)
    r.section("Frame times")
    bench_points(r, ctx, lib)
    r.section("Draw-call cost")
    bench_draw_count(r, ctx, lib)
    r.section("Pipeline creation vs a cache hit")
    bench_pipeline_creation(r, ctx, lib)
    finish(r)


if __name__ == "__main__":
    main()
