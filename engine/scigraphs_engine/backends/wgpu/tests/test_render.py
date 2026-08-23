# A real render, asserted on the pixels. "Not all background" would pass on a
# single stray pixel, so every assertion names where the ink is. The last check
# compares the readback against tests/golden/metrics.py::from_blender, and fails
# rather than skips if that module will not import, or the layouts drift apart.

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from harness import Report, finish   # noqa: E402

from scigraphs_engine import mesh                          # noqa: E402
from scigraphs_engine.mesh import DrawState, ShaderRef     # noqa: E402
from scigraphs_engine.backends.wgpu import (               # noqa: E402
    Renderer, ShaderLibrary, Target, camera, device,
)

# engine/scigraphs_engine/backends/wgpu/tests -> the repository root.
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

BG = (0.10, 0.20, 0.30, 1.0)


# This device turns 0.10 (= 25.5/255) into 25/255 where Python's round() gives
# 26, so the clear checks split: all pixels identical, and within one step.
LSB = 1.0 / 255.0


def _u8(v):
    """What a [0, 1] float becomes after a round trip, to within a step."""
    return round(v * 255.0) / 255.0


def check_clear(r, ctx):
    target = Target(ctx, 64, 48)
    target.clear(BG)
    img = target.read()
    r.check("readback is (H, W, 4) float32",
            img.shape == (48, 64, 4) and img.dtype == np.float32,
            f"{img.shape} {img.dtype}")
    r.check("values are in [0, 1]", 0.0 <= img.min() and img.max() <= 1.0)
    want = np.array(BG, np.float32)
    r.check("every pixel is IDENTICAL to the top-left one",
            bool(np.all(img == img[0, 0])),
            f"{int((img != img[0, 0]).any(axis=-1).sum())} pixels differ")
    r.check("and within one 8-bit step of the color asked for",
            bool(np.abs(img - want).max() <= LSB),
            f"asked {want}, got {img[0, 0]}, "
            f"max deviation {float(np.abs(img - want).max()) * 255:.2f}/255")
    for name, px in (("top-left", img[0, 0]), ("top-right", img[0, -1]),
                     ("bottom-left", img[-1, 0]), ("bottom-right", img[-1, -1])):
        r.check(f"{name} corner", bool(np.abs(px - want).max() <= LSB),
                str(px))
    target.clear((1.0, 1.0, 1.0, 1.0))
    r.check("a second clear replaces the first",
            float(target.read().min()) == 1.0)
    target.destroy()


def check_points_land_where_predicted(r, ctx, lib):
    coords = np.array([[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0],
                       [-1.0, 0.0, 1.0], [1.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0]], np.float32)
    colors = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1],
                       [1, 1, 0, 1], [1, 0, 1, 1]], np.float32)
    w, h, size = 320, 240, 11.0

    target = Target(ctx, w, h)
    renderer = Renderer(ctx, target, library=lib)
    spec = mesh.MeshSpec(topology=mesh.POINTS, shader=ShaderRef("round_point"),
                         attrs={"pos": coords, "color": colors},
                         state=DrawState(point_size=size))
    up = renderer.upload(spec)
    r.check("the spec uploaded", up is not None)
    cam = camera.fit_view(coords, target.size, margin=1.4)
    stats = renderer.render(up, cam, clear_color=BG)
    r.check("one draw of five instances",
            stats["draws"] == 1 and up.draws[0].instances == 5, str(stats))

    img = target.read()
    px = cam.project(coords, w, h)
    bg = np.array([_u8(c) for c in BG[:3]], np.float32)

    hits = 0
    for i, (x, y, z) in enumerate(px):
        xi, yi = int(round(x)), int(round(y))
        inside = 0 <= xi < w and 0 <= yi < h
        r.check(f"node {i} projects inside the frame", inside, f"({xi}, {yi})")
        if not inside:
            continue
        got = img[yi, xi, :3]
        want = colors[i, :3]
        ok = bool(np.abs(got - want).max() < 2.0 / 255.0)
        r.check(f"node {i} paints its own color at its own pixel", ok,
                f"got {got}, want {want}")
        hits += ok

        # Miss the w multiply and the disc grows with depth, so check that
        # just outside it is still background.
        off = int(size) + 4
        ox = min(max(xi + off, 0), w - 1)
        r.check(f"node {i} is background {off}px to its side",
                bool(np.abs(img[yi, ox, :3] - bg).max() < 2.0 / 255.0),
                str(img[yi, ox, :3]))

    r.check("all five nodes were found", hits == 5, f"{hits}/5")

    ink = float((np.abs(img[..., :3] - bg).max(axis=-1) > 8 / 255).mean())
    expected = 5 * np.pi * (size / 2) ** 2 / (w * h)
    r.check("ink fraction is within 25% of five discs of that size",
            abs(ink - expected) < expected * 0.25,
            f"got {ink:.5f}, five discs would be {expected:.5f}")
    up.destroy()
    target.destroy()
    return img


def check_point_size_is_honored(r, ctx, lib):
    n = 40
    coords = np.random.default_rng(11).normal(size=(n, 3)).astype(np.float32)
    colors = np.ones((n, 4), np.float32)
    target = Target(ctx, 256, 256)
    renderer = Renderer(ctx, target, library=lib)
    cam = camera.fit_view(coords, target.size, margin=1.3)
    bg = np.array([_u8(c) for c in BG[:3]], np.float32)

    fractions = []
    for size in (3.0, 12.0, 30.0):
        spec = mesh.MeshSpec(
            topology=mesh.POINTS, shader=ShaderRef("round_point"),
            attrs={"pos": coords, "color": colors},
            state=DrawState(point_size=size))
        up = renderer.upload(spec)
        renderer.render(up, cam, clear_color=BG)
        img = target.read()
        fractions.append(float(
            (np.abs(img[..., :3] - bg).max(axis=-1) > 8 / 255).mean()))
        up.destroy()
    r.check("ink grows strictly with point size",
            fractions[0] < fractions[1] < fractions[2],
            " -> ".join(f"{f:.4f}" for f in fractions))
    ratio = fractions[1] / max(fractions[0], 1e-9)
    r.check("4x the diameter is roughly 16x the ink (area, not length)",
            8.0 < ratio < 24.0, f"{ratio:.1f}x")
    target.destroy()


def check_per_draw_uniform_is_per_draw(r, ctx, lib):
    left = np.array([[-1.0, 0.0, 0.0]], np.float32)
    right = np.array([[1.0, 0.0, 0.0]], np.float32)
    white = np.ones((1, 4), np.float32)
    target = Target(ctx, 256, 128)
    renderer = Renderer(ctx, target, library=lib)
    cam = camera.fit_view(np.concatenate([left, right]), target.size,
                          margin=1.6)

    def spec(pos, size):
        return mesh.MeshSpec(topology=mesh.POINTS,
                             shader=ShaderRef("round_point"),
                             attrs={"pos": pos, "color": white},
                             state=DrawState(point_size=size))

    a = renderer.upload(spec(left, 4.0))
    b = renderer.upload(spec(right, 24.0))
    renderer.render([a, b], cam, clear_color=BG)
    img = target.read()
    bg = np.array([_u8(c) for c in BG[:3]], np.float32)
    ink = np.abs(img[..., :3] - bg).max(axis=-1) > 8 / 255
    half = img.shape[1] // 2
    small = int(ink[:, :half].sum())
    big = int(ink[:, half:].sum())
    lo, hi = min(small, big), max(small, big)
    r.check("the two halves of ONE render have different point sizes",
            lo > 0 and hi > lo * 8,
            f"{lo} px and {hi} px of ink")
    r.note("before the dynamic-offset ring both were the size of the last draw")
    a.destroy()
    b.destroy()
    target.destroy()


def check_depth(r, ctx, lib):
    """The far point is drawn last, so painter's order alone shows green."""
    near = np.array([[0.0, 0.0, 0.0]], np.float32)
    far = np.array([[0.0, 2.0, 0.0]], np.float32)
    red = np.array([[1.0, 0.0, 0.0, 1.0]], np.float32)
    green = np.array([[0.0, 1.0, 0.0, 1.0]], np.float32)
    target = Target(ctx, 128, 128)
    renderer = Renderer(ctx, target, library=lib)
    cam = camera.Camera(camera.look_at((0.0, -6.0, 0.0), (0.0, 0.0, 0.0)),
                        camera.perspective(45.0, 1.0, 0.1, 50.0), 0.1, 50.0)

    def spec(pos, col, state):
        return mesh.MeshSpec(topology=mesh.POINTS,
                             shader=ShaderRef("round_point"),
                             attrs={"pos": pos, "color": col}, state=state)

    on = DrawState(point_size=20.0, depth_test=True, depth_write=True)
    a = renderer.upload(spec(near, red, on))
    b = renderer.upload(spec(far, green, on))
    renderer.render([a, b], cam, clear_color=BG)   # near first, then far
    center = target.read()[64, 64, :3]
    r.check("with depth testing, the near point wins the pixel",
            center[0] > 0.9 and center[1] < 0.1, str(center))

    off = DrawState(point_size=20.0, depth_test=False, depth_write=False)
    a2 = renderer.upload(spec(near, red, off))
    b2 = renderer.upload(spec(far, green, off))
    renderer.render([a2, b2], cam, clear_color=BG)
    center = target.read()[64, 64, :3]
    r.check("with depth testing off, the last drawn wins instead",
            center[1] > 0.9 and center[0] < 0.1, str(center))
    r.note("both directions checked: a depth test that never rejects and one "
           "that always rejects would each pass only one of these")
    for m in (a, b, a2, b2):
        m.destroy()
    target.destroy()


def check_readback_matches_golden_layout(r, ctx, lib):
    sys.path.insert(0, os.path.join(REPO, "tests"))
    try:
        from golden import metrics
    except Exception as exc:    # noqa: BLE001
        r.check("tests/golden/metrics.py is importable", False,
                f"{type(exc).__name__}: {exc}")
        return
    r.check("tests/golden/metrics.py is importable", True,
            os.path.abspath(metrics.__file__))

    # Off-center ink, so a flipped row order cannot hide in symmetry.
    coords = np.array([[0.0, 0.0, 0.9], [0.2, 0.0, 0.85], [-0.2, 0.0, 0.85]],
                      np.float32)
    colors = np.ones((3, 4), np.float32)
    w, h = 200, 200
    target = Target(ctx, w, h)
    renderer = Renderer(ctx, target, library=lib)
    cam = camera.Camera(camera.look_at((0.0, -4.0, 0.0), (0.0, 0.0, 0.0)),
                        camera.perspective(45.0, 1.0, 0.1, 20.0), 0.1, 20.0)
    spec = mesh.MeshSpec(topology=mesh.POINTS, shader=ShaderRef("round_point"),
                         attrs={"pos": coords, "color": colors},
                         state=DrawState(point_size=14.0))
    up = renderer.upload(spec)
    renderer.render(up, cam, clear_color=BG)
    img = target.read()

    flat = np.asarray(img, np.float32)
    r.check("shape matches what from_blender returns for these dimensions",
            flat.shape == metrics.from_blender(
                np.zeros(w * h * 4, np.float32), w, h).shape,
            str(flat.shape))
    r.check("dtype matches", flat.dtype == metrics.from_blender(
        np.zeros(w * h * 4, np.float32), w, h).dtype)

    bg = metrics.background(img)
    r.check("metrics.background finds the clear color",
            bool(np.abs(bg - np.array(BG[:3], np.float32)).max() <= LSB),
            str(bg))

    fp = metrics.fingerprint(img)
    r.check("metrics.fingerprint runs and reports the right size",
            fp["size"] == [w, h], str(fp["size"]))
    r.check("it found ink", fp["ink_frac"] > 0.0, f"{fp['ink_frac']:.5f}")

    # Nodes at world +z land in the upper half; a flipped readback fails here.
    r.check("ink centroid is in the upper half, so row 0 is the top",
            fp["ink_centroid"][1] < 0.4,
            f"y={fp['ink_centroid'][1]:.4f} (a flipped readback gives "
            f"{1 - fp['ink_centroid'][1]:.4f})")
    r.check("and horizontally centered, as the geometry is",
            abs(fp["ink_centroid"][0] - 0.5) < 0.05,
            f"x={fp['ink_centroid'][0]:.4f}")

    delta = metrics.compare_fingerprints(fp, metrics.fingerprint(img))
    ok, bad = metrics.verdict(delta, metrics.TOLERANCE_FLOOR)
    r.check("an image compared against itself passes the golden verdict",
            ok and all(v == 0.0 for v in delta.values()), str(bad))

    png = os.path.join(HERE, "_out_render.png")
    target.save_png(png)
    r.check("save_png wrote a file", os.path.getsize(png) > 100,
            f"{os.path.getsize(png)} bytes")
    os.remove(png)
    up.destroy()
    target.destroy()


def check_degradation(r, ctx, lib):
    coords = np.zeros((3, 3), np.float32)
    spec = mesh.MeshSpec(topology=mesh.POINTS,
                         shader=ShaderRef("not_ported_yet"),
                         attrs={"pos": coords})
    target = Target(ctx, 32, 32)
    renderer = Renderer(ctx, target, library=lib)
    r.check("uploading an unported shader gives None, not an exception",
            renderer.upload(spec) is None)
    stats = renderer.render([], camera.fit_view(coords, target.size),
                            clear_color=BG)
    r.check("a render with nothing drawable still clears",
            stats["draws"] == 0
            and bool(np.abs(target.read()[16, 16]
                             - np.array(BG, np.float32)).max() <= LSB))
    target.destroy()


def main():
    r = Report("test_render", minimum=35)
    ctx = device.acquire()
    lib = ShaderLibrary(ctx)
    r.section("A solid-color clear")
    check_clear(r, ctx)
    r.section("Points land where the camera says")
    check_points_land_where_predicted(r, ctx, lib)
    r.section("Point size, which WebGPU does not have")
    check_point_size_is_honored(r, ctx, lib)
    r.section("The per-draw uniform is per draw")
    check_per_draw_uniform_is_per_draw(r, ctx, lib)
    r.section("Depth")
    check_depth(r, ctx, lib)
    r.section("Readback layout vs tests/golden/metrics.py")
    check_readback_matches_golden_layout(r, ctx, lib)
    r.section("Degradation when a shader is not ported")
    check_degradation(r, ctx, lib)
    finish(r)


if __name__ == "__main__":
    main()
