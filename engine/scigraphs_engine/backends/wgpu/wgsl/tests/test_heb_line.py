# heb_line: the cubic B-spline the vertex shader evaluates, checked against a
# numpy reference. The shader runs de Boor, the reference sums Cox-de Boor
# basis functions, so a wrong knot vector, degree drop or span cannot agree
# with itself. Hierarchy is not checked; these control polygons are made up.

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WGSL_DIR = os.path.dirname(HERE)
ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(WGSL_DIR))))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import json                                                       # noqa: E402
import re                                                         # noqa: E402

import wgpu                                                       # noqa: E402

from scigraphs_engine import mesh                                 # noqa: E402
from scigraphs_engine.backends.wgpu import (camera as cam,        # noqa: E402
                                            device, renderer,
                                            shaders, target)

FAILED = []
CHECKS = 0
MIN_CHECKS = 45


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


SIZE = 512
HALF = 4.0
EYE_Z = 10.0
NEAR, FAR = 1.0, 100.0
PX_PER_UNIT = SIZE / (2.0 * HALF)


def ortho_camera():
    view = cam.look_at((0.0, 0.0, EYE_Z), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    return cam.Camera(view, cam.orthographic(HALF, HALF, NEAR, FAR), NEAR, FAR)


def open_uniform_knots(degree, interior):
    return np.concatenate([
        np.zeros(degree + 1),
        (np.arange(1, interior + 1) / (interior + 1.0)) if interior else
        np.zeros(0),
        np.ones(degree + 1),
    ])


def basis(knots, i, p, x):
    """N_{i,p}(x), the textbook recursion, over an array of parameters."""
    if knots[i + p + 1] == knots[i]:
        return np.zeros_like(x)
    if p == 0:
        if knots[i + 1] == knots[i]:
            return np.zeros_like(x)
        inside = (knots[i] <= x) & (x < knots[i + 1])
        # Or C(1) is zero and the curve never reaches its last point.
        if knots[i + 1] == knots[-1]:
            inside = (knots[i] <= x) & (x <= knots[i + 1])
        return inside.astype(np.float64)
    out = np.zeros_like(x)
    d1 = knots[i + p] - knots[i]
    if d1 > 0:
        out = out + (x - knots[i]) / d1 * basis(knots, i, p - 1, x)
    d2 = knots[i + p + 1] - knots[i + 1]
    if d2 > 0:
        out = out + (knots[i + p + 1] - x) / d2 * basis(knots, i + 1, p - 1, x)
    return out


def straightened(points, beta):
    """Equation 1 of the paper, applied to the control points."""
    points = np.asarray(points, np.float64)
    n = points.shape[0]
    if beta >= 1.0:
        return points
    w = (np.arange(n) / (n - 1.0)) if n > 1 else np.zeros(n)
    line = points[0] + w[:, None] * (points[-1] - points[0])
    return beta * points + (1.0 - beta) * line


def reference_curve(points, beta, t):
    """The paper's curve through ``points``, at parameters ``t``."""
    points = np.asarray(points, np.float64)
    n = points.shape[0]
    degree = min(3, n - 1)
    if degree < 1:
        return np.tile(points[0], (len(t), 1))
    interior = n - degree - 1
    knots = open_uniform_knots(degree, interior)
    ctl = straightened(points, beta)
    t = np.asarray(t, np.float64)
    out = np.zeros((t.size, 3))
    for i in range(n):
        out += basis(knots, i, degree, t)[:, None] * ctl[i]
    return out


def _src():
    return open(os.path.join(WGSL_DIR, "heb_line.wgsl"), encoding="utf-8").read()


def _decl():
    return json.load(open(os.path.join(WGSL_DIR, "heb_line.json"),
                          encoding="utf-8"))


def declared_samples():
    m = re.search(r"const SAMPLES\s*:\s*u32\s*=\s*(\d+)u\s*;", _src())
    return int(m.group(1)) if m else None


SAMPLES = declared_samples() or 33


def test_the_sample_count_is_declared_once():
    """SAMPLES in the .wgsl and expand.vertex_count in the .json are one fact."""
    wgsl = declared_samples()
    check("heb_line.wgsl declares a SAMPLES constant", wgsl is not None,
          str(wgsl))
    js = (_decl().get("expand") or {}).get("vertex_count")
    check("and it equals expand.vertex_count in the .json", wgsl == js,
          f"{wgsl} vs {js}")


def test_heb_line_loads_and_declares_an_instanced_strip():
    lib = rig().library
    shader = lib.get("heb_line")
    check("heb_line loads and compiles", shader is not None)
    check("it declares the two per-edge color attributes",
          [a.name for a in shader.attributes] == ["col_a", "col_b"],
          str([a.name for a in shader.attributes]))
    check("it expands each MeshSpec row into one line-strip",
          shader.expand is not None
          and shader.expand.topology == "line-strip",
          str(shader.expand.topology if shader.expand else None))
    names = [b.name for b in shader.bindings]
    check("it binds camera, heb and ctrl and nothing else",
          names == ["camera", "heb", "ctrl"], str(names))
    by = {b.name: b for b in shader.bindings}
    check("ctrl is read-only storage at slot 6",
          by["ctrl"].type == "read-only-storage" and by["ctrl"].binding == 6,
          str(by["ctrl"]))
    check("heb is a 16-byte uniform at slot 5",
          by["heb"].type == "uniform" and by["heb"].binding == 5
          and by["heb"].size == 16, str(by["heb"]))


def test_the_camera_block_is_the_shared_one():
    def fields(src):
        m = re.search(r"^struct\s+Camera\s*\{(.*?)^\}", src, re.S | re.M)
        body = re.sub(r"//[^\n]*", "", m.group(1))
        return tuple(f.strip() for f in body.split(",") if f.strip())

    mine = fields(_src())
    other = fields(open(os.path.join(WGSL_DIR, "sphere.wgsl"),
                        encoding="utf-8").read())
    check("heb_line's Camera block is byte-identical to sphere.wgsl's",
          mine == other, "" if mine == other else str(mine))



class Rig:
    def __init__(self, size=SIZE):
        self.ctx = device.acquire()
        self.target = target.Target(self.ctx, size, size)
        self.size = size
        self.library = shaders.ShaderLibrary(self.ctx)
        self.renderer = renderer.Renderer(self.ctx, self.target,
                                          library=self.library)

    def draw(self, polygons, beta, colors=None):
        """One instanced draw over control polygons of any lengths."""
        stride = max(len(p) for p in polygons)
        packed = np.zeros((len(polygons) * stride, 4), np.float32)
        for e, p in enumerate(polygons):
            p = np.asarray(p, np.float32)
            packed[e * stride:e * stride + p.shape[0], :3] = p
            # The count rides in the .w of the edge's first point.
            packed[e * stride, 3] = len(p)
        ctrl = self.ctx.device.create_buffer_with_data(
            label="test-ctrl", data=packed,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        heb = self.renderer.block_buffer(
            "heb", np.array([beta, stride, 0.0, 0.0], np.float32))
        if colors is None:
            colors = [((1, 1, 1, 1), (1, 1, 1, 1))] * len(polygons)
        spec = mesh.MeshSpec(
            topology=mesh.LINES, shader=mesh.ShaderRef("heb_line"),
            attrs={"col_a": np.array([c[0] for c in colors], np.float32),
                   "col_b": np.array([c[1] for c in colors], np.float32)})
        m = self.renderer.upload(spec)
        if m is None:
            raise AssertionError("heb_line did not resolve; this test would "
                                 "otherwise pass over an empty render pass")
        try:
            self.renderer.render([m], ortho_camera(),
                                 blocks={"heb": heb, "ctrl": ctrl})
            return self.target.read()[:, :, :3].astype(np.float64)
        finally:
            m.destroy()
            ctrl.destroy()


_RIG = None


def rig():
    global _RIG
    if _RIG is None:
        _RIG = Rig()
    return _RIG


def ink(rgb, threshold=0.02):
    return rgb.max(axis=2) > threshold


SAMPLE_T = np.linspace(0.0, 1.0, SAMPLES)


def reference_pixels(polygons, beta, dense=24):
    """(N, 2) float screen coords of the polyline the shader should draw."""
    camera = ortho_camera()
    out = []
    for p in polygons:
        pts = reference_curve(p, beta, SAMPLE_T)
        px = camera.project(pts, SIZE, SIZE)[:, :2]
        for i in range(len(px) - 1):
            f = np.linspace(0.0, 1.0, dense)[:, None]
            out.append(px[i] * (1 - f) + px[i + 1] * f)
    return np.concatenate(out, axis=0)


def _mask_from(points, radius):
    m = np.zeros((SIZE, SIZE), bool)
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, SIZE - 1)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, SIZE - 1)
    m[ys, xs] = True
    if radius:
        acc = m.copy()
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                acc |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
        m = acc
    return m


def compare(label, polygons, beta):
    """Both directions: no ink off the reference, and the reference inked."""
    img = rig().draw(polygons, beta)
    lit = ink(img)
    ref = reference_pixels(polygons, beta)
    near = _mask_from(ref, 2)
    stray = int((lit & ~near).sum())
    check(f"{label}: no lit pixel is off the reference curve",
          stray == 0, f"{stray} stray of {int(lit.sum())} lit")
    camera = ortho_camera()
    misses = 0
    for p in polygons:
        px = camera.project(reference_curve(p, beta, SAMPLE_T), SIZE, SIZE)
        for x, y, _ in px:
            x, y = int(round(x)), int(round(y))
            if not lit[max(0, y - 1):y + 2, max(0, x - 1):x + 2].any():
                misses += 1
    check(f"{label}: every reference sample point is drawn",
          misses == 0, f"{misses} of {len(polygons) * SAMPLES} missing")
    check(f"{label}: the draw is not empty", int(lit.sum()) > 100,
          f"{int(lit.sum())} lit px")
    return img, lit


# An arch, an S and two short paths, exercising the degree drop 3 to 2 to 1.

ARCH = [(-3.0, -1.5, 0.0), (-1.5, 2.0, 0.0), (0.0, 2.6, 0.0),
        (1.5, 2.0, 0.0), (3.0, -1.5, 0.0)]
ESS = [(-3.0, -2.5, 0.0), (-2.0, 1.0, 0.0), (-0.5, -2.0, 0.0),
       (1.0, 2.0, 0.0), (2.0, -1.0, 0.0), (3.0, 2.5, 0.0),
       (3.4, 0.0, 0.0)]
TRIPLE = [(-3.0, 0.0, 0.0), (0.0, 3.0, 0.0), (3.0, 0.0, 0.0)]
PAIR = [(-3.0, -3.0, 0.0), (3.0, 3.0, 0.0)]
DEEP = [(-3.5, 0.0, 0.0)] + [(-3.5 + i * 0.5, 2.5 * np.sin(i * 0.7), 0.0)
                             for i in range(1, 14)] + [(3.5, 0.0, 0.0)]


def test_the_curve_matches_the_basis_sum_per_pixel():
    for name, poly in (("a 5-point arch", ARCH), ("a 7-point S", ESS),
                       ("a 15-point path", DEEP)):
        for beta in (1.0, 0.6):
            compare(f"{name}, beta {beta}", [poly], beta)


def test_the_degree_drops_when_the_path_is_too_short():
    compare("a 3-point path (quadratic)", [TRIPLE], 1.0)
    compare("a 2-point path (linear)", [PAIR], 1.0)
    img = rig().draw([PAIR], 1.0)
    camera = ortho_camera()
    ends = camera.project(np.array(PAIR, np.float64), SIZE, SIZE)[:, :2]
    ys, xs = np.nonzero(ink(img))
    d = ends[1] - ends[0]
    d = d / np.hypot(*d)
    perp = np.array([-d[1], d[0]])
    off = np.abs((np.stack([xs, ys], axis=1) - ends[0]) @ perp)
    check("a 2-point path draws the straight chord and nothing else",
          off.max() < 2.0, f"max deviation {off.max():.2f} px")


def test_the_curve_interpolates_its_endpoints():
    camera = ortho_camera()
    for name, poly in (("arch", ARCH), ("S", ESS), ("triple", TRIPLE)):
        for beta in (1.0, 0.4):
            got = reference_curve(poly, beta, [0.0, 1.0])
            ends = np.array([poly[0], poly[-1]], np.float64)
            err = np.abs(got - ends).max()
            check(f"{name} beta {beta}: the reference interpolates its ends",
                  err < 1e-12, f"max error {err:.2e}")
        img = rig().draw([poly], 1.0)
        lit = ink(img)
        px = camera.project(np.array([poly[0], poly[-1]], np.float64),
                            SIZE, SIZE)
        hit = all(lit[max(0, int(round(y)) - 1):int(round(y)) + 2,
                      max(0, int(round(x)) - 1):int(round(x)) + 2].any()
                  for x, y, _ in px)
        check(f"{name}: the DRAWN curve reaches both control-polygon ends", hit)


def test_beta_zero_is_exactly_straight_and_beta_is_exactly_proportional():
    chord_a, chord_b = np.array(ARCH[0]), np.array(ARCH[-1])
    d = chord_b - chord_a
    d = d / np.linalg.norm(d)

    def deviation(beta):
        pts = reference_curve(ARCH, beta, np.linspace(0, 1, 400))
        rel = pts - chord_a
        return float(np.abs(np.cross(rel, d)).max())

    at_one = deviation(1.0)
    check("beta = 0 is exactly straight in the reference",
          deviation(0.0) < 1e-12, f"{deviation(0.0):.2e}")
    errs = [abs(deviation(b) - b * at_one) for b in (0.25, 0.5, 0.75)]
    check("and the deviation is exactly proportional to beta",
          max(errs) < 1e-12, f"max error {max(errs):.2e}")

    camera = ortho_camera()
    ends = camera.project(np.array([ARCH[0], ARCH[-1]], np.float64),
                          SIZE, SIZE)[:, :2]
    u = ends[1] - ends[0]
    u = u / np.hypot(*u)
    perp = np.array([-u[1], u[0]])
    measured = []
    for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
        img = rig().draw([ARCH], beta)
        ys, xs = np.nonzero(ink(img))
        off = np.abs((np.stack([xs, ys], axis=1) - ends[0]) @ perp)
        measured.append(float(off.max()))
    check("the drawn curve at beta = 0 is the chord, to within a pixel",
          measured[0] < 2.0, f"{measured[0]:.2f} px off the chord")
    span = measured[-1]
    check("the drawn deviation really does span a lot at beta = 1",
          span > 40.0, f"{span:.1f} px, so the proportionality means something")
    errs = [abs(m - b * span)
            for m, b in zip(measured, (0.0, 0.25, 0.5, 0.75, 1.0))]
    check("and the drawn deviation is proportional to beta, in pixels",
          max(errs) < 2.5,
          f"{[round(m, 1) for m in measured]}, max error {max(errs):.2f} px")


def test_one_draw_carries_control_polygons_of_different_lengths():
    # Spread out in y so each curve owns its own band of the image.
    def shift(poly, dy, sx=1.0):
        return [(x * sx, y * 0.25 + dy, z) for x, y, z in poly]

    polys = [shift(PAIR, -3.0), shift(TRIPLE, -1.0), shift(ARCH, 1.0),
             shift(DEEP, 3.0)]
    compare("four polygons of length 2/3/5/15 in one draw", polys, 1.0)
    # Named individually, so a failure says which one.
    img = rig().draw(polys, 1.0)
    lit = ink(img)
    camera = ortho_camera()
    for i, p in enumerate(polys):
        ref = camera.project(reference_curve(p, 1.0, SAMPLE_T), SIZE, SIZE)
        miss = sum(0 if lit[max(0, int(round(y)) - 1):int(round(y)) + 2,
                             max(0, int(round(x)) - 1):int(round(x)) + 2].any()
                   else 1 for x, y, _ in ref)
        check(f"polygon {i} ({len(p)} control points) is drawn correctly",
              miss == 0, f"{miss} of {SAMPLES} sample points missing")


def test_each_edge_is_its_own_strip():
    a = [(-3.5, 3.0, 0.0), (-2.0, 3.5, 0.0), (-0.5, 3.0, 0.0)]
    b = [(0.5, -3.0, 0.0), (2.0, -3.5, 0.0), (3.5, -3.0, 0.0)]
    img = rig().draw([a, b], 1.0)
    lit = ink(img)
    camera = ortho_camera()
    (ex, ey, _), (sx, sy, _) = camera.project(
        np.array([a[-1], b[0]], np.float64), SIZE, SIZE)
    # The middle of the segment that would join them.
    f = np.linspace(0.25, 0.75, 40)
    xs = np.round(ex + (sx - ex) * f).astype(int)
    ys = np.round(ey + (sy - ey) * f).astype(int)
    joined = int(lit[ys, xs].sum())
    check("nothing is drawn between the end of one curve and the start of "
          "the next", joined == 0, f"{joined} of 40 sampled px are lit")
    check("and both curves really were drawn", int(lit.sum()) > 200,
          f"{int(lit.sum())} lit px")


def test_the_color_is_linear_between_the_two_endpoint_colors():
    img = rig().draw([ARCH], 1.0,
                     colors=[((1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0))])
    camera = ortho_camera()
    px = camera.project(reference_curve(ARCH, 1.0, [0.02, 0.5, 0.98]),
                        SIZE, SIZE)

    def sample(x, y):
        x, y = int(round(x)), int(round(y))
        win = img[max(0, y - 1):y + 2, max(0, x - 1):x + 2].reshape(-1, 3)
        win = win[win.max(axis=1) > 0.02]
        return win.mean(axis=0) if win.size else np.zeros(3)

    start, mid, end = (sample(x, y) for x, y, _ in px)
    check("the curve starts at col_a", start[0] > 0.9 and start[2] < 0.1,
          f"rgb {start.round(3).tolist()}")
    check("and ends at col_b", end[2] > 0.9 and end[0] < 0.1,
          f"rgb {end.round(3).tolist()}")
    check("and is halfway between them in the middle",
          abs(mid[0] - 0.5) < 0.08 and abs(mid[2] - 0.5) < 0.08,
          f"rgb {mid.round(3).tolist()}")


def test_moving_beta_touches_only_the_16_byte_block():
    """Changing bundling strength used to cost 1196 ms and 239 MB of re-upload.
    Now the polygons upload once and beta is a uniform."""
    stride = len(ARCH)
    packed = np.zeros((stride, 4), np.float32)
    packed[:, :3] = np.array(ARCH, np.float32)
    packed[0, 3] = stride
    ctx = rig().ctx
    ctrl = ctx.device.create_buffer_with_data(
        label="sweep-ctrl", data=packed,
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
    heb = ctx.device.create_buffer(
        label="sweep-heb", size=16,
        usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
    spec = mesh.MeshSpec(
        topology=mesh.LINES, shader=mesh.ShaderRef("heb_line"),
        attrs={"col_a": np.ones((1, 4), np.float32),
               "col_b": np.ones((1, 4), np.float32)})
    m = rig().renderer.upload(spec)
    check("the batch uploaded once", m is not None)
    try:
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            ctx.queue.write_buffer(heb, 0, np.array(
                [beta, stride, 0.0, 0.0], np.float32).tobytes())
            rig().renderer.render([m], ortho_camera(),
                                  blocks={"heb": heb, "ctrl": ctrl})
            img = rig().target.read()[:, :, :3].astype(np.float64)
            lit = ink(img)
            near = _mask_from(reference_pixels([ARCH], beta), 2)
            stray = int((lit & ~near).sum())
            check(f"beta {beta} over the SAME buffers matches the reference",
                  stray == 0 and int(lit.sum()) > 100,
                  f"{stray} stray of {int(lit.sum())} lit")
    finally:
        m.destroy()
        ctrl.destroy()


TESTS = [
    test_the_sample_count_is_declared_once,
    test_heb_line_loads_and_declares_an_instanced_strip,
    test_the_camera_block_is_the_shared_one,
    test_the_curve_matches_the_basis_sum_per_pixel,
    test_the_degree_drops_when_the_path_is_too_short,
    test_the_curve_interpolates_its_endpoints,
    test_beta_zero_is_exactly_straight_and_beta_is_exactly_proportional,
    test_one_draw_carries_control_polygons_of_different_lengths,
    test_each_edge_is_its_own_strip,
    test_the_color_is_linear_between_the_two_endpoint_colors,
    test_moving_beta_touches_only_the_16_byte_block,
]


def main(argv=None):
    import traceback

    patterns = list(argv or [])
    try:
        print(rig().ctx.describe())
    except Exception as exc:                            # noqa: BLE001
        print(f"NO GPU: {exc!r}")
        return 2

    ran = 0
    for fn in TESTS:
        if patterns and not any(p in fn.__name__ for p in patterns):
            continue
        ran += 1
        print(f"\n{fn.__name__}")
        try:
            fn()
        except Exception:                               # noqa: BLE001
            FAILED.append(fn.__name__)
            print(''.join('       ' + ln
                          for ln in traceback.format_exc().splitlines(True)))

    print(f"\n{ran} tests, {CHECKS} checks")
    if FAILED:
        print(f"{len(FAILED)} CHECKS FAILED")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    if ran == 0:
        print("NO TESTS RAN")
        return 1
    if not patterns and CHECKS < MIN_CHECKS:
        print(f"ONLY {CHECKS} CHECKS RAN, {MIN_CHECKS} DECLARED")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
