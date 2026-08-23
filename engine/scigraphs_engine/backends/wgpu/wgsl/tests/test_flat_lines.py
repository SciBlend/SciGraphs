# uniform_color and smooth_color, the two unfiltered flat lines. Both are a few
# lines of WGSL, so a suite can go green without reaching the code: every
# geometric check says where the ink is, computed from camera.project rather
# than counting pixels. One pixel wide, unlike Blender's (../README.md 3.3).

import dataclasses
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WGSL_DIR = os.path.dirname(HERE)
ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(WGSL_DIR))))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import wgpu                                                       # noqa: E402

from scigraphs_engine import mesh                                 # noqa: E402
from scigraphs_engine.mesh import DrawState                       # noqa: E402
from scigraphs_engine.backends.wgpu import (camera as cam,        # noqa: E402
                                            device, renderer,
                                            shaders, target)

FAILED = []
CHECKS = 0

# Floor for ../tests/run.py, set from a passing run.
MIN_CHECKS = 100

MINE = ("uniform_color", "smooth_color")


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


# Orthographic, so every pixel is a closed form. One test goes perspective, the
# only place the two interpolations differ.

SIZE = 256
HALF = 4.0
EYE_Z = 10.0
NEAR, FAR = 1.0, 100.0


def ortho_camera():
    view = cam.look_at((0.0, 0.0, EYE_Z), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    return cam.Camera(view, cam.orthographic(HALF, HALF, NEAR, FAR), NEAR, FAR)


def perspective_camera():
    view = cam.look_at((0.0, 0.0, 8.0), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    return cam.Camera(view, cam.perspective(45.0, 1.0, 0.5, 100.0), 0.5, 100.0)


def headlight(rim=0.0, ambient=0.05, key=0.9):
    return np.array([
        0.0, 0.0, 1.0, 0.0,
        key, key, key, rim,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        ambient, ambient, ambient, 0.0,
    ], dtype=np.float32)


class Rig:
    def __init__(self, size=SIZE):
        self.ctx = device.acquire()
        self.target = target.Target(self.ctx, size, size)
        self.size = size
        self.target.depth = self.ctx.device.create_texture(
            label="test-depth-readable", size=(size, size, 1),
            format=target.DEPTH_FORMAT,
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                   | wgpu.TextureUsage.COPY_SRC))
        self.target.depth_view = self.target.depth.create_view()
        self.library = shaders.ShaderLibrary(self.ctx)
        self.renderer = renderer.Renderer(self.ctx, self.target,
                                          library=self.library)
        self.lights = self.renderer.block_buffer("lights", headlight())

    def draw(self, specs, camera=None, tint=(1.0, 1.0, 1.0, 1.0),
             clear_color=(0.0, 0.0, 0.0, 1.0)):
        """Upload, render, return (rgb (H, W, 3) float64, depth (H, W))."""
        specs = [s for s in specs if s is not None]
        meshes = [self.renderer.upload(s) for s in specs]
        if any(m is None for m in meshes):
            # An unported shader leaves every ink check on an empty frame.
            missing = [shaders.variant_name(s.shader)
                       for s, m in zip(specs, meshes) if m is None]
            raise AssertionError(
                f"no shader for {missing}; without this raise the test would "
                f"pass over an empty render pass")
        try:
            self.renderer.render(meshes, camera or ortho_camera(),
                                 blocks={"lights": self.lights}, tint=tint,
                                 clear_color=clear_color)
            img = self.target.read()[:, :, :3].astype(np.float64)
            return img, self.read_depth()
        finally:
            for m in meshes:
                m.destroy()

    def read_depth(self):
        """(H, W) float32 depth in [0, 1]. 1.0 is the cleared value."""
        w = h = self.size
        raw = self.ctx.queue.read_texture(
            {"texture": self.target.depth, "mip_level": 0, "origin": (0, 0, 0),
             "aspect": "depth-only"},
            {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
            (w, h, 1))
        return np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()


_RIG = None


def rig():
    global _RIG
    if _RIG is None:
        _RIG = Rig()
    return _RIG


# Derived from camera.project: retyping it would test this file on itself.

def lit_mask(rgb, threshold=0.02):
    return rgb.max(axis=2) > threshold


def lit_pixels(rgb, threshold=0.02):
    """(N, 2) float64 of the (x, y) pixel centers that carry ink."""
    ys, xs = np.nonzero(lit_mask(rgb, threshold))
    return np.stack([xs + 0.5, ys + 0.5], axis=1).astype(np.float64)


def segment_param(points, a, b):
    """Each point's clamped parameter t in [0, 1] along a->b."""
    d = b - a
    denom = float(d @ d)
    t = ((points - a) @ d) / denom
    return np.clip(t, 0.0, 1.0)


def distance_to_segment(points, a, b):
    t = segment_param(points, a, b)
    foot = a + t[:, None] * (b - a)
    return np.linalg.norm(points - foot, axis=1)


def endpoints_2d(camera, p0, p1, size=SIZE):
    px = camera.project(np.array([p0, p1], np.float32), size, size)
    return px[0, :2].astype(np.float64), px[1, :2].astype(np.float64)


def assert_ink_is_the_segment(label, rgb, camera, p0, p1, tol=1.0):
    a, b = endpoints_2d(camera, p0, p1)
    pts = lit_pixels(rgb)
    check(f"{label}: something was drawn at all", len(pts) > 10,
          f"{len(pts)} lit px")
    if len(pts) == 0:
        return a, b
    dist = distance_to_segment(pts, a, b)
    check(f"{label}: every lit pixel is on the predicted segment",
          float(dist.max()) <= tol,
          f"max {dist.max():.2f} px from the segment "
          f"({a.round(1).tolist()} -> {b.round(1).tolist()})")

    # One-pixel steps, so a two-pixel gap anywhere is a failure.
    n = int(np.linalg.norm(b - a)) + 1
    ts = np.linspace(0.0, 1.0, max(n, 2))
    along = a + ts[:, None] * (b - a)
    gaps = 0
    for point in along:
        if np.min(np.linalg.norm(pts - point, axis=1)) > tol:
            gaps += 1
    check(f"{label}: every point of the segment has ink within {tol} px",
          gaps == 0, f"{gaps} of {len(along)} sample points uncovered")
    return a, b


# Always through scigraphs_engine.mesh: the last check asserts these draw.

def uniform_line(p0, p1):
    """mesh.segment_line_spec with no colors -> ShaderRef("uniform_color")."""
    return mesh.segment_line_spec(np.array([p0], np.float32),
                                  np.array([p1], np.float32))


def indexed_lines(coords, edges):
    return mesh.line_spec(np.asarray(coords, np.float32),
                          np.asarray(edges, np.int32))


def smooth_line(p0, p1, c0, c1):
    """mesh.segment_line_spec with colors -> ShaderRef("smooth_color")."""
    return mesh.segment_line_spec(np.array([p0], np.float32),
                                  np.array([p1], np.float32),
                                  np.array([c0], np.float32),
                                  np.array([c1], np.float32))


def sphere(center, radius, color):
    return mesh.sphere_spec(np.array([center], np.float32),
                            np.array([color], np.float32),
                            np.array([radius], np.float32))


def with_state(spec, **fields):
    return dataclasses.replace(spec, state=DrawState(**fields))


def _src(name):
    return open(os.path.join(WGSL_DIR, name + ".wgsl"), encoding="utf-8").read()


def _fields(source, name):
    m = re.search(r"^struct\s+%s\s*\{(.*?)^\}" % name, source, re.S | re.M)
    if m is None:
        return None
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return tuple(f.strip() for f in body.split(",") if f.strip())


def test_the_shared_blocks_are_still_one_fact():
    cameras, draws = {}, {}
    for entry in sorted(os.listdir(WGSL_DIR)):
        if not entry.endswith(".wgsl"):
            continue
        src = open(os.path.join(WGSL_DIR, entry), encoding="utf-8").read()
        got = _fields(src, "Camera")
        if got is not None:
            cameras[entry] = got
        got = _fields(src, "Draw")
        if got is not None:
            draws[entry] = got

    for name in MINE:
        check(f"{name}.wgsl carries the shared Camera block",
              name + ".wgsl" in cameras)
    check("Camera is byte-identical across every file in wgsl/",
          len(set(cameras.values())) == 1,
          f"{len(set(cameras.values()))} distinct across {len(cameras)} files")
    check("uniform_color.wgsl carries the shared Draw block",
          "uniform_color.wgsl" in draws)
    check("Draw is byte-identical wherever it appears",
          len(set(draws.values())) == 1, f"in {sorted(draws)}")

    fields = draws.get("uniform_color.wgsl") or ()
    names = [f.split(":")[0].strip() for f in fields]
    # The order is the layout; nothing converts between them.
    check("Draw fields match renderer.draw_block's packing",
          names == ["point_size", "line_width", "weight", "flags", "tint"],
          str(names))
    check("smooth_color declares no Draw block",
          "smooth_color.wgsl" not in draws,
          "it has no uniform to read; declaring one would make every caller "
          "supply a block it cannot use")


def test_both_shaders_load_and_compile():
    """Through the real loader, so a malformed .json fails here and not later."""
    lib = rig().library
    for name in MINE:
        try:
            got = lib.get(name)
        except Exception as exc:                        # noqa: BLE001
            check(f"{name} loads", False, repr(exc))
            continue
        check(f"{name} loads and compiles", got is not None)
    check("and both are in the library's own list of what exists",
          set(MINE) <= set(lib.available()), str(lib.available()))


def test_the_declared_bindings_are_the_ones_the_wgsl_reads():
    """Slot numbers and visibility against the folder table in README 1.1."""
    lib = rig().library
    uc = {b.name: b for b in lib.get("uniform_color").bindings}
    sc = {b.name: b for b in lib.get("smooth_color").bindings}

    check("uniform_color declares camera at 0.0 and draw at 0.1",
          set(uc) == {"camera", "draw"}
          and (uc["camera"].group, uc["camera"].binding) == (0, 0)
          and (uc["draw"].group, uc["draw"].binding) == (0, 1),
          str(sorted(uc)))
    check("uniform_color's draw block is dynamic", uc["draw"].dynamic,
          "one region per draw call, or every draw reads the last tint written")
    check("uniform_color reads camera in the VERTEX stage only",
          uc["camera"].visibility == wgpu.ShaderStage.VERTEX)
    check("uniform_color reads draw in the FRAGMENT stage only",
          uc["draw"].visibility == wgpu.ShaderStage.FRAGMENT,
          "the tint is the fragment's whole output; nothing in vs_main reads it")
    check("smooth_color declares camera and nothing else", set(sc) == {"camera"},
          str(sorted(sc)))
    check("smooth_color reads camera in the VERTEX stage only",
          sc["camera"].visibility == wgpu.ShaderStage.VERTEX)

    for name in MINE:
        src = _src(name)
        for slot, block in ((0, "camera"), (1, "draw")):
            declared = block in {b.name for b in lib.get(name).bindings}
            present = f"@group(0) @binding({slot})" in src
            check(f"{name}: the .json and the .wgsl agree about {block}",
                  declared == present,
                  f"json says {declared}, wgsl says {present}")


def test_neither_flat_line_writes_a_fragment_depth():
    for name in MINE:
        src = _src(name)
        code = "\n".join(re.sub(r"//.*$", "", ln) for ln in src.splitlines())
        check(f"{name} writes no @builtin(frag_depth)",
              "frag_depth" not in code,
              "" if "frag_depth" not in code else "found one")
        remap = re.search(r"0\.5\s*\+\s*0\.5", code)
        check(f"{name} contains no 0.5 depth remap", remap is None,
              "" if remap is None else f"found {remap.group(0)!r}")
        decl = rig().library.get(name)
        # true claims nothing, false is a veto no caller can lift (README 1.3).
        check(f"{name} leaves the depth write to the caller (writes_depth true)",
              decl.writes_depth is True)
    check("and it agrees with line_f, the filtered twin of the same "
          "representation",
          rig().library.get("line_f").writes_depth
          == rig().library.get("uniform_color").writes_depth,
          "a filtered and an unfiltered flat line occluding differently would "
          "be invisible until someone moved a slider")


def test_json_attributes_match_the_geometry_builders():
    lib = rig().library
    specs = {
        "uniform_color": [
            ("line_spec", indexed_lines([[0, 0, 0], [1, 1, 0]], [[0, 1]])),
            ("segment_line_spec", uniform_line((0, 0, 0), (1, 1, 0))),
        ],
        "smooth_color": [
            ("segment_line_spec+colors",
             smooth_line((0, 0, 0), (1, 1, 0), (1, 0, 0, 1), (0, 0, 1, 1))),
        ],
    }
    for name, cases in specs.items():
        shader = lib.get(name)
        declared = [a.name for a in shader.attributes]
        for builder, spec in cases:
            check(f"{name}: {builder} resolves to the {name} variant",
                  shaders.variant_name(spec.shader) == name,
                  shaders.variant_name(spec.shader))
            check(f"{name}: json attributes == {builder}.attrs, in order",
                  declared == list(spec.attrs),
                  f"{declared} vs {list(spec.attrs)}")
            bad = []
            for a in shader.attributes:
                arr = np.asarray(spec.attrs[a.name])
                cols = 1 if arr.ndim == 1 else arr.shape[1]
                if cols != a.columns:
                    bad.append(f"{a.name}: {a.format} over {cols} columns")
            check(f"{name}: declared formats match {builder}'s array widths",
                  not bad, "; ".join(bad))


# Axis-aligned lines survive a transposed matrix and the diagonals do not, and
# the two diagonals have opposite slopes, so a y flip cannot pass as a rotation.
ORIENTATIONS = [
    ("horizontal", (-2.5, 0.0, 0.0), (2.5, 0.0, 0.0)),
    ("vertical", (0.0, -2.5, 0.0), (0.0, 2.5, 0.0)),
    ("diagonal up", (-2.0, -2.0, 0.0), (2.0, 2.0, 0.0)),
    ("diagonal down", (-2.0, 2.0, 0.0), (2.0, -2.0, 0.0)),
    ("shallow", (-3.0, -1.0, 0.0), (3.0, 0.6, 0.0)),
]


def test_uniform_color_draws_the_predicted_segment():
    for label, p0, p1 in ORIENTATIONS:
        rgb, _ = rig().draw([uniform_line(p0, p1)], tint=(1.0, 1.0, 1.0, 1.0))
        assert_ink_is_the_segment(f"uniform_color {label}", rgb,
                                  ortho_camera(), p0, p1)


def test_uniform_color_takes_its_color_from_the_per_draw_tint():
    p0, p1 = (-2.5, 0.0, 0.0), (2.5, 0.0, 0.0)
    spec = uniform_line(p0, p1)
    masks = []
    for tint in ((1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0),
                 (0.25, 0.75, 0.5, 1.0)):
        rgb, _ = rig().draw([spec], tint=tint)
        lit = lit_mask(rgb)
        masks.append(lit)
        got = rgb[lit]
        # rgba8unorm round-trips within half an LSB.
        err = float(np.abs(got - np.array(tint[:3])).max()) if got.size else 9.9
        check(f"uniform_color is exactly the tint {tint[:3]}", err <= 1.0 / 255.0,
              f"max channel error {err:.4f} over {int(lit.sum())} px")
    check("...and the three tints lit exactly the same pixels",
          bool((masks[0] == masks[1]).all() and (masks[0] == masks[2]).all()),
          "the color must not move the geometry")


def test_uniform_color_draws_the_indexed_edge_set():
    g = (np.arange(4) - 1.5) * 1.8
    xx, yy = np.meshgrid(g, g)
    coords = np.stack([xx.ravel(), yy.ravel(), np.zeros(16)],
                      axis=1).astype(np.float32)
    idx = np.arange(16).reshape(4, 4)
    edges = []
    for r in range(4):
        for c in range(4):
            if c + 1 < 4:
                edges.append((idx[r, c], idx[r, c + 1]))
            if r + 1 < 4:
                edges.append((idx[r, c], idx[r + 1, c]))
    edges = np.array(edges, np.int32)

    spec = indexed_lines(coords, edges)
    check("line_spec kept one vertex per node, not two per edge",
          spec.vertex_count == 16 and spec.indices.shape[0] == len(edges),
          f"{spec.vertex_count} vertices for {len(edges)} edges")
    m = rig().renderer.upload(spec)
    check("...and the upload is indexed", m is not None
          and m.index_buffer is not None and m.vertex_count == 16,
          f"{m.vertex_count} vertices, index buffer "
          f"{'present' if m.index_buffer is not None else 'MISSING'}")
    m.destroy()

    rgb, _ = rig().draw([spec], tint=(1.0, 1.0, 1.0, 1.0))
    pts = lit_pixels(rgb)
    ends = ortho_camera().project(coords, SIZE, SIZE)[:, :2].astype(np.float64)
    worst = np.full(len(pts), np.inf)
    for a, b in edges:
        worst = np.minimum(worst, distance_to_segment(pts, ends[a], ends[b]))
    check("every lit pixel lies on one of the graph's edges",
          float(worst.max()) <= 1.0, f"max {worst.max():.2f} px, {len(pts)} px")
    missing = []
    for i, (a, b) in enumerate(edges):
        mid = 0.5 * (ends[a] + ends[b])
        if np.min(np.linalg.norm(pts - mid, axis=1)) > 1.0:
            missing.append(i)
    check("and every one of the graph's edges was drawn", not missing,
          f"{len(missing)} of {len(edges)} edges have no ink at their midpoint")


def test_the_line_is_exactly_one_pixel_wide():
    rgb, _ = rig().draw([with_state(uniform_line((-2.5, 0.0, 0.0),
                                                 (2.5, 0.0, 0.0)),
                                    line_width=8.0)],
                        tint=(1.0, 1.0, 1.0, 1.0))
    lit = lit_mask(rgb)
    per_column = lit.sum(axis=0)
    drawn = per_column[per_column > 0]
    check("a horizontal line is one pixel tall in every column it covers",
          drawn.size > 100 and int(drawn.max()) == 1,
          f"max {int(drawn.max()) if drawn.size else 0} px over "
          f"{drawn.size} columns, with line_width=8 in the draw state")


# smooth_color's geometry is uniform_color's, so what follows is all color.

def test_smooth_color_draws_the_predicted_segment():
    for label, p0, p1 in ORIENTATIONS[:3]:
        rgb, _ = rig().draw([smooth_line(p0, p1, (1, 1, 1, 1), (1, 1, 1, 1))])
        assert_ink_is_the_segment(f"smooth_color {label}", rgb, ortho_camera(),
                                  p0, p1)


def test_smooth_color_interpolates_between_its_endpoint_colors():
    p0, p1 = (-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)
    c0, c1 = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    rgb, _ = rig().draw([smooth_line(p0, p1, c0, c1)])
    a, b = endpoints_2d(ortho_camera(), p0, p1)
    pts = lit_pixels(rgb)
    check("smooth_color drew the segment", len(pts) > 100, f"{len(pts)} px")

    t = segment_param(pts, a, b)
    got = rgb[lit_mask(rgb)]
    order = np.argsort(t)
    t, got = t[order], got[order]
    want = (1.0 - t)[:, None] * np.array(c0[:3]) + t[:, None] * np.array(c1[:3])
    err = float(np.abs(got - want).max())
    # 1/255 quantization plus ~0.003 of pixel-center-to-parameter error.
    check("every pixel is the linear blend of the two endpoint colors",
          err < 0.01, f"max channel error {err:.4f} over {len(t)} px")

    mid = np.argmin(np.abs(t - 0.5))
    check("the midpoint is half of each",
          float(np.abs(got[mid] - np.array([0.5, 0.0, 0.5])).max()) < 0.01,
          f"t={t[mid]:.3f} -> {got[mid].round(3).tolist()}")
    check("the red end is at the first endpoint, not the second",
          got[0][0] > 0.9 and got[0][2] < 0.1
          and got[-1][2] > 0.9 and got[-1][0] < 0.1,
          f"{got[0].round(2).tolist()} -> {got[-1].round(2).tolist()}")
    check("and the gradient is not constant",
          float(got[:, 0].max() - got[:, 0].min()) > 0.9,
          f"red spans {got[:, 0].min():.2f}..{got[:, 0].max():.2f}")


def test_smooth_color_interpolation_is_perspective_correct():
    """The only thing separating this shader from @interpolate(linear); under
    an orthographic camera the two are identical, because every w is 1.
    Reference: c(s) = ((1-s)c0/w0 + s c1/w1) / ((1-s)/w0 + s/w1)."""
    camera = perspective_camera()
    # Steeply receding, so w0 and w1 are far apart.
    p0, p1 = (-1.5, 0.0, 3.0), (1.5, 0.0, -6.0)
    c0, c1 = (1.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)
    rgb, _ = rig().draw([smooth_line(p0, p1, c0, c1)], camera=camera)

    vp = camera.view_proj
    clip = np.concatenate([np.array([p0, p1], np.float64),
                           np.ones((2, 1))], axis=1) @ vp.T.astype(np.float64)
    w0, w1 = float(clip[0, 3]), float(clip[1, 3])
    check("the two endpoints really are at different depths",
          max(w0, w1) / min(w0, w1) > 2.0, f"w = {w0:.2f} and {w1:.2f}")

    a, b = endpoints_2d(camera, p0, p1)
    pts = lit_pixels(rgb)
    check("smooth_color drew the receding segment", len(pts) > 50,
          f"{len(pts)} px between {a.round(1).tolist()} and {b.round(1).tolist()}")
    s = segment_param(pts, a, b)
    got = rgb[lit_mask(rgb)]

    c0a, c1a = np.array(c0[:3]), np.array(c1[:3])
    inv = (1.0 - s) / w0 + s / w1
    persp = ((1.0 - s)[:, None] * c0a / w0 + s[:, None] * c1a / w1) / inv[:, None]
    linear = (1.0 - s)[:, None] * c0a + s[:, None] * c1a

    err_p = float(np.abs(got - persp).max())
    err_l = float(np.abs(got - linear).max())
    apart = float(np.abs(persp - linear).max())
    check("the perspective-correct and screen-linear answers are far apart here",
          apart > 0.15, f"they differ by up to {apart:.3f}")
    check("the drawn color is the perspective-correct one", err_p < 0.02,
          f"error {err_p:.4f} against perspective, {err_l:.4f} against linear")
    check("...and it is NOT the screen-linear one", err_l > 0.1,
          f"error {err_l:.4f}")


def test_the_flat_lines_pass_alpha_through():
    bg = (0.0, 0.4, 0.0, 1.0)
    alpha = 0.25
    rgb_in = (1.0, 0.0, 0.0)
    want = np.array(bg[:3]) * (1.0 - alpha) + np.array(rgb_in) * alpha
    p0, p1 = (-3.0, 0.0, 0.0), (3.0, 0.0, 0.0)

    cases = {
        "uniform_color (alpha from draw.tint)": (
            with_state(uniform_line(p0, p1), blend="ALPHA", depth_write=False),
            rgb_in + (alpha,)),
        "smooth_color (alpha from the attribute)": (
            with_state(smooth_line(p0, p1, rgb_in + (alpha,),
                                   rgb_in + (alpha,)),
                       blend="ALPHA", depth_write=False),
            (1.0, 1.0, 1.0, 1.0)),
    }
    for label, (spec, tint) in cases.items():
        rgb, _ = rig().draw([spec], tint=tint, clear_color=bg)
        # Red only ever comes from the line; the background has none.
        on_line = rgb[:, :, 0] > 1.0 / 255.0
        got = rgb[on_line]
        check(f"{label} composites as alpha {alpha} over the background",
              got.size > 0 and float(np.abs(got - want).max()) < 0.01,
              f"{got.mean(axis=0).round(3).tolist() if got.size else '-'} "
              f"vs {want.round(3).tolist()} over {int(on_line.sum())} px")

    check("...which is nothing like what forcing alpha to 1 would give",
          float(np.abs(want - np.array(rgb_in)).max()) > 0.5,
          f"opaque would be {np.array(rgb_in).round(3).tolist()}")


def test_a_flat_line_and_a_sphere_share_one_depth_buffer():
    r = 1.5
    behind = uniform_line((-3.0, 0.0, -r * 2.0), (3.0, 0.0, -r * 2.0))
    front = uniform_line((-3.0, 0.6, r * 2.0), (3.0, 0.6, r * 2.0))
    ball = sphere((0.0, 0.0, 0.0), r, (0.0, 0.0, 1.0, 1.0))
    camera = ortho_camera()

    rgb, depth = rig().draw([behind, ball, front], camera=camera,
                            tint=(1.0, 0.0, 0.0, 1.0))
    # Red only ever comes from the lines: the sphere is pure blue.
    line_ink = (rgb[:, :, 0] > 0.4) & (rgb[:, :, 2] < 0.4)

    a, b = endpoints_2d(camera, (-3.0, 0.0, -r * 2.0), (3.0, 0.0, -r * 2.0))
    row = int(round((a[1] + b[1]) * 0.5)) - 1
    covered = depth[row] < 1.0
    sphere_cols = np.nonzero(rgb[row, :, 2] > 0.2)[0]
    check("the sphere covers part of the back line's row",
          sphere_cols.size > 40, f"{sphere_cols.size} px of sphere on row {row}")
    inside = line_ink[row, sphere_cols]
    check("the line behind the sphere is hidden where the sphere is",
          not bool(inside.any()),
          f"{int(inside.sum())} px of line survived through the sphere")

    outside = np.ones(SIZE, bool)
    outside[sphere_cols] = False
    outside[:1] = outside[-1:] = False
    check("...and visible on both sides of it",
          bool(line_ink[row - 1:row + 2, outside].any(axis=0).sum() > 40),
          f"{int(line_ink[row - 1:row + 2, outside].any(axis=0).sum())} px")

    a2, b2 = endpoints_2d(camera, (-3.0, 0.6, r * 2.0), (3.0, 0.6, r * 2.0))
    row2 = int(round((a2[1] + b2[1]) * 0.5))
    over = [rw for rw in (row2 - 1, row2, row2 + 1)
            if line_ink[rw, sphere_cols].sum() > 40]
    check("the line in FRONT of the sphere is drawn over it", bool(over),
          f"rows {row2 - 1}..{row2 + 1}: "
          f"{[int(line_ink[rw, sphere_cols].sum()) for rw in (row2 - 1, row2, row2 + 1)]}")

    # Without the depth buffer this pass would be a painter's algorithm.
    rgb2, _ = rig().draw([front, ball, behind], camera=camera,
                         tint=(1.0, 0.0, 0.0, 1.0))
    check("and the image does not depend on the order the specs were listed",
          float(np.abs(rgb - rgb2).max()) == 0.0,
          f"max channel difference {float(np.abs(rgb - rgb2).max()):.4f}")


def test_two_flat_lines_that_cross_sort_by_depth():
    """Measure the pipeline's depth write rather than reading it off the .json."""
    near = uniform_line((-3.0, 0.0, 2.0), (3.0, 0.0, 2.0))     # horizontal
    far = uniform_line((0.0, -3.0, -2.0), (0.0, 3.0, -2.0))    # vertical
    camera = ortho_camera()
    cx, cy = camera.project(np.array([[0.0, 0.0, 0.0]], np.float32),
                            SIZE, SIZE)[0, :2]
    box = (slice(int(cy) - 1, int(cy) + 2), slice(int(cx) - 1, int(cx) + 2))

    first, _ = rig().draw([near, far], camera=camera, tint=(1.0, 1.0, 1.0, 1.0))
    second, _ = rig().draw([far, near], camera=camera, tint=(1.0, 1.0, 1.0, 1.0))
    check("both crossing lines were drawn",
          int(lit_mask(first).sum()) > 300,
          f"{int(lit_mask(first).sum())} px")
    check("the crossing pixel is the same whichever line was recorded first",
          float(np.abs(first[box] - second[box]).max()) == 0.0,
          f"max difference {float(np.abs(first[box] - second[box]).max()):.4f}")
    # The check means nothing unless the far line would have covered it.
    alone, depth_far = rig().draw([far], camera=camera, tint=(1, 1, 1, 1))
    check("...and the far line really does pass through that pixel alone",
          bool(lit_mask(alone)[box].any()),
          "otherwise there is nothing for the depth test to decide")
    _, depth_near = rig().draw([near], camera=camera, tint=(1, 1, 1, 1))
    check("the nearer line has the smaller depth value there",
          float(depth_near[box].min()) < float(depth_far[box].min()),
          f"{float(depth_near[box].min()):.4f} vs "
          f"{float(depth_far[box].min()):.4f}")
    _, both = rig().draw([far, near], camera=camera, tint=(1, 1, 1, 1))
    check("and the depth buffer after both draws holds the nearer one",
          abs(float(both[box].min()) - float(depth_near[box].min())) < 1e-6,
          f"{float(both[box].min()):.4f} vs "
          f"{float(depth_near[box].min()):.4f} for the near line alone")


def test_the_line_depth_is_the_rasterizers_and_it_is_right():
    camera = ortho_camera()
    p0, p1 = (-3.0, 1.0, -2.0), (3.0, -1.0, 2.0)
    rgb, depth = rig().draw([smooth_line(p0, p1, (1, 1, 1, 1), (1, 1, 1, 1))],
                            camera=camera)
    ends = camera.project(np.array([p0, p1], np.float32), SIZE, SIZE)
    a, b = ends[0, :2].astype(np.float64), ends[1, :2].astype(np.float64)
    z0, z1 = float(ends[0, 2]), float(ends[1, 2])
    check("the two ends really are at different depths", abs(z1 - z0) > 0.01,
          f"{z0:.4f} .. {z1:.4f}")

    pts = lit_pixels(rgb)
    t = segment_param(pts, a, b)
    ys, xs = np.nonzero(lit_mask(rgb))
    got = depth[ys, xs].astype(np.float64)
    want = (1.0 - t) * z0 + t * z1
    err = float(np.abs(got - want).max())
    check("the depth buffer holds the interpolated clip z, with no remap",
          err < 2e-3, f"max error {err:.6f} over {len(t)} px")
    remapped = want * 0.5 + 0.5
    check("...and that is measurably not the OpenGL remap of it",
          float(np.abs(got - remapped).min()) > 0.1,
          f"the remap would sit {float(np.abs(want - remapped).mean()):.3f} away")


def test_the_engines_own_specs_now_upload_instead_of_returning_none():
    seg_a = np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    seg_b = np.array([[0.0, 1.0, 0.0], [2.0, -1.0, 0.0]], np.float32)
    col = np.array([[1, 0, 0, 1], [0, 1, 0, 1]], np.float32)
    cases = {
        "mesh.line_spec": indexed_lines(
            [[-2, 0, 0], [0, 1.5, 0], [2, 0, 0]], [[0, 1], [1, 2]]),
        "mesh.segment_line_spec (no colors)":
            mesh.segment_line_spec(seg_a, seg_b),
        "mesh.segment_line_spec (colors)":
            mesh.segment_line_spec(seg_a, seg_b, col, col[::-1]),
    }
    for label, spec in cases.items():
        m = rig().renderer.upload(spec)
        check(f"{label} uploads (was None before this port)", m is not None,
              shaders.variant_name(spec.shader))
        if m is not None:
            m.destroy()
        rgb, _ = rig().draw([spec], tint=(1.0, 1.0, 1.0, 1.0))
        check(f"...and {label} puts ink on the screen",
              int(lit_mask(rgb).sum()) > 50,
              f"{int(lit_mask(rgb).sum())} px")

    still_missing = {}
    for label, ref in (("mesh.bucketed_line_spec", "polyline_uniform_color"),
                       ("mesh.bucketed_segment_spec", "polyline_smooth_color")):
        if rig().library.get(ref) is None:
            still_missing[label] = ref
    check("the bucketed builders are still unported, and say so",
          set(still_missing.values())
          == {"polyline_uniform_color", "polyline_smooth_color"},
          f"{still_missing} -- not in this port's scope; wgsl/README.md 9")


def test_a_line_tier_render_through_the_public_api_skips_nothing():
    from scigraphs_engine.api import Graph, Camera
    from scigraphs_engine.tests import graphs

    coords, edges, node_attrs, edge_attrs = graphs.clustered()
    g = Graph.from_arrays(coords, edges, node_attrs=node_attrs,
                          edge_attrs=edge_attrs)
    size = (192, 108)
    bg = (0.02, 0.02, 0.05, 1.0)
    camera = Camera.fit(g)

    def render(**style):
        styled = g.style(**style)
        img = styled.render(camera, size=size, background=bg)
        lit = img.array[..., :3].max(axis=2) > max(bg[:3]) + 0.05
        names = {sp.shader.base for sp in styled.geometry().specs}
        return img, lit, names

    _, lit_none, _ = render(edge_tier="none", nodes="point")
    for label, style in (
            ("uniform_color", dict(edge_tier="line", nodes="point")),
            ("smooth_color", dict(edge_tier="line", nodes="point",
                                  edges="hierarchical"))):
        img, lit, names = render(**style)
        check(f"the {label} scene really does emit {label}", label in names,
              str(sorted(names)))
        check(f"{label}: nothing was skipped for want of a shader",
              img.stats["skipped_shaders"] == (),
              str(img.stats["skipped_shaders"]))
        check(f"{label}: and no warning was attached",
              "warning" not in img.stats, img.stats.get("warning", "-"))
        new = int((lit & ~lit_none).sum())
        check(f"{label}: the edges are actually on the screen",
              new > 100,
              f"{new} pixels lit that edge_tier=none leaves dark, over a "
              f"{size[0]}x{size[1]} frame")
        check(f"{label}: ...and it did not lose any of the nodes",
              int((lit_none & ~lit).sum()) <= int(lit_none.sum()) * 0.02,
              f"{int((lit_none & ~lit).sum())} of {int(lit_none.sum())} "
              f"node pixels went dark")


TESTS = [
    test_the_shared_blocks_are_still_one_fact,
    test_both_shaders_load_and_compile,
    test_the_declared_bindings_are_the_ones_the_wgsl_reads,
    test_neither_flat_line_writes_a_fragment_depth,
    test_json_attributes_match_the_geometry_builders,
    test_uniform_color_draws_the_predicted_segment,
    test_uniform_color_takes_its_color_from_the_per_draw_tint,
    test_uniform_color_draws_the_indexed_edge_set,
    test_the_line_is_exactly_one_pixel_wide,
    test_smooth_color_draws_the_predicted_segment,
    test_smooth_color_interpolates_between_its_endpoint_colors,
    test_smooth_color_interpolation_is_perspective_correct,
    test_the_flat_lines_pass_alpha_through,
    test_a_flat_line_and_a_sphere_share_one_depth_buffer,
    test_two_flat_lines_that_cross_sort_by_depth,
    test_the_line_depth_is_the_rasterizers_and_it_is_right,
    test_the_engines_own_specs_now_upload_instead_of_returning_none,
    test_a_line_tier_render_through_the_public_api_skips_nothing,
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
