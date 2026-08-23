# Pixel tests for the WGSL impostor shaders. These compile with the depth write
# deleted and still draw roughly right, so compiling proves nothing. Each test
# renders a scene numpy can answer and checks every covered pixel; the depth
# tests also work out what a flat billboard would draw and assert they differ.

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WGSL_DIR = os.path.dirname(HERE)
# .../engine, so that `scigraphs_engine` imports from the working tree.
ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(WGSL_DIR))))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import wgpu                                                       # noqa: E402

from scigraphs_engine import mesh                                 # noqa: E402
from scigraphs_engine.backends.wgpu import (camera as cam,        # noqa: E402
                                            device, renderer,
                                            shaders, target)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


# Orthographic, so the expected value per pixel is a closed form. The camera
# sits on +Z, so view space is world space shifted by -EYE_Z in z.

SIZE = 256
HALF = 4.0            # ortho half-extent in world units
EYE_Z = 10.0
NEAR, FAR = 1.0, 100.0
PX_PER_UNIT = SIZE / (2.0 * HALF)     # 32 px per world unit


def ortho_camera():
    view = cam.look_at((0.0, 0.0, EYE_Z), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    proj = cam.orthographic(HALF, HALF, NEAR, FAR)
    return cam.Camera(view, proj, NEAR, FAR)


def pixel_grid():
    """(view_x, view_y) at the center of every pixel, as two (H, W) arrays."""
    px = (np.arange(SIZE, dtype=np.float64) + 0.5)
    ndc = px / SIZE * 2.0 - 1.0
    x = ndc * HALF
    y = -ndc * HALF
    return np.meshgrid(x, y)


def ortho_depth(view_z):
    """View-space z -> the [0, 1] depth value camera.orthographic produces."""
    return (view_z + NEAR) / (NEAR - FAR)


def vz(world_z):
    return world_z - EYE_Z


# Pointing straight at the camera, so the expected shading is a closed form.
def headlight(rim=0.0, ambient=0.05, key=0.9):
    return np.array([
        0.0, 0.0, 1.0, 0.0,          # key_dir
        key, key, key, rim,          # key_col, w = rim strength
        0.0, 0.0, 1.0, 0.0,          # fill_dir
        0.0, 0.0, 0.0, 0.0,          # fill_col
        ambient, ambient, ambient, 0.0,   # ambient
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

    def draw(self, specs, camera=None, lights=None):
        """Upload, render, return (rgb (H, W, 3), depth (H, W))."""
        blocks = {"lights": lights if lights is not None else self.lights}
        meshes = [self.renderer.upload(s) for s in specs if s is not None]
        missing = [s for s, m in zip([s for s in specs if s is not None],
                                     meshes) if m is None]
        if missing:
            raise AssertionError("shader missing for a spec; this test would "
                                 "otherwise pass over an empty render pass")
        self.renderer.render(meshes, camera or ortho_camera(), blocks=blocks)
        img = self.target.read()[:, :, :3].astype(np.float64)
        return img, self.read_depth()

    def read_depth(self):
        """(H, W) float32 depth in [0, 1]. 1.0 is the cleared value."""
        w = h = self.size
        # bytes_per_row must be a multiple of 256; 256*4 already is here.
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


MINE = ("sphere", "ribbon", "sphere_id", "ribbon_id", "arrow")


def _struct(source, name):
    m = re.search(r"^struct\s+%s\s*\{(.*?)^\}" % name, source,
                  re.S | re.M)
    if m is None:
        return None
    # Field declarations only: comments and whitespace may differ.
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return tuple(f.strip() for f in body.split(",") if f.strip())


def test_shared_uniform_blocks_are_identical():
    cameras, rigs = {}, {}
    for name in MINE:
        src = open(os.path.join(WGSL_DIR, name + ".wgsl"),
                   encoding="utf-8").read()
        cameras[name] = _struct(src, "Camera")
        got = _struct(src, "LightRig")
        if got is not None:
            rigs[name] = got

    distinct = {v for v in cameras.values()}
    check("Camera block identical in all five files", len(distinct) == 1,
          f"{len(distinct)} distinct")
    distinct = {v for v in rigs.values()}
    check("LightRig block identical wherever it appears", len(distinct) == 1,
          f"in {sorted(rigs)}")

    fields = next(iter(cameras.values())) or ()
    names = [f.split(":")[0].strip() for f in fields]
    check("Camera fields match README 1.3",
          names == ["view", "proj", "view_proj", "viewport", "params"],
          str(names))

    fields = next(iter(rigs.values())) or ()
    names = [f.split(":")[0].strip() for f in fields]
    # The order is the layout; nothing converts between them.
    check("LightRig fields match draw.py::_lighting_ubo packing",
          names == ["key_dir", "key_col", "fill_dir", "fill_col", "ambient"],
          str(names))
    check("LightRig is 5 vec4 == 80 bytes == what _lighting_ubo packs",
          len(fields) * 16 == 80, f"{len(fields)} fields")


def test_every_shader_compiles():
    """Through the real loader, so a malformed .json fails here and not later."""
    lib = rig().library
    for name in MINE:
        try:
            got = lib.get(name)
        except Exception as exc:                       # noqa: BLE001
            check(f"{name} loads", False, repr(exc))
            continue
        check(f"{name} loads and compiles", got is not None)


def test_json_attributes_match_the_geometry_builders():
    lib = rig().library
    coords = np.zeros((3, 3), np.float32)
    specs = {
        "sphere": mesh.sphere_spec(coords, np.ones((3, 4), np.float32),
                                   np.ones(3, np.float32)),
        "sphere_id": mesh.sphere_spec(coords, np.ones((3, 4), np.float32),
                                      np.ones(3, np.float32),
                                      shader_base="sphere_id"),
        "ribbon": mesh.ribbon_spec(coords, coords + 1.0,
                                   (1.0, 1.0, 1.0, 1.0),
                                   np.ones(3, np.float32),
                                   np.ones(3, np.float32)),
        "ribbon_id": mesh.ribbon_spec(coords, coords + 1.0,
                                      (1.0, 1.0, 1.0, 1.0),
                                      np.ones(3, np.float32),
                                      np.ones(3, np.float32),
                                      shader_base="ribbon_id"),
        "arrow": mesh.arrow_spec(coords, np.tile([0.0, 0.0, 1.0], (3, 1)),
                                 np.ones(3, np.float32),
                                 np.ones(3, np.float32),
                                 (1.0, 1.0, 1.0, 1.0)),
    }
    for name, spec in specs.items():
        shader = lib.get(name)
        declared = [a.name for a in shader.attributes]
        actual = list(spec.attrs)
        check(f"{name}: json attributes == MeshSpec.attrs, in order",
              declared == actual, f"{declared} vs {actual}")
        locs = [a.location for a in shader.attributes]
        check(f"{name}: locations are 0..n in that order",
              locs == list(range(len(locs))), str(locs))
        bad = []
        for a in shader.attributes:
            arr = np.asarray(spec.attrs[a.name])
            cols = 1 if arr.ndim == 1 else arr.shape[1]
            if cols != a.columns:
                bad.append(f"{a.name}: {a.format} over {cols} columns")
        check(f"{name}: declared formats match the array widths", not bad,
              "; ".join(bad))


def test_ids_declare_no_light_binding():
    lib = rig().library
    for name in ("sphere_id", "ribbon_id"):
        names = [b.name for b in lib.get(name).bindings]
        check(f"{name} binds camera only", names == ["camera"], str(names))
    for name in ("sphere", "ribbon", "arrow"):
        names = sorted(b.name for b in lib.get(name).bindings)
        check(f"{name} binds camera and lights", names == ["camera", "lights"],
              str(names))


def sphere(center, radius, color):
    return mesh.sphere_spec(np.array([center], np.float32),
                            np.array([color], np.float32),
                            np.array([radius], np.float32))


def ink(rgb, threshold=0.02):
    return rgb.max(axis=2) > threshold


def test_sphere_draws_a_disc_of_the_right_radius():
    r = 1.25
    rgb, depth = rig().draw([sphere((0.0, 0.0, 0.0), r, (1, 1, 1, 1))])
    covered = depth < 1.0

    n = int(covered.sum())
    check("sphere covers a non-empty region", n > 0, f"{n} px")
    measured = np.sqrt(n / np.pi)
    expected = r * PX_PER_UNIT
    check("disc radius matches the node radius",
          abs(measured - expected) < 1.0,
          f"{measured:.2f} px vs {expected:.2f} px expected")

    # Roundness, not area: an equal-area square quad would fail here.
    ys, xs = np.nonzero(covered)
    cx, cy = xs.mean(), ys.mean()
    rr = np.hypot(xs - cx, ys - cy)
    check("the covered region is a disc, not the quad",
          rr.max() - measured < 1.5,
          f"max radius {rr.max():.2f} px vs {measured:.2f} px")
    check("the disc is centered on the node",
          abs(cx - SIZE / 2) < 1.0 and abs(cy - SIZE / 2) < 1.0,
          f"center ({cx:.1f}, {cy:.1f})")


def test_sphere_is_shaded_not_flat():
    """Shading under a headlight is ambient + key * cos(theta), with
    cos(theta) the reconstructed normal's z, sqrt(1 - |p|^2)."""
    r = 2.0
    rgb, depth = rig().draw([sphere((0.0, 0.0, 0.0), r, (1, 1, 1, 1))])
    gx, gy = pixel_grid()
    p = np.hypot(gx, gy) / r
    inside = p < 0.98
    expect = 0.05 + 0.9 * np.sqrt(np.clip(1.0 - p * p, 0.0, None))
    got = rgb[:, :, 0]

    err = np.abs(got[inside] - expect[inside])
    check("shading matches ambient + key*cos(theta) per pixel",
          err.max() < 0.02, f"max error {err.max():.4f} over {inside.sum()} px")
    spread = got[inside].max() - got[inside].min()
    check("the disc is a gradient, not one constant color",
          spread > 0.5, f"range {spread:.3f}")


def test_sphere_rim_light_brightens_the_silhouette():
    """rim rides in key_col.w, packing inherited from the GLSL push-constant
    limit. This is the test that fails if the field moves."""
    r = 2.0
    rgb_no, _ = rig().draw([sphere((0, 0, 0), r, (0.2, 0.2, 0.2, 1))],
                           lights=rig().renderer.block_buffer(
                               "lights", headlight(rim=0.0)))
    rgb_yes, _ = rig().draw([sphere((0, 0, 0), r, (0.2, 0.2, 0.2, 1))],
                            lights=rig().renderer.block_buffer(
                                "lights", headlight(rim=0.8)))
    gx, gy = pixel_grid()
    p = np.hypot(gx, gy) / r
    edge = (p > 0.90) & (p < 0.97)
    middle = p < 0.3
    d_edge = (rgb_yes - rgb_no)[:, :, 0][edge].mean()
    d_mid = (rgb_yes - rgb_no)[:, :, 0][middle].mean()
    check("rim strength in key_col.w brightens the silhouette",
          d_edge > 0.2, f"+{d_edge:.3f} at the edge")
    check("rim leaves the center of the sphere alone",
          d_mid < 0.02, f"+{d_mid:.4f} in the middle")


def test_sphere_frag_depth_is_the_sphere_surface():
    r = 1.5
    rgb, depth = rig().draw([sphere((0.0, 0.0, 0.0), r, (1, 1, 1, 1))])
    gx, gy = pixel_grid()
    d2 = (gx * gx + gy * gy) / (r * r)
    inside = d2 < 0.95
    surface_z = vz(0.0) + r * np.sqrt(np.clip(1.0 - d2, 0.0, None))
    expect = ortho_depth(surface_z)
    flat = ortho_depth(vz(0.0))

    err = np.abs(depth[inside] - expect[inside])
    check("frag_depth equals the analytic sphere surface depth",
          err.max() < 2e-5,
          f"max error {err.max():.2e} over {int(inside.sum())} px")
    check("frag_depth is not constant over the disc",
          depth[inside].max() - depth[inside].min() > 1e-3,
          f"range {depth[inside].max() - depth[inside].min():.5f}")
    check("a flat billboard would fail this test",
          np.abs(flat - expect[inside]).max() > 1e-3,
          f"flat differs by up to "
          f"{np.abs(flat - expect[inside]).max():.5f}")


def _analytic_winner(surfaces):
    """(H, W) index of the nearest surface per pixel, -1 where none covers."""
    best = np.full(surfaces[0][0].shape, -1, np.int32)
    best_z = np.full(surfaces[0][0].shape, -np.inf)
    for i, (mask, z) in enumerate(surfaces):
        take = mask & (z > best_z)
        best[take] = i
        best_z[take] = z[take]
    return best, best_z


def test_two_spheres_occlude_volumetrically():
    r = 1.6
    a = sphere((-0.6, 0.0, 0.0), r, (1, 0, 0, 1))     # red,   nearer center
    b = sphere((0.6, 0.0, -0.9), r, (0, 1, 0, 1))     # green, farther center
    rgb, depth = rig().draw([a, b])

    gx, gy = pixel_grid()
    da = ((gx + 0.6) ** 2 + gy ** 2) / (r * r)
    db = ((gx - 0.6) ** 2 + gy ** 2) / (r * r)
    za = vz(0.0) + r * np.sqrt(np.clip(1.0 - da, 0.0, None))
    zb = vz(-0.9) + r * np.sqrt(np.clip(1.0 - db, 0.0, None))
    winner, _ = _analytic_winner([(da < 1.0, za), (db < 1.0, zb)])

    painted = np.where(rgb[:, :, 0] > rgb[:, :, 1], 0, 1)
    painted[depth >= 1.0] = -1

    # At the intersection curve the winner is a coin toss.
    seam = (da < 1.0) & (db < 1.0) & (np.abs(za - zb) < 0.02)
    judged = (winner >= 0) & ~seam
    wrong = int((painted[judged] != winner[judged]).sum())
    check("every pixel shows the nearer SURFACE, not the nearer center",
          wrong == 0, f"{wrong} wrong of {int(judged.sum())}")

    flat = _analytic_winner([(da < 1.0, np.full_like(za, vz(0.0))),
                             (db < 1.0, np.full_like(zb, vz(-0.9)))])[0]
    differ = int(((flat != winner) & judged).sum())
    check("a flat billboard would disagree on many pixels here",
          differ > 200, f"{differ} px would differ")


def test_occlusion_does_not_depend_on_draw_order():
    r = 1.6
    a = sphere((-0.6, 0.0, 0.0), r, (1, 0, 0, 1))
    b = sphere((0.6, 0.0, -0.9), r, (0, 1, 0, 1))
    first, _ = rig().draw([a, b])
    second, _ = rig().draw([b, a])
    diff = np.abs(first - second).max()
    check("swapping the draw order changes nothing", diff < 1e-6,
          f"max channel difference {diff:.2e}")


def test_sphere_under_perspective():
    """The one perspective camera, because ortho never divides by w."""
    r = 1.0
    view = cam.look_at((0.0, 0.0, EYE_Z), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    proj = cam.perspective(45.0, 1.0, NEAR, FAR)
    camera = cam.Camera(view, proj, NEAR, FAR)
    rgb, depth = rig().draw([sphere((0.0, 0.0, 0.0), r, (1, 1, 1, 1))],
                            camera=camera)
    covered = depth < 1.0
    n = int(covered.sum())
    check("perspective render is non-empty", n > 0, f"{n} px")

    cx, cy, _ = camera.project([[0.0, 0.0, 0.0]], SIZE, SIZE)[0]
    ys, xs = np.nonzero(covered)
    check("the disc is centered on the projected node center",
          abs(xs.mean() - cx) < 1.0 and abs(ys.mean() - cy) < 1.0,
          f"({xs.mean():.1f}, {ys.mean():.1f}) vs ({cx:.1f}, {cy:.1f})")

    pole = camera.project([[0.0, 0.0, r]], SIZE, SIZE)[0][2]
    got = depth[int(round(cy)), int(round(cx))]
    check("frag_depth at the center is the near pole's depth, w-divide included",
          abs(got - pole) < 1e-4, f"{got:.6f} vs {pole:.6f}")
    equator = camera.project([[0.0, 0.0, 0.0]], SIZE, SIZE)[0][2]
    check("and it differs from the flat billboard's depth",
          abs(equator - pole) > 1e-3, f"flat would be {equator:.6f}")


def ribbon(a, b, radius, color):
    return mesh.ribbon_spec(np.array([a], np.float32),
                            np.array([b], np.float32),
                            color,
                            np.array([radius], np.float32),
                            np.array([radius], np.float32))


def test_ribbon_draws_a_band_of_the_right_width():
    r = 0.5
    rgb, depth = rig().draw([ribbon((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                                    r, (1, 1, 1, 1))])
    covered = depth < 1.0
    ys, xs = np.nonzero(covered)
    check("ribbon covers a non-empty region", xs.size > 0, f"{xs.size} px")

    height = (ys.max() - ys.min() + 1) / PX_PER_UNIT
    check("the band is two radii across",
          abs(height - 2 * r) < 1.5 / PX_PER_UNIT,
          f"{height:.3f} world units vs {2 * r:.3f}")


def test_ribbon_square_cap_extends_half_a_radius_past_each_end():
    r = 0.5
    rgb, depth = rig().draw([ribbon((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                                    r, (1, 1, 1, 1))])
    covered = depth < 1.0
    xs = np.nonzero(covered.any(axis=0))[0]
    length = (xs.max() - xs.min() + 1) / PX_PER_UNIT

    uncapped = 4.0
    capped = uncapped + r     # half a radius at each end
    check("the quad is a full radius longer than its segment",
          abs(length - capped) < 1.5 / PX_PER_UNIT,
          f"{length:.3f} world units; uncapped would be {uncapped:.3f}")
    check("ink exists past the segment endpoint (the cap itself)",
          covered[SIZE // 2, int(round((2.0 + r * 0.4) * PX_PER_UNIT
                                       + SIZE / 2))],
          "sampled a fifth of a radius beyond the end")
    check("ink stops at the cap and does not run on",
          not covered[SIZE // 2, int(round((2.0 + r * 0.9) * PX_PER_UNIT
                                           + SIZE / 2))],
          "sampled nine tenths of a radius beyond the end")


def test_ribbon_is_shaded_like_a_cylinder():
    r = 1.5
    rgb, depth = rig().draw([ribbon((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                                    r, (1, 1, 1, 1))])
    gx, gy = pixel_grid()
    inner = (np.abs(gx) < 2.0) & (np.abs(gy) < r * 0.95)
    expect = 0.05 + 0.9 * np.sqrt(np.clip(1.0 - (gy / r) ** 2, 0.0, None))
    err = np.abs(rgb[:, :, 0][inner] - expect[inner])
    check("cylinder shading matches ambient + key*cos across the section",
          err.max() < 0.02, f"max error {err.max():.4f}")
    spread = rgb[:, :, 0][inner].max() - rgb[:, :, 0][inner].min()
    check("the band is a gradient, not one constant color",
          spread > 0.5, f"range {spread:.3f}")


def test_ribbon_frag_depth_is_the_tube_surface():
    r = 1.5
    rgb, depth = rig().draw([ribbon((-3.0, 0.0, 0.0), (3.0, 0.0, 0.0),
                                    r, (1, 1, 1, 1))])
    gx, gy = pixel_grid()
    inner = (np.abs(gx) < 2.0) & (np.abs(gy) < r * 0.95)
    surface_z = vz(0.0) + np.sqrt(np.clip(r * r - gy * gy, 0.0, None))
    expect = ortho_depth(surface_z)
    err = np.abs(depth[inner] - expect[inner])
    check("frag_depth equals the analytic cylinder surface depth",
          err.max() < 2e-5, f"max error {err.max():.2e}")
    flat = ortho_depth(vz(0.0))
    check("a flat billboard would fail this test",
          np.abs(flat - expect[inner]).max() > 1e-3,
          f"flat differs by up to {np.abs(flat - expect[inner]).max():.5f}")


def test_two_ribbons_occlude_volumetrically():
    r = 0.5
    za, zb = 0.0, -0.05
    h = ribbon((-2.0, 0.0, za), (2.0, 0.0, za), r, (1, 0, 0, 1))
    v = ribbon((0.0, -2.0, zb), (0.0, 2.0, zb), r, (0, 1, 0, 1))
    rgb, depth = rig().draw([h, v])

    gx, gy = pixel_grid()
    # The square cap extends each quad by r/2 past its endpoints.
    mask_h = (np.abs(gy) <= r) & (np.abs(gx) <= 2.0 + r * 0.5)
    mask_v = (np.abs(gx) <= r) & (np.abs(gy) <= 2.0 + r * 0.5)
    zh = vz(za) + np.sqrt(np.clip(r * r - gy * gy, 0.0, None))
    zv = vz(zb) + np.sqrt(np.clip(r * r - gx * gx, 0.0, None))
    winner, _ = _analytic_winner([(mask_h, zh), (mask_v, zv)])

    painted = np.where(rgb[:, :, 0] > rgb[:, :, 1], 0, 1)
    painted[depth >= 1.0] = -1
    seam = mask_h & mask_v & (np.abs(zh - zv) < 0.01)
    judged = (winner >= 0) & ~seam & (depth < 1.0)
    wrong = int((painted[judged] != winner[judged]).sum())
    check("crossing tubes resolve by surface, not by axis depth",
          wrong == 0, f"{wrong} wrong of {int(judged.sum())}")

    flat = _analytic_winner([(mask_h, np.full_like(zh, vz(za))),
                             (mask_v, np.full_like(zv, vz(zb)))])[0]
    differ = int(((flat != winner) & judged).sum())
    check("flat cards would disagree on many pixels here",
          differ > 100, f"{differ} px would differ")


def test_a_tube_and_a_sphere_share_one_depth_buffer():
    rs, rt = 1.0, 0.5
    zt = 0.3                    # the tube's axis is nearer than the sphere's center
    s = sphere((0.0, 0.0, 0.0), rs, (1, 0, 0, 1))
    t = ribbon((-3.0, 0.0, zt), (3.0, 0.0, zt), rt, (0, 1, 0, 1))
    rgb, depth = rig().draw([s, t])

    gx, gy = pixel_grid()
    d2 = (gx * gx + gy * gy) / (rs * rs)
    mask_s = d2 < 1.0
    mask_t = (np.abs(gy) <= rt) & (np.abs(gx) <= 3.0 + rt * 0.5)
    zs = vz(0.0) + rs * np.sqrt(np.clip(1.0 - d2, 0.0, None))
    ztu = vz(zt) + np.sqrt(np.clip(rt * rt - gy * gy, 0.0, None))
    winner, _ = _analytic_winner([(mask_s, zs), (mask_t, ztu)])

    painted = np.where(rgb[:, :, 0] > rgb[:, :, 1], 0, 1)
    painted[depth >= 1.0] = -1
    seam = mask_s & mask_t & (np.abs(zs - ztu) < 0.02)
    judged = (winner >= 0) & ~seam & (depth < 1.0)
    wrong = int((painted[judged] != winner[judged]).sum())
    check("sphere and tube resolve against each other by surface",
          wrong == 0, f"{wrong} wrong of {int(judged.sum())}")

    # The symptom: the node pokes through the tube crossing it.
    mid = rgb[SIZE // 2, SIZE // 2]
    check("the sphere's pole wins over the tube crossing it",
          mid[0] > mid[1], f"rgb {mid.round(3).tolist()}")
    fx = int(round(0.75 * rs * PX_PER_UNIT + SIZE / 2))
    flank = rgb[SIZE // 2, fx]
    check("the tube wins where the sphere has curved away",
          flank[1] > flank[0], f"rgb {flank.round(3).tolist()}")


def test_sphere_id_round_trips_the_encoded_index():
    ids = np.array([0, 1, 127, 128, 255, 256, 65535, 65536, 0x7FFFFF],
                   dtype=np.uint32)
    n = ids.size
    xs = np.linspace(-3.0, 3.0, n)
    coords = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
    spec = mesh.sphere_spec(coords, mesh.encode_ids_to_rgba(ids),
                            np.full(n, 0.25, np.float32),
                            shader_base="sphere_id")
    rgb, depth = rig().draw([spec])

    camera = ortho_camera()
    px = camera.project(coords, SIZE, SIZE)
    bad = []
    for i, (x, y, _) in enumerate(px):
        texel = np.round(rgb[int(round(y)), int(round(x))] * 255.0).astype(int)
        got = int(texel[0]) | (int(texel[1]) << 8) | (int(texel[2]) << 16)
        if got != int(ids[i]):
            bad.append(f"{int(ids[i])} -> {got}")
    check("every encoded id survives the round trip exactly", not bad,
          "; ".join(bad) if bad else f"{n} ids")


def test_id_passes_cover_exactly_what_the_beauty_passes_cover():
    cases = [
        ("sphere", sphere((-0.7, 0.2, 0.0), 1.3, (1, 1, 1, 1)),
         mesh.sphere_spec(np.array([[-0.7, 0.2, 0.0]], np.float32),
                          mesh.encode_ids_to_rgba(np.array([7])),
                          np.array([1.3], np.float32),
                          shader_base="sphere_id")),
        ("ribbon", ribbon((-2.0, -0.3, 0.0), (1.5, 1.1, 0.4), 0.6,
                          (1, 1, 1, 1)),
         mesh.ribbon_spec(np.array([[-2.0, -0.3, 0.0]], np.float32),
                          np.array([[1.5, 1.1, 0.4]], np.float32),
                          (0, 0, 0, 1),
                          np.array([0.6], np.float32),
                          np.array([0.6], np.float32),
                          shader_base="ribbon_id")),
    ]
    for name, beauty, ident in cases:
        _, d_beauty = rig().draw([beauty])
        _, d_id = rig().draw([ident])
        cov_b, cov_i = d_beauty < 1.0, d_id < 1.0
        check(f"{name}_id covers the same pixels as {name}",
              bool((cov_b == cov_i).all()),
              f"{int((cov_b != cov_i).sum())} px differ of {int(cov_b.sum())}")
        err = np.abs(d_beauty[cov_b] - d_id[cov_b]).max() if cov_b.any() else 0
        check(f"{name}_id writes the same depth as {name}", err < 1e-7,
              f"max difference {err:.2e}")


def test_arrow_draws_a_lit_cone():
    anchor = np.array([[1.5, 0.0, 0.0]], np.float32)
    direction = np.array([[1.0, 0.0, 0.0]], np.float32)
    spec = mesh.arrow_spec(anchor, direction,
                           np.array([0.25], np.float32),   # edge radius
                           np.array([0.0], np.float32),    # node radius
                           (1.0, 1.0, 1.0, 1.0))
    rgb, depth = rig().draw([spec])
    covered = depth < 1.0
    n = int(covered.sum())
    check("arrow covers a non-empty region", n > 0, f"{n} px")

    ys, xs = np.nonzero(covered)
    # mesh.arrow_spec: tip at the anchor, base 6*radius back along -direction.
    tip_x = (1.5 * PX_PER_UNIT + SIZE / 2)
    base_x = ((1.5 - 6 * 0.25) * PX_PER_UNIT + SIZE / 2)
    # Two tolerances: a base error would hide behind the tip's looser one.
    check("the cone's base cap is where mesh.arrow_spec puts it",
          abs(xs.min() - base_x) < 2.0,
          f"x min {xs.min()}, expected {base_x:.0f}")
    check("the cone reaches its tip", abs(xs.max() - tip_x) < 3.0,
          f"x max {xs.max()}, tip projects to {tip_x:.0f}")

    def column_height(x):
        col = np.nonzero(covered[:, int(round(x))])[0]
        return 0 if col.size == 0 else col.max() - col.min() + 1

    near_tip = column_height(tip_x - 3)
    near_base = column_height(base_x + 3)
    check("the cone tapers toward the tip", near_tip < near_base * 0.5,
          f"{near_tip} px vs {near_base} px")

    lit = rgb[:, :, 0][covered]
    check("the cone is shaded by its per-vertex normals, not flat",
          lit.max() - lit.min() > 0.1, f"range {lit.max() - lit.min():.3f}")


def test_arrow_is_two_sided():
    anchor = np.array([[1.0, 0.0, 0.0]], np.float32)
    forward = mesh.arrow_spec(anchor, np.array([[0.0, 0.0, 1.0]], np.float32),
                              np.array([0.3], np.float32),
                              np.array([0.0], np.float32),
                              (1.0, 1.0, 1.0, 1.0))
    away = mesh.arrow_spec(anchor, np.array([[0.0, 0.0, -1.0]], np.float32),
                           np.array([0.3], np.float32),
                           np.array([0.0], np.float32),
                           (1.0, 1.0, 1.0, 1.0))
    for name, spec in (("pointing at the camera", forward),
                       ("pointing away", away)):
        rgb, depth = rig().draw([spec])
        covered = depth < 1.0
        val = rgb[:, :, 0][covered]
        check(f"arrow {name} is drawn and lit",
              covered.any() and val.max() > 0.2,
              f"{int(covered.sum())} px, max {0.0 if not covered.any() else val.max():.3f}")


def test_impostor_depth_agrees_with_rasterizer_depth():
    """The arrow writes no frag_depth, the sphere overrides one. The GLSL's
    ``* 0.5 + 0.5`` remap, which OpenGL needed and WebGPU does not, pushes
    every impostor toward the far plane but still sorts them correctly among
    themselves, so only a test mixing the two paths catches it."""
    s = sphere((0.0, 0.0, 0.0), 1.0, (1, 0, 0, 1))
    a = mesh.arrow_spec(np.array([[1.5, 0.0, -2.0]], np.float32),
                        np.array([[1.0, 0.0, 0.0]], np.float32),
                        np.array([0.5], np.float32),
                        np.array([0.0], np.float32),
                        (0.0, 1.0, 0.0, 1.0))
    rgb, depth = rig().draw([s, a])
    # Measured, not guessed: a guessed region tests pixels only one reaches.
    _, only_sphere = rig().draw([s])
    _, only_arrow = rig().draw([a])
    contested = (only_sphere < 1.0) & (only_arrow < 1.0)
    check("the sphere and the arrowhead really do overlap on screen",
          int(contested.sum()) > 500, f"{int(contested.sum())} contested px")

    red = rgb[:, :, 0] > rgb[:, :, 1]
    check("a sphere in front of an arrowhead stays in front of it",
          bool(red[contested].all()),
          f"{int((~red[contested]).sum())} of {int(contested.sum())} lost to the arrow")


TESTS = [
    test_shared_uniform_blocks_are_identical,
    test_every_shader_compiles,
    test_json_attributes_match_the_geometry_builders,
    test_ids_declare_no_light_binding,
    test_sphere_draws_a_disc_of_the_right_radius,
    test_sphere_is_shaded_not_flat,
    test_sphere_rim_light_brightens_the_silhouette,
    test_sphere_frag_depth_is_the_sphere_surface,
    test_two_spheres_occlude_volumetrically,
    test_occlusion_does_not_depend_on_draw_order,
    test_sphere_under_perspective,
    test_ribbon_draws_a_band_of_the_right_width,
    test_ribbon_square_cap_extends_half_a_radius_past_each_end,
    test_ribbon_is_shaded_like_a_cylinder,
    test_ribbon_frag_depth_is_the_tube_surface,
    test_two_ribbons_occlude_volumetrically,
    test_a_tube_and_a_sphere_share_one_depth_buffer,
    test_sphere_id_round_trips_the_encoded_index,
    test_id_passes_cover_exactly_what_the_beauty_passes_cover,
    test_arrow_draws_a_lit_cone,
    test_arrow_is_two_sided,
    test_impostor_depth_agrees_with_rasterizer_depth,
]


def main(argv=None):
    import traceback

    patterns = list(argv or [])
    try:
        print(rig().ctx.describe())
    except Exception as exc:                            # noqa: BLE001
        # No adapter means exit 2, not a silent pass.
        print(f"NO GPU: {exc!r}")
        return 2

    ran = 0
    for fn in TESTS:
        if patterns and not any(p in fn.__name__ for p in patterns):
            continue
        ran += 1
        print(f"\n{fn.__name__}")
        before = len(FAILED)
        try:
            fn()
        except Exception:                               # noqa: BLE001
            FAILED.append(fn.__name__)
            print(''.join('       ' + ln
                          for ln in traceback.format_exc().splitlines(True)))
        if len(FAILED) == before:
            pass

    print(f"\n{ran} tests")
    if FAILED:
        print(f"{len(FAILED)} CHECKS FAILED")
        for name in FAILED:
            print(f"  - {name}")
        return 1
    if ran == 0:
        print("NO TESTS RAN")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
