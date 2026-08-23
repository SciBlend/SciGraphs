# arrow_flat, checked in pixels.
#
# Both ideas in this shader fail quietly. Per-vertex depth only shows on an
# edge pointing steeply at the camera, so these tests read the depth buffer
# back in a scene where it would have varied a lot. View-space and world-space
# directions coincide under the default camera, so those checks run rotated.

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WGSL_DIR = os.path.dirname(HERE)
ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(WGSL_DIR))))
if ENGINE_ROOT not in sys.path:
    sys.path.insert(0, ENGINE_ROOT)

import dataclasses                                                # noqa: E402

import wgpu                                                       # noqa: E402

from scigraphs_engine import mesh                                 # noqa: E402
from scigraphs_engine.backends.wgpu import (camera as cam,        # noqa: E402
                                            device, renderer,
                                            shaders, target)

FAILED = []
CHECKS = 0
MIN_CHECKS = 30


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


SIZE = 256
HALF = 4.0
EYE_Z = 10.0
NEAR, FAR = 1.0, 100.0
PX_PER_UNIT = SIZE / (2.0 * HALF)


def ortho_camera():
    view = cam.look_at((0.0, 0.0, EYE_Z), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    return cam.Camera(view, cam.orthographic(HALF, HALF, NEAR, FAR), NEAR, FAR)


def tilted_camera():
    """Off-axis, so view-space xy and world-space xy stop agreeing."""
    view = cam.look_at((6.0, 5.0, 7.0), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    return cam.Camera(view, cam.orthographic(HALF, HALF, NEAR, FAR), NEAR, FAR)


def ortho_depth(view_z):
    return (view_z + NEAR) / (NEAR - FAR)


class Rig:
    """Target with a readable depth texture; see test_impostors.py::Rig."""

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

    def draw(self, specs, camera=None):
        meshes = [self.renderer.upload(s) for s in specs if s is not None]
        if any(m is None for m in meshes):
            raise AssertionError("a shader was missing for a spec")
        try:
            self.renderer.render(meshes, camera or ortho_camera())
            return (self.target.read()[:, :, :3].astype(np.float64),
                    self.read_depth())
        finally:
            for m in meshes:
                m.destroy()

    def read_depth(self):
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


def arrow(a, b, edge_radius=0.25, node_radius=0.0, color=(1, 1, 1, 1),
          scale=1.0):
    return mesh.arrow_flat_spec(np.array([a], np.float32),
                                np.array([b], np.float32),
                                np.array([edge_radius], np.float32),
                                np.array([node_radius], np.float32),
                                color, arrow_scale=scale)


def covered(depth):
    return depth < 1.0


def test_arrow_flat_loads_and_declares_what_the_builder_builds():
    lib = rig().library
    shader = lib.get("arrow_flat")
    check("arrow_flat loads and compiles", shader is not None)
    spec = arrow((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    declared = [a.name for a in shader.attributes]
    check("json attributes == MeshSpec.attrs, in order",
          declared == list(spec.attrs), f"{declared} vs {list(spec.attrs)}")
    locs = [a.location for a in shader.attributes]
    check("locations are 0..n in the GLSL's order",
          locs == list(range(len(locs))), str(locs))
    bad = []
    for a in shader.attributes:
        arr = np.asarray(spec.attrs[a.name])
        cols = 1 if arr.ndim == 1 else arr.shape[1]
        if cols != a.columns:
            bad.append(f"{a.name}: {a.format} over {cols} columns")
    check("declared formats match the array widths", not bad, "; ".join(bad))
    check("the spec resolves to arrow_flat",
          shaders.variant_name(spec.shader) == "arrow_flat",
          shaders.variant_name(spec.shader))


def test_arrow_flat_binds_no_light_rig():
    names = [b.name for b in rig().library.get("arrow_flat").bindings]
    check("arrow_flat binds camera only", names == ["camera"], str(names))


def test_the_head_is_where_arrow_flat_spec_says_it_is():
    r = 0.25
    rgb, depth = rig().draw([arrow((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                                   edge_radius=r)])
    cov = covered(depth)
    n = int(cov.sum())
    check("the head covers a non-empty region", n > 0, f"{n} px")

    ys, xs = np.nonzero(cov)
    tip_x = 2.0 * PX_PER_UNIT + SIZE / 2
    base_x = (2.0 - r * mesh.FLAT_LEN) * PX_PER_UNIT + SIZE / 2
    check("the tip is at the target endpoint", abs(xs.max() - tip_x) < 3.0,
          f"x max {xs.max()}, tip projects to {tip_x:.0f}")
    check("the base is FLAT_LEN * radius back from it",
          abs(xs.min() - base_x) < 2.0,
          f"x min {xs.min()}, base at {base_x:.0f}")

    # At the base column, so a wide tip cannot mask a narrow base.
    col = np.nonzero(cov[:, int(round(base_x)) + 1])[0]
    width = (col.max() - col.min() + 1) / PX_PER_UNIT
    want = 2.0 * r * mesh.FLAT_HALF_WIDTH
    check("the base is 2 * FLAT_HALF_WIDTH * radius across",
          abs(width - want) < 2.0 / PX_PER_UNIT,
          f"{width:.3f} world units vs {want:.3f}")

    def column_height(x):
        c = np.nonzero(cov[:, int(round(x))])[0]
        return 0 if c.size == 0 else c.max() - c.min() + 1

    check("it tapers to the tip",
          column_height(tip_x - 3) < column_height(base_x + 3) * 0.4,
          f"{column_height(tip_x - 3)} px vs {column_height(base_x + 3)} px")


def test_the_head_backs_off_the_target_node_radius():
    nr = 0.75
    _, d0 = rig().draw([arrow((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                              node_radius=0.0)])
    _, d1 = rig().draw([arrow((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0),
                              node_radius=nr)])
    x0 = np.nonzero(covered(d0).any(axis=0))[0]
    x1 = np.nonzero(covered(d1).any(axis=0))[0]
    want = nr * PX_PER_UNIT
    check("a node radius moves the tip back by exactly that much",
          abs((x0.max() - x1.max()) - want) < 2.0,
          f"moved {x0.max() - x1.max()} px, expected {want:.0f}")
    check("and moves the base back by the same amount",
          abs((x0.min() - x1.min()) - want) < 2.0,
          f"moved {x0.min() - x1.min()} px, expected {want:.0f}")


def test_the_head_points_along_the_edge_in_every_direction():
    camera = ortho_camera()
    for k in range(8):
        th = k * np.pi / 4.0
        b = (2.5 * np.cos(th), 2.5 * np.sin(th), 0.0)
        rgb, depth = rig().draw([arrow((0.0, 0.0, 0.0), b, edge_radius=0.22)],
                                camera=camera)
        cov = covered(depth)
        ys, xs = np.nonzero(cov)
        if xs.size == 0:
            check(f"direction {k}: the head is drawn", False, "0 px")
            continue
        # The covered pixel furthest along the edge's screen direction.
        (ax, ay, _), (bx, by, _) = camera.project([(0.0, 0.0, 0.0), b],
                                                  SIZE, SIZE)
        d = np.array([bx - ax, by - ay])
        d = d / np.hypot(*d)
        proj = xs * d[0] + ys * d[1]
        tip = np.array([xs[proj.argmax()], ys[proj.argmax()]])
        check(f"direction {k}: the tip lands on the projected endpoint",
              np.hypot(tip[0] - bx, tip[1] - by) < 4.0,
              f"tip at {tip.tolist()}, endpoint projects to "
              f"({bx:.0f}, {by:.0f})")


def test_the_triangle_winding_does_not_depend_on_the_direction():
    """Why the .json says `cull: none`. Mutating ``cull`` on the loaded Shader
    builds distinct pipelines, since it is part of the pipeline cache key."""
    shader = rig().library.get("arrow_flat")
    original = shader.cull
    counts = {}
    try:
        for mode in ("back", "front"):
            shader.cull = mode
            drawn = []
            for k in range(8):
                th = k * np.pi / 4.0
                b = (2.5 * np.cos(th), 2.5 * np.sin(th), 0.0)
                _, depth = rig().draw([arrow((0.0, 0.0, 0.0), b,
                                             edge_radius=0.22)])
                drawn.append(int(covered(depth).sum()) > 0)
            counts[mode] = sum(drawn)
    finally:
        shader.cull = original
    check("one cull sense draws all eight directions and the other draws none",
          sorted(counts.values()) == [0, 8], str(counts))
    check("and the declaration keeps culling off", original == "none", original)


def test_all_three_vertices_share_the_tip_depth():
    a = (-1.0, -1.0, -3.0)
    b = (1.5, 1.0, 2.5)
    rgb, depth = rig().draw([arrow(a, b, edge_radius=0.35)])
    cov = covered(depth)
    check("the head is drawn", int(cov.sum()) > 100, f"{int(cov.sum())} px")

    got = depth[cov]
    spread = float(got.max() - got.min())
    check("frag depth is constant across the whole head",
          spread < 1e-6, f"range {spread:.3e} over {int(cov.sum())} px")

    want = ortho_depth(b[2] - EYE_Z)
    check("and it is the TIP's depth, not the tail's or the middle's",
          abs(float(got.mean()) - want) < 1e-5,
          f"{got.mean():.6f} vs {want:.6f}")

    # The discriminator: a per-vertex-depth head would span a lot of z here.
    r = 0.35
    length = r * mesh.FLAT_LEN
    d = np.array(b) - np.array(a)
    d = d / np.linalg.norm(d)
    base_z = b[2] - length * d[2]
    would = abs(ortho_depth(base_z - EYE_Z) - want)
    check("a per-vertex-depth head would have spanned a lot here",
          would > 1e-2, f"it would have spanned {would:.4f}")


def test_a_head_seen_almost_end_on_is_still_full_size():
    r = 0.3
    _, side = rig().draw([arrow((0.0, 0.0, 0.0), (2.5, 0.0, 0.0),
                                edge_radius=r)])
    _, endon = rig().draw([arrow((0.0, 0.0, -4.0), (0.02, 0.0, 4.0),
                                 edge_radius=r)])
    a, b = int(covered(side).sum()), int(covered(endon).sum())
    check("a nearly end-on head covers the same area as a sideways one",
          a > 0 and abs(a - b) <= max(4, a * 0.05), f"{a} px vs {b} px")


def test_an_edge_that_projects_to_a_point_is_culled():
    _, depth = rig().draw([arrow((0.0, 0.0, -2.0), (0.0, 0.0, 2.0),
                                 edge_radius=0.3)])
    check("an edge pointing exactly at the camera draws nothing at all",
          int(covered(depth).sum()) == 0, f"{int(covered(depth).sum())} px")
    # Discriminator: nudge it off-axis and it comes back.
    _, depth = rig().draw([arrow((0.0, 0.0, -2.0), (0.001, 0.0, 2.0),
                                 edge_radius=0.3)])
    check("and the smallest nudge off-axis brings it back",
          int(covered(depth).sum()) > 100, f"{int(covered(depth).sum())} px")


def test_the_direction_is_the_projected_one_not_the_world_one():
    camera = tilted_camera()
    a = np.array([-2.0, 0.5, -1.5], np.float32)
    b = np.array([1.5, -1.0, 2.0], np.float32)
    rgb, depth = rig().draw([arrow(tuple(a), tuple(b), edge_radius=0.35)],
                            camera=camera)
    cov = covered(depth)
    check("the tilted-camera head is drawn", int(cov.sum()) > 100,
          f"{int(cov.sum())} px")

    ys, xs = np.nonzero(cov)
    (ax, ay, _), (bx, by, _) = camera.project([a, b], SIZE, SIZE)

    def angle_to(dx, dy):
        n = np.hypot(dx, dy)
        u = np.array([dx / n, dy / n])
        proj = xs * u[0] + ys * u[1]
        tip = np.array([xs[proj.argmax()], ys[proj.argmax()]], float)
        centroid = np.array([xs.mean(), ys.mean()])
        axis = tip - centroid
        axis = axis / np.hypot(*axis)
        return float(np.degrees(np.arccos(np.clip(axis @ u, -1, 1))))

    projected = angle_to(bx - ax, by - ay)
    # World direction with z dropped: a pre-view-transform head's answer.
    w = b - a
    world = angle_to(float(w[0]), float(-w[1]))
    check("the head's axis agrees with the PROJECTED edge direction",
          projected < 6.0, f"{projected:.1f} degrees off")
    check("and the world-space direction is a visibly different answer here",
          world > 20.0,
          f"the world direction is {world:.1f} degrees off, so this "
          f"discriminates")


def test_the_head_is_opaque_even_when_its_color_is_not():
    from scigraphs_engine.mesh import DrawState
    spec = arrow((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0), edge_radius=0.35,
                 color=(1.0, 0.0, 0.0, 0.35))
    spec = dataclasses.replace(spec, state=DrawState(blend="ALPHA"))
    rgb, depth = rig().draw([spec])
    cov = covered(depth)
    check("the head is drawn under alpha blending", int(cov.sum()) > 100,
          f"{int(cov.sum())} px")
    red = rgb[:, :, 0][cov]
    check("it comes out fully opaque, not at 35% over black",
          red.min() > 0.99, f"darkest covered pixel is {red.min():.3f}; "
                            f"a forwarded alpha would give ~0.35")


TESTS = [
    test_arrow_flat_loads_and_declares_what_the_builder_builds,
    test_arrow_flat_binds_no_light_rig,
    test_the_head_is_where_arrow_flat_spec_says_it_is,
    test_the_head_backs_off_the_target_node_radius,
    test_the_head_points_along_the_edge_in_every_direction,
    test_the_triangle_winding_does_not_depend_on_the_direction,
    test_all_three_vertices_share_the_tip_depth,
    test_a_head_seen_almost_end_on_is_still_full_size,
    test_an_edge_that_projects_to_a_point_is_culled,
    test_the_direction_is_the_projected_one_not_the_world_one,
    test_the_head_is_opaque_even_when_its_color_is_not,
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
