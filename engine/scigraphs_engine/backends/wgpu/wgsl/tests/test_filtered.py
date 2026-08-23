# The four filtered WGSL shaders, checked against a numpy replay of the same
# predicate rather than the fact that they compiled. Three things a compile or
# a screenshot misses: the wrong subset, stray ink from a culled quad that
# collapsed to a stretched triangle, and drift in the unfiltered base shader.

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
from scigraphs_engine.backends.wgpu import (camera as cam,        # noqa: E402
                                            device, renderer,
                                            shaders, target)

FAILED = []
CHECKS = 0

# Floor for run.py, set from a passing run.
MIN_CHECKS = 200


def check(name, ok, detail=""):
    global CHECKS
    CHECKS += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


# Orthographic; nothing here is about projection.

SIZE = 256
HALF = 4.0
EYE_Z = 10.0
NEAR, FAR = 1.0, 100.0
PX_PER_UNIT = SIZE / (2.0 * HALF)


def ortho_camera():
    view = cam.look_at((0.0, 0.0, EYE_Z), (0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0))
    return cam.Camera(view, cam.orthographic(HALF, HALF, NEAR, FAR), NEAR, FAR)


def headlight(rim=0.0, ambient=0.05, key=0.9):
    return np.array([
        0.0, 0.0, 1.0, 0.0,
        key, key, key, rim,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        ambient, ambient, ambient, 0.0,
    ], dtype=np.float32)


# 6x6 grid, spaced so "was this drawn" is about one node: 1.2 world units is
# 38 px against an 8 px radius. Values on the bounds are planted, not random.

GRID = 6
NODES = GRID * GRID
RADIUS = 0.25


def _grid_coords():
    g = (np.arange(GRID) - (GRID - 1) / 2.0) * 1.2
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(NODES)],
                    axis=1).astype(np.float32)


COORDS = _grid_coords()

# Every grid neighbor, 60, none crossing: each edge owns its pixels.
def _grid_edges():
    idx = np.arange(NODES).reshape(GRID, GRID)
    out = []
    for r in range(GRID):
        for c in range(GRID):
            if c + 1 < GRID:
                out.append((idx[r, c], idx[r, c + 1]))
            if r + 1 < GRID:
                out.append((idx[r, c], idx[r + 1, c]))
    return np.array(out, dtype=np.int32)


EDGES = _grid_edges()
NUM_EDGES = EDGES.shape[0]


def _channels():
    """(NODES + NUM_EDGES, 4) float32, as filter_gpu.channel_rows packs it."""
    rng = np.random.default_rng(20240731)
    out = rng.random((NODES + NUM_EDGES, 4)).astype(np.float32)
    # These sit exactly on the bounds the stacks use, to exercise inclusivity.
    out[0, 0] = np.float32(0.5)
    out[1, 0] = np.float32(0.25)
    out[2, 0] = np.float32(0.75)
    out[NODES, 0] = np.float32(0.4)
    out[NODES + 1, 0] = np.float32(0.6)
    return out


CHAN = _channels()


def node_frows():
    """filter_gpu.node_frows: own row twice, -1 for "no edge row"."""
    rows = np.arange(NODES, dtype=np.float32)
    out = np.empty((NODES, 3), np.float32)
    out[:, 0] = rows
    out[:, 1] = rows
    out[:, 2] = -1.0
    return out


def edge_frows(swap=False):
    """filter_gpu.edge_frows: both endpoints, and the edge's own row."""
    a, b = (1, 0) if swap else (0, 1)
    out = np.empty((NUM_EDGES, 3), np.float32)
    out[:, 0] = EDGES[:, a]
    out[:, 1] = EDGES[:, b]
    out[:, 2] = np.arange(NUM_EDGES, dtype=np.float32) + NODES
    return out


# A transcription of scig_keep, not a second implementation.

def replay(frow, block):
    """(n,) bool: which vertices scig_keep accepts, on the CPU."""
    u = np.asarray(block, np.float32).reshape(8, 4)
    lo, hi, inv, act, elo, ehi, einv, eact = u

    def span(values, lo, hi, inv, act):
        ins = (values >= lo) & (values <= hi)
        keep = np.where(inv > 0.5, ~ins, ins)
        keep |= act <= 0.5
        return keep.all(axis=1)

    out = np.ones(frow.shape[0], bool)
    if (act != 0.0).any():
        out &= span(CHAN[frow[:, 0].astype(np.int64)], lo, hi, inv, act)
        # scig_keep skips this fetch when the rows are equal.
        out &= span(CHAN[frow[:, 1].astype(np.int64)], lo, hi, inv, act)
    has_edge = frow[:, 2] >= 0.0
    if (eact != 0.0).any():
        rows = np.where(has_edge, frow[:, 2], 0.0).astype(np.int64)
        out &= ~has_edge | span(CHAN[rows], elo, ehi, einv, eact)
    return out


def stack(node=(), edge=()):
    """A FilterStack's 8 vec4, as filter_gpu.uniform_rows packs them."""
    data = np.zeros((8, 4), np.float32)
    for base, clauses in ((0, node), (4, edge)):
        for slot, lo, hi, inv in clauses:
            data[base + 0, slot] = np.float32(lo)
            data[base + 1, slot] = np.float32(hi)
            data[base + 2, slot] = 1.0 if inv else 0.0
            data[base + 3, slot] = 1.0
    return data.ravel()


NODE_STACKS = [
    ("nothing active", stack()),
    ("a floor on slot 0", stack(node=[(0, 0.5, 1.0, False)])),
    ("a ceiling on slot 0", stack(node=[(0, 0.0, 0.25, False)])),
    ("a band on slot 1", stack(node=[(1, 0.3, 0.7, False)])),
    ("an inverted band", stack(node=[(0, 0.2, 0.6, True)])),
    ("two clauses in different slots",
     stack(node=[(0, 0.25, 1.0, False), (2, 0.0, 0.8, False)])),
    ("three clauses",
     stack(node=[(0, 0.1, 1.0, False), (1, 0.0, 0.9, False),
                 (3, 0.2, 1.0, False)])),
    ("four clauses, the block full",
     stack(node=[(0, 0.1, 1.0, False), (1, 0.05, 0.95, False),
                 (2, 0.0, 0.9, False), (3, 0.1, 0.95, True)])),
    # The values planted in _channels() sit exactly on these ends.
    ("a range whose ends are exact channel values",
     stack(node=[(0, 0.25, 0.75, False)])),
]

# Minus the full-block stack: it keeps 0 of 60 edges and stops discriminating.
EDGE_STACKS = [c for c in NODE_STACKS
               if c[0] != "four clauses, the block full"] + [
    ("four clauses, the block full",
     stack(node=[(0, 0.05, 1.0, False), (1, 0.0, 0.95, False),
                 (2, 0.05, 1.0, False), (3, 0.0, 0.95, False)])),
    ("an edge clause only", stack(edge=[(0, 0.4, 1.0, False)])),
    ("an inverted edge clause", stack(edge=[(1, 0.3, 0.7, True)])),
    ("a node clause and an edge clause",
     stack(node=[(0, 0.3, 1.0, False)], edge=[(0, 0.0, 0.6, False)])),
    ("an edge range whose ends are exact channel values",
     stack(edge=[(0, 0.4, 0.6, False)])),
]



class Rig:
    """Context, target, renderer, and the three buffers a filtered draw needs."""

    def __init__(self, size=SIZE):
        self.ctx = device.acquire()
        self.target = target.Target(self.ctx, size, size)
        self.size = size
        self.library = shaders.ShaderLibrary(self.ctx)
        from scigraphs_engine.mesh import DrawState
        self.renderer = renderer.Renderer(
            self.ctx, self.target, library=self.library,
            default_state=DrawState(point_size=9.0))
        self.lights = self.renderer.block_buffer("lights", headlight())
        self.fchan = self.ctx.device.create_buffer_with_data(
            label="test-fchan", data=np.ascontiguousarray(CHAN, np.float32),
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.fstack = self.ctx.device.create_buffer(
            label="test-fstack", size=128,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.blocks = {"lights": self.lights, "fstack": self.fstack,
                       "fchan": self.fchan}

    def set_stack(self, block):
        """Write the 128-byte block. This is all a slider ever does."""
        self.ctx.queue.write_buffer(
            self.fstack, 0,
            np.ascontiguousarray(block, np.float32).tobytes())

    def draw(self, specs, block=None, tint=(1.0, 1.0, 1.0, 1.0)):
        if block is not None:
            self.set_stack(block)
        meshes = [self.renderer.upload(s) for s in specs if s is not None]
        if any(m is None for m in meshes):
            raise AssertionError(
                "a shader was missing for a spec; without this the test would "
                "pass over an empty render pass")
        try:
            self.renderer.render(meshes, ortho_camera(), blocks=self.blocks,
                                 tint=tint)
            return self.target.read()[:, :, :3].astype(np.float64)
        finally:
            for m in meshes:
                m.destroy()

    def draw_uploaded(self, meshes, block=None, tint=(1.0, 1.0, 1.0, 1.0)):
        if block is not None:
            self.set_stack(block)
        self.renderer.render(meshes, ortho_camera(), blocks=self.blocks,
                             tint=tint)
        return self.target.read()[:, :, :3].astype(np.float64)


_RIG = None


def rig():
    global _RIG
    if _RIG is None:
        _RIG = Rig()
    return _RIG


FILTERED = ("sphere_f", "ribbon_f", "round_point_f", "line_f")


def _src(name):
    return open(os.path.join(WGSL_DIR, name + ".wgsl"), encoding="utf-8").read()


BLOCK_RE = re.compile(
    r"^// >>> SCIG FILTER BLOCK <<<\n.*?^// >>> END SCIG FILTER BLOCK <<<\n",
    re.S | re.M)


def test_filter_block_is_identical_in_every_variant():
    blocks = {}
    for name in FILTERED:
        m = BLOCK_RE.search(_src(name))
        check(f"{name} carries the marked filter block", m is not None)
        if m:
            blocks[name] = m.group(0)
    check("the filter block is byte-identical in all four",
          len(set(blocks.values())) == 1,
          f"{len(set(blocks.values()))} distinct across {sorted(blocks)}")


def _fields(source, name):
    m = re.search(r"^struct\s+%s\s*\{(.*?)^\}" % name, source, re.S | re.M)
    if m is None:
        return None
    body = re.sub(r"//[^\n]*", "", m.group(1))
    return tuple(f.strip() for f in body.split(",") if f.strip())


def test_filter_stack_matches_filter_gpu_uniform_rows():
    names = [f.split(":")[0].strip() for f in _fields(_src("sphere_f"),
                                                      "FilterStack") or ()]
    check("FilterStack fields match filter_gpu.uniform_rows' packing",
          names == ["lo", "hi", "inv", "act", "elo", "ehi", "einv", "eact"],
          str(names))
    check("FilterStack is 8 vec4 == 128 bytes == the guaranteed block",
          len(names) * 16 == 128, f"{len(names)} fields")
    declared = [b for b in rig().library.get("sphere_f").bindings
                if b.name == "fstack"][0]
    check("and the .json binds it at that size", declared.size == 128,
          str(declared.size))


def test_shared_camera_block_survived_the_new_files():
    cameras, rigs = {}, {}
    for entry in sorted(os.listdir(WGSL_DIR)):
        if not entry.endswith(".wgsl"):
            continue
        src = open(os.path.join(WGSL_DIR, entry), encoding="utf-8").read()
        got = _fields(src, "Camera")
        if got is not None:
            cameras[entry] = got
        got = _fields(src, "LightRig")
        if got is not None:
            rigs[entry] = got
    check("Camera block identical in all wgsl/ files",
          len(set(cameras.values())) == 1,
          f"{len(set(cameras.values()))} distinct across {len(cameras)} files")
    check("LightRig block identical wherever it appears",
          len(set(rigs.values())) == 1, f"in {sorted(rigs)}")


def _strip_comments(text):
    out = [re.sub(r"\s+$", "", re.sub(r"//.*$", "", ln))
           for ln in text.splitlines()]
    return "\n".join(ln for ln in out if ln)


def test_a_filtered_variant_is_its_base_plus_the_guard():
    """Strips the marked block, the frow attribute, scig_reject and the guard
    line. If anything else has to go to make this pass, the variant drifted."""
    for name, base in (("sphere_f", "sphere"), ("ribbon_f", "ribbon"),
                       ("round_point_f", "round_point")):
        src = BLOCK_RE.sub("", _src(name))
        # Struct member on the impostors, bare vs_main param on round_point.
        src = re.sub(r",\n\s*@location\(\d+\) frow\s*:\s*vec3<f32>\)", ")", src)
        src = re.sub(r"^\s*@location\(\d+\) frow\s*:.*\n", "", src, flags=re.M)
        src = re.sub(r"^fn scig_reject\(\) -> VertexOut \{.*?^\}\n", "", src,
                     flags=re.S | re.M)
        src = re.sub(r"^\s*if \(!scig_keep\(.*?\)\) \{ return scig_reject\(\); \}\n",
                     "", src, flags=re.M)
        stripped = _strip_comments(src)
        want = _strip_comments(_src(base))
        # Headers legitimately differ; compare from the first line of code.
        i, j = stripped.index("struct Camera"), want.index("struct Camera")
        check(f"{name} is {base} plus the guard, and nothing else",
              stripped[i:] == want[j:],
              "" if stripped[i:] == want[j:] else
              _first_difference(stripped[i:], want[j:]))


def _first_difference(a, b):
    la, lb = a.splitlines(), b.splitlines()
    for i in range(max(len(la), len(lb))):
        x = la[i] if i < len(la) else "<end>"
        y = lb[i] if i < len(lb) else "<end>"
        if x != y:
            return f"line {i}: {x!r} vs base {y!r}"
    return ""


def test_every_filtered_shader_compiles():
    """Through the real loader, so a malformed .json fails here and not later."""
    lib = rig().library
    for name in FILTERED + ("arrow_flat",):
        try:
            got = lib.get(name)
        except Exception as exc:                        # noqa: BLE001
            check(f"{name} loads", False, repr(exc))
            continue
        check(f"{name} loads and compiles", got is not None)


def test_filtered_shaders_declare_the_two_filter_bindings():
    lib = rig().library
    for name in FILTERED:
        by = {b.name: b for b in lib.get(name).bindings}
        check(f"{name} declares fstack at 0.3 as a uniform",
              "fstack" in by and by["fstack"].group == 0
              and by["fstack"].binding == 3 and by["fstack"].type == "uniform",
              str(by.get("fstack")))
        check(f"{name} declares fchan at 0.4 as read-only storage",
              "fchan" in by and by["fchan"].group == 0
              and by["fchan"].binding == 4
              and by["fchan"].type == "read-only-storage",
              str(by.get("fchan")))
        check(f"{name} declares fstack and fchan to the VERTEX stage only",
              by["fstack"].visibility == wgpu.ShaderStage.VERTEX
              and by["fchan"].visibility == wgpu.ShaderStage.VERTEX,
              "the predicate runs per vertex; a fragment-visible binding would "
              "claim otherwise")


def test_json_attributes_match_the_geometry_builders():
    lib = rig().library
    nf, ef = node_frows(), edge_frows()
    specs = {
        "sphere_f": sphere_spec_f(np.arange(NODES), stack_all=True),
        "ribbon_f": ribbon_spec_f(np.arange(NUM_EDGES)),
        "round_point_f": mesh.point_specs(
            COORDS, np.ones((NODES, 4), np.float32), frows=nf)[0][0],
        "line_f": mesh.filtered_line_spec(COORDS, EDGES, ef, edge_frows(True)),
    }
    want_frow_slot = {"sphere_f": 4, "ribbon_f": 6, "round_point_f": 2,
                      "line_f": 1}
    for name, spec in specs.items():
        shader = lib.get(name)
        declared = [a.name for a in shader.attributes]
        check(f"{name}: json attributes == MeshSpec.attrs, in order",
              declared == list(spec.attrs), f"{declared} vs {list(spec.attrs)}")
        slot = [a.location for a in shader.attributes
                if a.name == "frow"][0]
        check(f"{name}: frow is at the GLSL's slot", slot == want_frow_slot[name],
              f"{slot} vs {want_frow_slot[name]}")
        bad = []
        for a in shader.attributes:
            arr = np.asarray(spec.attrs[a.name])
            cols = 1 if arr.ndim == 1 else arr.shape[1]
            if cols != a.columns:
                bad.append(f"{a.name}: {a.format} over {cols} columns")
        check(f"{name}: declared formats match the array widths", not bad,
              "; ".join(bad))
        check(f"{name}: the spec resolves to the {name} variant",
              shaders.variant_name(spec.shader) == name,
              shaders.variant_name(spec.shader))


def sphere_spec_f(sel, stack_all=False):
    sel = np.asarray(sel)
    return mesh.sphere_spec(COORDS[sel], np.ones((sel.size, 4), np.float32),
                            np.full(sel.size, RADIUS, np.float32),
                            frows=node_frows()[sel])


def ribbon_spec_f(sel):
    sel = np.asarray(sel)
    a, b = COORDS[EDGES[sel, 0]], COORDS[EDGES[sel, 1]]
    return mesh.ribbon_spec(a, b, (1.0, 1.0, 1.0, 1.0),
                            np.full(sel.size, 0.12, np.float32),
                            np.full(sel.size, 0.12, np.float32),
                            frows=edge_frows()[sel])


def point_spec_f(sel):
    sel = np.asarray(sel)
    return mesh.point_specs(COORDS[sel], np.ones((sel.size, 4), np.float32),
                            frows=node_frows()[sel])[0][0]


def line_spec_f(sel):
    sel = np.asarray(sel)
    return mesh.filtered_line_spec(COORDS, EDGES[sel], edge_frows()[sel],
                                   edge_frows(True)[sel])


BUILDERS = {
    "sphere_f": (sphere_spec_f, node_frows, NODES, "node"),
    "round_point_f": (point_spec_f, node_frows, NODES, "node"),
    "ribbon_f": (ribbon_spec_f, edge_frows, NUM_EDGES, "edge"),
    "line_f": (line_spec_f, edge_frows, NUM_EDGES, "edge"),
}


def ink(rgb, threshold=0.02):
    return rgb.max(axis=2) > threshold


def _drawn_set(img, frow, kind):
    """(n,) bool: which primitives left ink, read from the image."""
    lit = ink(img)
    camera = ortho_camera()
    if kind == "node":
        pts = COORDS
    else:
        pts = 0.5 * (COORDS[EDGES[:, 0]] + COORDS[EDGES[:, 1]])
    px = camera.project(pts, SIZE, SIZE)
    out = np.zeros(pts.shape[0], bool)
    r = 3
    for i, (x, y, _) in enumerate(px):
        x, y = int(round(x)), int(round(y))
        out[i] = lit[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].any()
    return out


def _predicate_case(name, label, block):
    builder, frows, n, kind = BUILDERS[name]
    frow = frows()
    want = replay(frow, block)
    kept = int(want.sum())

    # An all-keep stack hides a deleted guard, an all-drop one an inversion.
    if label != "nothing active":
        check(f"{name} / {label}: the case discriminates "
              f"(a proper non-empty subset)",
              0 < kept < n, f"{kept} of {n} kept")

    full = builder(np.arange(n))
    img = rig().draw([full], block=block)
    got = _drawn_set(img, frow, kind)
    wrong = np.nonzero(got != want)[0]
    check(f"{name} / {label}: the drawn set is the numpy set, {kind} for {kind}",
          wrong.size == 0,
          f"{kept}/{n} kept; {wrong.size} wrong"
          + (f" at {wrong[:8].tolist()}" if wrong.size else ""))

    # Extra pixels mean a rejected primitive left ink, which per-node misses.
    if kept:
        subset = builder(np.nonzero(want)[0])
        ref = rig().draw([subset], block=stack())
    else:
        ref = rig().draw([], block=stack())
    diff = np.abs(img - ref).max()
    check(f"{name} / {label}: the frame is bit-identical to the kept subset "
          f"drawn alone", diff == 0.0, f"max channel difference {diff:.3e}")


def test_sphere_f_draws_the_numpy_set():
    for label, block in NODE_STACKS:
        _predicate_case("sphere_f", label, block)


def test_round_point_f_draws_the_numpy_set():
    for label, block in NODE_STACKS:
        _predicate_case("round_point_f", label, block)


def test_ribbon_f_draws_the_numpy_set():
    for label, block in EDGE_STACKS:
        _predicate_case("ribbon_f", label, block)


def test_line_f_draws_the_numpy_set():
    for label, block in EDGE_STACKS:
        _predicate_case("line_f", label, block)


def test_a_node_clause_takes_the_edges_that_would_dangle():
    block = stack(node=[(0, 0.45, 1.0, False)])
    node_keep = replay(node_frows(), block)
    edge_keep = replay(edge_frows(), block)
    both = node_keep[EDGES[:, 0]] & node_keep[EDGES[:, 1]]
    check("the numpy replay agrees an edge needs both endpoints",
          bool((edge_keep == both).all()),
          f"{int((edge_keep != both).sum())} disagree")
    # Some edges have one surviving endpoint, so reading only .x draws them.
    half = int((node_keep[EDGES[:, 0]] != node_keep[EDGES[:, 1]]).sum())
    check("this stack really does half-cut some edges", half > 3,
          f"{half} edges have exactly one surviving endpoint")

    img = rig().draw([ribbon_spec_f(np.arange(NUM_EDGES))], block=block)
    got = _drawn_set(img, edge_frows(), "edge")
    check("ribbon_f drops an edge whose FAR endpoint was filtered out",
          bool((got == both).all()),
          f"{int((got != both).sum())} wrong of {NUM_EDGES}")

    only_x = node_keep[EDGES[:, 0]]
    check("and reading only .x would have been visibly different here",
          int((only_x != both).sum()) > 3,
          f"{int((only_x != both).sum())} edges would differ")


def test_an_inactive_slot_ignores_its_bounds():
    live = stack(node=[(0, 0.5, 1.0, False)])
    dead = live.copy().reshape(8, 4)
    dead[0, 2], dead[1, 2] = 0.9, 0.95      # an absurd range in slot 2...
    dead[3, 2] = 0.0                        # ...with its active flag clear
    dead = dead.ravel()
    a = rig().draw([sphere_spec_f(np.arange(NODES))], block=live)
    b = rig().draw([sphere_spec_f(np.arange(NODES))], block=dead)
    check("an inactive slot's bounds change nothing",
          np.abs(a - b).max() == 0.0, f"max difference {np.abs(a - b).max():.3e}")
    armed = dead.copy().reshape(8, 4)
    armed[3, 2] = 1.0
    c = rig().draw([sphere_spec_f(np.arange(NODES))], block=armed.ravel())
    check("and the same bounds with the flag set do change it",
          np.abs(a - c).max() > 0.0, "otherwise this check proves nothing")


def test_the_range_is_inclusive_at_both_ends():
    block = stack(node=[(0, 0.25, 0.75, False)])
    want = replay(node_frows(), block)
    check("the node sitting exactly on lo is kept", bool(want[1]),
          f"channel {CHAN[1, 0]!r} against lo 0.25")
    check("the node sitting exactly on hi is kept", bool(want[2]),
          f"channel {CHAN[2, 0]!r} against hi 0.75")
    img = rig().draw([sphere_spec_f(np.arange(NODES))], block=block)
    got = _drawn_set(img, node_frows(), "node")
    check("and the shader agrees about both of them",
          bool(got[1]) and bool(got[2]),
          f"drawn: lo {bool(got[1])}, hi {bool(got[2])}")


def test_an_empty_stack_draws_what_the_unfiltered_shader_draws():
    cases = [
        ("sphere", sphere_spec_f(np.arange(NODES)),
         mesh.sphere_spec(COORDS, np.ones((NODES, 4), np.float32),
                          np.full(NODES, RADIUS, np.float32))),
        ("ribbon", ribbon_spec_f(np.arange(NUM_EDGES)),
         mesh.ribbon_spec(COORDS[EDGES[:, 0]], COORDS[EDGES[:, 1]],
                          (1.0, 1.0, 1.0, 1.0),
                          np.full(NUM_EDGES, 0.12, np.float32),
                          np.full(NUM_EDGES, 0.12, np.float32))),
        ("round_point", point_spec_f(np.arange(NODES)),
         mesh.point_specs(COORDS, np.ones((NODES, 4), np.float32))[0][0]),
    ]
    for name, filtered, plain in cases:
        a = rig().draw([filtered], block=stack())
        b = rig().draw([plain], block=stack())
        diff = np.abs(a - b).max()
        check(f"{name}_f with an empty stack == {name}, bit for bit",
              diff == 0.0, f"max channel difference {diff:.3e}")
        check(f"and {name} actually drew something", ink(b).sum() > 100,
              f"{int(ink(b).sum())} px")


def test_moving_a_threshold_touches_only_the_128_byte_block():
    """On the CPU a threshold rebuilds every batch, 1.2 s at 2M nodes. Here the
    mesh uploads once and the sweep writes nothing but the 128-byte block."""
    spec = sphere_spec_f(np.arange(NODES))
    mesh_obj = rig().renderer.upload(spec)
    check("the batch uploaded once", mesh_obj is not None)
    try:
        counts = []
        for floor in (0.0, 0.25, 0.5, 0.75, 1.01):
            block = stack(node=[(0, floor, 1.0, False)])
            img = rig().draw_uploaded([mesh_obj], block=block)
            got = _drawn_set(img, node_frows(), "node")
            want = replay(node_frows(), block)
            check(f"floor {floor}: the drawn set matches numpy over the SAME "
                  f"buffers", bool((got == want).all()),
                  f"{int(got.sum())} drawn, {int(want.sum())} expected")
            counts.append(int(got.sum()))
        check("the sweep is monotonically decreasing and actually sweeps",
              counts == sorted(counts, reverse=True) and counts[0] > counts[-1],
              str(counts))
        check("and it ends at nothing drawn", counts[-1] == 0, str(counts[-1]))
    finally:
        mesh_obj.destroy()


def test_a_rejected_vertex_leaves_no_ink_at_the_screen_edges():
    none_kept = stack(node=[(0, 2.0, 3.0, False)])
    img = rig().draw([sphere_spec_f(np.arange(NODES))], block=none_kept)
    check("everything rejected leaves an entirely empty frame",
          int(ink(img).sum()) == 0, f"{int(ink(img).sum())} lit px")

    # Pick the stack that keeps exactly the largest slot-0 value's node.
    top = int(np.argmax(CHAN[:NODES, 0]))
    lo = np.float32(CHAN[top, 0])
    one = stack(node=[(0, lo, 1.0, False)])
    want = replay(node_frows(), one)
    check("the one-node stack really keeps exactly one", int(want.sum()) == 1,
          f"{int(want.sum())} kept")
    img = rig().draw([sphere_spec_f(np.arange(NODES))], block=one)
    ref = rig().draw([sphere_spec_f([top])], block=stack())
    check("the 35 rejected quads leave no pixel anywhere",
          np.abs(img - ref).max() == 0.0,
          f"max difference {np.abs(img - ref).max():.3e}")
    border = np.zeros(img.shape[:2], bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    check("and specifically none on the frame border",
          not ink(img)[border].any(),
          f"{int(ink(img)[border].sum())} lit border px")


def test_every_vertex_of_a_primitive_gets_the_same_verdict():
    """A rejected primitive is degenerate wherever its vertices land, so
    mutating SCIG_CULL changes nothing the rest of the suite sees. Both shapes
    mesh.py builds: quads repeat one row across four corners, the line builder
    gives each end the same pair swapped."""
    nf, ef = node_frows(), edge_frows()
    quads = {
        "sphere_spec": mesh.sphere_spec(
            COORDS, np.ones((NODES, 4), np.float32),
            np.full(NODES, RADIUS, np.float32), frows=nf).attrs["frow"],
        "ribbon_spec": mesh.ribbon_spec(
            COORDS[EDGES[:, 0]], COORDS[EDGES[:, 1]], (1, 1, 1, 1),
            np.full(NUM_EDGES, 0.12, np.float32),
            np.full(NUM_EDGES, 0.12, np.float32), frows=ef).attrs["frow"],
    }
    for name, frow in quads.items():
        per_quad = frow.reshape(-1, 4, 3)
        same = bool((per_quad == per_quad[:, :1, :]).all())
        check(f"{name}: all four corners of a quad carry the same frow", same,
              f"{per_quad.shape[0]} quads")

    pts = mesh.point_specs(COORDS, np.ones((NODES, 4), np.float32),
                           frows=nf)[0][0]
    check("point_specs: one frow row per node, and the shader steps it per "
          "instance", pts.attrs["frow"].shape == (NODES, 3),
          str(pts.attrs["frow"].shape))

    line = mesh.filtered_line_spec(COORDS, EDGES, ef, edge_frows(True))
    rows = line.attrs["frow"]
    a, b = rows[0::2], rows[1::2]
    check("filtered_line_spec: the two ends really do carry different rows",
          not bool((a == b).all()),
          "otherwise the symmetry below would be vacuous")
    disagree = 0
    for _, block in EDGE_STACKS:
        disagree += int((replay(a, block) != replay(b, block)).sum())
    check("and the predicate gives them the same verdict under every stack",
          disagree == 0,
          f"{disagree} vertex pairs disagreed over {len(EDGE_STACKS)} stacks")


def test_line_f_takes_its_color_from_the_per_draw_tint():
    spec = line_spec_f(np.arange(NUM_EDGES))
    a = rig().draw([spec], block=stack(), tint=(1.0, 0.0, 0.0, 1.0))
    b = rig().draw([spec], block=stack(), tint=(0.0, 0.0, 1.0, 1.0))
    lit_a, lit_b = ink(a), ink(b)
    check("line_f drew the same pixels under both tints",
          bool((lit_a == lit_b).all()) and lit_a.sum() > 100,
          f"{int(lit_a.sum())} vs {int(lit_b.sum())} px")
    check("and it is red under a red tint",
          a[lit_a][:, 0].min() > 0.9 and a[lit_a][:, 2].max() < 0.1,
          f"mean rgb {a[lit_a].mean(axis=0).round(3).tolist()}")
    check("and blue under a blue tint",
          b[lit_b][:, 2].min() > 0.9 and b[lit_b][:, 0].max() < 0.1,
          f"mean rgb {b[lit_b].mean(axis=0).round(3).tolist()}")


TESTS = [
    test_filter_block_is_identical_in_every_variant,
    test_filter_stack_matches_filter_gpu_uniform_rows,
    test_shared_camera_block_survived_the_new_files,
    test_a_filtered_variant_is_its_base_plus_the_guard,
    test_every_filtered_shader_compiles,
    test_filtered_shaders_declare_the_two_filter_bindings,
    test_json_attributes_match_the_geometry_builders,
    test_sphere_f_draws_the_numpy_set,
    test_round_point_f_draws_the_numpy_set,
    test_ribbon_f_draws_the_numpy_set,
    test_line_f_draws_the_numpy_set,
    test_a_node_clause_takes_the_edges_that_would_dangle,
    test_an_inactive_slot_ignores_its_bounds,
    test_the_range_is_inclusive_at_both_ends,
    test_an_empty_stack_draws_what_the_unfiltered_shader_draws,
    test_moving_a_threshold_touches_only_the_128_byte_block,
    test_a_rejected_vertex_leaves_no_ink_at_the_screen_edges,
    test_every_vertex_of_a_primitive_gets_the_same_verdict,
    test_line_f_takes_its_color_from_the_per_draw_tint,
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
        # Going green by running fewer checks is a recorded failure mode.
        print(f"ONLY {CHECKS} CHECKS RAN, {MIN_CHECKS} DECLARED")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
