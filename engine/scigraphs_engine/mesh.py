# A MeshSpec is what a builder produces before ``batch_for_shader``. ``shader``
# is a name, not a compiled object, so nothing here needs a graphics context,
# and ``state`` rides along because WebGPU bakes it into a pipeline up front.

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple

import numpy as np

# Spelled as Blender spells them, so the first adapter is a pass-through.
POINTS = 'POINTS'
LINES = 'LINES'
TRIS = 'TRIS'


@dataclass(frozen=True)
class ShaderRef:
    base: str
    filtered: bool = False
    animated: bool = False


@dataclass(frozen=True)
class DrawState:
    """Fixed-function state a draw needs; ``None`` means leave it as set."""

    point_size: Optional[float] = None
    line_width: Optional[float] = None
    blend: Optional[str] = None          # 'NONE' | 'ALPHA' | 'ADDITIVE'
    depth_test: Optional[bool] = None
    depth_write: Optional[bool] = None


@dataclass(frozen=True)
class MeshGroup:
    """One draw call over a MeshSpec's shared attributes. Line width and point
    size cannot be vertex attributes, so they become ``weight`` classes over one
    buffer; a buffer per class took 2.8M segments from 123 ms to 1.44 s."""

    indices: Optional[np.ndarray] = None
    start: Optional[int] = None
    count: Optional[int] = None
    weight: Optional[float] = None
    state: DrawState = field(default_factory=DrawState)

    @property
    def is_range(self):
        return self.indices is None and self.start is not None


@dataclass(frozen=True)
class MeshSpec:
    topology: str
    shader: ShaderRef
    attrs: Mapping[str, np.ndarray]
    indices: Optional[np.ndarray] = None
    state: DrawState = field(default_factory=DrawState)
    # A backend that sees groups draws those and ignores the index set.
    groups: Optional[Tuple[MeshGroup, ...]] = None

    @property
    def vertex_count(self):
        for arr in self.attrs.values():
            return int(arr.shape[0])
        return 0

    def validate(self):
        """A reason string, or None: a short attribute uploads and draws garbage."""
        if not self.attrs:
            return "no attributes"
        n = None
        for name, arr in self.attrs.items():
            arr = np.asarray(arr)
            if n is None:
                n = arr.shape[0]
            elif arr.shape[0] != n:
                return f"{name} has {arr.shape[0]} rows, expected {n}"
        checks = []
        if self.indices is not None:
            checks.append(("indices", self.indices))
        for i, g in enumerate(self.groups or ()):
            if g.is_range:
                if g.start < 0 or g.start + g.count > n:
                    return (f"group {i}: range [{g.start}, "
                            f"{g.start + g.count}) outside {n} vertices")
            elif g.indices is not None:
                checks.append((f"group {i}", g.indices))
            else:
                return f"group {i}: neither indices nor a range"
        for label, idx in checks:
            idx = np.asarray(idx)
            if idx.size and int(idx.max()) >= n:
                return f"{label}: index {int(idx.max())} out of range for {n}"
        return None


# Corner order must match the fragment shader's normal reconstruction.
QUAD_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                        dtype=np.float32)
QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def _quad_faces(n):
    return (np.arange(n, dtype=np.int32)[:, None, None] * 4
            + QUAD_TRIS[None]).reshape(-1, 3)


# Bit 23 of the 24-bit id marks an edge segment: one ID pass and depth test.
EDGE_ID_FLAG = 0x800000


def encode_ids_to_rgba(ids):
    ids = np.asarray(ids, dtype=np.uint32)
    return np.stack([
        (ids & 0xFF).astype(np.float32) / 255.0,
        ((ids >> 8) & 0xFF).astype(np.float32) / 255.0,
        ((ids >> 16) & 0xFF).astype(np.float32) / 255.0,
        np.ones(ids.shape, dtype=np.float32),
    ], axis=1)


def sphere_spec(coords, colors, radii, frows=None, animated=False,
                shader_base="sphere"):
    """Sphere impostors, or None: 4 vertices per node, expanded in view space."""
    coords = np.asarray(coords, dtype=np.float32)
    m = coords.shape[0]
    if m == 0:
        return None

    attrs = {
        "pos": np.repeat(coords, 4, axis=0),
        "corner": np.tile(QUAD_CORNERS, (m, 1)),
        "radius": np.repeat(np.asarray(radii, dtype=np.float32), 4),
        "color": np.repeat(np.asarray(colors, dtype=np.float32), 4, axis=0),
    }
    if frows is not None:
        attrs["frow"] = np.repeat(frows, 4, axis=0)

    return MeshSpec(
        topology=TRIS,
        shader=ShaderRef(shader_base, filtered=frows is not None,
                         animated=animated),
        attrs=attrs,
        indices=_quad_faces(m),
    )


def ribbon_spec(seg_a, seg_b, edge_color, radius_a, radius_b,
                col_a=None, col_b=None, frows=None, animated=False,
                shader_base="ribbon"):
    """Cylinder impostors, one quad per segment, or None. Vertex order is
    (A-, B-, B+, A+) and ``endsel`` says which end, so a taper can thin it."""
    if seg_a is None:
        return None
    seg_a = np.asarray(seg_a, dtype=np.float32)
    e = seg_a.shape[0]
    if e == 0:
        return None
    seg_b = np.asarray(seg_b, dtype=np.float32)

    radius = np.stack(
        [radius_a, radius_b, radius_b, radius_a], axis=1
    ).reshape(-1).astype(np.float32)

    # The shader lights a tube opaque, so only the color's hue matters.
    if col_a is None:
        color = np.tile(np.asarray(edge_color, dtype=np.float32), (e * 4, 1))
    else:
        color = np.stack([col_a, col_b, col_b, col_a],
                         axis=1).reshape(-1, 4).astype(np.float32)

    attrs = {
        "pos_a": np.repeat(seg_a, 4, axis=0),
        "pos_b": np.repeat(seg_b, 4, axis=0),
        "side": np.tile(np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32), e),
        "endsel": np.tile(np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32), e),
        "radius": radius,
        "color": color,
    }
    if frows is not None:
        attrs["frow"] = np.repeat(frows, 4, axis=0)

    return MeshSpec(
        topology=TRIS,
        shader=ShaderRef(shader_base, filtered=frows is not None,
                         animated=animated),
        attrs=attrs,
        indices=_quad_faces(e),
    )


def line_spec(coords, edges):
    """Indexed lines over a shared position buffer, or None."""
    if edges is None:
        return None
    edges = np.asarray(edges)
    if edges.shape[0] == 0:
        return None
    return MeshSpec(
        topology=LINES,
        shader=ShaderRef("uniform_color"),
        attrs={"pos": np.asarray(coords, dtype=np.float32)},
        indices=edges,
    )


def bucketed_line_spec(coords, edges, weights, buckets=12):
    """Indexed lines in ``buckets`` weight classes; empty ones cost a state change."""
    if edges is None or weights is None:
        return None
    edges = np.asarray(edges)
    if edges.shape[0] == 0:
        return None

    idx = np.clip((np.asarray(weights) * buckets).astype(int), 0, buckets - 1)
    groups = []
    for b in range(buckets):
        sel = idx == b
        if not sel.any():
            continue
        groups.append(MeshGroup(
            indices=np.ascontiguousarray(edges[sel], dtype=np.int32),
            weight=(b + 0.5) / buckets))
    if not groups:
        return None

    return MeshSpec(
        topology=LINES,
        shader=ShaderRef("polyline_uniform_color"),
        attrs={"pos": np.ascontiguousarray(coords, dtype=np.float32)},
        groups=tuple(groups),
    )


def segment_line_spec(seg_a, seg_b, col_a=None, col_b=None):
    """Flat lines over explicit segment endpoints, or None. Unindexed."""
    if seg_a is None:
        return None
    seg_a = np.asarray(seg_a, dtype=np.float32)
    n = seg_a.shape[0]
    if n == 0:
        return None
    seg_b = np.asarray(seg_b, dtype=np.float32)

    pos = np.empty((n * 2, 3), dtype=np.float32)
    pos[0::2] = seg_a
    pos[1::2] = seg_b

    if col_a is None:
        return MeshSpec(topology=LINES, shader=ShaderRef("uniform_color"),
                        attrs={"pos": pos})

    col = np.empty((n * 2, 4), dtype=np.float32)
    col[0::2] = col_a
    col[1::2] = col_b
    return MeshSpec(topology=LINES, shader=ShaderRef("smooth_color"),
                    attrs={"pos": pos, "color": col})


def filtered_line_spec(coords, edges, frow_a, frow_b):
    """Lines with a filter row per vertex; unindexed, a shared vertex names no edge."""
    if edges is None:
        return None
    edges = np.asarray(edges)
    e = edges.shape[0]
    if e == 0:
        return None
    coords = np.asarray(coords, dtype=np.float32)

    pos = np.empty((e * 2, 3), dtype=np.float32)
    pos[0::2] = coords[edges[:, 0]]
    pos[1::2] = coords[edges[:, 1]]

    frow = np.empty((e * 2, 3), dtype=np.float32)
    frow[0::2] = frow_a
    frow[1::2] = frow_b

    return MeshSpec(topology=LINES, shader=ShaderRef("line", filtered=True),
                    attrs={"pos": pos, "frow": frow})


def block_line_specs(edge_groups, coords=None, edges=None,
                     seg_a=None, seg_b=None, col_a=None, col_b=None):
    """One spec per block; ``None`` holds an empty block's slot in the list."""
    out = []
    for eids in edge_groups:
        eids = np.asarray(eids)
        if eids.size == 0:
            out.append(None)
            continue
        if seg_a is not None:
            out.append(segment_line_spec(
                seg_a[eids], seg_b[eids],
                None if col_a is None else col_a[eids],
                None if col_b is None else col_b[eids]))
            continue
        if edges is not None:
            out.append(line_spec(coords, edges[eids]))
            continue
        out.append(None)
    return out


def bucketed_segment_spec(seg_a, seg_b, widths, col_a=None, col_b=None,
                          buckets=12):
    """Width classes over a tessellated segment soup, as contiguous runs."""
    if seg_a is None or widths is None:
        return None
    seg_a = np.asarray(seg_a, dtype=np.float32)
    n = seg_a.shape[0]
    if n == 0:
        return None
    w = np.asarray(widths, dtype=np.float32)
    lo, hi = float(w.min()), float(w.max())
    if hi <= lo:
        return None
    seg_b = np.asarray(seg_b, dtype=np.float32)

    idx = np.clip(((w - lo) / (hi - lo) * buckets).astype(int), 0, buckets - 1)
    order = np.argsort(idx, kind="stable")
    counts = np.bincount(idx, minlength=buckets)
    starts = np.concatenate(([0], np.cumsum(counts)))

    verts = np.empty((n * 2, 3), dtype=np.float32)
    verts[0::2] = seg_a[order]
    verts[1::2] = seg_b[order]
    attrs = {"pos": verts}

    if col_a is not None:
        col = np.empty((n * 2, 4), dtype=np.float32)
        col[0::2] = np.asarray(col_a, dtype=np.float32)[order]
        col[1::2] = np.asarray(col_b, dtype=np.float32)[order]
        attrs["color"] = col

    groups = tuple(
        MeshGroup(start=int(starts[b]) * 2, count=int(counts[b]) * 2,
                  weight=(b + 0.5) / buckets)
        for b in range(buckets) if counts[b]
    )
    if not groups:
        return None

    return MeshSpec(
        topology=LINES,
        shader=ShaderRef("polyline_smooth_color" if col_a is not None
                         else "polyline_uniform_color"),
        attrs=attrs,
        groups=groups,
    )


def point_specs(coords, colors, norm=None, size_by_attr=False, frows=None,
                buckets=12, shader=None):
    """Size classes as ``[(MeshSpec, weight), ...]``; points have no index buffer."""
    coords = np.asarray(coords, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.float32)
    ref = shader if shader is not None else ShaderRef("round_point",
                                                      filtered=frows is not None)

    def spec(sel):
        attrs = {"pos": coords[sel] if sel is not None else coords,
                 "color": colors[sel] if sel is not None else colors}
        if frows is not None:
            attrs["frow"] = frows[sel] if sel is not None else frows
        return MeshSpec(topology=POINTS, shader=ref, attrs=attrs)

    if not (size_by_attr and norm is not None and coords.shape[0]):
        return [(spec(None), None)]

    idx = np.clip((np.asarray(norm) * buckets).astype(int), 0, buckets - 1)
    out = []
    for b in range(buckets):
        sel = idx == b
        if sel.any():
            out.append((spec(sel), (b + 0.5) / buckets))
    return out


ARROW_SIDES = 8


def arrow_spec(anchor, direction, edge_radii, node_radii, color,
               arrow_scale=1.0, sides=ARROW_SIDES):
    """Lit 3D cones at directed edge ends, or None: 18 vertices and 16 triangles
    each at 8 sides, tip pulled back so the arrow sits on the sphere surface."""
    if anchor is None:
        return None
    anchor = np.asarray(anchor, dtype=np.float32)
    e = anchor.shape[0]
    if e == 0:
        return None
    s = sides
    d = np.asarray(direction, dtype=np.float32)
    edge_radii = np.asarray(edge_radii, dtype=np.float32)
    length = edge_radii * 6.0 * arrow_scale
    base_r = edge_radii * 2.4 * arrow_scale

    tip = np.asarray(anchor, dtype=np.float32) \
        - d * np.asarray(node_radii, dtype=np.float32)[:, None]
    base_c = tip - d * length[:, None]

    # Per-arrow orthonormal basis perpendicular to the axis.
    up = np.where(np.abs(d[:, 2:3]) < 0.9,
                  np.array([0.0, 0.0, 1.0], dtype=np.float32),
                  np.array([1.0, 0.0, 0.0], dtype=np.float32))
    u = np.cross(d, up)
    u /= np.maximum(np.linalg.norm(u, axis=1), 1e-12)[:, None]
    v = np.cross(d, u)

    ang = (2.0 * np.pi / s) * np.arange(s, dtype=np.float32)
    radial = (np.cos(ang)[None, :, None] * u[:, None, :]
              + np.sin(ang)[None, :, None] * v[:, None, :])   # (E, S, 3)
    ring = base_c[:, None, :] + radial * base_r[:, None, None]
    # Smooth cone side normal: radial weighted by height, axial by base radius.
    n_side = radial * length[:, None, None] + d[:, None, :] * base_r[:, None, None]
    n_side /= np.maximum(np.linalg.norm(n_side, axis=2), 1e-12)[:, :, None]

    nv = 2 * s + 2
    verts = np.empty((e, nv, 3), dtype=np.float32)
    norms = np.empty_like(verts)
    verts[:, 0] = tip
    norms[:, 0] = d
    verts[:, 1:s + 1] = ring
    norms[:, 1:s + 1] = n_side
    verts[:, s + 1] = base_c
    norms[:, s + 1] = -d
    verts[:, s + 2:] = ring
    norms[:, s + 2:] = -d[:, None, :]

    tri_side = [[0, 1 + k, 1 + (k + 1) % s] for k in range(s)]
    tri_base = [[s + 1, s + 2 + (k + 1) % s, s + 2 + k] for k in range(s)]
    tris = np.array(tri_side + tri_base, dtype=np.int32)
    faces = (np.arange(e, dtype=np.int32)[:, None, None] * nv
             + tris[None]).reshape(-1, 3)
    colors = np.tile(np.asarray(color, dtype=np.float32), (e * nv, 1))

    return MeshSpec(
        topology=TRIS,
        shader=ShaderRef("arrow"),
        attrs={"pos": verts.reshape(-1, 3), "nrm": norms.reshape(-1, 3),
               "color": colors},
        indices=faces,
    )


# Flat head proportions in radii; the cone uses the same numbers.
FLAT_LEN = 6.0
FLAT_HALF_WIDTH = 2.4


def arrow_flat_spec(anchor_a, anchor_b, edge_radii, node_radii, color,
                    arrow_scale=1.0):
    """Flat billboard arrowheads, one triangle each, or None. Two endpoints and a
    corner index travel instead of a direction, which the shader resolves in view
    space."""
    if anchor_a is None:
        return None
    anchor_a = np.asarray(anchor_a, dtype=np.float32)
    e = anchor_a.shape[0]
    if e == 0:
        return None

    r = np.asarray(edge_radii, dtype=np.float32) * float(arrow_scale)
    cone = np.empty((e, 3), dtype=np.float32)
    cone[:, 0] = r * FLAT_LEN
    cone[:, 1] = r * FLAT_HALF_WIDTH
    cone[:, 2] = np.asarray(node_radii, dtype=np.float32)

    return MeshSpec(
        topology=TRIS,
        shader=ShaderRef("arrow_flat"),
        attrs={
            "pos_a": np.repeat(anchor_a, 3, axis=0),
            "pos_b": np.repeat(np.asarray(anchor_b, np.float32), 3, axis=0),
            "corner": np.tile(np.array([0.0, 1.0, 2.0], np.float32), e),
            "cone": np.repeat(cone, 3, axis=0),
            "color": np.tile(np.asarray(color, np.float32), (e * 3, 1)),
        },
        indices=np.arange(e * 3, dtype=np.int32).reshape(-1, 3),
    )
