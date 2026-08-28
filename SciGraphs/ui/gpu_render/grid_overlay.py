import numpy as np

import gpu
from gpu_extras.batch import batch_for_shader

_CUBE_EDGES = np.array([
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
], dtype=np.int64)

_CORNERS = np.array([[(i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(8)],
                    dtype=np.float32)

_SHADER = None


def _shader():
    """POLYLINE_FLAT_COLOR, not UNIFORM_COLOR with line_width_set: the state
    call is capped by the driver at 10 px on OpenGL, and this one widens in the
    shader, so a width means the same thing on both backends."""
    global _SHADER
    if _SHADER is None:
        _SHADER = gpu.shader.from_builtin('POLYLINE_FLAT_COLOR')
    return _SHADER


def cells_of(pos, res):
    """``(lo, span, occupied, counts)`` for *pos* on a *res*^3 grid.

    The same three lines the far field uses, so what gets drawn is the grid the
    forces are actually being read off, not a lookalike.
    """
    pos = np.asarray(pos, dtype=np.float32)
    if pos.size == 0:
        return None
    lo = pos.min(axis=0)
    span = np.maximum(np.ptp(pos, axis=0), 1e-9)
    cell = np.clip(((pos - lo) / span * res).astype(np.int32), 0, res - 1)
    flat = (cell[:, 0] * res + cell[:, 1]) * res + cell[:, 2]
    occupied, counts = np.unique(flat, return_counts=True)
    return lo, span, occupied, counts


def _unflatten(occupied, res):
    z = occupied % res
    y = (occupied // res) % res
    x = occupied // (res * res)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def tree_cells(pos, max_level=6):
    """``(lower, upper, counts)`` for the leaves of Graphviz's quadtree, or None.

    Leaves only, and that is what makes the picture readable: a quadtree's
    leaves partition the root exactly, while drawing the internal cells too
    hides each box under its own eight children. On a 30k clustered graph that
    is 14,334 internal cells over 29,655 leaves, and the result reads as one
    flat mesh.

    *max_level* is a depth limit, not a detail slider for its own sake. Graphviz
    splits until a leaf holds a single point, so at the default depth of 10 some
    18,000 of those leaves sit at sides of 0.03 and 0.016 against a root of 16,
    which is under a pixel. Stopping earlier is what leaves the nesting visible.
    """
    try:
        from scigraphs_utils import quadtree_cells
    except Exception:  # noqa: BLE001 - an old wheel has no quadtree
        return None
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    if pos.shape[0] < 2:
        return None
    cells = quadtree_cells(pos, max_level=int(max_level))
    leaf = np.asarray(cells.is_leaf, dtype=bool)
    if not leaf.any():
        return None
    return (np.asarray(cells.lower, dtype=np.float32)[leaf],
            np.asarray(cells.upper, dtype=np.float32)[leaf],
            np.asarray(cells.count, dtype=np.int64)[leaf])


def box_geometry(lower, upper, counts, base, hot=None):
    """``(coords, colors)`` from explicit cell bounds, 24 vertices per cell."""
    if lower is None or not len(lower):
        return None, None
    size = (upper - lower)
    corners = lower[:, None, :] + _CORNERS[None, :, :] * size[:, None, :]
    coords = corners[:, _CUBE_EDGES.ravel(), :].reshape(-1, 3)
    return coords, _shade(counts, coords.shape[0], base, hot)


def _shade(counts, nverts, base, hot):
    base = np.asarray(base, dtype=np.float32)
    if hot is None:
        return np.tile(base, (nverts, 1))
    t = np.log1p(counts.astype(np.float32))
    t /= max(float(t.max()), 1e-9)
    mix = base[None, :] + (np.asarray(hot, dtype=np.float32) - base)[None, :] \
        * t[:, None]
    return np.repeat(mix, 24, axis=0)


def geometry(lo, span, res, occupied, counts, base, hot=None):
    """``(coords, colors)`` for a LINES batch: 24 vertices per occupied cell.

    With *hot* the color runs from *base* to it by how full the cell is, on a
    log scale because occupancy spans four orders of magnitude on a settled
    graph and a linear ramp shows one red box and nothing else.
    """
    if occupied is None or not len(occupied):
        return None, None
    step = np.asarray(span, dtype=np.float32) / float(res)
    origin = np.asarray(lo, dtype=np.float32) + _unflatten(occupied, res) * step
    corners = origin[:, None, :] + _CORNERS[None, :, :] * step[None, None, :]
    coords = corners[:, _CUBE_EDGES.ravel(), :].reshape(-1, 3)

    return coords, _shade(counts, coords.shape[0], base, hot)


def _source_grid(source, res):
    """The far grid in use. The compute path keeps the one it uploaded, since
    that is refreshed on a cadence and a fresh one would not be what the forces
    used; the numpy path rebuilds its grid every step, so recomputing is exact.
    """
    stashed = getattr(source, "grid_snapshot", None)
    if stashed is not None:
        return stashed
    sim = getattr(source, "sim", None)
    pos = getattr(sim, "pos", None)
    if pos is None:
        pos = getattr(source, "positions", None)
    return cells_of(pos, res) if pos is not None else None


def build(scene, source):
    """``(coords, colors)`` for whatever the settings ask for, or ``(None, None)``."""
    from scigraphs_core.mesh.layouts import simulation as ref
    from . import sim_gpu

    mode = str(getattr(scene, "scigraphs_preview_grid_mode", 'FAR'))
    base = tuple(getattr(scene, "scigraphs_preview_grid_color",
                         (0.25, 0.7, 1.0, 0.35)))
    hot = tuple(getattr(scene, "scigraphs_preview_grid_hot", (1.0, 0.25, 0.1, 0.9))) \
        if bool(getattr(scene, "scigraphs_preview_grid_by_count", True)) else None

    chunks = []
    if mode == 'TREE':
        sim = getattr(source, "sim", None)
        pos = getattr(sim, "pos", None)
        if pos is None:
            pos = getattr(source, "positions", None)
        depth = int(getattr(scene, "scigraphs_preview_tree_depth", 6))
        cells = tree_cells(pos, max_level=depth) if pos is not None else None
        if cells is not None:
            chunks.append(box_geometry(*cells, base, hot))
    if mode in ('FAR', 'BOTH'):
        grid = _source_grid(source, ref.FAR_RES)
        if grid is not None:
            lo, span, occ, cnt = grid
            chunks.append(geometry(lo, span, ref.FAR_RES, occ, cnt, base, hot))
    if mode in ('NEAR', 'BOTH'):
        near = getattr(source, "near_snapshot", None)
        if near is not None:
            lo, span, occ, cnt = near
            chunks.append(geometry(lo, span, sim_gpu.BIN_RES, occ, cnt, base, hot))

    chunks = [c for c in chunks if c[0] is not None]
    if not chunks:
        return None, None
    if len(chunks) == 1:
        return chunks[0]
    return (np.concatenate([c[0] for c in chunks]),
            np.concatenate([c[1] for c in chunks]))


def draw(scene, viewport, source=None, obj=None):
    """Draw the overlay in whatever matrix state the caller has set up."""
    if not bool(getattr(scene, "scigraphs_preview_grid_show", False)):
        return False
    if source is None:
        from . import playback
        pb = playback.get(obj) if obj is not None else None
        source = getattr(pb, "source", None)
    if source is None:
        return False
    coords, colors = build(scene, source)
    if coords is None:
        return False

    shader = _shader()
    batch = batch_for_shader(shader, 'LINES',
                             {"pos": coords.tolist(), "color": colors.tolist()})
    prev_blend = gpu.state.blend_get()
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("viewportSize", (float(viewport[0]), float(viewport[1])))
    shader.uniform_float("lineWidth",
                         float(getattr(scene, "scigraphs_preview_grid_width", 1.5)))
    batch.draw(shader)
    gpu.state.blend_set(prev_blend)
    return True
