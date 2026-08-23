# Every mode ends in one control polygon per edge, so the textures, batch,
# shading and draw call all live here. A mode exposes
# `check(params, edges, ctx) -> (ok, reason)` and
# `produce(coords, edges, params, ctx) -> (ctrl (E,K,3) float32, counts (E,) int32)`.
# `producer` imports on demand: a mode may need a compute shader this Blender
# lacks, and an eager import would make that failure fatal for the whole engine.

import importlib

import numpy as np

import gpu

from . import shaders

# Texel row length. Any power of two well inside max_texture_size works.
TEX_ROW = 4096

# Style enum value -> module. One table, so a module that fails to import
# cannot silently drop itself from the UI.
PRODUCERS = {
    'HIERARCHICAL': "heb_gpu",
    'FDEB': "fdeb_gpu",
    'SBEB': "sbeb_gpu",
    'ROUTED': "routed_gpu",
    'MINGLE': "mingle_gpu",
}


def producer(mode):
    """Return the module implementing ``mode``, or None if it will not import."""
    name = PRODUCERS.get(mode)
    if name is None:
        return None
    try:
        return importlib.import_module("." + name, __package__)
    except Exception:  # noqa: BLE001 - a mode that cannot load is skipped
        return None


def available_modes():
    """Return the modes whose producer imports here, in table order."""
    return [m for m in PRODUCERS if producer(m) is not None]


def has_parallel(edges):
    """Whether any node pair carries more than one edge."""
    lo = np.minimum(edges[:, 0], edges[:, 1]).astype(np.int64)
    hi = np.maximum(edges[:, 0], edges[:, 1]).astype(np.int64)
    key = lo * (int(max(hi.max(), lo.max())) + 1) + hi
    return np.unique(key).size != key.size


def common_refusals(params, edges):
    """Reason no mode can route these edges, or "" if there is none. These
    depend on the edges and settings, not on the routing."""
    if edges is None or edges.shape[0] == 0:
        return "no edges"
    if np.any(edges[:, 0] == edges[:, 1]):
        return "self-loops are drawn as circles, not routed paths"
    if (params.get("auto_offset_parallel") and params.get("parallel_offset", 0.0) > 0.0
            and has_parallel(edges)):
        return "parallel offsets move the endpoints before routing"
    return ""


def usable(mode, params, edges, ctx=None):
    """Whether ``mode`` can draw these edges, with the reason if not."""
    mod = producer(mode)
    if mod is None:
        return False, "no producer for %s" % mode
    why = common_refusals(params, edges)
    if why:
        return False, why
    if shaders.get_heb_line_shader() is None:
        return False, "the shader did not compile"
    return mod.check(params, edges, ctx or {})


def endpoint_colors(coords, edges, params, node_colors, edge_color):
    """Return (E, 2, 4) endpoint colors and the edge spans. Shading is affine
    in the curve parameter, so two texels per edge is enough."""
    from . import edge_styles_gpu

    base = np.asarray(edge_color if edge_color is not None
                      else (1.0, 1.0, 1.0, 1.0), dtype=np.float32)
    out = np.empty((edges.shape[0], 2, 4), dtype=np.float32)
    grad = params.get("heb_gradient", 'NONE')
    if grad == 'DIRECTION':
        out[:, 0, :3] = np.asarray(edge_styles_gpu.HEB_SOURCE_RGB, np.float32)
        out[:, 1, :3] = np.asarray(edge_styles_gpu.HEB_TARGET_RGB, np.float32)
    elif grad == 'NODES' and node_colors is not None:
        out[:, 0, :3] = node_colors[edges[:, 0]][:, :3]
        out[:, 1, :3] = node_colors[edges[:, 1]][:, :3]
    else:
        out[:, :, :3] = base[:3]

    # Reach, not arc length: bundling lengthens a curve without moving its
    # nodes, so fading by the arc would make an edge recede for being bundled.
    span = np.linalg.norm(coords[edges[:, 1]] - coords[edges[:, 0]], axis=1)
    fade = float(params.get("heb_fade_long", 0.0))
    if fade > 0.0 and span.max() > 0.0:
        alpha = base[3] * (1.0 - fade * span / span.max())
    else:
        alpha = np.full(edges.shape[0], base[3], dtype=np.float32)
    out[:, 0, 3] = alpha
    out[:, 1, 3] = alpha
    return out, span


# Bisection steps for the straightening solve; twelve resolves 1 part in 4096.
_TURN_STEPS = 12


def _straight_polygon(ctrl, counts):
    """Point i at i/(n-1) along the chord, what the shader straightens toward."""
    rows = np.arange(ctrl.shape[0])
    first = ctrl[:, 0]
    last = ctrl[rows, np.maximum(counts - 1, 0)]
    w = np.zeros((ctrl.shape[0], ctrl.shape[1], 1), dtype=np.float32)
    idx = np.arange(ctrl.shape[1], dtype=np.float32)[None, :]
    span = np.maximum(counts - 1, 1)[:, None].astype(np.float32)
    np.clip(idx / span, 0.0, 1.0, out=w[:, :, 0])
    return first[:, None, :] + w * (last - first)[:, None, :]


def _turns_within(ctrl, counts, cos_limit):
    """(E,) whether every corner is inside the limit. Coincident neighbors pass:
    no angle to measure, and counting them would straighten the route outright."""
    a = ctrl[:, 1:-1] - ctrl[:, :-2]
    b = ctrl[:, 2:] - ctrl[:, 1:-1]
    na = np.linalg.norm(a, axis=2)
    nb = np.linalg.norm(b, axis=2)
    cos_turn = np.sum(a * b, axis=2) / np.maximum(na * nb, 1e-12)
    idx = np.arange(ctrl.shape[1])[None, 1:-1]
    live = ((idx > 0) & (idx < counts[:, None] - 1)
            & (na > 1e-6) & (nb > 1e-6))
    return np.all(~live | (cos_turn >= cos_limit), axis=1)


def limit_turn_angle(ctrl, counts, limit_deg, steps=_TURN_STEPS):
    """Straighten each route by the least that puts its corners inside the limit.

    ``limit_deg`` is the angle between a corner's two arms: 180 is straight,
    small values are hairpins, the opposite of the deviation angle the
    arithmetic uses. Gansner et al. add this limit to agglomerative bundling.
    Bisected over [0, 1] toward the chord rather than relaxed: adjacent points
    share a corner, so nudging both toward their neighbors' midpoint leaves it
    untouched (90 degrees under a 120 degree limit for any number of passes;
    114 and 134 under a limit of 150). On the polygon because a corner needs a
    point's two neighbors and the shader fetches one at a time.
    """
    if limit_deg >= 180.0 or ctrl.shape[1] < 3:
        return ctrl
    cos_limit = float(np.cos(np.deg2rad(180.0 - limit_deg)))
    straight = _straight_polygon(ctrl, counts)

    lo = np.zeros(ctrl.shape[0], dtype=np.float32)
    hi = np.ones(ctrl.shape[0], dtype=np.float32)
    hi[_turns_within(ctrl, counts, cos_limit)] = 0.0
    if not hi.any():
        return ctrl

    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        cand = ctrl + mid[:, None, None] * (straight - ctrl)
        ok = _turns_within(cand, counts, cos_limit)
        hi = np.where(ok, mid, hi)
        lo = np.where(ok, lo, mid)
    return ctrl + hi[:, None, None] * (straight - ctrl)


# Direction bins per axis for the company estimate: 30 degrees a sector, about
# where angle compatibility stops calling two edges parallel.
_COMPANY_DIRS = 6

# Target average occupancy of a company bucket: too fine and every edge is
# alone, too coarse and every edge has company.
_COMPANY_OCCUPANCY = 12.0


def company_beta(coords, edges, amount, dirs=_COMPANY_DIRS):
    """(E,) per-edge multiplier from how much company an edge has. An edge with
    nothing to bundle with only loses by bending. Bucket by midpoint and
    direction, count what landed in each: O(E), against O(E^2) pairwise."""
    if amount <= 0.0 or edges.shape[0] == 0:
        return None
    a = coords[edges[:, 0]].astype(np.float64)
    b = coords[edges[:, 1]].astype(np.float64)
    mid = 0.5 * (a + b)

    lo = mid.min(axis=0)
    extent = np.maximum(mid.max(axis=0) - lo, 1e-9)
    # Cells sized for the target occupancy, never fewer than three per axis.
    # Without the floor a small graph collapses to one cell: one dense corridor
    # plus forty strays gives 2x separation at one cell, 19x at three.
    per_axis = max(3, int(round((edges.shape[0] / (_COMPANY_OCCUPANCY
                                                   * dirs * dirs)) ** (1 / 3))))
    cell = np.floor((mid - lo) / extent * (per_axis - 1e-6)).astype(np.int64)
    np.clip(cell, 0, per_axis - 1, out=cell)

    d = b - a
    n = np.linalg.norm(d, axis=1, keepdims=True)
    d = d / np.maximum(n, 1e-12)
    # Undirected, so the sector is taken on the half-sphere.
    d = np.where(d[:, 2:3] < 0.0, -d, d)
    azim = ((np.arctan2(d[:, 1], d[:, 0]) / np.pi + 1.0) * 0.5) % 1.0
    elev = np.arccos(np.clip(d[:, 2], -1.0, 1.0)) / np.pi
    sector = (np.minimum((azim * dirs).astype(np.int64), dirs - 1) * dirs
              + np.minimum((elev * dirs).astype(np.int64), dirs - 1))

    key = ((cell[:, 0] * per_axis + cell[:, 1]) * per_axis + cell[:, 2])
    key = key * (dirs * dirs) + sector
    counts = np.bincount(key)
    company = counts[key].astype(np.float32)

    # A high percentile, not the maximum: one very busy corridor would otherwise
    # push every other edge toward zero and straighten the graph.
    top = float(np.percentile(company, 95))
    if top <= 1.0:
        return None
    scaled = np.clip((company - 1.0) / (top - 1.0), 0.0, 1.0)
    return ((1.0 - amount) + amount * scaled).astype(np.float32)


def adaptive_beta(span, amount):
    """(E,) per-edge multiplier on the bundling strength, from reach. Long links
    across the graph are what bundling is for; a short one between neighbors
    just loses its direct reading. At ``amount`` 1 the shortest straightens."""
    if amount <= 0.0 or span.size == 0:
        return None
    hi = float(span.max())
    if hi <= 0.0:
        return None
    return ((1.0 - amount) + amount * (span / hi)).astype(np.float32)


# Cells per axis for the occupancy estimate. Coarse on purpose: at fine
# resolution no two curves share a cell and nothing has company.
_DENSITY_CELLS = 64


def route_density(ctrl, counts, cells=_DENSITY_CELLS):
    """(E,) how crowded each edge's route is, normalized to 0..1. Measured on
    the control points, not the drawn samples: at this cell size they agree."""
    e, k, _ = ctrl.shape
    flat = ctrl.reshape(-1, 3)
    lo = flat.min(axis=0)
    hi = flat.max(axis=0)
    extent = np.maximum(hi - lo, 1e-9)
    grid = np.floor((flat - lo) / extent * (cells - 1e-6)).astype(np.int64)
    np.clip(grid, 0, cells - 1, out=grid)
    keys = (grid[:, 0] * cells + grid[:, 1]) * cells + grid[:, 2]

    real = (np.arange(k)[None, :] < counts[:, None]).ravel()
    counts_per_cell = np.bincount(keys[real], minlength=cells ** 3)
    per_point = np.where(real, counts_per_cell[keys], 0.0).reshape(e, k)
    mean = per_point.sum(axis=1) / np.maximum(counts, 1)
    top = float(mean.max())
    return (mean / top).astype(np.float32) if top > 0.0 else None


def texture(values, channels=4):
    """Return an RGBA32F texture holding a flat array of texels, tiled by row."""
    n = values.shape[0]
    rows = int(np.ceil(n / TEX_ROW))
    padded = np.zeros((rows * TEX_ROW, channels), dtype=np.float32)
    padded[:n] = values
    return gpu.types.GPUTexture(
        (TEX_ROW, rows), format='RGBA32F',
        data=gpu.types.Buffer('FLOAT', padded.size, padded.ravel()))


def pack(ctrl, counts, coords, edges, params, node_colors=None,
         edge_color=None, edge_widths=None, mode='HIERARCHICAL'):
    """Turn control polygons into the textures, batch and counts that draw them."""
    kmax = int(ctrl.shape[1])
    colors, span = endpoint_colors(coords, edges, params, node_colors, edge_color)

    turn = float(params.get("bundle_turn_limit", 180.0))
    if turn < 180.0:
        ctrl = limit_turn_angle(ctrl, counts, turn)

    crowding = float(params.get("bundle_density_opacity", 0.0))
    if crowding > 0.0:
        dens = route_density(ctrl, counts)
        if dens is not None:
            # Crowded routes fade instead of sparse ones brightening, which
            # would push the whole image up until only the background is dark.
            colors[:, :, 3] *= (1.0 - crowding * dens)[:, None]

    # Multiplied because the two are independent: a short link with plenty of
    # company still straightens, and so does a long one going where nobody goes.
    betas = adaptive_beta(span, float(params.get("bundle_adaptive_beta", 0.0)))
    keep = company_beta(coords, edges, float(params.get("bundle_company", 0.0)))
    if keep is not None:
        betas = keep if betas is None else (betas * keep).astype(np.float32)

    # Long curves first, so a short link lands on top of the long ones crossing
    # it. The CPU path sorts the same way, so the two match under blending.
    order = np.argsort(-span, kind="stable")
    ctrl, counts, colors = ctrl[order], counts[order], colors[order]
    if betas is not None:
        betas = betas[order]
    widths = None
    if edge_widths is not None:
        # Same permutation: the shader looks widths up by instance.
        widths = np.asarray(edge_widths, dtype=np.float32).ravel()[order]

    # The control point count rides in the unused .w of the first point.
    packed = np.zeros((ctrl.shape[0] * kmax, 4), dtype=np.float32)
    packed[:, :3] = ctrl.reshape(-1, 3)
    packed[::kmax, 3] = counts

    # Two texels per edge, three when the strength varies per edge. The shader
    # reads the stride, so a uniform strength keeps the same layout and bytes.
    meta_stride = 2 if betas is None else 3
    meta = colors
    if betas is not None:
        meta = np.zeros((colors.shape[0], 3, 4), dtype=np.float32)
        meta[:, :2] = colors
        meta[:, 2, 0] = betas

    try:
        ctrl_tex = texture(packed)
        meta_tex = texture(meta.reshape(-1, 4))
        width_tex = None
        if widths is not None:
            wt = np.zeros((widths.size, 4), dtype=np.float32)
            wt[:, 0] = widths
            width_tex = texture(wt)
    except Exception:  # noqa: BLE001 - texture limits, out of VRAM
        return None

    samples = max(1, int(params["segments"])) + 1
    t = np.linspace(0.0, 1.0, samples, dtype=np.float32)

    fmt = gpu.types.GPUVertFormat()
    if width_tex is None:
        # A LINES batch over the sample parameters, which is all the vertex
        # data there is. It does not grow with the graph.
        pairs = np.empty((samples - 1) * 2, dtype=np.float32)
        pairs[0::2] = t[:-1]
        pairs[1::2] = t[1:]
        fmt.attr_add(id="sample_t", comp_type='F32', len=1, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(len=pairs.size, format=fmt)
        vbo.attr_fill("sample_t", pairs)
        vbo_bytes = pairs.nbytes
        batch = gpu.types.GPUBatch(type='LINES', buf=vbo)
    else:
        # Quads instead, so each span can be as thick as its edge asks for.
        # Still one instance's worth, whatever the graph size.
        spans = samples - 1
        seg = np.empty((spans, 4, 4), dtype=np.float32)
        seg[:, :, 0] = t[:-1, None]                  # t0
        seg[:, :, 1] = t[1:, None]                   # t1
        seg[:, :, 2] = (0.0, 0.0, 1.0, 1.0)          # which end
        seg[:, :, 3] = (-1.0, 1.0, -1.0, 1.0)        # which side
        fmt.attr_add(id="seg", comp_type='F32', len=4, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(len=spans * 4, format=fmt)
        vbo.attr_fill("seg", seg.reshape(-1, 4))
        vbo_bytes = seg.nbytes
        base = (np.arange(spans, dtype=np.int32) * 4)[:, None]
        quad = np.array([0, 1, 2, 2, 1, 3], dtype=np.int32)
        ibo = gpu.types.GPUIndexBuf(
            type='TRIS', seq=np.ascontiguousarray((base + quad).reshape(-1, 3)))
        batch = gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo)

    # The eight corners of what is actually drawn, for the view-dependent
    # strength. Padding rows repeat the last real point, so they sit inside.
    flat = ctrl.reshape(-1, 3)
    b_lo, b_hi = flat.min(axis=0), flat.max(axis=0)
    corners = [(b_lo[0] if i & 1 else b_hi[0],
                b_lo[1] if i & 2 else b_hi[1],
                b_lo[2] if i & 4 else b_hi[2]) for i in range(8)]

    return {
        "mode": mode,
        "batch": batch,
        "bounds": corners,
        "ctrl_tex": ctrl_tex,
        "meta_tex": meta_tex,
        "width_tex": width_tex,
        "instances": int(ctrl.shape[0]),
        "stride": int(kmax),
        "meta_stride": meta_stride,
        "samples": int(samples),
        "vertices": int(ctrl.shape[0] * (samples - 1) * 2),
        "bytes": int(packed.nbytes + meta.nbytes + vbo_bytes),
    }


def build(mode, coords, edges, params, ctx=None, node_colors=None,
          edge_color=None, edge_widths=None):
    """Textures, batch and uniforms for drawing ``mode``, or None. Nothing here
    depends on the bundling strength: that is a push constant, so the slider
    costs a redraw and not a rebuild."""
    ctx = ctx or {}
    ok, _why = usable(mode, params, edges, ctx)
    if not ok:
        return None
    made = producer(mode).produce(coords, edges, params, ctx)
    if made is None:
        return None
    ctrl, counts = made
    return pack(ctrl, counts, coords, edges, params, node_colors=node_colors,
                edge_color=edge_color, edge_widths=edge_widths, mode=mode)


# Edges per on-screen pixel where the view-dependent strength reaches the
# user's setting. The airports fixture, 18,833 routes as a whole world map,
# comes to 0.0146 edges per pixel of its own footprint, the zoom where the
# straight drawing is a wash. This knee puts that map at 0.89 of the setting
# and the North Atlantic at 0.13; 0.05 never got the map past 0.54.
_VIEW_KNEE = 0.015


def view_beta(data, beta, amount, mvp, viewport):
    """Scale ``beta`` by how crowded the graph is on screen right now. Crowding
    is edges per pixel of the graph's own footprint, taken from the control
    points and not the object's bounding box: a filter or a backbone cut leaves
    the object as large while drawing a fraction of it."""
    if amount <= 0.0 or data is None or not viewport:
        return beta
    pts = data.get("bounds")
    if pts is None or len(pts) == 0:
        return beta
    pts = np.asarray(pts, dtype=np.float64)
    m = np.asarray(mvp, dtype=np.float64)
    clip = np.column_stack([pts, np.ones(pts.shape[0])]) @ m.T
    w = clip[:, 3]
    if np.any(np.abs(w) < 1e-9):
        return beta
    ndc = clip[:, :2] / w[:, None]
    px = np.abs(ndc.max(axis=0) - ndc.min(axis=0)) * 0.5 * np.asarray(viewport)
    area = float(px[0] * px[1])
    if area <= 1.0:
        return beta
    crowding = data["instances"] / area
    # Square root: crowding is a count over an area, so it goes up fourfold per
    # halving of the zoom and a linear dial would be spent in one wheel click.
    t = float(np.clip(np.sqrt(crowding / _VIEW_KNEE), 0.0, 1.0))
    return beta * ((1.0 - amount) + amount * t)


def draw(data, mvp, beta, shader=None, width_range=None, viewport=None):
    """Draw a bundle. ``beta`` is applied here, not baked into the data.
    ``width_range`` is the (lo, hi) half-widths in device pixels that per-edge
    weights map onto, and only matters for a bundle built with ``edge_widths``;
    without one the curve takes the fixed-function width."""
    if data is None:
        return False
    wide = data.get("width_tex") is not None
    if shader is None:
        shader = shaders.get_heb_line_wide_shader() if wide \
            else shaders.get_heb_line_shader()
    if shader is None:
        return False
    shader.bind()
    shader.uniform_float("u_mvp", mvp)
    shader.uniform_float("u_beta", float(beta))
    shader.uniform_int("u_stride", data["stride"])
    shader.uniform_int("u_ctrl_row", TEX_ROW)
    shader.uniform_int("u_meta_row", TEX_ROW)
    shader.uniform_int("u_meta_stride", int(data.get("meta_stride", 2)))
    shader.uniform_sampler("u_ctrl", data["ctrl_tex"])
    shader.uniform_sampler("u_meta", data["meta_tex"])
    if wide:
        lo, hi = width_range if width_range else (1.0, 1.0)
        vp = viewport if viewport else (1920.0, 1080.0)
        shader.uniform_int("u_width_row", TEX_ROW)
        shader.uniform_sampler("u_width", data["width_tex"])
        shader.uniform_float("u_viewport", (float(vp[0]), float(vp[1])))
        # Full widths, as the line classes use them; the shader halves them.
        shader.uniform_float("u_w_lo", float(lo))
        shader.uniform_float("u_w_hi", float(hi))
    data["batch"].draw_instanced(shader, instance_count=data["instances"])
    return True
