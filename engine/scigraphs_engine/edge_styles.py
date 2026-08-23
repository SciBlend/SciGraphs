# Edge tessellation, every edge at once by broadcasting: a style or curvature
# change costs a batch rebuild, milliseconds for hundreds of thousands of edges.

from functools import lru_cache
import numpy as np

BUNDLE_MAX_EDGES = 4096
HEB_DEGREE = 3
HEB_SOURCE_RGB = (0.30, 0.85, 0.35)
HEB_TARGET_RGB = (0.95, 0.30, 0.25)
_Z_UP = np.array([0.0, 0.0, 1.0])
_X_AXIS = np.array([1.0, 0.0, 0.0])


# Every key ``tessellate`` reads; the values match the add-on's properties.
STYLE_PARAM_DEFAULTS = {
    "style_type": 'STRAIGHT',
    "curvature": 0.3,
    "segments": 8,
    "direction": 'AUTO',
    "orthogonal_style": 'CENTERED',
    "self_loop_radius": 0.2,
    "parallel_offset": 0.05,
    "auto_offset_parallel": True,
    "bundle_strength": 0.6,
    "bundle_iterations": 6,
    "bundle_threshold": 0.6,
    "heb_beta": 0.85,
    "heb_remove_lca": True,
    "heb_fade_long": 0.6,
    "heb_gradient": 'NODES',
    "taper_start": 1.0,
    "taper_end": 0.3,
}


def default_style_params(**overrides):
    """Return a complete params dict with ``overrides`` applied; unknown keys raise."""
    unknown = sorted(set(overrides) - set(STYLE_PARAM_DEFAULTS))
    if unknown:
        raise KeyError(f"unknown edge style parameter(s): {unknown}; "
                       f"known: {sorted(STYLE_PARAM_DEFAULTS)}")
    params = dict(STYLE_PARAM_DEFAULTS)
    params.update(overrides)
    return params



def recover_logical_edges(edges, node_mask):
    """Collapse baked curve polylines back to node-to-node edges. ``node_mask``
    flags real nodes; curve points have exactly two incident segments. Returns
    (E, 2) pairs and a mesh-edge index each, or None on messier topology."""
    if edges is None:
        return None
    num_edges = edges.shape[0]
    if node_mask is None or node_mask.all():
        return edges, np.arange(num_edges, dtype=np.int32)

    direct_rows = np.nonzero(node_mask[edges[:, 0]] & node_mask[edges[:, 1]])[0]
    direct = edges[direct_rows]

    # Neighbor table for curve points (degree must be exactly 2).
    n_verts = node_mask.size
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    mesh_row = np.concatenate([np.arange(num_edges), np.arange(num_edges)])
    forward = np.concatenate([np.ones(num_edges, bool),
                              np.zeros(num_edges, bool)])
    cp_sel = ~node_mask[src]
    cp_src, cp_dst = src[cp_sel], dst[cp_sel]
    deg = np.bincount(cp_src, minlength=n_verts)
    if (deg[~node_mask] != 2).any():
        return None
    order = np.argsort(cp_src, kind="stable")
    nb = np.full((n_verts, 2), -1, dtype=np.int64)
    nb[cp_src[order[0::2]], 0] = cp_dst[order[0::2]]
    nb[cp_src[order[1::2]], 1] = cp_dst[order[1::2]]

    start_sel = node_mask[src] & ~node_mask[dst]
    prev = src[start_sel].copy()
    cur = dst[start_sel].copy()
    start = prev.copy()
    chain_rows = mesh_row[start_sel]
    chain_fwd = forward[start_sel]
    active = ~node_mask[cur]
    for _ in range(n_verts):
        if not active.any():
            break
        c = cur[active]
        n0, n1 = nb[c, 0], nb[c, 1]
        nxt = np.where(n0 != prev[active], n0, n1)
        prev[active] = c
        cur[active] = nxt
        active = ~node_mask[cur]
    else:
        return None  # a cycle of curve points with no node: give up

    chains = np.stack([start, cur], axis=1)
    if direct.size:
        both = np.concatenate([direct, chains])
        rows = np.concatenate([direct_rows, chain_rows])
        fwd = np.concatenate([np.ones(direct_rows.size, bool), chain_fwd])
    else:
        both, rows, fwd = chains, chain_rows, chain_fwd
    if both.size == 0:
        return None
    # Each chain is found from both ends; forward-first keeps arrow direction.
    pref = np.argsort(~fwd, kind="stable")
    key = np.sort(both[pref], axis=1)
    _, uniq = np.unique(key, axis=0, return_index=True)
    sel = np.sort(pref[uniq])
    return both[sel].astype(np.int32), rows[sel].astype(np.int32)


def _perp_unit(d_unit):
    perp = np.cross(d_unit, _Z_UP)
    ln = np.linalg.norm(perp, axis=1)
    vertical = ln < 1e-10
    if vertical.any():
        perp[vertical] = np.cross(d_unit[vertical], _X_AXIS)
        ln = np.linalg.norm(perp, axis=1)
    ln = np.maximum(ln, 1e-12)
    return perp / ln[:, None]


def _direction_sign(a, b, direction, edge_ids):
    if direction == 'CLOCKWISE':
        return np.ones(a.shape[0])
    if direction == 'COUNTER_CLOCKWISE':
        return -np.ones(a.shape[0])
    if direction == 'ALTERNATING':
        return np.where(edge_ids % 2 == 0, 1.0, -1.0)
    # AUTO: deterministic from endpoint positions.
    return np.where(a[:, 0] + a[:, 1] > b[:, 0] + b[:, 1], 1.0, -1.0)


def _parallel_offsets(edges, params):
    """Per-edge (rank, group_size) among edges sharing the same node pair."""
    key = np.sort(edges, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True,
                               return_counts=True)
    order = np.argsort(inv, kind="stable")
    rank = np.empty(edges.shape[0], dtype=np.int64)
    starts = np.zeros(counts.size, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    rank[order] = np.arange(edges.shape[0]) - starts[inv[order]]
    return rank, counts[inv]


def _bezier_cubic(p0, c1, c2, p1, t):
    """(E, 3) control points x (K,) t -> (E, K, 3)."""
    u = 1.0 - t
    return (u**3)[None, :, None] * p0[:, None] \
        + (3 * u**2 * t)[None, :, None] * c1[:, None] \
        + (3 * u * t**2)[None, :, None] * c2[:, None] \
        + (t**3)[None, :, None] * p1[:, None]


def _bezier_quadratic(p0, c, p1, t):
    u = 1.0 - t
    return (u**2)[None, :, None] * p0[:, None] \
        + (2 * u * t)[None, :, None] * c[:, None] \
        + (t**2)[None, :, None] * p1[:, None]


def _curved_points(a, b, params, edge_ids, use_cubic, t):
    length = np.linalg.norm(b - a, axis=1)
    offs = length * params["curvature"] * 0.5
    sign = _direction_sign(a, b, params["direction"], edge_ids)
    perp = _perp_unit((b - a) / np.maximum(length, 1e-12)[:, None])
    perp = perp * (offs * sign)[:, None]
    if use_cubic:
        c1 = a + (b - a) * 0.25 + perp * 0.5
        c2 = a + (b - a) * 0.75 + perp * 0.5
        return _bezier_cubic(a, c1, c2, b, t)
    return _bezier_quadratic(a, (a + b) * 0.5 + perp, b, t)


def _arc_points(a, b, params, edge_ids, t):
    chord = np.linalg.norm(b - a, axis=1)
    sagitta = chord * params["curvature"] * 0.5
    flat = sagitta <= 1e-3
    radius = np.where(
        flat, 1.0, chord**2 / np.maximum(8.0 * sagitta, 1e-12) + sagitta / 2.0
    )
    sign = _direction_sign(a, b, params["direction"], edge_ids)
    perp = _perp_unit((b - a) / np.maximum(chord, 1e-12)[:, None])
    perp = perp * sign[:, None]
    mid = (a + b) * 0.5
    center = mid - perp * (radius - sagitta)[:, None]

    v0 = a - center
    v1 = b - center
    v0n = v0 / np.maximum(np.linalg.norm(v0, axis=1), 1e-12)[:, None]
    v1n = v1 / np.maximum(np.linalg.norm(v1, axis=1), 1e-12)[:, None]
    vi = v0n[:, None] * (1.0 - t)[None, :, None] + v1n[:, None] * t[None, :, None]
    vi = vi / np.maximum(np.linalg.norm(vi, axis=2), 1e-12)[:, :, None]
    pts = center[:, None] + vi * radius[:, None, None]
    if flat.any():  # curvature ~ 0 degenerates: straight lerp
        straight = a[:, None] * (1.0 - t)[None, :, None] \
            + b[:, None] * t[None, :, None]
        pts[flat] = straight[flat]
    return pts


def _orthogonal_points(a, b, params):
    """(E, 4, 3) right-angle routings; tessellate drops the duplicated bends."""
    style = params["orthogonal_style"]
    mid = (a + b) * 0.5
    if style == 'CENTERED':
        p_a = np.stack([mid[:, 0], a[:, 1], a[:, 2]], axis=1)
        p_b = np.stack([mid[:, 0], b[:, 1], mid[:, 2]], axis=1)
    elif style == 'HORIZONTAL_FIRST':
        p_a = np.stack([b[:, 0], a[:, 1], a[:, 2]], axis=1)
        p_b = np.stack([b[:, 0], b[:, 1], a[:, 2]], axis=1)
    elif style == 'VERTICAL_FIRST':
        p_a = np.stack([a[:, 0], b[:, 1], a[:, 2]], axis=1)
        p_b = np.stack([a[:, 0], b[:, 1], b[:, 2]], axis=1)
    else:  # SHORTEST: bend along the dominant axis
        horiz = np.abs(b[:, 0] - a[:, 0]) > np.abs(b[:, 1] - a[:, 1])
        p_a = np.where(horiz[:, None],
                       np.stack([b[:, 0], a[:, 1], a[:, 2]], axis=1),
                       np.stack([a[:, 0], b[:, 1], a[:, 2]], axis=1))
        p_b = p_a
    return np.stack([a, p_a, p_b, b], axis=1)


def _bundled_points(a, b, params, t):
    """Vectorized force-directed edge bundling (FDEB)."""
    e = a.shape[0]
    pts = a[:, None] * (1.0 - t)[None, :, None] + b[:, None] * t[None, :, None]
    if e < 2:
        return pts

    d = b - a
    length = np.linalg.norm(d, axis=1)
    ok = length > 1e-10
    dn = d / np.maximum(length, 1e-12)[:, None]

    angle = np.abs(dn @ dn.T)
    l_i = length[:, None]
    l_j = length[None, :]
    l_avg = (l_i + l_j) * 0.5
    scale = 2.0 / (l_avg / np.minimum(l_i, l_j)
                   + np.maximum(l_i, l_j) / l_avg)
    mid = (a + b) * 0.5
    mid_dist = np.linalg.norm(mid[:, None] - mid[None, :], axis=2)
    pos = l_avg / (l_avg + mid_dist)

    compat = angle * scale * pos
    compat[compat < params["bundle_threshold"]] = 0.0
    np.fill_diagonal(compat, 0.0)
    compat[~ok] = 0.0
    compat[:, ~ok] = 0.0
    row_sum = compat.sum(axis=1)

    strength = params["bundle_strength"]
    iterations = max(1, params["bundle_iterations"])
    step = 0.1 * strength
    # Row-sum division caps a step at the weighted mean. Unscaled it diverges
    # within a dozen iterations on a few hundred edges.
    inv = 1.0 / np.maximum(row_sum, 1e-12)
    for it in range(iterations):
        cur = step * (1.0 - it / iterations)
        inner = pts[:, 1:-1]
        spring = (pts[:, :-2] + pts[:, 2:]) * 0.5 - inner
        attract = (np.tensordot(compat, inner, axes=(1, 0))
                   - row_sum[:, None, None] * inner) * inv[:, None, None]
        pts[:, 1:-1] = inner + (spring * 0.5 + attract * strength) * cur
    return pts


# Hierarchical edge bundling (Holten 2006). An edge routes up to the least
# common ancestor in simplify.build_hierarchy's tree and back down; that path is
# the B-spline control polygon. Linear in edges, not FDEB's (E, E) matrix.

@lru_cache(maxsize=64)
def _bspline_basis(n_ctrl, n_samples, degree=HEB_DEGREE):
    """(n_samples, n_ctrl) basis of an open uniform B-spline. End knots repeat
    ``degree + 1`` times so the curve hits its first and last control point."""
    p = min(degree, n_ctrl - 1)
    interior = n_ctrl - p - 1
    knots = np.concatenate([
        np.zeros(p + 1),
        np.arange(1, interior + 1) / (interior + 1.0) if interior > 0
        else np.empty(0),
        np.ones(p + 1),
    ])
    t = np.linspace(0.0, 1.0, n_samples)

    # Degree zero: one indicator per half-open knot span; t = 1 falls off the
    # end of every one and is put back into the last by hand.
    spans = knots.size - 1
    basis = np.zeros((n_samples, spans))
    for i in range(spans):
        if knots[i] < knots[i + 1]:
            basis[:, i] = (t >= knots[i]) & (t < knots[i + 1])
    basis[t >= 1.0, np.nonzero(knots[:-1] < knots[1:])[0][-1]] = 1.0

    for d in range(1, p + 1):  # Cox-de Boor, one degree at a time
        nxt = np.zeros((n_samples, spans - d))
        for i in range(spans - d):
            lo = knots[i + d] - knots[i]
            hi = knots[i + d + 1] - knots[i + 1]
            if lo > 0.0:
                nxt[:, i] += (t - knots[i]) / lo * basis[:, i]
            if hi > 0.0:
                nxt[:, i] += (knots[i + d + 1] - t) / hi * basis[:, i + 1]
        basis = nxt
    return basis


def _heb_points(a, b, a_idx, b_idx, levels, params, n_samples):
    """(E, n_samples, 3) curves routed along the hierarchy. ``a``/``b`` already
    carry any parallel offset; the indices look up ancestry."""
    depth = len(levels)
    # member_of already holds every node's ancestor chain, so no tree walk.
    anc_a = np.stack([lv["member_of"][a_idx] for lv in levels], axis=1)
    anc_b = np.stack([lv["member_of"][b_idx] for lv in levels], axis=1)
    same = anc_a == anc_b
    # Least common ancestor: shallowest level where both fall in one community.
    # First match, not a level count, so imperfect nesting still works.
    lca = np.where(same.any(axis=1), same.argmax(axis=1), depth)

    beta = float(params["heb_beta"])
    remove_lca = bool(params["heb_remove_lca"])
    pts = np.empty((a.shape[0], n_samples, 3), dtype=np.float64)

    for level in np.unique(lca):
        rows = np.nonzero(lca == level)[0]
        up = [levels[l]["centers"][anc_a[rows, l]].astype(np.float64)
              for l in range(level)]
        down = [levels[l]["centers"][anc_b[rows, l]].astype(np.float64)
                for l in reversed(range(level))]
        # Disconnected endpoints share no ancestor: up one side, down the other.
        shared = ([levels[level]["centers"][anc_a[rows, level]].astype(np.float64)]
                  if level < depth else [])
        chain = [a[rows]] + up + shared + down + [b[rows]]

        # Dropping the common ancestor keeps siblings from detouring through the
        # parent. Skipped at three control points: every pair would be one line.
        if remove_lca and shared and len(chain) > 3:
            del chain[len(up) + 1]

        ctrl = np.stack(chain, axis=1)
        n_ctrl = ctrl.shape[1]
        if beta < 1.0:
            # Straighten the control polygon, not the curve. Looks the same and
            # costs one operation per control point instead of per sample.
            w = np.linspace(0.0, 1.0, n_ctrl)
            straight = ctrl[:, :1] + w[None, :, None] * (ctrl[:, -1:] - ctrl[:, :1])
            ctrl = beta * ctrl + (1.0 - beta) * straight
        pts[rows] = np.einsum(
            "mn,gnj->gmj", _bspline_basis(n_ctrl, n_samples), ctrl)
    return pts


def _heb_shading(span, a_idx, b_idx, params, base_color, node_colors,
                 n_samples):
    """(E, n_samples, 4) color per curve, opacity falling with ``span``, the
    straight-line reach. By arc length an edge would recede for being bundled."""
    rgba = np.empty((span.size, n_samples, 4), dtype=np.float32)
    grad = params["heb_gradient"]
    t = np.linspace(0.0, 1.0, n_samples)[None, :, None]
    if grad == 'DIRECTION':
        src = np.asarray(HEB_SOURCE_RGB, dtype=np.float32)[None, None]
        dst = np.asarray(HEB_TARGET_RGB, dtype=np.float32)[None, None]
        rgba[..., :3] = src + t * (dst - src)
    elif grad == 'NODES' and node_colors is not None:
        src = node_colors[a_idx][:, None, :3]
        dst = node_colors[b_idx][:, None, :3]
        rgba[..., :3] = src + t * (dst - src)
    else:
        rgba[..., :3] = np.asarray(base_color[:3], dtype=np.float32)

    fade = float(params["heb_fade_long"])
    if fade > 0.0 and span.max() > 0.0:
        rgba[..., 3] = (base_color[3]
                        * (1.0 - fade * span / span.max()))[:, None]
    else:
        rgba[..., 3] = base_color[3]
    return rgba


def _self_loop_points(centers, params):
    """(L, S+1, 3) closed polylines for self-loop edges."""
    radius = params["self_loop_radius"]
    segs = max(params["segments"], 8)
    ang = np.linspace(0.0, 2.0 * np.pi, segs + 1)
    circle = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    loop_center = centers + np.array([radius * 0.5, 0.0, radius * 0.5])
    return loop_center[:, None] + circle[None] * radius


def tessellate(coords, edges, params, edge_widths=None, hierarchy=None,
               node_colors=None, edge_color=None):
    """Turn logical edges into styled segments. ``edge_widths`` is an optional
    (E,) radius multiplier folded into the style's own taper; ``hierarchy`` is
    ``simplify.build_hierarchy``'s tree, required by HIERARCHICAL, and
    ``node_colors``/``edge_color`` feed only its shading. Returns None when there
    is nothing to draw, else ``seg_a``/``seg_b`` (S, 3) endpoints, ``scale_a``/
    ``scale_b`` (S,) radius multipliers, ``seg_edge`` (S,) owning edge ids, four
    ``arrow_*`` arrays per non-loop edge, and ``col_a``/``col_b`` (S, 4)."""
    if edges is None or edges.shape[0] == 0:
        return None

    if params is None or set(params) != set(STYLE_PARAM_DEFAULTS):
        params = default_style_params(**(params or {}))

    style = params["style_type"]
    heb = style == 'HIERARCHICAL' and bool(hierarchy)
    # A None reaching the self-loop branch becomes an (N, 1) array of NaN, which
    # then fails to concatenate against the (N, 4) colors from the curves.
    base_rgba = (1.0, 1.0, 1.0, 1.0) if edge_color is None else edge_color
    edge_ids = np.arange(edges.shape[0], dtype=np.int64)
    loops_sel = edges[:, 0] == edges[:, 1]
    plain_rows = np.flatnonzero(~loops_sel)
    plain = edges[~loops_sel]
    plain_ids = edge_ids[~loops_sel]
    if edge_widths is None:
        edge_widths = np.ones(edges.shape[0], dtype=np.float32)
    w_plain = np.asarray(edge_widths, dtype=np.float32)[~loops_sel]

    seg_a_parts, seg_b_parts = [], []
    col_a_parts, col_b_parts = [], []
    scale_a_parts, scale_b_parts = [], []
    # Which logical edge each segment came from; a polyline hides that.
    owner_parts = []
    arrows = None

    if plain.shape[0]:
        a = coords[plain[:, 0]].astype(np.float64)
        b = coords[plain[:, 1]].astype(np.float64)

        if params["auto_offset_parallel"] and params["parallel_offset"] > 0.0:
            rank, group = _parallel_offsets(plain, params)
            multi = group > 1
            if multi.any():
                base = params["parallel_offset"]
                amount = (-base * (group - 1) * 0.5 + rank * base)
                perp = _perp_unit(
                    (b - a) / np.maximum(
                        np.linalg.norm(b - a, axis=1), 1e-12)[:, None]
                )
                shift = perp * np.where(multi, amount, 0.0)[:, None]
                a = a + shift
                b = b + shift

        segs = max(1, params["segments"])
        t = np.linspace(0.0, 1.0, segs + 1)

        if heb:
            # Long curves first, so short ones land on top.
            order = np.argsort(-np.linalg.norm(b - a, axis=1), kind="stable")
            a, b = a[order], b[order]
            plain, plain_ids = plain[order], plain_ids[order]
            plain_rows, w_plain = plain_rows[order], w_plain[order]

        # CURVED rides a cubic; QUADRATIC and TAPERED ride a quadratic.
        if (style in ('CURVED', 'QUADRATIC', 'TAPERED')
                and params["curvature"] >= 1e-3):
            pts = _curved_points(a, b, params, plain_ids,
                                 style == 'CURVED', t)
        elif style == 'ARC' and params["curvature"] >= 1e-3:
            pts = _arc_points(a, b, params, plain_ids, t)
        elif style == 'ORTHOGONAL':
            pts = _orthogonal_points(a, b, params)
            t = np.linspace(0.0, 1.0, pts.shape[1])
        elif style == 'BUNDLED':
            if plain.shape[0] <= BUNDLE_MAX_EDGES:
                pts = _bundled_points(a, b, params, t)
            else:
                pts = _curved_points(a, b, params, plain_ids, True, t)
        elif heb:
            pts = _heb_points(a, b, plain[:, 0], plain[:, 1], hierarchy,
                              params, t.size)
        else:  # STRAIGHT (or a curved style at curvature ~ 0): one segment
            pts = np.stack([a, b], axis=1)
            t = np.array([0.0, 1.0])

        if heb:
            rgba = _heb_shading(
                np.linalg.norm(b - a, axis=1), plain[:, 0], plain[:, 1], params,
                base_rgba, node_colors, pts.shape[1])
            col_a_parts.append(rgba[:, :-1].reshape(-1, 4))
            col_b_parts.append(rgba[:, 1:].reshape(-1, 4))

        if style == 'TAPERED':
            taper = params["taper_start"] + (
                params["taper_end"] - params["taper_start"]) * t
        else:
            taper = np.ones(t.size)

        n_edges = pts.shape[0]
        segs_per_edge = pts.shape[1] - 1
        seg_a_parts.append(pts[:, :-1].reshape(-1, 3))
        seg_b_parts.append(pts[:, 1:].reshape(-1, 3))
        w_rep = np.repeat(w_plain, segs_per_edge)
        scale_a_parts.append(np.tile(taper[:-1], n_edges) * w_rep)
        scale_b_parts.append(np.tile(taper[1:], n_edges) * w_rep)
        # Row-major reshape, so each edge's segments are contiguous.
        owner_parts.append(np.repeat(plain_rows, segs_per_edge))

        # At the end of the last segment, so arrows follow curvature.
        last_dir = pts[:, -1] - pts[:, -2]
        last_len = np.maximum(np.linalg.norm(last_dir, axis=1), 1e-12)
        end_width = w_plain * (taper[-1] if style == 'TAPERED' else 1.0)
        arrows = {
            "arrow_pos": pts[:, -1].astype(np.float32),
            "arrow_dir": (last_dir / last_len[:, None]).astype(np.float32),
            "arrow_width": end_width.astype(np.float32),
            "arrow_target": plain[:, 1].astype(np.int32),
        }

    if loops_sel.any():
        centers = coords[edges[loops_sel, 0]].astype(np.float64)
        pts = _self_loop_points(centers, params)
        seg_a_parts.append(pts[:, :-1].reshape(-1, 3))
        seg_b_parts.append(pts[:, 1:].reshape(-1, 3))
        n_seg = pts.shape[1] - 1
        w_loops = np.repeat(
            np.asarray(edge_widths, dtype=np.float32)[loops_sel], n_seg)
        scale_a_parts.append(w_loops)
        scale_b_parts.append(w_loops)
        owner_parts.append(np.repeat(np.flatnonzero(loops_sel), n_seg))
        if col_a_parts:  # a loop has no length and no direction: flat color
            loop_rgba = np.tile(np.asarray(base_rgba, dtype=np.float32),
                                (pts.shape[0] * n_seg, 1))
            col_a_parts.append(loop_rgba)
            col_b_parts.append(loop_rgba)

    if not seg_a_parts:
        return None

    seg_a = np.concatenate(seg_a_parts).astype(np.float32)
    seg_b = np.concatenate(seg_b_parts).astype(np.float32)
    scale_a = np.concatenate(scale_a_parts).astype(np.float32)
    scale_b = np.concatenate(scale_b_parts).astype(np.float32)
    seg_edge = np.concatenate(owner_parts).astype(np.int64)
    col_a = np.concatenate(col_a_parts) if col_a_parts else None
    col_b = np.concatenate(col_b_parts) if col_b_parts else None

    # Zero-length segments (duplicate bends, coincident nodes) billboard as blobs.
    keep = np.linalg.norm(seg_b - seg_a, axis=1) > 1e-9
    if not keep.all():
        seg_a, seg_b = seg_a[keep], seg_b[keep]
        scale_a, scale_b = scale_a[keep], scale_b[keep]
        seg_edge = seg_edge[keep]
        if col_a is not None:
            col_a, col_b = col_a[keep], col_b[keep]
    if seg_a.shape[0] == 0:
        return None

    out = {
        "seg_a": seg_a,
        "seg_b": seg_b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "seg_edge": seg_edge,
    }
    if col_a is not None:
        out["col_a"] = col_a.astype(np.float32)
        out["col_b"] = col_b.astype(np.float32)
    if arrows is not None:
        out.update(arrows)
    return out
