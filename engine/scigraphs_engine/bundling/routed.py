# Geometry-based routed bundling, "winding roads" (Lambert et al. 2010).
#
# Edges share a grid, so two routes through the same cells coincide there, and
# a cell that carries traffic gets cheaper. Dijkstra wants a priority queue a
# compute shader cannot have, so this is Jacobi Bellman-Ford on a regular grid,
# and the round count follows the grid diameter, not the edge count. The
# full-box stencil is 8% off Euclidean on a diagonal at worst, where
# 4-/6-connectivity would make the metric Manhattan. The add-on reimplements
# the relaxation and the trace as compute kernels and calls the rest from here.

import itertools

import numpy as np

# "Unreachable": above any real distance, below float32's overflow on a step.
INF = 1e30

# A road with more than a dozen turns is one the spline smooths away.
MAX_CTRL = 16

# Cost of crossing the densest cell at ``routed_avoid_nodes = 1``, against
# empty space. 8 buys a detour of eight cells, visible on a 128-cell grid.
AVOID_GAIN = 8.0

# Floor on a reinforced cell: at zero cost the trace has nothing to descend.
MIN_COST = 0.05

# Cells left empty around the node box, so a route can bow outside the hull.
MARGIN = 2

# Ceilings on ``sources * cells``, the only array that scales with both the
# graph and the resolution. A numpy round costs eight full-size temporaries.
GPU_FIELD_TEXELS = 16_000_000
NP_FIELD_TEXELS = 2_000_000


def _frame(coords, dims):
    """Return the origin, orthonormal basis and node coordinates in it. Rows by
    decreasing variance, so the routing plane is the graph's own plane."""
    coords = np.asarray(coords, dtype=np.float64)
    origin = coords.mean(axis=0)
    q = coords - origin
    _vals, vecs = np.linalg.eigh(q.T @ q)
    basis = vecs[:, ::-1].T                       # eigh sorts ascending
    if np.linalg.det(basis) < 0.0:
        basis[2] = -basis[2]
    local = q @ basis.T
    return origin, basis, local[:, :dims], local[:, dims:]


def _grid(local, res, margin=MARGIN):
    """Return ``(lo, cell, counts)`` for a grid covering ``local``. Cells are
    cubic; with anisotropic ones a diagonal's cost depends on which diagonal."""
    lo = local.min(axis=0)
    span = np.maximum(local.max(axis=0) - lo, 1e-9)
    cell = float(span.max()) / max(int(res), 1)
    counts = np.maximum(np.ceil(span / cell), 1).astype(np.int64) + 2 * margin
    return lo - margin * cell, cell, counts


def _cell_of(local, lo, cell, counts):
    """Return the flat cell index per point, x fastest (shader decodes by divmod)."""
    ix = np.floor((local - lo) / cell).astype(np.int64)
    np.clip(ix, 0, counts - 1, out=ix)
    strides = np.concatenate([[1], np.cumprod(counts[:-1])])
    return ix @ strides


# float32(sqrt(2)) and float32(sqrt(3)) as literals: the spec allows GLSL's
# ``length()`` two ULP, and one ULP makes the backends break a tie differently.
_STEP_LEN = {1: 1.0, 2: 1.4142135381698608, 3: 1.7320507764816284}


def _offsets(dims):
    """Return the stencil, array-axis order, with each offset's Euclidean length."""
    off = [o for o in itertools.product((-1, 0, 1), repeat=dims) if any(o)]
    return off, np.array([_STEP_LEN[sum(c * c for c in o)] for o in off],
                         dtype=np.float32)


def _shift(a, off, fill=INF):
    """Shift ``a`` so element ``i`` reads ``a[i + off]``, ``fill`` outside. A
    padded slice, not ``np.roll``, whose wrap joins opposite grid edges."""
    p = np.pad(a, [(1, 1)] * a.ndim, mode="constant", constant_values=fill)
    return p[tuple(slice(1 + o, 1 + o + n) for o, n in zip(off, a.shape))]


def _blur(a):
    """One separable 1-2-1 pass, clamped at the border. An isolated spike
    leaves the relaxation nothing to descend and no road to be pulled onto."""
    out = a
    for ax in range(a.ndim):
        pad = [(1, 1) if i == ax else (0, 0) for i in range(a.ndim)]
        p = np.pad(out, pad, mode="edge")
        lo = tuple(slice(0, n) if i == ax else slice(None)
                   for i, n in enumerate(out.shape))
        hi = tuple(slice(2, 2 + n) if i == ax else slice(None)
                   for i, n in enumerate(out.shape))
        out = 0.25 * (p[lo] + 2.0 * out + p[hi])
    return out


def density_cost(coords, node_cell, shape, res, avoid, build_field=None):
    """Return the base cost per cell: 1 everywhere, more where the nodes crowd.
    ``build_field(coords, res)`` is the node-density source and returns a
    mapping with a ``node_density`` entry, or None. With no source, or one that
    returns None, every cell costs the same."""
    base = np.ones(shape, dtype=np.float32)
    if avoid <= 0.0 or build_field is None:
        return base

    # Half the routing resolution: building the field is the expensive half.
    field = build_field(np.asarray(coords, dtype=np.float64),
                        max(8, int(res) // 2))
    if field is None:
        return base
    dens = np.asarray(field["node_density"], dtype=np.float64)
    if dens.size == 0 or not np.isfinite(dens).any():
        return base

    # The high percentile the field uses on its own channel. The densest voxel
    # is several times the next, so dividing by that avoids one spot only.
    hot = float(np.percentile(dens, 99.5))
    if hot <= 0.0:
        return base
    per_cell = np.bincount(node_cell, weights=np.clip(dens / hot, 0.0, 1.0),
                           minlength=base.size).reshape(shape)
    per_cell = _blur(per_cell)
    peak = float(per_cell.max())
    if peak <= 0.0:
        return base
    return (1.0 + float(avoid) * AVOID_GAIN
            * (per_cell / peak)).astype(np.float32)


def _reinforced(base, usage, reinforce):
    """Return the cost after used roads have been made cheaper. Recomputed from
    the base each round, not compounded: re-discounting drives every used cell
    to the floor in three or four rounds. log1p because usage is heavy-tailed."""
    if reinforce <= 0.0 or usage.max() <= 0:
        return base
    spread = _blur(usage.astype(np.float32))
    top = float(spread.max())
    if top <= 0.0:
        return base
    frac = np.log1p(spread) / np.log1p(top)
    return (base * np.maximum(1.0 - float(reinforce) * frac,
                              MIN_COST)).astype(np.float32)


def _relax_numpy(cost, seeds, cell, eps, max_rounds):
    """Relax Bellman-Ford to a fixed point. Returns ``(dist, rounds, converged)``.
    ``dist`` is ``(S, ncells)``, one field per source, all relaxed in the same
    rounds: free in wall clock, not in memory. The round cap stops a NaN in the
    cost field from hanging the caller."""
    shape = tuple(int(n) for n in cost.shape)
    dims = cost.ndim
    offs, lens = _offsets(dims)
    ncells = int(np.prod(shape))
    s = int(np.asarray(seeds).size)

    # Mean of the two cells' costs, so the metric stays symmetric: otherwise a
    # routes to b differently than b to a, and two edges between the same
    # clusters refuse to share a road. Off the grid the cost clamps to the
    # border, not INF, whose product with a step length is a NaN. Kernel's
    # order (half, sum, step, size): a folded constant moves an answer an ULP.
    half = np.float32(0.5)
    size = np.float32(cell)
    weights = [half * (cost + _shift(cost, o, fill=1.0)) * l * size
               for o, l in zip(offs, lens)]

    dist = np.full((s,) + shape, INF, dtype=np.float32)
    dist.reshape(s, -1)[np.arange(s), np.asarray(seeds)] = 0.0

    # Refilled in place: the biggest allocation in the loop, needed once.
    buf = np.full((s,) + tuple(n + 2 for n in shape), INF, dtype=np.float32)
    inner = (slice(None),) + (slice(1, -1),) * dims

    rounds = 0
    for rounds in range(1, int(max_rounds) + 1):
        buf[inner] = dist
        new = dist.copy()
        for o, w in zip(offs, weights):
            sl = (slice(None),) + tuple(slice(1 + d, 1 + d + n)
                                        for d, n in zip(o, shape))
            np.minimum(new, buf[sl] + w, out=new)
        improved = bool(np.any(new < dist - eps))
        dist = new
        if not improved:
            return dist.reshape(s, ncells), rounds, True
    return dist.reshape(s, ncells), rounds, False


def _neighbor_table(shape):
    """Return ``(ncells, K)`` neighbor cell indices, -1 off the grid."""
    dims = len(shape)
    offs, lens = _offsets(dims)
    idx = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    pad = np.full(tuple(n + 2 for n in shape), -1, dtype=np.int64)
    pad[(slice(1, -1),) * dims] = idx
    table = np.stack([
        pad[tuple(slice(1 + d, 1 + d + n) for d, n in zip(o, shape))].ravel()
        for o in offs], axis=1)
    return table, lens


def _trace_numpy(dist, cost_flat, table, lens, cell, field, src, dst, max_steps):
    """Walk every edge back from its destination. Returns ``(routes, lengths)``.
    The step minimizes ``dist[n] + w(n, x)``, not ``dist[n]`` alone: descending
    on distance reaches the source over a path whose cost is not ``dist[dst]``."""
    e = int(src.size)
    routes = np.full((e, int(max_steps)), -1, dtype=np.int64)
    lengths = np.zeros(e, dtype=np.int64)
    cur = np.asarray(dst, dtype=np.int64).copy()
    alive = np.ones(e, dtype=bool)
    half = np.float32(0.5)
    size = np.float32(cell)

    for s in range(int(max_steps)):
        live = np.flatnonzero(alive)
        if live.size == 0:
            break
        routes[live, s] = cur[live]
        lengths[live] = s + 1
        alive[live[cur[live] == src[live]]] = False
        live = np.flatnonzero(alive)
        if live.size == 0:
            break

        nbr = table[cur[live]]                          # (m, K)
        ok = nbr >= 0
        safe = np.where(ok, nbr, 0)
        dn = np.where(ok, dist[field[live][:, None], safe], INF)
        # Same multiplication order as the relaxation; see ``_relax_numpy``.
        w = (half * (cost_flat[cur[live]][:, None] + cost_flat[safe])
             * lens[None, :] * size)
        pick = np.argmin(np.where(ok, dn + w, INF), axis=1)
        rows = np.arange(live.size)
        nxt = nbr[rows, pick]

        # Stop rather than loop: a non-decreasing step is a plateau.
        stuck = (nxt < 0) | (dn[rows, pick] >= dist[field[live], cur[live]])
        alive[live[stuck]] = False
        good = live[~stuck]
        cur[good] = nxt[~stuck]
    return routes, lengths


def _plan(coords, edges, params, budget, density=None):
    """Build the parts of the routing that do not change between rounds. Both
    backends share this verbatim, so the oracle only checks the solve and the
    trace. ``density`` replaces :func:`density_cost`."""
    # 2D by default. Measured on a non-planar test graph (240 nodes, 592 edges,
    # three real principal axes, 168 fields), GPU path, matched cells:
    #
    #     R   grid cells          solve   distinct cells   routes per road cell
    #     24  28x28    =   784   13.5 ms    416 ->  299           11.07
    #     24  28x28x19 = 14896  125.9 ms   1368 ->  663            5.22
    #     48  51x52    =  2652   33.7 ms   1441 ->  708            8.60
    #     48  51x52x33 = 87516 1244.9 ms   3930 -> 1599            3.96
    #
    # 37x the cost at R = 48 and growing: a 3D grid adds an axis of cells while
    # the round count stays with the diameter. At R = 128 the field misses the
    # texel budget and the guard below drops R to 32. The last column says
    # whether routes became roads. A road is codimension 1 in a plane and 2 in
    # a volume, so a crossing path meets it in 2D and misses in 3D. The 2D path
    # lifts the out-of-plane coordinate linearly along the edge.
    dims = 3 if params.get("routed_3d", False) else 2
    res = max(8, int(params.get("routed_resolution", 128)))
    avoid = float(np.clip(params.get("routed_avoid_nodes", 0.0), 0.0, 1.0))

    coords = np.asarray(coords, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    origin, basis, local, out_of_plane = _frame(coords, dims)

    # One distance field per distinct source, so the solve's memory follows the
    # node count: 592 edges over 240 nodes give 168 fields. Taking the
    # higher-degree endpoint is a one-line greedy at the set cover, worth about
    # that much (168 fields against 204 for "always endpoint 0", 1153 against
    # 1188 on a 6k-edge scale-free graph). The win is grouping at all.
    degree = np.bincount(edges.ravel(), minlength=coords.shape[0])
    take_a = degree[edges[:, 0]] >= degree[edges[:, 1]]
    src_node = np.where(take_a, edges[:, 0], edges[:, 1])
    dst_node = np.where(take_a, edges[:, 1], edges[:, 0])

    lo, cell, counts = _grid(local, res)
    ncells = int(np.prod(counts))
    sources = np.unique(src_node)
    # Drop the resolution, not the graph: halving keeps the cells cubic.
    while sources.size * ncells > budget and res > 8:
        res //= 2
        lo, cell, counts = _grid(local, res)
        ncells = int(np.prod(counts))

    shape = tuple(int(n) for n in counts[::-1])
    node_cell = _cell_of(local, lo, cell, counts)
    field = np.searchsorted(sources, src_node)

    src_cell = node_cell[src_node]
    dst_cell = node_cell[dst_node]
    straight = src_cell == dst_cell

    cost_of = density_cost if density is None else density
    return {
        "dims": dims, "res": res, "origin": origin, "basis": basis,
        "local": local, "out_of_plane": out_of_plane,
        "lo": lo, "cell": cell, "counts": counts, "shape": shape,
        "ncells": ncells, "node_cell": node_cell,
        "sources": sources, "field": field,
        "src_node": src_node, "dst_node": dst_node,
        "src_cell": src_cell, "dst_cell": dst_cell, "straight": straight,
        "base_cost": cost_of(coords, node_cell, shape, res, avoid),
        # A shortest path never revisits a cell, so 4x the diameter bounds any
        # route that is not a bug on an unconverged field.
        "max_steps": int(min(4 * int(max(counts)), 4096)),
        "max_rounds": int(min(8 * int(max(counts)), 4096)),
        # Relative to the cheapest step, so the test holds at any scale.
        "eps": float(1e-5 * cell),
    }


def _ragged_arange(counts):
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    return (np.arange(total, dtype=np.int64)
            - np.repeat(np.cumsum(counts) - counts, counts))


def _control(solution, coords, edges):
    """Return ``(ctrl, counts)``: the routes as padded control polygons.
    Interior cells only, since the end cells are the nodes' own up to half a
    cell off and the B-spline interpolates its ends. Resampled by arc length,
    because cell spacing alternates between one and root two."""
    plan = solution["plan"]
    routes, lengths = solution["routes"], solution["lengths"]
    dims = plan["dims"]
    e = int(edges.shape[0])
    coords = np.asarray(coords, dtype=np.float64)

    # Routes run destination -> source; which endpoint was the source decides
    # whether that is already 0 -> 1.
    reverse = plan["src_node"] == edges[:, 0]

    good = lengths >= 2
    good &= routes[np.arange(e), np.maximum(lengths - 1, 0)] == plan["src_cell"]

    inner_n = np.where(good, np.maximum(lengths - 2, 0), 0)
    poly_n = inner_n + 2
    starts = np.concatenate([[0], np.cumsum(poly_n)[:-1]])
    total = int(poly_n.sum())

    # Built in the routing frame, then transformed once.
    a_loc = np.concatenate([plan["local"][edges[:, 0]],
                            plan["out_of_plane"][edges[:, 0]]], axis=1)
    b_loc = np.concatenate([plan["local"][edges[:, 1]],
                            plan["out_of_plane"][edges[:, 1]]], axis=1)

    pts = np.zeros((total, 3), dtype=np.float64)
    pts[starts] = a_loc
    pts[starts + poly_n - 1] = b_loc

    if int(inner_n.sum()) > 0:
        rows = np.repeat(np.arange(e), inner_n)
        step = _ragged_arange(inner_n) + 1
        # Reversed routes are walked from the far end, giving 0 -> 1 order.
        pick = np.where(reverse[rows], lengths[rows] - 2 - step + 1, step)
        cells = routes[rows, pick]
        ix = np.stack(np.unravel_index(cells, plan["shape"]), axis=1)[:, ::-1]
        center = plan["lo"] + (ix + 0.5) * plan["cell"]
        slot = starts[rows] + step
        pts[slot, :dims] = center

    # In-plane part only. The out-of-plane coordinate is derived from this, so
    # including it makes the parameterization depend on its own answer.
    seg = np.zeros(total, dtype=np.float64)
    seg[1:] = np.linalg.norm(pts[1:, :dims] - pts[:-1, :dims], axis=1)
    seg[starts] = 0.0
    cum = np.cumsum(seg)
    cum -= np.repeat(cum[starts], poly_n)
    span = np.repeat(np.maximum(cum[starts + poly_n - 1], 1e-12), poly_n)
    s = cum / span

    if dims == 2:
        wa = np.repeat(a_loc[:, 2], poly_n)
        wb = np.repeat(b_loc[:, 2], poly_n)
        pts[:, 2] = (1.0 - s) * wa + s * wb

    world = plan["origin"] + pts @ plan["basis"]

    # One monotone key over every polyline, so the resampling is a single
    # searchsorted; 0.999 keeps an edge's last key below the next edge's first.
    row = np.repeat(np.arange(e), poly_n)
    key = row + 0.999 * s

    k = np.clip(poly_n, 3, MAX_CTRL)
    krow = np.repeat(np.arange(e), k)
    kslot = _ragged_arange(k)
    t = kslot / np.maximum(np.repeat(k, k) - 1, 1)
    q = krow + 0.999 * t

    lo_i = np.repeat(starts, k)
    hi_i = np.repeat(starts + poly_n - 1, k)
    j = np.clip(np.searchsorted(key, q, side='left') - 1, lo_i, hi_i - 1)
    denom = np.maximum(key[j + 1] - key[j], 1e-12)
    frac = np.clip((q - key[j]) / denom, 0.0, 1.0)[:, None]
    sampled = world[j] + frac * (world[j + 1] - world[j])

    kmax = int(k.max())
    ctrl = np.zeros((e, kmax, 3), dtype=np.float32)
    ctrl[krow, kslot] = sampled
    # Exactly the node positions: the beta slider pivots on the curve's start.
    ctrl[:, 0] = coords[edges[:, 0]]
    ctrl[np.arange(e), k - 1] = coords[edges[:, 1]]
    # Padded by repeating the last real point; zeros would look like a bug.
    tail = np.arange(kmax)[None, :] >= k[:, None]
    last = np.broadcast_to(ctrl[np.arange(e), k - 1][:, None, :],
                           (e, kmax, 3))
    ctrl[tail] = last[tail]
    return ctrl, k.astype(np.int32)
