# Force-directed edge bundling, Holten and van Wijk 2009: each edge becomes a
# polyline, pulled toward compatible neighbors and sprung along its own length.
#
# Interactions truncate at a radius instead of summing over all pairs: memory
# O(E * P), time O(E * P * k * I). Compatibility is recomputed per pair rather
# than stored, since an (E, E) matrix is 134 MB in float64 at 4096 edges.
#
# No relaxation pass here, and there must not be one: the add-on's draw shader
# already interpolates toward the straight line, and beta is that post-process.

from itertools import product

import numpy as np

# Resolution ceiling per axis, and the bucket budget the cell loop coarsens
# down to. Cells are at least one radius wide, so a cell plus its 26 neighbors
# hold the ball exactly.
RES_MAX = 64
MAX_BUCKETS = 65536

# Corresponding points one source point takes per bucket, over the 54 visited
# (27 cells, two point indices each). A cap on the running total instead would
# spend the budget in the first cell and pull every bundle toward -x-y-z.
# Bucketing by cell alone left 49,662 points in the busiest cell against 1,910
# in the busiest bucket, on 20k edges in a two-cluster graph.
BUCKET_CAP = 64

# The paper's initial step, a fraction of the way to the target rather than a
# distance, so the schedule is scale-free.
STEP0 = 0.6

# 0.5 to match the vectorized FDEB in edge_styles.
SPRING_GAIN = 0.5

# Bundled points converge, so the attraction is evaluated at its singularity.
SOFTEN_FRAC = 0.01

DEFAULT_CYCLES = 6
DEFAULT_RADIUS = 0.05
DEFAULT_VISIBILITY = True


def schedule(params):
    """The paper's schedule as [(P, S, I)] per cycle: subdivision points, step,
    iterations. P is capped at the count the curve is drawn with."""
    cycles = max(1, int(params.get("fdeb_cycles", DEFAULT_CYCLES)))
    p_cap = max(1, int(params["segments"]))
    p, step = 1, STEP0
    iters = max(1, int(params["bundle_iterations"]))
    out = []
    for _ in range(cycles):
        out.append((min(p, p_cap), step, iters))
        p *= 2
        step *= 0.5
        # I drops by a third per cycle; later cycles only refine.
        iters = max(1, int(round(iters * 2.0 / 3.0)))
    return out


def frames(coords, edges):
    """Per-edge start, end, unit direction, length and midpoint. Carrying the
    direction rather than the far endpoint saves a normalize per pair."""
    coords = np.asarray(coords, dtype=np.float32)
    a = coords[edges[:, 0]]
    b = coords[edges[:, 1]]
    d = b - a
    length = np.linalg.norm(d, axis=1).astype(np.float32)
    unit = (d / np.maximum(length, np.float32(1e-12))[:, None]).astype(np.float32)
    return {"a": a, "b": b, "u": unit, "len": length,
            "mid": (np.float32(0.5) * (a + b)).astype(np.float32)}


def _visibility(fr, i, j):
    """How much of P lies in Q's shadow: Q's endpoints project onto the line
    through P as an interval, and V is how far P's midpoint is outside it."""
    ai, ui, li = fr["a"][i], fr["u"][i], fr["len"][i]
    aj, uj, lj = fr["a"][j], fr["u"][j], fr["len"][j]
    t0 = ((aj - ai) * ui).sum(axis=1)
    t1 = ((aj + uj * lj[:, None] - ai) * ui).sum(axis=1)
    mid = np.float32(0.5) * (t0 + t1)
    width = np.abs(t1 - t0)
    # A perpendicular Q has zero width; the guard sends it to zero.
    return np.maximum(np.float32(1.0)
                      - np.float32(2.0) * np.abs(li * np.float32(0.5) - mid)
                      / np.maximum(width, np.float32(1e-12)),
                      np.float32(0.0))


def compatibility(fr, i, j, visibility=True):
    """Ce = Ca * Cs * Cp * Cv over the index pairs (i, j), a list rather than
    the (E, E) grid.

    Cv is on here, where the vectorized FDEB in edge_styles sets it to 1: it
    stops two edges from bundling when one lies off to the side of the other.
    Being a fourth factor, it makes a threshold tuned against the three-term
    product stricter: 0.05 in the paper, ``bundle_threshold`` 0.6 here.
    """
    ui, uj = fr["u"][i], fr["u"][j]
    li, lj = fr["len"][i], fr["len"][j]
    angle = np.abs((ui * uj).sum(axis=1))
    l_avg = np.float32(0.5) * (li + lj)
    scale = np.float32(2.0) / (
        l_avg / np.maximum(np.minimum(li, lj), np.float32(1e-12))
        + np.maximum(li, lj) / np.maximum(l_avg, np.float32(1e-12)))
    gap = np.linalg.norm(fr["mid"][i] - fr["mid"][j], axis=1)
    pos = l_avg / (l_avg + gap)
    out = angle * scale * pos
    if visibility:
        out = out * np.minimum(_visibility(fr, i, j), _visibility(fr, j, i))
    return out.astype(np.float32)


def grid(coords, radius, k_max):
    """(lo, 1/width, resolution, cell count) for a grid over the node box.
    Built once: an update is a convex combination of nearby points, so no
    subdivision point leaves the box. The 27-cell walk is a superset of the
    ball, so the loop below may coarsen freely."""
    lo = np.asarray(coords, dtype=np.float64).min(axis=0)
    hi = np.asarray(coords, dtype=np.float64).max(axis=0)
    span = np.maximum(hi - lo, 1e-12)
    # Widest axis, not a per-axis clamp afterwards: that gives cells narrower
    # than R and a walk that misses covered neighbors.
    width = max(float(radius), float(span.max()) / RES_MAX)
    budget = max(1, MAX_BUCKETS // max(1, int(k_max)))
    res = np.maximum(1, np.floor(span / width).astype(np.int64))
    while int(np.prod(res)) > budget:
        width *= 1.25
        res = np.maximum(1, np.floor(span / width).astype(np.int64))
    return (lo.astype(np.float32), (res / span).astype(np.float32), res,
            int(np.prod(res)))


def _cell_of(pts, lo, inv, res):
    """(n, 3) clamped cell coords, as the add-on's GPU kernel computes them."""
    return np.clip(np.floor((pts - lo) * inv).astype(np.int64), 0, res - 1)


def _cell_id(c, res):
    return (c[..., 0] * res[1] + c[..., 1]) * res[2] + c[..., 2]


def _initial(fr, k):
    """Return (E, k, 3): each edge as a straight polyline of k even points."""
    t = np.linspace(0.0, 1.0, k, dtype=np.float32)[None, :, None]
    return (fr["a"][:, None, :] * (np.float32(1.0) - t)
            + fr["b"][:, None, :] * t).astype(np.float32)


def resample(pts, k_out):
    """Even spacing by arc length, not by inserting midpoints: the spring term
    is a Laplacian, so a straight line is a fixed point of it."""
    e, k_in, _ = pts.shape
    if k_in == k_out:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=1), axis=2).astype(np.float32)
    cum = np.concatenate([np.zeros((e, 1), np.float32),
                          np.cumsum(seg, axis=1)], axis=1).astype(np.float32)
    frac = np.linspace(0.0, 1.0, k_out, dtype=np.float32)
    target = (cum[:, -1:] * frac[None, :]).astype(np.float32)

    # Counting segments short of the target picks the one the GPU kernel takes.
    s = np.clip((cum[:, 1:, None] < target[:, None, :]).sum(axis=1),
                0, k_in - 2)
    lo = np.take_along_axis(cum, s, axis=1)
    length = np.take_along_axis(cum, s + 1, axis=1) - lo
    f = np.where(length > 1e-12, (target - lo) / np.maximum(length, 1e-12), 0.0)
    f = np.clip(f, 0.0, 1.0).astype(np.float32)[:, :, None]
    gather = np.repeat(s[:, :, None], 3, axis=2)
    a = np.take_along_axis(pts, gather, axis=1)
    b = np.take_along_axis(pts, gather + 1, axis=1)
    out = (a * (np.float32(1.0) - f) + b * f).astype(np.float32)
    out[:, 0] = pts[:, 0]
    out[:, -1] = pts[:, -1]
    return out


def _pairs_exact(flat, k, fr, radius):
    """Every interacting pair by brute force, O(E^2), to check the grid."""
    e_count = fr["len"].shape[0]
    r2 = np.float32(radius) ** 2
    dots = fr["u"] @ fr["u"].T
    ip, jp = [], []
    for p in range(1, k - 1):
        src = np.arange(e_count) * k + p
        for j in range(e_count):
            other = j * k + np.where(dots[:, j] >= 0.0, p, k - 1 - p)
            d = flat[other] - flat[src]
            keep = ((d * d).sum(axis=1) <= r2) & (np.arange(e_count) != j)
            ip.append(src[keep])
            jp.append(other[keep])
    if not ip:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(ip), np.concatenate(jp)


def _pairs_grid(flat, k, fr, lo, inv, res, ncells, radius, cap=BUCKET_CAP):
    """Every interacting pair from the grid, the set the add-on's compute
    kernel walks. "Corresponding" is p on an edge pointing the same way and
    k-1-p on one pointing the other way: angle compatibility is |cos|, and
    raw-index pairing opens an antiparallel bundle into an X. ``cap`` takes the
    lowest point indices; the kernel takes whatever its scatter put first."""
    n = flat.shape[0]
    idx = np.arange(n, dtype=np.int64)
    edge_of = idx // k
    slot = idx - edge_of * k

    point_cells = _cell_of(flat, lo, inv, res)
    key = _cell_id(point_cells, res) * k + slot
    nbuckets = ncells * k
    counts = np.bincount(key, minlength=nbuckets)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    order = np.argsort(key, kind="stable")

    src = np.nonzero((slot > 0) & (slot < k - 1))[0]
    if src.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    src_cell, src_slot = point_cells[src], slot[src]
    r2 = np.float32(radius) ** 2

    out_i, out_j = [], []
    # One pass over the 27 cells, in the order three nested loops walked them:
    # the pair order decides how the float32 sums accumulate downstream.
    for dx, dy, dz in product((-1, 0, 1), repeat=3):
        neighbor = src_cell + np.array([dx, dy, dz])
        inside = np.all((neighbor >= 0) & (neighbor < res), axis=1)
        cell = _cell_id(np.clip(neighbor, 0, res - 1), res)
        for flipped in (False, True):
            want = (k - 1 - src_slot) if flipped else src_slot
            bkey = cell * k + want
            cnt = np.where(inside, np.minimum(counts[bkey], cap), 0)
            total = int(cnt.sum())
            if total == 0:
                continue
            rep = np.repeat(np.arange(src.size), cnt)
            within = np.arange(total) - (np.cumsum(cnt) - cnt)[rep]
            jp = order[starts[bkey][rep] + within]
            ip = src[rep]
            ei, ej = edge_of[ip], edge_of[jp]
            dot = (fr["u"][ei] * fr["u"][ej]).sum(axis=1)
            keep = (dot < 0.0) if flipped else (dot >= 0.0)
            keep &= ei != ej
            d = flat[jp] - flat[ip]
            keep &= (d * d).sum(axis=1) <= r2
            if keep.any():
                out_i.append(ip[keep])
                out_j.append(jp[keep])
    if not out_i:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(out_i), np.concatenate(out_j)


def _produce_numpy(coords, edges, params, exact=False, cap=None):
    """Bundle on the CPU: the reference, and the fallback when compute is
    unavailable. ``exact`` swaps the grid for brute force."""
    cap = BUCKET_CAP if cap is None else int(cap)
    fr = frames(coords, edges)
    sched = schedule(params)
    strength = np.float32(np.clip(params["bundle_strength"], 0.0, 1.0))
    thresh = np.float32(params["bundle_threshold"])
    use_vis = bool(params.get("fdeb_visibility", DEFAULT_VISIBILITY))
    diag = float(np.linalg.norm(np.ptp(np.asarray(coords, np.float64), axis=0)))
    radius = float(params.get("fdeb_radius", DEFAULT_RADIUS)) * diag
    soften = np.float32((SOFTEN_FRAC * radius) ** 2)
    k_max = max(p for p, _s, _i in sched) + 2
    lo, inv, res, ncells = grid(coords, radius, k_max)

    e_count = int(edges.shape[0])
    pts = _initial(fr, sched[0][0] + 2)
    for p_count, step, iters in sched:
        k = p_count + 2
        pts = resample(pts, k)
        edge_of = np.repeat(np.arange(e_count, dtype=np.int64), k)
        n = e_count * k
        for _ in range(iters):
            flat = pts.reshape(-1, 3)
            if exact:
                ip, jp = _pairs_exact(flat, k, fr, radius)
            else:
                ip, jp = _pairs_grid(flat, k, fr, lo, inv, res, ncells, radius,
                                     cap)

            acc = np.zeros((n, 3), dtype=np.float32)
            wsum = np.zeros(n, dtype=np.float32)
            if ip.size:
                cm = compatibility(fr, edge_of[ip], edge_of[jp], use_vis)
                live = cm >= thresh
                ip, jp, cm = ip[live], jp[live], cm[live]
            if ip.size:
                d = flat[jp] - flat[ip]
                w = (cm / ((d * d).sum(axis=1) + soften)).astype(np.float32)
                for axis in range(3):
                    acc[:, axis] = np.bincount(ip, weights=w * d[:, axis],
                                               minlength=n)
                wsum = np.bincount(ip, weights=w, minlength=n).astype(np.float32)

            attract = np.zeros_like(acc)
            pulled = wsum > 0.0
            attract[pulled] = acc[pulled] / wsum[pulled, None]

            spring = np.zeros((e_count, k, 3), dtype=np.float32)
            spring[:, 1:-1] = (np.float32(0.5) * (pts[:, :-2] + pts[:, 2:])
                               - pts[:, 1:-1])
            pts = (flat + np.float32(step)
                   * (np.float32(SPRING_GAIN) * spring.reshape(-1, 3)
                      + strength * attract)).reshape(e_count, k, 3)
            # Pinned, not just unforced: an edge has to touch its two nodes.
            pts[:, 0] = fr["a"]
            pts[:, -1] = fr["b"]
    return (np.ascontiguousarray(pts, dtype=np.float32),
            np.full(e_count, pts.shape[1], np.int32))
