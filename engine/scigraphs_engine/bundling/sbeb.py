# Skeleton-based edge bundling, image-based (Ersoy et al. 2011): the numpy half.
#
# The method draws the edges into an image, thresholds the density, dilates the
# drawn set into a shape, skeletonizes it, and pulls every subdivision point
# toward the skeleton. Cost is O(image pixels) per iteration, not O(edges).
#
# The distance and feature transforms and the skeleton detector are a jump flood
# needing a GPU texture for their state, so those and the iteration loop live on
# the add-on side.

import numpy as np

# Radius of the shape around the drawn edges, as a fraction of the raster side.
# Too small and every edge keeps a private tube whose skeleton is its own
# centerline. Drop in raster texels covered, ten-community ring of 1711 edges:
#     0.02 ->  2%   0.05 ->  8%   0.10 -> 26%
#     0.15 -> 42%   0.20 -> 49% but unstable, 30% one step away
# 0.15 is the largest value with a flat neighborhood; past it the clusters merge
# into blobs and the response starts depending on the graph.
DILATE_FRAC = 0.15

# Derived, not chosen: the shape is the drawing grown by exactly DILATE_FRAC.
# Anything less clips it, the skeleton follows the margin, and a bundle heads for
# a corner it is not in. Same 1711 edges: 29% ink at margin 0.06, 42% at 0.15.
MARGIN = DILATE_FRAC

# One texel per control point, so E*K tiled TEX_ROW wide must stay inside
# max_texture_size; past a few dozen the polygon is the curve anyway.
MAX_K = 64

# Edges rasterized per cluster per iteration, at most: everything else is
# O(res^2) while rasterization is O(total ink). Sampling is safe because the
# threshold is a fraction of peak density, so scaling every count alike leaves
# the set where it was. Cap of 8 at half a million edges: 9.87 to 10.26 ms.
RASTER_MAX_EDGES = 8192

# Fit on a sample, assign everything: O(E), not O(E * k * rounds).
KMEANS_ROUNDS = 12
KMEANS_SAMPLE = 20000
KMEANS_SEED = 0


DEFAULTS = {
    "sbeb_resolution": 512,
    "sbeb_iterations": 5,
    "sbeb_clusters": 16,
    "sbeb_threshold": 0.1,
    "sbeb_attraction": 0.6,
    "sbeb_smooth": 2,
    "sbeb_recluster": False,
}


def settings(params):
    """Return the mode's parameters; no ``sbeb_*`` key is registered yet."""
    s = {k[5:]: params.get(k, v) for k, v in DEFAULTS.items()}
    s["resolution"] = max(64, int(s["resolution"]))
    s["iterations"] = max(0, int(s["iterations"]))
    s["clusters"] = max(1, int(s["clusters"]))
    s["threshold"] = float(np.clip(float(s["threshold"]), 0.0, 1.0))
    s["attraction"] = float(np.clip(float(s["attraction"]), 0.0, 1.0))
    s["smooth"] = max(0, int(s["smooth"]))
    s["recluster"] = bool(s["recluster"])
    s["k"] = int(np.clip(int(params.get("segments", 24)) + 1, 3, MAX_K))
    return s


def frame(coords, ctx=None):
    """Return (origin, e1, e2, e3) spanning the graph's dominant plane, or None
    if the SVD fails. ``ctx["bundle_axes"]`` wins when given, so a caller who
    knows where the camera looks can bundle in the view plane. Axis signs are
    pinned: an SVD may negate a column, and two loads would then differ."""
    origin = np.asarray(coords, dtype=np.float64).mean(axis=0)
    axes = (ctx or {}).get("bundle_axes")
    given = axes is not None
    if given:
        e1 = np.asarray(axes[0], dtype=np.float64)
        e2 = np.asarray(axes[1], dtype=np.float64)
    else:
        centered = np.asarray(coords, dtype=np.float64) - origin
        # SVD, not an eigendecomposition of the covariance: same answer without
        # squaring the condition number of a nearly degenerate thin axis.
        try:
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        e1, e2 = vt[0], vt[1] if vt.shape[0] > 1 else np.array([0.0, 1.0, 0.0])

    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = e2 - e1 * float(e1 @ e2)
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    if not given:
        # SVD axes only; pinning a caller's plane would mirror its raster.
        for e in (e1, e2):
            if e[int(np.argmax(np.abs(e)))] < 0.0:
                e *= -1.0
    e3 = np.cross(e1, e2)
    return origin, e1, e2, e3


def _image_map(uv):
    """(lo, span). Square: the distance transform is Euclidean in texels."""
    lo = uv.min(axis=0)
    hi = uv.max(axis=0)
    span = float(np.max(hi - lo)) if uv.size else 1.0
    span = max(span, 1e-9) / max(1.0 - 2.0 * MARGIN, 1e-6)
    center = 0.5 * (lo + hi)
    return center - 0.5 * span, span


def _setup(coords, edges, params, ctx):
    """Frame, image map, straight polygons, labels. Shared by both paths, so the
    GPU-against-numpy comparison does not cover it."""
    s = settings(params)
    fr = frame(coords, ctx)
    if fr is None:
        return None
    origin, e1, e2, e3 = fr

    c = np.asarray(coords, dtype=np.float64) - origin
    uv_nodes = np.stack([c @ e1, c @ e2], axis=1)
    lo, span = _image_map(uv_nodes)

    k = s["k"]
    t = np.linspace(0.0, 1.0, k, dtype=np.float64)[None, :, None]
    a = uv_nodes[edges[:, 0]][:, None, :]
    b = uv_nodes[edges[:, 1]][:, None, :]
    # The out-of-plane coordinate never reaches the image; see ``_lift``.
    pts = ((a + (b - a) * t) - lo) / span

    labels = cluster_edges(coords, edges, s["clusters"])
    return {
        "s": s, "origin": origin, "e1": e1, "e2": e2, "e3": e3,
        "lo": lo, "span": span,
        "pts": np.ascontiguousarray(pts, dtype=np.float32),
        "labels": labels,
    }


def _lift(pts, setup, coords, edges):
    """Image coordinates back to (E, K, 3) world control polygons: free in the
    plane, keeping the straight edge's coordinate in the third direction."""
    origin, e1, e2, e3 = (setup["origin"], setup["e1"], setup["e2"],
                          setup["e3"])
    k = setup["s"]["k"]
    plane = pts.astype(np.float64) * setup["span"] + setup["lo"]

    c = np.asarray(coords, dtype=np.float64) - origin
    n_nodes = c @ e3
    t = np.linspace(0.0, 1.0, k, dtype=np.float64)[None, :]
    na = n_nodes[edges[:, 0]][:, None]
    nb = n_nodes[edges[:, 1]][:, None]
    n = na + (nb - na) * t

    ctrl = (origin[None, None, :]
            + plane[..., 0:1] * e1[None, None, :]
            + plane[..., 1:2] * e2[None, None, :]
            + n[..., None] * e3[None, None, :])
    # No kernel writes k=0 or k=K-1, so this removes float noise, not movement.
    ctrl[:, 0, :] = coords[edges[:, 0]]
    ctrl[:, -1, :] = coords[edges[:, 1]]
    return np.ascontiguousarray(ctrl, dtype=np.float32)


def cluster_edges(coords, edges, k, positions=None):
    """k-means over the 6D vector (ax, ay, az, bx, by, bz). Deterministic, and
    the slow step, so re-clustering every iteration is off by default. Endpoints
    go into a canonical order first, or two copies of one edge land in different
    clusters. Centroids are fitted on a bounded sample, which keeps an edge-count
    term out of the one method that has none. ``positions`` overrides the vector,
    and is how re-clustering sees the bundling so far."""
    e = np.asarray(edges, dtype=np.int64)
    swap = e[:, 0] > e[:, 1]
    if positions is None:
        a = np.asarray(coords, dtype=np.float64)[e[:, 0]]
        b = np.asarray(coords, dtype=np.float64)[e[:, 1]]
        vec = np.concatenate(_oriented(a, b, swap), axis=1)
    else:
        p = np.asarray(positions, dtype=np.float64)
        # (start, midpoint, end); only the ends decide the orientation.
        half = p.shape[1] // 3
        lo, hi = _oriented(p[:, :half], p[:, 2 * half:], swap)
        vec = np.concatenate([lo, p[:, half:2 * half], hi], axis=1)

    n = vec.shape[0]
    k = int(min(max(1, k), n))
    if k <= 1:
        return np.zeros(n, dtype=np.int64)

    # One scalar, not per column: per-axis variances do not survive a rotation,
    # and standardizing each column bundled a rotated graph 38% of its own
    # extent away. An isotropic scale still ignores the layout's units.
    mu = vec.mean(axis=0)
    sd = max(float(np.sqrt(np.mean(np.sum((vec - mu) ** 2, axis=1)))), 1e-9)
    z = (vec - mu) / sd

    rng = np.random.default_rng(KMEANS_SEED)
    if n > KMEANS_SAMPLE:
        # A stride, not a random draw: no dependence on the RNG's stream.
        fit = z[::max(1, n // KMEANS_SAMPLE)][:KMEANS_SAMPLE]
    else:
        fit = z

    centers = _kmeans_pp(fit, k, rng)
    for _ in range(KMEANS_ROUNDS):
        lab = _assign(fit, centers)
        new = centers.copy()
        for c in range(k):
            m = lab == c
            if m.any():
                new[c] = fit[m].mean(axis=0)
            else:
                # Re-seeded at the worst-served point: the count is user-facing.
                d = np.linalg.norm(fit - centers[lab], axis=1)
                new[c] = fit[int(np.argmax(d))]
        if np.allclose(new, centers):
            centers = new
            break
        centers = new
    return _assign(z, centers)


def _oriented(a, b, swap):
    """Reorder ``(a, b)`` per row so the lower-numbered node comes first. By
    index, not coordinates, which turn: a rotated graph bundled 29% away."""
    return (np.where(swap[:, None], b, a), np.where(swap[:, None], a, b))


def _kmeans_pp(x, k, rng):
    """k-means++ seeding; deterministic given the seeded generator."""
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[0]
    d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(d2.sum())
        if total <= 0.0:
            centers[i] = x[i % x.shape[0]]
        else:
            centers[i] = x[int(np.searchsorted(np.cumsum(d2),
                                               rng.random() * total))]
        d2 = np.minimum(d2, np.sum((x - centers[i]) ** 2, axis=1))
    return centers


def _assign(x, centers):
    """Nearest center per row, one center at a time: a broadcast allocates an
    (E, k) float64 temporary, 64 MB at 500k edges and 16 clusters."""
    best = np.zeros(x.shape[0], dtype=np.int64)
    bd = np.full(x.shape[0], np.inf)
    for c in range(centers.shape[0]):
        d = np.sum((x - centers[c]) ** 2, axis=1)
        take = d < bd
        bd[take] = d[take]
        best[take] = c
    return best


def cluster_ranges(labels, k):
    """Return (order, starts, counts). Each cluster becomes one contiguous run,
    so dispatching it is an index and a count, not an indirection."""
    order = np.argsort(labels, kind="stable")
    counts = np.bincount(labels, minlength=k).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return order, starts, counts


def _sampled(count):
    stride = max(1, int(np.ceil(count / RASTER_MAX_EDGES)))
    return int(count // stride) if stride > 1 else int(count), stride


def raster_numpy(pts, res, edge0, n_draw, stride, k):
    """One cluster's density image, by the same DDA the kernel walks."""
    dens = np.zeros(res * res, dtype=np.int64)
    if n_draw <= 0 or k < 2:
        return dens.reshape(res, res)
    e = edge0 + np.arange(n_draw, dtype=np.int64) * stride
    idx = (e[:, None] * k + np.arange(k - 1, dtype=np.int64)[None, :]).ravel()
    a = pts[idx] * res
    b = pts[idx + 1] * res

    d = np.abs(b - a)
    n = np.clip(np.max(d, axis=1).astype(np.int64) + 1, 1, 4 * res)
    total = int((n + 1).sum())
    seg = np.repeat(np.arange(n.size, dtype=np.int64), n + 1)
    ends = np.cumsum(n + 1)
    step = np.arange(total, dtype=np.int64) - np.repeat(ends - (n + 1), n + 1)
    t = (step / n[seg])[:, None]
    p = a[seg] + (b[seg] - a[seg]) * t
    q = np.clip(p.astype(np.int64), 0, res - 1)
    np.add.at(dens, q[:, 1] * res + q[:, 0], 1)
    return dens.reshape(res, res)
