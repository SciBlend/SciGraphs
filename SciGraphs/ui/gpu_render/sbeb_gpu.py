# SciGraphs GPU render - skeleton-based bundling, image-based (Ersoy et al. 2011)
#
# Cost is O(image pixels) per iteration, not O(edges): an edge is drawn into an
# image and never read again. Threshold, dilate, distance and feature transform
# and skeletonize live in jfa.py, which keeps its state in RGBA32F textures
# because Blender's gpu module has no storage buffers. The rasterizer kernel,
# the attraction, the smoothing and both iteration loops are here; the
# projection, the k-means, the numpy rasterizer and the lift to world space are
# in core.render.bundling.sbeb. Relaxation is not implemented and must not be:
# the draw shader already interpolates toward the straight line.

import time

import numpy as np

import gpu

from . import bundle_gpu, jfa
from .glsl_source import glsl

from ...core.render.bundling import sbeb as _sbeb  # noqa: E402,F401
from ...core.render.bundling.sbeb import (  # noqa: E402,F401
    DEFAULTS, DILATE_FRAC, KMEANS_ROUNDS, KMEANS_SAMPLE, KMEANS_SEED, MARGIN,
    MAX_K, RASTER_MAX_EDGES, _assign, _image_map, _kmeans_pp, _lift, _oriented,
    _sampled, _setup, cluster_edges, cluster_ranges, frame, raster_numpy,
    settings,
)

TEX_ROW = bundle_gpu.TEX_ROW

# As in sim_gpu: a wavefront on AMD, two warps on NVIDIA, memory-bound.
GROUP = 64

# Skeleton detector constants; see jfa.py. Not exposed, because a ratio test
# does not need retuning per graph.
SKEL_TAU = 1.0
SKEL_MIN_DIST = 2.0

# 1.0 converges on the straight line and undoes the bundling. Half a move takes
# the corners off in place.
SMOOTH_LAMBDA = 0.5

# Diagnostics from the last ``produce``. Nothing in the draw path reads it.
LAST = {}


def check(params, edges, ctx):
    """Return (ok, reason) for routing these edges through an image."""
    if params.get("style_type") != 'SBEB':
        return False, "not skeleton-based bundling"
    s = settings(params)
    rows = int(np.ceil(edges.shape[0] * s["k"] / TEX_ROW))
    if rows > 16384:
        return False, ("%d edges at %d points each needs a taller control "
                       "texture than the backend allows"
                       % (edges.shape[0], s["k"]))
    return True, ""


def produce(coords, edges, params, ctx):
    """Control polygons bundled toward each cluster's skeleton."""
    try:
        if jfa.available() and _available():
            out = _produce_gpu(coords, edges, params, ctx)
            if out is not None:
                return out
    except Exception:  # noqa: BLE001 - a driver refusing is a fallback, not a crash
        pass
    try:
        return _produce_numpy(coords, edges, params, ctx)
    except Exception:  # noqa: BLE001
        return None


_TEXEL = f"""
ivec2 scig_texel(int i) {{ return ivec2(i % {TEX_ROW}, i / {TEX_ROW}); }}
"""

_RASTER_SRC = _TEXEL + glsl("sbeb_raster")

_PEAK_SRC = glsl("sbeb_peak")

_ATTRACT_SRC = _TEXEL + glsl("sbeb_attract")

_SMOOTH_SRC = _TEXEL + glsl("sbeb_smooth")


_SHADERS = {}


def _shader(name):
    if name in _SHADERS:
        return _SHADERS[name]
    try:
        if gpu.platform.backend_type_get() == 'NONE':
            return None
    except Exception:  # noqa: BLE001 - raises before gpu.init()
        return None
    try:
        info = gpu.types.GPUShaderCreateInfo()
        if name == "raster":
            info.local_group_size(GROUP)
            info.image(0, 'RGBA32F', 'FLOAT_2D', "pts", qualifiers={'READ'})
            info.image(1, 'R32UI', 'UINT_2D', "dens",
                       qualifiers={'READ', 'WRITE'})
            for u in ("u_nseg", "u_km1", "u_k", "u_edge0", "u_estride",
                      "u_res", "u_max_steps"):
                info.push_constant('INT', u)
            src = _RASTER_SRC
        elif name == "peak":
            info.local_group_size(jfa.GROUP, jfa.GROUP)
            info.image(0, 'R32UI', 'UINT_2D', "dens", qualifiers={'READ'})
            info.image(1, 'R32UI', 'UINT_2D', "peak",
                       qualifiers={'READ', 'WRITE'})
            info.push_constant('INT', "u_res")
            src = _PEAK_SRC
        elif name == "attract":
            info.local_group_size(GROUP)
            info.image(0, 'RGBA32F', 'FLOAT_2D', "pts",
                       qualifiers={'READ', 'WRITE'})
            info.image(1, jfa.FEATURE_FORMAT, 'FLOAT_2D', "skel_ft",
                       qualifiers={'READ'})
            for u in ("u_np", "u_p0", "u_k", "u_res"):
                info.push_constant('INT', u)
            info.push_constant('FLOAT', "u_alpha")
            src = _ATTRACT_SRC
        elif name == "smooth":
            info.local_group_size(GROUP)
            info.image(0, 'RGBA32F', 'FLOAT_2D', "src", qualifiers={'READ'})
            info.image(1, 'RGBA32F', 'FLOAT_2D', "dst", qualifiers={'WRITE'})
            info.push_constant('INT', "u_n")
            info.push_constant('INT', "u_k")
            info.push_constant('FLOAT', "u_lambda")
            src = _SMOOTH_SRC
        else:
            return None
        info.compute_source(src)
        _SHADERS[name] = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001 - unsupported backend
        _SHADERS[name] = None
    return _SHADERS[name]


def _available():
    return all(_shader(n) is not None
               for n in ("raster", "peak", "attract", "smooth"))


def available():
    """Whether the GPU path can run here at all."""
    return jfa.available() and _available()


def _point_texture(pts):
    """Upload (E, K, 2) image coordinates as one RGBA32F texel per point."""
    flat = pts.reshape(-1, 2)
    n = flat.shape[0]
    rows = max(1, int(np.ceil(n / TEX_ROW)))
    padded = np.zeros((rows * TEX_ROW, 4), dtype=np.float32)
    padded[:n, :2] = flat
    return gpu.types.GPUTexture(
        (TEX_ROW, rows), format='RGBA32F',
        data=gpu.types.Buffer('FLOAT', padded.size, padded.ravel())), rows


def _blank_points(rows):
    tex = gpu.types.GPUTexture((TEX_ROW, rows), format='RGBA32F')
    tex.clear(format='FLOAT', value=(0.0, 0.0, 0.0, 0.0))
    return tex


def _produce_gpu(coords, edges, params, ctx):
    t_start = time.perf_counter()

    setup = _setup(coords, edges, params, ctx)
    if setup is None:
        return None
    s = setup["s"]
    res, k = s["resolution"], s["k"]
    e_count = int(edges.shape[0])
    labels = setup["labels"]
    order, starts, counts = cluster_ranges(labels, s["clusters"])

    pts_tex, rows = _point_texture(setup["pts"][order])
    pts_b = _blank_points(rows)
    n_points = e_count * k

    dens = jfa.mask_texture(res)
    peak = gpu.types.GPUTexture((1, 1), format='R32UI')
    omega = jfa.mask_texture(res)
    bnd = jfa.mask_texture(res)
    skel = jfa.mask_texture(res)
    ft_a = jfa.feature_texture(res)
    ft_b = jfa.feature_texture(res)

    raster = _shader("raster")
    peak_sh = _shader("peak")
    attract = _shader("attract")
    smooth = _shader("smooth")
    radius = max(2.0, DILATE_FRAC * res)
    img_groups = (res + jfa.GROUP - 1) // jfa.GROUP
    t_setup = time.perf_counter()

    for it in range(s["iterations"]):
        if s["recluster"] and it > 0:
            # The one O(E) readback in the loop, and why recluster is off by
            # default: it has to see where the curves got to, so they come back.
            cur = _read_points(pts_tex, n_points).reshape(e_count, k, 2)
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            cur = cur[inv]
            vec = np.concatenate([cur[:, 0], cur[:, k // 2], cur[:, -1]], axis=1)
            labels = cluster_edges(coords, edges, s["clusters"], positions=vec)
            order, starts, counts = cluster_ranges(labels, s["clusters"])
            pts_tex, rows = _point_texture(cur[order])
            pts_b = _blank_points(rows)

        for c in range(s["clusters"]):
            n_edges = int(counts[c])
            if n_edges == 0:
                continue
            n_draw, stride = _sampled(n_edges)
            n_seg = n_draw * (k - 1)

            dens.clear(format='UINT', value=(0,))
            peak.clear(format='UINT', value=(0,))
            raster.bind()
            raster.image('pts', pts_tex)
            raster.image('dens', dens)
            raster.uniform_int("u_nseg", n_seg)
            raster.uniform_int("u_km1", k - 1)
            raster.uniform_int("u_k", k)
            raster.uniform_int("u_edge0", int(starts[c]))
            raster.uniform_int("u_estride", stride)
            raster.uniform_int("u_res", res)
            raster.uniform_int("u_max_steps", 4 * res)
            gpu.compute.dispatch(raster, (n_seg + GROUP - 1) // GROUP, 1, 1)

            peak_sh.bind()
            peak_sh.image('dens', dens)
            peak_sh.image('peak', peak)
            peak_sh.uniform_int("u_res", res)
            gpu.compute.dispatch(peak_sh, img_groups, img_groups, 1)

            # The drawn set, thresholded, then dilated. The flood is capped
            # because only the field within the radius decides the level set.
            jfa.seed(dens, peak, ft_a, res, fraction=s["threshold"], floor=1)
            ft = jfa.flood(ft_a, ft_b, res, max_step=int(np.ceil(radius)))
            jfa.level_set(ft, omega, res, radius)

            jfa.boundary(omega, bnd, res)
            jfa.seed(bnd, peak, ft_a, res, fraction=0.0, floor=1)
            ft = jfa.flood(ft_a, ft_b, res)
            jfa.skeleton(ft, omega, skel, res, tau=SKEL_TAU,
                         min_dist=SKEL_MIN_DIST)

            # The feature transform to the skeleton: the whole per-edge cost.
            jfa.seed(skel, peak, ft_a, res, fraction=0.0, floor=1)
            ft = jfa.flood(ft_a, ft_b, res)

            n_pts = n_edges * k
            attract.bind()
            attract.image('pts', pts_tex)
            attract.image('skel_ft', ft)
            attract.uniform_int("u_np", n_pts)
            attract.uniform_int("u_p0", int(starts[c]) * k)
            attract.uniform_int("u_k", k)
            attract.uniform_int("u_res", res)
            attract.uniform_float("u_alpha", s["attraction"])
            gpu.compute.dispatch(attract, (n_pts + GROUP - 1) // GROUP, 1, 1)

        for _ in range(s["smooth"]):
            smooth.bind()
            smooth.image('src', pts_tex)
            smooth.image('dst', pts_b)
            smooth.uniform_int("u_n", n_points)
            smooth.uniform_int("u_k", k)
            smooth.uniform_float("u_lambda", SMOOTH_LAMBDA)
            gpu.compute.dispatch(smooth, (n_points + GROUP - 1) // GROUP, 1, 1)
            pts_tex, pts_b = pts_b, pts_tex

    # Dispatches are asynchronous, so timing the loop without this would measure
    # Python. The full readback below waits anyway; one texel first moves that
    # wait where it belongs, for one round trip (0.12 ms).
    np.asarray(pts_tex.read()[0][0])
    t_loop = time.perf_counter()

    out = _read_points(pts_tex, n_points).reshape(e_count, k, 2)
    t_read = time.perf_counter()

    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    ctrl = _lift(out[inv], setup, coords, edges)
    counts_out = np.full(e_count, k, dtype=np.int32)

    LAST.clear()
    LAST.update({
        "path": "gpu", "edges": e_count, "points": n_points,
        "clusters": int(s["clusters"]), "resolution": res, "k": k,
        "iterations": int(s["iterations"]),
        # Only loop_ms is the mode's own work; the rest is O(E) transfer.
        "setup_ms": (t_setup - t_start) * 1e3,
        "loop_ms": (t_loop - t_setup) * 1e3,
        "read_ms": (t_read - t_loop) * 1e3,
        "lift_ms": (time.perf_counter() - t_read) * 1e3,
        "total_ms": (time.perf_counter() - t_start) * 1e3,
        "rasterized": int(sum(_sampled(int(c))[0] for c in counts)),
    })
    return ctrl, counts_out


def _read_points(tex, n):
    """(n, 2) image coordinates; the only O(E) transfer in the loop."""
    return np.asarray(tex.read()).reshape(-1, 4)[:n, :2].copy()


# numpy: term for term the same pipeline, same order, same constants, so the GPU
# path can be checked against it. It calls jfa.py's numpy transforms, which is
# why the loop sits here rather than beside the rasterizer it shares.

def _produce_numpy(coords, edges, params, ctx):
    t_start = time.perf_counter()

    setup = _setup(coords, edges, params, ctx)
    if setup is None:
        return None
    s = setup["s"]
    res, k = s["resolution"], s["k"]
    e_count = int(edges.shape[0])
    labels = setup["labels"]
    order, starts, counts = cluster_ranges(labels, s["clusters"])
    pts = setup["pts"][order].reshape(-1, 2).astype(np.float64)

    radius = max(2.0, DILATE_FRAC * res)
    kk = np.arange(e_count * k) % k
    interior = (kk != 0) & (kk != k - 1)
    t_setup = time.perf_counter()

    for it in range(s["iterations"]):
        if s["recluster"] and it > 0:
            cur = pts.reshape(e_count, k, 2)
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)
            cur = cur[inv]
            vec = np.concatenate([cur[:, 0], cur[:, k // 2], cur[:, -1]], axis=1)
            labels = cluster_edges(coords, edges, s["clusters"], positions=vec)
            order, starts, counts = cluster_ranges(labels, s["clusters"])
            pts = cur[order].reshape(-1, 2)

        for c in range(s["clusters"]):
            n_edges = int(counts[c])
            if n_edges == 0:
                continue
            n_draw, stride = _sampled(n_edges)
            dens = raster_numpy(pts, res, int(starts[c]), n_draw, stride, k)
            peak = int(dens.max())

            ft = jfa.flood_from_mask(dens, max_step=int(np.ceil(radius)),
                                     fraction=s["threshold"], floor=1,
                                     peak=peak)
            omega = jfa.level_set_numpy(ft, radius)
            bnd = jfa.boundary_numpy(omega)
            ft = jfa.flood_from_mask(bnd, fraction=0.0, floor=1, peak=peak)
            skel = jfa.skeleton_numpy(ft, omega, tau=SKEL_TAU,
                                      min_dist=SKEL_MIN_DIST)
            ft = jfa.flood_from_mask(skel, fraction=0.0, floor=1, peak=peak)

            lo = int(starts[c]) * k
            hi = lo + n_edges * k
            sl = slice(lo, hi)
            q = np.clip((pts[sl] * res).astype(np.int64), 0, res - 1)
            f = ft[q[:, 1], q[:, 0]]
            move = (f[:, 2] > 0.5) & interior[sl]
            target = (f[:, :2] + 0.5) / res
            pts[sl] = np.where(move[:, None],
                               pts[sl] + s["attraction"] * (target - pts[sl]),
                               pts[sl])

        for _ in range(s["smooth"]):
            grid = pts.reshape(e_count, k, 2)
            mid = 0.5 * (grid[:, :-2] + grid[:, 2:])
            grid[:, 1:-1] += SMOOTH_LAMBDA * (mid - grid[:, 1:-1])
            pts = grid.reshape(-1, 2)

    t_loop = time.perf_counter()
    inv = np.empty_like(order)
    inv[order] = np.arange(order.size)
    ctrl = _lift(pts.reshape(e_count, k, 2)[inv], setup, coords, edges)

    LAST.clear()
    LAST.update({
        "path": "numpy", "edges": e_count, "points": e_count * k,
        "clusters": int(s["clusters"]), "resolution": res, "k": k,
        "iterations": int(s["iterations"]),
        "setup_ms": (t_setup - t_start) * 1e3,
        "loop_ms": (t_loop - t_setup) * 1e3,
        # No transfer here; the key is kept so both paths report one shape.
        "read_ms": 0.0,
        "lift_ms": (time.perf_counter() - t_loop) * 1e3,
        "total_ms": (time.perf_counter() - t_start) * 1e3,
        "rasterized": int(sum(_sampled(int(c))[0] for c in counts)),
    })
    return ctrl, np.full(e_count, k, dtype=np.int32)


# Only the raster and the projection are 2D. Jump flooding is not, which is why
# jfa.py's feature texture reserves .w for a third coordinate. A volumetric
# version was dropped: the medial set of a solid is a surface, not a curve, so
# there is no 1D attractor to bundle toward, and 512^3 is 134 million texels
# against 512^2's 262144, 2 GB at RGBA32F for one ping-pong pair.
