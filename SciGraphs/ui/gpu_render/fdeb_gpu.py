# Force-directed edge bundling (Holten and van Wijk), compute half. It mirrors
# scigraphs_engine.bundling.fdeb, the numpy half, term for term.
#
# Endpoints never move, so a compatibility entry is recomputed (~15 ops on data
# already in a texture) instead of read back from an (E, E) matrix. Cells are at
# least one radius wide on every axis, so a point's cell and its 26 neighbors
# hold the ball exactly. There must be no relaxation pass: bundle_gpu's draw
# shader interpolates toward the straight line, and beta is that post-process.

import numpy as np

import gpu

from . import bundle_gpu
from .glsl_source import glsl

from ...core.render.bundling import fdeb as _fdeb  # noqa: E402,F401
from ...core.render.bundling.fdeb import (  # noqa: E402,F401
    BUCKET_CAP, DEFAULT_CYCLES, DEFAULT_RADIUS, DEFAULT_VISIBILITY,
    MAX_BUCKETS, RES_MAX, SOFTEN_FRAC, SPRING_GAIN, STEP0,
    _cell_id, _cell_of, _initial, _pairs_exact, _pairs_grid, _produce_numpy,
    _visibility, compatibility, frames, grid, resample, schedule,
)

# Aliased, not copied: two texel-row lengths would drift apart.
TEX_ROW = bundle_gpu.TEX_ROW

# As in sim_gpu: a wavefront on AMD, two warps on NVIDIA, all memory-bound.
GROUP = 64

# Workgroups per dispatch dimension; the spec floor is 65535, so spill into y.
MAX_GROUPS = 32768

# Under BUCKET_CAP this path and the oracle agree to float noise; over it,
# _produce_gpu's ``report`` gives the occupancy.

# Set False to force the oracle. Only the test does this.
PREFER_GPU = True


def check(params, edges, ctx):
    """Return (ok, reason) for whether this mode has anything to do."""
    if params["style_type"] != 'FDEB':
        return False, "not force-directed bundling"
    if float(params.get("fdeb_radius", DEFAULT_RADIUS)) <= 0.0:
        return False, "a zero bundling radius leaves every edge alone"
    if int(params.get("fdeb_cycles", DEFAULT_CYCLES)) < 1:
        return False, "at least one cycle is needed to subdivide anything"
    return True, ""


def usable(params, edges):
    return bundle_gpu.usable('FDEB', params, edges, {})


_TEXEL = f"""
ivec2 scig_texel(int i) {{ return ivec2(i % {TEX_ROW}, i / {TEX_ROW}); }}

int scig_index()
{{
  return int(gl_GlobalInvocationID.y) * u_row + int(gl_GlobalInvocationID.x);
}}
"""

# Not in _TEXEL: resample has no grid, and unused push constants won't compile.
_CELL = glsl("fdeb_cell")

_COUNT_SRC = _TEXEL + _CELL + glsl("fdeb_count")

_SCATTER_SRC = _TEXEL + _CELL + glsl("fdeb_scatter")

_COMPAT_GLSL = glsl("fdeb_compat")

_FORCE_SRC = _TEXEL + _CELL + _COMPAT_GLSL + f"""
void main()
{{
  int idx = scig_index();
  if (idx >= u_count) {{ return; }}
  int e = idx / u_k;
  int p = idx - e * u_k;
  vec4 self = imageLoad(pos_in, scig_texel(idx));
  if (p == 0 || p == u_k - 1) {{
    imageStore(pos_out, scig_texel(idx), self);
    return;
  }}
  vec3 P = self.xyz;

  // Laplacian: the offset to the midpoint of this point's two neighbors along
  // its own polyline. Zero on an evenly spaced straight line, so zero strength
  // gives back the straight edge.
  vec3 spring = 0.5 * (imageLoad(pos_in, scig_texel(idx - 1)).xyz
                       + imageLoad(pos_in, scig_texel(idx + 1)).xyz) - P;

  vec4 ea = imageLoad(edge_img, scig_texel(2 * e));
  vec4 eu = imageLoad(edge_img, scig_texel(2 * e + 1));

  vec3 acc = vec3(0.0);
  float wsum = 0.0;
  ivec3 c0 = scig_cell(P);
  float r2 = u_radius * u_radius;

  for (int dx = -1; dx <= 1; ++dx) {{
    for (int dy = -1; dy <= 1; ++dy) {{
      for (int dz = -1; dz <= 1; ++dz) {{
        ivec3 c = c0 + ivec3(dx, dy, dz);
        if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, u_res))) {{
          continue;
        }}
        int cell = scig_cell_id(c);
        // Two buckets: which of the neighbor's points corresponds to this one
        // depends on which way that neighbor runs, and the bucket has to be
        // chosen before the neighbor is known. Antiparallel edges are
        // compatible (the angle measure is |cos|), so pairing them by raw index
        // would drag each one's head toward the other's tail and open the
        // bundle into an X. The two passes take complementary halves of the
        // same cell, so a point landing in both buckets is still counted once.
        for (int side = 0; side < 2; ++side) {{
          int want = (side == 0) ? p : (u_k - 1 - p);
          int bucket = cell * u_k + want;
          int start = int(imageLoad(cell_start, scig_texel(bucket)).x + 0.5);
          int n = min(int(imageLoad(cell_count_f, scig_texel(bucket)).x + 0.5),
                      u_cap);
          for (int m = 0; m < n; ++m) {{
            int jp = int(imageLoad(sorted_idx, scig_texel(start + m)).x);
            int j = jp / u_k;
            if (j == e) {{ continue; }}
            vec4 qa = imageLoad(edge_img, scig_texel(2 * j));
            vec4 qu = imageLoad(edge_img, scig_texel(2 * j + 1));
            bool same_way = dot(eu.xyz, qu.xyz) >= 0.0;
            if (same_way != (side == 0)) {{ continue; }}
            vec3 d = imageLoad(pos_in, scig_texel(jp)).xyz - P;
            float d2 = dot(d, d);
            if (d2 > r2) {{ continue; }}
            float cm = scig_compat(ea.xyz, eu.xyz, ea.w, qa.xyz, qu.xyz, qa.w);
            if (cm < u_thresh) {{ continue; }}
            float w = cm / (d2 + u_soften);
            acc += d * w;
            wsum += w;
          }}
        }}
      }}
    }}
  }}

  // The offset to the compatibility-weighted mean, not the sum: one step is
  // then a convex combination whatever the neighborhood holds, which bounds
  // the run.
  vec3 attract = (wsum > 0.0) ? acc / wsum : vec3(0.0);
  vec3 moved = P + u_step * (u_spring * spring + u_strength * attract);
  imageStore(pos_out, scig_texel(idx), vec4(moved, 0.0));
}}
"""

# Walks the old polyline twice; u_kin is bounded, so both loops are short.
_RESAMPLE_SRC = _TEXEL + glsl("fdeb_resample")

_SHADERS = {}


def _backend_ready():
    try:
        return gpu.platform.backend_type_get() != 'NONE'
    except Exception:  # noqa: BLE001 - raises before gpu.init()
        return False


def _grid_constants(info):
    info.push_constant('VEC3', "u_lo")
    info.push_constant('VEC3', "u_inv")
    info.push_constant('IVEC3', "u_res")
    info.push_constant('INT', "u_k")


def _shader(name):
    """Compile one kernel once, cached. None if this backend refuses it."""
    if name in _SHADERS:
        return _SHADERS[name]
    if not _backend_ready():
        return None
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(GROUP)
        info.push_constant('INT', "u_count")
        info.push_constant('INT', "u_row")
        if name == "count":
            info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_img", qualifiers={'READ'})
            info.image(1, 'R32UI', 'UINT_2D', "cell_count",
                       qualifiers={'READ', 'WRITE'})
            _grid_constants(info)
            src = _COUNT_SRC
        elif name == "scatter":
            info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_img", qualifiers={'READ'})
            info.image(1, 'R32UI', 'UINT_2D', "cell_cursor",
                       qualifiers={'READ', 'WRITE'})
            info.image(2, 'RGBA32F', 'FLOAT_2D', "cell_start",
                       qualifiers={'READ'})
            info.image(3, 'R32UI', 'UINT_2D', "sorted_idx", qualifiers={'WRITE'})
            _grid_constants(info)
            src = _SCATTER_SRC
        elif name == "resample":
            info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_in", qualifiers={'READ'})
            info.image(1, 'RGBA32F', 'FLOAT_2D', "pos_out", qualifiers={'WRITE'})
            info.push_constant('INT', "u_kin")
            info.push_constant('INT', "u_kout")
            src = _RESAMPLE_SRC
        else:
            info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_in", qualifiers={'READ'})
            info.image(1, 'RGBA32F', 'FLOAT_2D', "edge_img", qualifiers={'READ'})
            info.image(2, 'RGBA32F', 'FLOAT_2D', "cell_start",
                       qualifiers={'READ'})
            info.image(3, 'RGBA32F', 'FLOAT_2D', "cell_count_f",
                       qualifiers={'READ'})
            info.image(4, 'R32UI', 'UINT_2D', "sorted_idx", qualifiers={'READ'})
            info.image(5, 'RGBA32F', 'FLOAT_2D', "pos_out", qualifiers={'WRITE'})
            _grid_constants(info)
            info.push_constant('INT', "u_vis")
            info.push_constant('INT', "u_cap")
            info.push_constant('FLOAT', "u_radius")
            info.push_constant('FLOAT', "u_soften")
            info.push_constant('FLOAT', "u_thresh")
            info.push_constant('FLOAT', "u_step")
            info.push_constant('FLOAT', "u_spring")
            info.push_constant('FLOAT', "u_strength")
            src = _FORCE_SRC
        info.compute_source(src)
        _SHADERS[name] = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001 - unsupported backend, driver refusal
        _SHADERS[name] = None
    return _SHADERS[name]


def available():
    """True once all four kernels have compiled."""
    return all(_shader(n) is not None
               for n in ("count", "scatter", "force", "resample"))


def _texture(values):
    n, c = values.shape
    rows = max(1, int(np.ceil(n / TEX_ROW)))
    padded = np.zeros((rows * TEX_ROW, c), dtype=np.float32)
    padded[:n] = values
    return gpu.types.GPUTexture(
        (TEX_ROW, rows), format='RGBA32F',
        data=gpu.types.Buffer('FLOAT', padded.size, padded.ravel()))


def _blank(count):
    rows = max(1, int(np.ceil(count / TEX_ROW)))
    tex = gpu.types.GPUTexture((TEX_ROW, rows), format='RGBA32F')
    tex.clear(format='FLOAT', value=(0.0, 0.0, 0.0, 0.0))
    return tex


def _uint_tex(count):
    rows = max(1, int(np.ceil(count / TEX_ROW)))
    tex = gpu.types.GPUTexture((TEX_ROW, rows), format='R32UI')
    tex.clear(format='UINT', value=(0,))
    return tex


def _dispatch(shader, count):
    """Cover ``count`` invocations, spilling into y past MAX_GROUPS."""
    groups = max(1, (count + GROUP - 1) // GROUP)
    gx = min(groups, MAX_GROUPS)
    gy = (groups + gx - 1) // gx
    shader.uniform_int("u_count", int(count))
    shader.uniform_int("u_row", gx * GROUP)
    gpu.compute.dispatch(shader, gx, gy, 1)


def edge_texture(fr):
    """Two texels per edge: (start, length) and (unit direction, 0)."""
    e = fr["len"].shape[0]
    buf = np.zeros((e * 2, 4), dtype=np.float32)
    buf[0::2, :3] = fr["a"]
    buf[0::2, 3] = fr["len"]
    buf[1::2, :3] = fr["u"]
    return _texture(buf)


class _Bins:
    """Subdivision points bucketed into the fixed grid by counting sort. The
    prefix sum runs on the CPU; only one count per cell reads back."""

    def __init__(self, capacity, nbuckets):
        self.capacity = int(capacity)
        self.nbuckets = int(nbuckets)
        self.cell_count = _uint_tex(self.nbuckets)
        self.cell_cursor = _uint_tex(self.nbuckets)
        self.sorted_idx = _uint_tex(self.capacity)
        self.cell_start = None
        self.count_f = None
        self.counts = None
        self.busiest = 0

    def build(self, pos_tex, count, k, lo, inv, res, ncells):
        count_sh, scatter_sh = _shader("count"), _shader("scatter")
        if count_sh is None or scatter_sh is None:
            return False
        used = int(ncells) * int(k)
        lo_t = tuple(float(v) for v in lo)
        inv_t = tuple(float(v) for v in inv)
        res_t = tuple(int(v) for v in res)

        self.cell_count.clear(format='UINT', value=(0,))
        count_sh.bind()
        count_sh.image('pos_img', pos_tex)
        count_sh.image('cell_count', self.cell_count)
        count_sh.uniform_float("u_lo", lo_t)
        count_sh.uniform_float("u_inv", inv_t)
        count_sh.uniform_int("u_res", res_t)
        count_sh.uniform_int("u_k", int(k))
        _dispatch(count_sh, count)

        counts = np.asarray(self.cell_count.read()).ravel()[:used]
        self.counts = counts.astype(np.int64)
        self.busiest = max(self.busiest, int(self.counts.max(initial=0)))
        starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        pair = np.zeros((used, 4), dtype=np.float32)
        pair[:, 0] = starts
        # Held on the object; built inline, a GPUTexture can be collected
        # before the dispatch it was bound to runs.
        self.cell_start = _texture(pair)
        pair = np.zeros((used, 4), dtype=np.float32)
        pair[:, 0] = self.counts
        self.count_f = _texture(pair)

        self.cell_cursor.clear(format='UINT', value=(0,))
        scatter_sh.bind()
        scatter_sh.image('pos_img', pos_tex)
        scatter_sh.image('cell_cursor', self.cell_cursor)
        scatter_sh.image('cell_start', self.cell_start)
        scatter_sh.image('sorted_idx', self.sorted_idx)
        scatter_sh.uniform_float("u_lo", lo_t)
        scatter_sh.uniform_float("u_inv", inv_t)
        scatter_sh.uniform_int("u_res", res_t)
        scatter_sh.uniform_int("u_k", int(k))
        _dispatch(scatter_sh, count)
        return True


def _produce_gpu(coords, edges, params, report=None, cap=None):
    """Bundle on the GPU, None if this backend refuses. ``report`` puts the
    busiest bucket against the cap; over the cap the oracle disagrees."""
    if not available():
        return None
    cap = BUCKET_CAP if cap is None else int(cap)
    force_sh, resample_sh = _shader("force"), _shader("resample")

    fr = frames(coords, edges)
    sched = schedule(params)
    strength = float(np.clip(params["bundle_strength"], 0.0, 1.0))
    thresh = float(params["bundle_threshold"])
    use_vis = bool(params.get("fdeb_visibility", DEFAULT_VISIBILITY))
    diag = float(np.linalg.norm(np.ptp(np.asarray(coords, np.float64), axis=0)))
    radius = float(params.get("fdeb_radius", DEFAULT_RADIUS)) * diag
    soften = float((SOFTEN_FRAC * radius) ** 2)
    e_count = int(edges.shape[0])
    k_max = max(p for p, _s, _i in sched) + 2
    lo, inv, res, ncells = grid(coords, radius, k_max)
    lo_t = tuple(float(v) for v in lo)
    inv_t = tuple(float(v) for v in inv)
    res_t = tuple(int(v) for v in res)

    try:
        edge_tex = edge_texture(fr)
        k = sched[0][0] + 2
        seed = np.zeros((e_count * k_max, 4), dtype=np.float32)
        seed[:e_count * k, :3] = _initial(fr, k).reshape(-1, 3)
        pos_a = _texture(seed)
        pos_b = _blank(e_count * k_max)
        bins = _Bins(e_count * k_max, ncells * k_max)

        for p_count, step, iters in sched:
            k_new = p_count + 2
            if k_new != k:
                resample_sh.bind()
                resample_sh.image('pos_in', pos_a)
                resample_sh.image('pos_out', pos_b)
                resample_sh.uniform_int("u_kin", k)
                resample_sh.uniform_int("u_kout", k_new)
                _dispatch(resample_sh, e_count * k_new)
                pos_a, pos_b = pos_b, pos_a
                k = k_new
            n_points = e_count * k
            for _ in range(iters):
                if not bins.build(pos_a, n_points, k, lo, inv, res, ncells):
                    return None
                force_sh.bind()
                force_sh.image('pos_in', pos_a)
                force_sh.image('edge_img', edge_tex)
                force_sh.image('cell_start', bins.cell_start)
                force_sh.image('cell_count_f', bins.count_f)
                force_sh.image('sorted_idx', bins.sorted_idx)
                force_sh.image('pos_out', pos_b)
                force_sh.uniform_float("u_lo", lo_t)
                force_sh.uniform_float("u_inv", inv_t)
                force_sh.uniform_int("u_res", res_t)
                force_sh.uniform_int("u_k", k)
                force_sh.uniform_int("u_vis", 1 if use_vis else 0)
                force_sh.uniform_int("u_cap", int(cap))
                force_sh.uniform_float("u_radius", radius)
                force_sh.uniform_float("u_soften", soften)
                force_sh.uniform_float("u_thresh", thresh)
                force_sh.uniform_float("u_step", float(step))
                force_sh.uniform_float("u_spring", SPRING_GAIN)
                force_sh.uniform_float("u_strength", strength)
                _dispatch(force_sh, n_points)
                pos_a, pos_b = pos_b, pos_a

        # The one readback in the run.
        out = np.asarray(pos_a.read()).reshape(-1, 4)[:e_count * k, :3]
        ctrl = np.ascontiguousarray(out.reshape(e_count, k, 3), dtype=np.float32)
    except Exception:  # noqa: BLE001 - texture limits, out of VRAM, driver
        return None

    if report is not None:
        report.update({"busiest_bucket": bins.busiest, "cap": cap,
                       "capped": bins.busiest > cap,
                       "cells": ncells, "res": tuple(int(v) for v in res),
                       "points": e_count * k, "k": k})

    if not np.isfinite(ctrl).all():
        return None
    # Pinned rather than trusted: the kernel copies endpoints through untouched.
    ctrl[:, 0] = fr["a"]
    ctrl[:, -1] = fr["b"]
    return ctrl, np.full(e_count, k, np.int32)


def produce(coords, edges, params, ctx):
    """Return control polygons from the bundled polylines (producer contract)."""
    if PREFER_GPU:
        made = _produce_gpu(coords, edges, params)
        if made is not None:
            return made
    return _produce_numpy(coords, edges, params)


def build(coords, edges, params, node_colors=None, edge_color=None,
          edge_widths=None):
    return bundle_gpu.build(
        'FDEB', coords, edges, params, ctx={}, node_colors=node_colors,
        edge_color=edge_color, edge_widths=edge_widths)


def draw(data, mvp, beta, shader=None, width_range=None, viewport=None):
    return bundle_gpu.draw(data, mvp, beta, shader=shader,
                           width_range=width_range, viewport=viewport)
