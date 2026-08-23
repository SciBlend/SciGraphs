# Routed bundling, "winding roads" (Lambert et al. 2010).
#
# Edges route along a shared grid, so two edges picking the same cells are
# coincident there, and a road that carries traffic gets cheaper and carries
# more. Dijkstra wants a priority queue a compute shader cannot have, so this
# is Jacobi Bellman-Ford instead: a cell is a texel, relaxation is a stencil,
# reinforcement is ``imageAtomicAdd``, and the round count follows the grid
# diameter rather than the edge count. The stencil is the full box, because
# 4-/6-connectivity makes the metric Manhattan, off Euclidean by 8% at worst.

import numpy as np

import gpu

from . import bundle_gpu
from .glsl_source import glsl

from ...core.render.bundling import routed as _routed  # noqa: E402,F401
from ...core.render.bundling.routed import (  # noqa: E402,F401
    AVOID_GAIN, GPU_FIELD_TEXELS, INF, MARGIN, MAX_CTRL, MIN_COST,
    NP_FIELD_TEXELS, _STEP_LEN, _blur, _cell_of, _control, _frame, _grid,
    _neighbor_table, _offsets, _ragged_arange, _reinforced, _relax_numpy,
    _shift, _trace_numpy, density_cost,
)

TEX_ROW = bundle_gpu.TEX_ROW

# As in sim_gpu: a wavefront on AMD, two warps on NVIDIA, memory-bound.
GROUP = 64

# Rounds between readbacks of the "did anything change" counter, trading a sync
# per round against up to k-1 wasted dispatches. Swept on the 240-node graph at
# R = 128 (four route/reinforce rounds, best of three): 384 ms at k=1, 363 at
# k=2, 365 at k=4, 363 at k=8, 388 at k=16. Flat between 2 and 8; a 5% effect.
CONVERGE_CHECK = 4


def check(params, edges, ctx):
    """Whether routed bundling can route these edges (producer contract).
    Nodes sharing a cell fall back to a straight polygon instead of refusing."""
    if params["style_type"] != 'ROUTED':
        return False, "not routed bundling"
    return True, ""


def _node_density(coords, res):
    """Node density shaped for ``routed.density_cost``. volume.py needs a GPU,
    so it is imported here rather than at module scope."""
    from . import volume

    return volume.build_field(coords, res=res)


def _density_cost(coords, node_cell, shape, res, avoid):
    return density_cost(coords, node_cell, shape, res, avoid,
                        build_field=_node_density)


def _plan(coords, edges, params, budget):
    return _routed._plan(coords, edges, params, budget, density=_density_cost)


# R32F rather than the RGBA32F used elsewhere: the distance field is sources
# times cells, so a quarter of the bandwidth decides whether a graph fits. Only
# RGBA32F is verified on this build, so the shader falls back if R32F refuses.
_FLOAT_FMT = 'R32F'

_GRID_COMMON = f"""
ivec2 scig_texel(int i) {{ return ivec2(i % {TEX_ROW}, i / {TEX_ROW}); }}

int scig_flat(ivec3 c) {{ return (c.z * u_ny + c.y) * u_nx + c.x; }}

ivec3 scig_coord(int c)
{{
  int plane = u_nx * u_ny;
  int z = c / plane;
  int r = c - z * plane;
  int y = r / u_nx;
  return ivec3(r - y * u_nx, y, z);
}}

bool scig_outside(ivec3 c)
{{
  return c.x < 0 || c.y < 0 || c.z < 0
      || c.x >= u_nx || c.y >= u_ny || c.z >= u_nz;
}}

// The three step lengths, as literals rather than length(vec3(...)): the spec
// allows length() two ULP, and the numpy reference has to agree to the last one
// or the two pick different members of a tied pair of routes. Same constants as
// _STEP_LEN on the Python side.
float scig_step(int dx, int dy, int dz)
{{
  int q = dx * dx + dy * dy + dz * dz;
  return q == 1 ? 1.0 : (q == 2 ? 1.4142135381698608 : 1.7320507764816284);
}}
"""

# One round of Jacobi Bellman-Ford over every (source, cell) pair. The 2D case
# is u_nz == 1, whose dz = +-1 neighbors fall outside, so both see the same set.
_RELAX_SRC = _GRID_COMMON + glsl("routed_relax")

# One invocation per edge walks back to its source, counting cells atomically.
# One pass, so the usage image is done when the last route is.
_TRACE_SRC = _GRID_COMMON + glsl("routed_trace")

_SHADERS = {}


def _backend_live():
    try:
        return gpu.platform.backend_type_get() != 'NONE'
    except Exception:  # noqa: BLE001 - raises before gpu.init()
        return False


def _compile(name, src, images, pushes):
    """Compile one kernel against ``_FLOAT_FMT``, retrying in RGBA32F. The retry
    rewrites the module-wide format, and :func:`available` compiles both kernels
    before any texture exists, so they cannot disagree."""
    global _FLOAT_FMT
    for fmt in (_FLOAT_FMT, 'RGBA32F'):
        try:
            info = gpu.types.GPUShaderCreateInfo()
            info.local_group_size(GROUP)
            for slot, (iname, ifmt, quals) in enumerate(images):
                info.image(slot, fmt if ifmt is None else ifmt,
                           'UINT_2D' if ifmt == 'R32UI' else 'FLOAT_2D',
                           iname, qualifiers=quals)
            for ptype, pname in pushes:
                info.push_constant(ptype, pname)
            info.compute_source(src)
            shader = gpu.shader.create_from_info(info)
            _FLOAT_FMT = fmt
            return shader
        except Exception:  # noqa: BLE001 - format or backend unsupported
            continue
    return None


_GRID_PUSH = [('INT', "u_nx"), ('INT', "u_ny"), ('INT', "u_nz"),
              ('INT', "u_cells"), ('FLOAT', "u_cell")]


def _shader(name):
    if name in _SHADERS:
        return _SHADERS[name]
    if not _backend_live():
        return None
    if name == "relax":
        sh = _compile(
            name, _RELAX_SRC,
            [("dist_in", None, {'READ'}), ("cost_img", None, {'READ'}),
             ("dist_out", None, {'WRITE'}),
             ("changed", 'R32UI', {'READ', 'WRITE'})],
            _GRID_PUSH + [('INT', "u_total"), ('FLOAT', "u_eps")])
    else:
        sh = _compile(
            name, _TRACE_SRC,
            [("dist_img", None, {'READ'}), ("cost_img", None, {'READ'}),
             ("edge_img", 'RGBA32F', {'READ'}), ("usage", 'R32UI',
                                                 {'READ', 'WRITE'}),
             ("route_img", None, {'WRITE'}), ("len_img", None, {'WRITE'})],
            _GRID_PUSH + [('INT', "u_edges"), ('INT', "u_max_steps")])
    _SHADERS[name] = sh
    return sh


def available():
    """Whether both kernels compiled."""
    return _shader("relax") is not None and _shader("trace") is not None


def _channels():
    return 1 if _FLOAT_FMT == 'R32F' else 4


def _float_tex(values):
    ch = _channels()
    n = int(values.size)
    rows = max(1, int(np.ceil(n / TEX_ROW)))
    padded = np.zeros((rows * TEX_ROW, ch), dtype=np.float32)
    padded[:n, 0] = np.asarray(values, dtype=np.float32).ravel()
    return gpu.types.GPUTexture(
        (TEX_ROW, rows), format=_FLOAT_FMT,
        data=gpu.types.Buffer('FLOAT', padded.size, padded.ravel()))


def _rgba_tex(values):
    n = values.shape[0]
    rows = max(1, int(np.ceil(n / TEX_ROW)))
    padded = np.zeros((rows * TEX_ROW, 4), dtype=np.float32)
    padded[:n] = values
    return gpu.types.GPUTexture(
        (TEX_ROW, rows), format='RGBA32F',
        data=gpu.types.Buffer('FLOAT', padded.size, padded.ravel()))


def _uint_tex(count):
    rows = max(1, int(np.ceil(count / TEX_ROW)))
    tex = gpu.types.GPUTexture((TEX_ROW, rows), format='R32UI')
    tex.clear(format='UINT', value=(0,))
    return tex


def _read_float(tex, n):
    return np.asarray(tex.read()).ravel()[:n * _channels():_channels()]


def _bind_grid(sh, counts, ncells, cell):
    nx, ny, nz = (list(counts) + [1, 1])[:3]
    sh.uniform_int("u_nx", int(nx))
    sh.uniform_int("u_ny", int(ny))
    sh.uniform_int("u_nz", int(nz))
    sh.uniform_int("u_cells", int(ncells))
    sh.uniform_float("u_cell", float(cell))


def _relax_gpu(cost, seeds, counts, cell, eps, max_rounds):
    """Relax to the fixed point numpy reaches; the field stays a texture."""
    sh = _shader("relax")
    if sh is None:
        return None
    ncells = int(cost.size)
    s = int(np.asarray(seeds).size)
    total = s * ncells

    init = np.full(total, INF, dtype=np.float32)
    init[np.arange(s) * ncells + np.asarray(seeds)] = 0.0
    dist_a = _float_tex(init)
    dist_b = _float_tex(np.zeros(total, dtype=np.float32))
    cost_tex = _float_tex(cost.ravel())
    changed = _uint_tex(1)

    groups = (total + GROUP - 1) // GROUP
    rounds = 0
    converged = False
    while rounds < int(max_rounds):
        window = min(CONVERGE_CHECK, int(max_rounds) - rounds)
        changed.clear(format='UINT', value=(0,))
        for _ in range(window):
            sh.bind()
            sh.image('dist_in', dist_a)
            sh.image('cost_img', cost_tex)
            sh.image('dist_out', dist_b)
            sh.image('changed', changed)
            _bind_grid(sh, counts, ncells, cell)
            sh.uniform_int("u_total", total)
            sh.uniform_float("u_eps", float(eps))
            gpu.compute.dispatch(sh, groups, 1, 1)
            dist_a, dist_b = dist_b, dist_a
            rounds += 1
        if int(np.asarray(changed.read()).ravel()[0]) == 0:
            converged = True
            break
    # Held on the result: an unreferenced GPUTexture is collectable at return.
    return {"dist": dist_a, "cost": cost_tex, "rounds": rounds,
            "converged": converged, "cells": ncells, "sources": s}


def _trace_gpu(solved, counts, cell, field, src, dst, straight, max_steps):
    """Trace and reinforce in one dispatch: ``(routes, lengths, usage)``."""
    sh = _shader("trace")
    if sh is None:
        return None
    e = int(src.size)
    ncells = solved["cells"]

    spec = np.zeros((e, 4), dtype=np.float32)
    spec[:, 0] = field
    spec[:, 1] = src
    spec[:, 2] = dst
    spec[:, 3] = straight.astype(np.float32)
    edge_tex = _rgba_tex(spec)
    usage = _uint_tex(ncells)
    route_tex = _float_tex(np.full(e * int(max_steps), -1.0, dtype=np.float32))
    len_tex = _float_tex(np.zeros(e, dtype=np.float32))

    sh.bind()
    sh.image('dist_img', solved["dist"])
    sh.image('cost_img', solved["cost"])
    sh.image('edge_img', edge_tex)
    sh.image('usage', usage)
    sh.image('route_img', route_tex)
    sh.image('len_img', len_tex)
    _bind_grid(sh, counts, ncells, cell)
    sh.uniform_int("u_edges", e)
    sh.uniform_int("u_max_steps", int(max_steps))
    gpu.compute.dispatch(sh, (e + GROUP - 1) // GROUP, 1, 1)

    lengths = _read_float(len_tex, e).astype(np.int64)
    routes = _read_float(route_tex, e * int(max_steps)).astype(np.int64)
    routes = routes.reshape(e, int(max_steps))
    used = np.asarray(usage.read()).ravel()[:ncells].astype(np.int64)
    return routes, lengths, used


def route(coords, edges, params, backend=None, want_dist=False):
    """Run the whole route/reinforce loop and report what it did. ``backend``
    is 'gpu', 'numpy', or None to take the GPU if it is there."""
    if backend is None:
        backend = "gpu" if available() else "numpy"
    on_gpu = backend == "gpu"
    if on_gpu and not available():
        return None

    plan = _plan(coords, edges, params,
                 GPU_FIELD_TEXELS if on_gpu else NP_FIELD_TEXELS)
    iterations = max(1, int(params.get("routed_iterations", 4)))
    reinforce = float(np.clip(params.get("routed_reinforce", 0.5), 0.0, 0.99))

    # The seed cell of each field, through a sorted view of `field`.
    order = np.argsort(plan["field"], kind="stable")
    first = order[np.searchsorted(plan["field"][order],
                                  np.arange(plan["sources"].size))]
    seeds = plan["src_cell"][first]

    table = lens = None
    if not on_gpu:
        table, lens = _neighbor_table(plan["shape"])

    cost = plan["base_cost"]
    usage = np.zeros(plan["ncells"], dtype=np.int64)
    history = []
    routes = lengths = dist = None

    for it in range(iterations):
        if on_gpu:
            solved = _relax_gpu(cost, seeds, plan["counts"], plan["cell"],
                                plan["eps"], plan["max_rounds"])
            if solved is None:
                return None
            traced = _trace_gpu(solved, plan["counts"], plan["cell"],
                                plan["field"], plan["src_cell"],
                                plan["dst_cell"], plan["straight"],
                                plan["max_steps"])
            if traced is None:
                return None
            routes, lengths, usage = traced
            rounds, converged = solved["rounds"], solved["converged"]
            dist = (_read_float(solved["dist"],
                                solved["sources"] * solved["cells"]
                                ).reshape(solved["sources"], solved["cells"])
                    if want_dist else None)
        else:
            dist, rounds, converged = _relax_numpy(
                cost.reshape(plan["shape"]), seeds, plan["cell"], plan["eps"],
                plan["max_rounds"])
            routes, lengths = _trace_numpy(
                dist, cost.ravel(), table, lens, plan["cell"], plan["field"],
                plan["src_cell"], plan["dst_cell"], plan["max_steps"])
            routes[plan["straight"]] = -1
            lengths[plan["straight"]] = 0
            usage = np.bincount(routes[routes >= 0].ravel(),
                                minlength=plan["ncells"])
            if not want_dist:
                dist = None

        distinct = int(np.count_nonzero(usage))
        history.append({"iteration": it, "rounds": rounds,
                        "converged": bool(converged),
                        "distinct_cells": distinct,
                        "cells_used": int(usage.sum()),
                        "reached": int(np.count_nonzero(
                            (lengths > 0)
                            & (routes[np.arange(routes.shape[0]),
                                      np.maximum(lengths - 1, 0)]
                               == plan["src_cell"])))})
        if it < iterations - 1:
            cost = _reinforced(plan["base_cost"], usage.reshape(plan["shape"]),
                               reinforce).ravel()
        else:
            cost = cost.ravel()

    return {"plan": plan, "routes": routes, "lengths": lengths,
            "usage": usage, "cost": cost.reshape(plan["shape"]),
            "dist": dist, "history": history, "backend": backend,
            "seeds": seeds}


def _produce_numpy(coords, edges, params, ctx=None):
    """Control polygons from numpy: the oracle and the fallback. A Bellman-Ford
    that stops a round early is wrong in only a few far cells and still looks
    reasonable, which is why the oracle earns its keep here. It shares the grid,
    the cost field and the resampling, and no line of the relaxation or trace."""
    solution = route(coords, edges, params, backend="numpy")
    if solution is None:
        return None
    return _control(solution, coords, np.asarray(edges, dtype=np.int64))


def _produce_gpu(coords, edges, params, ctx=None):
    solution = route(coords, edges, params, backend="gpu")
    if solution is None:
        return None
    return _control(solution, coords, np.asarray(edges, dtype=np.int64))


def produce(coords, edges, params, ctx):
    """Routes along a shared grid, as control polygons (producer contract).
    GPU first, numpy on any failure: the same algorithm either way."""
    try:
        made = _produce_gpu(coords, edges, params, ctx)
        if made is not None:
            return made
    except Exception:  # noqa: BLE001 - a driver that refuses is a fallback
        pass
    try:
        return _produce_numpy(coords, edges, params, ctx)
    except Exception:  # noqa: BLE001 - a mode that cannot route declines
        return None
