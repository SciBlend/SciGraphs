# The layout simulation, on the GPU. The force model matches
# `core/mesh/layouts/simulation.py` term for term, the numpy oracle that
# test_sim_gpu checks against. 1M nodes (RTX 4060, Vulkan): 5.65 ms per frame
# from the CPU, 0.08 ms in a compute shader. Python `gpu` has no storage
# buffers, so every array is a texture, and `GPUTexture` is FLOAT-only, which
# caps exact indices at 2^24.

import numpy as np

import gpu

from . import shaders

# Shared with dynamic.py's position texture, so a node index means one thing.
TEX_ROW = shaders.POS_TEX_ROW

# 64 is a wavefront on AMD and two warps on NVIDIA.
GROUP = 64

_MAX_EXACT_INDEX = 1 << 24


def available():
    return _force_shader() is not None


def _texture(values, fmt='RGBA32F'):
    n, c = values.shape
    rows = max(1, int(np.ceil(n / TEX_ROW)))
    padded = np.zeros((rows * TEX_ROW, c), dtype=np.float32)
    padded[:n] = values
    return gpu.types.GPUTexture(
        (TEX_ROW, rows), format=fmt,
        data=gpu.types.Buffer('FLOAT', padded.size, padded.ravel()))


def build_adjacency(edges, num_nodes):
    """CSR adjacency, (start, count) in offsets .xy; a scatter needs atomics."""
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return None
    if num_nodes >= _MAX_EXACT_INDEX:
        raise ValueError(
            f"{num_nodes} nodes exceeds the {_MAX_EXACT_INDEX} that a float32 "
            "index can address exactly")

    # Both directions: the force is symmetric and each endpoint must see it.
    both = np.concatenate([edges, edges[:, ::-1]], axis=0)
    order = np.argsort(both[:, 0], kind="stable")
    src = both[order, 0]
    dst = both[order, 1]

    degree = np.bincount(src, minlength=num_nodes).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(degree)[:-1]])

    offsets = np.zeros((num_nodes, 4), dtype=np.float32)
    offsets[:, 0] = starts
    offsets[:, 1] = degree
    nbrs = np.zeros((dst.size, 4), dtype=np.float32)
    nbrs[:, 0] = dst

    return {
        "offsets": _texture(offsets),
        "neighbors": _texture(nbrs),
        "degree": degree.astype(np.float32),
        "max_degree": int(degree.max()) if degree.size else 0,
        "count": int(dst.size),
    }


def build_nodes(coords, mass):
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    n = coords.shape[0]
    pos = np.zeros((n, 4), dtype=np.float32)
    pos[:, :3] = coords
    pos[:, 3] = np.asarray(mass, dtype=np.float32)
    return _texture(pos)


_FORCE_SRC = f"""
ivec2 scig_texel(int i) {{ return ivec2(i % {TEX_ROW}, i / {TEX_ROW}); }}

void main()
{{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= u_count) {{ return; }}

  vec4 self = imageLoad(pos_img, scig_texel(i));
  vec3 p = self.xyz;
  float m = self.w;
  vec3 force = vec3(0.0);

  // Attraction: walk this node's slice of the CSR and sum the pulls. Linear in
  // the separation (ForceAtlas2), divided by the source mass when the outbound
  // distribution is on -- the same two lines as the numpy reference.
  vec4 off = imageLoad(adj_off, scig_texel(i));
  int start = int(off.x + 0.5);
  int deg = int(off.y + 0.5);
  for (int e = 0; e < deg; ++e) {{
    int j = int(imageLoad(adj_nbr, scig_texel(start + e)).x + 0.5);
    vec3 q = imageLoad(pos_img, scig_texel(j)).xyz;
    vec3 d = q - p;
    float len = length(d);
    float factor = u_attraction;
    if (u_model == 1) {{           // FR: d^2 / k
      factor *= len / max(u_k, 1e-9);
    }} else if (u_model == 2) {{   // LinLog: log(1 + d)
      factor *= log(1.0 + len) / max(len, 1e-9);
    }}
    if (u_model != 1) {{ factor /= max(m, 1e-9); }}
    force += d * factor;
  }}

  // Near-field repulsion: exact, over this node's K nearest neighbors.
  //
  // A fixed-width gather -- K scattered reads per invocation, with thousands of
  // invocations in flight to hide their latency. This term costs 46 of the 85 ms
  // a step takes in numpy at a hundred thousand nodes, bound there by that
  // latency.
  //
  // The neighbor list is built on the CPU and refreshed on a cadence rather
  // than every step, as the reference does: who is nearby changes far more
  // slowly than where things are.
  for (int nk = 0; nk < u_near_k; ++nk) {{
    int j = int(imageLoad(near_idx, scig_texel(i * u_near_k + nk)).x + 0.5);
    vec4 other = imageLoad(pos_img, scig_texel(j));
    vec3 d = p - other.xyz;
    float d2 = dot(d, d) + u_near_soften;
    force += d * ((u_repulsion * m * other.w) / d2);
  }}

  // Far-field repulsion, from the coarse grid of monopoles. One texel per
  // occupied cell holding (centroid.xyz, mass); the node's own cell is skipped
  // because the near field answers that exactly.
  //
  // The force is measured from the node's own cell centroid, not from the node:
  // one force per cell, handed to every node in it, scaled by the node's mass.
  // Measuring from the node would be more accurate and just as cheap, but it is
  // not what the numpy reference computes, and agreement is what lets the
  // reference validate the kernel. Upgrading both sides to node-to-cell is a
  // change to make together.
  int own = int(imageLoad(cell_of, scig_texel(i)).x + 0.5);
  vec3 from = imageLoad(cell_buf, scig_texel(own)).xyz;
  for (int c = 0; c < u_cells; ++c) {{
    if (c == own) {{ continue; }}
    vec4 cell = imageLoad(cell_buf, scig_texel(c));
    vec3 d = from - cell.xyz;
    float d2 = dot(d, d) + u_far_soften;
    force += d * ((u_repulsion * m * cell.w) / d2);
  }}

  imageStore(force_img, scig_texel(i), vec4(force, 0.0));
}}
"""


_SHADER = None
_TRIED = False


def _force_shader():
    global _SHADER, _TRIED
    if _TRIED:
        return _SHADER
    _TRIED = True
    try:
        if gpu.platform.backend_type_get() == 'NONE':
            _TRIED = False
            return None
    except Exception:  # noqa: BLE001 - raises before gpu.init()
        _TRIED = False
        return None
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(GROUP)
        info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_img", qualifiers={'READ'})
        info.image(1, 'RGBA32F', 'FLOAT_2D', "adj_off", qualifiers={'READ'})
        info.image(2, 'RGBA32F', 'FLOAT_2D', "adj_nbr", qualifiers={'READ'})
        info.image(3, 'RGBA32F', 'FLOAT_2D', "cell_buf", qualifiers={'READ'})
        info.image(4, 'RGBA32F', 'FLOAT_2D', "cell_of", qualifiers={'READ'})
        info.image(5, 'RGBA32F', 'FLOAT_2D', "near_idx", qualifiers={'READ'})
        info.image(6, 'RGBA32F', 'FLOAT_2D', "force_img", qualifiers={'WRITE'})
        info.push_constant('INT', "u_count")
        info.push_constant('INT', "u_cells")
        info.push_constant('INT', "u_model")
        info.push_constant('INT', "u_near_k")
        info.push_constant('FLOAT', "u_attraction")
        info.push_constant('FLOAT', "u_repulsion")
        info.push_constant('FLOAT', "u_k")
        info.push_constant('FLOAT', "u_far_soften")
        info.push_constant('FLOAT', "u_near_soften")
        info.compute_source(_FORCE_SRC)
        _SHADER = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001 - unsupported backend
        _SHADER = None
    return _SHADER


MODEL_CODES = {'FA2': 0, 'FR': 1, 'LINLOG': 2}


def build_near(near_idx):
    """(N, K) ``near_idx`` to a row-major (node, k) texture, built on the CPU."""
    idx = np.ascontiguousarray(near_idx, dtype=np.float32)
    flat = idx.reshape(-1, 1)
    out = np.zeros((flat.shape[0], 4), dtype=np.float32)
    out[:, 0] = flat[:, 0]
    return _texture(out), int(idx.shape[1])


def compute_forces(pos_tex, adj, cells, cell_of, params, count, near=None,
                   near_k=0):
    """``cells`` and ``cell_of`` are reference numpy; only the kernel is tested."""
    shader = _force_shader()
    if shader is None:
        return None
    rows = max(1, int(np.ceil(count / TEX_ROW)))
    force_tex = gpu.types.GPUTexture((TEX_ROW, rows), format='RGBA32F')
    force_tex.clear(format='FLOAT', value=(0.0, 0.0, 0.0, 0.0))

    shader.bind()
    shader.image('pos_img', pos_tex)
    shader.image('adj_off', adj["offsets"])
    shader.image('adj_nbr', adj["neighbors"])
    shader.image('cell_buf', cells)
    shader.image('cell_of', cell_of)
    # The sampler must point somewhere valid even at K = 0.
    shader.image('near_idx', near if near is not None else cell_of)
    shader.image('force_img', force_tex)
    shader.uniform_int("u_count", count)
    shader.uniform_int("u_cells", int(params["cells"]))
    shader.uniform_int("u_model", MODEL_CODES.get(params.get("model", 'FA2'), 0))
    shader.uniform_int("u_near_k", int(near_k) if near is not None else 0)
    shader.uniform_float("u_attraction", float(params["attraction"]))
    shader.uniform_float("u_repulsion", float(params["repulsion"]))
    shader.uniform_float("u_k", float(params["k"]))
    shader.uniform_float("u_far_soften", float(params["far_soften"]))
    shader.uniform_float("u_near_soften", float(params.get("near_soften", 0.0)))
    gpu.compute.dispatch(shader, (count + GROUP - 1) // GROUP, 1, 1)
    return force_tex


def read_forces(force_tex, count):
    """Validation only. The draw path never reads forces back."""
    return np.asarray(force_tex.read()).reshape(-1, 4)[:count, :3]


# Spatial binning on the GPU: a counting sort in three passes, atomic count per
# cell, prefix sum, atomic scatter. It replaces the kd-tree that fed the near
# field, 308 ms at 400k nodes against 19 ms for the two dispatches it served.
# The prefix sum stays on the CPU: 4096 values, constant in graph size.

BIN_RES = 16          # cells per axis; 16^3 = 4096
BIN_CELLS = BIN_RES ** 3


_BIN_COMMON = f"""
ivec2 scig_texel(int i) {{ return ivec2(i % {TEX_ROW}, i / {TEX_ROW}); }}

int scig_cell(vec3 p)
{{
  vec3 t = (p - u_lo) / max(u_span, vec3(1e-9));
  ivec3 c = clamp(ivec3(floor(t * float({BIN_RES}))), ivec3(0),
                  ivec3({BIN_RES} - 1));
  return (c.x * {BIN_RES} + c.y) * {BIN_RES} + c.z;
}}
"""

_BIN_COUNT_SRC = _BIN_COMMON + f"""
void main()
{{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= u_count) {{ return; }}
  vec3 p = imageLoad(pos_img, scig_texel(i)).xyz;
  imageAtomicAdd(cell_count, scig_texel(scig_cell(p)), 1u);
}}
"""

_BIN_SCATTER_SRC = _BIN_COMMON + f"""
void main()
{{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= u_count) {{ return; }}
  vec3 p = imageLoad(pos_img, scig_texel(i)).xyz;
  int c = scig_cell(p);
  // The cursor starts at zero and counts up; adding the cell's start offset
  // gives this node a slot no other node can be given.
  uint slot = imageAtomicAdd(cell_cursor, scig_texel(c), 1u);
  int start = int(imageLoad(cell_start, scig_texel(c)).x + 0.5);
  imageStore(sorted_idx, scig_texel(start + int(slot)), uvec4(uint(i), 0u, 0u, 0u));
}}
"""

_BIN_SHADERS = {}


def _bin_shader(name):
    if name in _BIN_SHADERS:
        return _BIN_SHADERS[name]
    try:
        if gpu.platform.backend_type_get() == 'NONE':
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(GROUP)
        info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_img", qualifiers={'READ'})
        if name == "count":
            info.image(1, 'R32UI', 'UINT_2D', "cell_count",
                       qualifiers={'READ', 'WRITE'})
            src = _BIN_COUNT_SRC
        else:
            info.image(1, 'R32UI', 'UINT_2D', "cell_cursor",
                       qualifiers={'READ', 'WRITE'})
            info.image(2, 'RGBA32F', 'FLOAT_2D', "cell_start",
                       qualifiers={'READ'})
            info.image(3, 'R32UI', 'UINT_2D', "sorted_idx",
                       qualifiers={'WRITE'})
            src = _BIN_SCATTER_SRC
        info.push_constant('INT', "u_count")
        info.push_constant('VEC3', "u_lo")
        info.push_constant('VEC3', "u_span")
        info.compute_source(src)
        _BIN_SHADERS[name] = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001
        _BIN_SHADERS[name] = None
    return _BIN_SHADERS[name]


def binning_available():
    return _bin_shader("count") is not None and _bin_shader("scatter") is not None


def _uint_tex(count):
    rows = max(1, int(np.ceil(count / TEX_ROW)))
    tex = gpu.types.GPUTexture((TEX_ROW, rows), format='R32UI')
    tex.clear(format='UINT', value=(0,))
    return tex


class SpatialBins:

    def __init__(self, count):
        self.n = int(count)
        self.cell_count = _uint_tex(BIN_CELLS)
        self.cell_cursor = _uint_tex(BIN_CELLS)
        self.sorted_idx = _uint_tex(self.n)
        self.cell_start = None
        self.counts = None
        self.starts = None

    def build(self, pos_tex, lo, span):
        count_sh = _bin_shader("count")
        scatter_sh = _bin_shader("scatter")
        if count_sh is None or scatter_sh is None:
            return False
        groups = (self.n + GROUP - 1) // GROUP
        lo = tuple(float(v) for v in lo)
        span = tuple(float(v) for v in span)

        self.cell_count.clear(format='UINT', value=(0,))
        count_sh.bind()
        count_sh.image('pos_img', pos_tex)
        count_sh.image('cell_count', self.cell_count)
        count_sh.uniform_int("u_count", self.n)
        count_sh.uniform_float("u_lo", lo)
        count_sh.uniform_float("u_span", span)
        gpu.compute.dispatch(count_sh, groups, 1, 1)

        counts = np.asarray(self.cell_count.read()).ravel()[:BIN_CELLS]
        counts = counts.astype(np.int64)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.counts, self.starts = counts, starts
        start_f = np.zeros((BIN_CELLS, 4), dtype=np.float32)
        start_f[:, 0] = starts
        self.cell_start = _texture(start_f)

        self.cell_cursor.clear(format='UINT', value=(0,))
        scatter_sh.bind()
        scatter_sh.image('pos_img', pos_tex)
        scatter_sh.image('cell_cursor', self.cell_cursor)
        scatter_sh.image('cell_start', self.cell_start)
        scatter_sh.image('sorted_idx', self.sorted_idx)
        scatter_sh.uniform_int("u_count", self.n)
        scatter_sh.uniform_float("u_lo", lo)
        scatter_sh.uniform_float("u_span", span)
        gpu.compute.dispatch(scatter_sh, groups, 1, 1)
        return True

    def read_members(self):
        order = np.asarray(self.sorted_idx.read()).ravel()[:self.n]
        return self.counts, self.starts, order.astype(np.int64)


# Neighbor list from the bins: up to K of whoever shares the node's 27 cells, in
# the (N, K) shape the near field consumes. The near field was already 6% off
# exact all-pairs, so this swaps one approximation for another.
_NEIGHBOR_SRC = _BIN_COMMON + f"""
void main()
{{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= u_count) {{ return; }}
  vec3 p = imageLoad(pos_img, scig_texel(i)).xyz;

  vec3 t = (p - u_lo) / max(u_span, vec3(1e-9));
  ivec3 c0 = clamp(ivec3(floor(t * float({BIN_RES}))), ivec3(0),
                   ivec3({BIN_RES} - 1));

  int written = 0;
  int base = i * u_k;
  for (int dx = -1; dx <= 1 && written < u_k; ++dx) {{
    for (int dy = -1; dy <= 1 && written < u_k; ++dy) {{
      for (int dz = -1; dz <= 1 && written < u_k; ++dz) {{
        ivec3 c = c0 + ivec3(dx, dy, dz);
        if (any(lessThan(c, ivec3(0))) ||
            any(greaterThanEqual(c, ivec3({BIN_RES})))) {{ continue; }}
        int cell = (c.x * {BIN_RES} + c.y) * {BIN_RES} + c.z;
        int start = int(imageLoad(cell_start, scig_texel(cell)).x + 0.5);
        int n = int(imageLoad(cell_count_f, scig_texel(cell)).x + 0.5);
        for (int e = 0; e < n && written < u_k; ++e) {{
          int j = int(imageLoad(sorted_idx, scig_texel(start + e)).x);
          if (j == i) {{ continue; }}
          imageStore(near_out, scig_texel(base + written),
                     vec4(float(j), 0.0, 0.0, 0.0));
          written += 1;
        }}
      }}
    }}
  }}
  // Pad with the node itself: the force term then subtracts a zero separation,
  // which the softening keeps finite, so a short list contributes nothing rather
  // than counting a real neighbor twice.
  for (int e = written; e < u_k; ++e) {{
    imageStore(near_out, scig_texel(base + e), vec4(float(i), 0.0, 0.0, 0.0));
  }}
}}
"""


def _neighbor_shader():
    if "neighbors" in _BIN_SHADERS:
        return _BIN_SHADERS["neighbors"]
    try:
        if gpu.platform.backend_type_get() == 'NONE':
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(GROUP)
        info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_img", qualifiers={'READ'})
        info.image(1, 'RGBA32F', 'FLOAT_2D', "cell_start", qualifiers={'READ'})
        info.image(2, 'RGBA32F', 'FLOAT_2D', "cell_count_f", qualifiers={'READ'})
        info.image(3, 'R32UI', 'UINT_2D', "sorted_idx", qualifiers={'READ'})
        info.image(4, 'RGBA32F', 'FLOAT_2D', "near_out", qualifiers={'WRITE'})
        info.push_constant('INT', "u_count")
        info.push_constant('INT', "u_k")
        info.push_constant('VEC3', "u_lo")
        info.push_constant('VEC3', "u_span")
        info.compute_source(_NEIGHBOR_SRC)
        _BIN_SHADERS["neighbors"] = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001
        _BIN_SHADERS["neighbors"] = None
    return _BIN_SHADERS["neighbors"]


def neighbors_from_bins(bins, pos_tex, lo, span, k):
    shader = _neighbor_shader()
    if shader is None or bins.cell_start is None:
        return None
    count_f = np.zeros((BIN_CELLS, 4), dtype=np.float32)
    count_f[:, 0] = bins.counts
    # Held on the bins, not inline: a GPUTexture created as a call argument can
    # be collected before its dispatch runs, and the kernel reads garbage.
    bins.count_f_tex = _texture(count_f)
    tex = _blank(bins.n * k)
    shader.bind()
    shader.image('pos_img', pos_tex)
    shader.image('cell_start', bins.cell_start)
    shader.image('cell_count_f', bins.count_f_tex)
    shader.image('sorted_idx', bins.sorted_idx)
    shader.image('near_out', tex)
    shader.uniform_int("u_count", bins.n)
    shader.uniform_int("u_k", int(k))
    shader.uniform_float("u_lo", tuple(float(v) for v in lo))
    shader.uniform_float("u_span", tuple(float(v) for v in span))
    gpu.compute.dispatch(shader, (bins.n + GROUP - 1) // GROUP, 1, 1)
    return tex


def neighbors_numpy(coords, lo, span, k):
    """The same selection in numpy, unvectorized and in the shader's cell order."""
    n = coords.shape[0]
    cell = cell_index_numpy(coords, lo, span)
    order = np.argsort(cell, kind="stable")
    counts = np.bincount(cell, minlength=BIN_CELLS)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    t = (np.asarray(coords, np.float64) - np.asarray(lo)) / np.maximum(
        np.asarray(span, np.float64), 1e-9)
    c0 = np.clip(np.floor(t * BIN_RES).astype(np.int64), 0, BIN_RES - 1)

    # The 27 cell offsets in the order the shader scans its neighborhood in.
    cell_offsets = [(dx, dy, dz)
                    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    out = np.empty((n, k), dtype=np.int64)
    for i in range(n):
        got = []
        for offset in cell_offsets:
            if len(got) >= k:
                break
            c = c0[i] + np.array(offset)
            if np.any(c < 0) or np.any(c >= BIN_RES):
                continue
            idx = (c[0] * BIN_RES + c[1]) * BIN_RES + c[2]
            members = order[starts[idx]:starts[idx] + counts[idx]]
            others = [int(j) for j in members if j != i]
            got.extend(others[:k - len(got)])
        got.extend([i] * (k - len(got)))
        out[i] = got[:k]
    return out


def cell_index_numpy(coords, lo, span):
    t = (np.asarray(coords, dtype=np.float64) - np.asarray(lo)) / np.maximum(
        np.asarray(span, dtype=np.float64), 1e-9)
    c = np.clip(np.floor(t * BIN_RES).astype(np.int64), 0, BIN_RES - 1)
    return (c[:, 0] * BIN_RES + c[:, 1]) * BIN_RES + c[:, 2]


_INTEGRATE_SRC = f"""
ivec2 scig_texel(int i) {{ return ivec2(i % {TEX_ROW}, i / {TEX_ROW}); }}

void main()
{{
  int i = int(gl_GlobalInvocationID.x);
  if (i >= u_count) {{ return; }}
  ivec2 t = scig_texel(i);

  vec4 self = imageLoad(pos_in, t);
  vec3 p = self.xyz;
  float m = self.w;
  vec3 force = imageLoad(force_img, t).xyz;

  // Gravity, towards the centroid. Strong gravity drops the 1/d, which is the
  // only difference between the two modes.
  vec3 delta = p - u_center;
  if (u_strong_gravity != 0) {{
    force -= (u_gravity * m) * delta;
  }} else {{
    float d = max(length(delta), 1e-9);
    force -= ((u_gravity * m) / d) * delta;
  }}
  if (u_dims < 3) {{ force.z = 0.0; }}

  // ForceAtlas2's per-node step, damped by how much this node reversed
  // direction since the previous force. That damping is what stops a node
  // pulled by two neighbors at once from oscillating.
  vec3 prev = imageLoad(prev_img, t).xyz;
  float swing = length(force - prev);
  float factor = u_speed / (1.0 + u_speed * sqrt(swing));
  vec3 disp = force * factor;

  // No node moves more than one optimal distance in a step.
  float n = length(disp);
  if (n > u_cap) {{ disp *= u_cap / n; }}

  vec3 np_ = p + disp;
  if (u_dims < 3) {{ np_.z = 0.0; }}
  imageStore(pos_out, t, vec4(np_, m));
  imageStore(prev_out, t, vec4(force, 0.0));
  // Everything the global reduction needs, per node: the two mass-weighted sums
  // ForceAtlas2's adaptive speed is a ratio of, and the displacement the caller
  // reports as "how much did it move".
  float traction = length(0.5 * (force + prev));
  imageStore(step_img, t, vec4(m * swing, m * traction, length(disp), 0.0));
}}
"""

# Gravity needs the centroid, the loop's last O(N) transfer. Fixed lanes stride
# the array accumulating privately, and one readback of the 256 partials sums
# them exactly with no atomics, 4 KB at any graph size. The same pass totals the
# speed's two mass-weighted terms from the previous step, so the speed runs one
# step stale; making it current would cost three dispatches with barriers.
REDUCE_LANES = 256

_REDUCE_SRC = f"""
ivec2 scig_texel(int i) {{ return ivec2(i % {TEX_ROW}, i / {TEX_ROW}); }}

void main()
{{
  int lane = int(gl_GlobalInvocationID.x);
  if (lane >= {REDUCE_LANES}) {{ return; }}
  vec3 psum = vec3(0.0);
  vec3 ssum = vec3(0.0);
  for (int i = lane; i < u_count; i += {REDUCE_LANES}) {{
    ivec2 t = scig_texel(i);
    psum += imageLoad(pos_img, t).xyz;
    ssum += imageLoad(step_img, t).xyz;
  }}
  imageStore(part_pos, scig_texel(lane), vec4(psum, 0.0));
  imageStore(part_step, scig_texel(lane), vec4(ssum, 0.0));
}}
"""

_RED_SHADER = None
_RED_TRIED = False


def _reduce_shader():
    global _RED_SHADER, _RED_TRIED
    if _RED_TRIED:
        return _RED_SHADER
    try:
        if gpu.platform.backend_type_get() == 'NONE':
            return None
    except Exception:  # noqa: BLE001
        return None
    _RED_TRIED = True
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(GROUP)
        info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_img", qualifiers={'READ'})
        info.image(1, 'RGBA32F', 'FLOAT_2D', "step_img", qualifiers={'READ'})
        info.image(2, 'RGBA32F', 'FLOAT_2D', "part_pos", qualifiers={'WRITE'})
        info.image(3, 'RGBA32F', 'FLOAT_2D', "part_step", qualifiers={'WRITE'})
        info.push_constant('INT', "u_count")
        info.compute_source(_REDUCE_SRC)
        _RED_SHADER = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001
        _RED_SHADER = None
    return _RED_SHADER


_INT_SHADER = None
_INT_TRIED = False


def _integrate_shader():
    global _INT_SHADER, _INT_TRIED
    if _INT_TRIED:
        return _INT_SHADER
    try:
        if gpu.platform.backend_type_get() == 'NONE':
            return None
    except Exception:  # noqa: BLE001
        return None
    _INT_TRIED = True
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(GROUP)
        info.image(0, 'RGBA32F', 'FLOAT_2D', "pos_in", qualifiers={'READ'})
        info.image(1, 'RGBA32F', 'FLOAT_2D', "force_img", qualifiers={'READ'})
        info.image(2, 'RGBA32F', 'FLOAT_2D', "prev_img", qualifiers={'READ'})
        info.image(3, 'RGBA32F', 'FLOAT_2D', "pos_out", qualifiers={'WRITE'})
        info.image(4, 'RGBA32F', 'FLOAT_2D', "prev_out", qualifiers={'WRITE'})
        info.image(5, 'RGBA32F', 'FLOAT_2D', "step_img", qualifiers={'WRITE'})
        info.push_constant('INT', "u_count")
        info.push_constant('INT', "u_strong_gravity")
        info.push_constant('INT', "u_dims")
        info.push_constant('FLOAT', "u_gravity")
        info.push_constant('FLOAT', "u_speed")
        info.push_constant('FLOAT', "u_cap")
        info.push_constant('VEC3', "u_center")
        info.compute_source(_INTEGRATE_SRC)
        _INT_SHADER = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001
        _INT_SHADER = None
    return _INT_SHADER


def _blank(count):
    rows = max(1, int(np.ceil(count / TEX_ROW)))
    tex = gpu.types.GPUTexture((TEX_ROW, rows), format='RGBA32F')
    tex.clear(format='FLOAT', value=(0.0, 0.0, 0.0, 0.0))
    return tex


class GpuSim:
    """Positions live on the GPU and stay there. One step is two dispatches, with
    two of every texture, since an invocation cannot read what another writes."""

    def __init__(self, coords, edges, mass, k, params):
        self.n = int(coords.shape[0])
        self.k = float(k)
        self.params = dict(params)
        self.pos_a = build_nodes(coords, mass)
        self.pos_b = _blank(self.n)
        self.prev_a = _blank(self.n)
        self.prev_b = _blank(self.n)
        self.force = _blank(self.n)
        self.step_tex = _blank(self.n)
        self.adj = build_adjacency(edges, self.n)
        self.near = None
        self.near_k = 0
        self.bins = None
        self.iteration = 0
        self.part_pos = _blank(REDUCE_LANES)
        self.part_step = _blank(REDUCE_LANES)
        self.speed = 1.0
        self.jitter_tolerance = float(self.params.get("jitter_tolerance", 1.0))
        self.moved = 0.0
        self.center = np.zeros(3, dtype=np.float32)

    def reduce(self):
        """Centroid, adaptive speed and movement: 2 x 256 texels back, always."""
        shader = _reduce_shader()
        if shader is None:
            return False
        shader.bind()
        shader.image('pos_img', self.pos_a)
        shader.image('step_img', self.step_tex)
        shader.image('part_pos', self.part_pos)
        shader.image('part_step', self.part_step)
        shader.uniform_int("u_count", self.n)
        gpu.compute.dispatch(shader, (REDUCE_LANES + GROUP - 1) // GROUP, 1, 1)

        psum = np.asarray(self.part_pos.read()).reshape(-1, 4)[:REDUCE_LANES, :3]
        ssum = np.asarray(self.part_step.read()).reshape(-1, 4)[:REDUCE_LANES, :3]
        self.center = (psum.sum(axis=0) / max(self.n, 1)).astype(np.float32)
        total_swing, total_traction, moved = ssum.sum(axis=0)
        self.moved = float(moved)

        # ForceAtlas2's rule, clamped to 50% per step against oscillation.
        if total_swing > 0.0:
            target = self.jitter_tolerance * total_traction / total_swing
            self.speed = float(np.clip(target, self.speed * 0.5,
                                       self.speed * 1.5))
        self.speed = float(np.clip(self.speed, 1e-4, 10.0))
        return True

    def advance(self, cells, cell_of, gravity, strong_gravity=False, dims=3):
        """One step, centroid and speed from :meth:`reduce`. The first iteration
        has no previous force to swing against and keeps the starting speed."""
        if not self.reduce():
            return False
        return self.step(cells, cell_of, self.center, self.speed, gravity,
                         strong_gravity=strong_gravity, dims=dims)

    def set_near(self, near_idx):
        if near_idx is None:
            self.near, self.near_k = None, 0
        else:
            self.near, self.near_k = build_near(near_idx)

    def refresh_structures(self, lo, span, k=16):
        """Rebuild the bins and neighbor list on the GPU. Only the 4096 cell
        counts come back: 0.21 ms against ``_refresh_near``'s 329 ms at 400k."""
        if self.bins is None or self.bins.n != self.n:
            self.bins = SpatialBins(self.n)
        if not self.bins.build(self.pos_a, lo, span):
            return False
        tex = neighbors_from_bins(self.bins, self.pos_a, lo, span, k)
        if tex is None:
            return False
        self.near, self.near_k = tex, int(k)
        return True

    def usable(self):
        return (_force_shader() is not None
                and _integrate_shader() is not None
                and self.adj is not None)

    def positions(self):
        return np.asarray(self.pos_a.read()).reshape(-1, 4)[:self.n, :3].copy()

    def step(self, cells, cell_of, center, speed, gravity, strong_gravity=False,
             dims=3):
        fshader = _force_shader()
        ishader = _integrate_shader()
        if fshader is None or ishader is None:
            return False

        p = dict(self.params)
        p["cells"] = int(p.get("cells", 0))
        tex = compute_forces(self.pos_a, self.adj, cells, cell_of, p, self.n,
                             near=self.near, near_k=self.near_k)
        if tex is None:
            return False

        ishader.bind()
        ishader.image('pos_in', self.pos_a)
        ishader.image('force_img', tex)
        ishader.image('prev_img', self.prev_a)
        ishader.image('pos_out', self.pos_b)
        ishader.image('prev_out', self.prev_b)
        ishader.image('step_img', self.step_tex)
        ishader.uniform_int("u_count", self.n)
        ishader.uniform_int("u_strong_gravity", 1 if strong_gravity else 0)
        ishader.uniform_int("u_dims", int(dims))
        ishader.uniform_float("u_gravity", float(gravity))
        ishader.uniform_float("u_speed", float(speed))
        ishader.uniform_float("u_cap", float(self.k))
        ishader.uniform_float("u_center", tuple(float(c) for c in center))
        gpu.compute.dispatch(ishader, (self.n + GROUP - 1) // GROUP, 1, 1)

        self.pos_a, self.pos_b = self.pos_b, self.pos_a
        self.prev_a, self.prev_b = self.prev_b, self.prev_a
        self.iteration += 1
        return True
