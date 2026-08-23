# Multilevel agglomerative bundling by ink minimization (Gansner et al. 2011),
# the compute half. An edge is a chain a -> p -> q -> b, with every edge in a
# bundle sharing p and q:
#
#     ink(u) = sum_{s in S} |s - p| + |p - q| + sum_{t in T} |t - q|
#
# The GPU does the kNN and scores the k*B candidates. The greedy merge and the
# route assembly stay on the CPU.

import numpy as np

import gpu

from . import bundle_gpu

from ...core.render.bundling import mingle as _mingle  # noqa: E402,F401
from ...core.render.bundling.mingle import (  # noqa: E402,F401
    DEFAULTS, MAX_GROUP, MAX_K, PASSES_PER_ROUND, SOLVE_EPS_FRAC, SOLVE_ITERS,
    _assemble, _group_ink, _level_size, _proximity_points, _seed_level,
    _solve_numpy, _trunk_level, candidates, drawn_ink, knn_numpy, matching,
    merge, score_numpy, settings,
)

TEX_ROW = bundle_gpu.TEX_ROW

# As in sim_gpu and fdeb_gpu: a wavefront on AMD, two warps on NVIDIA.
GROUP = 64

# Workgroups per dispatch dimension, spilling into y past this.
MAX_GROUPS = 32768

# Set False to force the oracle. Only the test does this.
PREFER_GPU = True

# Diagnostics from the last ``produce``. Nothing in the draw path reads it.
LAST = {}


def check(params, edges, ctx):
    """Report whether this mode can route these edges (producer contract)."""
    if params.get("style_type") != 'MINGLE':
        return False, "not agglomerative ink-minimizing bundling"
    if int(params.get("mingle_rounds", DEFAULTS["mingle_rounds"])) < 1:
        return False, "at least one round is needed to merge anything"
    if int(params.get("mingle_neighbors", DEFAULTS["mingle_neighbors"])) < 1:
        return False, "a proximity graph with no neighbors has no candidates"
    if float(params.get("mingle_min_gain", 0.0)) >= 1.0:
        return False, ("a minimum gain of 1 or more asks a merge to save all the "
                       "ink of both edges, which no merge can do")
    if edges.shape[0] < 2:
        return False, "a single edge has nothing to bundle with"
    s = settings(params)
    rows = int(np.ceil(edges.shape[0] * (2 * s["rounds"] + 2) / TEX_ROW))
    if rows > 16384:
        return False, ("%d edges at %d rounds needs a taller control texture "
                       "than the backend allows" % (edges.shape[0], s["rounds"]))
    return True, ""


def usable(params, edges):
    return bundle_gpu.usable('MINGLE', params, edges, {})


_TEXEL = f"""
ivec2 scig_texel(int i) {{ return ivec2(i % {TEX_ROW}, i / {TEX_ROW}); }}

int scig_index()
{{
  return int(gl_GlobalInvocationID.y) * u_row + int(gl_GlobalInvocationID.x);
}}
"""

# One invocation per bundle, walking every other bundle. Brute force: a grid in
# 6D is nearly all empty (3^6 = 729 cells), and a random projection only gets
# close, which turns the greedy merge downstream into a different algorithm.
# Insertion uses strict comparisons, so the k best come out by (distance,
# index), which is what numpy's stable argsort gives. Measured on an RTX 4060,
# round 0 pass 0:
#
#     edges      kNN     scoring    greedy merge     whole run
#       913    0.8 ms     0.8 ms        2.7 ms         31 ms
#      8330    3.5 ms     8.4 ms       33.0 ms        158 ms
#     23373   11.4 ms    25.7 ms      110.4 ms        528 ms
#
# B halves every pass and the full B^2 is paid once, so the quadratic term is 2%
# of the run at 23k edges. The CPU merge is what stays.
_KNN_SRC = _TEXEL + f"""
void main()
{{
  int i = scig_index();
  if (i >= u_count) {{ return; }}
  vec3 p0 = imageLoad(prox_img, scig_texel(2 * i)).xyz;
  vec3 p1 = imageLoad(prox_img, scig_texel(2 * i + 1)).xyz;

  float best_d[{MAX_K}];
  int best_j[{MAX_K}];
  for (int m = 0; m < u_k; ++m) {{ best_d[m] = 1e30; best_j[m] = -1; }}

  for (int j = 0; j < u_count; ++j) {{
    if (j == i) {{ continue; }}
    vec3 q0 = imageLoad(prox_img, scig_texel(2 * j)).xyz - p0;
    vec3 q1 = imageLoad(prox_img, scig_texel(2 * j + 1)).xyz - p1;
    precise float d2 = dot(q0, q0) + dot(q1, q1);
    if (d2 < best_d[u_k - 1]) {{
      int m = u_k - 1;
      while (m > 0 && d2 < best_d[m - 1]) {{
        best_d[m] = best_d[m - 1];
        best_j[m] = best_j[m - 1];
        --m;
      }}
      best_d[m] = d2;
      best_j[m] = j;
    }}
  }}

  for (int m = 0; m < u_k; ++m) {{
    imageStore(near_out, scig_texel(i * u_k + m),
               vec4(float(best_j[m]), best_d[m], 0.0, 0.0));
  }}
}}
"""

# One invocation per candidate merge, addressed from its two bundle indices: a
# flat k*B-way problem, not a traversal. The accumulators here and in the kNN
# are ``precise``, and lengths come from an explicit sqrt(dot(d, d)) rather than
# length(), because Weiszfeld is iterative and Vulkan's SPIR-V compiler
# contracts a + b*c into an FMA where OpenGL does not. In another iterative
# kernel here that one-ULP gap grew into a different converged result.
_SCORE_SRC = _TEXEL + f"""
vec4 scig_pt(int base, int m, int from_target)
{{
  return (from_target != 0) ? imageLoad(dst_img, scig_texel(base + m))
                            : imageLoad(src_img, scig_texel(base + m));
}}

void main()
{{
  int c = scig_index();
  if (c >= u_count) {{ return; }}
  vec4 cd = imageLoad(cand_img, scig_texel(c));
  int u = int(cd.x + 0.5);
  int v = int(cd.y + 0.5);

  vec4 mu0 = imageLoad(meet_img, scig_texel(2 * u));      // p_u, ink(u)
  vec4 mu1 = imageLoad(meet_img, scig_texel(2 * u + 1));  // q_u, live count
  vec4 mv0 = imageLoad(meet_img, scig_texel(2 * v));
  vec4 mv1 = imageLoad(meet_img, scig_texel(2 * v + 1));
  int nu = int(mu1.w + 0.5);
  int nv = int(mv1.w + 0.5);
  int bu = u * {MAX_GROUP};
  int bv = v * {MAX_GROUP};
  float solo = mu0.w + mv0.w;

  float best_gain = -1e30;
  vec3 best_p = mu0.xyz;
  vec3 best_q = mu1.xyz;
  int best_flip = 0;
  float best_move = 0.0;

  for (int flip = 0; flip < 2; ++flip) {{
    // flip 0 pairs the two heads; flip 1 pairs u's head with v's tail, which is
    // the same edge run the other way. An edge has no direction, so both are
    // legal drawings and the cheaper one wins.
    int vh = flip;
    int vt = 1 - flip;

    precise vec3 ca = vec3(0.0);
    precise vec3 cb = vec3(0.0);
    precise float wa0 = 0.0;
    precise float wb0 = 0.0;
    for (int m = 0; m < nu; ++m) {{
      vec4 s = imageLoad(src_img, scig_texel(bu + m));
      ca += s.xyz * s.w; wa0 += s.w;
      vec4 t = imageLoad(dst_img, scig_texel(bu + m));
      cb += t.xyz * t.w; wb0 += t.w;
    }}
    for (int m = 0; m < nv; ++m) {{
      vec4 s = scig_pt(bv, m, vh);
      ca += s.xyz * s.w; wa0 += s.w;
      vec4 t = scig_pt(bv, m, vt);
      cb += t.xyz * t.w; wb0 += t.w;
    }}
    vec3 P = ca / max(wa0, 1e-20);
    vec3 Q = cb / max(wb0, 1e-20);

    float move = 0.0;
    for (int it = 0; it < {SOLVE_ITERS}; ++it) {{
      precise vec3 na = vec3(0.0);
      precise vec3 nb = vec3(0.0);
      precise float wa = 0.0;
      precise float wb = 0.0;
      for (int m = 0; m < nu; ++m) {{
        vec4 s = imageLoad(src_img, scig_texel(bu + m));
        vec3 d = s.xyz - P;
        float w = s.w / max(sqrt(dot(d, d)), u_eps);
        na += s.xyz * w; wa += w;
        vec4 t = imageLoad(dst_img, scig_texel(bu + m));
        vec3 e = t.xyz - Q;
        float x = t.w / max(sqrt(dot(e, e)), u_eps);
        nb += t.xyz * x; wb += x;
      }}
      for (int m = 0; m < nv; ++m) {{
        vec4 s = scig_pt(bv, m, vh);
        vec3 d = s.xyz - P;
        float w = s.w / max(sqrt(dot(d, d)), u_eps);
        na += s.xyz * w; wa += w;
        vec4 t = scig_pt(bv, m, vt);
        vec3 e = t.xyz - Q;
        float x = t.w / max(sqrt(dot(e, e)), u_eps);
        nb += t.xyz * x; wb += x;
      }}
      vec3 pq = P - Q;
      float inv = 1.0 / max(sqrt(dot(pq, pq)), u_eps);
      // Jacobi: both meeting points step from the same previous state. Under
      // Gauss-Seidel the numpy path would also have to reproduce the order the
      // kernel happened to run in.
      vec3 Pn = (na + Q * inv) / (wa + inv);
      vec3 Qn = (nb + P * inv) / (wb + inv);
      if (it == {SOLVE_ITERS} - 1) {{
        vec3 dp = Pn - P;
        vec3 dq = Qn - Q;
        move = max(sqrt(dot(dp, dp)), sqrt(dot(dq, dq)));
      }}
      P = Pn;
      Q = Qn;
    }}

    // Unweighted ink: one line per source, one trunk, one line per target,
    // whatever those lines carry.
    precise float ink = 0.0;
    for (int m = 0; m < nu; ++m) {{
      vec3 d = imageLoad(src_img, scig_texel(bu + m)).xyz - P;
      ink += sqrt(dot(d, d));
      vec3 e = imageLoad(dst_img, scig_texel(bu + m)).xyz - Q;
      ink += sqrt(dot(e, e));
    }}
    for (int m = 0; m < nv; ++m) {{
      vec3 d = scig_pt(bv, m, vh).xyz - P;
      ink += sqrt(dot(d, d));
      vec3 e = scig_pt(bv, m, vt).xyz - Q;
      ink += sqrt(dot(e, e));
    }}
    vec3 pq = P - Q;
    ink += sqrt(dot(pq, pq));

    float g = solo - ink;
    if (g > best_gain) {{
      best_gain = g;
      best_p = P;
      best_q = Q;
      best_flip = flip;
      best_move = move;
    }}
  }}

  imageStore(gain_out, scig_texel(2 * c), vec4(best_p, best_gain));
  imageStore(gain_out, scig_texel(2 * c + 1),
             vec4(best_q, float(best_flip)));
  imageStore(move_out, scig_texel(c), vec4(best_move, 0.0, 0.0, 0.0));
}}
"""

_SHADERS = {}


def _backend_ready():
    try:
        return gpu.platform.backend_type_get() != 'NONE'
    except Exception:  # noqa: BLE001 - raises before gpu.init()
        return False


def _shader(name):
    """Compile one kernel once. None if this backend refuses it."""
    if name in _SHADERS:
        return _SHADERS[name]
    if not _backend_ready():
        return None
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(GROUP)
        info.push_constant('INT', "u_count")
        info.push_constant('INT', "u_row")
        if name == "knn":
            info.image(0, 'RGBA32F', 'FLOAT_2D', "prox_img", qualifiers={'READ'})
            info.image(1, 'RGBA32F', 'FLOAT_2D', "near_out", qualifiers={'WRITE'})
            info.push_constant('INT', "u_k")
            src = _KNN_SRC
        else:
            info.image(0, 'RGBA32F', 'FLOAT_2D', "src_img", qualifiers={'READ'})
            info.image(1, 'RGBA32F', 'FLOAT_2D', "dst_img", qualifiers={'READ'})
            info.image(2, 'RGBA32F', 'FLOAT_2D', "meet_img", qualifiers={'READ'})
            info.image(3, 'RGBA32F', 'FLOAT_2D', "cand_img", qualifiers={'READ'})
            info.image(4, 'RGBA32F', 'FLOAT_2D', "gain_out", qualifiers={'WRITE'})
            info.image(5, 'RGBA32F', 'FLOAT_2D', "move_out", qualifiers={'WRITE'})
            info.push_constant('FLOAT', "u_eps")
            src = _SCORE_SRC
        info.compute_source(src)
        _SHADERS[name] = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001 - unsupported backend, driver refusal
        _SHADERS[name] = None
    return _SHADERS[name]


def available():
    return all(_shader(n) is not None for n in ("knn", "score"))


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


def _dispatch(shader, count):
    """Cover ``count`` invocations, spilling into y past MAX_GROUPS."""
    groups = max(1, (count + GROUP - 1) // GROUP)
    gx = min(groups, MAX_GROUPS)
    gy = (groups + gx - 1) // gx
    shader.uniform_int("u_count", int(count))
    shader.uniform_int("u_row", gx * GROUP)
    gpu.compute.dispatch(shader, gx, gy, 1)


def knn_gpu(x6, k):
    """Return (B, k) neighbor indices from the compute path, or None."""
    sh = _shader("knn")
    if sh is None:
        return None
    b = int(x6.shape[0])
    kk = int(min(k, max(b - 1, 0)))
    if kk == 0:
        return np.full((b, k), -1, dtype=np.int64)
    try:
        flat = np.zeros((b * 2, 4), dtype=np.float32)
        flat[0::2, :3] = x6[:, :3]
        flat[1::2, :3] = x6[:, 3:]
        prox = _texture(flat)
        out = _blank(b * kk)
        sh.bind()
        sh.image('prox_img', prox)
        sh.image('near_out', out)
        sh.uniform_int("u_k", kk)
        _dispatch(sh, b)
        got = np.asarray(out.read()).reshape(-1, 4)[:b * kk, 0]
    except Exception:  # noqa: BLE001 - texture limits, out of VRAM, driver
        return None
    near = np.full((b, k), -1, dtype=np.int64)
    near[:, :kk] = np.rint(got).astype(np.int64).reshape(b, kk)
    return near


def score_gpu(level, cand, eps):
    """Return (gain, flip, p, q, residual) from the compute path, or None."""
    sh = _shader("score")
    if sh is None:
        return None
    c = int(cand.shape[0])
    if c == 0:
        return (np.zeros(0, np.float32), np.zeros(0, np.int32),
                np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32),
                np.zeros(0, np.float32))
    b = int(level["p"].shape[0])
    live = _level_size(level).astype(np.float32)
    try:
        sp = np.zeros((b * MAX_GROUP, 4), dtype=np.float32)
        sp[:, :3] = level["src"].reshape(-1, 3)
        sp[:, 3] = level["sw"].ravel()
        tp = np.zeros((b * MAX_GROUP, 4), dtype=np.float32)
        tp[:, :3] = level["dst"].reshape(-1, 3)
        tp[:, 3] = level["tw"].ravel()
        meet = np.zeros((b * 2, 4), dtype=np.float32)
        meet[0::2, :3] = level["p"]
        meet[0::2, 3] = level["ink"]
        meet[1::2, :3] = level["q"]
        meet[1::2, 3] = live
        cd = np.zeros((c, 4), dtype=np.float32)
        cd[:, 0] = cand[:, 0]
        cd[:, 1] = cand[:, 1]

        src_tex = _texture(sp)
        dst_tex = _texture(tp)
        meet_tex = _texture(meet)
        cand_tex = _texture(cd)
        gain_tex = _blank(c * 2)
        move_tex = _blank(c)

        sh.bind()
        sh.image('src_img', src_tex)
        sh.image('dst_img', dst_tex)
        sh.image('meet_img', meet_tex)
        sh.image('cand_img', cand_tex)
        sh.image('gain_out', gain_tex)
        sh.image('move_out', move_tex)
        sh.uniform_float("u_eps", float(eps))
        _dispatch(sh, c)

        got = np.asarray(gain_tex.read()).reshape(-1, 4)[:c * 2]
        mov = np.asarray(move_tex.read()).reshape(-1, 4)[:c, 0]
    except Exception:  # noqa: BLE001 - texture limits, out of VRAM, driver
        return None
    gain = np.ascontiguousarray(got[0::2, 3], dtype=np.float32)
    out_p = np.ascontiguousarray(got[0::2, :3], dtype=np.float32)
    out_q = np.ascontiguousarray(got[1::2, :3], dtype=np.float32)
    flip = np.rint(got[1::2, 3]).astype(np.int32)
    if not (np.isfinite(gain).all() and np.isfinite(out_p).all()
            and np.isfinite(out_q).all()):
        return None
    return gain, flip, out_p, out_q, np.ascontiguousarray(mov, dtype=np.float32)


def _bundle(coords, edges, params, backend, report=None):
    """``backend`` picks the kNN and the scorer; the rest is CPU either way."""
    if backend == 'gpu':
        return _mingle.bundle(coords, edges, params, report, knn=knn_gpu,
                              score=score_gpu, backend='gpu')
    return _mingle.bundle(coords, edges, params, report)


def _produce_numpy(coords, edges, params, report=None):
    return _bundle(coords, edges, params, 'numpy', report)


def _produce_gpu(coords, edges, params, report=None):
    if not available():
        return None
    try:
        return _bundle(coords, edges, params, 'gpu', report)
    except Exception:  # noqa: BLE001 - a driver refusing is a fallback
        return None


def produce(coords, edges, params, ctx):
    """Produce chained lines through shared meeting points (producer contract)."""
    LAST.clear()
    if PREFER_GPU:
        made = _produce_gpu(coords, edges, params, LAST)
        if made is not None:
            return made
    try:
        return _produce_numpy(coords, edges, params, LAST)
    except Exception:  # noqa: BLE001
        return None


def build(coords, edges, params, node_colors=None, edge_color=None,
          edge_widths=None):
    return bundle_gpu.build(
        'MINGLE', coords, edges, params, ctx={}, node_colors=node_colors,
        edge_color=edge_color, edge_widths=edge_widths)


def draw(data, mvp, beta, shader=None, width_range=None, viewport=None):
    return bundle_gpu.draw(data, mvp, beta, shader=shader,
                           width_range=width_range, viewport=viewport)
