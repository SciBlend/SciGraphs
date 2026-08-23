# Jump flooding: the distance and feature transforms, shared by sbeb_gpu and
# cushion shading. Felzenszwalb's exact scan is a dependency chain and the Python
# `gpu` module has no shader storage to carry one; jump flooding answers in
# log2(n) dispatches with all state in one RGBA32F texture. It is not exact, a
# seed can hide behind a nearer one, and Rong and Tan's JFA+1 pass removes most
# of what is left. Tests measure both against brute force.

import numpy as np

import gpu

from .glsl_source import glsl

# 8x8 tiles: a square wavefront minimizes the step-1 halo (10x10 vs 3x66).
GROUP = 8

# .x/.y hold the nearest seed's texel coords, .z the found flag, .w is spare.
FEATURE_FORMAT = 'RGBA32F'

# No-seed marker; negative so a bug ignoring .z reads a huge distance.
EMPTY = (-1.0, -1.0, 0.0, 0.0)


_SHADERS = {}


def _shader(name):
    """Compile one kernel once. Failures before ``gpu.init()`` are not cached."""
    if name in _SHADERS:
        return _SHADERS[name]
    try:
        if gpu.platform.backend_type_get() == 'NONE':
            return None
    except Exception:  # noqa: BLE001 - raises before gpu.init()
        return None
    try:
        info = gpu.types.GPUShaderCreateInfo()
        info.local_group_size(GROUP, GROUP)
        info.push_constant('IVEC2', "u_size")
        if name == "seed":
            info.image(0, 'R32UI', 'UINT_2D', "mask_img", qualifiers={'READ'})
            info.image(1, 'R32UI', 'UINT_2D', "peak_img", qualifiers={'READ'})
            info.image(2, FEATURE_FORMAT, 'FLOAT_2D', "out_img",
                       qualifiers={'WRITE'})
            info.push_constant('FLOAT', "u_fraction")
            info.push_constant('INT', "u_floor")
            src = glsl("jfa_seed")
        elif name == "step":
            info.image(0, FEATURE_FORMAT, 'FLOAT_2D', "in_img",
                       qualifiers={'READ'})
            info.image(1, FEATURE_FORMAT, 'FLOAT_2D', "out_img",
                       qualifiers={'WRITE'})
            info.push_constant('INT', "u_step")
            src = glsl("jfa_step")
        elif name == "distance":
            info.image(0, FEATURE_FORMAT, 'FLOAT_2D', "ft_img",
                       qualifiers={'READ'})
            info.image(1, 'R32F', 'FLOAT_2D', "out_img", qualifiers={'WRITE'})
            info.push_constant('FLOAT', "u_empty")
            src = glsl("jfa_distance")
        elif name == "level":
            info.image(0, FEATURE_FORMAT, 'FLOAT_2D', "ft_img",
                       qualifiers={'READ'})
            info.image(1, 'R32UI', 'UINT_2D', "out_img", qualifiers={'WRITE'})
            info.push_constant('FLOAT', "u_radius")
            src = glsl("jfa_level")
        elif name == "boundary":
            info.image(0, 'R32UI', 'UINT_2D', "mask_img", qualifiers={'READ'})
            info.image(1, 'R32UI', 'UINT_2D', "out_img", qualifiers={'WRITE'})
            src = glsl("jfa_boundary")
        elif name == "skeleton":
            info.image(0, FEATURE_FORMAT, 'FLOAT_2D', "ft_img",
                       qualifiers={'READ'})
            info.image(1, 'R32UI', 'UINT_2D', "inside_img", qualifiers={'READ'})
            info.image(2, 'R32UI', 'UINT_2D', "out_img", qualifiers={'WRITE'})
            info.push_constant('FLOAT', "u_tau")
            info.push_constant('FLOAT', "u_min_dist")
            src = glsl("jfa_skeleton")
        else:
            return None
        info.compute_source(src)
        _SHADERS[name] = gpu.shader.create_from_info(info)
    except Exception:  # noqa: BLE001 - unsupported backend
        _SHADERS[name] = None
    return _SHADERS[name]


def available():
    return all(_shader(n) is not None for n in
               ("seed", "step", "distance", "level", "boundary", "skeleton"))


def schedule(res, max_step=None, extra=1):
    """Jump sizes, largest first, then ``extra`` JFA+1 passes at 1. ``max_step``
    caps the first jump; past roughly twice that radius the field is wrong."""
    first = 1
    while first * 2 < max(int(res), 2):
        first *= 2
    if max_step is not None:
        cap = 1
        while cap < max(int(max_step), 1):
            cap *= 2
        first = min(first, cap)
    steps = []
    s = first
    while s >= 1:
        steps.append(s)
        s //= 2
    steps.extend([1] * max(0, int(extra)))
    return steps


def feature_texture(res):
    tex = gpu.types.GPUTexture((int(res), int(res)), format=FEATURE_FORMAT)
    tex.clear(format='FLOAT', value=EMPTY)
    return tex


def mask_texture(res, value=0):
    tex = gpu.types.GPUTexture((int(res), int(res)), format='R32UI')
    tex.clear(format='UINT', value=(int(value),))
    return tex


def _groups(res):
    return (int(res) + GROUP - 1) // GROUP


def seed(mask_tex, peak_tex, out_tex, res, fraction=0.0, floor=1):
    """A texel seeds itself at ``max(floor, fraction * peak)``, ``peak`` from
    texel (0, 0) of ``peak_tex``, which is bound even at ``fraction=0``."""
    shader = _shader("seed")
    if shader is None:
        return False
    shader.bind()
    shader.image('mask_img', mask_tex)
    shader.image('peak_img', peak_tex)
    shader.image('out_img', out_tex)
    shader.uniform_int("u_size", (int(res), int(res)))
    shader.uniform_float("u_fraction", float(fraction))
    shader.uniform_int("u_floor", int(floor))
    gpu.compute.dispatch(shader, _groups(res), _groups(res), 1)
    return True


def flood(ft_a, ft_b, res, max_step=None, extra=1):
    """Use the return value, not the texture you seeded: in place, a round would
    read seeds it had already replaced."""
    shader = _shader("step")
    if shader is None:
        return None
    g = _groups(res)
    src, dst = ft_a, ft_b
    for s in schedule(res, max_step=max_step, extra=extra):
        shader.bind()
        shader.image('in_img', src)
        shader.image('out_img', dst)
        shader.uniform_int("u_size", (int(res), int(res)))
        shader.uniform_int("u_step", int(s))
        gpu.compute.dispatch(shader, g, g, 1)
        src, dst = dst, src
    return src


def distance_texture(ft_tex, out_tex, res, empty=-1.0):
    """|texel - feature| into R32F, the field Ersoy's shading mode samples."""
    shader = _shader("distance")
    if shader is None:
        return None
    shader.bind()
    shader.image('ft_img', ft_tex)
    shader.image('out_img', out_tex)
    shader.uniform_int("u_size", (int(res), int(res)))
    shader.uniform_float("u_empty", float(empty))
    gpu.compute.dispatch(shader, _groups(res), _groups(res), 1)
    return out_tex


def level_set(ft_tex, out_tex, res, radius):
    shader = _shader("level")
    if shader is None:
        return False
    shader.bind()
    shader.image('ft_img', ft_tex)
    shader.image('out_img', out_tex)
    shader.uniform_int("u_size", (int(res), int(res)))
    shader.uniform_float("u_radius", float(radius))
    gpu.compute.dispatch(shader, _groups(res), _groups(res), 1)
    return True


def boundary(mask_tex, out_tex, res):
    shader = _shader("boundary")
    if shader is None:
        return False
    shader.bind()
    shader.image('mask_img', mask_tex)
    shader.image('out_img', out_tex)
    shader.uniform_int("u_size", (int(res), int(res)))
    gpu.compute.dispatch(shader, _groups(res), _groups(res), 1)
    return True


def skeleton(ft_tex, inside_tex, out_tex, res, tau=1.0, min_dist=2.0):
    shader = _shader("skeleton")
    if shader is None:
        return False
    shader.bind()
    shader.image('ft_img', ft_tex)
    shader.image('inside_img', inside_tex)
    shader.image('out_img', out_tex)
    shader.uniform_int("u_size", (int(res), int(res)))
    shader.uniform_float("u_tau", float(tau))
    shader.uniform_float("u_min_dist", float(min_dist))
    gpu.compute.dispatch(shader, _groups(res), _groups(res), 1)
    return True


def read(ft_tex):
    """(res, res, 3) of (fx, fy, valid), [y][x]. The reserved channel drops."""
    arr = np.asarray(ft_tex.read())
    return np.ascontiguousarray(arr[..., :3])


def read_mask(mask_tex):
    return np.asarray(mask_tex.read()).reshape(mask_tex.height,
                                               mask_tex.width).astype(np.int64)


def empty_field(res):
    ft = np.zeros((int(res), int(res), 3), dtype=np.float32)
    ft[..., 0] = -1.0
    ft[..., 1] = -1.0
    return ft


def seed_numpy(mask, fraction=0.0, floor=1, peak=None):
    mask = np.asarray(mask)
    if peak is None:
        peak = int(mask.max()) if mask.size else 0
    cut = max(int(floor), int(float(fraction) * float(peak)))
    res = mask.shape[0]
    ft = empty_field(res)
    ys, xs = np.nonzero(mask >= cut)
    ft[ys, xs, 0] = xs
    ft[ys, xs, 1] = ys
    ft[ys, xs, 2] = 1.0
    return ft


def _shift(ft, dx, dy):
    """``ft`` at (x + dx, y + dy); out-of-range texels read as empty."""
    h, w = ft.shape[:2]
    out = np.zeros_like(ft)
    out[..., 0] = -1.0
    out[..., 1] = -1.0
    dy0, dy1 = max(0, -dy), h - max(0, dy)
    dx0, dx1 = max(0, -dx), w - max(0, dx)
    if dy0 < dy1 and dx0 < dx1:
        out[dy0:dy1, dx0:dx1] = ft[dy0 + dy:dy1 + dy, dx0 + dx:dx1 + dx]
    return out


def flood_numpy(ft, max_step=None, extra=1):
    """The kernel round for round, tie-breaks included, one pass per round."""
    res = ft.shape[0]
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    cur = ft.copy()
    for s in schedule(res, max_step=max_step, extra=extra):
        best = cur.copy()
        bd = np.where(cur[..., 2] > 0.5,
                      np.hypot(cur[..., 0] - xx, cur[..., 1] - yy),
                      np.float32(3.4e38))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cand = _shift(cur, dx * s, dy * s)
                d = np.where(cand[..., 2] > 0.5,
                             np.hypot(cand[..., 0] - xx, cand[..., 1] - yy),
                             np.float32(3.4e38))
                take = d < bd
                bd = np.where(take, d, bd)
                best[take] = cand[take]
        cur = best
    return cur


def exact_numpy(mask, fraction=0.0, floor=1, peak=None):
    """Nearest seed by exhaustive search: a kd-tree oracle could itself be wrong."""
    seeds = seed_numpy(mask, fraction=fraction, floor=floor, peak=peak)
    ys, xs = np.nonzero(seeds[..., 2] > 0.5)
    res = mask.shape[0]
    out = empty_field(res)
    if xs.size == 0:
        return out
    yy, xx = np.mgrid[0:res, 0:res]
    d2 = ((xs[None, None, :] - xx[..., None]) ** 2
          + (ys[None, None, :] - yy[..., None]) ** 2)
    pick = np.argmin(d2, axis=2)
    out[..., 0] = xs[pick]
    out[..., 1] = ys[pick]
    out[..., 2] = 1.0
    return out


def distance_numpy(ft, empty=-1.0):
    res = ft.shape[0]
    yy, xx = np.mgrid[0:res, 0:res].astype(np.float32)
    d = np.hypot(ft[..., 0] - xx, ft[..., 1] - yy)
    return np.where(ft[..., 2] > 0.5, d, np.float32(empty))


def level_set_numpy(ft, radius):
    d = distance_numpy(ft, empty=np.inf)
    return ((ft[..., 2] > 0.5) & (d <= float(radius))).astype(np.int64)


def boundary_numpy(mask):
    m = np.asarray(mask) != 0
    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), dtype=bool)
    pad[1:-1, 1:-1] = m
    inner = (pad[1:-1, 2:] & pad[1:-1, :-2] & pad[2:, 1:-1] & pad[:-2, 1:-1])
    return (m & ~inner).astype(np.int64)


def skeleton_numpy(ft, inside, tau=1.0, min_dist=2.0):
    res = ft.shape[0]
    d = distance_numpy(ft, empty=0.0)
    sep = np.zeros((res, res), dtype=np.float32)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        g = _shift(ft, dx, dy)
        s = np.where(g[..., 2] > 0.5,
                     np.hypot(g[..., 0] - ft[..., 0], g[..., 1] - ft[..., 1]),
                     0.0)
        sep = np.maximum(sep, s)
    ok = ((np.asarray(inside) != 0) & (ft[..., 2] > 0.5)
          & (d >= float(min_dist)) & (sep > float(tau) * d))
    return ok.astype(np.int64)


def flood_from_mask(mask, max_step=None, extra=1, fraction=0.0, floor=1,
                    peak=None):
    return flood_numpy(seed_numpy(mask, fraction=fraction, floor=floor,
                                  peak=peak),
                       max_step=max_step, extra=extra)
