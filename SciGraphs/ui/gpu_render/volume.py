# Nodes splatted into a voxel grid at build time, the grid traversed at draw
# time. Sliced, not ray-marched: a ray-march needs the opaque depth along each
# ray, and in the viewport that is the buffer being drawn into. A slice is a
# screen quad at one NDC depth. Texture RGB is mean color, A is density.

import gpu
import numpy as np

from scigraphs_core.coloring.colormaps import colormap_exists, sample_colormap
def _settings(scene, st=None):
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)


LUT_SIZE = 256

# RGBA16F at 8 bytes a voxel, so this ceiling is 64 MB of GPU memory.
VOXEL_MAX = 8_000_000

# Two box passes give a triangular kernel; a true Gaussian would need scipy.
BLUR_PASSES = 2


def _grid_shape(coords, res, pad):
    """Near-cubic grid over ``coords``, plus ``pad`` empty voxels for the blur.
    The thin axis gets a floor, or a flat z = 0 layout has no volume."""
    lo = coords.min(axis=0).astype(np.float64)
    hi = coords.max(axis=0).astype(np.float64)
    span = hi - lo

    thin = max(float(span.max()), 1e-9) / max(int(res), 1)
    extent = np.maximum(span, thin)
    lo = 0.5 * (lo + hi) - 0.5 * extent

    shape = np.clip(np.rint(res * extent / extent.max()), 1, res).astype(np.int64)
    voxel = extent / shape

    shape = shape + 2 * pad
    lo = lo - pad * voxel

    while int(np.prod(shape)) > VOXEL_MAX:
        shape = np.maximum(shape // 2, 1)
        voxel = voxel * 2.0

    return lo, voxel, shape


def _voxel_index(coords, lo, voxel, shape):
    # x fastest, matching the 3D texture layout.
    v = np.floor((coords - lo) / voxel).astype(np.int64)
    np.clip(v, 0, shape - 1, out=v)
    return (v[:, 2] * shape[1] + v[:, 1]) * shape[0] + v[:, 0]


def _blur_axis(a, radius, axis):
    if radius < 1:
        return a
    n = a.shape[axis]
    k = 2 * radius + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius + 1, radius)
    c = np.cumsum(np.pad(a, pad, mode="constant"), axis=axis)
    hi = np.take(c, np.arange(k, k + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return (hi - lo) / k


def _blur(a, radius, passes=BLUR_PASSES):
    if radius < 1:
        return a
    out = a
    for _ in range(passes):
        for axis in range(3):
            out = _blur_axis(out, radius, axis)
    return out


def build_field(coords, colors=None, res=64, weights=None, smooth=1.5):
    """``colors=None`` skips three quarters of the blur work."""
    n = coords.shape[0]
    if n == 0:
        return None

    radius = max(0, int(round(float(smooth))))
    lo, voxel, shape = _grid_shape(coords, int(res),
                                   pad=BLUR_PASSES * radius + 1)
    n_vox = int(np.prod(shape))
    flat = _voxel_index(coords, lo, voxel, shape)

    w = (np.ones(n, dtype=np.float64) if weights is None
         else np.maximum(np.asarray(weights, dtype=np.float64), 0.0))

    grid_shape = (int(shape[2]), int(shape[1]), int(shape[0]))
    mass = np.bincount(flat, weights=w, minlength=n_vox).reshape(grid_shape)
    sm_mass = _blur(mass, radius)

    if colors is None:
        mean_col = 0.0
    else:
        # Mass-weighted: the only mean defined in voxels holding no node.
        sm_col = np.stack([
            _blur(np.bincount(flat, weights=w * colors[:, c],
                              minlength=n_vox).reshape(grid_shape), radius)
            for c in range(3)
        ], axis=-1)
        mean_col = sm_col / np.maximum(sm_mass, 1e-12)[..., None]

    voxel_volume = float(np.prod(voxel))
    density = sm_mass / max(voxel_volume, 1e-30)

    # The densest voxel is often several times the next, so normalizing by it
    # flattens the rest. Not clipped at 1: that loses a few percent of mass.
    nonzero = density[density > 0.0]
    dmax = float(np.percentile(nonzero, 99.5)) if nonzero.size else 1.0
    if dmax <= 0.0:
        dmax = 1.0

    field = np.empty(grid_shape + (4,), dtype=np.float32)
    field[..., :3] = mean_col
    field[..., 3] = density / dmax

    return {
        "field": field,
        "lo": lo.astype(np.float32),
        "voxel": voxel.astype(np.float32),
        "extent": (voxel * shape).astype(np.float32),
        "shape": grid_shape,
        "dmax": dmax,
        "dtrue": float(density.max()) if n_vox else 0.0,
        "voxel_volume": voxel_volume,
        # Full precision, so a threshold means "nodes per unit volume".
        "node_density": density.reshape(-1)[flat].astype(np.float32),
    }


def build_lut(scene, st=None):
    """Density transfer function, (LUT_SIZE, 4) RGBA; alpha carries the filter."""
    st = _settings(scene, st)
    cmap = st.colormap
    if not colormap_exists(cmap):
        cmap = "viridis"
    rgba = np.asarray(
        sample_colormap(cmap, LUT_SIZE,
                        bool(st.reverse_colormap)),
        dtype=np.float32,
    ).reshape(LUT_SIZE, 4)

    t = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
    floor = float(st.volume_threshold)
    gamma = float(st.volume_falloff)
    opacity = np.clip((t - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0) ** gamma
    rgba[:, 3] = opacity
    return rgba


def upload(data, lut):
    if data is None:
        return None
    field = np.ascontiguousarray(data["field"], dtype=np.float32)
    nz, ny, nx = data["shape"]
    try:
        tex = gpu.types.GPUTexture(
            (nx, ny, nz), format='RGBA16F',
            data=gpu.types.Buffer('FLOAT', field.size, field.ravel()))
        lut_arr = np.ascontiguousarray(lut, dtype=np.float32)
        lut_tex = gpu.types.GPUTexture(
            (LUT_SIZE, 1), format='RGBA32F',
            data=gpu.types.Buffer('FLOAT', lut_arr.size, lut_arr.ravel()))
    except Exception:  # noqa: BLE001 - backend without 3D textures
        return None

    return {
        "tex": tex,
        "lut": lut_tex,
        "lo": data["lo"],
        "extent": data["extent"],
        "shape": data["shape"],
        "dmax": data["dmax"],
        "dtrue": data["dtrue"],
        "voxel_volume": data["voxel_volume"],
        "bytes": int(field.nbytes),
    }


def build(scene, coords, colors, weights=None, st=None):
    """``(gpu_dict, cpu_data)``, either None when that step failed."""
    st = _settings(scene, st)
    use_color = st.volume_mode == 'COLOR'
    data = build_field(
        coords, colors if use_color else None,
        res=int(st.volume_res),
        weights=weights,
        smooth=float(st.volume_smooth),
    )
    if data is None:
        return None, None
    return upload(data, build_lut(scene)), data


def _ndc_bounds(lo, extent, obj_to_clip):
    """``(x0, y0, x1, y1, z0, z1)`` in NDC, None when the box is off screen.
    A box straddling the camera plane gets the whole screen, for the shader."""
    hi = lo + extent
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])], dtype=np.float64)
    homog = np.concatenate([corners, np.ones((8, 1))], axis=1)
    clip = homog @ obj_to_clip.T
    w = clip[:, 3]
    if np.all(w <= 1e-6):
        return None
    if np.any(w <= 1e-6):
        return -1.0, -1.0, 1.0, 1.0, -1.0, 1.0

    ndc = clip[:, :3] / w[:, None]
    lo_n = ndc.min(axis=0)
    hi_n = ndc.max(axis=0)
    if np.any(hi_n < -1.0) or np.any(lo_n > 1.0):
        return None
    x0, y0, z0 = np.maximum(lo_n, -1.0)
    x1, y1, z1 = np.minimum(hi_n, 1.0)
    return float(x0), float(y0), float(x1), float(y1), float(z0), float(z1)


def _slice_depths(z0, z1, count, proj):
    """``count`` slice depths, evenly spaced in *view* depth, not in NDC."""
    edges = np.linspace(z0, z1, count + 1)
    if abs(float(proj[3][3])) < 1e-9:  # perspective
        a = float(proj[2][2])
        b = float(proj[2][3])
        denom = edges + a
        safe = np.abs(denom) > 1e-9
        if safe.all():
            view_z = b / denom
            v_edges = np.linspace(view_z[0], view_z[-1], count + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                edges = b / v_edges - a
            edges = np.clip(np.nan_to_num(edges, nan=z1), min(z0, z1),
                            max(z0, z1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    steps = np.diff(edges)
    return centers, steps


def draw(vol, scene, modelview, proj, st=None):
    """Composite after the opaque graph, leaving depth untouched so it tints."""
    st = _settings(scene, st)
    from . import shaders

    shader = shaders.get_volume_shader()
    if shader is None or vol is None:
        return False

    obj_to_clip = np.asarray(proj, dtype=np.float64) @ np.asarray(
        modelview, dtype=np.float64)
    bounds = _ndc_bounds(vol["lo"].astype(np.float64),
                         vol["extent"].astype(np.float64), obj_to_clip)
    if bounds is None:
        return False
    x0, y0, x1, y1, z0, z1 = bounds
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return False

    try:
        clip_to_obj = np.linalg.inv(obj_to_clip)
    except np.linalg.LinAlgError:
        return False

    to_tex = np.eye(4)
    inv_extent = 1.0 / np.maximum(vol["extent"].astype(np.float64), 1e-12)
    to_tex[:3, :3] = np.diag(inv_extent)
    to_tex[:3, 3] = -vol["lo"].astype(np.float64) * inv_extent
    clip_to_vol = to_tex @ clip_to_obj

    count = max(2, int(st.volume_slices))
    centers, steps = _slice_depths(z0, z1, count, proj)

    batch = shaders.get_volume_batch(shader)
    if batch is None:
        return False

    from mathutils import Matrix
    mat = Matrix([[float(v) for v in row] for row in clip_to_vol])

    prev_blend = gpu.state.blend_get()
    prev_mask = gpu.state.depth_mask_get()
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_mask_set(False)

    shader.bind()
    shader.uniform_sampler("field", vol["tex"])
    shader.uniform_sampler("lut", vol["lut"])
    shader.uniform_float("u_clip_to_vol", mat)
    shader.uniform_float("u_rect", (x0, y0, x1, y1))
    nz, ny, nx = vol["shape"]
    shader.uniform_float("u_shape", (float(nx), float(ny), float(nz)))
    # Gain is optical depth across the grid, so divide out the grid size.
    shader.uniform_float("u_field", (
        float(st.volume_gain) / float(max(nx, ny, nz)),
        1.0 if st.volume_mode == 'COLOR' else 0.0,
        float(st.volume_levels),
        float(st.volume_shell),
    ))

    for i in range(count - 1, -1, -1):
        shader.uniform_float("u_slab", (float(centers[i]), float(steps[i])))
        batch.draw(shader)

    gpu.state.depth_mask_set(prev_mask)
    gpu.state.blend_set(prev_blend)
    return True


def crowded_mask(node_density, percentile):
    """Nodes dense enough to leave to the cloud; a percentile, not a density."""
    if node_density is None or node_density.size == 0:
        return None
    pct = float(np.clip(percentile, 0.0, 100.0))
    if pct >= 100.0:
        return None
    cut = float(np.percentile(node_density, pct))
    mask = node_density > cut
    return mask if mask.any() else None
