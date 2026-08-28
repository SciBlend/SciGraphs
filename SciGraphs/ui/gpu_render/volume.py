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

BOX_MARGIN = 0.15

DMAX_DRIFT = 2.0

MASS_EPS = 1e-12


def _bounds(coords):
    """``coords.min(axis=0)`` on an (N, 3) array runs 11x slower than a column
    at a time: 33.8 ms against 3.0 at a million nodes. An axis that is not
    finite collapses to a point: a diverged simulation must not raise here."""
    lo = np.empty(3, dtype=np.float64)
    hi = np.empty(3, dtype=np.float64)
    for c in range(3):
        col = coords[:, c]
        lo[c] = col.min()
        hi[c] = col.max()
    bad = ~(np.isfinite(lo) & np.isfinite(hi))
    if bad.any():
        lo[bad] = 0.0
        hi[bad] = 0.0
    return lo, hi


def _grid_shape(coords, res, pad, margin=0.0):
    """Near-cubic grid over ``coords``, plus ``pad`` empty voxels for the blur.
    The thin axis gets a floor, or a flat z = 0 layout has no volume. Returns
    the fitted box too: what a frozen grid tests the next frame against."""
    lo, hi = _bounds(coords)
    span = hi - lo

    thin = max(float(span.max()), 1e-9) / max(int(res), 1)
    extent = np.maximum(span, thin) * (1.0 + 2.0 * float(margin))
    lo = 0.5 * (lo + hi) - 0.5 * extent
    fit = (lo.copy(), lo + extent)

    shape = np.clip(np.rint(res * extent / extent.max()), 1, res).astype(np.int64)
    voxel = extent / shape

    shape = shape + 2 * pad
    lo = lo - pad * voxel

    while int(np.prod(shape)) > VOXEL_MAX:
        shape = np.maximum(shape // 2, 1)
        voxel = voxel * 2.0

    return lo, voxel, shape, fit


def _voxel_index(coords, lo, voxel, shape):
    """x fastest, matching the 3D texture layout. A column at a time, in place:
    the (N, 3) form cost 36.0 ms at a million nodes against 9.8. Still float64,
    because float32 puts every node of a graph sitting at 1e7 in the wrong
    voxel. Clipping after the cast is what keeps a NaN coordinate in bounds."""
    inv = 1.0 / np.asarray(voxel, dtype=np.float64)
    nx, ny = int(shape[0]), int(shape[1])
    flat = None
    for c, mul in ((0, 1), (1, nx), (2, nx * ny)):
        t = coords[:, c].astype(np.float64)
        t -= lo[c]
        t *= inv[c]
        np.floor(t, out=t)
        i = t.astype(np.int64)
        np.clip(i, 0, int(shape[c]) - 1, out=i)
        if flat is None:
            flat = i
        else:
            i *= mul
            flat += i
    return flat


def _blur_axis(a, radius, axis, out, tmp, gain=None):
    """One box pass, ``out[i] = sum(a[i-r : i+r])`` with zeros outside, and no
    division unless ``gain`` says so. Sliced, not gathered: ``np.take`` with an
    index array copies where a slice is a view, and the padding the offsets used
    to need was a whole extra array. ``tmp`` may alias ``a``, ``out`` may not."""
    n = a.shape[axis]

    def cut(x, lo, hi):
        s = [slice(None)] * a.ndim
        s[axis] = slice(lo, hi)
        return x[tuple(s)]

    c = np.cumsum(a, axis=axis, out=tmp)
    b0 = min(radius + 1, n)
    b1 = max(n - radius, b0)
    m = min(max(n - radius, 0), b0)
    cut(out, 0, m)[...] = cut(c, radius, radius + m)
    if m < b0:
        cut(out, m, b0)[...] = cut(c, n - 1, n)
    if b1 > b0:
        np.subtract(cut(c, b0 + radius, b1 + radius),
                    cut(c, b0 - radius - 1, b1 - radius - 1),
                    out=cut(out, b0, b1))
    if n > b1:
        np.subtract(cut(c, n - 1, n),
                    cut(c, b1 - radius - 1, n - radius - 1),
                    out=cut(out, b1, n))
    if gain is not None:
        out *= gain
    return out


def _blur(a, radius, passes=BLUR_PASSES, ping=None, out=None):
    """float32 throughout, and the 1 / k of every pass folded into the last:
    50 ms against 123 for a 138-cubed grid. The intermediate passes run k times
    hot, which float32 has room for, and the 1e-5 relative error left in the
    visible voxels is far under what RGBA16F carries anyway."""
    if radius < 1:
        if out is None:
            return a
        np.copyto(out, a)
        return out
    if ping is None:
        ping = (np.empty(a.shape, dtype=np.float32),
                np.empty(a.shape, dtype=np.float32))
    total = 3 * passes
    gain = float(2 * radius + 1) ** -total
    src = a
    for i in range(total):
        last = i == total - 1
        dst = out if (last and out is not None) else ping[i & 1]
        _blur_axis(src, radius, i // passes, dst, ping[1 - (i & 1)],
                   gain=gain if last else None)
        src = dst
    return src


def _scratch(state, grid_shape, want_color):
    """Blur ping-pong plus the outputs. Kept on ``state`` across frames: four
    grids and the field, 84 MB at res=128, versus repaging them every frame."""
    buf = None if state is None else state.get("buf")
    if buf is None or buf["shape"] != grid_shape \
            or (want_color and buf["col"] is None):
        buf = {
            "shape": grid_shape,
            "ping": (np.empty(grid_shape, dtype=np.float32),
                     np.empty(grid_shape, dtype=np.float32)),
            "mass": np.empty(grid_shape, dtype=np.float32),
            "col": np.empty(grid_shape, dtype=np.float32) if want_color else None,
            "field": np.zeros(grid_shape + (4,), dtype=np.float32),
            "zeroed": True,
        }
        if state is not None:
            state["buf"] = buf
    return buf


def build_field(coords, colors=None, res=64, weights=None, smooth=1.5,
                state=None):
    """``colors=None`` skips three quarters of the blur work. Pass an empty dict
    as ``state`` to freeze the grid and keep the scratch for :func:`refresh_field`;
    ``node_density`` is then None, since a moving layout cannot re-cut the
    batches it feeds anyway."""
    n = coords.shape[0]
    if n == 0:
        return None

    radius = max(0, int(round(float(smooth))))
    refit = state is None or state.get("shape") is None
    if refit:
        lo, voxel, shape, fit = _grid_shape(
            coords, int(res), pad=BLUR_PASSES * radius + 1,
            margin=0.0 if state is None else BOX_MARGIN)
    else:
        lo, voxel, shape, fit = (state["lo"], state["voxel"], state["shape"],
                                 state["fit"])

    n_vox = int(np.prod(shape))
    flat = _voxel_index(coords, lo, voxel, shape)

    w = (None if weights is None
         else np.maximum(np.asarray(weights, dtype=np.float64), 0.0))

    grid_shape = (int(shape[2]), int(shape[1]), int(shape[0]))
    buf = _scratch(state, grid_shape, colors is not None)
    mass = np.bincount(flat, weights=w, minlength=n_vox).reshape(grid_shape)
    sm_mass = _blur(mass, radius, ping=buf["ping"], out=buf["mass"])

    voxel_volume = float(np.prod(voxel))
    scale = 1.0 / max(voxel_volume, 1e-30)
    dtrue = float(sm_mass.max()) * scale

    if refit:
        nonzero = sm_mass[sm_mass > 0.0]
        dmax = (float(np.percentile(nonzero, 99.5)) * scale
                if nonzero.size else 1.0)
        if dmax <= 0.0:
            dmax = 1.0
    else:
        dmax = state["dmax"]

    field = buf["field"]
    np.multiply(sm_mass, scale / dmax, out=field[..., 3])

    node_density = None
    if state is None:
        # Full precision, so a threshold means "nodes per unit volume".
        node_density = sm_mass.reshape(-1)[flat] * np.float32(scale)

    if colors is None:
        if not buf["zeroed"]:
            field[..., :3] = 0.0
            buf["zeroed"] = True
    else:
        buf["zeroed"] = False
        np.maximum(sm_mass, MASS_EPS, out=sm_mass)
        np.reciprocal(sm_mass, out=sm_mass)
        for c in range(3):
            cw = colors[:, c] if w is None else w * colors[:, c]
            sm_col = _blur(
                np.bincount(flat, weights=cw, minlength=n_vox
                            ).reshape(grid_shape),
                radius, ping=buf["ping"], out=buf["col"])
            np.multiply(sm_col, sm_mass, out=sm_col)
            field[..., c] = sm_col

    if state is not None:
        state.update({
            "lo": lo, "voxel": voxel, "shape": shape, "fit": fit,
            "dmax": dmax, "dtrue": dtrue, "count": n,
            "res": int(res), "smooth": float(smooth),
            "colors": colors, "weights": weights,
        })
        if refit:
            state["peak"] = dtrue
            state["span"] = (float((fit[1] - fit[0]).max())
                             / (1.0 + 2.0 * BOX_MARGIN))

    return {
        "field": field,
        "lo": lo.astype(np.float32),
        "voxel": voxel.astype(np.float32),
        "extent": (voxel * shape).astype(np.float32),
        "shape": grid_shape,
        "dmax": dmax,
        "dtrue": dtrue,
        "voxel_volume": voxel_volume,
        "node_density": node_density,
    }


def build_lut(scene, st=None):
    """Density transfer function, (LUT_SIZE, 4) RGBA; alpha carries the filter."""
    from .attributes import srgb_to_linear

    st = _settings(scene, st)
    cmap = st.colormap
    if not colormap_exists(cmap):
        cmap = "viridis"
    rgba = np.asarray(
        sample_colormap(cmap, LUT_SIZE,
                        bool(st.reverse_colormap)),
        dtype=np.float32,
    ).reshape(LUT_SIZE, 4)
    rgba[:, :3] = srgb_to_linear(rgba[:, :3])

    t = np.linspace(0.0, 1.0, LUT_SIZE, dtype=np.float32)
    floor = float(st.volume_threshold)
    gamma = float(st.volume_falloff)
    opacity = np.clip((t - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0) ** gamma
    rgba[:, 3] = opacity
    return rgba


def _field_texture(data):
    field = np.ascontiguousarray(data["field"], dtype=np.float32)
    nz, ny, nx = data["shape"]
    return gpu.types.GPUTexture(
        (nx, ny, nz), format='RGBA16F',
        data=gpu.types.Buffer('FLOAT', field.size, field.ravel())), field.nbytes


def _adopt(vol, data):
    vol.update({
        "lo": data["lo"],
        "extent": data["extent"],
        "shape": data["shape"],
        "dmax": data["dmax"],
        "dtrue": data["dtrue"],
        "voxel_volume": data["voxel_volume"],
    })


def upload(data, lut):
    if data is None:
        return None
    try:
        tex, nbytes = _field_texture(data)
        lut_arr = np.ascontiguousarray(lut, dtype=np.float32)
        lut_tex = gpu.types.GPUTexture(
            (LUT_SIZE, 1), format='RGBA32F',
            data=gpu.types.Buffer('FLOAT', lut_arr.size, lut_arr.ravel()))
    except Exception:  # noqa: BLE001 - backend without 3D textures
        return None

    vol = {"tex": tex, "lut": lut_tex, "bytes": int(nbytes),
           "state": None, "pending": None}
    _adopt(vol, data)
    return vol


def build(scene, coords, colors, weights=None, st=None, animated=False,
          idx=None):
    """``(gpu_dict, cpu_data)``, either None when that step failed. ``animated``
    arms :func:`refresh_field`; ``idx`` is the row of the caller's full
    coordinate array each splatted node came from, so a frame handler can hand
    over every node and let the cloud pick its own back out."""
    st = _settings(scene, st)
    use_color = st.volume_mode == 'COLOR'
    state = {} if animated else None
    data = build_field(
        coords, colors if use_color else None,
        res=int(st.volume_res),
        weights=weights,
        smooth=float(st.volume_smooth),
        state=state,
    )
    if data is None:
        return None, None
    vol = upload(data, build_lut(scene))
    if vol is not None and state is not None:
        idx = None if idx is None else np.asarray(idx)
        state["idx"] = idx
        state["reach"] = 0 if idx is None or not idx.size else int(idx.max()) + 1
        vol["state"] = state
    return vol, data


def _grid_holds(state, coords):
    """Whether the frozen box still deserves to be kept. Refitting it every
    frame is what makes the cloud swim; never refitting piles the escapees onto
    the faces of the box, since :func:`_voxel_index` clamps them."""
    lo, hi = _bounds(coords)
    fit_lo, fit_hi = state["fit"]
    if np.any(lo < fit_lo) or np.any(hi > fit_hi):
        return False
    return float((hi - lo).max()) >= state["span"] / (1.0 + 2.0 * BOX_MARGIN)


def refresh_field(vol, coords):
    """Re-splat a moved layout into the frozen grid, parking the result for
    :func:`flush`. Numpy only, so it is safe from a frame-change handler that
    has no GPU context bound; calling it twice just keeps the later coords."""
    state = vol.get("state") if vol else None
    if state is None or coords.shape[0] < state["reach"]:
        return False
    idx = state["idx"]
    sub = coords if idx is None else coords[idx]
    if not sub.shape[0] or sub.shape[0] != state["count"]:
        return False

    drift = state["dtrue"] / max(state["peak"], 1e-30)
    if not _grid_holds(state, sub) or not 1.0 / DMAX_DRIFT < drift < DMAX_DRIFT:
        state["shape"] = None

    data = build_field(sub, state["colors"], res=state["res"],
                       weights=state["weights"], smooth=state["smooth"],
                       state=state)
    if data is None:
        return False
    vol["pending"] = data
    return True


def flush(vol):
    """Upload what :func:`refresh_field` parked. Draw time only, and a no-op
    with nothing parked, so any number of viewports may call it."""
    data = vol.get("pending") if vol else None
    if data is None:
        return False
    vol["pending"] = None
    try:
        tex, nbytes = _field_texture(data)
    except Exception:  # noqa: BLE001 - out of VRAM mid-animation
        return False
    vol["tex"] = tex
    vol["bytes"] = int(nbytes)
    _adopt(vol, data)
    return True


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
