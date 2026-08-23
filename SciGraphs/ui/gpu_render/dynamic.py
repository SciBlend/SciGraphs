# Node position as draw-time state. Baked into the vertex buffer, a moving node
# forces a full rebuild: 33 ms at 31k nodes, 175 ms at 262k, 1.2 s on the 2M
# road network. Here a node is one RGBA32F texel plus an index per vertex,
# 7 floats down to 4. `GPUTexture` has no in-place write, so updates recreate it.

import numpy as np

import gpu

from . import edge_styles_gpu, shaders
def _settings(scene, st=None):
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)


# Defined once in shaders.py: the GLSL is compiled against this row length.
TEX_ROW = shaders.POS_TEX_ROW

_CACHE = {}
_CACHE_MAX = 4

_IDLE_TEX = None


def eligible(obj, scene, st=None):
    """``(ok, why)``. Every rejection below is a stage that already consumed the
    coordinates on the CPU and cannot be told they moved."""
    st = _settings(scene, st)
    if "is_intersection" in obj.data.attributes:
        return False, "a curve point is not a node and has no position texel"
    if edge_styles_gpu.enabled(scene):
        return False, "styled edges are tessellated from the positions"
    if st.volume_mode != 'OFF':
        return False, "the density field is splatted from the positions"
    if bool(st.coarsen):
        return False, "a supernode sits at the centroid of its members"
    if bool(st.adaptive):
        return False, "the adaptive cut places stand-ins by position"
    if bool(st.use_blocks):
        return False, "blocks bin nodes by position and cull by block"
    if bool(st.render_id_pass):
        return False, "the ID pass bakes its own positions and is read back"
    return True, ""


def _rows_for(count):
    return max(1, int(np.ceil(count / TEX_ROW)))


def pack(coords, scratch=None):
    """(rows*TEX_ROW, 4) staging. Reusing ``scratch`` saves 16 MB per frame."""
    n = coords.shape[0]
    rows = _rows_for(n)
    if scratch is None or scratch.shape[0] != rows * TEX_ROW:
        scratch = np.zeros((rows * TEX_ROW, 4), dtype=np.float32)
        scratch[:, 3] = 1.0
    scratch[:n, :3] = coords
    return scratch


def texture(coords, scratch=None):
    padded = pack(coords, scratch)
    try:
        return gpu.types.GPUTexture(
            (TEX_ROW, padded.shape[0] // TEX_ROW), format='RGBA32F',
            data=gpu.types.Buffer('FLOAT', padded.size, padded.ravel()))
    except Exception:  # noqa: BLE001 - texture limits, out of VRAM
        return None


def build(obj, coords):
    """Keyed on object name and node count. New coords go to :func:`refresh`."""
    key = (obj.name, coords.shape[0])
    entry = _CACHE.get(key)
    if entry is not None:
        return entry

    scratch = pack(coords)
    tex = texture(coords, scratch)
    if tex is None:
        return None
    entry = {"tex": tex, "scratch": scratch, "count": coords.shape[0]}
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = entry
    return entry


def refresh(entry, coords):
    if entry is None or coords.shape[0] != entry["count"]:
        return False
    tex = texture(coords, entry["scratch"])
    if tex is None:
        return False
    entry["tex"] = tex
    return True


def idle():
    """One-texel stand-in for an animated shader bound with no graph."""
    global _IDLE_TEX
    if _IDLE_TEX is None:
        _IDLE_TEX = texture(np.zeros((1, 3), dtype=np.float32))
    return _IDLE_TEX


def bind(shader, entry):
    shader.uniform_sampler("u_pos", entry["tex"] if entry else idle())


def drop_cache(obj=None):
    if obj is None:
        _CACHE.clear()
        return
    for key in [k for k in _CACHE if k[0] == obj.name]:
        del _CACHE[key]


def node_rows(node_idx):
    return np.ascontiguousarray(node_idx, dtype=np.float32)


def quad_rows(node_idx):
    return np.repeat(np.asarray(node_idx, dtype=np.float32), 4)


def edge_rows(edges):
    """(E*2, 2) as (this end, other end), so a ribbon vertex needs no side flag."""
    e = edges.shape[0]
    out = np.empty((e * 2, 2), dtype=np.float32)
    out[0::2, 0] = edges[:, 0]
    out[0::2, 1] = edges[:, 1]
    out[1::2, 0] = edges[:, 1]
    out[1::2, 1] = edges[:, 0]
    return out


# Copied from batches: importing it is circular. A test checks they match.
_QUAD_CORNERS = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
                         dtype=np.float32)
_QUAD_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def build_point_batches(shader, node_idx, colors_sub, norm_sub, size_by_attr,
                        frows=None, size_buckets=12):
    from gpu_extras.batch import batch_for_shader

    out = []
    node_idx = np.asarray(node_idx)
    if size_by_attr and norm_sub is not None and node_idx.size:
        buckets = np.clip((norm_sub * size_buckets).astype(int),
                          0, size_buckets - 1)
        for b in range(size_buckets):
            sel = buckets == b
            if not sel.any():
                continue
            attrs = {"n_idx": node_rows(node_idx[sel]), "color": colors_sub[sel]}
            if frows is not None:
                attrs["frow"] = frows[sel]
            out.append((batch_for_shader(shader, 'POINTS', attrs),
                        (b + 0.5) / size_buckets))
    else:
        attrs = {"n_idx": node_rows(node_idx), "color": colors_sub}
        if frows is not None:
            attrs["frow"] = frows
        out.append((batch_for_shader(shader, 'POINTS', attrs), None))
    return out


def build_sphere_batch(node_idx, colors_sub, radii_sub, frows=None):
    shader = shaders.get_sphere_shader(filtered=frows is not None, animated=True)
    node_idx = np.asarray(node_idx)
    m = node_idx.size
    if shader is None or m == 0:
        return None
    from gpu_extras.batch import batch_for_shader

    faces = (np.arange(m, dtype=np.int32)[:, None, None] * 4
             + _QUAD_TRIS[None]).reshape(-1, 3)
    attrs = {
        "n_idx": quad_rows(node_idx),
        "corner": np.tile(_QUAD_CORNERS, (m, 1)),
        "radius": np.repeat(radii_sub, 4),
        "color": np.repeat(colors_sub, 4, axis=0),
    }
    if frows is not None:
        attrs["frow"] = np.repeat(frows, 4, axis=0)
    return batch_for_shader(shader, 'TRIS', attrs, indices=faces)


# Must match batches.build_ribbon_batch: (A-, B-, B+, A+) loops around the quad,
# and the wrong triangle pairing turns the tube into a bow tie.
_RIBBON_SIDE = np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float32)
_RIBBON_ENDSEL = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
_RIBBON_TRIS = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)


def build_line_batch(edges, filtered_rows=None):
    """Indexed LINES over node indices: one vertex per node, no positions."""
    shader = get_line_shader(filtered_rows is not None)
    if shader is None or edges is None or edges.shape[0] == 0:
        return None
    from gpu_extras.batch import batch_for_shader

    count = int(edges.max()) + 1
    attrs = {"n_idx": node_rows(np.arange(count))}
    if filtered_rows is not None:
        attrs["frow"] = filtered_rows
    return batch_for_shader(shader, 'LINES', attrs,
                            indices=np.ascontiguousarray(edges, dtype=np.int32))


def get_line_shader(filtered=False):
    return shaders.get_line_dynamic_shader(filtered=filtered)


def build_line_width_buckets(edges, widths, num_nodes, buckets=12):
    """Width classes for the animated line tier. Drawn as quads because the
    rasterizer's own line width does not reach these thicknesses."""
    shader = shaders.get_line_dynamic_wide_shader()
    if shader is None or edges is None or edges.shape[0] == 0 or widths is None:
        return None, None

    idx = np.clip((np.asarray(widths) * buckets).astype(int), 0, buckets - 1)
    out = []
    for b in range(buckets):
        sel = idx == b
        if not sel.any():
            continue
        sub = np.ascontiguousarray(edges[sel], dtype=np.float32)
        e = sub.shape[0]
        # (index_a, index_b, end, side) per corner: both ends, both sides.
        corners = np.empty((e, 4, 4), dtype=np.float32)
        corners[:, :, 0] = sub[:, 0:1]
        corners[:, :, 1] = sub[:, 1:2]
        corners[:, :, 2] = (0.0, 0.0, 1.0, 1.0)
        corners[:, :, 3] = (-1.0, 1.0, -1.0, 1.0)
        fmt = gpu.types.GPUVertFormat()
        fmt.attr_add(id="edge", comp_type='F32', len=4, fetch_mode='FLOAT')
        vbo = gpu.types.GPUVertBuf(len=e * 4, format=fmt)
        vbo.attr_fill("edge", corners.reshape(-1, 4))
        base = (np.arange(e, dtype=np.int32) * 4)[:, None]
        quad = np.array([0, 1, 2, 2, 1, 3], dtype=np.int32)
        ibo = gpu.types.GPUIndexBuf(
            type='TRIS',
            seq=np.ascontiguousarray((base + quad).reshape(-1, 3)))
        out.append((gpu.types.GPUBatch(type='TRIS', buf=vbo, elem=ibo),
                    (b + 0.5) / buckets))
    return (out or None), shader


def build_ribbon_batch(edges, radius_a, radius_b, color, filtered_rows=None):
    """Ribbon quads that fetch their own endpoints. Per-endpoint radii still ride
    in the batch, since a taper is not a function of position."""
    shader = shaders.get_ribbon_shader(filtered=filtered_rows is not None,
                                       animated=True)
    if shader is None or edges is None or edges.shape[0] == 0:
        return None
    from gpu_extras.batch import batch_for_shader

    e = edges.shape[0]
    pair = np.repeat(np.asarray(edges, dtype=np.float32), 4, axis=0)
    side = np.tile(_RIBBON_SIDE, e)
    endsel = np.tile(_RIBBON_ENDSEL, e)
    # Radius follows endsel around the loop: (A, B, B, A).
    ra = np.asarray(radius_a, dtype=np.float32)
    rb = np.asarray(radius_b, dtype=np.float32)
    radius = np.stack([ra, rb, rb, ra], axis=1).reshape(-1)
    colors = np.tile(np.asarray(color, dtype=np.float32), (e * 4, 1))
    faces = (np.arange(e, dtype=np.int32)[:, None, None] * 4
             + _RIBBON_TRIS[None]).reshape(-1, 3)

    attrs = {"n_pair": pair, "side": side, "endsel": endsel,
             "radius": radius, "color": colors}
    if filtered_rows is not None:
        attrs["frow"] = np.repeat(filtered_rows, 4, axis=0)
    return batch_for_shader(shader, 'TRIS', attrs, indices=faces)


def _arrow_triangles(count, nv):
    s = shaders.ARROW_SIDES
    tri_side = [[0, 1 + k, 1 + (k + 1) % s] for k in range(s)]
    tri_base = [[s + 1, s + 2 + (k + 1) % s, s + 2 + k] for k in range(s)]
    tris = np.array(tri_side + tri_base, dtype=np.int32)
    return (np.arange(count, dtype=np.int32)[:, None, None] * nv
            + tris[None]).reshape(-1, 3)


def build_arrow_flat_batch(edges, head_radii, node_radii, color,
                           arrow_scale=1.0):
    shader = shaders.get_arrow_flat_shader(animated=True)
    if shader is None or edges is None or edges.shape[0] == 0:
        return None
    from gpu_extras.batch import batch_for_shader
    from . import batches as _b

    e = edges.shape[0]
    r = np.asarray(head_radii, dtype=np.float32) * float(arrow_scale)
    cone = np.empty((e, 3), dtype=np.float32)
    cone[:, 0] = r * _b._FLAT_LEN
    cone[:, 1] = r * _b._FLAT_HALF_WIDTH
    cone[:, 2] = np.asarray(node_radii, dtype=np.float32)

    attrs = {
        "n_pair": np.repeat(np.asarray(edges, np.float32), 3, axis=0),
        "corner": np.tile(np.array([0.0, 1.0, 2.0], np.float32), e),
        "cone": np.repeat(cone, 3, axis=0),
        "color": np.tile(np.asarray(color, np.float32), (e * 3, 1)),
    }
    faces = np.arange(e * 3, dtype=np.int32).reshape(-1, 3)
    return batch_for_shader(shader, 'TRIS', attrs, indices=faces)


def build_arrow_batch(edges, head_radii, node_radii, color, arrow_scale=1.0):
    """Arrow cones built in the vertex shader. ``node_radii`` is the target
    node's radius. Arrow scale folds into the lengths; push constants are full."""
    shader = shaders.get_arrow_dynamic_shader()
    if shader is None or edges is None or edges.shape[0] == 0:
        return None
    from gpu_extras.batch import batch_for_shader

    e = edges.shape[0]
    nv = 2 * shaders.ARROW_SIDES + 2

    head = np.asarray(head_radii, dtype=np.float32)
    cone = np.empty((e, 3), dtype=np.float32)
    cone[:, 0] = head * 6.0 * arrow_scale
    cone[:, 1] = head * 2.4 * arrow_scale
    cone[:, 2] = np.asarray(node_radii, dtype=np.float32)

    pair = np.repeat(np.asarray(edges, dtype=np.float32), nv, axis=0)
    vert = np.tile(np.arange(nv, dtype=np.float32), e)
    cone = np.repeat(cone, nv, axis=0)
    colors = np.tile(np.asarray(color, dtype=np.float32), (e * nv, 1))

    return batch_for_shader(
        shader, 'TRIS',
        {"n_pair": pair, "a_vert": vert, "cone": cone, "color": colors},
        indices=_arrow_triangles(e, nv),
    )
