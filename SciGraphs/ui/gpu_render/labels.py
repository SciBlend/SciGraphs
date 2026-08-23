# Node labels drawn with blf straight into the render output. The compositor PNG
# overlay in core/visualization/text_overlay.py ray-casts real geometry and so
# cannot see GPU impostors; this path reads the depth buffer instead.

import blf
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from ...core.visualization import text_overlay as t_ov
from .state import CACHE
def _settings(scene, st=None):
    if st is not None:
        return st
    from .host import settings_from_scene
    return settings_from_scene(scene)



def labels_enabled(scene, st=None):
    st = _settings(scene, st)
    return (
        bool(st.render_labels)
        and getattr(scene, "scigraphs", None) is not None
    )


def _settings_from_props(props):
    return t_ov.TextOverlaySettings(
        size_mode=props.text_size_mode,
        fixed_size=props.text_size_fixed,
        size_scale=props.text_size_scale,
        max_distance=props.text_max_distance,
        text_color=tuple(props.text_color),
        background_enabled=props.text_background_enabled,
        background_color=tuple(props.text_background_color),
        background_alpha=props.text_background_alpha,
        depth_occlusion=props.text_depth_occlusion,
        filter_enabled=props.text_filter_enabled,
        filter_attribute=props.text_filter_attribute,
        filter_operator=props.text_filter_operator,
        filter_value=props.text_filter_value,
        format_type=props.text_format_type,
        float_decimals=props.text_float_decimals,
        format_prefix=props.text_format_prefix,
        format_suffix=props.text_format_suffix,
        thousands_separator=props.text_thousands_separator,
        font_path=t_ov.get_font_path(props),
    )


# A node's pixel depth is its own sphere's front, nearer than its center.
_OCC_REL = 0.10
_OCC_ABS = 0.2


def collect_label_items(obj, scene, persp, width, height, dist_buf=None):
    """``persp`` is the (4, 4) projection @ view matrix, row major; ``dist_buf``
    is an optional (height, width) eye-distance array for occlusion."""
    props = getattr(scene, "scigraphs", None)
    if props is None:
        return []
    # No per-node labels while coarsened: those nodes are not drawn one by one.
    entry = CACHE.get(obj.as_pointer())
    if entry is not None and entry.get("_coarse_active"):
        return []
    settings = _settings_from_props(props)

    names = t_ov.resolve_node_names(obj)
    if not names:
        return []
    mesh = obj.data
    n = min(len(names), len(mesh.vertices))
    if n == 0:
        return []

    co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)[:n]
    mw = np.asarray(obj.matrix_world, dtype=np.float64)
    world = co @ mw[:3, :3].T + mw[:3, 3]

    p = np.asarray(persp, dtype=np.float64)
    clip = world @ p[:3, :3].T + p[:3, 3]
    wc = world @ p[3, :3].T + p[3, 3]

    visible = wc > 1e-6
    ndc = np.zeros((n, 3))
    ndc[visible] = clip[visible] / wc[visible, None]
    visible &= (np.abs(ndc[:, 0]) <= 1.05) & (np.abs(ndc[:, 1]) <= 1.05)

    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (ndc[:, 1] * 0.5 + 0.5) * height
    dist = np.maximum(wc, 1e-6)  # eye depth along the view axis

    occluded = np.zeros(n, dtype=bool)
    if settings.depth_occlusion and dist_buf is not None:
        # The depth buffer may not match the label resolution (capped DoF).
        bh, bw = dist_buf.shape
        ix = np.clip((px * bw / width).astype(np.int64), 0, bw - 1)
        iy = np.clip((py * bh / height).astype(np.int64), 0, bh - 1)
        near = dist_buf[iy, ix]
        occluded = near < (dist * (1.0 - _OCC_REL) - _OCC_ABS)

    nodes = []
    for i in range(n):
        if not visible[i] or occluded[i]:
            continue
        nodes.append(t_ov.ProjectedNode(
            name=names[i], x=float(px[i]), y=float(py[i]),
            distance=float(dist[i]), visible=True, occluded=False,
            attribute_value=None,
        ))

    nodes = t_ov.apply_distance_filter(nodes, settings.max_distance)
    if settings.filter_enabled and settings.filter_attribute not in ("", 'NONE'):
        nodes = t_ov.apply_attribute_filter(nodes, obj, settings)

    texts = {}
    if (props.text_source == 'ATTRIBUTE'
            and props.text_attribute not in ("", 'NONE')):
        vals = t_ov.get_node_attribute_values(obj, props.text_attribute)
        for node in nodes:
            if node.name in vals:
                texts[node.name] = t_ov.format_value(vals[node.name], settings)

    items = []
    for node in nodes:
        text = texts.get(node.name, node.name)
        size = t_ov.calculate_text_size(node, settings, None)
        items.append((text, node.x, node.y, size, node.distance))
    return items


_FONT_CACHE = {}


def _font_alive(fid):
    blf.size(fid, 12)
    return blf.dimensions(fid, "M")[0] > 0.0


def _font_id(path):
    """A loaded font does not survive opening a .blend, and the stale id measures
    every string zero wide instead of raising. Id 0 is Blender's own, always valid."""
    if not path:
        return 0
    fid = _FONT_CACHE.get(path)
    if fid is not None and (fid == 0 or _font_alive(fid)):
        return fid
    try:
        fid = blf.load(path)
    except Exception:  # noqa: BLE001 - bad font file: fall back to default
        fid = 0
    if fid < 0:
        fid = 0
    _FONT_CACHE[path] = fid
    return fid


_GRID_CELL = 64.0
_PAD = 2.0


def _cells(rect):
    x0, y0, x1, y1 = rect
    for cx in range(int(x0 // _GRID_CELL), int(x1 // _GRID_CELL) + 1):
        for cy in range(int(y0 // _GRID_CELL), int(y1 // _GRID_CELL) + 1):
            yield (cx, cy)


def _rect_free(grid, rect):
    x0, y0, x1, y1 = rect
    for cell in _cells(rect):
        for ox0, oy0, ox1, oy1 in grid.get(cell, ()):
            if x0 < ox1 and x1 > ox0 and y0 < oy1 and y1 > oy0:
                return False
    return True


def _rect_add(grid, rect):
    for cell in _cells(rect):
        grid.setdefault(cell, []).append(rect)


def _candidates(x, y, size, tw, th):
    off = size * 0.4
    gap = size * 0.3
    base = (
        (x - tw * 0.5, y + off),
        (x - tw * 0.5, y - off - th),
        (x + off + gap, y - th * 0.5),
        (x - off - gap - tw, y - th * 0.5),
        (x + off * 0.7, y + off * 0.7),
        (x - off * 0.7 - tw, y + off * 0.7),
        (x + off * 0.7, y - off * 0.7 - th),
        (x - off * 0.7 - tw, y - off * 0.7 - th),
    )
    ring2 = tuple((x + (bx - x) * 2.2, y + (by - y) * 2.2) for bx, by in base)
    return base + ring2


def _declutter(measured):
    """Resolve overlaps, nearest label first, first free anchor wins. Returns
    placed ``(text, bx, by, size, tw, th)``; no free anchor means no label."""
    measured.sort(key=lambda m: (m[4], m[0]))
    grid = {}
    placed = []
    for text, x, y, size, _dist, tw, th in measured:
        for bx, by in _candidates(x, y, size, tw, th):
            rect = (bx - _PAD, by - _PAD, bx + tw + _PAD, by + th + _PAD)
            if _rect_free(grid, rect):
                _rect_add(grid, rect)
                placed.append((text, bx, by, size, tw, th))
                break
    return placed


def draw_label_items(items, scene, width, height, st=None):
    """Coordinates are buffer pixels, origin bottom left, matching blf. The
    pixel-space ortho is explicit because an offscreen has no region projection."""
    st = _settings(scene, st)
    if not items:
        return
    props = getattr(scene, "scigraphs", None)
    if props is None:
        return
    from mathutils import Matrix

    settings = _settings_from_props(props)
    fid = _font_id(settings.font_path)

    prev_blend = gpu.state.blend_get()
    gpu.state.blend_set('ALPHA')

    measured = []
    for text, x, y, size, dist in items:
        blf.size(fid, size)
        tw, th = blf.dimensions(fid, text)
        measured.append((text, x, y, size, dist, tw, th))

    if bool(st.labels_declutter):
        placed = _declutter(measured)
    else:
        placed = [(text, x - tw * 0.5, y + size * 0.4, size, tw, th)
                  for text, x, y, size, _dist, tw, th in measured]

    ortho = Matrix((
        (2.0 / width, 0.0, 0.0, -1.0),
        (0.0, 2.0 / height, 0.0, -1.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ))
    with gpu.matrix.push_pop():
        with gpu.matrix.push_pop_projection():
            gpu.matrix.load_matrix(Matrix.Identity(4))
            gpu.matrix.load_projection_matrix(ortho)

            if settings.background_enabled and settings.background_alpha > 0.0:
                pad = 3.0
                coords = []
                indices = []
                for k, (_t, bx, by, _s, tw, th) in enumerate(placed):
                    x0, x1 = bx - pad, bx + tw + pad
                    y0, y1 = by - pad, by + th + pad
                    base = k * 4
                    coords.extend(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))
                    indices.extend(((base, base + 1, base + 2),
                                    (base, base + 2, base + 3)))
                sh = gpu.shader.from_builtin('UNIFORM_COLOR')
                batch = batch_for_shader(sh, 'TRIS', {"pos": coords},
                                         indices=indices)
                sh.bind()
                bc = settings.background_color
                sh.uniform_float("color", (bc[0], bc[1], bc[2],
                                           float(settings.background_alpha)))
                batch.draw(sh)

            tc = settings.text_color
            blf.color(fid, tc[0], tc[1], tc[2], 1.0)
            for text, bx, by, size, _tw, _th in placed:
                blf.size(fid, size)
                blf.position(fid, bx, by, 0.0)
                blf.draw(fid, text)

    gpu.state.blend_set(prev_blend)
