import blf
import bpy
from bpy.app.handlers import persistent
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from ...core.render import lod
from ...core.visualization import text_overlay as t_ov
from .attributes import read_attribute_values, scalar_attr_items
from .picking import pick_nearest_node
from .state import is_graph_object

_PAD = 8.0
_LINE_GAP = 3.0
_TITLE_PT = 12.0
_LINE_PT = 11.0
_CURSOR_GAP = 16.0
_MAX_ATTRS = 6

_BG = (0.06, 0.06, 0.06, 0.88)
_FRAME = (0.55, 0.55, 0.55, 0.55)
_TITLE = (0.95, 0.95, 0.95, 1.0)
_DIM = (0.72, 0.72, 0.72, 1.0)

_HANDLE = None
_STATE = {"x": 0, "y": 0, "idx": None, "obj": None, "area": None}


def node_name(obj, idx):
    """The imported label for a node, or None.

    Through the shared resolver, not a split of the raw property: names live as
    a JSON list under ``node_names`` and as a comma-joined string under
    ``nodes_data`` depending on the importer, and splitting the JSON on commas
    yields ``["Goroka Airport"`` with the bracket attached.
    """
    try:
        names = t_ov.resolve_node_names(obj)
    except Exception:  # noqa: BLE001 - an overlay must not break the draw
        return None
    if not names or not 0 <= idx < len(names):
        return None
    name = str(names[idx]).strip()
    return name or None


def _degree(obj, idx):
    mesh = obj.data
    n = len(mesh.edges)
    if not n:
        return 0
    edges = np.empty(n * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    return int(np.count_nonzero(edges == idx))


def node_info(obj, idx, max_attrs=_MAX_ATTRS):
    """``(title, [(label, text), ...])`` for one node.

    Degree is counted here rather than read from an attribute: it is the one
    fact every graph has, and a mesh that was never analysed carries no
    ``degree`` attribute to read.
    """
    if obj is None or idx is None:
        return None, []
    if not 0 <= idx < len(obj.data.vertices):
        return None, []
    name = node_name(obj, idx)
    title = f"#{idx}" if not name else f"{name}  (#{idx})"

    rows = [("degree", f"{_degree(obj, idx):,}")]
    for entry in scalar_attr_items(obj)[:max_attrs]:
        attr = entry[0]
        if attr == "degree":
            continue
        try:
            vals = read_attribute_values(obj.data, attr)
        except Exception:  # noqa: BLE001 - an overlay must not break the draw
            continue
        if vals is None or vals.size <= idx:
            continue
        v = float(vals[idx])
        rows.append((attr, "n/a" if not np.isfinite(v) else f"{v:,.4g}"))
    return title, rows


_SEL_CACHE = {"key": None, "title": None, "rows": []}
_EPOCH = 0


@persistent
def invalidate_selection(*_args):
    global _EPOCH
    _EPOCH += 1


def selection_mask(obj):
    """Selected vertices as a bool array, or None.

    Edit Mode keeps the live selection in the BMesh and the datablock is stale,
    which is exactly the mode a box or lasso select runs in. Reading it back
    through ``update_from_editmode`` is correct but slower, 2.39 ms against
    1.09 at 21k nodes, because it rebuilds the whole datablock.
    """
    mesh = getattr(obj, "data", None)
    if mesh is None:
        return None
    if obj.mode == 'EDIT':
        try:
            import bmesh
            bm = bmesh.from_edit_mesh(mesh)
            return np.fromiter((v.select for v in bm.verts),
                               dtype=bool, count=len(bm.verts))
        except Exception:  # noqa: BLE001 - an overlay must not break the draw
            return None
    mask = np.zeros(len(mesh.vertices), dtype=bool)
    mesh.vertices.foreach_get("select", mask)
    return mask


def _spread(values):
    """``mean (min - max)``, or a single number when they agree."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-12:
        return f"{lo:,.4g}"
    return f"{values.mean():,.4g}  ({lo:,.4g} - {hi:,.4g})"


def selection_stats(obj, mask, max_attrs=_MAX_ATTRS):
    """``(title, rows)`` describing every selected node together, or None."""
    if mask is None:
        return None, []
    n = int(mask.sum())
    if n < 2:
        return None, []
    if n >= mask.size:
        return None, []

    mesh = obj.data
    rows = [("nodes", f"{n:,}")]

    n_edges = len(mesh.edges)
    if n_edges:
        edges = np.empty(n_edges * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edges)
        edges = edges.reshape(-1, 2)
        both = mask[edges[:, 0]] & mask[edges[:, 1]]
        inside = int(both.sum())
        crossing = int((mask[edges[:, 0]] ^ mask[edges[:, 1]]).sum())
        rows.append(("edges inside", f"{inside:,}"))
        rows.append(("edges crossing", f"{crossing:,}"))
        deg = np.bincount(edges.ravel(), minlength=len(mesh.vertices))
        rows.append(("degree", _spread(deg[mask].astype(np.float64))))

    for entry in scalar_attr_items(obj)[:max_attrs]:
        attr = entry[0]
        if attr == "degree":
            continue
        try:
            vals = read_attribute_values(mesh, attr)
        except Exception:  # noqa: BLE001
            continue
        if vals is None or vals.size != mask.size:
            continue
        sel = vals[mask]
        finite = sel[np.isfinite(sel)]
        if finite.size == 0:
            continue
        rows.append((attr, _spread(finite.astype(np.float64))))
    return "Selection", rows


def selection_key(obj):
    """What must change before the figures are recomputed."""
    mesh = obj.data
    count = mesh.total_vert_sel if obj.mode == 'EDIT' else -1
    return (obj.as_pointer(), obj.mode, count, _EPOCH)


def cached_selection(obj):
    """The selection block, recomputed only when the key moves."""
    if obj is None:
        return None, []
    own = picked_mask(obj)
    if own is not None:
        key = ("picked", obj.as_pointer(), int(own.sum()), _EPOCH)
        if _SEL_CACHE["key"] == key:
            return _SEL_CACHE["title"], _SEL_CACHE["rows"]
        title, rows = selection_stats(obj, own)
        _SEL_CACHE.update(key=key, title=title, rows=rows)
        return title, rows
    if obj.mode == 'EDIT' and obj.data.total_vert_sel < 2:
        return None, []
    key = selection_key(obj)
    if _SEL_CACHE["key"] == key:
        return _SEL_CACHE["title"], _SEL_CACHE["rows"]
    title, rows = selection_stats(obj, selection_mask(obj))
    _SEL_CACHE.update(key=key, title=title, rows=rows)
    return title, rows


_BOX = {"active": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0}
_PICKED = {"obj": None, "mask": None}


def project_nodes(obj, region, rv3d):
    """Region-space x, y per node plus a mask of what is in front of the camera.

    Shares its arithmetic with pick_nearest_node on purpose: a box that
    disagreed with the readout about where a node is would select one thing and
    describe another.
    """
    mesh = obj.data
    n = len(mesh.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    mat = np.array(rv3d.perspective_matrix @ obj.matrix_world, dtype=np.float64)
    homog = np.empty((n, 4), dtype=np.float64)
    homog[:, :3] = co
    homog[:, 3] = 1.0
    clip = homog @ mat.T
    w = clip[:, 3]
    front = w > 1e-6
    sx = np.full(n, np.inf)
    sy = np.full(n, np.inf)
    sx[front] = (clip[front, 0] / w[front] * 0.5 + 0.5) * region.width
    sy[front] = (clip[front, 1] / w[front] * 0.5 + 0.5) * region.height
    return sx, sy, front


def nodes_in_box(obj, region, rv3d, rect):
    """Bool mask of the nodes whose projection falls inside ``rect``."""
    x0, y0, x1, y1 = rect
    lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
    lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
    sx, sy, front = project_nodes(obj, region, rv3d)
    return front & (sx >= lo_x) & (sx <= hi_x) & (sy >= lo_y) & (sy <= hi_y)


def picked_mask(obj):
    """The gesture's selection for ``obj``, or None."""
    if obj is None or _PICKED["obj"] is None:
        return None
    if _PICKED["obj"] != obj.name:
        return None
    mask = _PICKED["mask"]
    if mask is None or mask.size != len(obj.data.vertices):
        return None
    return mask


def clear_picked():
    _PICKED.update(obj=None, mask=None)


def _text(fid, x, y, pt, text, color):
    blf.size(fid, pt)
    blf.color(fid, *color)
    blf.position(fid, x, y, 0.0)
    blf.draw(fid, text)


def _measure(fid, pt, text):
    blf.size(fid, pt)
    return blf.dimensions(fid, text)


def _rect(rect, color):
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    x0, y0, x1, y1 = rect
    batch = batch_for_shader(
        shader, 'TRIS',
        {"pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]},
        indices=[(0, 1, 2), (0, 2, 3)])
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)


def _outline(rect, color, w):
    x0, y0, x1, y1 = rect
    for r in ((x0, y0, x1, y0 + w), (x0, y1 - w, x1, y1),
              (x0, y0, x0 + w, y1), (x1 - w, y0, x1, y1)):
        _rect(r, color)


def box_origin(x, y, box_w, box_h, region_w, region_h, gap):
    """Keep the box on screen: flip to the other side of the cursor rather than
    clamping, so it never sits under the pointer it describes."""
    ox = x + gap
    if ox + box_w > region_w:
        ox = max(0.0, x - gap - box_w)
    oy = y + gap
    if oy + box_h > region_h:
        oy = max(0.0, y - gap - box_h)
    return ox, oy


def draw_tooltip(region_w, region_h, x, y, sections, fid=0):
    """``sections`` is ``[(title, rows), ...]``; empty ones are dropped, so the
    caller can pass the hovered node and the selection without checking which
    of the two exists."""
    sections = [(t, r) for t, r in sections if t]
    if not sections:
        return
    s = lod.px_scale(region_h)
    title_pt, line_pt = _TITLE_PT * s, _LINE_PT * s
    pad, gap = _PAD * s, _LINE_GAP * s

    plan = []
    widths = []
    for si, (title, rows) in enumerate(sections):
        if si:
            plan.append(("gap", None, gap * 2.0))
        tw, th = _measure(fid, title_pt, title)
        widths.append(tw)
        plan.append(("title", title, th))
        for label, value in rows:
            line = f"{label}   {value}"
            w, h = _measure(fid, line_pt, line)
            widths.append(w)
            plan.append(("line", line, h))

    box_w = max(widths) + pad * 2.0
    box_h = sum(h for _k, _t, h in plan) + gap * (len(plan) - 1) + pad * 2.0
    ox, oy = box_origin(x, y, box_w, box_h, region_w, region_h, _CURSOR_GAP * s)

    prev = gpu.state.blend_get()
    gpu.state.blend_set('ALPHA')
    try:
        _rect((ox, oy, ox + box_w, oy + box_h), _BG)
        _outline((ox, oy, ox + box_w, oy + box_h), _FRAME, max(1.0, s))
        cy = oy + box_h - pad
        for i, (kind, text, h) in enumerate(plan):
            if i:
                cy -= gap
            cy -= h
            if kind == 'title':
                _text(fid, ox + pad, cy, title_pt, text, _TITLE)
            elif kind == 'line':
                _text(fid, ox + pad, cy, line_pt, text, _DIM)
    finally:
        gpu.state.blend_set(prev)


def area_under(window, mx, my):
    """The 3D view and its WINDOW region containing a window-space point.

    Resolved from the event rather than from ``context.area``: a modal started
    from a timer has no area on its context, so trusting it meant the overlay
    never matched the area it was asked to draw in, and nothing appeared.

    A missing ``window`` scans them all rather than giving up. The context a
    modal runs under is not guaranteed to carry one, and returning None there
    is indistinguishable from "the cursor is over no 3D view".
    """
    windows = ([window] if window is not None
               else list(getattr(bpy.context.window_manager, "windows", ())))
    for win in windows:
        for area in win.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            if not (area.x <= mx < area.x + area.width
                    and area.y <= my < area.y + area.height):
                continue
            for region in area.regions:
                if region.type != 'WINDOW':
                    continue
                if (region.x <= mx < region.x + region.width
                        and region.y <= my < region.y + region.height):
                    return area, region
    return None, None


def _draw_rubber_band():
    x0, y0 = _BOX["x0"], _BOX["y0"]
    x1, y1 = _BOX["x1"], _BOX["y1"]
    rect = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    prev = gpu.state.blend_get()
    gpu.state.blend_set('ALPHA')
    try:
        _rect(rect, (0.35, 0.6, 1.0, 0.12))
        _outline(rect, (0.5, 0.75, 1.0, 0.9), 1.0)
    finally:
        gpu.state.blend_set(prev)


def _active_graph(context):
    """The graph the overlay is about. getattr, not attribute access: the
    context a modal is handed does not always carry a view layer."""
    obj = getattr(context, "active_object", None)
    if obj is None:
        layer = getattr(context, "view_layer", None)
        obj = getattr(getattr(layer, "objects", None), "active", None)
    return obj if is_graph_object(obj) else None


def _callback():
    ctx = bpy.context
    area = getattr(ctx, "area", None)
    region = getattr(ctx, "region", None)
    if area is None or region is None:
        return
    target = _STATE["area"]
    if target is None or area.as_pointer() != target.as_pointer():
        return
    if getattr(ctx.scene.render, "engine", "") != 'SCIGRAPHS':
        return
    if _BOX["active"]:
        _draw_rubber_band()
    idx, obj = _STATE["idx"], _STATE["obj"]
    if idx is None or obj is None:
        return
    try:
        sections = [node_info(obj, idx)]
        sections.append(cached_selection(obj))
        draw_tooltip(region.width, region.height,
                     _STATE["x"], _STATE["y"], sections)
    except Exception:  # noqa: BLE001 - an overlay must not break the viewport
        pass


class SCIGRAPHS_OT_node_hover(bpy.types.Operator):
    """Track the pointer and name the node under it.

    Modal rather than a draw handler alone, because a draw handler is never
    told where the mouse is. Everything it does is store three numbers; the
    reading and the drawing are module functions so they can be tested without
    a window.
    """

    bl_idname = "scigraphs.node_hover"
    bl_label = "Node Hover Info"
    bl_description = ("Show a floating readout for the node under the cursor "
                      "while the SciGraphs engine is active")

    def _gesture(self, context, event, area, region):
        """Ctrl and drag: a box over the nodes. Returns None when the event is
        not ours, so everything else still reaches Blender untouched."""
        if area is None or region is None:
            return None
        obj = _active_graph(context)
        if obj is None:
            return None
        x = event.mouse_x - region.x
        y = event.mouse_y - region.y

        if event.type == 'ESC' and event.value == 'PRESS':
            if _BOX["active"] or picked_mask(obj) is not None:
                _BOX["active"] = False
                clear_picked()
                invalidate_selection()
                area.tag_redraw()
            return None

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS' and event.ctrl:
            _BOX.update(active=True, x0=x, y0=y, x1=x, y1=y)
            _STATE["area"] = area
            area.tag_redraw()
            return {'RUNNING_MODAL'}

        if not _BOX["active"]:
            return None

        if event.type == 'MOUSEMOVE':
            _BOX.update(x1=x, y1=y)
            area.tag_redraw()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            _BOX["active"] = False
            rv3d = getattr(area.spaces.active, "region_3d", None)
            rect = (_BOX["x0"], _BOX["y0"], x, y)
            try:
                mask = nodes_in_box(obj, region, rv3d, rect)
            except Exception:  # noqa: BLE001
                mask = None
            if mask is not None and mask.any():
                _PICKED.update(obj=obj.name, mask=mask)
            else:
                clear_picked()
            invalidate_selection()
            area.tag_redraw()
            return {'RUNNING_MODAL'}

        return None

    def modal(self, context, event):
        scene = getattr(context, "scene", None)
        if scene is None or getattr(scene.render, "engine", "") != 'SCIGRAPHS':
            _stop()
            return {'CANCELLED'}
        area, region = area_under(getattr(context, "window", None),
                                  event.mouse_x, event.mouse_y)
        gesture = self._gesture(context, event, area, region)
        if gesture is not None:
            return gesture
        if event.type != 'MOUSEMOVE':
            return {'PASS_THROUGH'}

        if area is None or region is None:
            _STATE.update(idx=None, area=None)
            return {'PASS_THROUGH'}
        obj = _active_graph(context)
        if obj is None:
            _STATE.update(idx=None, area=None)
            return {'PASS_THROUGH'}

        rv3d = getattr(area.spaces.active, "region_3d", None)
        _STATE.update(x=event.mouse_x - region.x, y=event.mouse_y - region.y,
                      obj=obj, area=area)
        try:
            _STATE["idx"] = pick_nearest_node(
                context, obj, _STATE["x"], _STATE["y"],
                region=region, rv3d=rv3d)
        except Exception:  # noqa: BLE001
            _STATE["idx"] = None
        area.tag_redraw()
        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        _start(context)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


def _start(context):
    global _HANDLE
    if _HANDLE is None:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _callback, (), 'WINDOW', 'POST_PIXEL')


def _stop():
    global _HANDLE
    if _HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, 'WINDOW')
        _HANDLE = None
    _STATE.update(idx=None, obj=None, area=None)


def running():
    return _HANDLE is not None


_WARNED = False


def _first_view3d():
    """A window, area and region to invoke into.

    ``bpy.ops`` from a timer runs against a context with no area, and an
    INVOKE_DEFAULT there gives a modal that can never tell where the mouse is.
    Overriding onto a real 3D view is what makes the overlay start at all.
    """
    for window in getattr(bpy.context.window_manager, "windows", ()):
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            region = next((r for r in area.regions if r.type == 'WINDOW'), None)
            if region is not None:
                return window, area, region
    return None, None, None


POLL_SECONDS = 1.0


def ensure():
    """Keep the hover matching the render engine. Returns the poll interval, so
    the timer that runs it keeps running."""
    global _WARNED
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return POLL_SECONDS
    want = getattr(scene.render, "engine", "") == 'SCIGRAPHS'
    if not want:
        if running():
            _stop()
        return POLL_SECONDS
    if running():
        return POLL_SECONDS

    window, area, region = _first_view3d()
    if window is None:
        return POLL_SECONDS
    try:
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.scigraphs.node_hover('INVOKE_DEFAULT')
    except Exception as exc:  # noqa: BLE001
        if not _WARNED:
            _WARNED = True
            print(f"SciGraphs: node hover could not start: {exc!r}")
    return POLL_SECONDS


def start_watch():
    if not bpy.app.timers.is_registered(ensure):
        bpy.app.timers.register(ensure, first_interval=0.5, persistent=True)
    handlers = bpy.app.handlers.depsgraph_update_post
    if invalidate_selection not in handlers:
        handlers.append(invalidate_selection)


def stop_watch():
    if bpy.app.timers.is_registered(ensure):
        bpy.app.timers.unregister(ensure)
    handlers = bpy.app.handlers.depsgraph_update_post
    if invalidate_selection in handlers:
        handlers.remove(invalidate_selection)


def unregister_hover():
    stop_watch()
    _stop()
