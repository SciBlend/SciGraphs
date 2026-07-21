# SciGraphs GPU Preview
# Fast viewport preview that draws graph nodes as points and edges as lines
# directly through the GPU module, bypassing Geometry Nodes entirely.
#
# The graph mesh (vertices + edges) is uploaded to a couple of GPU batches and
# redrawn with one draw call per primitive type, so per-frame cost stays low
# even for very large graphs. On top of the raw preview it supports:
#   - color per node (flat / active color attribute / scalar attribute + colormap)
#   - size per node from a scalar attribute (needs the "fancy" shader)
#   - round points and additive edge blending (density look)
#   - live threshold filtering by a scalar attribute
#   - level-of-detail subsampling for huge graphs
#   - click picking of the nearest node
# This is an interactive "explore" mode; Geometry Nodes remain the path for
# final Cycles/EEVEE renders.

import bpy
import gpu
import numpy as np
from bpy.app.handlers import persistent
from gpu_extras.batch import batch_for_shader

try:
    from bpy_extras import view3d_utils  # noqa: F401 - used in picking
except Exception:  # noqa: BLE001
    view3d_utils = None

from ..core.coloring.attributes import find_attribute, read_attribute_values
from ..core.coloring.colormaps import colormap_exists, values_to_rgba

_HANDLE = None
_ENABLED = False

# Cache keyed by object pointer -> batch bundle (see _build_batches).
_CACHE = {}

# Objects whose native viewport display we replaced with 'BOUNDS' while the
# preview is on, so we can restore them: {object_name: original_display_type}.
_HIDDEN = {}
# Guard against re-entrancy when we tweak display_type from the depsgraph handler.
_SYNCING = False

# Compiled custom shaders (lazy). None means "not available / failed".
_ROUND_POINT_SHADER = None
_ROUND_TRIED = False

# Number of point-size buckets used for "size by attribute".
_SIZE_BUCKETS = 12


# ---------------------------------------------------------------------------
# Custom round-point shader. Falls back to builtin square points if it fails.
#
# We deliberately do NOT write gl_PointSize here: it is unreliable across
# Blender's GPU backends (Metal/Vulkan). Point size is driven by
# gpu.state.point_size_set instead (fixed-function point size), which the
# shader inherits. Roundness is done by discarding fragments outside the disk.
# ---------------------------------------------------------------------------

_ROUND_POINT_VERT = """
uniform mat4 u_mvp;
in vec3 pos;
in vec4 color;
out vec4 f_color;
void main()
{
    gl_Position = u_mvp * vec4(pos, 1.0);
    f_color = color;
}
"""

_ROUND_POINT_FRAG = """
in vec4 f_color;
out vec4 fragColor;
void main()
{
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) {
        discard;
    }
    fragColor = f_color;
}
"""


def _get_round_point_shader():
    """Compile (once) the round-point shader; return None if unsupported."""
    global _ROUND_POINT_SHADER, _ROUND_TRIED
    if _ROUND_TRIED:
        return _ROUND_POINT_SHADER
    _ROUND_TRIED = True
    try:
        _ROUND_POINT_SHADER = gpu.types.GPUShader(
            _ROUND_POINT_VERT, _ROUND_POINT_FRAG
        )
    except Exception:  # noqa: BLE001 - unsupported backend / compile error
        _ROUND_POINT_SHADER = None
    return _ROUND_POINT_SHADER


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _preview_prop(scene, name, default):
    return getattr(scene, f"scigraphs_preview_{name}", default)


def _is_graph_object(obj):
    """True when ``obj`` is a graph mesh we can draw."""
    return (
        obj is not None
        and obj.type == 'MESH'
        and "num_nodes" in obj
        and len(obj.data.vertices) > 0
    )


def is_enabled():
    """True when the preview draw handler is active and switched on."""
    scene = getattr(bpy.context, "scene", None)
    return _ENABLED and scene is not None and bool(_preview_prop(scene, "enabled", False))


def _active_point_color_attribute(mesh):
    """Return the active POINT-domain color attribute, or ``None``."""
    color_attrs = getattr(mesh, "color_attributes", None)
    if not color_attrs:
        return None
    active = color_attrs.active_color
    if active is None or active.domain != 'POINT':
        return None
    if len(active.data) != len(mesh.vertices):
        return None
    return active


def compute_visible_mask(obj, scene=None):
    """Return ``(mask, active)`` for the current preview filter.

    ``mask`` is a boolean numpy array of length ``num_verts`` (True = visible).
    ``active`` is True only when a filter is actually narrowing the set, so
    callers can decide whether offering a "filtered only" option makes sense.
    """
    if scene is None:
        scene = bpy.context.scene
    mesh = obj.data
    num_verts = len(mesh.vertices)
    mask = np.ones(num_verts, dtype=bool)

    if not bool(_preview_prop(scene, "filter_enabled", False)):
        return mask, False

    values, vmin, vmax = _read_value_channel(
        mesh, _preview_prop(scene, "attr_name", ""), num_verts
    )
    if values is None:
        return mask, False

    if vmax > vmin:
        norm = (values - vmin) / (vmax - vmin)
    else:
        norm = np.zeros(num_verts, dtype=np.float32)

    fmin = float(_preview_prop(scene, "filter_min", 0.0))
    fmax = float(_preview_prop(scene, "filter_max", 1.0))
    mask = (norm >= fmin) & (norm <= fmax)
    return mask, True


def write_visibility_attribute(obj, mask, name="scigraphs_visible"):
    """Write ``mask`` as a BOOLEAN POINT attribute and return its name.

    A boolean attribute can be fed straight into a Geometry Nodes ``Selection``
    input via an Input Named Attribute node, avoiding any INT->bool comparison.
    """
    mesh = obj.data
    if name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[name])
        except RuntimeError:
            pass
    attr = mesh.attributes.new(name=name, type='BOOLEAN', domain='POINT')
    attr.data.foreach_set("value", np.ascontiguousarray(mask, dtype=bool))
    return name


def _read_value_channel(mesh, attr_name, num_verts):
    """Read a POINT-domain scalar attribute, return (values, vmin, vmax) or (None, 0, 1)."""
    if not attr_name:
        return None, 0.0, 1.0
    attr = find_attribute(mesh, attr_name)
    if attr is None or getattr(attr, "domain", "") != 'POINT':
        return None, 0.0, 1.0
    values = read_attribute_values(mesh, attr_name)
    if values.size != num_verts:
        return None, 0.0, 1.0
    finite = np.isfinite(values)
    if not finite.any():
        return None, 0.0, 1.0
    vmin = float(values[finite].min())
    vmax = float(values[finite].max())
    return values.astype(np.float32), vmin, vmax


# ---------------------------------------------------------------------------
# Batch building
# ---------------------------------------------------------------------------

def _object_signature(obj, scene):
    """Fingerprint of everything that changes the CPU-baked batch content."""
    mesh = obj.data
    color_attr = _active_point_color_attribute(mesh)
    return (
        obj.name,
        len(mesh.vertices),
        len(mesh.edges),
        color_attr.name if color_attr else "",
        _preview_prop(scene, "color_mode", 'VERTEX'),
        _preview_prop(scene, "attr_name", ""),
        _preview_prop(scene, "colormap", "viridis"),
        bool(_preview_prop(scene, "reverse_colormap", False)),
        tuple(_preview_prop(scene, "node_color", (0.3, 0.7, 1.0, 1.0))),
        bool(_preview_prop(scene, "size_by_attr", False)),
        bool(_preview_prop(scene, "round_points", True)),
        bool(_preview_prop(scene, "filter_enabled", False)),
        round(float(_preview_prop(scene, "filter_min", 0.0)), 4),
        round(float(_preview_prop(scene, "filter_max", 1.0)), 4),
        bool(_preview_prop(scene, "lod_enabled", False)),
        int(_preview_prop(scene, "lod_max_points", 300000)),
        int(obj.get("scigraphs_preview_epoch", 0)),
    )


def _compute_colors(mesh, num_verts, scene, values, vmin, vmax):
    """Return an (N, 4) float32 color array according to the color mode."""
    mode = _preview_prop(scene, "color_mode", 'VERTEX')

    if mode == 'ATTRIBUTE' and values is not None:
        cmap = _preview_prop(scene, "colormap", "viridis")
        if colormap_exists(cmap):
            rgba = values_to_rgba(
                values, cmap_name=cmap, vmin=vmin, vmax=vmax,
                reverse=bool(_preview_prop(scene, "reverse_colormap", False)),
            )
            return np.asarray(rgba, dtype=np.float32).reshape(-1, 4)

    if mode in ('VERTEX', 'ATTRIBUTE'):
        color_attr = _active_point_color_attribute(mesh)
        if color_attr is not None:
            colors = np.empty(num_verts * 4, dtype=np.float32)
            color_attr.data.foreach_get("color", colors)
            return colors.reshape(num_verts, 4)

    flat = np.array(_preview_prop(scene, "node_color", (0.3, 0.7, 1.0, 1.0)),
                    dtype=np.float32)
    return np.tile(flat, (num_verts, 1))


def _build_batches(obj, scene):
    """Build point + line GPU batches from the object's mesh data."""
    mesh = obj.data
    num_verts = len(mesh.vertices)

    coords = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(num_verts, 3)

    num_edges = len(mesh.edges)
    if num_edges:
        edges = np.empty(num_edges * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", edges)
        edges = edges.reshape(num_edges, 2)
    else:
        edges = None

    values, vmin, vmax = _read_value_channel(
        mesh, _preview_prop(scene, "attr_name", ""), num_verts
    )
    # Normalized value in [0, 1] for size + filter.
    if values is not None and vmax > vmin:
        norm = (values - vmin) / (vmax - vmin)
    elif values is not None:
        norm = np.zeros(num_verts, dtype=np.float32)
    else:
        norm = None

    colors = _compute_colors(mesh, num_verts, scene, values, vmin, vmax)

    # Visibility mask from the threshold filter.
    visible = np.ones(num_verts, dtype=bool)
    if bool(_preview_prop(scene, "filter_enabled", False)) and norm is not None:
        fmin = float(_preview_prop(scene, "filter_min", 0.0))
        fmax = float(_preview_prop(scene, "filter_max", 1.0))
        visible = (norm >= fmin) & (norm <= fmax)

    point_idx = np.nonzero(visible)[0]

    # Level-of-detail subsampling of the visible points.
    lod = bool(_preview_prop(scene, "lod_enabled", False))
    lod_max = max(1000, int(_preview_prop(scene, "lod_max_points", 300000)))
    if lod and point_idx.size > lod_max:
        stride = int(np.ceil(point_idx.size / lod_max))
        point_idx = point_idx[::stride]

    point_coords = coords[point_idx]
    point_colors = colors[point_idx]

    # Point shader: custom round shader when requested & supported, else builtin.
    round_pts = bool(_preview_prop(scene, "round_points", True))
    round_shader = _get_round_point_shader() if round_pts else None
    if round_shader is not None:
        point_shader = round_shader
        point_kind = 'ROUND'
    else:
        point_shader = gpu.shader.from_builtin('FLAT_COLOR')
        point_kind = 'FLAT'

    # Size by attribute: bucket points by normalized value and draw each bucket
    # with its own point size (fixed-function), which works on every backend.
    # Each bucket stores its representative normalized value so the actual pixel
    # size can be computed live at draw time from node_size / size_max_mult.
    size_by_attr = bool(_preview_prop(scene, "size_by_attr", False))
    point_batches = []
    if size_by_attr and norm is not None and point_idx.size:
        nvals = norm[point_idx]
        buckets = np.clip((nvals * _SIZE_BUCKETS).astype(int), 0, _SIZE_BUCKETS - 1)
        for b in range(_SIZE_BUCKETS):
            sel = buckets == b
            if not sel.any():
                continue
            batch = batch_for_shader(
                point_shader, 'POINTS',
                {"pos": point_coords[sel], "color": point_colors[sel]},
            )
            point_batches.append((batch, (b + 0.5) / _SIZE_BUCKETS))
    else:
        batch = batch_for_shader(
            point_shader, 'POINTS', {"pos": point_coords, "color": point_colors}
        )
        point_batches.append((batch, None))

    # Edges: keep only those whose endpoints are both visible.
    line_shader = None
    line_batch = None
    if edges is not None:
        keep = visible[edges[:, 0]] & visible[edges[:, 1]]
        kept = edges[keep]
        if lod and kept.shape[0] > lod_max:
            stride = int(np.ceil(kept.shape[0] / lod_max))
            kept = kept[::stride]
        if kept.shape[0] > 0:
            line_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            line_batch = batch_for_shader(
                line_shader, 'LINES', {"pos": coords}, indices=kept
            )

    return {
        "point_batches": point_batches,
        "point_shader": point_shader,
        "point_kind": point_kind,
        "line_batch": line_batch,
        "line_shader": line_shader,
        "visible_count": int(point_idx.size),
    }


def _get_cache_entry(obj, scene):
    """Return cached batches for ``obj``, rebuilding them if stale."""
    key = obj.as_pointer()
    sig = _object_signature(obj, scene)
    entry = _CACHE.get(key)
    if entry is None or entry.get("sig") != sig:
        try:
            data = _build_batches(obj, scene)
        except Exception:  # noqa: BLE001 - never let drawing crash the viewport
            return None
        data["sig"] = sig
        _CACHE[key] = data
        entry = data
    return entry


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_callback():
    """Draw handler: points for nodes, lines for edges, in world space."""
    context = bpy.context
    if context.area is None or context.area.type != 'VIEW_3D':
        return

    scene = context.scene
    if scene is None or not _preview_prop(scene, "enabled", False):
        return

    obj = context.active_object
    if not _is_graph_object(obj):
        return

    entry = _get_cache_entry(obj, scene)
    if entry is None:
        return

    node_size = float(_preview_prop(scene, "node_size", 6.0))
    size_max_mult = float(_preview_prop(scene, "size_max_mult", 4.0))
    show_edges = bool(_preview_prop(scene, "show_edges", True))
    edge_color = tuple(_preview_prop(scene, "edge_color", (0.5, 0.5, 0.55, 0.35)))
    edge_width = float(_preview_prop(scene, "edge_width", 1.0))
    depth_test = bool(_preview_prop(scene, "depth_test", True))
    additive = bool(_preview_prop(scene, "additive_edges", False))

    prev_blend = gpu.state.blend_get()
    if depth_test:
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)

    with gpu.matrix.push_pop():
        gpu.matrix.multiply_matrix(obj.matrix_world)

        # Edges first so nodes sit visually on top.
        if show_edges and entry["line_batch"] is not None:
            gpu.state.blend_set('ADDITIVE' if additive else 'ALPHA')
            gpu.state.line_width_set(edge_width)
            shader = entry["line_shader"]
            shader.bind()
            shader.uniform_float("color", edge_color)
            entry["line_batch"].draw(shader)
            gpu.state.line_width_set(1.0)

        gpu.state.blend_set('ALPHA')
        shader = entry["point_shader"]
        shader.bind()
        if entry["point_kind"] == 'ROUND':
            mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
            shader.uniform_float("u_mvp", mvp)
        for batch, mid in entry["point_batches"]:
            if mid is None:
                size = node_size
            else:
                size = node_size * (1.0 + mid * (size_max_mult - 1.0))
            gpu.state.point_size_set(size)
            batch.draw(shader)
        gpu.state.point_size_set(1.0)

    if depth_test:
        gpu.state.depth_test_set('NONE')
        gpu.state.depth_mask_set(False)
    gpu.state.blend_set(prev_blend)


def _tag_redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


# ---------------------------------------------------------------------------
# Native mesh hiding
# ---------------------------------------------------------------------------

def _sync_native_display(context):
    """Hide the native mesh draw of the active graph while the preview is on."""
    global _SYNCING
    if _SYNCING or context is None:
        return

    scene = getattr(context, "scene", None)
    hide = bool(_preview_prop(scene, "hide_mesh", True)) if scene else False
    active_on = _ENABLED and bool(_preview_prop(scene, "enabled", False))

    desired = None
    if active_on and hide:
        obj = getattr(context, "active_object", None)
        if _is_graph_object(obj):
            desired = obj
    desired_name = desired.name if desired is not None else None

    _SYNCING = True
    try:
        for name in list(_HIDDEN.keys()):
            if name == desired_name:
                continue
            obj = bpy.data.objects.get(name)
            original = _HIDDEN.pop(name)
            if obj is not None and obj.display_type != original:
                obj.display_type = original

        if desired is not None and desired_name not in _HIDDEN:
            _HIDDEN[desired_name] = desired.display_type
            if desired.display_type != 'BOUNDS':
                desired.display_type = 'BOUNDS'
    finally:
        _SYNCING = False


# ---------------------------------------------------------------------------
# Handlers / lifecycle
# ---------------------------------------------------------------------------

def _ensure_depsgraph_handler(add):
    handlers = bpy.app.handlers.depsgraph_update_post
    present = _depsgraph_update_handler in handlers
    if add and not present:
        handlers.append(_depsgraph_update_handler)
    elif not add and present:
        handlers.remove(_depsgraph_update_handler)


def enable_preview():
    global _HANDLE, _ENABLED
    if not _ENABLED:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_callback, (), 'WINDOW', 'POST_VIEW'
        )
        _ENABLED = True
    _ensure_depsgraph_handler(True)
    _sync_native_display(bpy.context)
    _tag_redraw()


def disable_preview():
    global _HANDLE, _ENABLED
    if _ENABLED and _HANDLE:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, 'WINDOW')
        _HANDLE = None
        _ENABLED = False
    _ensure_depsgraph_handler(False)
    _sync_native_display(bpy.context)
    _CACHE.clear()
    _tag_redraw()


def invalidate(obj=None):
    """Force a rebuild of cached batches (call after a layout/color change)."""
    if obj is None:
        _CACHE.clear()
    else:
        _CACHE.pop(obj.as_pointer(), None)
    _tag_redraw()


@persistent
def _on_load_post(*_args):
    """Re-apply the preview state after opening a .blend (module state is reset)."""
    scene = getattr(bpy.context, "scene", None)
    if scene is None:
        return
    if bool(_preview_prop(scene, "enabled", True)):
        enable_preview()
    else:
        disable_preview()


@persistent
def _depsgraph_update_handler(scene, depsgraph):
    """Auto-invalidate cached batches when a graph mesh changes."""
    if not _ENABLED:
        return

    _sync_native_display(bpy.context)

    if not _CACHE:
        return

    dirty = False
    for update in depsgraph.updates:
        id_data = update.id
        if not isinstance(id_data, bpy.types.Object) or id_data.type != 'MESH':
            continue
        if update.is_updated_geometry or update.is_updated_transform:
            _CACHE.pop(id_data.original.as_pointer(), None)
            dirty = True

    if dirty:
        _tag_redraw()


# ---------------------------------------------------------------------------
# Property update callbacks
# ---------------------------------------------------------------------------

def _on_enabled_update(self, context):
    if getattr(self, "scigraphs_preview_enabled", False):
        enable_preview()
    else:
        disable_preview()


def _on_setting_update(self, context):
    _tag_redraw()


def _on_rebuild_update(self, context):
    # Settings that change the baked batch content.
    invalidate()


def _on_hide_mesh_update(self, context):
    _sync_native_display(context)
    _tag_redraw()


def apply_display_engine(context):
    """Apply the scene display engine to the active graph object.

    GPU  -> turn on the GPU preview, hide the Geometry Nodes modifier so it
            neither draws nor evaluates.
    CPU  -> turn off the GPU preview, ensure the Geometry Nodes visualization
            exists and is shown.
    """
    scene = getattr(context, "scene", None)
    if scene is None:
        return
    engine = getattr(scene, "scigraphs_display_engine", 'GPU')
    obj = getattr(context, "active_object", None)

    if engine == 'GPU':
        if _is_graph_object(obj):
            mod = obj.modifiers.get("SciGraphs_Viz")
            if mod is not None and mod.show_viewport:
                mod.show_viewport = False
        if not scene.scigraphs_preview_enabled:
            scene.scigraphs_preview_enabled = True
        else:
            enable_preview()
    else:  # GEOMETRY_NODES / CPU
        if scene.scigraphs_preview_enabled:
            scene.scigraphs_preview_enabled = False
        else:
            disable_preview()
        if _is_graph_object(obj):
            mod = obj.modifiers.get("SciGraphs_Viz")
            if mod is None:
                try:
                    from ..core.mesh.geometry import setup_geometry_nodes_visualization
                    setup_geometry_nodes_visualization(obj)
                    mod = obj.modifiers.get("SciGraphs_Viz")
                except Exception:  # noqa: BLE001
                    mod = None
            if mod is not None and not mod.show_viewport:
                mod.show_viewport = True
    _tag_redraw()


def _on_engine_update(self, context):
    apply_display_engine(context)


# ---------------------------------------------------------------------------
# Picking
# ---------------------------------------------------------------------------

def _pick_nearest_node(context, obj, mouse_x, mouse_y, radius_px=20.0):
    """Return the index of the node closest to the mouse, or None."""
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return None

    mesh = obj.data
    num_verts = len(mesh.vertices)
    coords = np.empty(num_verts * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", coords)
    coords = coords.reshape(num_verts, 3)

    mat = np.array(rv3d.perspective_matrix @ obj.matrix_world, dtype=np.float64)
    homog = np.empty((num_verts, 4), dtype=np.float64)
    homog[:, :3] = coords
    homog[:, 3] = 1.0
    clip = homog @ mat.T

    w = clip[:, 3]
    in_front = w > 1e-6
    if not in_front.any():
        return None

    ndc_x = np.full(num_verts, np.inf)
    ndc_y = np.full(num_verts, np.inf)
    ndc_x[in_front] = clip[in_front, 0] / w[in_front]
    ndc_y[in_front] = clip[in_front, 1] / w[in_front]

    sx = (ndc_x * 0.5 + 0.5) * region.width
    sy = (ndc_y * 0.5 + 0.5) * region.height

    dx = sx - mouse_x
    dy = sy - mouse_y
    dist2 = dx * dx + dy * dy
    dist2[~in_front] = np.inf

    nearest = int(np.argmin(dist2))
    if dist2[nearest] > radius_px * radius_px:
        return None
    return nearest


class SCIGRAPHS_OT_pick_node(bpy.types.Operator):
    """Click a node in the GPU preview to inspect it"""
    bl_idname = "scigraphs.pick_node"
    bl_label = "Pick Node"
    bl_description = "Click the nearest node to read its index and attributes (Esc/right-click to stop)"

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.workspace.status_text_set(None)
            return {'CANCELLED'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if context.area is None or context.area.type != 'VIEW_3D':
                return {'PASS_THROUGH'}
            obj = context.active_object
            if not _is_graph_object(obj):
                return {'RUNNING_MODAL'}
            idx = _pick_nearest_node(
                context, obj,
                event.mouse_region_x, event.mouse_region_y,
            )
            if idx is None:
                self.report({'INFO'}, "No node under cursor")
            else:
                obj["scigraphs_preview_picked"] = idx
                attr_bits = []
                for name, *_ in _scalar_attr_items(obj)[:4]:
                    vals = read_attribute_values(obj.data, name)
                    if vals.size > idx:
                        attr_bits.append(f"{name}={vals[idx]:.3g}")
                extra = ("  |  " + ", ".join(attr_bits)) if attr_bits else ""
                self.report({'INFO'}, f"Node #{idx}{extra}")
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if not is_enabled():
            self.report({'WARNING'}, "Enable the GPU preview first")
            return {'CANCELLED'}
        context.workspace.status_text_set("Pick Node: click nodes  |  Esc/Right-click to stop")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def _scalar_attr_items(obj):
    from ..core.coloring.attributes import list_scalar_attributes
    if not _is_graph_object(obj):
        return []
    return list_scalar_attributes(obj.data)


class SCIGRAPHS_OT_set_preview_attr(bpy.types.Operator):
    """Pick the scalar attribute used by the GPU preview"""
    bl_idname = "scigraphs.set_preview_attr"
    bl_label = "Set Preview Attribute"
    bl_property = "attr"

    def _items(self, context):
        obj = getattr(context, "active_object", None)
        items = [("", "None", "No attribute")]
        for name, data_type, domain in _scalar_attr_items(obj):
            if domain == 'POINT':
                items.append((name, name, f"{data_type} on {domain}"))
        return items

    attr: bpy.props.EnumProperty(name="Attribute", items=_items)

    def execute(self, context):
        context.scene.scigraphs_preview_attr_name = self.attr
        invalidate()
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'FINISHED'}


class SCIGRAPHS_OT_toggle_gpu_preview(bpy.types.Operator):
    """Toggle the fast GPU point/line preview of the active graph"""
    bl_idname = "scigraphs.toggle_gpu_preview"
    bl_label = "Toggle GPU Preview"
    bl_description = (
        "Draw the active graph as GPU points and lines for fast viewport "
        "navigation (bypasses Geometry Nodes)"
    )

    def execute(self, context):
        scene = context.scene
        scene.scigraphs_preview_enabled = not scene.scigraphs_preview_enabled
        state = "enabled" if scene.scigraphs_preview_enabled else "disabled"
        self.report({'INFO'}, f"GPU preview {state}")
        return {'FINISHED'}


class SCIGRAPHS_OT_refresh_gpu_preview(bpy.types.Operator):
    """Rebuild the GPU preview from the current graph positions/colors"""
    bl_idname = "scigraphs.refresh_gpu_preview"
    bl_label = "Refresh GPU Preview"
    bl_description = "Rebuild the GPU preview after a layout or coloring change"

    def execute(self, context):
        invalidate()
        self.report({'INFO'}, "GPU preview refreshed")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_gpu_preview(bpy.types.Panel):
    """Fast GPU preview settings."""
    bl_label = "GPU Preview (Fast)"
    bl_parent_id = "SCIGRAPHS_PT_visualization"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene, "scigraphs_preview_enabled", text="")

    @staticmethod
    def _draw_attr_row(col, scene):
        row = col.row(align=True)
        row.prop(scene, "scigraphs_preview_attr_name", text="Attribute")
        row.operator("scigraphs.set_preview_attr", text="", icon='DOWNARROW_HLT')

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        obj = context.active_object

        layout.use_property_split = True
        layout.use_property_decorate = False

        if not _is_graph_object(obj):
            layout.label(text="Select a graph object", icon='INFO')
            return

        enabled = scene.scigraphs_preview_enabled
        entry = _CACHE.get(obj.as_pointer()) if enabled else None
        shown = entry["visible_count"] if entry else len(obj.data.vertices)
        info = layout.row()
        info.enabled = enabled
        info.label(
            text=f"{shown:,}/{len(obj.data.vertices):,} nodes  |  {len(obj.data.edges):,} edges",
            icon='SNAP_VERTEX',
        )

        # Nodes
        col = layout.column(align=True)
        col.enabled = enabled
        col.label(text="Nodes", icon='MESH_CIRCLE')
        col.prop(scene, "scigraphs_preview_node_size", text="Point Size")
        col.prop(scene, "scigraphs_preview_round_points", text="Round Points")
        if scene.scigraphs_preview_round_points and _ROUND_TRIED and _ROUND_POINT_SHADER is None:
            col.label(text="Round points unsupported on this backend", icon='ERROR')
        col.prop(scene, "scigraphs_preview_color_mode", text="Color")
        if scene.scigraphs_preview_color_mode == 'FLAT':
            col.prop(scene, "scigraphs_preview_node_color", text="")
        elif scene.scigraphs_preview_color_mode == 'ATTRIBUTE':
            self._draw_attr_row(col, scene)
            col.prop(scene, "scigraphs_preview_colormap", text="Colormap")
            col.prop(scene, "scigraphs_preview_reverse_colormap", text="Reverse")

        # Size by attribute
        col = layout.column(align=True)
        col.enabled = enabled
        col.prop(scene, "scigraphs_preview_size_by_attr", text="Size by Attribute")
        if scene.scigraphs_preview_size_by_attr:
            if scene.scigraphs_preview_color_mode != 'ATTRIBUTE':
                self._draw_attr_row(col, scene)
            col.prop(scene, "scigraphs_preview_size_max_mult", text="Max Multiplier")

        # Edges
        col = layout.column(align=True)
        col.enabled = enabled
        col.label(text="Edges", icon='CURVE_PATH')
        col.prop(scene, "scigraphs_preview_show_edges", text="Show Edges")
        sub = col.column(align=True)
        sub.enabled = scene.scigraphs_preview_show_edges
        sub.prop(scene, "scigraphs_preview_edge_width", text="Width")
        sub.prop(scene, "scigraphs_preview_edge_color", text="Color")
        sub.prop(scene, "scigraphs_preview_additive_edges", text="Additive (Density)")

        # Filtering
        col = layout.column(align=True)
        col.enabled = enabled
        col.prop(scene, "scigraphs_preview_filter_enabled", text="Filter by Attribute")
        if scene.scigraphs_preview_filter_enabled:
            if scene.scigraphs_preview_color_mode != 'ATTRIBUTE':
                self._draw_attr_row(col, scene)
            row = col.row(align=True)
            row.prop(scene, "scigraphs_preview_filter_min", text="Min", slider=True)
            row.prop(scene, "scigraphs_preview_filter_max", text="Max", slider=True)

        # Level of detail
        col = layout.column(align=True)
        col.enabled = enabled
        col.prop(scene, "scigraphs_preview_lod_enabled", text="Level of Detail")
        if scene.scigraphs_preview_lod_enabled:
            col.prop(scene, "scigraphs_preview_lod_max_points", text="Max Points")

        # Misc
        col = layout.column(align=True)
        col.enabled = enabled
        col.prop(scene, "scigraphs_preview_depth_test", text="Depth Test")
        col.prop(scene, "scigraphs_preview_hide_mesh", text="Hide Native Mesh")
        if enabled and not scene.scigraphs_preview_hide_mesh:
            col.row().label(text="Mesh drawn twice (slower)", icon='ERROR')

        layout.separator()
        row = layout.row(align=True)
        row.enabled = enabled
        row.operator("scigraphs.pick_node", icon='RESTRICT_SELECT_OFF')
        row.operator("scigraphs.refresh_gpu_preview", text="", icon='FILE_REFRESH')

        picked = obj.get("scigraphs_preview_picked", None)
        if picked is not None:
            layout.label(text=f"Picked node: #{picked}", icon='PINNED')


_CLASSES = [
    SCIGRAPHS_OT_toggle_gpu_preview,
    SCIGRAPHS_OT_refresh_gpu_preview,
    SCIGRAPHS_OT_pick_node,
    SCIGRAPHS_OT_set_preview_attr,
    SCIGRAPHS_PT_gpu_preview,
]


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

def _register_properties():
    S = bpy.types.Scene
    S.scigraphs_display_engine = bpy.props.EnumProperty(
        name="Display Engine",
        description="How graphs are shown in the viewport",
        items=[
            ('GPU', "GPU (Fast)",
             "Draw nodes/edges directly on the GPU. Fast, interactive, ideal "
             "for exploring large graphs. Not shown in Cycles/EEVEE renders",
             'SHADING_RENDERED', 0),
            ('GEOMETRY_NODES', "CPU (Geometry Nodes)",
             "Instanced spheres/tubes via Geometry Nodes. Slower but renderable "
             "in Cycles/EEVEE with full materials",
             'GEOMETRY_NODES', 1),
        ],
        default='GPU', update=_on_engine_update,
    )
    S.scigraphs_preview_enabled = bpy.props.BoolProperty(
        name="GPU Preview",
        description="Draw the active graph as fast GPU points and lines",
        default=True, update=_on_enabled_update,
    )
    S.scigraphs_preview_node_size = bpy.props.FloatProperty(
        name="Point Size", description="Size of node points in pixels",
        default=6.0, min=1.0, max=64.0, update=_on_setting_update,
    )
    S.scigraphs_preview_round_points = bpy.props.BoolProperty(
        name="Round Points",
        description="Draw circular points instead of squares (uses a custom shader when supported)",
        default=True, update=_on_rebuild_update,
    )
    S.scigraphs_preview_color_mode = bpy.props.EnumProperty(
        name="Color Mode",
        description="How node colors are chosen",
        items=[
            ('FLAT', "Flat", "Single flat color"),
            ('VERTEX', "Node Colors", "Active per-vertex color attribute (e.g. from Coloring)"),
            ('ATTRIBUTE', "Attribute + Colormap", "Map a scalar attribute through a colormap"),
        ],
        default='VERTEX', update=_on_rebuild_update,
    )
    S.scigraphs_preview_node_color = bpy.props.FloatVectorProperty(
        name="Node Color", description="Flat node color",
        subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.3, 0.7, 1.0, 1.0), update=_on_rebuild_update,
    )
    S.scigraphs_preview_attr_name = bpy.props.StringProperty(
        name="Attribute",
        description="Scalar POINT attribute used for color / size / filter",
        default="", update=_on_rebuild_update,
    )
    S.scigraphs_preview_colormap = bpy.props.StringProperty(
        name="Colormap", description="Colormap name (e.g. viridis, plasma, magma)",
        default="viridis", update=_on_rebuild_update,
    )
    S.scigraphs_preview_reverse_colormap = bpy.props.BoolProperty(
        name="Reverse Colormap", default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_size_by_attr = bpy.props.BoolProperty(
        name="Size by Attribute",
        description="Scale node point size by the selected scalar attribute",
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_size_max_mult = bpy.props.FloatProperty(
        name="Max Size Multiplier",
        description="Point size multiplier for the highest attribute value",
        default=4.0, min=1.0, max=32.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_show_edges = bpy.props.BoolProperty(
        name="Show Edges", description="Draw graph edges as GPU lines",
        default=True, update=_on_setting_update,
    )
    S.scigraphs_preview_edge_color = bpy.props.FloatVectorProperty(
        name="Edge Color", description="Color (and alpha) used for the edge lines",
        subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.5, 0.5, 0.55, 0.35), update=_on_setting_update,
    )
    S.scigraphs_preview_edge_width = bpy.props.FloatProperty(
        name="Edge Width", description="Line width of edges in pixels",
        default=1.0, min=1.0, max=16.0, update=_on_setting_update,
    )
    S.scigraphs_preview_additive_edges = bpy.props.BoolProperty(
        name="Additive Edges",
        description="Use additive blending so dense edge regions glow (density map look)",
        default=False, update=_on_setting_update,
    )
    S.scigraphs_preview_filter_enabled = bpy.props.BoolProperty(
        name="Filter", description="Show only nodes whose normalized attribute is in range",
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_filter_min = bpy.props.FloatProperty(
        name="Filter Min", default=0.0, min=0.0, max=1.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_filter_max = bpy.props.FloatProperty(
        name="Filter Max", default=1.0, min=0.0, max=1.0, update=_on_rebuild_update,
    )
    S.scigraphs_preview_lod_enabled = bpy.props.BoolProperty(
        name="Level of Detail",
        description="Subsample points/edges when the graph exceeds Max Points",
        default=False, update=_on_rebuild_update,
    )
    S.scigraphs_preview_lod_max_points = bpy.props.IntProperty(
        name="Max Points", description="Target maximum number of drawn points",
        default=300000, min=1000, max=10000000, update=_on_rebuild_update,
    )
    S.scigraphs_preview_depth_test = bpy.props.BoolProperty(
        name="Depth Test",
        description="Occlude points/lines behind other geometry",
        default=True, update=_on_setting_update,
    )
    S.scigraphs_preview_hide_mesh = bpy.props.BoolProperty(
        name="Hide Native Mesh",
        description=(
            "Switch the graph mesh to Bounds display while the preview is on "
            "so Blender does not draw it a second time (drawn twice = slower)"
        ),
        default=True, update=_on_hide_mesh_update,
    )


_PROP_NAMES = (
    "scigraphs_preview_enabled",
    "scigraphs_preview_node_size",
    "scigraphs_preview_round_points",
    "scigraphs_preview_color_mode",
    "scigraphs_preview_node_color",
    "scigraphs_preview_attr_name",
    "scigraphs_preview_colormap",
    "scigraphs_preview_reverse_colormap",
    "scigraphs_preview_size_by_attr",
    "scigraphs_preview_size_max_mult",
    "scigraphs_preview_show_edges",
    "scigraphs_preview_edge_color",
    "scigraphs_preview_edge_width",
    "scigraphs_preview_additive_edges",
    "scigraphs_preview_filter_enabled",
    "scigraphs_preview_filter_min",
    "scigraphs_preview_filter_max",
    "scigraphs_preview_lod_enabled",
    "scigraphs_preview_lod_max_points",
    "scigraphs_preview_depth_test",
    "scigraphs_preview_hide_mesh",
)


def _unregister_properties():
    S = bpy.types.Scene
    for name in _PROP_NAMES:
        if hasattr(S, name):
            delattr(S, name)


def register():
    _register_properties()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    if _on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_load_post)

    # GPU is the default engine: turn the preview on so imported graphs are
    # drawn on the GPU straight away.
    scene = getattr(bpy.context, "scene", None)
    if scene is not None and bool(_preview_prop(scene, "enabled", True)):
        enable_preview()


def unregister():
    if _on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load_post)
    disable_preview()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    _unregister_properties()
