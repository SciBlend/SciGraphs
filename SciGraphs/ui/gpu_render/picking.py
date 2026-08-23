# Click to inspect: the nearest node within a pixel radius.

import bpy
import numpy as np

from .attributes import read_attribute_values, scalar_attr_items
from .state import is_enabled, is_graph_object


def pick_nearest_node(context, obj, mouse_x, mouse_y, radius_px=20.0):
    """Node index, or None when nothing is inside ``radius_px``."""
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
    bl_idname = "scigraphs.pick_node"
    bl_label = "Pick Node"
    bl_description = "Click the nearest node to read its index and attributes (Esc/right-click to stop)"

    def modal(self, context, event):
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            context.workspace.status_text_set(None)
            return {'CANCELLED'}

        if event.type != 'LEFTMOUSE' or event.value != 'PRESS':
            return {'PASS_THROUGH'}
        if context.area is None or context.area.type != 'VIEW_3D':
            return {'PASS_THROUGH'}
        obj = context.active_object
        if not is_graph_object(obj):
            return {'RUNNING_MODAL'}
        idx = pick_nearest_node(
            context, obj,
            event.mouse_region_x, event.mouse_region_y,
        )
        if idx is None:
            self.report({'INFO'}, "No node under cursor")
            return {'RUNNING_MODAL'}

        obj["scigraphs_preview_picked"] = idx
        attr_bits = []
        for name, *_ in scalar_attr_items(obj)[:4]:
            vals = read_attribute_values(obj.data, name)
            if vals.size > idx:
                attr_bits.append(f"{name}={vals[idx]:.3g}")
        extra = ("  |  " + ", ".join(attr_bits)) if attr_bits else ""
        self.report({'INFO'}, f"Node #{idx}{extra}")
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        if not is_enabled():
            self.report({'WARNING'}, "Enable the GPU preview first")
            return {'CANCELLED'}
        context.workspace.status_text_set("Pick Node: click nodes  |  Esc/Right-click to stop")
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
