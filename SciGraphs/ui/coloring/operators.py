"""Operator classes for the floating coloring toolbar."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    IntProperty,
    StringProperty,
)

from ...core.coloring.colormaps import (
    COLORMAP_CATALOG,
    QUICK_COLORMAPS,
)
from . import functions as fn


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tag_3d_views(context):
    screen = getattr(context, "screen", None)
    if not screen:
        return
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


# ---------------------------------------------------------------------------
# Toolbar visibility / switching
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_color_toggle_toolbar(bpy.types.Operator):
    """Show or hide the floating coloring toolbar."""
    bl_idname = "scigraphs.color_toggle_toolbar"
    bl_label = "Toggle SciGraphs Color Toolbar"
    bl_description = "Show or hide the floating coloring toolbar at the top of the 3D view"

    def execute(self, context):
        wm = context.window_manager
        wm.scigraphs_show_color_toolbar = not wm.scigraphs_show_color_toolbar
        _tag_3d_views(context)
        status = "shown" if wm.scigraphs_show_color_toolbar else "hidden"
        self.report({'INFO'}, f"Coloring toolbar {status}")
        return {'FINISHED'}


class SCIGRAPHS_OT_color_show_toolbar(bpy.types.Operator):
    """Force-show the floating coloring toolbar."""
    bl_idname = "scigraphs.color_show_toolbar"
    bl_label = "Show SciGraphs Color Toolbar"
    bl_description = "Make sure the floating coloring toolbar is visible"

    def execute(self, context):
        context.window_manager.scigraphs_show_color_toolbar = True
        _tag_3d_views(context)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Drag handle
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_color_drag_toolbar(bpy.types.Operator):
    """Drag the floating coloring toolbar."""
    bl_idname = "scigraphs.color_drag_toolbar"
    bl_label = "Move SciGraphs Color Toolbar"
    bl_description = "Drag to reposition the floating coloring toolbar"

    _start_mouse = None
    _start_pos = None

    def invoke(self, context, event):
        wm = context.window_manager
        region = context.region
        if region is None:
            return {'CANCELLED'}

        if wm.scigraphs_color_toolbar_x < 0 or wm.scigraphs_color_toolbar_y < 0:
            wm.scigraphs_color_toolbar_x = int(region.width * 0.5)
            wm.scigraphs_color_toolbar_y = int(region.height - 64)

        self._start_mouse = (event.mouse_region_x, event.mouse_region_y)
        self._start_pos = (
            wm.scigraphs_color_toolbar_x,
            wm.scigraphs_color_toolbar_y,
        )
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        wm = context.window_manager

        if event.type == 'MOUSEMOVE':
            dx = event.mouse_region_x - self._start_mouse[0]
            dy = event.mouse_region_y - self._start_mouse[1]
            wm.scigraphs_color_toolbar_x = max(8, int(self._start_pos[0] + dx))
            wm.scigraphs_color_toolbar_y = max(8, int(self._start_pos[1] + dy))
            _tag_3d_views(context)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            wm.scigraphs_color_toolbar_x, wm.scigraphs_color_toolbar_y = self._start_pos
            _tag_3d_views(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def execute(self, context):
        # Plain click without drag: re-center against the top of the viewport.
        region = context.region
        wm = context.window_manager
        if region is not None:
            wm.scigraphs_color_toolbar_x = int(region.width * 0.5)
            wm.scigraphs_color_toolbar_y = int(region.height - 64)
            _tag_3d_views(context)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Apply / refresh / colormap chips
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_color_apply(bpy.types.Operator):
    """Apply the current coloring settings to the active mesh."""
    bl_idname = "scigraphs.color_apply"
    bl_label = "Apply Coloring"
    bl_description = "Map the selected attribute to vertex colors using the current colormap"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return fn.active_mesh_object(context) is not None

    def execute(self, context):
        ok, message = fn.apply_coloring(context)
        if not ok:
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        self.report({'INFO'}, message)
        _tag_3d_views(context)
        return {'FINISHED'}


class SCIGRAPHS_OT_color_refresh_range(bpy.types.Operator):
    """Refresh ``vmin``/``vmax`` from the currently selected attribute."""
    bl_idname = "scigraphs.color_refresh_range"
    bl_label = "Refresh Color Range"
    bl_description = "Recalculate vmin/vmax from the selected attribute and re-apply if requested"
    bl_options = {'REGISTER', 'UNDO'}

    reapply: BoolProperty(
        name="Re-apply",
        description="Re-apply the colormap after recomputing the range",
        default=True,
        options={'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        return fn.active_mesh_object(context) is not None

    def execute(self, context):
        obj = fn.active_mesh_object(context)
        props = fn.coloring_props(context)
        if obj is None or props is None:
            self.report({'WARNING'}, "No mesh / coloring settings available")
            return {'CANCELLED'}

        attribute_name = fn.resolve_attribute_name(props, obj)
        if not attribute_name:
            self.report({'WARNING'}, "No scalar attribute available on this mesh")
            return {'CANCELLED'}

        props.attribute_name = attribute_name
        vmin, vmax = fn.update_property_range(props, obj, attribute_name)
        if vmin is None:
            self.report({'WARNING'}, f"Attribute '{attribute_name}' is empty")
            return {'CANCELLED'}

        props.auto_range = True
        if self.reapply:
            ok, message = fn.apply_coloring(context)
            if not ok:
                self.report({'WARNING'}, message)
                return {'CANCELLED'}
            self.report(
                {'INFO'},
                f"Range refreshed: [{vmin:.4g} … {vmax:.4g}] – {message}",
            )
        else:
            self.report({'INFO'}, f"Range refreshed: [{vmin:.4g} … {vmax:.4g}]")
        _tag_3d_views(context)
        return {'FINISHED'}


class SCIGRAPHS_OT_color_set_colormap(bpy.types.Operator):
    """Pick a colormap (from a chip or the popup) and optionally apply it."""
    bl_idname = "scigraphs.color_set_colormap"
    bl_label = "Set Colormap"
    bl_description = "Switch to the chosen colormap and re-apply when configured to do so"
    bl_options = {'REGISTER', 'UNDO'}

    colormap: StringProperty(
        name="Colormap",
        description="Identifier of the colormap to apply",
        default="viridis",
        options={'SKIP_SAVE'},
    )

    apply_now: BoolProperty(
        name="Apply Now",
        description="Re-color the mesh after switching colormap",
        default=True,
        options={'SKIP_SAVE'},
    )

    def execute(self, context):
        props = fn.coloring_props(context)
        if props is None:
            self.report({'WARNING'}, "Coloring settings not registered")
            return {'CANCELLED'}

        catalog = {ident for ident, *_ in COLORMAP_CATALOG}
        if self.colormap not in catalog:
            self.report({'WARNING'}, f"Unknown colormap '{self.colormap}'")
            return {'CANCELLED'}

        props.colormap = self.colormap

        if not self.apply_now:
            self.report({'INFO'}, f"Colormap set to {self.colormap}")
            return {'FINISHED'}

        if not props.auto_apply_on_chip:
            self.report({'INFO'}, f"Colormap set to {self.colormap}")
            return {'FINISHED'}

        if fn.active_mesh_object(context) is None:
            self.report({'INFO'}, f"Colormap set to {self.colormap} (no active mesh to apply on)")
            return {'FINISHED'}

        ok, message = fn.apply_coloring(context)
        if not ok:
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        self.report({'INFO'}, message)
        _tag_3d_views(context)
        return {'FINISHED'}


class SCIGRAPHS_OT_color_toggle_reverse(bpy.types.Operator):
    """Toggle the ``reverse`` flag and re-apply the current colormap."""
    bl_idname = "scigraphs.color_toggle_reverse"
    bl_label = "Reverse Colormap"
    bl_description = "Flip the colormap orientation and re-apply"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = fn.coloring_props(context)
        if props is None:
            self.report({'WARNING'}, "Coloring settings not registered")
            return {'CANCELLED'}

        props.reverse = not props.reverse
        if fn.active_mesh_object(context) is None:
            self.report({'INFO'}, f"Reverse = {props.reverse}")
            return {'FINISHED'}

        ok, message = fn.apply_coloring(context)
        if not ok:
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Reverse = {props.reverse}  ·  {message}")
        _tag_3d_views(context)
        return {'FINISHED'}


class SCIGRAPHS_OT_color_cycle_attribute(bpy.types.Operator):
    """Cycle through the scalar attributes of the active mesh."""
    bl_idname = "scigraphs.color_cycle_attribute"
    bl_label = "Cycle Attribute"
    bl_description = "Switch to the next scalar attribute on the active mesh and re-apply"
    bl_options = {'REGISTER', 'UNDO'}

    step: IntProperty(
        name="Step",
        description="+1 for next attribute, -1 for previous",
        default=1,
        options={'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        return fn.active_mesh_object(context) is not None

    def execute(self, context):
        obj = fn.active_mesh_object(context)
        props = fn.coloring_props(context)
        if obj is None or props is None:
            self.report({'WARNING'}, "No mesh / coloring settings available")
            return {'CANCELLED'}

        names = fn.attribute_names(obj)
        if not names:
            self.report({'WARNING'}, "No scalar attribute on this mesh")
            return {'CANCELLED'}

        current = fn.resolve_attribute_name(props, obj)
        next_name = fn.cycle_attribute_name(current, names, self.step)
        props.attribute_name = next_name
        try:
            props.attribute_enum = next_name
        except TypeError:
            pass

        fn.update_property_range(props, obj, next_name)

        ok, message = fn.apply_coloring(context)
        if not ok:
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Attribute -> {next_name}  ·  {message}")
        _tag_3d_views(context)
        return {'FINISHED'}


class SCIGRAPHS_OT_color_set_attribute(bpy.types.Operator):
    """Set the active scalar attribute by name and re-apply coloring."""
    bl_idname = "scigraphs.color_set_attribute"
    bl_label = "Set Coloring Attribute"
    bl_description = "Switch the coloring toolbar to a specific scalar attribute"
    bl_options = {'REGISTER', 'UNDO'}

    attribute: StringProperty(
        name="Attribute",
        description="Name of the scalar attribute to use",
        default="",
        options={'SKIP_SAVE'},
    )

    @classmethod
    def poll(cls, context):
        return fn.active_mesh_object(context) is not None

    def execute(self, context):
        obj = fn.active_mesh_object(context)
        props = fn.coloring_props(context)
        if obj is None or props is None:
            self.report({'WARNING'}, "No mesh / coloring settings available")
            return {'CANCELLED'}

        names = fn.attribute_names(obj)
        if not names:
            self.report({'WARNING'}, "No scalar attribute on this mesh")
            return {'CANCELLED'}

        target = self.attribute.strip()
        if target not in names:
            self.report({'WARNING'}, f"Attribute '{target}' not found on this mesh")
            return {'CANCELLED'}

        props.attribute_name = target
        try:
            props.attribute_enum = target
        except TypeError:
            pass

        fn.update_property_range(props, obj, target)

        ok, message = fn.apply_coloring(context)
        if not ok:
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        self.report({'INFO'}, f"Attribute -> {target}  ·  {message}")
        _tag_3d_views(context)
        return {'FINISHED'}


class SCIGRAPHS_OT_color_pick_attribute(bpy.types.Operator):
    """Open a popup menu to choose the scalar attribute to color with."""
    bl_idname = "scigraphs.color_pick_attribute"
    bl_label = "Pick Coloring Attribute"
    bl_description = "Open a menu listing every scalar attribute on the active mesh"

    @classmethod
    def poll(cls, context):
        obj = fn.active_mesh_object(context)
        return obj is not None and bool(fn.attribute_names(obj))

    def invoke(self, context, _event):
        bpy.ops.wm.call_menu(name=SCIGRAPHS_MT_color_attribute_menu.bl_idname)
        return {'FINISHED'}

    def execute(self, context):
        return self.invoke(context, None)


class SCIGRAPHS_MT_color_attribute_menu(bpy.types.Menu):
    """Popup menu listing every scalar attribute available on the active mesh."""
    bl_idname = "SCIGRAPHS_MT_color_attribute_menu"
    bl_label = "Coloring Attribute"

    def draw(self, context):
        layout = self.layout
        obj = fn.active_mesh_object(context)
        props = fn.coloring_props(context)

        if obj is None:
            layout.label(text="No active mesh", icon='ERROR')
            return

        names = fn.attribute_names(obj)
        if not names:
            layout.label(text="No scalar attributes on this mesh", icon='INFO')
            return

        current = fn.resolve_attribute_name(props, obj) if props else ""

        for name in names:
            icon = 'RADIOBUT_ON' if name == current else 'RADIOBUT_OFF'
            op = layout.operator(
                "scigraphs.color_set_attribute",
                text=name,
                icon=icon,
            )
            op.attribute = name


class SCIGRAPHS_OT_color_remove(bpy.types.Operator):
    """Remove the last color attribute written by the toolbar."""
    bl_idname = "scigraphs.color_remove"
    bl_label = "Remove Coloring"
    bl_description = "Remove the most recent color attribute created by the coloring toolbar"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return fn.active_mesh_object(context) is not None

    def execute(self, context):
        ok, message = fn.remove_coloring(context)
        if not ok:
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        self.report({'INFO'}, message)
        _tag_3d_views(context)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Settings popup
# ---------------------------------------------------------------------------

class SCIGRAPHS_OT_color_settings_dialog(bpy.types.Operator):
    """Open a popup with the full coloring configuration (attribute, colormap, range, ...)."""
    bl_idname = "scigraphs.color_settings_dialog"
    bl_label = "Coloring Settings"
    bl_description = "Configure attribute, colormap, range, opacity, and material auto-setup"

    def invoke(self, context, _event):
        # Sync the enum with the stored string so the popup reflects the
        # currently active attribute.
        props = fn.coloring_props(context)
        obj = fn.active_mesh_object(context)
        if props and obj:
            attribute_name = fn.resolve_attribute_name(props, obj)
            if attribute_name:
                props.attribute_name = attribute_name
                try:
                    props.attribute_enum = attribute_name
                except TypeError:
                    pass
                if props.auto_range:
                    fn.update_property_range(props, obj, attribute_name)
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        props = fn.coloring_props(context)
        obj = fn.active_mesh_object(context)
        if props is None:
            layout.label(text="Coloring settings unavailable", icon='ERROR')
            return

        if obj is None:
            warn = layout.box()
            warn.label(text="Select a mesh object to color", icon='INFO')
            return

        if not fn.has_scalar_attributes(obj):
            warn = layout.box()
            warn.label(text="Active mesh has no scalar attributes", icon='ERROR')
            warn.label(
                text="Compute centrality, accessibility, or any custom attribute first.",
                icon='QUESTION',
            )
            return

        source_box = layout.box()
        source_box.label(text="Source", icon='OUTLINER_DATA_POINTCLOUD')
        # The enum's update callback (in properties.py) writes the chosen
        # name into ``attribute_name``; we must NOT touch ID properties here
        # because Blender disallows ID writes during draw().
        source_box.prop(props, "attribute_enum", text="Attribute")
        info_row = source_box.row(align=True)
        info_row.alignment = 'RIGHT'
        info_row.label(
            text=f"Last range: [{props.last_vmin:.4g} … {props.last_vmax:.4g}]",
            icon='IPO_LINEAR',
        )
        op = source_box.operator(
            "scigraphs.color_refresh_range",
            text="Refresh Range",
            icon='FILE_REFRESH',
        )
        op.reapply = False

        cmap_box = layout.box()
        cmap_box.label(text="Colormap", icon='COLOR')
        cmap_box.prop(props, "colormap", text="Colormap")
        cmap_box.prop(props, "reverse")

        range_box = layout.box()
        range_box.label(text="Value Range", icon='ARROW_LEFTRIGHT')
        range_box.prop(props, "auto_range")
        sub = range_box.column(align=True)
        sub.enabled = not props.auto_range
        sub.prop(props, "vmin")
        sub.prop(props, "vmax")

        out_box = layout.box()
        out_box.label(text="Output", icon='RENDER_RESULT')
        out_box.prop(props, "color_domain")
        out_box.prop(props, "color_attribute_name", text="Layer Name")
        out_box.prop(props, "opacity")
        out_box.prop(props, "auto_setup_material")
        out_box.prop(props, "auto_apply_on_chip")

        gate_box = layout.box()
        gate_box.label(text="Node Filter", icon='SNAP_VERTEX')
        has_intersection = (
            obj.type == 'MESH'
            and 'is_intersection' in obj.data.attributes
        )
        has_visual = (
            hasattr(obj, "modifiers") and obj.modifiers.get("SciGraphs_Viz") is not None
        )
        gate_available = has_intersection or has_visual
        sub = gate_box.column(align=True)
        sub.enabled = gate_available
        sub.prop(props, "nodes_only")
        if props.nodes_only:
            sub.prop(props, "edge_color", text="Edge Color")
        if not gate_available:
            gate_box.label(
                text="No 'is_intersection' attribute and no SciGraphs_Viz modifier",
                icon='INFO',
            )
        elif has_visual:
            gate_box.label(
                text="Gating via 'scigraphs_is_node' marker stamped by SciGraphs_Viz",
                icon='CHECKMARK',
            )

    def execute(self, context):
        ok, message = fn.apply_coloring(context)
        if not ok:
            self.report({'WARNING'}, message)
            return {'CANCELLED'}
        self.report({'INFO'}, message)
        _tag_3d_views(context)
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (
    SCIGRAPHS_OT_color_toggle_toolbar,
    SCIGRAPHS_OT_color_show_toolbar,
    SCIGRAPHS_OT_color_drag_toolbar,
    SCIGRAPHS_OT_color_apply,
    SCIGRAPHS_OT_color_refresh_range,
    SCIGRAPHS_OT_color_set_colormap,
    SCIGRAPHS_OT_color_toggle_reverse,
    SCIGRAPHS_OT_color_cycle_attribute,
    SCIGRAPHS_OT_color_set_attribute,
    SCIGRAPHS_MT_color_attribute_menu,
    SCIGRAPHS_OT_color_pick_attribute,
    SCIGRAPHS_OT_color_remove,
    SCIGRAPHS_OT_color_settings_dialog,
)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


__all__ = [
    "QUICK_COLORMAPS",
    "SCIGRAPHS_OT_color_toggle_toolbar",
    "SCIGRAPHS_OT_color_show_toolbar",
    "SCIGRAPHS_OT_color_drag_toolbar",
    "SCIGRAPHS_OT_color_apply",
    "SCIGRAPHS_OT_color_refresh_range",
    "SCIGRAPHS_OT_color_set_colormap",
    "SCIGRAPHS_OT_color_toggle_reverse",
    "SCIGRAPHS_OT_color_cycle_attribute",
    "SCIGRAPHS_OT_color_set_attribute",
    "SCIGRAPHS_OT_color_pick_attribute",
    "SCIGRAPHS_MT_color_attribute_menu",
    "SCIGRAPHS_OT_color_remove",
    "SCIGRAPHS_OT_color_settings_dialog",
]
