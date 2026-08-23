import bpy
import numpy as np

from . import draw, dynamic, playback
from .attributes import scalar_attr_items
from .state import is_graph_object


class SCIGRAPHS_OT_animate_layout(bpy.types.Operator):
    bl_idname = "scigraphs.animate_layout"
    bl_label = "Animate Layout"
    bl_description = (
        "Run a force-directed layout that advances with the timeline, so "
        "pressing Play shows the graph arranging itself. The trajectory is "
        "recorded as it runs, so scrubbing back and rendering show exactly "
        "what was watched"
    )

    def execute(self, context):
        scene = context.scene
        obj = context.active_object
        if not is_graph_object(obj):
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}

        if playback.is_running(obj):
            playback.stop(obj)
            self.report({'INFO'}, "Layout animation stopped; mesh updated")
            return {'FINISHED'}

        # Switch to the animated draw path instead of silently using the slow one.
        if not scene.scigraphs_preview_animate:
            scene.scigraphs_preview_animate = True
        ok, why = dynamic.eligible(obj, scene)
        if not ok:
            self.report({'ERROR'}, f"Cannot animate this graph: {why}")
            return {'CANCELLED'}

        scene.frame_set(scene.frame_start)
        # Build the bundle first or the handler finds no entry and switches off.
        draw.invalidate(obj)
        draw.get_cache_entry(obj, scene)

        pb = playback.start(obj, scene)
        playback.register_handler()
        n = pb.frames[0].shape[0]
        self.report(
            {'INFO'},
            f"Animating {n:,} nodes; press Play. "
            f"Recording up to {pb.max_frames} frames "
            f"({playback.trajectory_bytes(n, pb.max_frames) / 1e6:.0f} MB)")
        return {'FINISHED'}


class SCIGRAPHS_OT_reset_animation(bpy.types.Operator):
    """Discard the recorded trajectory and leave the mesh where it started"""
    bl_idname = "scigraphs.reset_animation"
    bl_label = "Reset Animation"

    def execute(self, context):
        playback.drop_cache(context.active_object)
        draw.invalidate(context.active_object)
        self.report({'INFO'}, "Trajectory discarded")
        return {'FINISHED'}


class SCIGRAPHS_OT_set_preview_attr(bpy.types.Operator):
    """Pick the scalar attribute used by the GPU preview"""
    bl_idname = "scigraphs.set_preview_attr"
    bl_label = "Set Preview Attribute"
    bl_property = "attr"

    def _items(self, context):
        obj = getattr(context, "active_object", None)
        items = [("", "None", "No attribute")]
        for name, data_type, domain in scalar_attr_items(obj):
            if domain == 'POINT':
                items.append((name, name, f"{data_type} on {domain}"))
        return items

    attr: bpy.props.EnumProperty(name="Attribute", items=_items)

    def execute(self, context):
        context.scene.scigraphs_preview_attr_name = self.attr
        draw.invalidate()
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'FINISHED'}


class SCIGRAPHS_OT_set_edge_attr(bpy.types.Operator):
    """Pick the EDGE attribute that drives edge width"""
    bl_idname = "scigraphs.set_edge_attr"
    bl_label = "Set Edge Width Attribute"
    bl_property = "attr"

    def _items(self, context):
        from .attributes import edge_scalar_attr_items
        obj = getattr(context, "active_object", None)
        items = [("", "None", "No attribute")]
        for name in edge_scalar_attr_items(obj):
            items.append((name, name, "EDGE scalar"))
        return items

    attr: bpy.props.EnumProperty(name="Attribute", items=_items)

    def execute(self, context):
        context.scene.scigraphs_preview_edge_attr_name = self.attr
        draw.invalidate()
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.invoke_search_popup(self)
        return {'FINISHED'}


class SCIGRAPHS_OT_bake_edge_channel(bpy.types.Operator):
    bl_idname = "scigraphs.bake_edge_channel"
    bl_label = "Bake Edge Channel"
    bl_description = (
        "Compute one of the engine's edge measures and store it as an EDGE "
        "attribute. Answers the 'where do weights come from' question for a "
        "graph that has none: length, triangle support, neighborhood overlap "
        "and the rest are properties of the graph itself"
    )
    bl_options = {'REGISTER', 'UNDO'}

    channel: bpy.props.EnumProperty(
        name="Channel",
        items=lambda self, ctx: _edge_channel_items(),
    )

    def execute(self, context):
        from . import filters
        obj = context.active_object
        scene = context.scene
        if not is_graph_object(obj):
            self.report({'ERROR'}, "No graph object selected")
            return {'CANCELLED'}
        try:
            values = filters.channel(obj, scene, self.channel, "")[0]
        except Exception as exc:  # noqa: BLE001
            self.report({'ERROR'}, f"Could not compute {self.channel}: {exc}")
            return {'CANCELLED'}
        if values is None or values.size != len(obj.data.edges):
            self.report({'ERROR'},
                        f"{self.channel} did not return one value per edge")
            return {'CANCELLED'}

        name = f"edge_{self.channel.lower()}"
        mesh = obj.data
        attr = mesh.attributes.get(name)
        if attr is not None and attr.domain != 'EDGE':
            self.report({'ERROR'}, f"'{name}' exists on another domain")
            return {'CANCELLED'}
        if attr is None:
            attr = mesh.attributes.new(name=name, type='FLOAT', domain='EDGE')
        attr.data.foreach_set("value",
                              np.asarray(values, dtype=np.float32).ravel())
        mesh.update()

        scene.scigraphs_preview_edge_attr_name = name
        scene.scigraphs_preview_edge_size_by_attr = True
        draw.invalidate_geometry(obj)
        self.report({'INFO'}, f"Baked '{name}' and set it as the width source")
        return {'FINISHED'}


def _edge_channel_items():
    from . import filters
    out = []
    for cid in filters.EDGE_CHANNELS:
        if cid == 'EDGE_ATTR':
            continue  # reads an attribute; baking it would be a copy
        out.append((cid, filters.CHANNEL_LABELS.get(cid, cid), ""))
    return out


class SCIGRAPHS_OT_toggle_gpu_preview(bpy.types.Operator):
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
    bl_idname = "scigraphs.refresh_gpu_preview"
    bl_label = "Refresh GPU Preview"
    bl_description = "Rebuild the GPU preview after a layout or coloring change"

    def execute(self, context):
        # Geometry level: this button exists for changes no signature catches.
        draw.invalidate_geometry()
        self.report({'INFO'}, "GPU preview refreshed")
        return {'FINISHED'}


class SCIGRAPHS_OT_filter_add(bpy.types.Operator):
    bl_idname = "scigraphs.filter_add"
    bl_label = "Add Filter"
    bl_description = (
        "Add a clause to the filter stack. Clauses are combined with AND, and a "
        "new one starts at its full range, so adding it changes nothing until "
        "it is narrowed"
    )

    channel: bpy.props.StringProperty(default='DEGREE', options={'HIDDEN'})

    def execute(self, context):
        scene = context.scene
        slot = scene.scigraphs_filters.add()
        slot.channel = self.channel
        scene.scigraphs_filters_index = len(scene.scigraphs_filters) - 1
        draw.invalidate()
        return {'FINISHED'}


class SCIGRAPHS_OT_filter_remove(bpy.types.Operator):
    """Remove the active clause from the filter stack"""
    bl_idname = "scigraphs.filter_remove"
    bl_label = "Remove Filter"

    @classmethod
    def poll(cls, context):
        return bool(getattr(context.scene, "scigraphs_filters", None))

    def execute(self, context):
        scene = context.scene
        index = scene.scigraphs_filters_index
        if 0 <= index < len(scene.scigraphs_filters):
            scene.scigraphs_filters.remove(index)
            scene.scigraphs_filters_index = min(
                index, max(0, len(scene.scigraphs_filters) - 1))
            draw.invalidate()
        return {'FINISHED'}


class SCIGRAPHS_OT_filter_seed(bpy.types.Operator):
    bl_idname = "scigraphs.filter_seed"
    bl_label = "Seed From Selection"
    bl_description = (
        "Write the current vertex selection as the seed set, so Hops From Seed "
        "measures distance to it. Stored as a mesh attribute rather than read "
        "from the selection every frame, which is what makes an ego network "
        "survive saving the file -- and what keeps the seed still while the "
        "selection moves"
    )

    clear: bpy.props.BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        obj = getattr(context, "active_object", None)
        return obj is not None and obj.type == 'MESH'

    def execute(self, context):
        import numpy as np

        from . import filters
        obj = context.active_object
        mesh = obj.data

        # Edit Mode keeps the live selection in the BMesh; the datablock is stale.
        if obj.mode == 'EDIT':
            import bmesh
            bm = bmesh.from_edit_mesh(mesh)
            mask = np.zeros(len(bm.verts), dtype=bool)
            for i, vert in enumerate(bm.verts):
                mask[i] = vert.select
        else:
            mask = np.zeros(len(mesh.vertices), dtype=bool)
            mesh.vertices.foreach_get("select", mask)

        if self.clear:
            mask[:] = False
        elif not mask.any():
            self.report({'WARNING'}, "Nothing is selected")
            return {'CANCELLED'}

        name = filters.SEED_ATTR
        if name in mesh.attributes:
            mesh.attributes.remove(mesh.attributes[name])
        if mask.any():
            attr = mesh.attributes.new(name=name, type='BOOLEAN',
                                       domain='POINT')
            attr.data.foreach_set("value", mask)

        # The only writer. The panel reads this count rather than every boolean.
        obj[filters.SEED_COUNT] = int(mask.sum())

        # The seed feeds a cached channel no signature sees, so go geometry level.
        draw.invalidate_geometry(obj)
        self.report({'INFO'}, f"{int(mask.sum()):,} nodes seeded")
        return {'FINISHED'}


class SCIGRAPHS_OT_filter_move(bpy.types.Operator):
    bl_idname = "scigraphs.filter_move"
    bl_label = "Move Filter"
    bl_description = (
        "Reorder the stack. The result is the same either way -- AND does not "
        "care about order -- so this is for reading, not for meaning"
    )

    direction: bpy.props.EnumProperty(
        items=[('UP', "Up", ""), ('DOWN', "Down", "")], default='UP',
        options={'HIDDEN'},
    )

    @classmethod
    def poll(cls, context):
        return len(getattr(context.scene, "scigraphs_filters", ())) > 1

    def execute(self, context):
        scene = context.scene
        slots = scene.scigraphs_filters
        index = scene.scigraphs_filters_index
        target = index - 1 if self.direction == 'UP' else index + 1
        if 0 <= target < len(slots):
            slots.move(index, target)
            scene.scigraphs_filters_index = target
        return {'FINISHED'}
