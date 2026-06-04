# Text overlay operators for graph node labels
#
# Provides operators to generate text overlay images with node labels
# projected from 3D positions to screen coordinates.

import bpy

from ....core import text_overlay
from ....core.visualization.text_overlay import (
    get_font_path,
    get_settings_snapshot_with_object,
)
from ....utils.logger import log


class SCIGRAPHS_OT_GenerateTextOverlay(bpy.types.Operator):
    """Generate text overlay image for graph node labels"""
    bl_idname = "scigraphs.generate_text_overlay"
    bl_label = "Generate Text Overlay"
    bl_description = "Generate PNG with node labels and add to compositor"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            return False
        has_graph_data = (
            "node_names" in obj or 
            "num_nodes" in obj or 
            "nodes_data" in obj or
            len(obj.data.vertices) > 0
        )
        return has_graph_data
    
    def execute(self, context):
        scene = context.scene
        props = scene.scigraphs
        obj = context.active_object
        
        log(f"Generating text overlay for object: {obj.name}")
        
        camera = scene.camera
        if camera is None:
            self.report({'ERROR'}, "No active camera in scene. Add a camera first.")
            return {'CANCELLED'}
        
        if camera.type != 'CAMERA':
            self.report({'ERROR'}, "Active camera is not a Camera object")
            return {'CANCELLED'}
        
        render = scene.render
        resolution = (
            int(render.resolution_x * render.resolution_percentage / 100),
            int(render.resolution_y * render.resolution_percentage / 100)
        )
        
        font_path = get_font_path(props)
        
        settings = text_overlay.TextOverlaySettings(
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
            font_path=font_path,
        )
        
        projected = text_overlay.project_nodes_to_screen(
            obj, camera, scene, resolution
        )
        
        if not projected:
            self.report({'WARNING'}, "No nodes to project")
            return {'CANCELLED'}
        
        log(f"Projected {len(projected)} nodes to screen")
        
        if settings.depth_occlusion:
            projected = text_overlay.test_depth_occlusion(
                context, obj, projected, camera
            )
            visible_count = sum(1 for n in projected if not n.occluded)
            log(f"After occlusion: {visible_count} visible nodes")
        
        if settings.max_distance > 0:
            projected = text_overlay.apply_distance_filter(
                projected, settings.max_distance
            )
            log(f"After distance filter: {len(projected)} nodes")
        
        if settings.filter_enabled and settings.filter_attribute and settings.filter_attribute != 'NONE':
            projected = text_overlay.apply_attribute_filter(
                projected, obj, settings
            )
            log(f"After attribute filter: {len(projected)} nodes")
        
        if props.text_source == 'ATTRIBUTE' and props.text_attribute and props.text_attribute != 'NONE':
            attr_values = text_overlay.get_node_attribute_values(obj, props.text_attribute)
            if not attr_values:
                self.report(
                    {'WARNING'},
                    f"Attribute '{props.text_attribute}' has no values; showing node IDs"
                )
            for node in projected:
                if node.name in attr_values:
                    node.name = repr(attr_values[node.name])
        
        image_path = text_overlay.generate_text_image(
            projected, resolution, settings, camera
        )
        
        if image_path is None:
            self.report({'ERROR'}, "Failed to generate text overlay image")
            return {'CANCELLED'}
        
        success = text_overlay.setup_compositor_overlay(scene, image_path)
        
        if not success:
            self.report({'ERROR'}, "Failed to setup compositor")
            return {'CANCELLED'}
        
        props.text_overlay_enabled = True
        
        visible_count = sum(1 for n in projected if n.visible and not n.occluded)
        self.report({'INFO'}, f"Text overlay generated with {visible_count} labels")
        
        return {'FINISHED'}


class SCIGRAPHS_OT_RemoveTextOverlay(bpy.types.Operator):
    """Remove text overlay from compositor"""
    bl_idname = "scigraphs.remove_text_overlay"
    bl_label = "Remove Text Overlay"
    bl_description = "Remove text overlay nodes from compositor"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        props = scene.scigraphs
        
        success = text_overlay.remove_compositor_overlay(scene)
        
        if success:
            props.text_overlay_enabled = False
            self.report({'INFO'}, "Text overlay removed")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "Could not remove text overlay")
            return {'CANCELLED'}


class SCIGRAPHS_OT_StartAutoUpdate(bpy.types.Operator):
    """Start automatic text overlay update when view or settings change"""
    bl_idname = "scigraphs.start_auto_update"
    bl_label = "Start Auto Update"
    bl_description = "Continuously update text overlay as view or settings change"
    
    _timer = None
    _last_snapshot = None
    _target_object_name = None
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if obj is None or obj.type != 'MESH':
            return False
        return context.scene.camera is not None
    
    def modal(self, context, event):
        if not context.window_manager.get("scigraphs_auto_update_running", False):
            self.cancel(context)
            return {'CANCELLED'}
        
        if event.type == 'TIMER':
            scene = context.scene
            camera = scene.camera
            
            if camera is None:
                self.cancel(context)
                return {'CANCELLED'}
            
            obj = bpy.data.objects.get(self._target_object_name)
            if obj is None:
                log(f"Auto-update: Target object '{self._target_object_name}' not found")
                self.cancel(context)
                return {'CANCELLED'}
            
            current_snapshot = get_settings_snapshot_with_object(context, obj)
            
            changed = (self._last_snapshot is None or 
                       current_snapshot != self._last_snapshot)
            
            if changed:
                self._last_snapshot = current_snapshot
                
                try:
                    with context.temp_override(active_object=obj):
                        bpy.ops.scigraphs.generate_text_overlay()
                except Exception as e:
                    log(f"Auto-update error: {e}")
        
        return {'PASS_THROUGH'}
    
    def execute(self, context):
        props = context.scene.scigraphs
        interval = props.text_auto_update_interval
        
        self._target_object_name = context.active_object.name
        
        context.window_manager["scigraphs_auto_update_running"] = True
        
        wm = context.window_manager
        self._timer = wm.event_timer_add(interval, window=context.window)
        
        self._last_snapshot = None
        
        wm.modal_handler_add(self)
        
        self.report({'INFO'}, f"Auto-update started for '{self._target_object_name}'")
        return {'RUNNING_MODAL'}
    
    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None
        
        context.window_manager["scigraphs_auto_update_running"] = False
        log("Auto-update stopped")


class SCIGRAPHS_OT_StopAutoUpdate(bpy.types.Operator):
    """Stop automatic text overlay update"""
    bl_idname = "scigraphs.stop_auto_update"
    bl_label = "Stop Auto Update"
    bl_description = "Stop continuous overlay updates"
    
    def execute(self, context):
        context.window_manager["scigraphs_auto_update_running"] = False
        self.report({'INFO'}, "Auto-update stopped")
        return {'FINISHED'}


classes = [
    SCIGRAPHS_OT_GenerateTextOverlay,
    SCIGRAPHS_OT_RemoveTextOverlay,
    SCIGRAPHS_OT_StartAutoUpdate,
    SCIGRAPHS_OT_StopAutoUpdate,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
