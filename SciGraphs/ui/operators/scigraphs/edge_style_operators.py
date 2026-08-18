"""Operators for applying, resetting and presetting graph edge styles."""

import bpy
from bpy.props import EnumProperty, BoolProperty

from ....core import geometry
from scigraphs_core import edge_styles
from scigraphs_core.logger import log


# Shared by the operator's enum and by its invoke(), which validates against it.
_PRESET_ITEMS = [
    ('GEPHI_DEFAULT', "Gephi Default", ""),
    ('CYTOSCAPE_BEZIER', "Cytoscape Bezier", ""),
    ('SCHEMATIC', "Schematic", ""),
    ('BUNDLED_DENSE', "Bundled (Dense)", ""),
    ('FLOW_DIAGRAM', "Flow Diagram", ""),
    ('MINIMAL', "Minimal", ""),
]
_PRESET_IDS = {item[0] for item in _PRESET_ITEMS}


class SCIGRAPHS_OT_ApplyEdgeStyle(bpy.types.Operator):
    bl_idname = "scigraphs.apply_edge_style"
    bl_label = "Apply Edge Style"
    bl_description = "Apply the selected edge style to graph edges"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and "num_nodes" in obj
    
    def execute(self, context):
        obj = context.active_object
        props = context.scene.scigraphs
        
        style_params = edge_styles.get_style_params_from_props(props)
        success = geometry.apply_edge_style_to_graph(obj, style_params)

        if success:
            self.report({'INFO'}, f"Applied '{props.edge_style_type}' edge style")

            mod = obj.modifiers.get("SciGraphs_Viz")
            if mod:
                obj.data.update()
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        else:
            self.report({'ERROR'}, "Failed to apply edge style")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ResetEdgeStyle(bpy.types.Operator):
    bl_idname = "scigraphs.reset_edge_style"
    bl_label = "Reset to Straight"
    bl_description = "Reset all edges to straight lines"
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and "num_nodes" in obj
    
    def execute(self, context):
        obj = context.active_object
        
        success = geometry.reset_edge_style(obj)
        
        if success:
            self.report({'INFO'}, "Edges reset to straight lines")
            
            props = context.scene.scigraphs
            props.edge_style_type = 'STRAIGHT'
            props.edge_style_preset = 'MINIMAL'

            obj.data.update()
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        else:
            self.report({'ERROR'}, "Failed to reset edge style")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ApplyEdgeStylePreset(bpy.types.Operator):
    bl_idname = "scigraphs.apply_edge_style_preset"
    bl_label = "Apply Edge Preset"
    bl_description = "Apply a predefined edge style configuration"
    bl_options = {'REGISTER', 'UNDO'}
    
    preset: EnumProperty(
        name="Preset",
        items=_PRESET_ITEMS,
        default='GEPHI_DEFAULT',
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and "num_nodes" in obj
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        self.preset = props.edge_style_preset if props.edge_style_preset in _PRESET_IDS else 'GEPHI_DEFAULT'
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        edge_styles.apply_preset(props, self.preset)
        props.edge_style_preset = self.preset

        obj = context.active_object
        style_params = edge_styles.get_style_params_from_props(props)
        success = geometry.apply_edge_style_to_graph(obj, style_params)
        
        if success:
            self.report({'INFO'}, f"Applied '{self.preset}' preset")
            obj.data.update()
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        else:
            self.report({'ERROR'}, f"Failed to apply preset")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_PreviewEdgeStyle(bpy.types.Operator):
    bl_idname = "scigraphs.preview_edge_style"
    bl_label = "Preview Style"
    bl_description = "Preview the edge style on the graph"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and "num_nodes" in obj
    
    def execute(self, context):
        # No partial preview yet, so this just applies the style outright.
        return bpy.ops.scigraphs.apply_edge_style()


def update_preset_callback(self, context):
    if self.edge_style_preset != 'CUSTOM':
        edge_styles.apply_preset(self, self.edge_style_preset)


def register():
    bpy.utils.register_class(SCIGRAPHS_OT_ApplyEdgeStyle)
    bpy.utils.register_class(SCIGRAPHS_OT_ResetEdgeStyle)
    bpy.utils.register_class(SCIGRAPHS_OT_ApplyEdgeStylePreset)
    bpy.utils.register_class(SCIGRAPHS_OT_PreviewEdgeStyle)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_OT_PreviewEdgeStyle)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ApplyEdgeStylePreset)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ResetEdgeStyle)
    bpy.utils.unregister_class(SCIGRAPHS_OT_ApplyEdgeStyle)
