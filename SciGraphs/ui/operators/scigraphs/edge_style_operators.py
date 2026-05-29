"""
Edge Style Operators for SciGraphs

Operators for applying and managing edge styles in graph visualizations.
"""

import bpy
from bpy.props import EnumProperty, BoolProperty

from ....core import geometry
from ....core import edge_styles
from ....utils.logger import log


class SCIGRAPHS_OT_ApplyEdgeStyle(bpy.types.Operator):
    """Apply the selected edge style to the graph."""
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
        
        # Get style parameters from properties
        style_params = edge_styles.get_style_params_from_props(props)
        
        # Apply the style
        success = geometry.apply_edge_style_to_graph(obj, style_params)
        
        if success:
            self.report({'INFO'}, f"Applied '{props.edge_style_type}' edge style")
            
            # Refresh visualization if active
            mod = obj.modifiers.get("SciGraphs_Viz")
            if mod:
                # Force viewport update
                obj.data.update()
                for area in context.screen.areas:
                    if area.type == 'VIEW_3D':
                        area.tag_redraw()
        else:
            self.report({'ERROR'}, "Failed to apply edge style")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ResetEdgeStyle(bpy.types.Operator):
    """Reset edges to straight lines."""
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
            
            # Update properties to match
            props = context.scene.scigraphs
            props.edge_style_type = 'STRAIGHT'
            props.edge_style_preset = 'MINIMAL'
            
            # Refresh viewport
            obj.data.update()
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()
        else:
            self.report({'ERROR'}, "Failed to reset edge style")
            return {'CANCELLED'}
        
        return {'FINISHED'}


class SCIGRAPHS_OT_ApplyEdgeStylePreset(bpy.types.Operator):
    """Apply a predefined edge style preset."""
    bl_idname = "scigraphs.apply_edge_style_preset"
    bl_label = "Apply Edge Preset"
    bl_description = "Apply a predefined edge style configuration"
    bl_options = {'REGISTER', 'UNDO'}
    
    preset: EnumProperty(
        name="Preset",
        items=[
            ('GEPHI_DEFAULT', "Gephi Default", ""),
            ('CYTOSCAPE_BEZIER', "Cytoscape Bezier", ""),
            ('SCHEMATIC', "Schematic", ""),
            ('BUNDLED_DENSE', "Bundled (Dense)", ""),
            ('FLOW_DIAGRAM', "Flow Diagram", ""),
            ('MINIMAL', "Minimal", ""),
        ],
        default='GEPHI_DEFAULT',
    )
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and "num_nodes" in obj
    
    def invoke(self, context, event):
        props = context.scene.scigraphs
        valid_presets = {item.identifier for item in self.bl_rna.properties["preset"].enum_items}
        self.preset = props.edge_style_preset if props.edge_style_preset in valid_presets else 'GEPHI_DEFAULT'
        return context.window_manager.invoke_props_dialog(self)
    
    def execute(self, context):
        props = context.scene.scigraphs
        
        # Apply preset to properties
        edge_styles.apply_preset(props, self.preset)
        
        # Update the preset selector
        props.edge_style_preset = self.preset
        
        # Apply to graph
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
    """Preview edge style without fully applying (shows on selected edges only)."""
    bl_idname = "scigraphs.preview_edge_style"
    bl_label = "Preview Style"
    bl_description = "Preview the edge style on the graph"
    bl_options = {'REGISTER'}
    
    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and "num_nodes" in obj
    
    def execute(self, context):
        # For now, preview is the same as apply
        # In future could implement partial/preview mode
        return bpy.ops.scigraphs.apply_edge_style()


def update_preset_callback(self, context):
    """Callback when preset selection changes."""
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
