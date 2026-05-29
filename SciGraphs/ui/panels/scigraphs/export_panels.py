# Export and utility panels

import bpy

class SCIGRAPHS_PT_export(bpy.types.Panel):
    """Main export and tools panel."""
    bl_label = "Export & Tools"
    bl_parent_id = "SCIGRAPHS_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        
        if not obj or "num_nodes" not in obj:
            box = layout.box()
            box.label(text="No graph loaded", icon='ERROR')
            box.label(text="Create a graph first in Data panel")
            return
        
        layout.label(text="Export and utility tools", icon='EXPORT')


class SCIGRAPHS_PT_export_options(bpy.types.Panel):
    """Export options for graph data."""
    bl_label = "Export Options"
    bl_parent_id = "SCIGRAPHS_PT_export"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Export settings
        box = layout.box()
        box.label(text="Export Graph Data", icon='EXPORT')
        col = box.column(align=True)
        col.prop(props, "export_filepath", text="Path")
        col.prop(props, "export_format", text="Format")
        col.prop(props, "export_include_attributes", text="Attributes")
        
        # Export buttons
        box.separator()
        row = box.row(align=True)
        row.scale_y = 1.3
        row.operator("scigraphs.export_graph", text="Export Graph", icon='EXPORT')
        row.operator("scigraphs.export_positions", text="Positions", icon='EMPTY_AXIS')
        
        layout.separator()
        box = layout.box()
        box.label(text="Quick Export (Format-Specific)", icon='FILE_BLANK')
        col = box.column(align=True)
        row = col.row(align=True)
        row.operator("scigraphs.export_gexf", text="GEXF", icon='FILE')
        row.operator("scigraphs.export_graphml", text="GraphML", icon='FILE')
        row = col.row(align=True)
        row.operator("scigraphs.export_pajek", text="Pajek", icon='FILE')
        row.operator("scigraphs.export_json", text="JSON", icon='FILE')


class SCIGRAPHS_PT_export_utilities(bpy.types.Panel):
    """Utility tools."""
    bl_label = "Utilities"
    bl_parent_id = "SCIGRAPHS_PT_export"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        # Statistics report
        box = layout.box()
        box.label(text="Statistics Report", icon='TEXT')
        box.prop(props, "report_include_powerlaw", text="Power Law Fit")
        
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.generate_statistics_report", text="Generate Report", icon='FILE_TEXT')


def register():
    bpy.utils.register_class(SCIGRAPHS_PT_export)
    bpy.utils.register_class(SCIGRAPHS_PT_export_options)
    bpy.utils.register_class(SCIGRAPHS_PT_export_utilities)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_PT_export_utilities)
    bpy.utils.unregister_class(SCIGRAPHS_PT_export_options)
    bpy.utils.unregister_class(SCIGRAPHS_PT_export)

