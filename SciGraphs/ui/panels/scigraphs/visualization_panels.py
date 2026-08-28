# Visualization and appearance panels

import bpy


class _MeshRenderPanel:
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'render'
    bl_options = {'DEFAULT_CLOSED'}
    COMPAT_ENGINES = {'BLENDER_EEVEE', 'CYCLES', 'BLENDER_WORKBENCH'}

    @classmethod
    def poll(cls, context):
        return context.engine in cls.COMPAT_ENGINES


def draw_text_style(layout, props, obj):
    """The label style, shared by both render paths.

    The SciGraphs engine draws its own overlay and reads these very
    properties, so the two panels have to agree. One function instead of
    two copies: the engine's panel pointed at the sidebar for style, and
    once that panel moved into the Render tab under EEVEE and Cycles,
    size, font and depth occlusion had no reachable control at all under
    the SciGraphs engine.

    Declutter is not here. Each path has its own, and the engine's also
    chooses which labels survive.
    """
    box = layout.box()
    box.label(text="Text Source", icon='TEXT')

    col = box.column(align=True)
    col.prop(props, "text_source", text="Source")

    if props.text_source == 'ATTRIBUTE':
        col.prop(props, "text_attribute", text="Attribute")

        col.separator()
        col.label(text="Number Format:")
        col.prop(props, "text_format_type", text="Type")

        if props.text_format_type in ('FLOAT', 'SCIENTIFIC', 'PERCENTAGE'):
            col.prop(props, "text_float_decimals", text="Decimals")

        col.prop(props, "text_thousands_separator", text="Thousands Separator")

        row = col.row(align=True)
        row.prop(props, "text_format_prefix", text="Prefix")
        row.prop(props, "text_format_suffix", text="Suffix")

    layout.separator()
    box = layout.box()
    box.label(text="Size Settings", icon='FIXED_SIZE')

    col = box.column(align=True)
    col.prop(props, "text_size_mode", text="Mode")

    if props.text_size_mode in ('FIXED', 'ADAPTIVE'):
        col.prop(props, "text_size_fixed", text="Base Size")

    if props.text_size_mode in ('PROPORTIONAL', 'ADAPTIVE'):
        col.prop(props, "text_size_scale", text="Scale Factor")

    layout.separator()
    box = layout.box()
    box.label(text="Visibility", icon='HIDE_OFF')

    col = box.column(align=True)
    col.prop(props, "text_max_distance", text="Max Distance")
    col.prop(props, "text_depth_occlusion", text="Depth Occlusion")

    layout.separator()
    box = layout.box()
    header_row = box.row()
    header_row.prop(props, "text_filter_enabled", text="")
    header_row.label(text="Attribute Filter", icon='FILTER')

    if props.text_filter_enabled:
        col = box.column(align=True)
        col.prop(props, "text_filter_attribute", text="Attribute")
        col.prop(props, "text_filter_operator", text="Operator")
        col.prop(props, "text_filter_value", text="Value")

    layout.separator()
    box = layout.box()
    box.label(text="Font", icon='OUTLINER_DATA_FONT')

    col = box.column(align=True)
    col.prop(props, "text_font_source", text="Source")

    if props.text_font_source == 'SYSTEM':
        col.prop(props, "text_font_system", text="Font")
    else:
        col.prop(props, "text_font_custom", text="")

    layout.separator()
    box = layout.box()
    box.label(text="Appearance", icon='COLOR')

    col = box.column(align=True)
    col.prop(props, "text_color", text="Text Color")

    col.separator()
    col.prop(props, "text_background_enabled", text="Show Background")

    if props.text_background_enabled:
        col.prop(props, "text_background_color", text="Background")
        col.prop(props, "text_background_alpha", text="Opacity", slider=True)


class SCIGRAPHS_PT_visualization_appearance(_MeshRenderPanel, bpy.types.Panel):
    """Node and edge appearance settings."""
    bl_label = "Appearance"
    bl_order = 0
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Nodes", icon='MESH_CIRCLE')
        col = box.column(align=True)
        col.prop(props, "node_size", text="Size", slider=True)
        col.prop(props, "node_resolution", text="Resolution")
        col.prop(props, "node_shape", text="Shape")
        col.prop(props, "node_scale_multiplier", text="Attr Mult")
        
        box = layout.box()
        box.label(text="Edges", icon='CURVE_PATH')
        col = box.column(align=True)
        col.prop(props, "edge_thickness", text="Thickness", slider=True)
        col.prop(props, "edge_resolution", text="Resolution")
        col.prop(props, "edge_thickness_multiplier", text="Attr Mult")

        box = layout.box()
        box.label(text="Direction Arrows", icon='FORWARD')
        col = box.column(align=True)
        col.prop(props, "show_arrows", text="Show Arrows")
        if props.show_arrows:
            col.prop(props, "arrow_size", text="Size")
            col.prop(props, "arrow_position", text="Position", slider=True)
        
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("scigraphs.update_appearance", text="Update Appearance", icon='SHADING_RENDERED')


class SCIGRAPHS_PT_visualization_edge_style(_MeshRenderPanel, bpy.types.Panel):
    """Edge style settings for curved, bundled, and styled edges."""
    bl_label = "Edge Style"
    bl_order = 1
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        box = layout.box()
        box.label(text="Style", icon='CURVE_DATA')
        
        col = box.column(align=True)
        col.prop(props, "edge_style_preset", text="Preset")
        col.separator()
        col.prop(props, "edge_style_type", text="Type")
        
        if obj and "edge_style_applied" in obj:
            row = box.row()
            row.alert = False
            row.label(text=f"Current: {obj['edge_style_applied']}", icon='CHECKMARK')
        
        if props.edge_style_type in ('CURVED', 'QUADRATIC', 'ARC', 'TAPERED'):
            layout.separator()
            box = layout.box()
            box.label(text="Curve Settings", icon='MOD_CURVE')
            
            col = box.column(align=True)
            col.prop(props, "edge_curvature", text="Curvature", slider=True)
            col.prop(props, "edge_segments", text="Segments")
            col.prop(props, "edge_curve_direction", text="Direction")
        
        if props.edge_style_type == 'ORTHOGONAL':
            layout.separator()
            box = layout.box()
            box.label(text="Orthogonal Settings", icon='SNAP_PERPENDICULAR')
            
            col = box.column(align=True)
            col.prop(props, "edge_orthogonal_style", text="Style")
            col.prop(props, "edge_segments", text="Segments")
        
        # These need something the mesh operator lacks: Hierarchical the cluster
        # tree, the rest a compute shader. Baking any of them would give straight
        # edges, so say so rather than silently do it.
        if props.edge_style_type in ('HIERARCHICAL', 'FDEB', 'SBEB', 'ROUTED',
                                    'MINGLE'):
            layout.separator()
            box = layout.box()
            box.label(text="Available in GPU Preview only", icon='INFO')
            box.label(text="Enable GPU Edge Styles to use this shape")

        if props.edge_style_type == 'BUNDLED':
            layout.separator()
            box = layout.box()
            box.label(text="Bundling Settings", icon='GP_MULTIFRAME_EDITING')
            
            col = box.column(align=True)
            col.prop(props, "edge_bundle_strength", text="Strength", slider=True)
            col.prop(props, "edge_bundle_iterations", text="Iterations")
            col.prop(props, "edge_bundle_compatibility_threshold", text="Compatibility", slider=True)
            col.prop(props, "edge_segments", text="Segments")
        
        if props.edge_style_type == 'TAPERED':
            layout.separator()
            box = layout.box()
            box.label(text="Taper Settings", icon='MOD_THICKNESS')
            
            col = box.column(align=True)
            col.prop(props, "edge_taper_start", text="Start Thickness")
            col.prop(props, "edge_taper_end", text="End Thickness")
        
        layout.separator()
        box = layout.box()
        box.label(text="Multi-Edge Handling", icon='MOD_ARRAY')
        
        col = box.column(align=True)
        col.prop(props, "edge_auto_offset_parallel", text="Auto-Offset Parallel Edges")
        
        if props.edge_auto_offset_parallel:
            col.prop(props, "edge_parallel_offset", text="Offset Distance")
        
        col.separator()
        col.prop(props, "edge_self_loop_radius", text="Self-Loop Radius")
        
        if obj and obj.get("is_osmnx", False):
            layout.separator()
            box = layout.box()
            box.label(text="OSMnx Options", icon='WORLD_DATA')
            col = box.column()
            col.prop(props, "edge_style_preserve_osmnx", text="Preserve Street Geometry")
        
        layout.separator()
        box = layout.box()
        box.label(text="Quick Presets", icon='PRESET')
        
        row = box.row(align=True)
        op = row.operator("scigraphs.apply_edge_style_preset", text="Gephi")
        op.preset = 'GEPHI_DEFAULT'
        op = row.operator("scigraphs.apply_edge_style_preset", text="Cytoscape")
        op.preset = 'CYTOSCAPE_BEZIER'
        op = row.operator("scigraphs.apply_edge_style_preset", text="Schematic")
        op.preset = 'SCHEMATIC'
        
        row = box.row(align=True)
        op = row.operator("scigraphs.apply_edge_style_preset", text="Bundled")
        op.preset = 'BUNDLED_DENSE'
        op = row.operator("scigraphs.apply_edge_style_preset", text="Flow")
        op.preset = 'FLOW_DIAGRAM'
        op = row.operator("scigraphs.apply_edge_style_preset", text="Minimal")
        op.preset = 'MINIMAL'
        
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("scigraphs.apply_edge_style", text="Apply Edge Style", icon='PLAY')
        row.operator("scigraphs.preview_edge_style", text="", icon='HIDE_OFF')
        row.operator("scigraphs.reset_edge_style", text="", icon='LOOP_BACK')
        
        if obj and "num_curve_verts" in obj:
            info_box = layout.box()
            info_box.scale_y = 0.7
            info_box.label(text=f"Curve vertices: {obj['num_curve_verts']}")


class SCIGRAPHS_PT_visualization_text(_MeshRenderPanel, bpy.types.Panel):
    """Text overlay settings for node labels."""
    bl_label = "Text Labels"
    bl_order = 2
    
    def draw_header(self, context):
        props = context.scene.scigraphs
        self.layout.prop(props, "text_overlay_enabled", text="")
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        if props.text_overlay_enabled:
            row = layout.row()
            row.alert = False
            row.label(text="Overlay active in compositor", icon='CHECKMARK')
        
        draw_text_style(layout, props, obj)
        layout.separator()
        row = layout.row()
        row.prop(props, "text_declutter", text="Declutter")
        
        layout.separator()
        box = layout.box()
        header_row = box.row()
        header_row.prop(props, "text_auto_update", text="")
        header_row.label(text="Auto Update", icon='PLAY')
        
        if props.text_auto_update:
            col = box.column(align=True)
            col.prop(props, "text_auto_update_interval", text="Interval (s)")
            
            if context.window_manager.get("scigraphs_auto_update_running", False):
                col.operator("scigraphs.stop_auto_update", text="Stop Auto Update", icon='PAUSE')
            else:
                col.operator("scigraphs.start_auto_update", text="Start Auto Update", icon='PLAY')
        
        layout.separator()
        row = layout.row(align=True)
        row.scale_y = 1.5
        
        if props.text_overlay_enabled:
            row.operator("scigraphs.generate_text_overlay", text="Update Overlay", icon='FILE_REFRESH')
            row.operator("scigraphs.remove_text_overlay", text="", icon='X')
        else:
            row.operator("scigraphs.generate_text_overlay", text="Generate Text Overlay", icon='OUTLINER_DATA_FONT')
        
        info_box = layout.box()
        info_box.scale_y = 0.7
        info_box.label(text="Uses scene camera and render resolution")
        info_box.label(text="Image is composited over final render")


def register():
    bpy.utils.register_class(SCIGRAPHS_PT_visualization_appearance)
    bpy.utils.register_class(SCIGRAPHS_PT_visualization_edge_style)
    bpy.utils.register_class(SCIGRAPHS_PT_visualization_text)


def unregister():
    bpy.utils.unregister_class(SCIGRAPHS_PT_visualization_text)
    bpy.utils.unregister_class(SCIGRAPHS_PT_visualization_edge_style)
    bpy.utils.unregister_class(SCIGRAPHS_PT_visualization_appearance)

