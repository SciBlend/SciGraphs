# SciGraphs Viewport Menus
# Quick actions for the active graph - no complex inputs needed.

import bpy


class SCIGRAPHS_MT_viewport_layout(bpy.types.Menu):
    """Apply layout algorithms to active graph."""
    bl_idname = "SCIGRAPHS_MT_viewport_layout"
    bl_label = "Layout"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.apply_layout", icon='PLAY')
        layout.operator("scigraphs.reset_layout", icon='FILE_REFRESH')
        layout.separator()
        layout.operator("scigraphs.execute_layout_step", icon='FRAME_NEXT')
        layout.operator("scigraphs.bake_animation", icon='RENDER_ANIMATION')


class SCIGRAPHS_MT_viewport_visualization(bpy.types.Menu):
    """Visual settings for active graph."""
    bl_idname = "SCIGRAPHS_MT_viewport_visualization"
    bl_label = "Visualization"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.setup_visualization", icon='GEOMETRY_NODES')
        layout.operator("scigraphs.update_appearance", icon='MATERIAL')
        layout.separator()
        layout.operator("scigraphs.apply_edge_style_preset", icon='PRESET')
        layout.operator("scigraphs.reset_edge_style", icon='LOOP_BACK')
        layout.separator()
        layout.operator("scigraphs.apply_rendering_preset", icon='SCENE')
        layout.operator("scigraphs.setup_lighting", icon='LIGHT_SUN')


class SCIGRAPHS_MT_viewport_analysis(bpy.types.Menu):
    """Graph analysis on active graph."""
    bl_idname = "SCIGRAPHS_MT_viewport_analysis"
    bl_label = "Analysis"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.calculate_centrality", icon='LIGHT_POINT')
        layout.operator("scigraphs.calculate_clustering", icon='STICKY_UVS_LOC')
        layout.operator("scigraphs.apply_clustering", icon='GROUP_VERTEX')
        layout.separator()
        layout.operator("scigraphs.find_sccs", icon='LINKED')
        layout.operator("scigraphs.detect_patterns", icon='SHADERFX')
        layout.separator()
        layout.operator("scigraphs.compute_mst", icon='OUTLINER_OB_ARMATURE')


class SCIGRAPHS_MT_viewport_topology(bpy.types.Menu):
    """Topological analysis and visualization."""
    bl_idname = "SCIGRAPHS_MT_viewport_topology"
    bl_label = "Topology"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.check_planarity", icon='MESH_PLANE')
        layout.operator("scigraphs.calculate_genus", icon='SURFACE_NSPHERE')
        layout.operator("scigraphs.compute_faces", icon='FACESEL')
        layout.operator("scigraphs.validate_crossings", icon='PIVOT_CURSOR')
        layout.separator()
        layout.operator("scigraphs.visualize_surface", icon='SURFACE_DATA')
        layout.operator("scigraphs.toggle_topo_surface", icon='HIDE_OFF')
        layout.operator("scigraphs.remove_topo_surface", icon='X')
        layout.separator()
        layout.operator("scigraphs.create_dual_graph", icon='MOD_WIREFRAME')
        layout.operator("scigraphs.toggle_dual_graph", icon='HIDE_OFF')
        layout.operator("scigraphs.remove_dual_graph", icon='X')


class SCIGRAPHS_MT_viewport_temporal(bpy.types.Menu):
    """Temporal graph playback controls."""
    bl_idname = "SCIGRAPHS_MT_viewport_temporal"
    bl_label = "Temporal"

    def draw(self, context):
        layout = self.layout
        row = layout.row(align=True)
        row.operator("scigraphs.temporal_first", text="", icon='REW')
        row.operator("scigraphs.temporal_previous", text="", icon='PREV_KEYFRAME')
        row.operator("scigraphs.temporal_play", text="", icon='PLAY')
        row.operator("scigraphs.temporal_next", text="", icon='NEXT_KEYFRAME')
        row.operator("scigraphs.temporal_last", text="", icon='FF')
        layout.separator()
        layout.operator("scigraphs.temporal_refresh", icon='FILE_REFRESH')
        layout.operator("scigraphs.temporal_clear", icon='X')


class SCIGRAPHS_MT_viewport_osmnx(bpy.types.Menu):
    """OSMnx operations on active network."""
    bl_idname = "SCIGRAPHS_MT_viewport_osmnx"
    bl_label = "OSMnx"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.osmnx_centrality", icon='LIGHT_POINT')
        layout.operator("scigraphs.osmnx_attr_to_colors", icon='BRUSH_DATA')
        layout.separator()
        layout.operator("scigraphs.osmnx_add_edge_lengths", icon='DRIVER_DISTANCE')
        layout.operator("scigraphs.osmnx_add_edge_bearings", icon='ORIENTATION_GIMBAL')
        layout.operator("scigraphs.osmnx_add_edge_speeds", icon='DRIVER')
        layout.operator("scigraphs.osmnx_add_travel_times", icon='TIME')
        layout.separator()
        layout.operator("scigraphs.osmnx_simplify", icon='MOD_DECIM')
        layout.operator("scigraphs.osmnx_consolidate", icon='STICKY_UVS_VERT')
        layout.operator("scigraphs.osmnx_largest_component", icon='OUTLINER_OB_POINTCLOUD')
        layout.separator()
        layout.operator("scigraphs.osmnx_to_undirected", icon='ARROW_LEFTRIGHT')
        layout.operator("scigraphs.osmnx_to_digraph", icon='FORWARD')
        layout.separator()
        layout.operator("scigraphs.osmnx_basic_stats", icon='INFO')
        layout.operator("scigraphs.osmnx_orientation_entropy", icon='FORCE_VORTEX')
        layout.operator("scigraphs.osmnx_circuity", icon='CON_FOLLOWPATH')


class SCIGRAPHS_MT_viewport_osmnx_elevation(bpy.types.Menu):
    """Elevation operations on active network."""
    bl_idname = "SCIGRAPHS_MT_viewport_osmnx_elevation"
    bl_label = "Elevation"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.osmnx_apply_elevation_3d", icon='EMPTY_SINGLE_ARROW')
        layout.operator("scigraphs.osmnx_flatten_network", icon='MESH_PLANE')
        layout.separator()
        layout.operator("scigraphs.osmnx_toggle_terrain", icon='HIDE_OFF')
        layout.operator("scigraphs.osmnx_remove_terrain", icon='X')


class SCIGRAPHS_MT_viewport_export(bpy.types.Menu):
    """Export active graph."""
    bl_idname = "SCIGRAPHS_MT_viewport_export"
    bl_label = "Export"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.export_gexf", icon='FILE')
        layout.operator("scigraphs.export_graphml", icon='FILE')
        layout.operator("scigraphs.export_pajek", icon='FILE')
        layout.operator("scigraphs.export_json", icon='FILE')
        layout.separator()
        layout.operator("scigraphs.export_positions", icon='EMPTY_AXIS')
        layout.separator()
        layout.operator("scigraphs.calculate_global_statistics", icon='INFO')
        layout.operator("scigraphs.generate_statistics_report", icon='TEXT')


class SCIGRAPHS_MT_viewport_repro(bpy.types.Menu):
    """Reproducible pipeline operations."""
    bl_idname = "SCIGRAPHS_MT_viewport_repro"
    bl_label = "Reproducibility"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.run_pipeline", icon='PLAY')
        layout.operator("scigraphs.validate_pipeline", icon='CHECKMARK')
        layout.separator()
        layout.operator("scigraphs.export_pipeline_template", icon='FILE_NEW')
        layout.operator("scigraphs.export_current_repro_spec", icon='EXPORT')
        layout.separator()
        layout.operator("scigraphs.open_artifacts_folder", icon='FOLDER_REDIRECT')


class SCIGRAPHS_MT_viewport_tools(bpy.types.Menu):
    """Interactive graph tools."""
    bl_idname = "SCIGRAPHS_MT_viewport_tools"
    bl_label = "Tools"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.select_node_tool", icon='VERTEXSEL')
        layout.operator("scigraphs.move_node_tool", icon='OBJECT_ORIGIN')
        layout.operator("scigraphs.path_tool", icon='TRACKING')
        layout.operator("scigraphs.lasso_select_tool", icon='STROKE')
        layout.separator()
        layout.operator("scigraphs.highlight_communities", icon='GROUP_VERTEX')
        layout.operator("scigraphs.visualize_centrality_interactive", icon='LIGHT_POINT')
        layout.operator("scigraphs.preview_layout", icon='PLAY')
        layout.separator()
        layout.operator("scigraphs.toggle_hud", icon='INFO')
        layout.operator("scigraphs.toggle_node_gizmos", icon='GIZMO')


class SCIGRAPHS_MT_viewport_pie(bpy.types.Menu):
    """Pie menu shortcuts."""
    bl_idname = "SCIGRAPHS_MT_viewport_pie"
    bl_label = "Pie Menus"

    def draw(self, context):
        layout = self.layout
        layout.operator("scigraphs.pie_main", text="Main (Shift+Q)", icon='MESH_CIRCLE')
        layout.operator("scigraphs.pie_layout", text="Layout (Ctrl+Shift+L)", icon='NODETREE')
        layout.operator("scigraphs.pie_analysis", text="Analysis (Ctrl+Shift+A)", icon='OUTLINER_DATA_GP_LAYER')
        layout.operator("scigraphs.pie_visualization", text="Visualization (Ctrl+Shift+V)", icon='SHADING_RENDERED')
        layout.operator("scigraphs.pie_topology", text="Topology (Ctrl+Shift+T)", icon='SURFACE_DATA')


class SCIGRAPHS_MT_viewport_toolbars(bpy.types.Menu):
    """Floating toolbar visibility and profiles."""
    bl_idname = "SCIGRAPHS_MT_viewport_toolbars"
    bl_label = "Floating Toolbars"

    def draw(self, context):
        layout = self.layout
        wm = context.window_manager
        is_visible = getattr(wm, "scigraphs_show_quick_toolbar", True)

        icon = 'HIDE_OFF' if not is_visible else 'HIDE_ON'
        text = "Show Floating Toolbar" if not is_visible else "Hide Floating Toolbar"
        layout.operator("scigraphs.toggle_quick_toolbar", text=text, icon=icon)

        layout.separator()
        op = layout.operator("scigraphs.set_quick_toolbar", text="SciGraphs Toolbar", icon='GRAPH')
        op.toolbar = 'SCIGRAPHS'
        op = layout.operator("scigraphs.set_quick_toolbar", text="OSMnx Toolbar", icon='WORLD')
        op.toolbar = 'OSMNX'
        op = layout.operator("scigraphs.set_quick_toolbar", text="City2Graph Toolbar", icon='OUTLINER_OB_CURVE')
        op.toolbar = 'CITY2GRAPH'

        layout.separator()
        is_color_visible = getattr(wm, "scigraphs_show_color_toolbar", True)
        color_icon = 'HIDE_OFF' if not is_color_visible else 'HIDE_ON'
        color_text = (
            "Show Color Toolbar" if not is_color_visible else "Hide Color Toolbar"
        )
        layout.operator(
            "scigraphs.color_toggle_toolbar",
            text=color_text,
            icon=color_icon,
        )
        layout.operator(
            "scigraphs.color_settings_dialog",
            text="Color Settings...",
            icon='COLOR',
        )


class SCIGRAPHS_MT_viewport(bpy.types.Menu):
    """Main SciGraphs menu."""
    bl_idname = "SCIGRAPHS_MT_viewport"
    bl_label = "SciGraphs"

    def draw(self, context):
        layout = self.layout
        layout.menu("SCIGRAPHS_MT_viewport_toolbars", icon='GIZMO')
        layout.separator()
        layout.menu("SCIGRAPHS_MT_viewport_tools", icon='TOOL_SETTINGS')
        layout.menu("SCIGRAPHS_MT_viewport_pie", icon='MESH_CIRCLE')
        layout.separator()
        layout.menu("SCIGRAPHS_MT_viewport_layout", icon='NODETREE')
        layout.menu("SCIGRAPHS_MT_viewport_visualization", icon='SHADING_RENDERED')
        layout.menu("SCIGRAPHS_MT_viewport_analysis", icon='OUTLINER_DATA_GP_LAYER')
        layout.menu("SCIGRAPHS_MT_viewport_topology", icon='SURFACE_DATA')
        layout.menu("SCIGRAPHS_MT_viewport_temporal", icon='TIME')
        layout.separator()
        layout.menu("SCIGRAPHS_MT_viewport_osmnx", icon='WORLD')
        layout.menu("SCIGRAPHS_MT_viewport_osmnx_elevation", icon='EMPTY_SINGLE_ARROW')
        layout.separator()
        layout.menu("SCIGRAPHS_MT_viewport_export", icon='EXPORT')
        layout.menu("SCIGRAPHS_MT_viewport_repro", icon='FILE_CACHE')


def _draw_scigraphs_menu(self, context):
    """Draw function appended to VIEW3D_HT_header."""
    self.layout.menu("SCIGRAPHS_MT_viewport")


_MENU_CLASSES = [
    SCIGRAPHS_MT_viewport_layout,
    SCIGRAPHS_MT_viewport_visualization,
    SCIGRAPHS_MT_viewport_analysis,
    SCIGRAPHS_MT_viewport_topology,
    SCIGRAPHS_MT_viewport_temporal,
    SCIGRAPHS_MT_viewport_osmnx,
    SCIGRAPHS_MT_viewport_osmnx_elevation,
    SCIGRAPHS_MT_viewport_export,
    SCIGRAPHS_MT_viewport_repro,
    SCIGRAPHS_MT_viewport_tools,
    SCIGRAPHS_MT_viewport_pie,
    SCIGRAPHS_MT_viewport_toolbars,
    SCIGRAPHS_MT_viewport,
]


def register():
    for cls in _MENU_CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.VIEW3D_HT_header.append(_draw_scigraphs_menu)


def unregister():
    bpy.types.VIEW3D_HT_header.remove(_draw_scigraphs_menu)
    for cls in reversed(_MENU_CLASSES):
        bpy.utils.unregister_class(cls)
