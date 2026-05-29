# SciGraphs viewport node handles
# Draws safe, non-RNA gizmo-like handles for graph nodes.

import bpy
import numpy as np
import gpu
import blf
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector


_HANDLE = None
_GIZMOS_ENABLED = False
_TOOLTIP_HANDLE = None
_TOOLTIP_STATE = None

_TOOLBAR_SCIGRAPHS = 'SCIGRAPHS'
_TOOLBAR_OSMNX = 'OSMNX'
_TOOLBAR_CITY2GRAPH = 'CITY2GRAPH'

_ACTION_DEFINITIONS = (
    ("panel_data", "Data", "DATA", "Open Data import and graph creation tools", 'IMPORT'),
    ("panel_layout", "Layout", "LAYOUT", "Open layout and positioning tools", 'NODETREE'),
    ("panel_algorithms", "Algorithms", "ALGORITHMS", "Open graph algorithm tools", 'SCRIPT'),
    ("panel_analysis", "Analysis", "ANALYSIS", "Open graph analysis tools", 'VIEWZOOM'),
    ("panel_visualization", "Visualization", "VISUAL", "Open visualization tools", 'GEOMETRY_NODES'),
    ("panel_export", "Export", "TOOLS", "Open export and utility tools", 'EXPORT'),
    ("osmnx_import", "Import Network", "OSMNX", "Import a street network from OSM by place, point, address, bbox, polygon, or XML", 'WORLD'),
    ("osmnx_project", "Project CRS", "OSMNX", "Project the network to a local CRS", 'ORIENTATION_GLOBAL'),
    ("osmnx_simplify", "Simplify", "OSMNX", "Simplify network topology", 'MOD_DECIM'),
    ("osmnx_consolidate", "Consolidate", "OSMNX", "Consolidate nearby intersections", 'AUTOMERGE_ON'),
    ("osmnx_speeds", "Speeds", "ROUTING", "Infer edge speeds", 'DRIVER'),
    ("osmnx_travel_times", "Travel Times", "ROUTING", "Infer edge travel times", 'TIME'),
    ("osmnx_shortest_path", "Shortest Path", "ROUTING", "Calculate an OSMnx shortest path", 'TRACKING'),
    ("osmnx_k_routes", "K Routes", "ROUTING", "Compute route alternatives", 'MOD_ARRAY'),
    ("osmnx_batch_routes", "Batch Routes", "ROUTING", "Compute many random origin-destination routes", 'STICKY_UVS_LOC'),
    ("osmnx_route_summary", "Route Summary", "ROUTING", "Summarize the active OSMnx route", 'INFO'),
    ("osmnx_isochrones", "Isochrones", "ACCESS", "Generate OSMnx isochrones", 'MESH_CIRCLE'),
    ("osmnx_stats", "Basic Stats", "ANALYSIS", "Compute OSMnx network statistics and density indicators", 'INFO'),
    ("osmnx_centrality", "Centrality", "ANALYSIS", "Compute OSMnx centrality", 'LIGHT_POINT'),
    ("osmnx_attr_colors", "Attribute Colors", "VISUAL", "Apply an OSMnx attribute to colors", 'COLOR'),
    ("osmnx_orientations", "Orientation Tools", "ANALYSIS", "Analyze street bearings, entropy, and orientation rose", 'ORIENTATION_GIMBAL'),
    ("osmnx_elevation_api", "Elevation API", "TERRAIN", "Add node elevations from API", 'URL'),
    ("osmnx_grades", "Grades", "TERRAIN", "Calculate edge grades", 'EMPTY_SINGLE_ARROW'),
    ("osmnx_features", "Features", "DATA", "Download OSM features for the current settings", 'POINTCLOUD_DATA'),
    ("osmnx_cache", "Cache", "DATA", "Manage cached OSMnx graphs", 'FILE_CACHE'),
    ("osmnx_export", "Export OSMnx", "DATA", "Export the active OSMnx network", 'EXPORT'),
    ("c2g_boundary", "Boundary", "CITY2GRAPH", "Geocode an urban boundary", 'VIEWZOOM'),
    ("c2g_overture", "Overture Data", "CITY2GRAPH", "Download Overture Maps data from a place name or bounding box", 'WORLD'),
    ("c2g_file", "Load File", "CITY2GRAPH", "Load City2Graph data from file", 'FILE_FOLDER'),
    ("c2g_process_segments", "Process Segments", "CITY2GRAPH", "Clean Overture street segments", 'MOD_EDGESPLIT'),
    ("c2g_segments_graph", "Segments Graph", "CITY2GRAPH", "Convert segments to graph", 'OUTLINER_OB_CURVE'),
    ("c2g_tessellation", "Tessellation", "MORPHOLOGY", "Generate urban tessellation", 'MOD_TRIANGULATE'),
    ("c2g_morphology", "Morphology", "MORPHOLOGY", "Generate morphological graph", 'FORCE_MAGNETIC'),
    ("c2g_proximity", "Proximity", "PROXIMITY", "Generate a proximity graph", 'PIVOT_INDIVIDUAL'),
    ("c2g_multilayer", "Multilayer Graph", "PROXIMITY", "Bridge multiple feature layers into a heterogeneous graph", 'LINKED'),
    ("c2g_group_nodes", "Group Nodes", "PROXIMITY", "Group points by polygons", 'GROUP_VERTEX'),
    ("c2g_gtfs", "GTFS", "TRANSPORT", "Import GTFS transit data", 'TIME'),
    ("c2g_travel", "Travel Graph", "TRANSPORT", "Create a travel summary graph", 'OUTLINER_OB_CURVE'),
    ("c2g_od_matrix", "OD Matrix", "MOBILITY", "Load an origin-destination matrix", 'SPREADSHEET'),
    ("c2g_od_graph", "OD Graph", "MOBILITY", "Convert OD matrix to graph", 'FORCE_FORCE'),
    ("c2g_dual", "Dual Graph", "METAPATH", "Create street dual graph", 'MOD_WIREFRAME'),
    ("c2g_bridge_amenities", "Bridge Amenities", "METAPATH", "Bridge amenities to streets", 'STICKY_UVS_LOC'),
    ("c2g_metapath_wizard", "Metapath Wizard", "METAPATH", "Run the guided metapath analysis workflow", 'CURVE_PATH'),
    ("c2g_weighted_metapaths", "Weighted Paths", "METAPATH", "Compute weighted metapaths", 'DRIVER_DISTANCE'),
    ("c2g_mesh", "Path Mesh", "METAPATH", "Convert metapaths to mesh", 'MESH_DATA'),
    ("c2g_graph_tools", "Graph Tools", "TOOLS", "Filter, clip, clean, or create isochrones from City2Graph graphs", 'MODIFIER'),
    ("c2g_export", "Export C2G", "DATA", "Export the active City2Graph graph", 'EXPORT'),
)

_TOOLBAR_PROFILES = {
    _TOOLBAR_SCIGRAPHS: (
        "panel_data",
        "panel_layout",
        "panel_algorithms",
        "panel_analysis",
        "panel_visualization",
        "panel_export",
    ),
    _TOOLBAR_OSMNX: (
        "osmnx_import",
        "osmnx_project",
        "osmnx_simplify",
        "osmnx_consolidate",
        "osmnx_speeds",
        "osmnx_travel_times",
        "osmnx_shortest_path",
        "osmnx_k_routes",
        "osmnx_batch_routes",
        "osmnx_route_summary",
        "osmnx_isochrones",
        "osmnx_stats",
        "osmnx_centrality",
        "osmnx_attr_colors",
        "osmnx_orientations",
        "osmnx_elevation_api",
        "osmnx_grades",
        "osmnx_features",
        "osmnx_cache",
        "osmnx_export",
    ),
    _TOOLBAR_CITY2GRAPH: (
        "c2g_boundary",
        "c2g_overture",
        "c2g_file",
        "c2g_process_segments",
        "c2g_segments_graph",
        "c2g_tessellation",
        "c2g_morphology",
        "c2g_proximity",
        "c2g_multilayer",
        "c2g_group_nodes",
        "c2g_gtfs",
        "c2g_travel",
        "c2g_od_matrix",
        "c2g_od_graph",
        "c2g_dual",
        "c2g_bridge_amenities",
        "c2g_metapath_wizard",
        "c2g_weighted_metapaths",
        "c2g_mesh",
        "c2g_graph_tools",
        "c2g_export",
    ),
}

_ACTION_INFO = {
    action: {
        "label": label,
        "category": category,
        "description": description,
    }
    for action, label, category, description, _icon in _ACTION_DEFINITIONS
}


def set_active_toolbar(context, toolbar):
    """Select the toolbar profile to show when the quick toolbar is visible."""
    wm = context.window_manager
    if toolbar not in _TOOLBAR_PROFILES:
        toolbar = _TOOLBAR_SCIGRAPHS

    if hasattr(wm, "scigraphs_active_quick_toolbar") and wm.scigraphs_active_quick_toolbar != toolbar:
        wm.scigraphs_active_quick_toolbar = toolbar
        _tag_3d_views(context)


def _active_toolbar(context):
    toolbar = getattr(context.window_manager, "scigraphs_active_quick_toolbar", _TOOLBAR_SCIGRAPHS)
    if toolbar not in _TOOLBAR_PROFILES:
        return _TOOLBAR_SCIGRAPHS
    return toolbar


def _current_actions(context):
    return _TOOLBAR_PROFILES[_active_toolbar(context)]


def _tag_3d_views(context):
    screen = getattr(context, "screen", None)
    if not screen:
        return

    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.tag_redraw()


def _draw_rect(x, y, width, height, color):
    vertices = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ]
    indices = [(0, 1, 2), (0, 2, 3)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _draw_tooltip_text(x, y, text, size, color):
    font_id = 0
    blf.size(font_id, size)
    blf.color(font_id, *color)
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text)


def _wrap_tooltip_text(text, max_chars=54):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def _set_toolbar_tooltip(context, title, description, category, center_x, toolbar_y):
    global _TOOLTIP_STATE

    region = getattr(context, "region", None)
    if region is None:
        _TOOLTIP_STATE = None
        return

    description_lines = _wrap_tooltip_text(description)
    line_height = 17
    width = 360
    height = 44 + len(description_lines) * line_height
    x = int(center_x - width * 0.5)
    y = int(toolbar_y + 34)

    x = max(12, min(x, region.width - width - 12))
    if y + height > region.height - 12:
        y = max(12, int(toolbar_y - height - 28))

    _TOOLTIP_STATE = {
        "title": title,
        "description_lines": description_lines,
        "category": category,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def _clear_toolbar_tooltip():
    global _TOOLTIP_STATE
    _TOOLTIP_STATE = None


def _draw_toolbar_tooltip():
    context = bpy.context
    if context.area is None or context.area.type != 'VIEW_3D':
        return
    if not getattr(context.window_manager, "scigraphs_show_quick_toolbar", True):
        return
    if not _TOOLTIP_STATE:
        return

    state = _TOOLTIP_STATE
    x = state["x"]
    y = state["y"]
    width = state["width"]
    height = state["height"]

    _draw_rect(x, y, width, height, (0.035, 0.045, 0.06, 0.90))
    _draw_rect(x, y + height - 26, width, 26, (0.08, 0.16, 0.22, 0.94))

    category = state["category"]
    title = state["title"]
    _draw_tooltip_text(x + 12, y + height - 19, f"{title}  ·  {category}", 13, (0.58, 0.86, 1.0, 1.0))

    text_y = y + height - 45
    for line in state["description_lines"]:
        _draw_tooltip_text(x + 12, text_y, line, 12, (0.92, 0.92, 0.92, 0.95))
        text_y -= 17


def enable_toolbar_tooltips():
    global _TOOLTIP_HANDLE
    if _TOOLTIP_HANDLE is None:
        _TOOLTIP_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_toolbar_tooltip, (), 'WINDOW', 'POST_PIXEL'
        )


def disable_toolbar_tooltips():
    global _TOOLTIP_HANDLE
    if _TOOLTIP_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_TOOLTIP_HANDLE, 'WINDOW')
        _TOOLTIP_HANDLE = None
    _clear_toolbar_tooltip()


def _toolbar_origin(context, action_count):
    wm = context.window_manager
    region = context.region
    spacing = 30 if action_count > 16 else 34
    width = action_count * spacing

    x = wm.scigraphs_quick_toolbar_x
    y = wm.scigraphs_quick_toolbar_y

    if x < 0:
        x = max(42, (region.width - width) * 0.5)
    if y < 0:
        y = 74

    return x, y, spacing


def _active_graph_positions(context):
    obj = context.active_object
    if not obj or "num_nodes" not in obj:
        return None, None

    pos_flat = obj.get("node_positions", [])
    if not pos_flat:
        return obj, None

    try:
        positions = np.array(pos_flat, dtype=float).reshape(-1, 3)
    except ValueError:
        return obj, None

    world_positions = [(obj.matrix_world @ Vector(pos))[:] for pos in positions]
    return obj, world_positions


def _draw_node_handles():
    if not _GIZMOS_ENABLED:
        return

    context = bpy.context
    if context.area is None or context.area.type != 'VIEW_3D':
        return

    _obj, positions = _active_graph_positions(context)
    if not positions:
        return

    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')

    batch = batch_for_shader(shader, 'POINTS', {"pos": positions})
    gpu.state.point_size_set(8.0)
    shader.bind()
    shader.uniform_float("color", (0.2, 0.75, 1.0, 0.8))
    batch.draw(shader)

    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def enable_gizmos():
    global _HANDLE, _GIZMOS_ENABLED
    _GIZMOS_ENABLED = True
    if _HANDLE is None:
        _HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_node_handles, (), 'WINDOW', 'POST_VIEW'
        )


def disable_gizmos():
    global _HANDLE, _GIZMOS_ENABLED
    _GIZMOS_ENABLED = False
    if _HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_HANDLE, 'WINDOW')
        _HANDLE = None


class SCIGRAPHS_OT_enable_node_gizmos(bpy.types.Operator):
    """Show/hide viewport node handles."""
    bl_idname = "scigraphs.toggle_node_gizmos"
    bl_label = "Toggle Node Handles"
    bl_description = "Show/hide visual handles for graph nodes"

    def execute(self, context):
        if _GIZMOS_ENABLED:
            disable_gizmos()
            status = "disabled"
        else:
            enable_gizmos()
            status = "enabled"

        for area in context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

        self.report({'INFO'}, f"Node handles {status}. Use Move Node to drag nodes.")
        return {'FINISHED'}


class SCIGRAPHS_OT_toggle_quick_toolbar(bpy.types.Operator):
    """Show/hide the floating SciGraphs toolbar."""
    bl_idname = "scigraphs.toggle_quick_toolbar"
    bl_label = "Toggle SciGraphs Quick Toolbar"
    bl_description = "Show or hide the floating SciGraphs quick toolbar"

    def execute(self, context):
        wm = context.window_manager
        wm.scigraphs_show_quick_toolbar = not wm.scigraphs_show_quick_toolbar
        _tag_3d_views(context)
        status = "shown" if wm.scigraphs_show_quick_toolbar else "hidden"
        self.report({'INFO'}, f"SciGraphs quick toolbar {status}")
        return {'FINISHED'}


class SCIGRAPHS_OT_set_quick_toolbar(bpy.types.Operator):
    """Show and switch the SciGraphs floating toolbar profile."""
    bl_idname = "scigraphs.set_quick_toolbar"
    bl_label = "Set SciGraphs Quick Toolbar"
    bl_description = "Show the floating toolbar and switch to a specific profile"

    toolbar: bpy.props.EnumProperty(
        name="Toolbar",
        items=[
            (_TOOLBAR_SCIGRAPHS, "SciGraphs", "Show graph analysis and visualization tools"),
            (_TOOLBAR_OSMNX, "OSMnx", "Show OSMnx network tools"),
            (_TOOLBAR_CITY2GRAPH, "City2Graph", "Show City2Graph urban analytics tools"),
        ],
        default=_TOOLBAR_SCIGRAPHS,
        options={'SKIP_SAVE'},
    )

    def execute(self, context):
        context.window_manager.scigraphs_show_quick_toolbar = True
        set_active_toolbar(context, self.toolbar)
        self.report({'INFO'}, f"{self.toolbar} quick toolbar shown")
        return {'FINISHED'}


class SCIGRAPHS_OT_quick_import_dialog(bpy.types.Operator):
    """Floating import dialog for SciGraphs graph data."""
    bl_idname = "scigraphs.quick_import_dialog"
    bl_label = "SciGraphs Quick Import"
    bl_description = "Configure and start a SciGraphs data import"

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="What do you want to import?", icon='IMPORT')
        box.prop(props, "data_source", text="Source")

        if props.data_source == 'FILE':
            box.prop(props, "filepath", text="File")
            box.prop(props, "csv_delimiter", text="Delimiter")
            row = box.row(align=True)
            row.operator("scigraphs.load_columns", text="Load Columns", icon='IMPORT')
            row.operator("scigraphs.create_graph", text="Create Graph", icon='ADD')

        elif props.data_source == 'DATABASE':
            box.prop(props, "db_profile_index", text="Connection")
            box.prop(props, "sql_query", text="SQL")
            row = box.row(align=True)
            row.operator("scigraphs.load_sql_columns", text="Load Columns", icon='IMPORT')
            row.operator("scigraphs.preview_sql_query", text="Preview", icon='VIEWZOOM')
            box.operator("scigraphs.create_graph_from_sql", text="Create Graph", icon='ADD')
            if props.sql_query_status:
                status = box.box()
                status.scale_y = 0.75
                status.label(text=props.sql_query_status, icon='INFO')

        elif props.data_source == 'SUITESPARSE':
            box.prop(props, "suitesparse_id", text="Matrix")
            box.prop(props, "suitesparse_mode", text="Mode")
            box.prop(props, "suitesparse_giant_only")
            row = box.row(align=True)
            row.operator("scigraphs.download_suitesparse", text="Download & Import", icon='IMPORT')
            row.operator("scigraphs.browse_suitesparse", text="Browse", icon='URL')
            if props.suitesparse_status:
                status = box.box()
                status.scale_y = 0.75
                status.label(text=props.suitesparse_status, icon='INFO')

        elif props.data_source == 'REPRO':
            repro = context.scene.scigraphs_repro
            box.prop(repro, "pipeline_path", text="Pipeline")
            row = box.row(align=True)
            row.operator("scigraphs.validate_pipeline", text="Validate", icon='CHECKMARK')
            row.operator("scigraphs.run_pipeline", text="Run Pipeline", icon='PLAY')

            template_box = layout.box()
            template_box.label(text="Templates and artifacts", icon='FILE_CACHE')
            row = template_box.row(align=True)
            row.operator("scigraphs.export_pipeline_template", text="Template", icon='EXPORT')
            row.operator("scigraphs.export_current_repro_spec", text="Export Scene", icon='SCENE_DATA')
            template_box.prop(repro, "artifacts_path", text="Artifacts")
            template_box.operator("scigraphs.open_artifacts_folder", text="Open Artifacts", icon='FILEBROWSER')

        if props.data_source not in {'SUITESPARSE', 'REPRO'}:
            layout.separator()
            graph_box = layout.box()
            graph_box.label(text="Graph structure", icon='OUTLINER')
            graph_box.prop(props, "source_column", text="Source")
            graph_box.prop(props, "target_column", text="Target")
            graph_box.prop(props, "is_directed", text="Directed")
            graph_box.prop(props, "remove_self_loops", text="Remove Self-Loops")
            graph_box.prop(props, "weight_column", text="Weight")

        if props.data_source != 'REPRO':
            layout.separator()
            layout.prop(props, "auto_layout_on_import", text="Auto Layout")
            if props.auto_layout_on_import:
                layout.prop(props, "layout_algorithm", text="Layout")

    def execute(self, context):
        props = context.scene.scigraphs

        if props.data_source == 'DATABASE':
            return bpy.ops.scigraphs.load_sql_columns()
        if props.data_source == 'SUITESPARSE':
            return bpy.ops.scigraphs.download_suitesparse()
        if props.data_source == 'REPRO':
            return bpy.ops.scigraphs.run_pipeline('INVOKE_DEFAULT')
        return bpy.ops.scigraphs.load_columns()


class SCIGRAPHS_OT_osmnx_quick_import_dialog(bpy.types.Operator):
    """Floating import dialog for OSMnx street networks."""
    bl_idname = "scigraphs.osmnx_quick_import_dialog"
    bl_label = "OSMnx Quick Import"
    bl_description = "Configure and import an OSMnx network"

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=540)

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Area of interest", icon='WORLD')
        box.prop(props, "osmnx_download_method", text="Method")

        method = props.osmnx_download_method
        if method == 'PLACE':
            box.prop(props, "osmnx_place_name", text="Place")
            box.prop(props, "osmnx_which_result", text="Result #")
        elif method == 'MULTI_PLACE':
            box.prop(props, "osmnx_place_list", text="Places")
        elif method == 'POINT':
            box.prop(props, "osmnx_latitude", text="Latitude")
            box.prop(props, "osmnx_longitude", text="Longitude")
            box.prop(props, "osmnx_distance", text="Radius (m)")
        elif method == 'ADDRESS':
            box.prop(props, "osmnx_address", text="Address")
            box.prop(props, "osmnx_distance", text="Radius (m)")
        elif method == 'BBOX':
            col = box.column(align=True)
            col.prop(props, "osmnx_bbox_north", text="North")
            col.prop(props, "osmnx_bbox_south", text="South")
            col.prop(props, "osmnx_bbox_east", text="East")
            col.prop(props, "osmnx_bbox_west", text="West")
        elif method == 'POLYGON':
            box.prop_search(props, "osmnx_polygon_object", bpy.data, "objects", text="Polygon Object")
        elif method == 'XML':
            box.prop(props, "osmnx_xml_filepath", text="OSM XML")

        filter_box = layout.box()
        filter_box.label(text="Network and filters", icon='FILTER')
        filter_box.prop(props, "osmnx_custom_filter_preset", text="Preset")
        if props.osmnx_custom_filter_preset == 'NONE':
            filter_box.prop(props, "osmnx_network_type", text="Network")
        filter_box.prop(props, "osmnx_custom_filter_text", text="Custom Filter")

        options = layout.box()
        options.label(text="Import options", icon='PREFERENCES')
        options.prop(props, "osmnx_simplify", text="Simplify")
        options.prop(props, "osmnx_retain_geometry", text="Keep Curves")
        options.prop(props, "osmnx_truncate_by_edge")
        options.prop(props, "osmnx_retain_all")
        options.prop(props, "osmnx_scale", text="Scale")

    def execute(self, context):
        return bpy.ops.scigraphs.import_osm_graph()


class SCIGRAPHS_OT_osmnx_quick_action_dialog(bpy.types.Operator):
    """Quick parameter dialog for OSMnx toolbar actions."""
    bl_idname = "scigraphs.osmnx_quick_action_dialog"
    bl_label = "OSMnx Quick Action"
    bl_description = "Configure and run an OSMnx toolbar action"

    action: bpy.props.StringProperty(options={'SKIP_SAVE'})
    feature_source: bpy.props.EnumProperty(
        name="Feature Source",
        description="Where to download OSM features from",
        items=[
            ('PLACE', "Place", "Download features by place name"),
            ('ADDRESS', "Address", "Download features around a geocoded address"),
            ('POINT', "Point", "Download features around latitude/longitude"),
            ('BBOX', "BBox", "Download features within a bounding box"),
            ('POLYGON', "Polygon", "Download features inside the active polygon object"),
            ('XML', "XML", "Parse features from a local OSM XML file"),
        ],
        default='PLACE',
        options={'SKIP_SAVE'},
    )
    orientation_action: bpy.props.EnumProperty(
        name="Orientation Action",
        description="Orientation analysis to run",
        items=[
            ('ENTROPY', "Entropy", "Calculate street-network orientation entropy"),
            ('DISTRIBUTION', "Distribution", "Calculate a bearing histogram"),
            ('ROSE', "3D Rose", "Create a 3D orientation rose mesh"),
        ],
        default='ENTROPY',
        options={'SKIP_SAVE'},
    )

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=520)

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        layout.use_property_split = True
        layout.use_property_decorate = False

        if self.action in {'shortest_path', 'k_routes'}:
            box = layout.box()
            title = "Shortest Path" if self.action == 'shortest_path' else "K Shortest Routes"
            box.label(text=title, icon='TRACKING')
            box.prop(props, "osmnx_path_weight", text="Weight")
            if props.osmnx_path_weight == 'elevation_impedance':
                box.prop(props, "osmnx_impedance_alpha", text="Elevation Penalty")
            row = box.row(align=True)
            row.prop(props, "osmnx_shortest_path_source", text="Source")
            row.operator("scigraphs.osmnx_select_path_source", text="", icon='EYEDROPPER')
            row = box.row(align=True)
            row.prop(props, "osmnx_shortest_path_target", text="Target")
            row.operator("scigraphs.osmnx_select_path_target", text="", icon='EYEDROPPER')
            if self.action == 'k_routes':
                box.prop(props, "osmnx_k_shortest", text="Routes")

        elif self.action == 'isochrones':
            box = layout.box()
            box.label(text="Isochrones", icon='MESH_CIRCLE')
            box.prop(props, "osmnx_iso_center_node", text="Center Node")
            box.prop(props, "osmnx_selected_node_id", text="Selected Node")
            box.prop(props, "osmnx_iso_trip_times", text="Times")
            box.prop(props, "osmnx_iso_travel_speed", text="Speed km/h")
            box.prop(props, "osmnx_iso_mode", text="Mode")
            if props.osmnx_iso_mode == 'BUFFER_UNION':
                box.prop(props, "osmnx_iso_buffer", text="Buffer")
            box.operator("scigraphs.osmnx_select_nearest_node", text="Pick Center Node", icon='EYEDROPPER')

        elif self.action == 'attr_colors':
            box = layout.box()
            box.label(text="Attribute Colors", icon='COLOR')
            box.prop(props, "osmnx_color_attr_name", text="Attribute")
            box.prop(props, "osmnx_colormap", text="Colormap")

        elif self.action == 'batch_routes':
            box = layout.box()
            box.label(text="Batch Routes", icon='STICKY_UVS_LOC')
            box.prop(props, "osmnx_path_weight", text="Weight")
            if props.osmnx_path_weight == 'elevation_impedance':
                box.prop(props, "osmnx_impedance_alpha", text="Elevation Penalty")
            box.prop(props, "osmnx_od_random_n", text="Random OD Pairs")
            box.prop(props, "osmnx_od_batch_cpus", text="CPU Cores")

        elif self.action == 'stats':
            box = layout.box()
            box.label(text="Basic Stats", icon='INFO')
            row = box.row(align=True)
            row.prop(props, "osmnx_network_area", text="Area km2")
            row.operator("scigraphs.osmnx_estimate_area", text="", icon='FILE_REFRESH')
            box.label(text="Area is used for node/edge/street density metrics.", icon='INFO')

        elif self.action == 'speeds':
            box = layout.box()
            box.label(text="Edge Speeds", icon='DRIVER')
            box.prop(props, "osmnx_fallback_speed", text="Fallback km/h")

        elif self.action == 'consolidate':
            box = layout.box()
            box.label(text="Consolidate Intersections", icon='AUTOMERGE_ON')
            box.prop(props, "osmnx_simplification_tolerance", text="Tolerance (m)")
            box.label(text="Project the graph first for meter-based tolerance.", icon='INFO')

        elif self.action == 'features':
            box = layout.box()
            box.label(text="OSM Features", icon='POINTCLOUD_DATA')
            box.prop(self, "feature_source", text="Download by")
            box.prop(props, "osmnx_feature_type", text="Feature")
            if props.osmnx_feature_type == 'CUSTOM':
                box.prop(props, "osmnx_custom_tags", text="Tags")
            if self.feature_source == 'PLACE':
                box.prop(props, "osmnx_features_place", text="Place")
                box.label(text="Falls back to the network place when empty.", icon='INFO')
            elif self.feature_source == 'ADDRESS':
                box.prop(props, "osmnx_geocode_address", text="Address")
                box.prop(props, "osmnx_features_distance", text="Radius")
            elif self.feature_source == 'POINT':
                box.prop(props, "osmnx_latitude", text="Latitude")
                box.prop(props, "osmnx_longitude", text="Longitude")
                box.prop(props, "osmnx_features_distance", text="Radius")
            elif self.feature_source == 'BBOX':
                col = box.column(align=True)
                col.prop(props, "osmnx_bbox_north", text="North")
                col.prop(props, "osmnx_bbox_south", text="South")
                col.prop(props, "osmnx_bbox_east", text="East")
                col.prop(props, "osmnx_bbox_west", text="West")
            elif self.feature_source == 'POLYGON':
                box.label(text="Uses the active polygon mesh as the boundary.", icon='MESH_CIRCLE')
            elif self.feature_source == 'XML':
                box.label(text="A file selector will open for the OSM XML file.", icon='FILE')
            box.separator()
            box.prop(props, "osmnx_poi_snap_mode", text="Snap Mode")
            box.operator("scigraphs.osmnx_snap_pois", text="Snap Active POIs", icon='EMPTY_SINGLE_ARROW')

        elif self.action == 'orientations':
            box = layout.box()
            box.label(text="Orientation Analysis", icon='ORIENTATION_GIMBAL')
            box.prop(self, "orientation_action", text="Action")
            if self.orientation_action == 'DISTRIBUTION':
                box.prop(props, "osmnx_bearing_num_bins", text="Bins")
            elif self.orientation_action == 'ROSE':
                box.prop(props, "osmnx_rose_bins", text="Bins")
                box.prop(props, "osmnx_rose_radius", text="Radius")
                box.prop(props, "osmnx_rose_height_scale", text="Height")

    @staticmethod
    def _feature_type(props, allowed):
        feature_type = getattr(props, "osmnx_feature_type", "BUILDING")
        if feature_type in allowed:
            return feature_type
        return 'BUILDING'

    @staticmethod
    def _ensure_active_osmnx_graph(context):
        obj = context.active_object
        if obj and obj.get("is_osmnx", False):
            return obj

        for candidate in context.scene.objects:
            if candidate.get("is_osmnx", False):
                context.view_layer.objects.active = candidate
                candidate.select_set(True)
                return candidate
        return None

    def _run_with_osmnx_graph(self, context, operator_call, *args, **kwargs):
        if self._ensure_active_osmnx_graph(context) is None:
            self.report({'ERROR'}, "Select or import an OSMnx graph first")
            return {'CANCELLED'}

        try:
            return operator_call(*args, **kwargs)
        except RuntimeError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

    def execute(self, context):
        props = context.scene.scigraphs
        if self.action == 'shortest_path':
            return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_shortest_path)
        if self.action == 'k_routes':
            return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_k_shortest)
        if self.action == 'isochrones':
            return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_isochrones)
        if self.action == 'attr_colors':
            return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_attr_to_colors)
        if self.action == 'batch_routes':
            return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_batch_routes)
        if self.action == 'stats':
            return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_basic_stats)
        if self.action == 'speeds':
            return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_add_edge_speeds)
        if self.action == 'consolidate':
            return self._run_with_osmnx_graph(
                context,
                bpy.ops.scigraphs.osmnx_consolidate,
                tolerance=props.osmnx_simplification_tolerance,
            )
        if self.action == 'features':
            if self.feature_source == 'PLACE':
                place = props.osmnx_features_place.strip() or props.osmnx_place_name.strip()
                return bpy.ops.scigraphs.osmnx_features_place(
                    place=place,
                    feature_type=self._feature_type(props, {'BUILDING', 'AMENITY', 'RESTAURANT', 'AMENITY_METAPATH', 'LANDUSE', 'NATURAL', 'CUSTOM', 'HIGHWAY'}),
                    custom_tags=props.osmnx_custom_tags,
                )
            if self.feature_source == 'ADDRESS':
                return bpy.ops.scigraphs.osmnx_features_address(
                    address=props.osmnx_geocode_address,
                    distance=props.osmnx_features_distance,
                    feature_preset=self._feature_type(props, {'BUILDING', 'AMENITY', 'RESTAURANT', 'SHOP', 'LEISURE', 'PARKING', 'BUS_STOP', 'RAIL_STATION', 'PARK', 'EDUCATION', 'HEALTH', 'AMENITY_METAPATH', 'LANDUSE', 'NATURAL', 'WATER', 'HIGHWAY', 'CUSTOM'}),
                    custom_tags=props.osmnx_custom_tags,
                )
            if self.feature_source == 'POINT':
                return bpy.ops.scigraphs.osmnx_features_point(
                    latitude=props.osmnx_latitude,
                    longitude=props.osmnx_longitude,
                    distance=props.osmnx_features_distance,
                    feature_type=self._feature_type(props, {'BUILDING', 'AMENITY', 'RESTAURANT', 'AMENITY_METAPATH', 'NATURAL', 'LANDUSE', 'HIGHWAY', 'CUSTOM'}),
                    custom_tags=props.osmnx_custom_tags,
                )
            if self.feature_source == 'BBOX':
                return bpy.ops.scigraphs.osmnx_features_bbox(
                    feature_type=self._feature_type(props, {'BUILDING', 'AMENITY', 'AMENITY_METAPATH', 'LANDUSE', 'NATURAL', 'HIGHWAY', 'CUSTOM'}),
                    custom_tags=props.osmnx_custom_tags,
                )
            if self.feature_source == 'POLYGON':
                return bpy.ops.scigraphs.osmnx_features_polygon(
                    feature_preset=self._feature_type(props, {'BUILDING', 'AMENITY', 'RESTAURANT', 'SHOP', 'LEISURE', 'PARKING', 'BUS_STOP', 'RAIL_STATION', 'PARK', 'EDUCATION', 'HEALTH', 'AMENITY_METAPATH', 'LANDUSE', 'NATURAL', 'WATER', 'HIGHWAY', 'CUSTOM'}),
                    custom_tags=props.osmnx_custom_tags,
                )
            if self.feature_source == 'XML':
                return bpy.ops.scigraphs.osmnx_features_xml('INVOKE_DEFAULT')
        if self.action == 'orientations':
            if self.orientation_action == 'ENTROPY':
                return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_orientation_entropy)
            if self.orientation_action == 'DISTRIBUTION':
                return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_bearings_distribution)
            if self.orientation_action == 'ROSE':
                return self._run_with_osmnx_graph(context, bpy.ops.scigraphs.osmnx_orientation_rose)
        self.report({'WARNING'}, f"Unknown OSMnx quick action: {self.action}")
        return {'CANCELLED'}


class SCIGRAPHS_OT_city2graph_quick_action_dialog(bpy.types.Operator):
    """Quick parameter dialog for City2Graph toolbar actions."""
    bl_idname = "scigraphs.city2graph_quick_action_dialog"
    bl_label = "City2Graph Quick Action"
    bl_description = "Configure and run a City2Graph toolbar action"

    action: bpy.props.StringProperty(options={'SKIP_SAVE'})

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=560)

    @staticmethod
    def _prop(layout, props, name, **kwargs):
        if hasattr(props, name):
            layout.prop(props, name, **kwargs)

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()

        if self.action == 'boundary':
            box.label(text="Boundary", icon='VIEWZOOM')
            self._prop(box, props, "geocode_place_name", text="Place")

        elif self.action == 'overture':
            box.label(text="Overture Maps", icon='WORLD')
            self._prop(box, props, "geocode_place_name", text="Place")
            box.label(text="If Place is empty, the REST/bbox download is used.", icon='INFO')
            self._prop(box, props, "c2g_use_osmnx_bbox", text="Use OSMnx BBox")
            if not getattr(props, "c2g_use_osmnx_bbox", True):
                self._prop(box, props, "c2g_bbox_north", text="North")
                self._prop(box, props, "c2g_bbox_south", text="South")
                self._prop(box, props, "c2g_bbox_east", text="East")
                self._prop(box, props, "c2g_bbox_west", text="West")
            box.separator()
            self._prop(box, props, "c2g_overture_building", text="Buildings")
            self._prop(box, props, "c2g_overture_segment", text="Segments")
            self._prop(box, props, "c2g_overture_connector", text="Connectors")
            self._prop(box, props, "c2g_overture_place", text="Places")
            self._prop(box, props, "c2g_overture_water", text="Water")
            self._prop(box, props, "c2g_overture_land", text="Land")

        elif self.action == 'tessellation':
            box.label(text="Tessellation", icon='MOD_TRIANGULATE')
            self._prop(box, props, "c2g_tessellation_shrink", text="Shrink")
            self._prop(box, props, "c2g_tessellation_segment", text="Segment")

        elif self.action == 'morphology':
            box.label(text="Morphological Graph", icon='FORCE_MAGNETIC')
            self._prop(box, props, "morpho_use_center_from_osmnx", text="Use OSMnx Center")
            if not getattr(props, "morpho_use_center_from_osmnx", False):
                self._prop(box, props, "morpho_center_lat", text="Center Latitude")
                self._prop(box, props, "morpho_center_lon", text="Center Longitude")
            box.separator()
            self._prop(box, props, "morpho_distance", text="Radius")
            self._prop(box, props, "morpho_clipping_buffer", text="Clipping Buffer")
            self._prop(box, props, "morpho_contiguity", text="Contiguity")
            box.separator()
            self._prop(box, props, "morpho_keep_buildings", text="Keep Buildings")
            self._prop(box, props, "morpho_keep_segments", text="Keep Segments")
            box.label(text="Select tessellation and street network before running.", icon='INFO')

        elif self.action == 'proximity':
            box.label(text="Proximity Graph", icon='PIVOT_INDIVIDUAL')
            self._prop(box, props, "prox_feature_object", text="Features")
            self._prop(box, props, "prox_graph_type", text="Graph")
            self._prop(box, props, "prox_distance_metric", text="Distance")
            if getattr(props, "prox_distance_metric", None) == 'NETWORK':
                self._prop(box, props, "prox_network_object", text="Network")
            box.separator()
            graph_type = getattr(props, "prox_graph_type", "")
            if graph_type == 'KNN':
                self._prop(box, props, "prox_knn_k", text="K")
            elif graph_type == 'FIXED_RADIUS':
                self._prop(box, props, "prox_radius", text="Radius")
            elif graph_type == 'WAXMAN':
                self._prop(box, props, "prox_waxman_beta", text="Beta")
                self._prop(box, props, "prox_waxman_r0", text="R0")
                self._prop(box, props, "prox_waxman_seed", text="Seed")
            elif graph_type == 'CONTIGUITY':
                self._prop(box, props, "prox_contiguity_type", text="Contiguity")
            box.separator()
            self._prop(box, props, "prox_deduplicate", text="Deduplicate")
            if getattr(props, "prox_deduplicate", False):
                self._prop(box, props, "prox_dedup_tolerance", text="Tolerance")
            self._prop(box, props, "prox_curve_thickness", text="Edge Thickness")
            self._prop(box, props, "prox_visualize_limit", text="Viz Limit")
            self._prop(box, props, "prox_color_by_attribute", text="Color by Attribute")

        elif self.action == 'multilayer':
            box.label(text="Multilayer Graph", icon='LINKED')
            self._prop(box, props, "prox_layer1_object", text="Layer 1")
            self._prop(box, props, "prox_layer2_object", text="Layer 2")
            self._prop(box, props, "prox_layer3_object", text="Layer 3")
            self._prop(box, props, "prox_multilayer_method", text="Method")
            if getattr(props, "prox_multilayer_method", None) == 'KNN':
                self._prop(box, props, "prox_multilayer_k", text="K")
            else:
                self._prop(box, props, "prox_multilayer_radius", text="Radius")
            self._prop(box, props, "prox_distance_metric", text="Distance")
            if getattr(props, "prox_distance_metric", None) == 'NETWORK':
                self._prop(box, props, "prox_network_object", text="Network")

        elif self.action == 'group_nodes':
            box.label(text="Group Nodes", icon='GROUP_VERTEX')
            self._prop(box, props, "prox_polygons_object", text="Polygons")
            self._prop(box, props, "prox_points_object", text="Points")
            self._prop(box, props, "prox_group_predicate", text="Predicate")
            self._prop(box, props, "prox_distance_metric", text="Distance")
            if getattr(props, "prox_distance_metric", None) == 'NETWORK':
                self._prop(box, props, "prox_network_object", text="Network")

        elif self.action == 'travel':
            box.label(text="Travel Graph", icon='OUTLINER_OB_CURVE')
            self._prop(box, props, "gtfs_calendar_start", text="Start")
            self._prop(box, props, "gtfs_calendar_end", text="End")

        elif self.action == 'od_graph':
            box.label(text="OD Graph", icon='FORCE_FORCE')
            self._prop(box, props, "od_zones_object", text="Zones")
            self._prop(box, props, "od_zone_id_col", text="Zone ID")
            self._prop(box, props, "od_matrix_type", text="Matrix")
            box.separator()
            self._prop(box, props, "od_source_col", text="Source Column")
            self._prop(box, props, "od_target_col", text="Target Column")
            self._prop(box, props, "od_weight_col", text="Weight Column")
            box.separator()
            self._prop(box, props, "od_threshold", text="Threshold")
            self._prop(box, props, "od_directed", text="Directed")

        elif self.action == 'graph_tools':
            box.label(text="Graph Tools", icon='MODIFIER')
            self._prop(box, props, "graph_tool_action", text="Tool")
            tool = getattr(props, "graph_tool_action", "")
            if tool == 'FILTER':
                self._prop(box, props, "graph_filter_center", text="Center")
                self._prop(box, props, "graph_filter_threshold", text="Distance")
            elif tool == 'CLIP':
                self._prop(box, props, "graph_filter_center", text="Clip Polygon")
            elif tool == 'ISOCHRONE':
                self._prop(box, props, "isochrone_center_object", text="Center")
                self._prop(box, props, "isochrone_threshold", text="Cost")
                self._prop(box, props, "isochrone_weight_attr", text="Weight")
            elif tool == 'REMOVE_ISOLATED':
                box.label(text="Keeps only the largest connected component.", icon='INFO')
            box.label(text="Run on the active graph object.", icon='INFO')

        elif self.action == 'bridge_amenities':
            box.label(text="Bridge Amenities", icon='STICKY_UVS_LOC')
            self._prop(box, props, "metapath_amenities_object", text="Amenities")
            self._prop(box, props, "metapath_k_neighbors", text="K Neighbors")
            self._prop(box, props, "metapath_amenity_limit", text="Limit")

        elif self.action == 'weighted_metapaths':
            box.label(text="Weighted Metapaths", icon='DRIVER_DISTANCE')
            self._prop(box, props, "metapath_weight_attr", text="Weight Attribute")
            self._prop(box, props, "metapath_weight_threshold", text="Max")
            self._prop(box, props, "metapath_weight_min_threshold", text="Min")
            self._prop(box, props, "metapath_endpoint_type", text="Endpoint")

    def execute(self, context):
        action_ops = {
            'boundary': bpy.ops.scigraphs.c2g_geocode_boundaries,
            'tessellation': bpy.ops.scigraphs.c2g_generate_tessellation,
            'morphology': bpy.ops.scigraphs.c2g_morphological_graph,
            'proximity': bpy.ops.scigraphs.generate_proximity_graph,
            'multilayer': bpy.ops.scigraphs.generate_multilayer_graph,
            'group_nodes': bpy.ops.scigraphs.generate_group_nodes_graph,
            'travel': bpy.ops.scigraphs.c2g_travel_summary_graph,
            'od_graph': bpy.ops.scigraphs.c2g_od_to_graph,
            'graph_tools': bpy.ops.scigraphs.c2g_graph_tool_apply,
            'bridge_amenities': bpy.ops.scigraphs.bridge_amenities,
            'weighted_metapaths': bpy.ops.scigraphs.compute_metapaths_by_weight,
        }
        if self.action == 'overture':
            props = context.scene.city2graph
            if getattr(props, "geocode_place_name", "").strip():
                return bpy.ops.scigraphs.c2g_load_overture_place()
            return bpy.ops.scigraphs.c2g_load_overture()
        op = action_ops.get(self.action)
        if op is None:
            self.report({'WARNING'}, f"Unknown City2Graph quick action: {self.action}")
            return {'CANCELLED'}
        return op()


class SCIGRAPHS_OT_panel_quick_dialog(bpy.types.Operator):
    """Panel-oriented quick controls for the SciGraphs toolbar."""
    bl_idname = "scigraphs.panel_quick_dialog"
    bl_label = "SciGraphs Panel Tools"
    bl_description = "Open quick actions for a SciGraphs subpanel"

    panel: bpy.props.StringProperty(options={'SKIP_SAVE'})

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self, width=620)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        drawers = {
            "DATA": self._draw_data,
            "LAYOUT": self._draw_layout,
            "ALGORITHMS": self._draw_algorithms,
            "ANALYSIS": self._draw_analysis,
            "VISUALIZATION": self._draw_visualization,
            "EXPORT": self._draw_export,
        }

        drawer = drawers.get(self.panel)
        if drawer is None:
            layout.label(text="Unknown SciGraphs panel", icon='ERROR')
            return

        drawer(context, layout)

    def execute(self, context):
        return {'FINISHED'}

    @staticmethod
    def _active_graph(context):
        obj = context.active_object
        if obj is not None and "num_nodes" in obj:
            return obj
        return None

    @staticmethod
    def _prop(layout, props, name, **kwargs):
        if hasattr(props, name):
            layout.prop(props, name, **kwargs)
            return True
        return False

    @staticmethod
    def _no_graph_box(layout):
        box = layout.box()
        box.label(text="No graph loaded", icon='ERROR')
        box.label(text="Create or import a graph first in Data.")

    def _draw_data(self, context, layout):
        props = context.scene.scigraphs

        box = layout.box()
        box.label(text="Data", icon='IMPORT')
        box.operator("scigraphs.quick_import_dialog", text="Open Full Import Dialog", icon='IMPORT')

        row = box.row(align=True)
        row.prop_enum(props, "data_source", 'FILE')
        row.prop_enum(props, "data_source", 'DATABASE')
        row.prop_enum(props, "data_source", 'SUITESPARSE')
        row.prop_enum(props, "data_source", 'REPRO')

        if props.data_source == 'FILE':
            box.prop(props, "filepath", text="File")
            box.prop(props, "csv_delimiter", text="Delimiter")
            row = box.row(align=True)
            row.operator("scigraphs.load_columns", text="Load Columns", icon='IMPORT')
            row.operator("scigraphs.create_graph", text="Create Graph", icon='ADD')
        elif props.data_source == 'DATABASE':
            box.prop(props, "db_profile_index", text="Connection")
            box.prop(props, "sql_query", text="SQL")
            row = box.row(align=True)
            row.operator("scigraphs.load_sql_columns", text="Load Columns", icon='IMPORT')
            row.operator("scigraphs.preview_sql_query", text="Preview", icon='VIEWZOOM')
            box.operator("scigraphs.create_graph_from_sql", text="Create Graph", icon='ADD')
        elif props.data_source == 'SUITESPARSE':
            box.prop(props, "suitesparse_id", text="Matrix")
            box.prop(props, "suitesparse_mode", text="Mode")
            box.prop(props, "suitesparse_giant_only")
            row = box.row(align=True)
            row.operator("scigraphs.download_suitesparse", text="Download & Import", icon='IMPORT')
            row.operator("scigraphs.browse_suitesparse", text="Browse", icon='URL')
        elif props.data_source == 'REPRO':
            repro = context.scene.scigraphs_repro
            box.prop(repro, "pipeline_path", text="Pipeline")
            row = box.row(align=True)
            row.operator("scigraphs.validate_pipeline", text="Validate", icon='CHECKMARK')
            row.operator("scigraphs.run_pipeline", text="Run Pipeline", icon='PLAY')

        if props.data_source not in {'SUITESPARSE', 'REPRO'}:
            structure = layout.box()
            structure.label(text="Graph Structure", icon='OUTLINER')
            structure.prop(props, "source_column", text="Source")
            structure.prop(props, "target_column", text="Target")
            structure.prop(props, "is_directed", text="Directed")
            structure.prop(props, "remove_self_loops", text="Remove Self-Loops")
            structure.prop(props, "weight_column", text="Weight")

        obj = self._active_graph(context)
        if obj is not None:
            tools = layout.box()
            tools.label(text=f"Active graph: {obj.get('num_nodes', 0)} nodes", icon='MESH_DATA')
            tools.operator("scigraphs.import_node_attributes", text="Import Node Attributes", icon='SPREADSHEET')
            tools.operator("scigraphs.setup_visualization", text="Setup Visualization", icon='GEOMETRY_NODES')

    def _draw_layout(self, context, layout):
        props = context.scene.scigraphs
        if self._active_graph(context) is None:
            self._no_graph_box(layout)
            return

        box = layout.box()
        box.label(text="Layout & Positioning", icon='NODETREE')
        box.prop(props, "layout_algorithm", text="Algorithm")
        box.prop(props, "layout_scale", text="Scale")
        box.prop(props, "iterations", text="Iterations")
        box.operator("scigraphs.apply_layout", text="Apply Layout", icon='PLAY')

        interactive = layout.box()
        interactive.label(text="Interactive Layout", icon='FRAME_NEXT')
        self._prop(interactive, props, "iterations_per_frame", text="Iterations / Frame")
        row = interactive.row(align=True)
        row.operator("scigraphs.execute_layout_step", text="Step", icon='FRAME_NEXT')
        row.operator("scigraphs.reset_layout", text="Reset", icon='LOOP_BACK')
        interactive.operator("scigraphs.bake_animation", text="Bake Animation", icon='REC')

        splitter = layout.box()
        splitter.label(text="Network Splitter 3D", icon='MOD_EXPLODE')
        row = splitter.row(align=True)
        row.operator("scigraphs.network_splitter_3d", text="Split Layers", icon='MOD_EXPLODE')
        row.operator("scigraphs.reset_splitter", text="Flatten Z", icon='LOOP_BACK')

    def _draw_algorithms(self, context, layout):
        props = context.scene.scigraphs
        obj = self._active_graph(context)
        if obj is None:
            self._no_graph_box(layout)
            return

        traversal = layout.box()
        traversal.label(text="Traversal", icon='ORIENTATION_VIEW')
        traversal.prop(props, "traversal_algorithm", text="Algorithm")
        traversal.prop(props, "traversal_start_mode", text="Start")
        if props.traversal_start_mode == 'MANUAL':
            traversal.prop(props, "traversal_start_nodes", text="Nodes")
        traversal.prop(props, "traversal_animation_mode", text="Mode")
        traversal.operator("scigraphs.animate_traversal", text="Animate Traversal", icon='PLAY')

        path = layout.box()
        path.label(text="Pathfinding", icon='CURVE_PATH')
        path.prop(props, "pathfinding_algorithm", text="Algorithm")
        row = path.row(align=True)
        row.prop(props, "pathfinding_source", text="Source")
        op = row.operator("scigraphs.pick_path_node", text="", icon='EYEDROPPER')
        op.target = 'SOURCE'
        row = path.row(align=True)
        row.prop(props, "pathfinding_target", text="Target")
        op = row.operator("scigraphs.pick_path_node", text="", icon='EYEDROPPER')
        op.target = 'TARGET'
        path.operator("scigraphs.find_shortest_path", text="Find Shortest Path", icon='PLAY')
        path.operator("scigraphs.path_tool", text="Pick Source and Target in Viewport", icon='TRACKING')

        spanning = layout.box()
        spanning.label(text="Spanning Trees", icon='OUTLINER_OB_FORCE_FIELD')
        spanning.prop(props, "spanning_algorithm", text="Algorithm")
        spanning.operator("scigraphs.compute_mst", text="Compute MST", icon='PLAY')

        flow = layout.box()
        flow.label(text="Network Flow", icon='MOD_FLUIDSIM')
        if obj.get("is_directed", False):
            flow.prop(props, "flow_source", text="Source")
            flow.prop(props, "flow_sink", text="Sink")
            row = flow.row(align=True)
            row.operator("scigraphs.compute_max_flow", text="Max Flow", icon='PLAY')
            row.operator("scigraphs.compute_min_cut", text="Min Cut", icon='SELECT_DIFFERENCE')
        else:
            flow.label(text="Requires a directed graph", icon='INFO')

    def _draw_analysis(self, context, layout):
        props = context.scene.scigraphs
        obj = self._active_graph(context)
        if obj is None:
            self._no_graph_box(layout)
            return

        centrality = layout.box()
        centrality.label(text="Centrality Metrics", icon='DRIVER')
        centrality.prop(props, "centrality_method", text="Method")
        row = centrality.row(align=True)
        row.operator("scigraphs.calculate_centrality", text="Centrality", icon='PLAY')
        row.operator("scigraphs.calculate_clustering", text="Local Clustering", icon='MESH_UVSPHERE')

        community = layout.box()
        community.label(text="Community Detection", icon='GROUP')
        community.prop(props, "clustering_algorithm", text="Algorithm")
        community.operator("scigraphs.apply_clustering", text="Detect Communities", icon='PLAY')

        stats = layout.box()
        stats.label(text="Statistics", icon='INFO')
        row = stats.row(align=True)
        row.operator("scigraphs.calculate_global_statistics", text="Global Stats", icon='VIEWZOOM')
        row.operator("scigraphs.generate_statistics_report", text="Report", icon='TEXT')

        directed = layout.box()
        directed.label(text="Directed Analysis", icon='FORWARD')
        if obj.get("is_directed", False):
            directed.prop(props, "directed_centrality_method", text="Centrality")
            directed.operator("scigraphs.calculate_directed_centrality", text="Directed Centrality", icon='PLAY')
            row = directed.row(align=True)
            row.operator("scigraphs.detect_patterns", text="Patterns", icon='VIEWZOOM')
            row.operator("scigraphs.find_sccs", text="SCCs", icon='STICKY_UVS_LOC')
            row = directed.row(align=True)
            row.operator("scigraphs.analyze_flow", text="Flow Roles", icon='MOD_FLUIDSIM')
            row.operator("scigraphs.animate_flow", text="Animate Flow", icon='ANIM')
        else:
            directed.label(text="Requires a directed graph", icon='INFO')

        topology = layout.box()
        topology.label(text="Topological Analysis", icon='SURFACE_DATA')
        row = topology.row(align=True)
        row.operator("scigraphs.check_planarity", text="Planarity", icon='CHECKMARK')
        row.operator("scigraphs.calculate_genus", text="Genus", icon='SURFACE_DATA')
        row = topology.row(align=True)
        row.operator("scigraphs.compute_faces", text="Faces", icon='FACESEL')
        row.operator("scigraphs.visualize_surface", text="Embedding", icon='MOD_WIREFRAME')
        topology.operator("scigraphs.create_dual_graph", text="Create Dual Graph", icon='MOD_WIREFRAME')

    def _draw_visualization(self, context, layout):
        props = context.scene.scigraphs
        if self._active_graph(context) is None:
            self._no_graph_box(layout)
            return

        setup = layout.box()
        setup.label(text="Visualization", icon='GEOMETRY_NODES')
        setup.operator("scigraphs.setup_visualization", text="Setup Geometry Nodes", icon='GEOMETRY_NODES')

        appearance = layout.box()
        appearance.label(text="Appearance", icon='MATERIAL')
        for name, label in (
            ("node_size", "Node Size"),
            ("node_resolution", "Node Resolution"),
            ("node_shape", "Node Shape"),
            ("edge_thickness", "Edge Thickness"),
            ("edge_resolution", "Edge Resolution"),
        ):
            self._prop(appearance, props, name, text=label)
        appearance.operator("scigraphs.update_appearance", text="Update Appearance", icon='SHADING_RENDERED')

        edges = layout.box()
        edges.label(text="Edge Style", icon='CURVE_BEZCURVE')
        self._prop(edges, props, "edge_style_preset", text="Preset")
        self._prop(edges, props, "edge_style_type", text="Type")
        row = edges.row(align=True)
        row.operator("scigraphs.apply_edge_style", text="Apply", icon='PLAY')
        row.operator("scigraphs.preview_edge_style", text="Preview", icon='HIDE_OFF')
        row.operator("scigraphs.reset_edge_style", text="Reset", icon='LOOP_BACK')

        scene = layout.box()
        scene.label(text="Scene Setup", icon='LIGHT')
        self._prop(scene, props, "rendering_preset", text="Material")
        self._prop(scene, props, "lighting_setup", text="Lighting")
        row = scene.row(align=True)
        row.operator("scigraphs.apply_rendering_preset", text="Material", icon='MATERIAL')
        row.operator("scigraphs.setup_lighting", text="Lighting", icon='LIGHT')

        text = layout.box()
        text.label(text="Text Labels", icon='FONT_DATA')
        self._prop(text, props, "text_overlay_enabled", text="Enabled")
        self._prop(text, props, "text_source", text="Source")
        if getattr(props, "text_source", "NODE_ID") == 'ATTRIBUTE':
            self._prop(text, props, "text_attribute", text="Attribute")
        row = text.row(align=True)
        row.operator("scigraphs.generate_text_overlay", text="Generate", icon='FONT_DATA')
        row.operator("scigraphs.remove_text_overlay", text="Remove", icon='X')

    def _draw_export(self, context, layout):
        props = context.scene.scigraphs
        if self._active_graph(context) is None:
            self._no_graph_box(layout)
            return

        export = layout.box()
        export.label(text="Export & Tools", icon='EXPORT')
        self._prop(export, props, "export_filepath", text="File")
        self._prop(export, props, "export_format", text="Format")
        self._prop(export, props, "export_include_attributes", text="Attributes")
        row = export.row(align=True)
        row.operator("scigraphs.export_graph", text="Export", icon='EXPORT')
        row.operator("scigraphs.export_positions", text="Positions", icon='EMPTY_AXIS')

        formats = layout.box()
        formats.label(text="Direct Formats", icon='FILE_TICK')
        row = formats.row(align=True)
        row.operator("scigraphs.export_gexf", text="GEXF")
        row.operator("scigraphs.export_graphml", text="GraphML")
        row.operator("scigraphs.export_pajek", text="Pajek")
        row.operator("scigraphs.export_json", text="JSON")

        utilities = layout.box()
        utilities.label(text="Utilities", icon='TOOL_SETTINGS')
        self._prop(utilities, props, "report_include_powerlaw", text="Power-law Fit")
        utilities.operator("scigraphs.generate_statistics_report", text="Statistics Report", icon='TEXT')
        utilities.operator("scigraphs.toggle_node_gizmos", text="Toggle Node Handles", icon='EMPTY_AXIS')


class SCIGRAPHS_OT_drag_quick_toolbar(bpy.types.Operator):
    """Drag the floating SciGraphs toolbar."""
    bl_idname = "scigraphs.drag_quick_toolbar"
    bl_label = "Move SciGraphs Toolbar"
    bl_description = "Drag to reposition the floating SciGraphs toolbar"

    _start_mouse = None
    _start_pos = None

    def invoke(self, context, event):
        wm = context.window_manager
        if wm.scigraphs_quick_toolbar_x < 0 or wm.scigraphs_quick_toolbar_y < 0:
            x, y, _spacing = _toolbar_origin(context, len(_current_actions(context)) + 2)
            wm.scigraphs_quick_toolbar_x = int(x)
            wm.scigraphs_quick_toolbar_y = int(y)

        self._start_mouse = (event.mouse_region_x, event.mouse_region_y)
        self._start_pos = (wm.scigraphs_quick_toolbar_x, wm.scigraphs_quick_toolbar_y)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        wm = context.window_manager

        if event.type == 'MOUSEMOVE':
            dx = event.mouse_region_x - self._start_mouse[0]
            dy = event.mouse_region_y - self._start_mouse[1]
            wm.scigraphs_quick_toolbar_x = max(8, int(self._start_pos[0] + dx))
            wm.scigraphs_quick_toolbar_y = max(8, int(self._start_pos[1] + dy))
            _tag_3d_views(context)
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE' and event.value == 'RELEASE':
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            wm.scigraphs_quick_toolbar_x, wm.scigraphs_quick_toolbar_y = self._start_pos
            _tag_3d_views(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}

    def execute(self, context):
        x, y, _spacing = _toolbar_origin(context, len(_current_actions(context)) + 2)
        context.window_manager.scigraphs_quick_toolbar_x = int(x)
        context.window_manager.scigraphs_quick_toolbar_y = int(y)
        _tag_3d_views(context)
        return {'FINISHED'}


class SCIGRAPHS_OT_quick_toolbar_action(bpy.types.Operator):
    """Run an action from the floating SciGraphs toolbar."""
    bl_idname = "scigraphs.quick_toolbar_action"
    bl_label = "SciGraphs Quick Action"
    bl_description = "Run a SciGraphs quick toolbar action"

    action: bpy.props.StringProperty(options={'SKIP_SAVE'})

    def execute(self, context):
        action = self.action

        try:
            panel_actions = {
                "panel_data": "DATA",
                "panel_layout": "LAYOUT",
                "panel_algorithms": "ALGORITHMS",
                "panel_analysis": "ANALYSIS",
                "panel_visualization": "VISUALIZATION",
                "panel_export": "EXPORT",
            }

            if action in panel_actions:
                bpy.ops.scigraphs.panel_quick_dialog('INVOKE_DEFAULT', panel=panel_actions[action])
            elif action == "pie_main":
                bpy.ops.wm.call_menu_pie(name="SCIGRAPHS_MT_PIE_main")
            elif action == "load_columns":
                bpy.ops.scigraphs.quick_import_dialog('INVOKE_DEFAULT')
            elif action == "create_graph":
                props = context.scene.scigraphs
                if props.data_source == 'DATABASE':
                    bpy.ops.scigraphs.create_graph_from_sql()
                else:
                    bpy.ops.scigraphs.create_graph()
            elif action == "apply_layout":
                bpy.ops.scigraphs.apply_layout('INVOKE_DEFAULT')
            elif action == "layout_step":
                bpy.ops.scigraphs.execute_layout_step()
            elif action == "reset_layout":
                bpy.ops.scigraphs.reset_layout()
            elif action == "centrality":
                bpy.ops.scigraphs.calculate_centrality()
            elif action == "clustering":
                bpy.ops.scigraphs.apply_clustering()
            elif action == "setup_viz":
                bpy.ops.scigraphs.setup_visualization()
            elif action == "appearance":
                bpy.ops.scigraphs.update_appearance('INVOKE_DEFAULT')
            elif action == "edge_style":
                bpy.ops.scigraphs.apply_edge_style_preset('INVOKE_DEFAULT')
            elif action == "topology":
                bpy.ops.wm.call_menu_pie(name="SCIGRAPHS_MT_PIE_topology")
            elif action == "node_handles":
                bpy.ops.scigraphs.toggle_node_gizmos()
            elif action == "export":
                bpy.ops.scigraphs.export_graph()
            elif action == "osmnx_import":
                bpy.ops.scigraphs.osmnx_quick_import_dialog('INVOKE_DEFAULT')
            elif action == "osmnx_project":
                bpy.ops.scigraphs.osmnx_project_graph('INVOKE_DEFAULT')
            elif action == "osmnx_simplify":
                bpy.ops.scigraphs.osmnx_simplify()
            elif action == "osmnx_consolidate":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='consolidate')
            elif action == "osmnx_speeds":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='speeds')
            elif action == "osmnx_travel_times":
                bpy.ops.scigraphs.osmnx_add_travel_times()
            elif action == "osmnx_shortest_path":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='shortest_path')
            elif action == "osmnx_k_routes":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='k_routes')
            elif action == "osmnx_batch_routes":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='batch_routes')
            elif action == "osmnx_route_summary":
                bpy.ops.scigraphs.osmnx_route_summary()
            elif action == "osmnx_isochrones":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='isochrones')
            elif action == "osmnx_stats":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='stats')
            elif action == "osmnx_centrality":
                bpy.ops.scigraphs.osmnx_centrality('INVOKE_DEFAULT')
            elif action == "osmnx_attr_colors":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='attr_colors')
            elif action == "osmnx_orientations":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='orientations')
            elif action == "osmnx_elevation_api":
                bpy.ops.scigraphs.osmnx_add_elevations_api()
            elif action == "osmnx_grades":
                bpy.ops.scigraphs.osmnx_add_edge_grades()
            elif action == "osmnx_features":
                bpy.ops.scigraphs.osmnx_quick_action_dialog('INVOKE_DEFAULT', action='features')
            elif action == "osmnx_cache":
                bpy.ops.scigraphs.osmnx_view_cached_graphs('INVOKE_DEFAULT')
            elif action == "osmnx_export":
                bpy.ops.scigraphs.osmnx_export('INVOKE_DEFAULT')
            elif action == "c2g_boundary":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='boundary')
            elif action == "c2g_overture":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='overture')
            elif action == "c2g_file":
                bpy.ops.scigraphs.c2g_load_file('INVOKE_DEFAULT')
            elif action == "c2g_process_segments":
                bpy.ops.scigraphs.c2g_process_segments()
            elif action == "c2g_segments_graph":
                bpy.ops.scigraphs.c2g_segments_to_graph()
            elif action == "c2g_tessellation":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='tessellation')
            elif action == "c2g_morphology":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='morphology')
            elif action == "c2g_proximity":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='proximity')
            elif action == "c2g_multilayer":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='multilayer')
            elif action == "c2g_group_nodes":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='group_nodes')
            elif action == "c2g_gtfs":
                bpy.ops.scigraphs.c2g_import_gtfs('INVOKE_DEFAULT')
            elif action == "c2g_travel":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='travel')
            elif action == "c2g_od_matrix":
                bpy.ops.scigraphs.c2g_load_od_matrix('INVOKE_DEFAULT')
            elif action == "c2g_od_graph":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='od_graph')
            elif action == "c2g_dual":
                bpy.ops.scigraphs.create_street_dual_graph()
            elif action == "c2g_bridge_amenities":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='bridge_amenities')
            elif action == "c2g_metapath_wizard":
                bpy.ops.scigraphs.compute_metapaths_wizard('INVOKE_DEFAULT')
            elif action == "c2g_weighted_metapaths":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='weighted_metapaths')
            elif action == "c2g_mesh":
                bpy.ops.scigraphs.convert_metapaths_to_mesh('INVOKE_DEFAULT')
            elif action == "c2g_graph_tools":
                bpy.ops.scigraphs.city2graph_quick_action_dialog('INVOKE_DEFAULT', action='graph_tools')
            elif action == "c2g_export":
                bpy.ops.scigraphs.c2g_export_graph('INVOKE_DEFAULT')
            else:
                self.report({'WARNING'}, f"Unknown SciGraphs action: {action}")
                return {'CANCELLED'}
        except RuntimeError as exc:
            self.report({'WARNING'}, str(exc))
            return {'CANCELLED'}

        return {'FINISHED'}


class SCIGRAPHS_GGT_quick_toolbar(bpy.types.GizmoGroup):
    """Floating horizontal SciGraphs action toolbar."""
    bl_idname = "SCIGRAPHS_GGT_quick_toolbar"
    bl_label = "SciGraphs Quick Toolbar"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'PERSISTENT', 'SCALE'}

    @classmethod
    def poll(cls, context):
        return bool(getattr(context.window_manager, "scigraphs_show_quick_toolbar", True))

    def setup(self, context):
        self._buttons = []
        self._action_buttons = {}

        move_button = self.gizmos.new("GIZMO_GT_button_2d")
        move_button.icon = 'VIEW_PAN'
        move_button.color = (0.08, 0.18, 0.28)
        move_button.color_highlight = (0.18, 0.42, 0.62)
        move_button.alpha = 0.55
        move_button.alpha_highlight = 0.9
        move_button.scale_basis = 13
        move_button.target_set_operator("scigraphs.drag_quick_toolbar")
        self._buttons.append(move_button)

        for action, _label, category, _description, icon in _ACTION_DEFINITIONS:
            button = self.gizmos.new("GIZMO_GT_button_2d")
            button.icon = icon
            button.color = self._color_for_category(category)
            button.color_highlight = (0.15, 0.55, 0.85)
            button.alpha = 0.62
            button.alpha_highlight = 0.95
            button.scale_basis = 13
            op_props = button.target_set_operator("scigraphs.quick_toolbar_action")
            op_props.action = action
            self._action_buttons[action] = button

        close_button = self.gizmos.new("GIZMO_GT_button_2d")
        close_button.icon = 'PANEL_CLOSE'
        close_button.color = (0.32, 0.08, 0.08)
        close_button.color_highlight = (0.72, 0.18, 0.18)
        close_button.alpha = 0.58
        close_button.alpha_highlight = 0.95
        close_button.scale_basis = 12
        close_button.target_set_operator("scigraphs.toggle_quick_toolbar")
        self._buttons.append(close_button)

    def draw_prepare(self, context):
        current_actions = _current_actions(context)
        x, y, spacing = _toolbar_origin(context, len(current_actions) + len(self._buttons))

        visible_buttons = [self._buttons[0]]
        visible_buttons.extend(self._action_buttons[action] for action in current_actions)
        visible_buttons.append(self._buttons[-1])

        hidden_pos = Matrix.Translation((-10000, -10000, 0))
        for button in self._action_buttons.values():
            button.matrix_basis = hidden_pos

        for index, button in enumerate(visible_buttons):
            button.matrix_basis = Matrix.Translation((x + index * spacing, y, 0))

        tooltip = None
        if getattr(visible_buttons[0], "is_highlight", False):
            tooltip = ("Move Toolbar", "Drag to reposition the floating SciGraphs toolbar.", "TOOLBAR", x, y)
        elif getattr(visible_buttons[-1], "is_highlight", False):
            close_x = x + (len(visible_buttons) - 1) * spacing
            tooltip = ("Hide Toolbar", "Hide the floating toolbar. You can re-enable it from the SciGraphs viewport menu.", "TOOLBAR", close_x, y)
        else:
            for action_index, action in enumerate(current_actions, start=1):
                button = self._action_buttons[action]
                if getattr(button, "is_highlight", False):
                    info = _ACTION_INFO.get(action)
                    if info:
                        button_x = x + action_index * spacing
                        tooltip = (
                            info["label"],
                            info["description"],
                            info["category"],
                            button_x,
                            y,
                        )
                    break

        if tooltip is None:
            _clear_toolbar_tooltip()
        else:
            title, description, category, button_x, button_y = tooltip
            _set_toolbar_tooltip(context, title, description, category, button_x, button_y)

    @staticmethod
    def _color_for_category(category):
        colors = {
            "SCIGRAPHS": (0.05, 0.22, 0.30),
            "LAYOUT": (0.10, 0.18, 0.36),
            "ALGORITHMS": (0.16, 0.12, 0.28),
            "ANALYSIS": (0.18, 0.12, 0.34),
            "TOPOLOGY": (0.20, 0.18, 0.08),
            "VISUAL": (0.10, 0.26, 0.18),
            "TOOLS": (0.18, 0.18, 0.20),
            "DATA": (0.25, 0.15, 0.07),
            "OSMNX": (0.04, 0.24, 0.22),
            "ROUTING": (0.12, 0.20, 0.32),
            "ACCESS": (0.10, 0.22, 0.30),
            "TERRAIN": (0.18, 0.24, 0.10),
            "CITY2GRAPH": (0.22, 0.12, 0.28),
            "MORPHOLOGY": (0.28, 0.14, 0.10),
            "PROXIMITY": (0.16, 0.20, 0.30),
            "METAPATH": (0.24, 0.10, 0.18),
            "TRANSPORT": (0.20, 0.16, 0.08),
            "MOBILITY": (0.22, 0.18, 0.06),
        }
        return colors.get(category, (0.12, 0.12, 0.12))


_CLASSES = [
    SCIGRAPHS_OT_enable_node_gizmos,
    SCIGRAPHS_OT_toggle_quick_toolbar,
    SCIGRAPHS_OT_set_quick_toolbar,
    SCIGRAPHS_OT_quick_import_dialog,
    SCIGRAPHS_OT_osmnx_quick_import_dialog,
    SCIGRAPHS_OT_osmnx_quick_action_dialog,
    SCIGRAPHS_OT_city2graph_quick_action_dialog,
    SCIGRAPHS_OT_panel_quick_dialog,
    SCIGRAPHS_OT_drag_quick_toolbar,
    SCIGRAPHS_OT_quick_toolbar_action,
    SCIGRAPHS_GGT_quick_toolbar,
]


def register():
    bpy.types.WindowManager.scigraphs_show_quick_toolbar = bpy.props.BoolProperty(
        name="Show SciGraphs Quick Toolbar",
        description="Show the floating horizontal SciGraphs toolbar in the 3D View",
        default=True,
    )
    bpy.types.WindowManager.scigraphs_active_quick_toolbar = bpy.props.EnumProperty(
        name="SciGraphs Active Quick Toolbar",
        description="Current contextual profile for the floating SciGraphs toolbar",
        items=[
            (_TOOLBAR_SCIGRAPHS, "SciGraphs", ""),
            (_TOOLBAR_OSMNX, "OSMnx", ""),
            (_TOOLBAR_CITY2GRAPH, "City2Graph", ""),
        ],
        default=_TOOLBAR_SCIGRAPHS,
    )
    bpy.types.WindowManager.scigraphs_quick_toolbar_x = bpy.props.IntProperty(
        name="SciGraphs Toolbar X",
        description="Horizontal position for the floating SciGraphs toolbar",
        default=-1,
    )
    bpy.types.WindowManager.scigraphs_quick_toolbar_y = bpy.props.IntProperty(
        name="SciGraphs Toolbar Y",
        description="Vertical position for the floating SciGraphs toolbar",
        default=-1,
    )

    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    enable_toolbar_tooltips()


def unregister():
    disable_toolbar_tooltips()
    disable_gizmos()
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)

    del bpy.types.WindowManager.scigraphs_quick_toolbar_y
    del bpy.types.WindowManager.scigraphs_quick_toolbar_x
    del bpy.types.WindowManager.scigraphs_active_quick_toolbar
    del bpy.types.WindowManager.scigraphs_show_quick_toolbar
