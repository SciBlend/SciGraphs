import bpy


class SCIGRAPHS_PT_c2g_data(bpy.types.Panel):
    """Data import panel."""
    bl_label = "Data Import"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph
        scene_props = context.scene.scigraphs

        layout.use_property_split = True
        layout.use_property_decorate = False

        # --- API key status (compact, always on top) ---
        from ....preferences import get_preferences
        prefs = get_preferences()
        api_key_status = "DEMO" if (
            prefs and prefs.overture_api_key == "DEMO-API-KEY"
        ) else "Set"
        info = layout.box()
        info.scale_y = 0.7
        row = info.row()
        row.label(
            text=f"Overture API Key: {api_key_status}",
            icon='KEY_HLT' if api_key_status == "Set" else 'KEY_DEHLT',
        )
        if api_key_status == "DEMO":
            row.operator(
                "screen.userpref_show", text="", icon='PREFERENCES'
            ).section = 'ADDONS'

        # --- 1 · Define Area (mirror of the OSMnx download methods) ---
        box = layout.box()
        box.label(text="1 · Define Area", icon='WORLD')
        box.prop(props, "c2g_area_method", text="Method")

        method = props.c2g_area_method
        active = context.active_object
        active_is_osmnx = bool(active and active.get("is_osmnx"))

        sub = box.box()
        if method == 'FROM_OSMNX':
            if active_is_osmnx:
                sub.label(
                    text=f"Using OSMnx graph: {active.name}",
                    icon='CHECKMARK',
                )
                sub.scale_y = 0.85
            else:
                sub.alert = True
                sub.label(
                    text="No OSMnx graph selected",
                    icon='ERROR',
                )
                sub.label(text="Activate one in the viewport, or pick another method.")
        elif method == 'PLACE':
            sub.label(text="Place Name:", icon='VIEWZOOM')
            sub.prop(scene_props, "osmnx_place_name", text="")
            sub.prop(scene_props, "osmnx_which_result", text="Geocoder Result #")
        elif method == 'POINT':
            col = sub.column(align=True)
            col.prop(scene_props, "osmnx_latitude", text="Latitude")
            col.prop(scene_props, "osmnx_longitude", text="Longitude")
            col.prop(scene_props, "osmnx_distance", text="Radius (m)")
        elif method == 'ADDRESS':
            sub.prop(scene_props, "osmnx_address", text="")
            sub.prop(scene_props, "osmnx_distance", text="Radius (m)")
        elif method == 'BBOX':
            col = sub.column(align=True)
            col.prop(scene_props, "osmnx_bbox_north", text="North")
            col.prop(scene_props, "osmnx_bbox_south", text="South")
            col.prop(scene_props, "osmnx_bbox_east", text="East")
            col.prop(scene_props, "osmnx_bbox_west", text="West")
        elif method == 'POLYGON':
            sub.prop_search(
                scene_props, "osmnx_polygon_object",
                bpy.data, "objects", text="Mesh",
            )

        # If a graph is active but the user picked another method, hint
        # that the graph projection will still be used to align outputs.
        if method != 'FROM_OSMNX' and active_is_osmnx:
            tip = box.box()
            tip.scale_y = 0.7
            tip.label(
                text=f"Outputs will be aligned with '{active.name}'",
                icon='INFO',
            )

        # --- 2 · Feature Types ---
        layout.separator()
        box = layout.box()
        box.label(text="2 · Feature Types", icon='OUTLINER_OB_MESH')
        col = box.column(align=True)
        col.prop(props, "c2g_overture_building")
        col.prop(props, "c2g_overture_segment")
        col.prop(props, "c2g_overture_place")
        col.prop(props, "c2g_overture_water")
        col.prop(props, "c2g_overture_land")

        # Source badge: Overture REST for buildings/places, OSMnx for
        # the rest (matches what the c2g notebooks recommend).
        src = box.box()
        src.scale_y = 0.7
        src.label(text="Sources:", icon='INFO')
        src.label(text="• Buildings, Places → Overture Maps")
        src.label(text="• Segments, Water, Land → OSMnx (Overpass)")

        col2 = box.column(align=True)
        col2.prop(props, "c2g_overture_limit", text="Max Features / Type")
        if props.c2g_overture_limit > 15000:
            warn = col2.row()
            warn.scale_y = 0.85
            warn.alert = True
            warn.label(
                text="High limits may be slow or capped server-side",
                icon='ERROR',
            )

        # --- 3 · Download ---
        layout.separator()
        box = layout.box()
        box.label(text="3 · Download", icon='IMPORT')

        col = box.column(align=True)
        col.scale_y = 1.3
        row = col.row(align=True)
        row.operator(
            "scigraphs.c2g_load_overture",
            icon='MESH_UVSPHERE', text="Get as Polygons",
        )
        row.operator(
            "scigraphs.c2g_load_overture_points",
            icon='LIGHTPROBE_SPHERE', text="Get as Points",
        )

        col.separator()
        col.scale_y = 1.0
        col.operator(
            "scigraphs.c2g_convert_to_centroids",
            icon='SNAP_VERTEX', text="Convert Selected to Centroids",
        )

        # --- Local file fallback (kept, less prominent) ---
        layout.separator()
        box = layout.box()
        box.label(text="Local File (GeoJSON / Shapefile)", icon='FILE')
        box.operator("scigraphs.c2g_load_file", icon='IMPORT', text="Load from File")

        # --- Process Segments ---
        layout.separator()
        box = layout.box()
        box.label(text="Process Segments", icon='MOD_EDGESPLIT')
        info_box = box.box()
        info_box.scale_y = 0.7
        info_box.label(text="Select segment object; optionally also select connectors")
        box.operator("scigraphs.c2g_process_segments", icon='PLAY')

        # --- GTFS ---
        layout.separator()
        box = layout.box()
        box.label(text="GTFS Transit Data", icon='ANIM')
        box.prop(props, "c2g_gtfs_path", text="")
        col = box.column(align=True)
        col.operator("scigraphs.c2g_import_gtfs", icon='IMPORT', text="Import GTFS")

        if context.scene.get("c2g_gtfs_loaded") or "c2g_gtfs_data" in context.scene:
            col.separator()
            loaded = col.box()
            loaded.scale_y = 0.7
            tables = list(context.scene.get("c2g_gtfs_tables", []) or [])
            label = (
                f"Loaded · tables: {len(tables)}"
                if tables else "Loaded"
            )
            loaded.label(text=label, icon='CHECKMARK')

            col.prop(props, "c2g_gtfs_create_stops")
            col.prop(props, "c2g_gtfs_create_routes")
            col.prop(props, "c2g_gtfs_stop_size")
            col.separator()
            col.operator("scigraphs.c2g_visualize_gtfs", icon='VIEW3D')


class SCIGRAPHS_PT_c2g_morphology(bpy.types.Panel):
    """Urban morphology panel."""
    bl_label = "Urban Morphology"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph

        layout.use_property_split = True
        layout.use_property_decorate = False

        # --- Segments to Graph ---
        box = layout.box()
        box.label(text="Segments → Graph", icon='GRAPH')

        info_box = box.box()
        info_box.scale_y = 0.7
        info_box.label(text="Select processed segments object")

        box.operator("scigraphs.c2g_segments_to_graph", icon='OUTLINER_OB_CURVE')

        # --- Tessellation ---
        layout.separator()
        box = layout.box()
        box.label(text="Tessellation", icon='MESH_GRID')

        box.prop(props, "c2g_tessellation_shrink")
        box.prop(props, "c2g_tessellation_segment")

        box.separator()
        box.label(text="Select buildings (+ optional barriers)")
        box.operator("scigraphs.c2g_generate_tessellation", icon='MOD_TRIANGULATE')

        # --- Morphological Graph ---
        layout.separator()
        box = layout.box()
        box.label(text="Morphological Graph", icon='OUTLINER_OB_CURVE')

        col = box.column(align=True)
        col.prop(props, "morpho_use_center_from_osmnx")

        if not props.morpho_use_center_from_osmnx:
            col.prop(props, "morpho_center_lat")
            col.prop(props, "morpho_center_lon")

        col.separator()
        col.prop(props, "morpho_distance")
        col.prop(props, "morpho_clipping_buffer")
        col.prop(props, "morpho_contiguity")

        col.separator()
        col.prop(props, "morpho_keep_buildings")
        col.prop(props, "morpho_keep_segments")

        # --- Relation toggles ---
        col.separator()
        rel = box.box()
        rel.label(text="Relations to include", icon='LINKED')
        rcol = rel.column(align=True)
        rcol.prop(props, "morpho_rel_priv_priv")
        rcol.prop(props, "morpho_rel_pub_pub")
        rcol.prop(props, "morpho_rel_priv_pub")

        all_on = (
            props.morpho_rel_priv_priv
            and props.morpho_rel_pub_pub
            and props.morpho_rel_priv_pub
        )
        none_on = not (
            props.morpho_rel_priv_priv
            or props.morpho_rel_pub_pub
            or props.morpho_rel_priv_pub
        )
        hint = rel.box()
        hint.scale_y = 0.7
        if none_on:
            hint.alert = True
            hint.label(text="Enable at least one relation", icon='ERROR')
        elif all_on:
            hint.label(
                text="All relations: single heterogeneous graph",
                icon='INFO',
            )
        else:
            hint.label(
                text="Subset: one Blender object per active relation",
                icon='INFO',
            )

        box.separator()
        box.label(text="Select buildings + street network")
        box.operator("scigraphs.c2g_morphological_graph", icon='FORCE_MAGNETIC')


class SCIGRAPHS_PT_c2g_transport(bpy.types.Panel):
    """Transportation analysis panel."""
    bl_label = "Transportation"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph

        layout.use_property_split = True
        layout.use_property_decorate = False

        if not (context.scene.get("c2g_gtfs_loaded") or "c2g_gtfs_data" in context.scene):
            box = layout.box()
            box.label(text="No GTFS data loaded", icon='INFO')
            box.label(text="Import GTFS in the Data Import panel first")
            return

        # --- Travel Summary ---
        box = layout.box()
        box.label(text="Travel Summary Graph", icon='OUTLINER_OB_CURVE')

        col = box.column(align=True)
        col.prop(props, "gtfs_calendar_start", text="Calendar Start")
        col.prop(props, "gtfs_calendar_end", text="Calendar End")

        info = box.box()
        info.scale_y = 0.7
        info.label(text="Leave dates empty to use all calendar data", icon='INFO')

        box.separator()
        box.operator("scigraphs.c2g_travel_summary_graph", icon='OUTLINER_OB_CURVE')

        # --- OD Pairs ---
        layout.separator()
        box = layout.box()
        box.label(text="GTFS OD Graph", icon='TRACKING')

        info = box.box()
        info.scale_y = 0.7
        info.label(
            text="Aggregates trip legs into a SciGraphs graph",
            icon='INFO',
        )

        col = box.column(align=True)
        col.prop(props, "gtfs_od_directed")
        col.prop(props, "gtfs_od_top_n")
        if props.gtfs_od_top_n == 0:
            warn = box.row()
            warn.alert = True
            warn.scale_y = 0.85
            warn.label(
                text="No limit set — large feeds may exhaust memory",
                icon='ERROR',
            )

        box.separator()
        box.operator("scigraphs.c2g_get_od_pairs", icon='EXPORT')


class SCIGRAPHS_PT_c2g_mobility(bpy.types.Panel):
    """Mobility / OD Matrix panel."""
    bl_label = "Mobility (OD Matrix)"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph

        layout.use_property_split = True
        layout.use_property_decorate = False

        # --- Load OD Matrix ---
        box = layout.box()
        box.label(text="Load OD Matrix", icon='FILE_FOLDER')

        box.prop(props, "od_matrix_path", text="")
        box.operator("scigraphs.c2g_load_od_matrix", icon='IMPORT')

        if "c2g_od_data" in context.scene:
            loaded = box.box()
            loaded.scale_y = 0.7
            path = context.scene.get("c2g_od_path", "")
            loaded.label(text=f"Loaded: {path.split('/')[-1] if path else 'Yes'}", icon='CHECKMARK')

        # --- OD to Graph ---
        layout.separator()
        box = layout.box()
        box.label(text="OD to Graph", icon='OUTLINER_OB_CURVE')

        col = box.column(align=True)
        col.prop(props, "od_zones_object", text="Zones")
        col.prop(props, "od_zone_id_col")
        col.prop(props, "od_matrix_type")

        col.separator()
        col.prop(props, "od_source_col")
        col.prop(props, "od_target_col")
        col.prop(props, "od_weight_col")

        col.separator()
        col.prop(props, "od_threshold")
        col.prop(props, "od_directed")

        box.separator()
        box.operator("scigraphs.c2g_od_to_graph", icon='PLAY')


class SCIGRAPHS_PT_c2g_graph_tools(bpy.types.Panel):
    """Graph utility tools panel."""
    bl_label = "Graph Tools"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph

        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Graph Utilities", icon='MODIFIER')

        box.prop(props, "graph_tool_action", text="Tool")

        action = props.graph_tool_action

        if action == 'FILTER':
            col = box.column(align=True)
            col.prop(props, "graph_filter_center", text="Center")
            col.prop(props, "graph_filter_threshold")

        elif action == 'CLIP':
            col = box.column(align=True)
            col.prop(props, "graph_filter_center", text="Clip Polygon")

        elif action == 'ISOCHRONE':
            col = box.column(align=True)
            col.prop(props, "isochrone_center_object", text="Center")
            col.prop(props, "isochrone_threshold")
            col.prop(props, "isochrone_weight_attr")

        box.separator()

        info = box.box()
        info.scale_y = 0.7
        if action == 'REMOVE_ISOLATED':
            info.label(text="Keep only the largest connected component", icon='INFO')
        elif action == 'FILTER':
            info.label(text="Remove nodes beyond distance from center", icon='INFO')
        elif action == 'CLIP':
            info.label(text="Clip graph to a polygon boundary object", icon='INFO')
        elif action == 'ISOCHRONE':
            info.label(text="Generate reachability polygon from center", icon='INFO')

        box.separator()
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.c2g_graph_tool_apply", icon='PLAY', text="Apply")


class SCIGRAPHS_PT_c2g_export(bpy.types.Panel):
    """Export panel."""
    bl_label = "GeoAI Export"
    bl_parent_id = "SCIGRAPHS_PT_city2graph_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'City2Graph'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.city2graph

        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Graph Export for GNN", icon='EXPORT')

        box.prop(props, "c2g_export_format", text="Format")
        box.prop(props, "c2g_graph_type", text="Type")
        box.prop(props, "c2g_export_path", text="")

        box.separator()
        box.label(text="Select graph object to export")
        box.operator("scigraphs.c2g_export_graph", icon='EXPORT')

        layout.separator()

        box = layout.box()
        box.label(text="PyTorch Geometric", icon='FILE_SCRIPT')
        box.label(text="External loader script for PyG")
        box.operator("scigraphs.c2g_save_loader_script", icon='FILE_NEW', text="Save Loader Script")


classes = [
    SCIGRAPHS_PT_c2g_data,
    SCIGRAPHS_PT_c2g_morphology,
    SCIGRAPHS_PT_c2g_transport,
    SCIGRAPHS_PT_c2g_mobility,
    SCIGRAPHS_PT_c2g_graph_tools,
    SCIGRAPHS_PT_c2g_export,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
