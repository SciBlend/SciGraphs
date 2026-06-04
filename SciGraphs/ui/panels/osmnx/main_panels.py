"""OSMnx panels — organised into Download → Graph Ops → Attributes → Routing →
Accessibility → Statistics → Features → Elevation → IO/Export."""

import bpy
import os

from ..feature_selector import draw_feature_selector


def _is_osmnx_obj(obj):
    """True if the given object is an imported OSMnx network."""
    return bool(obj and obj.get("is_osmnx", False))


def _draw_no_graph_hint(layout, msg="Import an OSMnx graph first"):
    """Render a small "select a graph" hint inside a panel body."""
    box = layout.box()
    box.scale_y = 0.8
    box.label(text=msg, icon='INFO')
    box.label(text="Open the Download subpanel above.")


# ---------------------------------------------------------------------------
# Root panel
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_main(bpy.types.Panel):
    """OSMnx main panel - root for all OSMnx subpanels."""
    bl_label = "OSMnx"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'

    def draw(self, context):
        from ... import gizmos
        gizmos.set_active_toolbar(context, 'OSMNX')

        layout = self.layout
        obj = context.active_object

        box = layout.box()
        box.label(text="OpenStreetMap Network Analysis", icon='WORLD')

        if obj and obj.get("is_osmnx", False):
            info = box.box()
            info.scale_y = 0.8
            info.label(text=f"Network: {obj.name}", icon='MESH_DATA')

            nodes = obj.get("num_nodes", 0)
            edges = obj.get("num_edges", 0)
            info.label(text=f"Nodes: {nodes:,} | Edges: {edges:,}")

            query_name = obj.get("osmnx_query_name") or obj.get("osmnx_place")
            if query_name:
                info.label(text=f"Place: {query_name}")
            if obj.get("osmnx_projected", False):
                info.label(text=f"CRS: {obj.get('osmnx_crs', 'projected')}", icon='ORIENTATION_GLOBAL')
        else:
            info = box.box()
            info.scale_y = 0.8
            info.label(text="No OSMnx network selected", icon='INFO')
            info.label(text="Open 'Download' subpanel to import one")


# ---------------------------------------------------------------------------
# 1. Download  (renamed Import/Export split into Download + IO/Export)
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_download(bpy.types.Panel):
    bl_label = "Download"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_order = 0

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Import street networks from OSM", icon='WORLD')

        layout.prop(props, "osmnx_download_method")

        box = layout.box()
        method = props.osmnx_download_method

        if method == 'PLACE':
            box.label(text="Place Name:", icon='VIEWZOOM')
            box.prop(props, "osmnx_place_name", text="")
            box.prop(props, "osmnx_which_result", text="Geocoder Result #")
            info = box.box()
            info.scale_y = 0.7
            info.label(text="Examples: 'Madrid, Spain'")
            info.label(text="'Manhattan, New York, USA'")

        elif method == 'MULTI_PLACE':
            box.label(text="Multiple Places (one per line):", icon='PRESET')
            box.prop(props, "osmnx_place_list", text="")
            info = box.box()
            info.scale_y = 0.7
            info.label(text="Each line = a separate place, unioned into one graph")

        elif method == 'POINT':
            box.label(text="Center Coordinates:", icon='PIVOT_CURSOR')
            col = box.column(align=True)
            col.prop(props, "osmnx_latitude", text="Latitude")
            col.prop(props, "osmnx_longitude", text="Longitude")
            box.prop(props, "osmnx_distance", text="Radius (m)")

        elif method == 'ADDRESS':
            box.label(text="Postal Address:", icon='HOME')
            box.prop(props, "osmnx_address", text="")
            box.prop(props, "osmnx_distance", text="Radius (m)")

        elif method == 'BBOX':
            box.label(text="Bounding Box:", icon='MESH_PLANE')
            col = box.column(align=True)
            col.prop(props, "osmnx_bbox_north", text="North")
            col.prop(props, "osmnx_bbox_south", text="South")
            col.prop(props, "osmnx_bbox_east", text="East")
            col.prop(props, "osmnx_bbox_west", text="West")

        elif method == 'POLYGON':
            box.label(text="Boundary Polygon Object:", icon='MESH_DATA')
            box.prop_search(props, "osmnx_polygon_object", bpy.data, "objects", text="Mesh")
            info = box.box()
            info.scale_y = 0.7
            info.label(text="Mesh vertices treated as (lon, lat).")

        elif method == 'XML':
            box.label(text="Local .osm XML File:", icon='FILE')
            box.prop(props, "osmnx_xml_filepath", text="")

        # Filtering / network type
        layout.separator()
        box = layout.box()
        box.label(text="Network Type & Filter", icon='FILTER')
        col = box.column(align=True)

        col.prop(props, "osmnx_custom_filter_preset")
        if props.osmnx_custom_filter_preset == 'NONE':
            col.prop(props, "osmnx_network_type")
        else:
            info = col.box()
            info.scale_y = 0.7
            info.label(text="Network Type is ignored (preset active)")
        col.prop(props, "osmnx_custom_filter_text", text="Advanced Filter")

        # Options
        layout.separator()
        box = layout.box()
        box.label(text="Options:", icon='PREFERENCES')
        col = box.column(align=True)
        col.prop(props, "osmnx_simplify", text="Simplify")
        col.prop(props, "osmnx_retain_geometry", text="Keep Curved Geometry")
        col.prop(props, "osmnx_truncate_by_edge")
        col.prop(props, "osmnx_retain_all")
        col.prop(props, "osmnx_scale", text="Scale Factor")

        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("scigraphs.import_osm_graph", text="Import from OSM", icon='IMPORT')


# ---------------------------------------------------------------------------
# 2. Graph Operations  (projection, conversion, simplification)
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_graph_ops(bpy.types.Panel):
    bl_label = "Graph Operations"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 1

    def draw(self, context):
        layout = self.layout
        obj = context.active_object
        if not _is_osmnx_obj(obj):
            _draw_no_graph_hint(layout)
            return
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Projection", icon='ORIENTATION_GLOBAL')
        if obj.get("osmnx_projected", False):
            box.label(text=f"Current: {obj.get('osmnx_crs', 'Unknown')}", icon='CHECKMARK')
        else:
            box.label(text="Not projected (WGS84)", icon='ERROR')
        box.operator("scigraphs.osmnx_project_graph", text="Project to CRS", icon='FILE_REFRESH')

        layout.separator()
        box = layout.box()
        box.label(text="Graph Type Conversion", icon='ARROW_LEFTRIGHT')
        col = box.column(align=True)
        col.operator("scigraphs.osmnx_to_undirected", text="To Undirected", icon='DRIVER')
        col.operator("scigraphs.osmnx_to_digraph", text="To DiGraph", icon='FORCE_MAGNETIC')

        layout.separator()
        box = layout.box()
        box.label(text="GeoDataFrame Conversion", icon='MESH_DATA')
        col = box.column(align=True)
        col.operator("scigraphs.osmnx_graph_to_gdfs", text="Graph to GeoDataFrames", icon='EXPORT')
        col.operator("scigraphs.osmnx_gdfs_to_graph", text="GeoDataFrames to Graph", icon='IMPORT')

        layout.separator()
        box = layout.box()
        box.label(text="Simplification", icon='MOD_SIMPLIFY')
        col = box.column(align=True)
        col.operator("scigraphs.osmnx_simplify", text="Simplify Graph", icon='MESH_DATA')
        props = context.scene.scigraphs
        col.separator()
        col.prop(props, "osmnx_simplification_tolerance", text="Tolerance (m)")
        col.operator("scigraphs.osmnx_consolidate", text="Consolidate Intersections", icon='AUTOMERGE_ON')

        layout.separator()
        box = layout.box()
        box.label(text="Truncation", icon='MOD_MASK')
        col = box.column(align=True)
        col.operator("scigraphs.osmnx_truncate_bbox", text="Truncate to BBox", icon='MESH_PLANE')
        col.operator("scigraphs.osmnx_truncate_polygon", text="Truncate to Polygon", icon='MESH_CIRCLE')
        col.operator("scigraphs.osmnx_truncate_distance", text="Truncate by Distance", icon='DRIVER_DISTANCE')
        col.separator()
        col.operator("scigraphs.osmnx_largest_component", text="Keep Largest Component", icon='STICKY_UVS_LOC')


# ---------------------------------------------------------------------------
# 3. Attributes  (lengths, speeds, bearings, travel times)
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_attributes(bpy.types.Panel):
    bl_label = "Network Attributes"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 2

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        if not _is_osmnx_obj(obj):
            _draw_no_graph_hint(layout)
            return

        box = layout.box()
        box.label(text="Edge Metrics", icon='DRIVER_DISTANCE')
        col = box.column(align=True)

        row = col.row(align=True)
        if obj.get("osmnx_has_lengths", False):
            row.label(text="Lengths", icon='CHECKMARK')
        else:
            row.operator("scigraphs.osmnx_add_edge_lengths", text="Add Lengths", icon='DRIVER_DISTANCE')

        row = col.row(align=True)
        if obj.get("osmnx_has_bearings", False):
            row.label(text="Bearings", icon='CHECKMARK')
            entropy = obj.get("osmnx_bearing_entropy", 0)
            col.label(text=f"  Entropy: {entropy:.2f}")
        else:
            row.operator("scigraphs.osmnx_add_edge_bearings", text="Add Bearings", icon='ORIENTATION_CURSOR')

        layout.separator()
        box = layout.box()
        box.label(text="Speed & Travel Time", icon='TIME')
        col = box.column(align=True)
        col.prop(props, "osmnx_fallback_speed", text="Fallback Speed (km/h)")

        row = col.row(align=True)
        has_speeds = obj.get("osmnx_has_speeds", False)
        speeds_label = "Recalculate Speeds" if has_speeds else "Add Speeds"
        row.operator("scigraphs.osmnx_add_edge_speeds", text=speeds_label, icon='PLAY')
        if has_speeds:
            row.label(text="", icon='CHECKMARK')

        row = col.row(align=True)
        has_tt = obj.get("osmnx_has_travel_times", False)
        tt_label = "Recalculate Travel Times" if has_tt else "Add Travel Times"
        row.operator("scigraphs.osmnx_add_travel_times", text=tt_label, icon='PREVIEW_RANGE')
        row.enabled = has_speeds
        if has_tt:
            row.label(text="", icon='CHECKMARK')


# ---------------------------------------------------------------------------
# 4. Routing  (shortest / k-shortest / batch / summary)
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_routing(bpy.types.Panel):
    bl_label = "Routing & Paths"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 3

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        if not _is_osmnx_obj(obj):
            _draw_no_graph_hint(layout)
            return

        # Source / target
        box = layout.box()
        box.label(text="Source & Target Nodes", icon='CON_FOLLOWPATH')

        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(props, "osmnx_shortest_path_source", text="Source")
        row.operator("scigraphs.osmnx_select_path_source", text="", icon='EYEDROPPER')
        row = col.row(align=True)
        row.prop(props, "osmnx_shortest_path_target", text="Target")
        row.operator("scigraphs.osmnx_select_path_target", text="", icon='EYEDROPPER')

        col.separator()
        col.prop(props, "osmnx_path_weight", text="Minimize")
        if props.osmnx_path_weight == 'elevation_impedance':
            col.prop(props, "osmnx_impedance_alpha", text="Alpha")

        # Single shortest path
        layout.separator()
        box = layout.box()
        box.label(text="Shortest Path", icon='PLAY')
        row = box.row()
        row.scale_y = 1.3
        row.operator("scigraphs.osmnx_shortest_path", text="Calculate Path")

        if "osmnx_path_distance_km" in obj:
            result = box.box()
            result.scale_y = 0.8
            result.label(text="Last Path Result:", icon='INFO')
            result.label(text=f"  Distance: {obj.get('osmnx_path_distance_km', 0):.3f} km")
            result.label(text=f"  Nodes: {obj.get('osmnx_path_num_nodes', 0)}")
            result.label(text=f"  Edges: {obj.get('osmnx_path_num_edges', 0)}")
            if "osmnx_path_travel_time_min" in obj:
                result.label(text=f"  Time: {obj.get('osmnx_path_travel_time_min', 0):.1f} min")

            # Route summary + elevation profile
            row = box.row(align=True)
            row.operator("scigraphs.osmnx_route_summary", text="Summarize", icon='GRAPH')
            row.operator("scigraphs.osmnx_route_elev_profile", text="Elevation Profile", icon='CURVE_BEZCURVE')

            if "osmnx_route_rise_m" in obj:
                result2 = box.box()
                result2.scale_y = 0.8
                result2.label(text=f"  Rise: {obj.get('osmnx_route_rise_m', 0):.0f} m")
                result2.label(text=f"  |grade|: {obj.get('osmnx_route_mean_grade_abs', 0) * 100:.1f}%")

        # K-shortest
        layout.separator()
        box = layout.box()
        box.label(text="K Alternative Routes", icon='PRESET')
        col = box.column(align=True)
        col.prop(props, "osmnx_k_shortest", text="K")
        col.operator("scigraphs.osmnx_k_shortest", text="Compute K Alternatives", icon='PLAY')

        # Many-to-many batch
        layout.separator()
        box = layout.box()
        box.label(text="Batch (Many-to-Many)", icon='STICKY_UVS_LOC')
        col = box.column(align=True)
        col.prop(props, "osmnx_od_random_n", text="Random OD Pairs")
        col.prop(props, "osmnx_od_batch_cpus", text="CPU Cores (0=auto)")
        col.operator("scigraphs.osmnx_batch_routes", text="Run Batch Routing", icon='PLAY')

        if obj.get("osmnx_batch_total"):
            info = box.box()
            info.scale_y = 0.8
            info.label(text=f"Batch: {obj.get('osmnx_batch_reached', 0)}/{obj.get('osmnx_batch_total', 0)} reached")
            info.label(text=f"Mean dist: {obj.get('osmnx_batch_mean_dist_m', 0):.0f} m")


# ---------------------------------------------------------------------------
# 5. Accessibility  (isochrones / ego / DBSCAN)
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_accessibility(bpy.types.Panel):
    bl_label = "Accessibility"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 4

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        if not _is_osmnx_obj(obj):
            _draw_no_graph_hint(layout)
            return

        # Isochrones
        box = layout.box()
        box.label(text="Isochrones", icon='MESH_CIRCLE')
        col = box.column(align=True)
        col.prop(props, "osmnx_iso_center_node", text="Center Node")
        col.prop(props, "osmnx_iso_trip_times", text="Times (min)")
        col.prop(props, "osmnx_iso_travel_speed", text="Speed (km/h)")
        col.prop(props, "osmnx_iso_mode", text="Mode")
        if props.osmnx_iso_mode == 'BUFFER_UNION':
            col.prop(props, "osmnx_iso_buffer", text="Buffer (m)")
        row = col.row()
        row.scale_y = 1.3
        row.operator("scigraphs.osmnx_isochrones", text="Generate Isochrones", icon='PLAY')

        # Ego subgraph
        layout.separator()
        box = layout.box()
        box.label(text="Ego Subgraph", icon='STICKY_UVS_DISABLE')
        box.operator("scigraphs.osmnx_ego_subgraph", text="Extract Ego Subgraph", icon='PLAY')

        # Network DBSCAN
        layout.separator()
        box = layout.box()
        box.label(text="Network-Constrained DBSCAN", icon='STICKY_UVS_LOC')
        box.operator("scigraphs.osmnx_network_dbscan", text="Cluster by Network Distance", icon='PLAY')


# ---------------------------------------------------------------------------
# 6. Statistics  (basic stats + centrality + orientation rose)
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_statistics(bpy.types.Panel):
    bl_label = "Statistics & Centrality"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 5

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        if not _is_osmnx_obj(obj):
            _draw_no_graph_hint(layout)
            return

        # Basic stats
        box = layout.box()
        box.label(text="Basic Stats", icon='INFO')
        row = box.row(align=True)
        row.prop(props, "osmnx_network_area", text="Area (km2)")
        row.operator("scigraphs.osmnx_estimate_area", text="", icon='FILE_REFRESH')
        row = box.row()
        row.scale_y = 1.2
        row.operator("scigraphs.osmnx_basic_stats", text="Compute Statistics", icon='PLAY')

        if obj.get("osmnx_stats_calculated", False):
            info = box.box()
            info.scale_y = 0.8
            col = info.column(align=True)
            col.label(text=f"Nodes: {obj.get('osmnx_stat_n_nodes', 0):,}")
            col.label(text=f"Edges: {obj.get('osmnx_stat_n_edges', 0):,}")
            col.label(text=f"Total Length: {obj.get('osmnx_stat_total_length_km', 0):.2f} km")
            col.label(text=f"Avg Edge: {obj.get('osmnx_stat_avg_edge_length_m', 0):.1f} m")
            col.label(text=f"Avg Degree: {obj.get('osmnx_stat_avg_degree', 0):.2f}")
            circuity = obj.get('osmnx_stat_circuity_avg')
            if circuity:
                col.label(text=f"Circuity: {circuity:.3f}")
            col.label(text=f"Intersections: {obj.get('osmnx_stat_intersection_count', 0):,}")
            col.label(text=f"Dead Ends: {obj.get('osmnx_stat_dead_end_count', 0):,}")
            if obj.get('osmnx_stat_node_density_per_km2'):
                col.separator()
                col.label(text=f"Nodes/km2: {obj.get('osmnx_stat_node_density_per_km2', 0):.1f}")
                col.label(text=f"Edges/km2: {obj.get('osmnx_stat_edge_density_per_km2', 0):.1f}")
                col.label(text=f"Street km/km2: {obj.get('osmnx_stat_street_density_km_per_km2', 0):.2f}")

        # Circuity
        layout.separator()
        box = layout.box()
        box.label(text="Circuity", icon='CURVE_PATH')
        box.operator("scigraphs.osmnx_circuity", text="Calculate Circuity", icon='PLAY')
        if "circuity_avg" in obj:
            info = box.box()
            info.scale_y = 0.8
            info.label(text=f"Average: {obj.get('circuity_avg', 0):.3f}")

        # Centrality
        layout.separator()
        box = layout.box()
        box.label(text="Centrality", icon='PARTICLE_POINT')
        col = box.column(align=True)
        col.prop(props, "osmnx_centrality_kind", text="Kind")
        col.prop(props, "osmnx_centrality_weighted")
        col.prop(props, "osmnx_centrality_fast")
        col.operator("scigraphs.osmnx_centrality", text="Compute Centrality", icon='PLAY')

        # Bearing distribution + orientation rose
        layout.separator()
        box = layout.box()
        box.label(text="Orientation Analysis", icon='ORIENTATION_VIEW')
        col = box.column(align=True)
        col.operator("scigraphs.osmnx_orientation_entropy", text="Orientation Entropy", icon='GRAPH')
        col.operator("scigraphs.osmnx_bearings_distribution", text="Bearing Distribution", icon='DRIVER_ROTATIONAL_DIFFERENCE')
        col.separator()
        col.label(text="Orientation Rose (image):")
        col.prop(props, "osmnx_rose_bins", text="Bins")
        col.operator("scigraphs.osmnx_orientation_rose", text="Render Rose to Image", icon='IMAGE_DATA')

        # Node-pair distance calculator (replaces the manual lat/lon
        # calculator: pick two graph nodes, get straight-line, network,
        # and travel-time distances + circuity).
        layout.separator()
        box = layout.box()
        box.label(text="Node-Pair Distances", icon='DRIVER_DISTANCE')

        col = box.column(align=True)
        row = col.row(align=True)
        row.prop(props, "osmnx_dist_node_a", text="Node A")
        row.operator("scigraphs.osmnx_select_distance_node_a", text="", icon='EYEDROPPER')
        row = col.row(align=True)
        row.prop(props, "osmnx_dist_node_b", text="Node B")
        row.operator("scigraphs.osmnx_select_distance_node_b", text="", icon='EYEDROPPER')

        col.separator()
        run = col.row()
        run.scale_y = 1.2
        run.operator("scigraphs.osmnx_calc_node_pair_distance", text="Compute Distances", icon='PLAY')

        if "osmnx_pair_straight_m" in obj:
            result = box.box()
            result.scale_y = 0.85
            method = obj.get("osmnx_pair_straight_method", "")
            crs_label = obj.get("osmnx_pair_crs", "")
            straight_m = obj.get("osmnx_pair_straight_m", 0.0)
            network_m = obj.get("osmnx_pair_network_m", 0.0)
            circuity = obj.get("osmnx_pair_circuity", 0.0)
            num_nodes = obj.get("osmnx_pair_num_nodes", 0)
            num_edges = obj.get("osmnx_pair_num_edges", 0)
            result.label(text="Last computation:", icon='INFO')
            result.label(text=f"  Method: {method}")
            if crs_label:
                result.label(text=f"  CRS:    {crs_label}")
            result.label(text=f"  Straight: {straight_m:,.1f} m  ({straight_m/1000:.3f} km)")
            result.label(text=f"  Network:  {network_m:,.1f} m  ({network_m/1000:.3f} km)")
            result.label(text=f"  Circuity: {circuity:.3f}  (network / straight)")
            result.label(text=f"  Path:     {num_nodes} nodes, {num_edges} edges")
            if "osmnx_pair_travel_min" in obj:
                tm = obj.get("osmnx_pair_travel_min", 0.0)
                result.label(text=f"  Travel:   {tm:.1f} min")


# ---------------------------------------------------------------------------
# 7. Features (POIs)
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_features(bpy.types.Panel):
    bl_label = "Features & POIs"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 6

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        layout.use_property_split = True
        layout.use_property_decorate = False

        box = layout.box()
        box.label(text="Download OSM Features", icon='MESH_DATA')

        draw_feature_selector(layout, props, title="Feature Source & Tags")

        col = layout.column(align=True)
        col.label(text="Download by:")
        row = col.row(align=True)
        row.scale_y = 1.2
        row.operator("scigraphs.osmnx_features_place", text="Place", icon='WORLD')
        row.operator("scigraphs.osmnx_features_address", text="Address", icon='HOME')
        row = col.row(align=True)
        row.scale_y = 1.2
        row.operator("scigraphs.osmnx_features_point", text="Point", icon='PIVOT_CURSOR')
        row.operator("scigraphs.osmnx_features_bbox", text="BBox", icon='MESH_PLANE')
        row = col.row(align=True)
        row.scale_y = 1.2
        row.operator("scigraphs.osmnx_features_polygon", text="Polygon (active obj)", icon='MESH_CIRCLE')
        row.operator("scigraphs.osmnx_features_xml", text="From XML File", icon='FILE')

        layout.separator()
        box = layout.box()
        box.label(text="Snap POIs to Street Network", icon='EMPTY_SINGLE_ARROW')
        col = box.column(align=True)
        col.prop(props, "osmnx_poi_snap_mode", text="Mode")
        col.operator("scigraphs.osmnx_snap_pois", text="Snap Active POIs", icon='PLAY')

        # Geocoding helpers
        layout.separator()
        box = layout.box()
        box.label(text="Geocoding & Geometry", icon='URL')
        col = box.column(align=True)
        col.operator("scigraphs.osmnx_geocode", text="Geocode Address", icon='VIEWZOOM')
        col.operator("scigraphs.osmnx_geocode_to_gdf", text="Geocode to GeoDataFrame", icon='MESH_DATA')
        col.separator()
        col.operator("scigraphs.osmnx_bbox_from_point", text="BBox from Point", icon='PIVOT_BOUNDBOX')
        col.operator("scigraphs.osmnx_bbox_to_poly", text="BBox to Polygon", icon='MESH_PLANE')
        col.separator()
        col.operator("scigraphs.osmnx_buffer_geometry", text="Buffer Geometry", icon='MOD_OFFSET')
        col.operator("scigraphs.osmnx_interpolate_points", text="Interpolate Points", icon='CURVE_BEZCURVE')
        col.operator("scigraphs.osmnx_sample_points", text="Sample Points on Graph", icon='PARTICLE_POINT')


# ---------------------------------------------------------------------------
# 8. Elevation & Terrain
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_elevation(bpy.types.Panel):
    bl_label = "Elevation & Terrain"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 7

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object
        if not _is_osmnx_obj(obj):
            _draw_no_graph_hint(layout)
            return

        from ....preferences import get_preferences, ADDON_PACKAGE
        prefs = get_preferences()

        # ============================================================
        # 1. Get elevation data — unified pipeline
        # ============================================================
        box = layout.box()
        box.label(text="1 · Get Elevation Data", icon='WORLD')

        col = box.column(align=True)
        col.prop(props, "osmnx_dem_source", text="Source")

        source = props.osmnx_dem_source

        # Source-specific options.
        sub = box.box()
        sub.scale_y = 0.95
        if source == 'OPENTOPOGRAPHY':
            has_key = prefs and prefs.opentopography_api_key
            if not has_key:
                row = sub.row()
                row.alert = True
                row.label(text="API key required", icon='ERROR')
                row.operator("preferences.addon_show", text="Set Key").module = ADDON_PACKAGE
            else:
                sub.label(
                    text=f"Dataset: {prefs.opentopography_default_dataset} (change in preferences)",
                    icon='INFO',
                )
            sub.prop(props, "osmnx_dem_padding", text="Padding")
        elif source == 'OPEN_ELEVATION':
            sub.prop(props, "osmnx_dem_api_provider", text="Provider")
            sub.prop(props, "osmnx_dem_api_resolution", text="Resolution")
            sub.prop(props, "osmnx_dem_api_workers", text="Parallel Workers")
        elif source == 'LOCAL_GEOTIFF':
            sub.prop(props, "osmnx_dem_local_path", text="GeoTIFF")

        # What to do with the DEM.
        do = box.column(align=True)
        do.label(text="Apply to:")
        row = do.row(align=True)
        row.prop(props, "osmnx_dem_apply_to_nodes", toggle=True, icon='SNAP_VERTEX')
        row.prop(props, "osmnx_dem_create_terrain", toggle=True, icon='MESH_GRID')

        # Terrain method only when terrain is requested.
        if props.osmnx_dem_create_terrain:
            method = do.column(align=True)
            method.prop(props, "osmnx_dem_terrain_method", text="Method")
            if props.osmnx_dem_terrain_method == 'DISPLACE':
                method.prop(props, "osmnx_dem_subdivision_levels", text="Subdivisions")
            else:
                method.prop(props, "osmnx_dem_subsample", text="Subsample")

        # Vertical scale + offset apply to both nodes and terrain.
        vs = box.column(align=True)
        vs.prop(props, "osmnx_elevation_scale", text="Vertical Scale")
        vs.prop(props, "osmnx_elevation_offset", text="Base Offset (m)")

        run = box.row()
        run.scale_y = 1.3
        run.enabled = props.osmnx_dem_apply_to_nodes or props.osmnx_dem_create_terrain
        run.operator("scigraphs.osmnx_get_elevation", text="Get Elevation Data", icon='IMPORT')

        # Status badges.
        status = box.column(align=True)
        status.scale_y = 0.85
        if obj.get("osmnx_has_elevation", False):
            elev_min = obj.get("osmnx_elev_min", 0)
            elev_max = obj.get("osmnx_elev_max", 0)
            elev_range = obj.get("osmnx_elev_range", 0)
            status.label(
                text=f"Nodes elevated: {elev_min:.0f}m – {elev_max:.0f}m ({elev_range:.0f}m range)",
                icon='CHECKMARK',
            )
        terrain_name = obj.get("terrain_child") or obj.get("dem_terrain_child")
        if terrain_name and terrain_name in bpy.data.objects:
            terrain_obj = bpy.data.objects[terrain_name]
            status.label(
                text=f"Terrain: {terrain_obj.name} ({len(terrain_obj.data.vertices):,} verts)",
                icon='CHECKMARK',
            )

        # Helper for users who can't / don't want to use any of the built-in
        # sources: export the AOI bbox so they can grab the DEM elsewhere
        # (Copernicus, EarthData, etc.) and re-enter via "Local GeoTIFF".
        helper = box.row()
        helper.scale_y = 0.9
        helper.operator(
            "scigraphs.osmnx_export_aoi",
            text="Export AOI (for Copernicus / external DEM)",
            icon='EXPORT',
        )

        # ============================================================
        # 2. Apply / flatten 3D positions
        # ============================================================
        if obj.get("osmnx_has_elevation", False) or obj.get("osmnx_3d_applied", False):
            layout.separator()
            box = layout.box()
            box.label(text="2 · 3D Positions", icon='ORIENTATION_LOCAL')
            col = box.column(align=True)

            if obj.get("osmnx_has_elevation", False) and not obj.get("osmnx_3d_applied", False):
                col.scale_y = 1.15
                col.operator(
                    "scigraphs.osmnx_apply_elevation_3d",
                    text="Apply 3D Positions",
                    icon='ORIENTATION_LOCAL',
                )

            if obj.get("osmnx_3d_applied", False):
                scale_used = obj.get("osmnx_elev_scale_used", 1.0)
                col.label(text=f"Network in 3D · vertical scale {scale_used}x", icon='CHECKMARK')
                col.operator(
                    "scigraphs.osmnx_flatten_network",
                    text="Flatten to 2D",
                    icon='MESH_PLANE',
                )

        # ============================================================
        # 3. Slope / grade analysis
        # ============================================================
        if obj.get("osmnx_has_elevation", False):
            layout.separator()
            box = layout.box()
            box.label(text="3 · Slope / Grade", icon='TRIA_UP')
            box.operator("scigraphs.osmnx_add_edge_grades", text="Calculate Grades", icon='PLAY')
            if obj.get("osmnx_has_grades", False):
                info = box.box()
                info.scale_y = 0.8
                mean_grade = obj.get("osmnx_grade_mean_abs", 0) * 100
                max_grade = obj.get("osmnx_grade_max_abs", 0) * 100
                steep_pct = obj.get("osmnx_steep_pct", 0)
                info.label(text=f"Mean grade: {mean_grade:.1f}%")
                info.label(text=f"Max grade:  {max_grade:.1f}%")
                info.label(text=f"Steep edges (>5%): {steep_pct:.1f}%")

        # ============================================================
        # 4. Terrain object widget — only if terrain mesh exists
        # ============================================================
        if terrain_name and terrain_name in bpy.data.objects:
            layout.separator()
            terrain_obj = bpy.data.objects[terrain_name]
            tbox = layout.box()
            tbox.label(text="4 · Terrain Object", icon='MESH_GRID')
            visible = not terrain_obj.hide_viewport
            row = tbox.row(align=True)
            row.operator(
                "scigraphs.osmnx_toggle_terrain",
                text="Hide" if visible else "Show",
                icon='HIDE_OFF' if visible else 'HIDE_ON',
            )
            row.operator(
                "scigraphs.osmnx_remove_terrain_child",
                text="Remove",
                icon='X',
            )

        # ============================================================
        # 5. Basemap texture for the terrain mesh (optional)
        # ============================================================
        if terrain_name and terrain_name in bpy.data.objects:
            layout.separator()
            terrain_obj = bpy.data.objects[terrain_name]
            bm = layout.box()
            bm.label(text="5 · Basemap Texture", icon='IMAGE_DATA')

            col = bm.column(align=True)
            col.prop(props, "osmnx_basemap_source", text="Source")

            if props.osmnx_basemap_source == 'WMS':
                col.prop(props, "osmnx_wms_url", text="WMS URL")
                col.prop(props, "osmnx_wms_layer", text="Layer")
            else:
                col.prop(props, "osmnx_basemap_zoom", text="Zoom")
                # Padding is only used when the terrain has no recorded
                # DEM bbox (legacy fallback). Hide it otherwise so users
                # don't think it affects current basemap fetches.
                has_dem_bounds = 'dem_bounds_north' in terrain_obj
                if not has_dem_bounds:
                    col.prop(props, "osmnx_basemap_padding", text="Bbox Padding")

                # Live tile-count estimate so the user does not request
                # gigabytes by accident. We use the terrain's recorded
                # DEM bbox when available so the estimate matches what
                # the operator will actually request.
                try:
                    from ....core.geo import imagery, terrain as _terrain_mod
                    bounds = None
                    if all(k in terrain_obj for k in (
                        'dem_bounds_north', 'dem_bounds_south',
                        'dem_bounds_east', 'dem_bounds_west',
                    )):
                        bounds = {
                            'north': float(terrain_obj['dem_bounds_north']),
                            'south': float(terrain_obj['dem_bounds_south']),
                            'east': float(terrain_obj['dem_bounds_east']),
                            'west': float(terrain_obj['dem_bounds_west']),
                        }
                    if bounds is None:
                        bounds = _terrain_mod.get_osmnx_bounds(
                            obj, padding=props.osmnx_basemap_padding
                        )
                    if bounds:
                        n_tiles = imagery.estimate_tiles(
                            bounds, props.osmnx_basemap_zoom
                        )
                        info = bm.row()
                        info.scale_y = 0.85
                        warn = n_tiles > 200
                        info.label(
                            text=f"≈ {n_tiles} tiles will be downloaded",
                            icon='ERROR' if warn else 'INFO',
                        )
                except Exception:  # noqa: BLE001
                    pass

            # Key-required hint.
            if props.osmnx_basemap_source in ('MAPBOX', 'MAPTILER'):
                from ....preferences import get_preferences
                _prefs = get_preferences()
                _key = ""
                if _prefs is not None:
                    _key = (
                        _prefs.mapbox_api_key
                        if props.osmnx_basemap_source == 'MAPBOX'
                        else _prefs.maptiler_api_key
                    ) or ""
                if not _key.strip():
                    warn = bm.row()
                    warn.alert = True
                    warn.label(
                        text="Set the API key in Add-on Preferences",
                        icon='ERROR',
                    )

            row = bm.row(align=True)
            row.scale_y = 1.2
            row.operator(
                "scigraphs.osmnx_fetch_basemap",
                text="Fetch & Apply",
                icon='IMPORT',
            )
            if terrain_obj.get("basemap_image"):
                row.operator(
                    "scigraphs.osmnx_clear_basemap",
                    text="Clear",
                    icon='X',
                )

            if terrain_obj.get("basemap_image"):
                status = bm.column(align=True)
                status.scale_y = 0.85
                status.label(
                    text=f"Active: {terrain_obj.get('basemap_source', '?')}",
                    icon='CHECKMARK',
                )
                attribution = terrain_obj.get("basemap_attribution", "")
                if attribution:
                    sub = bm.box()
                    sub.scale_y = 0.75
                    sub.label(text=attribution, icon='INFO')


# ---------------------------------------------------------------------------
# 9. IO / Export  (GraphML, GeoPackage, OSM XML, Gephi, SVG + cache)
# ---------------------------------------------------------------------------

class SCIGRAPHS_PT_osmnx_io(bpy.types.Panel):
    bl_label = "IO / Export"
    bl_parent_id = "SCIGRAPHS_PT_osmnx_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OSMnx'
    bl_options = {'DEFAULT_CLOSED'}
    bl_order = 8

    def draw(self, context):
        layout = self.layout
        props = context.scene.scigraphs
        obj = context.active_object

        layout.use_property_split = True
        layout.use_property_decorate = False

        # GraphML IO (round trip)
        box = layout.box()
        box.label(text="GraphML (OSMnx native)", icon='FILE')
        col = box.column(align=True)
        col.prop(props, "osmnx_graphml_path", text="")
        row = col.row(align=True)
        row.operator("scigraphs.osmnx_save_graphml", text="Save", icon='EXPORT')
        row.operator("scigraphs.osmnx_load_graphml", text="Load", icon='IMPORT')

        # Multi-format export
        layout.separator()
        box = layout.box()
        box.label(text="Export Formats (GIS interop)", icon='EXPORT')
        col = box.column(align=True)
        col.prop(props, "osmnx_export_format")
        col.operator("scigraphs.osmnx_export", text="Export…", icon='EXPORT')

        # Cache management (only relevant for OSMnx objects)
        if obj and obj.get("is_osmnx", False):
            layout.separator()
            box = layout.box()
            box.label(text="Graph Cache", icon='FILE_CACHE')

            from ....core.osmnx import cache
            cache_dir = cache.get_cache_directory()
            info_box = box.box()
            info_box.scale_y = 0.8
            info_col = info_box.column(align=True)
            info_col.label(text="Cache Directory:")
            if len(cache_dir) > 40:
                parts = cache_dir.split(os.sep)
                if len(parts) > 2:
                    display_path = "..." + os.sep + os.sep.join(parts[-2:])
                else:
                    display_path = cache_dir[-40:]
                info_col.label(text=display_path)
            else:
                info_col.label(text=cache_dir)

            col = box.column(align=True)
            col.separator()
            col.operator("scigraphs.osmnx_save_to_cache", text="Save to Cache", icon='FILE_TICK')
            row = col.row(align=True)
            row.operator("scigraphs.osmnx_view_cached_graphs", text="Manage Cache", icon='PRESET')
            row.operator("scigraphs.osmnx_open_cache_directory", text="", icon='FILEBROWSER')
        else:
            layout.separator()
            box = layout.box()
            box.label(text="Graph Cache", icon='FILE_CACHE')
            box.operator("scigraphs.osmnx_view_cached_graphs", text="Manage Cache", icon='PRESET')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = [
    SCIGRAPHS_PT_osmnx_main,
    SCIGRAPHS_PT_osmnx_download,
    SCIGRAPHS_PT_osmnx_graph_ops,
    SCIGRAPHS_PT_osmnx_attributes,
    SCIGRAPHS_PT_osmnx_routing,
    SCIGRAPHS_PT_osmnx_accessibility,
    SCIGRAPHS_PT_osmnx_statistics,
    SCIGRAPHS_PT_osmnx_features,
    SCIGRAPHS_PT_osmnx_elevation,
    SCIGRAPHS_PT_osmnx_io,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
