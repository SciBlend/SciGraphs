import bpy
from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
    PointerProperty,
)


class City2GraphProperties(bpy.types.PropertyGroup):
    """Properties for City2Graph functionality."""
    
    c2g_data_source: EnumProperty(
        name="Data Source",
        description="Source of urban data",
        items=[
            ('OVERTURE', "Overture Maps", "Download from Overture Maps"),
            ('FILE', "From File", "Load from GeoJSON/Shapefile"),
        ],
        default='OVERTURE',
    )

    c2g_area_method: EnumProperty(
        name="Area Method",
        description=(
            "How to define the area for downloading features. Mirrors the "
            "OSMnx panel methods so the same workflow works without an "
            "imported graph"
        ),
        items=[
            ('FROM_OSMNX', "From OSMnx Graph", "Use the active OSMnx graph's bbox + projection"),
            ('PLACE', "By Place Name", "Geocode a place name (e.g. 'Madrid, Spain')"),
            ('POINT', "By Coordinates", "Bounding box around lat/lon + radius"),
            ('ADDRESS', "By Address", "Geocode a postal address + radius"),
            ('BBOX', "By Bounding Box", "Manual N/S/E/W"),
            ('POLYGON', "From Object Polygon", "Use a Blender mesh object's bounds"),
        ],
        default='FROM_OSMNX',
    )
    
    c2g_overture_building: BoolProperty(
        name="Buildings",
        description="Download building footprints",
        default=True,
    )
    
    c2g_overture_segment: BoolProperty(
        name="Road Segments",
        description="Download road segments",
        default=False,
    )
    
    c2g_overture_place: BoolProperty(
        name="Places (POIs)",
        description="Download places of interest",
        default=False,
    )
    
    c2g_overture_water: BoolProperty(
        name="Water",
        description="Download water features",
        default=False,
    )
    
    c2g_overture_land: BoolProperty(
        name="Land Use",
        description="Download land use polygons",
        default=False,
    )
    
    c2g_use_osmnx_bbox: BoolProperty(
        name="Use OSMnx BBox",
        description="Use bounding box from OSMnx properties",
        default=True,
    )

    c2g_overture_limit: IntProperty(
        name="Feature Limit",
        description=(
            "Maximum features per type returned by the Overture REST API. "
            "Lower values are faster and friendlier with the demo key; "
            "higher values may be capped server-side"
        ),
        default=10000,
        min=100,
        max=50000,
        soft_max=20000,
    )
    
    c2g_bbox_north: FloatProperty(
        name="North",
        description="Northern latitude bound",
        default=40.5,
        min=-90.0,
        max=90.0,
        precision=6,
    )
    
    c2g_bbox_south: FloatProperty(
        name="South",
        description="Southern latitude bound",
        default=40.3,
        min=-90.0,
        max=90.0,
        precision=6,
    )
    
    c2g_bbox_east: FloatProperty(
        name="East",
        description="Eastern longitude bound",
        default=-3.5,
        min=-180.0,
        max=180.0,
        precision=6,
    )
    
    c2g_bbox_west: FloatProperty(
        name="West",
        description="Western longitude bound",
        default=-3.9,
        min=-180.0,
        max=180.0,
        precision=6,
    )
    
    c2g_gtfs_path: StringProperty(
        name="GTFS File",
        description="Path to GTFS .zip file",
        subtype='FILE_PATH',
        default="",
    )
    
    c2g_tessellation_shrink: FloatProperty(
        name="Shrink Factor",
        description="Morphological tessellation shrink factor",
        default=0.4,
        min=0.0,
        max=1.0,
        precision=2,
    )
    
    c2g_tessellation_segment: FloatProperty(
        name="Segment Length",
        description="Discretization segment length for tessellation",
        default=0.5,
        min=0.1,
        max=5.0,
        precision=2,
    )
    
    c2g_graph_type: EnumProperty(
        name="Graph Type",
        description="Type of graph representation",
        items=[
            ('HOMOGENEOUS', "Homogeneous", "Single node/edge type"),
            ('HETEROGENEOUS', "Heterogeneous", "Multiple node/edge types"),
        ],
        default='HOMOGENEOUS',
    )
    
    c2g_export_format: EnumProperty(
        name="Export Format",
        description="Format for graph export",
        items=[
            ('JSON', "JSON", "JSON node-link format (recommended for PyG)"),
            ('GRAPHML', "GraphML", "GraphML XML format (standard)"),
        ],
        default='JSON',
    )
    
    c2g_export_path: StringProperty(
        name="Export Path",
        description="Path for exported graph file",
        subtype='FILE_PATH',
        default="",
    )
    
    c2g_gtfs_create_stops: BoolProperty(
        name="Create Stops",
        description="Visualize GTFS stops as spheres",
        default=True,
    )
    
    c2g_gtfs_create_routes: BoolProperty(
        name="Create Routes",
        description="Visualize GTFS routes as curves",
        default=True,
    )
    
    c2g_gtfs_stop_size: FloatProperty(
        name="Stop Size",
        description="Size of stop sphere visualizations",
        default=0.05,
        min=0.01,
        max=1.0,
    )
    
    metapath_hops: IntProperty(
        name="Hops",
        description="Number of segment-to-segment hops for metapath",
        default=3,
        min=1,
        max=10,
    )
    
    metapath_k_neighbors: IntProperty(
        name="K Neighbors",
        description="Number of nearest segments to bridge each amenity",
        default=1,
        min=1,
        max=5,
    )
    
    metapath_amenity_limit: IntProperty(
        name="Amenity Limit",
        description="Maximum number of amenities to analyze (for performance)",
        default=100,
        min=10,
        max=1000,
    )
    
    metapath_visualize_limit: IntProperty(
        name="Visualization Limit",
        description="Maximum metapaths to visualize as curves",
        default=200,
        min=10,
        max=1000,
    )
    
    metapath_curve_thickness: FloatProperty(
        name="Curve Thickness",
        description="Bevel depth for metapath curve visualization",
        default=0.0002,
        min=0.0001,
        max=0.01,
    )
    
    metapath_amenities_object: PointerProperty(
        name="Amenities Object",
        description="Object containing amenity features for metapath analysis",
        type=bpy.types.Object,
    )
    
    prox_graph_type: EnumProperty(
        name="Graph Type",
        description="Type of proximity graph to generate",
        items=[
            ('KNN', "K-Nearest Neighbors", "Connect to K nearest neighbors"),
            ('DELAUNAY', "Delaunay", "Delaunay triangulation"),
            ('FIXED_RADIUS', "Fixed Radius", "Connect within fixed distance (Gilbert graph)"),
            ('WAXMAN', "Waxman", "Probabilistic Waxman graph"),
            ('GABRIEL', "Gabriel", "Gabriel graph (subset of Delaunay)"),
            ('RNG', "Relative Neighborhood", "Relative Neighborhood Graph"),
            ('EMST', "EMST", "Euclidean Minimum Spanning Tree"),
            ('CONTIGUITY', "Contiguity", "Polygon contiguity graph"),
        ],
        default='KNN',
    )
    
    prox_distance_metric: EnumProperty(
        name="Distance Metric",
        description="Metric for distance calculation",
        items=[
            ('EUCLIDEAN', "Euclidean", "Straight-line distance"),
            ('MANHATTAN', "Manhattan", "City-block distance (L1 norm)"),
            ('NETWORK', "Network", "Shortest path along street network"),
        ],
        default='EUCLIDEAN',
    )
    
    prox_knn_k: IntProperty(
        name="K (Neighbors)",
        description="Number of nearest neighbors for KNN graph",
        default=5,
        min=1,
        max=100,
    )
    
    prox_radius: FloatProperty(
        name="Radius (meters)",
        description="Connection radius for fixed-radius graph",
        default=100.0,
        min=10.0,
        max=5000.0,
    )
    
    prox_waxman_beta: FloatProperty(
        name="Beta",
        description="Probability scaling parameter for Waxman graph (0-1)",
        default=0.5,
        min=0.0,
        max=1.0,
        precision=2,
    )
    
    prox_waxman_r0: FloatProperty(
        name="R0 (meters)",
        description="Maximum distance parameter for Waxman graph",
        default=100.0,
        min=10.0,
        max=5000.0,
    )
    
    prox_waxman_seed: IntProperty(
        name="Random Seed",
        description="Random seed for reproducibility (0 = random)",
        default=0,
        min=0,
        max=999999,
    )
    
    prox_contiguity_type: EnumProperty(
        name="Contiguity Type",
        description="Type of polygon contiguity",
        items=[
            ('QUEEN', "Queen", "Polygons sharing edges or vertices"),
            ('ROOK', "Rook", "Polygons sharing only edges"),
        ],
        default='QUEEN',
    )
    
    prox_multilayer_method: EnumProperty(
        name="Multi-Layer Method",
        description="Method for connecting layers",
        items=[
            ('KNN', "K-Nearest Neighbors", "Connect to K nearest in other layers"),
            ('FIXED_RADIUS', "Fixed Radius", "Connect within radius to other layers"),
        ],
        default='KNN',
    )
    
    prox_multilayer_k: IntProperty(
        name="K (Multi-Layer)",
        description="Number of neighbors per layer for multi-layer graph",
        default=1,
        min=1,
        max=20,
    )
    
    prox_multilayer_radius: FloatProperty(
        name="Radius (Multi-Layer)",
        description="Connection radius for multi-layer graph",
        default=100.0,
        min=10.0,
        max=5000.0,
    )
    
    prox_feature_object: PointerProperty(
        name="Feature Object",
        description="Primary OSM features object for proximity graph",
        type=bpy.types.Object,
    )
    
    prox_network_object: PointerProperty(
        name="Street Network",
        description="OSMnx street network for network distance metric",
        type=bpy.types.Object,
    )
    
    prox_curve_thickness: FloatProperty(
        name="Curve Thickness",
        description="Bevel depth for edge visualization curves",
        default=0.0002,
        min=0.0001,
        max=0.01,
        precision=4,
    )
    
    prox_visualize_limit: IntProperty(
        name="Visualization Limit",
        description="Maximum number of edges to materialise in the graph mesh",
        default=1000000,
        min=10,
        max=10000000,
    )
    
    prox_color_by_attribute: BoolProperty(
        name="Color by Attribute",
        description="Enable node coloring by layer attributes",
        default=True,
    )
    
    prox_layer1_object: PointerProperty(
        name="Layer 1",
        description="First layer for multi-layer graph",
        type=bpy.types.Object,
    )
    
    prox_layer2_object: PointerProperty(
        name="Layer 2",
        description="Second layer for multi-layer graph",
        type=bpy.types.Object,
    )
    
    prox_layer3_object: PointerProperty(
        name="Layer 3",
        description="Third layer for multi-layer graph (optional)",
        type=bpy.types.Object,
    )

    prox_deduplicate: BoolProperty(
        name="Deduplicate Points",
        description="Remove nearby duplicate points before graph generation",
        default=True,
    )
    
    prox_dedup_tolerance: FloatProperty(
        name="Tolerance (meters)",
        description="Distance threshold for considering points as duplicates",
        default=0.5,
        min=0.1,
        max=50.0,
        precision=2,
    )
    
    prox_contiguity_predicate: EnumProperty(
        name="Spatial Predicate",
        description="Spatial relationship for contiguity graph",
        items=[
            ('INTERSECTS', "Intersects", "Polygons that touch or overlap"),
            ('TOUCHES', "Touches", "Polygons that share boundary"),
        ],
        default='INTERSECTS',
    )
    
    prox_polygons_object: PointerProperty(
        name="Polygons Object",
        description="Polygon features for group_nodes graph",
        type=bpy.types.Object,
    )
    
    prox_points_object: PointerProperty(
        name="Points Object",
        description="Point features for group_nodes graph",
        type=bpy.types.Object,
    )
    
    prox_group_predicate: EnumProperty(
        name="Containment Predicate",
        description="How to determine if point is in polygon",
        items=[
            ('COVERED_BY', "Covered By", "Point fully covered by polygon"),
            ('WITHIN', "Within", "Point strictly within polygon interior"),
            ('INTERSECTS', "Intersects", "Point intersects polygon (boundary or interior)"),
        ],
        default='COVERED_BY',
    )
    
    geocode_place_name: StringProperty(
        name="Place Name",
        description="Place name to geocode (e.g., 'Liverpool, UK')",
        default="",
    )
    
    od_matrix_path: StringProperty(
        name="OD Matrix File",
        description="Path to OD matrix CSV file",
        subtype='FILE_PATH',
        default="",
    )
    
    od_zones_object: PointerProperty(
        name="Zones Object",
        description="GeoDataFrame with zone polygons",
        type=bpy.types.Object,
    )
    
    od_zone_id_col: StringProperty(
        name="Zone ID Column",
        description="Column name for zone identifiers",
        default="zone_id",
    )
    
    od_matrix_type: EnumProperty(
        name="Matrix Type",
        description="Format of the OD data",
        items=[
            ('EDGELIST', "Edge List", "CSV with source, target, weight columns"),
            ('ADJACENCY', "Adjacency Matrix", "Square matrix with zones as rows/columns"),
        ],
        default='EDGELIST',
    )
    
    od_source_col: StringProperty(
        name="Source Column",
        description="Column name for origin zones (edgelist)",
        default="source",
    )
    
    od_target_col: StringProperty(
        name="Target Column",
        description="Column name for destination zones (edgelist)",
        default="target",
    )
    
    od_weight_col: StringProperty(
        name="Weight Column",
        description="Column name for flow weights",
        default="flow",
    )
    
    od_threshold: FloatProperty(
        name="Flow Threshold",
        description="Minimum flow to include in graph",
        default=0.0,
        min=0.0,
    )
    
    od_directed: BoolProperty(
        name="Directed Graph",
        description="Create directed graph preserving flow direction",
        default=True,
    )
    
    isochrone_threshold: FloatProperty(
        name="Travel Cost",
        description="Maximum travel cost for isochrone (e.g., seconds, meters)",
        default=300.0,
        min=1.0,
    )
    
    isochrone_weight_attr: StringProperty(
        name="Weight Attribute",
        description="Edge attribute to use as travel cost",
        default="travel_time",
    )
    
    isochrone_center_object: PointerProperty(
        name="Center Point",
        description="Object or vertex to use as isochrone center",
        type=bpy.types.Object,
    )
    
    graph_filter_threshold: FloatProperty(
        name="Filter Distance",
        description="Distance threshold for graph filtering (CRS units)",
        default=1000.0,
        min=1.0,
    )
    
    graph_filter_center: PointerProperty(
        name="Filter Center",
        description="Center point for distance-based filtering",
        type=bpy.types.Object,
    )
    
    metapath_weight_attr: StringProperty(
        name="Weight Attribute",
        description="Edge attribute for weighted metapath computation",
        default="length",
    )
    
    metapath_weight_threshold: FloatProperty(
        name="Cost Threshold",
        description="Maximum travel cost for weighted metapaths",
        default=1000.0,
        min=0.0,
    )
    
    metapath_weight_min_threshold: FloatProperty(
        name="Min Cost Threshold",
        description="Minimum travel cost for weighted metapaths",
        default=0.0,
        min=0.0,
    )
    
    metapath_endpoint_type: StringProperty(
        name="Endpoint Type",
        description="Node type to connect via weighted metapaths",
        default="amenity",
    )
    
    # --- Morphological graph full parameters ---
    
    morpho_center_lat: FloatProperty(
        name="Center Latitude",
        description="Latitude of the analysis center point",
        default=0.0,
        min=-90.0,
        max=90.0,
        precision=6,
    )
    
    morpho_center_lon: FloatProperty(
        name="Center Longitude",
        description="Longitude of the analysis center point",
        default=0.0,
        min=-180.0,
        max=180.0,
        precision=6,
    )
    
    morpho_distance: FloatProperty(
        name="Analysis Radius (m)",
        description="Analysis radius in meters from center point",
        default=500.0,
        min=100.0,
        max=50000.0,
    )
    
    morpho_clipping_buffer: FloatProperty(
        name="Clipping Buffer (m)",
        description="Buffer around analysis area to avoid edge effects",
        default=300.0,
        min=0.0,
        max=10000.0,
    )
    
    morpho_contiguity: EnumProperty(
        name="Contiguity Rule",
        description="Adjacency rule for tessellation",
        items=[
            ('QUEEN', "Queen", "Polygons sharing edges or vertices"),
            ('ROOK', "Rook", "Polygons sharing only edges"),
        ],
        default='QUEEN',
    )
    
    morpho_keep_buildings: BoolProperty(
        name="Keep Buildings",
        description="Preserve building geometries in result",
        default=True,
    )
    
    morpho_keep_segments: BoolProperty(
        name="Keep Segments",
        description="Preserve street segment geometries in result",
        default=False,
    )
    
    morpho_use_center_from_osmnx: BoolProperty(
        name="Center from OSMnx",
        description="Use center coordinates from the selected OSMnx graph object",
        default=True,
    )

    # --- Relation toggles for the morphological graph ---
    # When all three are enabled the operator falls back to the
    # high-level c2g.morphological_graph (faster, builds a coherent
    # heterogeneous graph). When any is disabled, the operator wires
    # the requested subset using the individual c2g calls
    # (private_to_private_graph, public_to_public_graph,
    # private_to_public_graph) and produces one Blender object per
    # active relation type.

    morpho_rel_priv_priv: BoolProperty(
        name="Private ↔ Private",
        description=(
            "Adjacency between tessellation cells (parcel neighbours). "
            "Uses city2graph.morphology.private_to_private_graph"
        ),
        default=True,
    )

    morpho_rel_pub_pub: BoolProperty(
        name="Public ↔ Public",
        description=(
            "Connectivity between street segments. "
            "Uses city2graph.morphology.public_to_public_graph"
        ),
        default=True,
    )

    morpho_rel_priv_pub: BoolProperty(
        name="Private ↔ Public",
        description=(
            "Interface (façade) between parcels and street segments. "
            "Uses city2graph.morphology.private_to_public_graph"
        ),
        default=True,
    )
    
    # --- GTFS calendar dates ---
    
    # ``gtfs_calendar_start/end`` are populated dynamically with the
    # set of dates discovered in the active GTFS feed (calendar +
    # calendar_dates). The list lives in ``Scene["c2g_gtfs_dates"]``
    # and is rebuilt every time a feed is imported.
    def _gtfs_date_items(self, _context):  # noqa: ARG002 — Blender API
        scene = bpy.context.scene if bpy.context else None
        if scene is None:
            return [("", "All", "Use the full calendar range")]
        dates = list(scene.get("c2g_gtfs_dates", []) or [])
        items = [("", "All", "Use the full calendar range")]
        for d in dates:
            ds = str(d)
            label = (
                f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}" if len(ds) == 8 else ds
            )
            items.append((ds, label, f"Service date {label}"))
        return items

    gtfs_calendar_start: EnumProperty(
        name="Calendar Start",
        description=(
            "First service day to include. Pick 'All' to use the full "
            "calendar range loaded from the GTFS feed."
        ),
        items=_gtfs_date_items,
    )

    gtfs_calendar_end: EnumProperty(
        name="Calendar End",
        description=(
            "Last service day to include. Pick 'All' to use the full "
            "calendar range loaded from the GTFS feed."
        ),
        items=_gtfs_date_items,
    )

    gtfs_od_directed: BoolProperty(
        name="Directed OD",
        description=(
            "Preserve trip direction in the OD graph. When disabled, "
            "(A, B) and (B, A) are merged into a single undirected edge"
        ),
        default=False,
    )

    gtfs_od_top_n: IntProperty(
        name="Top N Pairs",
        description=(
            "Maximum number of OD pairs to materialise. Big GTFS feeds "
            "can produce millions of pairs; cap to avoid OOM. "
            "0 = no limit (use with caution)"
        ),
        default=10000,
        min=0,
        soft_max=200000,
    )
    
    # --- Overture download types ---
    
    c2g_overture_connector: BoolProperty(
        name="Connectors",
        description="Download network connectors (junction points)",
        default=False,
    )
    
    # --- Graph tools ---
    
    graph_tool_action: EnumProperty(
        name="Action",
        description="Graph tool to apply",
        items=[
            ('CLIP', "Clip to Area", "Clip graph to a polygon boundary"),
            ('FILTER', "Filter by Distance", "Remove nodes beyond a distance threshold"),
            ('ISOCHRONE', "Isochrone", "Generate isochrone polygon from center"),
            ('REMOVE_ISOLATED', "Remove Isolated", "Keep only the largest connected component"),
        ],
        default='REMOVE_ISOLATED',
    )


def register():
    bpy.utils.register_class(City2GraphProperties)
    bpy.types.Scene.city2graph = bpy.props.PointerProperty(type=City2GraphProperties)


def unregister():
    del bpy.types.Scene.city2graph
    bpy.utils.unregister_class(City2GraphProperties)

