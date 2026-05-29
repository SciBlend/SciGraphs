from bpy.props import (
    StringProperty,
    EnumProperty,
    BoolProperty,
    IntProperty,
    FloatProperty,
)

OSMNX_SCENE_PROPERTIES = {
    # ========================================
    # OSMnx PROPERTIES (OpenStreetMap Import)
    # ========================================

    'osmnx_download_method': EnumProperty(
        name="Download Method",
        description="Method to define the area of interest",
        items=[
            ('PLACE', "By Place Name", "Download by place name (e.g. 'Madrid, Spain')"),
            ('MULTI_PLACE', "Multi-Place", "Download from a list of places (one per line)"),
            ('POINT', "By Coordinates", "Download within radius of a point"),
            ('ADDRESS', "By Address", "Download around a postal address"),
            ('BBOX', "By Bounding Box", "Download within north/south/east/west bounds"),
            ('POLYGON', "From Object Polygon", "Use a Blender mesh object as the boundary polygon"),
            ('XML', "From OSM XML File", "Load graph from a local .osm XML file"),
        ],
        default='PLACE',
    ),

    'osmnx_place_list': StringProperty(
        name="Place List",
        description="List of place names (one per line) for Multi-Place download",
        default="",
    ),

    'osmnx_polygon_object': StringProperty(
        name="Polygon Object",
        description="Name of a Blender mesh object to use as the boundary polygon",
        default="",
    ),

    'osmnx_xml_filepath': StringProperty(
        name="OSM XML File",
        description="Path to a local .osm XML file",
        subtype='FILE_PATH',
        default="",
    ),

    'osmnx_retain_all': BoolProperty(
        name="Retain All",
        description="Keep disconnected components (islands / isolated subgraphs)",
        default=False,
    ),

    'osmnx_which_result': IntProperty(
        name="Which Result",
        description="Geocoder disambiguation: 1 = first OSM match, 2 = second, etc. (0 = auto)",
        default=0,
        min=0,
        max=10,
    ),

    'osmnx_custom_filter_preset': EnumProperty(
        name="Infrastructure Preset",
        description="Quick infrastructure filter (overrides Network Type)",
        items=[
            ('NONE', "None (use Network Type)", "Use the standard network_type dropdown"),
            ('RAIL', "Rail (all)", "All railways: rail, subway, tram, light_rail, monorail, narrow_gauge"),
            ('SUBWAY', "Subway / Metro", "Metro/subway systems only"),
            ('TRAM', "Tram / Light Rail", "Tram and light rail"),
            ('CYCLEWAY', "Cycleway", "Cycling infrastructure"),
            ('FOOTWAY', "Footway / Pedestrian", "Pedestrian paths, footways, steps"),
            ('WATERWAY', "Waterway", "Rivers, streams, canals"),
            ('MOTORWAY', "Motorway / Trunk", "Only major roads (motorway/trunk)"),
            ('SERVICE', "Service Roads", "Only service roads"),
            ('BUS_ONLY', "Busway", "Bus-only ways"),
        ],
        default='NONE',
    ),

    'osmnx_custom_filter_text': StringProperty(
        name="Custom Filter",
        description="Advanced: raw Overpass custom_filter string (overrides preset if non-empty)",
        default="",
    ),

    'osmnx_place_name': StringProperty(
        name="Place Name",
        description="Name of the place to download (e.g. 'Piedmont, California, USA')",
        default="",
    ),

    'osmnx_network_type': EnumProperty(
        name="Network Type",
        description="Type of street network to download",
        items=[
            ('drive', "Drive", "Roads for driving (excludes service roads)"),
            ('drive_service', "Drive + Service", "Roads for driving including service roads"),
            ('walk', "Walk", "Pedestrian paths and sidewalks"),
            ('bike', "Bike", "Cycling infrastructure"),
            ('all', "All", "All public and private ways"),
            ('all_public', "All Public", "All public ways"),
        ],
        default='drive',
    ),

    'osmnx_latitude': FloatProperty(
        name="Latitude",
        description="Center latitude for point-based download",
        default=40.4168,
        min=-90.0,
        max=90.0,
        precision=6,
    ),

    'osmnx_longitude': FloatProperty(
        name="Longitude",
        description="Center longitude for point-based download",
        default=-3.7038,
        min=-180.0,
        max=180.0,
        precision=6,
    ),

    'osmnx_distance': IntProperty(
        name="Distance (m)",
        description="Radius in meters for point/address download",
        default=1000,
        min=100,
        max=50000,
    ),

    'osmnx_address': StringProperty(
        name="Address",
        description="Postal address to geocode and download around",
        default="",
    ),

    'osmnx_bbox_north': FloatProperty(
        name="North",
        description="Northern latitude bound",
        default=40.5,
        min=-90.0,
        max=90.0,
        precision=6,
    ),

    'osmnx_bbox_south': FloatProperty(
        name="South",
        description="Southern latitude bound",
        default=40.3,
        min=-90.0,
        max=90.0,
        precision=6,
    ),

    'osmnx_bbox_east': FloatProperty(
        name="East",
        description="Eastern longitude bound",
        default=-3.5,
        min=-180.0,
        max=180.0,
        precision=6,
    ),

    'osmnx_bbox_west': FloatProperty(
        name="West",
        description="Western longitude bound",
        default=-3.9,
        min=-180.0,
        max=180.0,
        precision=6,
    ),

    'osmnx_simplify': BoolProperty(
        name="Simplify Graph",
        description="Remove intermediate nodes that are not intersections",
        default=True,
    ),

    'osmnx_retain_geometry': BoolProperty(
        name="Retain Street Geometry",
        description="Keep the curved geometry of streets (not just straight lines)",
        default=True,
    ),

    'osmnx_truncate_by_edge': BoolProperty(
        name="Truncate by Edge",
        description="Retain edges that cross the boundary (vs. strict truncation)",
        default=True,
    ),

    'osmnx_scale': FloatProperty(
        name="Scale Factor",
        description="Scale factor for the imported network (meters to Blender units)",
        default=0.001,
        min=0.0001,
        max=1.0,
        precision=4,
    ),

    'osmnx_bearing_lat1': FloatProperty(
        name="Latitude 1",
        description="First point latitude for bearing calculation",
        default=0.0,
        min=-90.0,
        max=90.0,
        precision=6,
    ),

    'osmnx_bearing_lon1': FloatProperty(
        name="Longitude 1",
        description="First point longitude for bearing calculation",
        default=0.0,
        min=-180.0,
        max=180.0,
        precision=6,
    ),

    'osmnx_bearing_lat2': FloatProperty(
        name="Latitude 2",
        description="Second point latitude for bearing calculation",
        default=0.0,
        min=-90.0,
        max=90.0,
        precision=6,
    ),

    'osmnx_bearing_lon2': FloatProperty(
        name="Longitude 2",
        description="Second point longitude for bearing calculation",
        default=0.0,
        min=-180.0,
        max=180.0,
        precision=6,
    ),

    'osmnx_bearing_result': FloatProperty(
        name="Bearing Result",
        description="Calculated bearing in degrees",
        default=0.0,
    ),

    'osmnx_bearing_num_bins': IntProperty(
        name="Number of Bins",
        description="Number of bins for bearing distribution",
        default=36,
        min=4,
        max=360,
    ),

    'osmnx_dist_y1': FloatProperty(
        name="Y1/Lat1",
        description="First point Y coordinate or latitude",
        default=0.0,
        precision=6,
    ),

    'osmnx_dist_x1': FloatProperty(
        name="X1/Lon1",
        description="First point X coordinate or longitude",
        default=0.0,
        precision=6,
    ),

    'osmnx_dist_y2': FloatProperty(
        name="Y2/Lat2",
        description="Second point Y coordinate or latitude",
        default=0.0,
        precision=6,
    ),

    'osmnx_dist_x2': FloatProperty(
        name="X2/Lon2",
        description="Second point X coordinate or longitude",
        default=0.0,
        precision=6,
    ),

    'osmnx_distance_result': FloatProperty(
        name="Distance Result",
        description="Calculated distance",
        default=0.0,
    ),

    # Node-pair distance calculator (eyedropper-driven, replaces the old
    # manual lat/lon calculator).
    'osmnx_dist_node_a': StringProperty(
        name="Node A",
        description="First graph node ID for the distance calculator",
        default="",
    ),

    'osmnx_dist_node_b': StringProperty(
        name="Node B",
        description="Second graph node ID for the distance calculator",
        default="",
    ),

    # ========================================
    # ELEVATION & TERRAIN — UNIFIED PIPELINE
    # ========================================
    # The user picks ONE source, plus what to do with it (apply to nodes /
    # create terrain mesh). Backed by a single dispatcher operator.

    'osmnx_dem_source': EnumProperty(
        name="Elevation Source",
        description="Where to fetch elevation data from",
        items=[
            ('OPENTOPOGRAPHY', "OpenTopography (high quality)",
                "SRTM / ALOS / Copernicus via OpenTopography. Requires API key."),
            ('OPEN_ELEVATION', "Open-Elevation API (free)",
                "Free public API, no key, lower resolution. Slow on big areas."),
            ('LOCAL_GEOTIFF', "Local GeoTIFF",
                "Use a DEM file you already downloaded (e.g. from Copernicus)."),
        ],
        default='OPENTOPOGRAPHY',
    ),

    'osmnx_dem_apply_to_nodes': BoolProperty(
        name="Apply to nodes",
        description="Sample the DEM at every graph node and write its 'elevation' attribute",
        default=True,
    ),

    'osmnx_dem_create_terrain': BoolProperty(
        name="Create terrain mesh",
        description="Build a visible terrain mesh under the network as visual context",
        default=True,
    ),

    'osmnx_dem_terrain_method': EnumProperty(
        name="Terrain Method",
        description="How to build the terrain mesh from the DEM",
        items=[
            ('DISPLACE', "Displace (fast)",
                "Subdivision Surface + Displace modifier. Fast and lightweight."),
            ('RAW_MESH', "Raw Mesh (accurate)",
                "One vertex per raster pixel. Slower, exact for analysis or close-ups."),
        ],
        default='DISPLACE',
    ),

    'osmnx_dem_subdivision_levels': IntProperty(
        name="Subdivision Levels",
        description="Displace method: subdivision steps (higher = more detail, slower)",
        default=6,
        min=1,
        max=10,
    ),

    'osmnx_dem_subsample': IntProperty(
        name="Subsample",
        description="Raw Mesh method: take 1 of every N pixels (1 = full resolution)",
        default=2,
        min=1,
        max=16,
    ),

    'osmnx_dem_local_path': StringProperty(
        name="GeoTIFF Path",
        description="Path to a local DEM GeoTIFF file",
        default="",
        subtype='FILE_PATH',
    ),

    'osmnx_dem_padding': FloatProperty(
        name="Padding",
        description="Extra padding around the network bounds for downloads / sampling (fraction)",
        default=0.1,
        min=0.0,
        max=0.5,
    ),

    'osmnx_dem_api_resolution': IntProperty(
        name="API Resolution",
        description="Open-Elevation: grid resolution (points per side). Higher = more detail, slower",
        default=50,
        min=10,
        max=300,
    ),

    'osmnx_dem_api_workers': IntProperty(
        name="API Workers",
        description="Open-Elevation: parallel HTTP workers",
        default=5,
        min=1,
        max=16,
    ),

    'osmnx_dem_api_provider': EnumProperty(
        name="API Provider",
        description="Backend used when source is Open-Elevation",
        items=[
            ('open-elevation', "Open-Elevation", "Free, global"),
            ('opentopodata', "OpenTopoData (SRTM)", "Free, SRTM 30m"),
        ],
        default='open-elevation',
    ),

    # ========================================
    # BASEMAP TEXTURE FOR TERRAIN
    # ========================================
    # Drape an aerial / satellite / map basemap onto the imported terrain
    # mesh. Sources are XYZ tile servers (Esri, OSM, Mapbox, MapTiler) or
    # an arbitrary WMS endpoint.

    'osmnx_basemap_source': EnumProperty(
        name="Basemap Source",
        description="Imagery provider used to texture the terrain mesh",
        items=[
            ('ESRI', "Satellite (Esri)",
                "Esri World Imagery, no key required (best default)"),
            ('OSM', "OSM Standard",
                "OpenStreetMap rendered map, no key required"),
            ('MAPBOX', "Mapbox Satellite",
                "Mapbox satellite tiles. Requires Mapbox token in addon preferences"),
            ('MAPTILER', "MapTiler Satellite",
                "MapTiler satellite tiles. Requires MapTiler key in addon preferences"),
            ('WMS', "Custom WMS",
                "Any WMS GetMap endpoint (e.g. PNOA, USGS, Sentinel Hub WMS)"),
        ],
        default='ESRI',
    ),

    'osmnx_basemap_zoom': IntProperty(
        name="Zoom Level",
        description=(
            "Slippy-map zoom (XYZ sources). Higher = more detail and "
            "exponentially more tiles. 14–17 is usually a sweet spot"
        ),
        default=16,
        min=10,
        max=20,
    ),

    'osmnx_basemap_padding': FloatProperty(
        name="Bbox Padding",
        description="Extra padding around the graph bbox before requesting imagery (fraction)",
        default=0.05,
        min=0.0,
        max=0.5,
    ),

    'osmnx_wms_url': StringProperty(
        name="WMS URL",
        description="Base URL of the WMS service (without query string parameters)",
        default="",
    ),

    'osmnx_wms_layer': StringProperty(
        name="WMS Layer",
        description="Layer name to request (the WMS LAYERS parameter)",
        default="",
    ),

    # ========================================
    # OSMnx ANALYSIS PROPERTIES
    # ========================================

    'osmnx_shortest_path_source': StringProperty(
        name="Source Node",
        description="Source node ID for shortest path calculation",
        default="",
    ),

    'osmnx_shortest_path_target': StringProperty(
        name="Target Node",
        description="Target node ID for shortest path calculation",
        default="",
    ),

    'osmnx_path_weight': EnumProperty(
        name="Path Weight",
        description="Attribute to minimize when finding shortest path",
        items=[
            ('length', "Distance (length)", "Minimize total distance in meters"),
            ('travel_time', "Travel Time", "Minimize travel time (requires speeds + travel_times)"),
            ('elevation_impedance', "Length + Elevation Penalty",
                "length * (1 + alpha * |grade|) — requires node elevations + edge grades"),
        ],
        default='length',
    ),

    'osmnx_selected_node_id': StringProperty(
        name="Selected Node",
        description="ID of the last selected node",
        default="",
    ),

    'osmnx_selected_edge_u': StringProperty(
        name="Selected Edge U",
        description="Source node of selected edge",
        default="",
    ),

    'osmnx_selected_edge_v': StringProperty(
        name="Selected Edge V",
        description="Target node of selected edge",
        default="",
    ),

    'osmnx_graphml_path': StringProperty(
        name="GraphML Path",
        description="File path for GraphML save/load",
        subtype='FILE_PATH',
        default="",
    ),

    'osmnx_network_area': FloatProperty(
        name="Network Area",
        description="Area covered by the network in square kilometers",
        default=0.0,
        min=0.0,
        precision=2,
        unit='AREA',
    ),

    'osmnx_fallback_speed': IntProperty(
        name="Fallback Speed",
        description="Default speed (km/h) for roads without speed data",
        default=30,
        min=5,
        max=130,
    ),

    # Elevation properties
    'osmnx_elevation_scale': FloatProperty(
        name="Elevation Scale",
        description="Vertical exaggeration factor for 3D elevation display",
        default=1.0,
        min=0.1,
        max=100.0,
    ),

    'osmnx_elevation_offset': FloatProperty(
        name="Elevation Offset",
        description="Base elevation offset in meters (added to all elevations)",
        default=0.0,
        min=-1000.0,
        max=10000.0,
    ),

    'osmnx_dem_filepath': StringProperty(
        name="DEM File",
        description="Path to local DEM raster file for elevation data",
        subtype='FILE_PATH',
        default="",
    ),

    # Terrain plane import properties
    'terrain_offset_x': FloatProperty(
        name="Offset X",
        description="Horizontal offset of terrain in X direction",
        default=0.0,
        min=-10000.0,
        max=10000.0,
    ),

    'terrain_offset_y': FloatProperty(
        name="Offset Y",
        description="Horizontal offset of terrain in Y direction",
        default=0.0,
        min=-10000.0,
        max=10000.0,
    ),

    'terrain_offset_z': FloatProperty(
        name="Offset Z",
        description="Vertical offset of terrain",
        default=0.0,
        min=-1000.0,
        max=1000.0,
    ),

    'terrain_scale_xy': FloatProperty(
        name="Scale XY",
        description="Horizontal scale of terrain plane",
        default=1.0,
        min=0.01,
        max=100.0,
    ),

    'terrain_opacity': FloatProperty(
        name="Opacity",
        description="Terrain texture opacity",
        default=1.0,
        min=0.0,
        max=1.0,
    ),

    'osmnx_simplification_tolerance': FloatProperty(
        name="Consolidation Tolerance",
        description="Distance threshold in meters for consolidating intersections",
        default=10.0,
        min=1.0,
        max=100.0,
    ),

    'osmnx_feature_type': EnumProperty(
        name="Feature Type",
        description="Type of OSM features to download",
        items=[
            ('BUILDING', "Buildings", "Download building footprints"),
            ('AMENITY', "Amenities (All)", "Download all amenities (shops, restaurants, etc.)"),
            ('AMENITY_METAPATH', "Amenities (Metapath)", "Download amenities for metapath analysis (cafe, restaurant, pub, bar, museum, theatre, cinema)"),
            ('LANDUSE', "Land Use", "Download land use polygons"),
            ('NATURAL', "Natural", "Download natural features (water, forest, etc.)"),
            ('HIGHWAY', "Highways", "Download highway/road features"),
            ('CUSTOM', "Custom", "Use custom OSM tags"),
        ],
        default='BUILDING',
    ),

    'osmnx_custom_tags': StringProperty(
        name="Custom Tags",
        description="Custom OSM tags as key=value pairs (comma-separated)",
        default="building=yes",
    ),

    'osmnx_features_place': StringProperty(
        name="Place for Features",
        description="Place name for feature download",
        default="",
    ),

    'osmnx_features_distance': IntProperty(
        name="Features Distance",
        description="Radius in meters for feature download from point",
        default=1000,
        min=100,
        max=10000,
    ),

    'osmnx_geocode_address': StringProperty(
        name="Address to Geocode",
        description="Address or place name for geocoding",
        default="",
    ),

    'osmnx_truncate_distance': FloatProperty(
        name="Truncation Distance",
        description="Network distance threshold for truncation",
        default=1000.0,
        min=100.0,
        max=50000.0,
    ),

    # ========================================
    # OSMnx ROUTING (advanced)
    # ========================================

    'osmnx_impedance_alpha': FloatProperty(
        name="Elevation Alpha",
        description="Elevation-impedance weight multiplier: length * (1 + alpha * |grade|)",
        default=5.0,
        min=0.0,
        max=50.0,
        precision=2,
    ),

    'osmnx_k_shortest': IntProperty(
        name="K Alternative Routes",
        description="Number of alternative routes to compute (k-shortest paths)",
        default=3,
        min=2,
        max=20,
    ),

    'osmnx_od_random_n': IntProperty(
        name="Random OD Pairs",
        description="Number of random origin-destination pairs to sample",
        default=10,
        min=1,
        max=1000,
    ),

    'osmnx_od_batch_cpus': IntProperty(
        name="CPU Cores",
        description="Parallel cores for batch shortest-path (0 = auto)",
        default=0,
        min=0,
        max=32,
    ),

    # ========================================
    # OSMnx ACCESSIBILITY / ISOCHRONES
    # ========================================

    'osmnx_iso_center_node': StringProperty(
        name="Center Node",
        description="Center node ID for isochrone / ego graph",
        default="",
    ),

    'osmnx_iso_trip_times': StringProperty(
        name="Trip Times (min)",
        description="Comma-separated list of travel-time thresholds in minutes",
        default="5, 10, 15, 20",
    ),

    'osmnx_iso_travel_speed': FloatProperty(
        name="Travel Speed (km/h)",
        description="Walking/cycling speed to use when edges have no travel_time attribute",
        default=4.5,
        min=1.0,
        max=200.0,
        precision=1,
    ),

    'osmnx_iso_mode': EnumProperty(
        name="Isochrone Mode",
        description="How to generate the isochrone polygon from reachable nodes/edges",
        items=[
            ('CONVEX_HULL', "Convex Hull", "Fast outline (convex hull of reachable nodes)"),
            ('BUFFER_UNION', "Buffer Union", "Accurate: buffer and union of reachable edges"),
        ],
        default='BUFFER_UNION',
    ),

    'osmnx_iso_buffer': FloatProperty(
        name="Edge Buffer (m)",
        description="Buffer size (meters) for buffer-union isochrone mode",
        default=25.0,
        min=1.0,
        max=500.0,
    ),

    # ========================================
    # OSMnx CENTRALITY / COLOR-BY-ATTR
    # ========================================

    'osmnx_centrality_kind': EnumProperty(
        name="Centrality",
        description="Type of centrality to compute",
        items=[
            ('BETWEENNESS_NODE', "Node Betweenness", "Betweenness centrality on nodes"),
            ('BETWEENNESS_EDGE', "Edge Betweenness (line graph)",
                "Betweenness computed on edges via line graph"),
            ('CLOSENESS', "Closeness", "Closeness centrality on nodes"),
        ],
        default='BETWEENNESS_NODE',
    ),

    'osmnx_centrality_weighted': BoolProperty(
        name="Length-Weighted",
        description="Use edge length as weight for centrality",
        default=True,
    ),

    'osmnx_centrality_fast': BoolProperty(
        name="Fast Mode (rustworkx)",
        description="Use rustworkx backend when available (much faster for large graphs)",
        default=True,
    ),

    'osmnx_colormap': EnumProperty(
        name="Colormap",
        description="Matplotlib colormap for attribute → vertex colors",
        items=[
            ('viridis', "Viridis", ""),
            ('plasma', "Plasma", ""),
            ('magma', "Magma", ""),
            ('inferno', "Inferno", ""),
            ('turbo', "Turbo", ""),
            ('coolwarm', "Cool-Warm", ""),
            ('RdYlBu_r', "Red→Yellow→Blue", ""),
            ('YlGnBu', "Yellow→Green→Blue", ""),
            ('hot', "Hot", ""),
        ],
        default='viridis',
    ),

    'osmnx_color_attr_name': StringProperty(
        name="Attribute",
        description="Node/edge attribute name to map to vertex colors",
        default="betweenness",
    ),

    'osmnx_color_target': EnumProperty(
        name="Color Target",
        description="Apply colors to nodes or edges",
        items=[
            ('NODES', "Nodes (vertex colors)", "Apply to mesh vertices"),
            ('EDGES', "Edges (edge attribute)", "Store as edge attribute for geometry nodes"),
        ],
        default='NODES',
    ),

    # ========================================
    # OSMnx ORIENTATION ROSE (3D polar mesh)
    # ========================================

    'osmnx_rose_bins': IntProperty(
        name="Rose Bins",
        description="Number of angular bins for the 3D orientation rose",
        default=36,
        min=8,
        max=360,
    ),

    'osmnx_rose_radius': FloatProperty(
        name="Rose Radius",
        description="Outer radius of the 3D orientation rose (Blender units)",
        default=2.0,
        min=0.1,
        max=1000.0,
    ),

    'osmnx_rose_height_scale': FloatProperty(
        name="Bar Height Scale",
        description="Height multiplier for orientation-rose bars",
        default=1.0,
        min=0.01,
        max=100.0,
    ),

    # ========================================
    # OSMnx FEATURES (POIs)
    # ========================================

    'osmnx_poi_snap_mode': EnumProperty(
        name="POI Snap Mode",
        description="How to link POI features to the street network",
        items=[
            ('ATTR_ONLY', "Attribute only", "Write nearest_node_id to each POI (no geometry change)"),
            ('MOVE_TO_NODE', "Move to node", "Move each POI point to its nearest node"),
            ('ADD_CONNECTOR', "Add connector edges", "Create visible edges from each POI to its nearest node"),
        ],
        default='ATTR_ONLY',
    ),

    # ========================================
    # OSMnx EXPORT (GIS interop)
    # ========================================

    'osmnx_export_filepath': StringProperty(
        name="Export Path",
        description="Output file path for GIS exports",
        subtype='FILE_PATH',
        default="",
    ),

    'osmnx_export_format': EnumProperty(
        name="Export Format",
        description="GIS interchange format",
        items=[
            ('GEOPACKAGE', "GeoPackage (.gpkg)", "QGIS-native, OGC GeoPackage"),
            ('OSM_XML', "OSM XML (.osm)", "OpenStreetMap XML (requires all_oneway=True)"),
            ('GRAPHML_GEPHI', "GraphML (Gephi)", "GraphML with Gephi-compatible attribute names"),
            ('SVG', "SVG (vector)", "Matplotlib SVG render of the projected graph"),
        ],
        default='GEOPACKAGE',
    ),

    # ========================================
    # OSMnx INTERPOLATE / SAMPLE POINTS
    # ========================================

    'osmnx_sample_n': IntProperty(
        name="Number of Samples",
        description="Number of points to sample uniformly from the graph",
        default=20,
        min=1,
        max=10000,
    ),

    'osmnx_interpolate_distance': FloatProperty(
        name="Interpolation Distance (m)",
        description="Distance between interpolated points along a LineString",
        default=50.0,
        min=0.1,
        max=10000.0,
    ),
}

