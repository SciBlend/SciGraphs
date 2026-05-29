import pandas as pd
import numpy as np
from ..algorithms import graph
from ...utils.logger import log


GRAPH_FILE_EXTENSIONS = {'.gexf'}


def _file_extension(filepath):
    """Return the normalized extension for a local filepath."""
    import os
    return os.path.splitext(str(filepath or ''))[1].lower()


def is_graph_file(filepath):
    """True when *filepath* is a native graph file handled by NetworkX."""
    return _file_extension(filepath) in GRAPH_FILE_EXTENSIONS


def _networkx_graph_to_edge_dataframe(G):
    """Convert a NetworkX graph into an edge table compatible with CSV import."""
    rows = []

    for u, v, data in G.edges(data=True):
        row = {'source': u, 'target': v}
        for key, value in (data or {}).items():
            if key not in row:
                row[key] = value
        rows.append(row)

    return pd.DataFrame(rows)


def _extract_networkx_node_coordinates(G):
    """Read common GEXF/NetworkX node position encodings if present."""
    coordinates = {}

    for node_id, attrs in G.nodes(data=True):
        coord = None

        viz = attrs.get('viz') if isinstance(attrs, dict) else None
        position = viz.get('position') if isinstance(viz, dict) else None
        if isinstance(position, dict) and 'x' in position and 'y' in position:
            coord = (
                float(position.get('x', 0.0)),
                float(position.get('y', 0.0)),
                float(position.get('z', 0.0)),
            )
        elif 'x' in attrs and 'y' in attrs:
            coord = (
                float(attrs.get('x', 0.0)),
                float(attrs.get('y', 0.0)),
                float(attrs.get('z', 0.0)),
            )

        if coord is not None:
            coordinates[node_id] = coord

    return coordinates


def load_native_graph_file(filepath):
    """Load native graph formats such as GEXF into GraphData."""
    ext = _file_extension(filepath)

    try:
        import networkx as nx
    except ImportError:
        log("Error: NetworkX is required to load native graph files.")
        return None

    try:
        if ext == '.gexf':
            G = nx.read_gexf(filepath)
        else:
            log(f"Unsupported graph file extension: {ext}")
            return None
    except Exception as e:
        log(f"Error reading graph file '{filepath}': {e}")
        return None

    df = _networkx_graph_to_edge_dataframe(G)
    if df.empty:
        log(f"Error: graph file '{filepath}' contains no edges.")
        return None

    graph_data = graph.GraphData(list(G.nodes()), list(G.edges()), df)
    graph_data.source_column_name = 'source'
    graph_data.target_column_name = 'target'
    graph_data.node_coordinates = _extract_networkx_node_coordinates(G)

    edge_weights = None
    if 'weight' in df.columns:
        edge_weights = pd.to_numeric(df['weight'], errors='coerce').fillna(0).values
    graph_data.edge_weights = edge_weights

    log(f"Loaded graph file: {len(graph_data.nodes):,} nodes, {len(graph_data.edges):,} edges")
    if graph_data.node_coordinates:
        log(f"  Node coordinates found: {len(graph_data.node_coordinates):,}")

    return graph_data


def load_graph_from_file(filepath, source_col, target_col, delimiter=','):
    """
    Reads a CSV/text file or native graph file and extracts graph data.
    Optimized for large files using numpy.
    Returns a custom graph data object.
    """
    if not filepath:
        return None

    if is_graph_file(filepath):
        return load_native_graph_file(filepath)
    
    log(f"Loading graph from {filepath}...")
    import time
    start_time = time.time()
    
    try:
        # Read CSV with pandas (still needed for dataframe operations)
        # But use more efficient dtype inference
        df = pd.read_csv(filepath, low_memory=False, delimiter=delimiter)
        log(f"  CSV read in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        log(f"Error reading file: {e}")
        return None
    
    # Get the actual column names from their indices
    try:
        source_col_name = df.columns[source_col]
        target_col_name = df.columns[target_col]
    except IndexError:
        log(f"Error: Column index out of range. CSV has {len(df.columns)} columns.")
        return None
    
    # Extract edges as numpy arrays (much faster)
    edges_start = time.time()
    source_values = df[source_col_name].values
    target_values = df[target_col_name].values
    edges = list(zip(source_values, target_values))
    log(f"  Edges extracted in {time.time() - edges_start:.2f}s")
    
    # Get unique nodes using numpy (faster than pandas)
    nodes_start = time.time()
    all_nodes = np.concatenate([source_values, target_values])
    
    # Convert to pandas Series to handle NaN and mixed types gracefully
    try:
        nodes = np.unique(all_nodes)
    except TypeError:
        # Handle mixed types (NaN + strings) by using pandas
        nodes = pd.Series(all_nodes).dropna().unique()
    
    log(f"  Nodes extracted in {time.time() - nodes_start:.2f}s")
    
    # Store in our custom graph structure
    graph_data = graph.GraphData(nodes, edges, df)
    
    print(f"Total load time: {time.time() - start_time:.2f}s")
    print(f"  Nodes: {len(nodes):,}, Edges: {len(edges):,}")
    
    return graph_data

def get_columns_from_file(filepath, delimiter=','):
    """
    Returns list of column names from CSV/text/native graph file.
    Optimized to read only the header.
    """
    if not filepath:
        return []

    if is_graph_file(filepath):
        graph_data = load_native_graph_file(filepath)
        if graph_data is None or graph_data.dataframe is None:
            return []
        return list(graph_data.dataframe.columns)
    
    try:
        # Use pandas but read only the header (much faster)
        df = pd.read_csv(filepath, nrows=0, delimiter=delimiter)
        return list(df.columns)
    except Exception as e:
        log(f"Error reading file columns: {e}")
        return []

def load_geospatial_graph(
    filepath,
    source_col,
    target_col,
    lat_col=None,
    lon_col=None,
    geocode_mode=False,
    time_col=None,
    time_agg='ALL',
    time_start=None,
    time_end=None,
    weight_col=None,
    delimiter=','
):
    """
    Load graph with geospatial and temporal awareness.
    
    Args:
        filepath: Path to CSV file
        source_col: Index of source column
        target_col: Index of target column
        lat_col: Index of latitude column (optional)
        lon_col: Index of longitude column (optional)
        geocode_mode: If True, geocode source/target as country names
        time_col: Index of time column (optional)
        time_agg: Time aggregation mode ('ALL', 'YEAR', 'MONTH', 'RANGE')
        time_start: Start period for RANGE mode
        time_end: End period for RANGE mode
        weight_col: Index of weight column (optional)
    
    Returns:
        GraphData object with node_coordinates attribute
    """
    from ..geo import geospatial
    
    if not filepath:
        return None
    
    log(f"\nLoading geospatial graph from {filepath}...")
    import time
    start_time = time.time()
    
    try:
        df = pd.read_csv(filepath, low_memory=False, delimiter=delimiter)
        log(f"  CSV read in {time.time() - start_time:.2f}s")
    except Exception as e:
        log(f"Error reading file: {e}")
        return None
    
    # Get column names BEFORE any filtering
    try:
        source_col_name = df.columns[source_col]
        target_col_name = df.columns[target_col]
        weight_col_name = df.columns[weight_col] if weight_col is not None else None
    except IndexError:
        log(f"Error: Column index out of range. CSV has {len(df.columns)} columns.")
        return None
    
    # Handle temporal filtering/aggregation
    if time_col is not None:
        time_col_name = df.columns[time_col]
        
        log(f"  Filtering temporal data: {time_agg} mode")
        df = geospatial.filter_temporal_data(
            df, 
            time_col_name, 
            time_agg, 
            time_start, 
            time_end,
            weight_col_name
        )
        log(f"  After temporal filtering: {len(df):,} rows")
    
    # Extract edges
    edges_start = time.time()
    source_values = df[source_col_name].values
    target_values = df[target_col_name].values
    edges = list(zip(source_values, target_values))
    log(f"  Edges extracted in {time.time() - edges_start:.2f}s")
    
    # Get unique nodes
    nodes_start = time.time()
    all_nodes = np.concatenate([source_values, target_values])
    nodes = np.unique(all_nodes)
    log(f"  Nodes extracted in {time.time() - nodes_start:.2f}s")
    
    # Handle geospatial coordinates
    node_coordinates = {}
    
    if lat_col is not None and lon_col is not None:
        # Use explicit lat/lon columns
        log("  Using explicit lat/lon columns...")
        try:
            lat_col_name = df.columns[lat_col]
            lon_col_name = df.columns[lon_col]
            
            # Create mapping from node name to coordinates
            for node in nodes:
                # Find first row with this node as source or target
                mask = (df[source_col_name] == node) | (df[target_col_name] == node)
                if mask.any():
                    row = df[mask].iloc[0]
                    lat = row[lat_col_name]
                    lon = row[lon_col_name]
                    
                    if pd.notna(lat) and pd.notna(lon):
                        node_coordinates[str(node)] = (float(lat), float(lon))
        except IndexError:
            log(f"Error: Lat/lon column index out of range. CSV has {len(df.columns)} columns.")
            # Continue without coordinates instead of failing
    
    elif geocode_mode:
        # Geocode node names as locations
        log("  Geocoding node names as countries...")
        geocode_start = time.time()
        node_coordinates = geospatial.geocode_locations(list(nodes))
        log(f"  Geocoding completed in {time.time() - geocode_start:.2f}s")
    
    # Store edge weights if provided
    edge_weights = None
    if weight_col_name is not None:
        # weight_col_name was set before any filtering, check if it exists in filtered df
        if weight_col_name in df.columns:
            edge_weights = pd.to_numeric(df[weight_col_name], errors='coerce').fillna(0).values
        else:
            log(f"  Warning: Weight column '{weight_col_name}' not found in filtered data")
    
    # Create graph data object
    graph_data = graph.GraphData(nodes, edges, df)
    graph_data.node_coordinates = node_coordinates
    graph_data.edge_weights = edge_weights
    
    # Store column names for source and target
    graph_data.source_column_name = source_col_name
    graph_data.target_column_name = target_col_name
    
    log(f"Total load time: {time.time() - start_time:.2f}s")
    log(f"  Nodes: {len(nodes):,}, Edges: {len(edges):,}")
    log(f"  Nodes with coordinates: {len(node_coordinates)}")
    
    return graph_data


# ============================================================================
# OSMnx IMPORT FUNCTIONS
# ============================================================================

# Preset custom_filter strings for specialised OSM infrastructure.
# Each preset is a Overpass-compatible "[key~'value']" filter used by OSMnx.
# Keep this list in sync with UI enum `osmnx_custom_filter_preset` in scene properties.
OSMNX_CUSTOM_FILTER_PRESETS = {
    'NONE': None,
    'RAIL': '["railway"~"rail|subway|tram|light_rail|monorail|narrow_gauge"]',
    'SUBWAY': '["railway"~"subway"]',
    'TRAM': '["railway"~"tram|light_rail"]',
    'CYCLEWAY': '["highway"~"cycleway"]',
    'FOOTWAY': '["highway"~"footway|path|pedestrian|steps"]',
    'WATERWAY': '["waterway"~"river|stream|canal"]',
    'MOTORWAY': '["highway"~"motorway|motorway_link|trunk|trunk_link"]',
    'SERVICE': '["highway"~"service"]',
    'BUS_ONLY': '["highway"~"busway"]',
}


def load_osmnx_graph(
    method='PLACE',
    place_name='',
    latitude=0.0,
    longitude=0.0,
    distance=1000,
    address='',
    bbox_north=0.0,
    bbox_south=0.0,
    bbox_east=0.0,
    bbox_west=0.0,
    polygon=None,
    xml_filepath='',
    place_list=None,
    network_type='drive',
    simplify=True,
    retain_geometry=True,
    truncate_by_edge=True,
    retain_all=False,
    custom_filter=None,
    which_result=None,
):
    """
    Download street network from OpenStreetMap using OSMnx.

    Args:
        method: Download method. One of:
            'PLACE'       -> ox.graph_from_place(place_name)
            'POINT'       -> ox.graph_from_point((lat, lon), dist=distance)
            'ADDRESS'     -> ox.graph_from_address(address, dist=distance)
            'BBOX'        -> ox.graph_from_bbox(bbox=(north, south, east, west))
            'POLYGON'     -> ox.graph_from_polygon(polygon)  (shapely Polygon)
            'XML'         -> ox.graph_from_xml(xml_filepath)  (local .osm file)
            'MULTI_PLACE' -> ox.graph_from_place(place_list)  (list of places)
        place_name: Name of the place for PLACE method.
        latitude, longitude: Center coordinates for POINT method.
        distance: Radius in meters for POINT/ADDRESS methods.
        address: Postal address for ADDRESS method.
        bbox_north/south/east/west: Bounding box coords for BBOX method.
        polygon: A shapely Polygon (or MultiPolygon) for POLYGON method.
        xml_filepath: Path to a local .osm XML file for XML method.
        place_list: List of place-name strings for MULTI_PLACE method.
        network_type: Type of network ('drive', 'walk', 'bike', 'all', ...).
        simplify: Remove intermediate nodes that are not intersections.
        retain_geometry: Keep curved street geometry in Blender edges.
        truncate_by_edge: Retain edges that cross the boundary.
        retain_all: Keep disconnected components (islands).
        custom_filter: Overpass custom_filter string (e.g. '["railway"~"rail"]')
            or a key of ``OSMNX_CUSTOM_FILTER_PRESETS``. When set, overrides
            ``network_type`` for infrastructure-specific downloads.
        which_result: For geocoder disambiguation in PLACE (1-indexed).

    Returns:
        Tuple of (GraphData, edge_geometries).  Returns (None, None) on error.
    """
    import time
    start_time = time.time()

    try:
        import osmnx as ox
    except ImportError:
        log("Error: OSMnx is not installed. Please install it with 'pip install osmnx'")
        return None, None

    # Resolve custom_filter preset keyword → actual filter string.
    if isinstance(custom_filter, str) and custom_filter in OSMNX_CUSTOM_FILTER_PRESETS:
        custom_filter = OSMNX_CUSTOM_FILTER_PRESETS[custom_filter]
    if custom_filter == '' or custom_filter == 'NONE':
        custom_filter = None

    log(f"\nDownloading OSM network ({method})...")
    if custom_filter:
        log(f"  custom_filter: {custom_filter}")
    if retain_all:
        log("  retain_all: True (disconnected components kept)")

    # Shared kwargs passed to every graph_from_* constructor.
    kwargs = dict(
        simplify=simplify,
        truncate_by_edge=truncate_by_edge,
        retain_all=retain_all,
    )
    if custom_filter is not None:
        kwargs['custom_filter'] = custom_filter
    else:
        kwargs['network_type'] = network_type

    G = None

    if method == 'PLACE':
        log(f"  Place: {place_name}")
        G = _osmnx_graph_from_place(
            ox, place_name, which_result=which_result, **kwargs
        )

    elif method == 'MULTI_PLACE':
        log(f"  Multi-place: {place_list}")
        G = _osmnx_graph_from_place(
            ox, place_list or [], which_result=None, **kwargs
        )

    elif method == 'POINT':
        log(f"  Point: ({latitude}, {longitude}), radius: {distance}m")
        G = _osmnx_graph_from_point(
            ox, latitude, longitude, distance, **kwargs
        )

    elif method == 'ADDRESS':
        log(f"  Address: {address}, radius: {distance}m")
        G = _osmnx_graph_from_address(
            ox, address, distance, **kwargs
        )

    elif method == 'BBOX':
        log(f"  Bounding box: N={bbox_north}, S={bbox_south}, E={bbox_east}, W={bbox_west}")
        G = _osmnx_graph_from_bbox(
            ox, bbox_north, bbox_south, bbox_east, bbox_west, **kwargs
        )

    elif method == 'POLYGON':
        if polygon is None:
            log("Error: POLYGON method requires a shapely Polygon.")
            return None, None
        log(f"  Polygon: {getattr(polygon, 'bounds', None)}")
        G = _osmnx_graph_from_polygon(ox, polygon, **kwargs)

    elif method == 'XML':
        # XML graphs are loaded from file; they do not accept network_type / retain_all.
        log(f"  XML file: {xml_filepath}")
        G = _osmnx_graph_from_xml(ox, xml_filepath, simplify=simplify)

    else:
        log(f"Error: unknown method '{method}'")
        return None, None

    if G is None:
        return None, None

    log(f"  Download completed in {time.time() - start_time:.2f}s")

    graph_data, edge_geometries = osmnx_to_graph_data(G, retain_geometry)

    log(f"Total import time: {time.time() - start_time:.2f}s")
    log(f"  Nodes: {len(graph_data.nodes):,}, Edges: {len(graph_data.edges):,}")

    return graph_data, edge_geometries


def _osmnx_graph_from_place(ox, place, which_result=None, **kwargs):
    """Download graph by place name or list of places."""
    try:
        call_kwargs = dict(kwargs)
        if which_result is not None:
            call_kwargs['which_result'] = which_result
        G = ox.graph_from_place(place, **call_kwargs)
        return G
    except Exception as e:
        log(f"Error downloading from place '{place}': {e}")
        return None


def _osmnx_graph_from_point(ox, latitude, longitude, distance, **kwargs):
    """Download graph around a point."""
    try:
        G = ox.graph_from_point(
            (latitude, longitude),
            dist=distance,
            **kwargs,
        )
        return G
    except Exception as e:
        log(f"Error downloading from point ({latitude}, {longitude}): {e}")
        return None


def _osmnx_graph_from_address(ox, address, distance, **kwargs):
    """Download graph around an address."""
    try:
        G = ox.graph_from_address(
            address,
            dist=distance,
            **kwargs,
        )
        return G
    except Exception as e:
        log(f"Error downloading from address '{address}': {e}")
        return None


def _osmnx_graph_from_bbox(ox, north, south, east, west, **kwargs):
    """Download graph within a bounding box.

    OSMnx 2.x switched from the historic ``(north, south, east, west)``
    tuple to ``(left, bottom, right, top) = (west, south, east, north)``
    for all spatial helpers. We detect the major version and reorder
    accordingly; otherwise the polygon Built from the tuple becomes
    degenerate (spans a hemisphere) and Overpass subdivides into
    thousands of sub-queries.
    """
    try:
        version = getattr(ox, "__version__", "1.0")
        major = int(str(version).split(".")[0])
    except (ValueError, AttributeError):
        major = 1
    bbox_tuple = (west, south, east, north) if major >= 2 else (north, south, east, west)
    try:
        G = ox.graph_from_bbox(
            bbox=bbox_tuple,
            **kwargs,
        )
        return G
    except Exception as e:
        log(f"Error downloading from bbox: {e}")
        return None


def _osmnx_graph_from_polygon(ox, polygon, **kwargs):
    """Download graph inside a shapely polygon."""
    try:
        G = ox.graph_from_polygon(polygon, **kwargs)
        return G
    except Exception as e:
        log(f"Error downloading from polygon: {e}")
        return None


def _osmnx_graph_from_xml(ox, xml_filepath, simplify=True):
    """Load graph from a local OSM XML file.

    ``graph_from_xml`` only accepts ``simplify``; ``network_type``,
    ``truncate_by_edge``, ``retain_all`` and ``custom_filter`` are not
    applicable to the XML loader.
    """
    try:
        import os
        if not xml_filepath or not os.path.exists(xml_filepath):
            log(f"Error: XML file not found: {xml_filepath}")
            return None
        G = ox.graph_from_xml(xml_filepath, simplify=simplify)
        return G
    except Exception as e:
        log(f"Error loading graph from XML '{xml_filepath}': {e}")
        return None


def osmnx_to_graph_data(G, retain_geometry=True):
    """
    Convert an OSMnx MultiDiGraph to our internal GraphData format.
    
    Args:
        G: OSMnx MultiDiGraph
        retain_geometry: Whether to extract curved street geometries
    
    Returns:
        Tuple of (GraphData, edge_geometries)
    """
    import osmnx as ox
    
    # Get nodes with their coordinates
    nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
    
    # Extract node information
    nodes = list(nodes_gdf.index)
    node_coordinates = {}
    
    for node_id in nodes:
        if node_id in nodes_gdf.index:
            row = nodes_gdf.loc[node_id]
            lat = row['y']
            lon = row['x']
            node_coordinates[node_id] = (lat, lon)
    
    # Extract edges and their geometries
    edges = []
    edge_geometries = {}
    edge_lengths = []
    
    for idx, row in edges_gdf.iterrows():
        u, v, key = idx  # MultiDiGraph index is (u, v, key)
        edges.append((u, v))
        
        # Get edge length
        length = row.get('length', 0)
        edge_lengths.append(length)
        
        # Extract geometry if available and requested
        if retain_geometry and 'geometry' in row and row['geometry'] is not None:
            geom = row['geometry']
            # Convert LineString to list of (lat, lon) tuples
            if hasattr(geom, 'coords'):
                # Note: OSMnx uses (lon, lat) order in geometries
                coords = [(y, x) for x, y in geom.coords]
                edge_geometries[(u, v)] = coords
    
    # Create GraphData object
    graph_data = graph.GraphData(nodes, edges, None)
    graph_data.node_coordinates = node_coordinates
    graph_data.edge_lengths = edge_lengths
    graph_data.is_osmnx = True
    # OSMnx returns MultiDiGraph by default; propagate the actual directedness
    # so downstream code in pathfinding / export_utils builds DiGraphs / writes
    # 'edgedefault="directed"' GraphML correctly.
    graph_data.is_directed = bool(getattr(G, "is_directed", lambda: True)())

    # Store original graph reference for additional analysis
    graph_data.osmnx_graph = G

    return graph_data, edge_geometries

