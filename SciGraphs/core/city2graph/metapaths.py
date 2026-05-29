"""
Metapath analysis for heterogeneous urban graphs.

This module provides functions to create dual graphs from street networks,
bridge amenities to street segments, and compute metapaths between amenities.
"""

def create_street_dual_graph_c2g(osmnx_graph):
    """
    Create dual graph using city2graph (street segments as nodes).
    
    CRITICAL: Fixes MultiIndex 3-level issue (x, y, nan) -> (x, y).
    
    Args:
        osmnx_graph: NetworkX MultiDiGraph from OSMnx
        
    Returns:
        tuple: (dual_nodes_gdf, dual_edges_gdf) with fixed indices
    """
    import pandas as pd
    import city2graph as c2g
    
    street_primary_nodes, street_primary_edges = c2g.nx_to_gdf(osmnx_graph)
    
    dual_nodes_gdf, dual_edges_gdf = c2g.dual_graph((street_primary_nodes, street_primary_edges))
    
    dual_nodes_gdf.geometry = dual_nodes_gdf.geometry.centroid
    
    if isinstance(dual_nodes_gdf.index, pd.MultiIndex) and dual_nodes_gdf.index.nlevels == 3:
        # Extract only (x, y), drop the nan third level
        new_index = [(idx[0], idx[1]) for idx in dual_nodes_gdf.index]
        dual_nodes_gdf.index = pd.Index(new_index)
        
        # Fix edges index to match
        new_edge_index = [
            ((idx[0][0], idx[0][1]), (idx[1][0], idx[1][1])) 
            if isinstance(idx, tuple) and len(idx) == 2 
            else idx 
            for idx in dual_edges_gdf.index
        ]
        dual_edges_gdf.index = pd.Index(new_edge_index)
    
    return dual_nodes_gdf, dual_edges_gdf


def prepare_amenities_from_features(features_obj, target_crs, limit=None):
    """
    Convert features object to amenities GeoDataFrame.
    
    Args:
        features_obj: Blender object with features (from OSMnx or City2Graph)
        target_crs: Target CRS to project to (should match dual graph)
        limit: Maximum number of amenities to return (None for all)
        
    Returns:
        GeoDataFrame: Amenities with Point geometry and clean integer index
    """
    import geopandas as gpd
    import osmnx as ox
    
    place = features_obj.get("place_name", "")
    
    if not place:
        raise ValueError("Features object missing 'place_name' property. Please re-download features using OSMnx operators.")
    
    # Try to download amenities fresh (only works for place-based queries)
    tags = {
        'amenity': ['cafe', 'restaurant', 'pub', 'bar', 'museum', 'theatre', 'cinema']
    }
    
    amenities_gdf = None
    
    # Check if this was downloaded from a place name (not point/bbox)
    if not place.startswith("Point") and not place.startswith("BBox"):
        try:
            amenities_gdf = ox.features_from_place(place, tags=tags)
        except Exception as e:
            print(f"Warning: Could not re-download amenities from '{place}': {e}")
            print("Attempting to extract from Blender object...")
    
    # If re-download failed or was from point/bbox, extract from Blender mesh
    if amenities_gdf is None or len(amenities_gdf) == 0:
        # Extract coordinates from Blender object
        if not features_obj.data or not features_obj.data.vertices:
            raise ValueError("Features object has no mesh data")
        
        # Get original CRS from object
        original_crs = features_obj.get("crs", "EPSG:4326")
        
        # Extract vertex positions and reverse transformation
        from shapely.geometry import Point
        
        points = []
        for vert in features_obj.data.vertices:
            # TODO: This needs proper coordinate transformation back to geographic/projected space
            # For now, this is a placeholder - the re-download approach is preferred
            points.append(Point(vert.co.x, vert.co.y))
        
        if not points:
            raise ValueError("No valid points extracted from features object")
        
        amenities_gdf = gpd.GeoDataFrame(geometry=points, crs=original_crs)
    
    # Reset index BEFORE projection to avoid MultiIndex issues
    amenities_gdf = amenities_gdf.reset_index(drop=True)
    
    # Project to target CRS
    amenities_gdf = amenities_gdf.to_crs(target_crs)
    
    # Convert to Point geometry (centroids)
    amenities_gdf['geometry'] = amenities_gdf.geometry.centroid
    
    # Create clean GeoDataFrame with only geometry
    amenities_gdf = gpd.GeoDataFrame(amenities_gdf[['geometry']], crs=amenities_gdf.crs)
    
    # Apply limit if specified
    if limit:
        amenities_gdf = amenities_gdf.head(limit)
    
    # Ensure clean integer index
    amenities_gdf = amenities_gdf.reset_index(drop=True)
    
    return amenities_gdf


def bridge_amenities_to_segments(amenities_gdf, dual_nodes_gdf, k=1):
    """
    Bridge amenities to nearest street segments using c2g.bridge_nodes.
    
    Args:
        amenities_gdf: GeoDataFrame of amenity points
        dual_nodes_gdf: GeoDataFrame of dual graph nodes (street segments)
        k: Number of nearest segments to connect each amenity to
        
    Returns:
        tuple: (nodes_dict, edges_dict) for heterogeneous graph
    """
    import city2graph as c2g
    
    # Ensure node_type attributes
    if 'node_type' not in amenities_gdf.columns:
        amenities_gdf['node_type'] = 'amenity'
    if 'node_type' not in dual_nodes_gdf.columns:
        dual_nodes_gdf['node_type'] = 'segment'
    
    # Create heterogeneous graph structure
    nodes_dict = {
        "amenity": amenities_gdf,
        "segment": dual_nodes_gdf
    }
    
    # Get dual edges from stored property or create empty
    # (This should be passed in, but we handle the structure here)
    edges_dict = {}
    
    # Bridge amenities to segments
    _, bridged_edges = c2g.bridge_nodes(
        nodes_dict=nodes_dict,
        proximity_method="knn",
        source_node_types=["amenity"],
        target_node_types=["segment"],
        k=k
    )
    
    # Update edges dictionary
    edges_dict.update(bridged_edges)
    
    return nodes_dict, edges_dict


def compute_metapaths(nodes_dict, edges_dict, hops=3, directed=False):
    """
    Compute metapaths using c2g.add_metapaths.
    
    Args:
        nodes_dict: Dictionary of node type -> GeoDataFrame
        edges_dict: Dictionary of (src_type, rel, tgt_type) -> GeoDataFrame
        hops: Number of segment-to-segment hops
        directed: Whether to treat graph as directed
        
    Returns:
        tuple: (result_nodes, result_edges) with metapath edges added
    """
    import city2graph as c2g
    
    sequence = [("amenity", "is_nearby", "segment")]
    for _ in range(hops):
        sequence.append(("segment", "connects_to", "segment"))
    sequence.append(("segment", "is_nearby", "amenity"))
    
    result_nodes, result_edges = c2g.add_metapaths(
        graph=(nodes_dict, edges_dict),
        sequence=sequence,
        edge_attr=None,
        edge_attr_agg="sum",
        directed=directed,
        trace_path=False,
        as_nx=False,
        multigraph=True
    )
    
    return result_nodes, result_edges


def compute_metapaths_by_weight(nodes_dict, edges_dict, weight_attr, threshold, 
                                 endpoint_type="amenity", min_threshold=0.0,
                                 directed=False, new_relation_name=None):
    """
    Connect nodes of a specific type if reachable within a cost threshold band.
    
    New in city2graph 0.3.1. Uses Dijkstra's algorithm for path finding.
    
    Args:
        nodes_dict: Dictionary of node type -> GeoDataFrame
        edges_dict: Dictionary of (src_type, rel, tgt_type) -> GeoDataFrame
        weight_attr: Edge attribute to use as cost (e.g., 'travel_time')
        threshold: Maximum cost threshold for connection
        endpoint_type: Node type to connect (e.g., 'amenity', 'building')
        min_threshold: Minimum cost threshold (default 0.0)
        directed: Whether to treat graph as directed
        new_relation_name: Name for the new edge relation
    
    Returns:
        tuple: (result_nodes, result_edges) with weighted metapath edges
    """
    import city2graph as c2g
    
    result_nodes, result_edges = c2g.add_metapaths_by_weight(
        graph=(nodes_dict, edges_dict),
        weight=weight_attr,
        threshold=threshold,
        min_threshold=min_threshold,
        endpoint_type=endpoint_type,
        new_relation_name=new_relation_name,
        directed=directed,
        as_nx=False,
        multigraph=False
    )
    
    return result_nodes, result_edges


def extract_metapath_connections(result_edges, metapath_key=('amenity', 'metapath_0', 'amenity'), 
                                 add_multiplicity=True):
    """
    Extract metapath connections from result and optionally add multiplicity attribute.
    
    Args:
        result_edges: Dictionary of edge types from c2g.add_metapaths
        metapath_key: Key for metapath edges (default searches for metapath_0)
        add_multiplicity: If True, groups by endpoints and adds multiplicity column
        
    Returns:
        GeoDataFrame: Metapath edges with 'multiplicity' column if add_multiplicity=True
    """
    import geopandas as gpd
    import pandas as pd
    
    # Try exact key first
    metapath_gdf = None
    if metapath_key in result_edges:
        metapath_gdf = result_edges[metapath_key]
    else:
        # Search for any metapath key
        for key in result_edges.keys():
            if 'metapath' in key[1]:
                metapath_gdf = result_edges[key]
                break
    
    if metapath_gdf is None or len(metapath_gdf) == 0:
        return None
    
    if not add_multiplicity:
        return metapath_gdf
    
    # Group by endpoints and calculate multiplicity
    grouped_metapaths = []
    
    for idx, row in metapath_gdf.iterrows():
        if hasattr(row, 'geometry') and row.geometry:
            coords = list(row.geometry.coords)
            if len(coords) >= 2:
                # Create key from start/end coordinates
                start = (round(coords[0][0], 6), round(coords[0][1], 6))
                end = (round(coords[-1][0], 6), round(coords[-1][1], 6))
                # Symmetric key for undirected
                key = tuple(sorted([start, end]))
                
                grouped_metapaths.append({
                    'geometry': row.geometry,
                    'key': key,
                    'index': idx
                })
    
    # Count multiplicity and create unique GeoDataFrame
    from collections import defaultdict
    key_to_data = defaultdict(list)
    
    for item in grouped_metapaths:
        key_to_data[item['key']].append(item)
    
    # Build result with multiplicity
    unique_rows = []
    for key, items in key_to_data.items():
        # Use first geometry as representative
        first_item = items[0]
        multiplicity = len(items)
        
        # Get original row data
        original_row = metapath_gdf.loc[first_item['index']].copy()
        
        # Create new row with multiplicity
        row_dict = {
            'geometry': first_item['geometry'],
            'multiplicity': multiplicity
        }
        
        # Copy other attributes if they exist
        for col in metapath_gdf.columns:
            if col != 'geometry':
                row_dict[col] = original_row[col]
        
        unique_rows.append(row_dict)
    
    # Create GeoDataFrame
    result_gdf = gpd.GeoDataFrame(unique_rows, crs=metapath_gdf.crs)
    
    print(f"Metapaths grouped: {len(metapath_gdf)} raw → {len(result_gdf)} unique with multiplicity")
    
    return result_gdf

