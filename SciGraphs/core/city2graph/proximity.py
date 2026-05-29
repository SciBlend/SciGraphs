"""
Proximity graph generation from Blender OSM feature objects.

Provides wrapper functions that convert Blender meshes to GeoDataFrames,
call city2graph proximity functions, and return results ready for visualization.
"""

def _extract_gdf_from_feature_object(obj, target_crs=None, deduplicate=True, tolerance=0.5):
    """
    Extract GeoDataFrame from Blender OSM feature object.
    
    Args:
        obj: Blender object with OSM features
        target_crs: Optional CRS to project to (defaults to UTM based on location)
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication (after CRS conversion)
        
    Returns:
        GeoDataFrame: Points extracted from vertices
        
    Note:
        OSM features can be nodes (points), ways (lines/polygons), or relations.
        When ways/relations are imported, each vertex becomes a point.
        Use deduplicate=True to consolidate nearby points (default).
    """
    if not obj or not obj.data:
        raise ValueError("Invalid object: no mesh data")
    
    if not obj.get("is_osm_features"):
        raise ValueError("Object is not an OSM features object")
    
    center_lat = obj.get("osmnx_center_lat")
    center_lon = obj.get("osmnx_center_lon")
    scale = obj.get("osmnx_scale", 0.001)
    
    if center_lat is None or center_lon is None:
        raise ValueError("Object missing coordinate transformation parameters")
    
    import geopandas as gpd
    from shapely.geometry import Point
    from ..mesh.geometry import _local_3d_to_latlon
    
    points = []
    for vert in obj.data.vertices:
        lat, lon = _local_3d_to_latlon(
            vert.co.x, vert.co.y,
            center_lat, center_lon, scale
        )
        points.append(Point(lon, lat))
    
    gdf = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")
    
    if target_crs:
        gdf = gdf.to_crs(target_crs)
    else:
        bounds = gdf.total_bounds
        center_lon_calc = (bounds[0] + bounds[2]) / 2
        import pyproj
        utm_crs = pyproj.CRS.from_proj4(
            f"+proj=utm +zone={int((center_lon_calc + 180) / 6) + 1} +datum=WGS84 +units=m +no_defs"
        )
        gdf = gdf.to_crs(utm_crs)
    
    if deduplicate and len(gdf) > 0:
        from shapely.ops import unary_union
        from shapely.geometry import MultiPoint
        
        buffered = gdf.geometry.buffer(tolerance / 2)
        dissolved = unary_union(buffered)
        
        if hasattr(dissolved, 'geoms'):
            centroids = [geom.centroid for geom in dissolved.geoms]
        else:
            centroids = [dissolved.centroid]
        
        gdf = gpd.GeoDataFrame(geometry=centroids, crs=gdf.crs)
    
    gdf = gdf.reset_index(drop=True)
    
    return gdf


def _extract_network_gdf(network_obj, target_crs=None):
    """
    Extract network GeoDataFrame from OSMnx street network object.
    
    Args:
        network_obj: Blender object with OSMnx street network
        target_crs: Optional CRS to project to (to match features CRS)
        
    Returns:
        GeoDataFrame: Street edges for network distance calculation
    """
    if not network_obj or not network_obj.get("is_osmnx"):
        raise ValueError("Invalid network object: not an OSMnx street network")
    
    import city2graph as c2g
    from .. import importer
    
    graph_id = network_obj.get("osmnx_graph_id")
    if not graph_id:
        raise ValueError("Network object missing graph ID")
    
    if not hasattr(importer, '_osmnx_graph_cache') or graph_id not in importer._osmnx_graph_cache:
        raise ValueError("Network graph not found in cache. Try re-importing the network.")
    
    osmnx_graph = importer._osmnx_graph_cache[graph_id]
    
    if osmnx_graph is None:
        raise ValueError("Could not retrieve graph from cache")
    
    _, edges_gdf = c2g.nx_to_gdf(osmnx_graph)
    
    if target_crs and edges_gdf.crs != target_crs:
        edges_gdf = edges_gdf.to_crs(target_crs)
    
    return edges_gdf


def generate_knn_graph_from_features(feature_obj, k=5, distance_metric="euclidean", 
                                     network_obj=None, deduplicate=True, tolerance=0.5,
                                     as_nx=False):
    """
    Generate K-Nearest Neighbors graph from OSM feature object.
    
    Args:
        feature_obj: Blender object with OSM features
        k: Number of nearest neighbors
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication
        as_nx: Return NetworkX graph instead of GeoDataFrames
        
    Returns:
        tuple: (nodes_gdf, edges_gdf) or NetworkX graph if as_nx=True
    """
    import city2graph as c2g
    
    gdf = _extract_gdf_from_feature_object(feature_obj, deduplicate=deduplicate, tolerance=tolerance)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=gdf.crs)
    
    result = c2g.knn_graph(
        gdf,
        k=k,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        as_nx=as_nx
    )
    
    return result


def generate_delaunay_graph_from_features(feature_obj, distance_metric="euclidean",
                                          network_obj=None, deduplicate=True, tolerance=0.5,
                                          as_nx=False):
    """
    Generate Delaunay triangulation graph from OSM feature object.
    
    Args:
        feature_obj: Blender object with OSM features
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication
        as_nx: Return NetworkX graph instead of GeoDataFrames
        
    Returns:
        tuple: (nodes_gdf, edges_gdf) or NetworkX graph if as_nx=True
    """
    import city2graph as c2g
    
    gdf = _extract_gdf_from_feature_object(feature_obj, deduplicate=deduplicate, tolerance=tolerance)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=gdf.crs)
    
    result = c2g.delaunay_graph(
        gdf,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        as_nx=as_nx
    )
    
    return result


def generate_fixed_radius_graph_from_features(feature_obj, radius=100.0, 
                                              distance_metric="euclidean",
                                              network_obj=None, deduplicate=True, tolerance=0.5,
                                              as_nx=False):
    """
    Generate fixed-radius (Gilbert) graph from OSM feature object.
    
    Args:
        feature_obj: Blender object with OSM features
        radius: Connection radius in meters
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication
        as_nx: Return NetworkX graph instead of GeoDataFrames
        
    Returns:
        tuple: (nodes_gdf, edges_gdf) or NetworkX graph if as_nx=True
    """
    import city2graph as c2g
    
    gdf = _extract_gdf_from_feature_object(feature_obj, deduplicate=deduplicate, tolerance=tolerance)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=gdf.crs)
    
    result = c2g.fixed_radius_graph(
        gdf,
        radius=radius,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        as_nx=as_nx
    )
    
    return result


def generate_waxman_graph_from_features(feature_obj, beta=0.5, r0=100.0, seed=None,
                                        distance_metric="euclidean", network_obj=None,
                                        deduplicate=True, tolerance=0.5,
                                        as_nx=False):
    """
    Generate Waxman probabilistic graph from OSM feature object.
    
    Args:
        feature_obj: Blender object with OSM features
        beta: Probability scaling parameter (0-1)
        r0: Maximum distance parameter
        seed: Random seed for reproducibility
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication
        as_nx: Return NetworkX graph instead of GeoDataFrames
        
    Returns:
        tuple: (nodes_gdf, edges_gdf) or NetworkX graph if as_nx=True
    """
    import city2graph as c2g
    
    gdf = _extract_gdf_from_feature_object(feature_obj, deduplicate=deduplicate, tolerance=tolerance)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=gdf.crs)
    
    result = c2g.waxman_graph(
        gdf,
        beta=beta,
        r0=r0,
        seed=seed,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        as_nx=as_nx
    )
    
    return result


def generate_gabriel_graph_from_features(feature_obj, distance_metric="euclidean",
                                        network_obj=None, deduplicate=True, tolerance=0.5,
                                        as_nx=False):
    """
    Generate Gabriel graph from OSM feature object.
    
    Args:
        feature_obj: Blender object with OSM features
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication
        as_nx: Return NetworkX graph instead of GeoDataFrames
        
    Returns:
        tuple: (nodes_gdf, edges_gdf) or NetworkX graph if as_nx=True
    """
    import city2graph as c2g
    
    gdf = _extract_gdf_from_feature_object(feature_obj, deduplicate=deduplicate, tolerance=tolerance)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=gdf.crs)
    
    result = c2g.gabriel_graph(
        gdf,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        as_nx=as_nx
    )
    
    return result


def generate_rng_graph_from_features(feature_obj, distance_metric="euclidean",
                                     network_obj=None, deduplicate=True, tolerance=0.5,
                                     as_nx=False):
    """
    Generate Relative Neighborhood Graph from OSM feature object.
    
    Args:
        feature_obj: Blender object with OSM features
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication
        as_nx: Return NetworkX graph instead of GeoDataFrames
        
    Returns:
        tuple: (nodes_gdf, edges_gdf) or NetworkX graph if as_nx=True
    """
    import city2graph as c2g
    
    gdf = _extract_gdf_from_feature_object(feature_obj, deduplicate=deduplicate, tolerance=tolerance)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=gdf.crs)
    
    result = c2g.relative_neighborhood_graph(
        gdf,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        as_nx=as_nx
    )
    
    return result


def generate_emst_graph_from_features(feature_obj, distance_metric="euclidean",
                                      network_obj=None, deduplicate=True, tolerance=0.5,
                                      as_nx=False):
    """
    Generate Euclidean Minimum Spanning Tree from OSM feature object.
    
    Args:
        feature_obj: Blender object with OSM features
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication
        as_nx: Return NetworkX graph instead of GeoDataFrames
        
    Returns:
        tuple: (nodes_gdf, edges_gdf) or NetworkX graph if as_nx=True
    """
    import city2graph as c2g
    
    gdf = _extract_gdf_from_feature_object(feature_obj, deduplicate=deduplicate, tolerance=tolerance)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=gdf.crs)
    
    result = c2g.euclidean_minimum_spanning_tree(
        gdf,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        as_nx=as_nx
    )
    
    return result


def generate_contiguity_graph_from_features(feature_obj, contiguity="queen",
                                            distance_metric="euclidean",
                                            network_obj=None, deduplicate=True, tolerance=0.5,
                                            as_nx=False):
    """
    Generate contiguity graph from polygon features.
    
    Args:
        feature_obj: Blender object with polygon features
        contiguity: 'queen' or 'rook'
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        deduplicate: If True, remove duplicate/nearby points
        tolerance: Distance threshold in meters for deduplication
        as_nx: Return NetworkX graph instead of GeoDataFrames
        
    Returns:
        tuple: (nodes_gdf, edges_gdf) or NetworkX graph if as_nx=True
        
    Note:
        Contiguity is determined by libpysal's spatial weights (Queen/Rook rules).
        No separate predicate parameter needed - contiguity type defines adjacency.
    """
    import city2graph as c2g
    
    gdf = _extract_gdf_from_feature_object(feature_obj, deduplicate=deduplicate, tolerance=tolerance)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=gdf.crs)
    
    result = c2g.contiguity_graph(
        gdf,
        contiguity=contiguity,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        as_nx=as_nx
    )
    
    return result


def generate_bridge_nodes_from_features(feature_objects, proximity_method="knn",
                                        k=1, radius=100.0, distance_metric="euclidean",
                                        network_obj=None, as_nx=False):
    """
    Generate multi-layer graph connecting different feature types.
    
    Args:
        feature_objects: Dictionary of {layer_name: blender_object}
        proximity_method: 'knn' or 'fixed_radius'
        k: Number of neighbors for KNN
        radius: Radius for fixed_radius
        distance_metric: 'euclidean', 'manhattan', or 'network'
        network_obj: Optional OSMnx street network for network distance
        as_nx: Return NetworkX graph instead of dictionaries
        
    Returns:
        tuple: (nodes_dict, edges_dict) or NetworkX graph if as_nx=True
        nodes_dict: {layer_name: gdf}
        edges_dict: {(src_layer, relation, tgt_layer): gdf}
    """
    import city2graph as c2g
    
    nodes_dict = {}
    for layer_name, obj in feature_objects.items():
        gdf = _extract_gdf_from_feature_object(obj)
        nodes_dict[layer_name] = gdf
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        first_gdf = list(nodes_dict.values())[0]
        network_gdf = _extract_network_gdf(network_obj, target_crs=first_gdf.crs)
    
    kwargs = {
        "distance_metric": distance_metric,
        "network_gdf": network_gdf,
    }
    
    if proximity_method == "knn":
        kwargs["k"] = k
    else:
        kwargs["radius"] = radius
    
    result = c2g.bridge_nodes(
        nodes_dict=nodes_dict,
        proximity_method=proximity_method,
        as_nx=as_nx,
        **kwargs
    )
    
    return result


def generate_group_nodes_from_features(polygons_obj, points_obj, 
                                       distance_metric="euclidean",
                                       predicate="covered_by",
                                       network_obj=None, as_nx=False):
    """
    Generate graph connecting polygon zones to contained points.
    
    Args:
        polygons_obj: Blender object with polygon features
        points_obj: Blender object with point features
        distance_metric: 'euclidean', 'manhattan', or 'network'
        predicate: Spatial predicate ('covered_by', 'within', 'intersects')
        network_obj: Optional OSMnx street network for network distance
        as_nx: Return NetworkX graph instead of dictionaries
        
    Returns:
        tuple: (nodes_dict, edges_dict) or NetworkX graph if as_nx=True
    """
    import city2graph as c2g
    
    polygons_gdf = _extract_gdf_from_feature_object(polygons_obj)
    points_gdf = _extract_gdf_from_feature_object(points_obj)
    
    network_gdf = None
    if distance_metric == "network" and network_obj:
        network_gdf = _extract_network_gdf(network_obj, target_crs=points_gdf.crs)
    
    result = c2g.group_nodes(
        polygons_gdf=polygons_gdf,
        points_gdf=points_gdf,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
        predicate=predicate,
        as_nx=as_nx
    )
    
    return result
