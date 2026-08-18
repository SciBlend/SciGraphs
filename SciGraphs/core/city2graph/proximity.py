"""
Proximity graph generation from Blender feature objects.

Wrapper functions that convert Blender meshes (OSMnx or Overture/city2graph
feature objects) to GeoDataFrames, call the matching city2graph proximity
function, and return results ready for visualization.

Every `generate_*_from_features` below shares the same parameters, documented
here rather than on each one:

    feature_obj      Blender object carrying OSM or Overture features
    distance_metric  'euclidean', 'manhattan', or 'network'
    network_obj      OSMnx street network, required by 'network' distance
    deduplicate      collapse duplicate and near-coincident points
    tolerance        deduplication threshold in meters, after CRS conversion
    as_nx            return a NetworkX graph instead of GeoDataFrames

and each returns `(nodes_gdf, edges_gdf)`, or a NetworkX graph when `as_nx`.
The docstrings below cover only what is specific to that generator.
"""

def _extract_gdf_from_feature_object(obj, target_crs=None, deduplicate=True,
                                     tolerance=0.5, as_points=True):
    """
    Extract a GeoDataFrame from a Blender feature object.
    
    `target_crs` defaults to the UTM zone for the object's location.

    A point object maps each vertex to one node. With `as_points`, polygon and
    line objects such as Overture buildings are reduced to one representative
    point per feature, so they can serve as proximity-graph nodes; pass False to
    keep the original geometry, which containment predicates need.
    """
    if not obj or not obj.data:
        raise ValueError("Invalid object: no mesh data")
    
    if not obj.get("is_osm_features") and not obj.get("is_city2graph"):
        raise ValueError("Object is not a feature object (OSM or city2graph)")
    
    from . import utils as c2g_utils
    
    gdf = c2g_utils.blender_to_geopandas(obj, crs="EPSG:4326")
    
    if gdf is None or len(gdf) == 0:
        raise ValueError("Object missing coordinate transformation parameters")
    
    if as_points and not (gdf.geom_type == "Point").all():
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.representative_point()
    
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
    
    if deduplicate and len(gdf) > 0 and (gdf.geom_type == "Point").all():
        import geopandas as gpd
        from shapely.ops import unary_union
        
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
    """Return the street edges of an OSMnx object, projected to ``target_crs``.

    The graph itself comes from the importer's in-memory cache, so a network
    imported in an earlier session raises and has to be re-imported.
    """
    if not network_obj or not network_obj.get("is_osmnx"):
        raise ValueError("Invalid network object: not an OSMnx street network")
    
    import city2graph as c2g
    from ...core import importer
    
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
    
    `k` is the number of nearest neighbors.
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
    """Generate Delaunay triangulation graph from OSM feature object."""
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
    
    `radius` is the connection radius in meters.
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
    
    `beta` scales the connection probability (0-1), `r0` is the maximum
    distance, and `seed` makes the draw reproducible.
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
    """Generate Gabriel graph from OSM feature object."""
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
    """Generate Relative Neighborhood Graph from OSM feature object."""
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
    """Generate Euclidean Minimum Spanning Tree from OSM feature object."""
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
    
    `feature_obj` must carry polygons. `contiguity` is 'queen' or 'rook', and
    adjacency comes from libpysal's spatial weights, so there is no separate
    predicate parameter.
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
    
    `feature_objects` maps layer name to Blender object. `proximity_method` is
    'knn' (using `k`) or 'fixed_radius' (using `radius`).

    Returns `(nodes_dict, edges_dict)`, keyed by layer name and by the
    `(src_layer, relation, tgt_layer)` triplet respectively, or a NetworkX graph
    when `as_nx`.
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
    
    `predicate` is the spatial test: 'covered_by', 'within' or 'intersects'.
    Returns `(nodes_dict, edges_dict)`, or a NetworkX graph when `as_nx`.
    """
    import city2graph as c2g
    
    polygons_gdf = _extract_gdf_from_feature_object(polygons_obj, as_points=False)
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
