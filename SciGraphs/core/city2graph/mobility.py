"""
Mobility / OD matrix utilities wrapper.

This module provides wrappers for city2graph's mobility functions,
which convert Origin-Destination (OD) data into spatial graph representations.

New in city2graph 0.3.1.
"""

from ...utils.logger import log
from .get_c2g import get_city2graph


def od_matrix_to_graph(od_data, zones_gdf, zone_id_col=None, matrix_type="edgelist",
                        source_col="source", target_col="target", weight_cols=None,
                        threshold=None, threshold_col=None, include_self_loops=False,
                        compute_edge_geometry=True, directed=True, as_nx=False):
    """
    Convert OD data (edge list or adjacency matrix) into graph structures.
    
    Creates spatially-aware graphs from OD data following city2graph's 
    GeoDataFrame-first design.
    
    Args:
        od_data: DataFrame or ndarray with OD flow data
        zones_gdf: GeoDataFrame of zones with unique identifiers
        zone_id_col: Name of the zone ID column in zones_gdf
        matrix_type: 'edgelist' or 'adjacency'
        source_col: Column name for origins (edgelist mode)
        target_col: Column name for destinations (edgelist mode)
        weight_cols: Edge list weight columns to preserve
        threshold: Minimum flow retained (>=) applied to threshold_col
        threshold_col: Column for thresholding when multiple weight_cols
        include_self_loops: Keep flows where origin == destination
        compute_edge_geometry: Build LineString geometries from centroids
        directed: Build directed or undirected graph
        as_nx: Return NetworkX graph instead of GeoDataFrames
    
    Returns:
        tuple(GeoDataFrame, GeoDataFrame) or NetworkX graph
    """
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.mobility import od_matrix_to_graph as c2g_od_to_graph
        
        log(f"Converting OD matrix to graph (type={matrix_type}, directed={directed})...")
        
        result = c2g_od_to_graph(
            od_data=od_data,
            zones_gdf=zones_gdf,
            zone_id_col=zone_id_col,
            matrix_type=matrix_type,
            source_col=source_col,
            target_col=target_col,
            weight_cols=weight_cols,
            threshold=threshold,
            threshold_col=threshold_col,
            include_self_loops=include_self_loops,
            compute_edge_geometry=compute_edge_geometry,
            directed=directed,
            as_nx=as_nx
        )
        
        if as_nx:
            log(f"Created NetworkX graph: {result.number_of_nodes()} nodes, {result.number_of_edges()} edges")
        else:
            nodes_gdf, edges_gdf = result
            log(f"Created graph: {len(nodes_gdf)} nodes, {len(edges_gdf)} edges")
        
        return result
        
    except Exception as e:
        log(f"Error converting OD matrix to graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_od_graph_blender(od_data, zones_gdf, zone_id_col, osmnx_obj=None,
                            matrix_type="edgelist", directed=True, 
                            weight_cols=None, threshold=None):
    """
    Create Blender mesh graph from OD matrix data.
    
    Args:
        od_data: DataFrame with OD flow data
        zones_gdf: GeoDataFrame of zones
        zone_id_col: Zone ID column name
        osmnx_obj: Optional OSMnx object for coordinate alignment
        matrix_type: 'edgelist' or 'adjacency'
        directed: Build directed graph
        weight_cols: Flow columns to preserve
        threshold: Minimum flow threshold
    
    Returns:
        Blender object with OD graph
    """
    result = od_matrix_to_graph(
        od_data=od_data,
        zones_gdf=zones_gdf,
        zone_id_col=zone_id_col,
        matrix_type=matrix_type,
        weight_cols=weight_cols,
        threshold=threshold,
        directed=directed,
        as_nx=True
    )
    
    if result is None:
        return None
    
    try:
        from .morphology import create_graph_from_networkx
        
        graph_obj = create_graph_from_networkx(
            result,
            name="OD_Graph",
            use_positions=True,
            osmnx_obj=osmnx_obj
        )
        
        if graph_obj:
            graph_obj["is_od_graph"] = True
            graph_obj["od_directed"] = directed
            if threshold:
                graph_obj["od_threshold"] = threshold
        
        return graph_obj
        
    except Exception as e:
        log(f"Error creating Blender OD graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_od_flows(od_data, zones_gdf, zone_id_col, feature_obj=None,
                       weight_col=None, threshold=None, curve_thickness=0.0002,
                       limit=1000):
    """
    Visualize OD flows as curves in Blender.
    
    Args:
        od_data: DataFrame with OD flow data
        zones_gdf: GeoDataFrame of zones
        zone_id_col: Zone ID column name
        feature_obj: Source feature object for coordinate transform
        weight_col: Column to use for flow weights
        threshold: Minimum flow to visualize
        curve_thickness: Bevel depth for curves
        limit: Maximum number of flows to visualize
    
    Returns:
        list: Created curve objects
    """
    result = od_matrix_to_graph(
        od_data=od_data,
        zones_gdf=zones_gdf,
        zone_id_col=zone_id_col,
        matrix_type="edgelist",
        weight_cols=[weight_col] if weight_col else None,
        threshold=threshold,
        directed=True,
        as_nx=False
    )
    
    if result is None:
        return []
    
    _, edges_gdf = result
    
    try:
        from ..mesh.geo_mesh import create_curves_from_gdf
        
        edges_gdf = edges_gdf.reset_index()
        
        curves_obj = create_curves_from_gdf(
            edges_gdf,
            name="OD_Flows",
            feature_obj=feature_obj,
            thickness=curve_thickness,
            limit=limit
        )
        
        if curves_obj:
            curves_obj["is_od_flow"] = True
            log(f"Created OD flow curves with {min(len(edges_gdf), limit)} edges")
            return [curves_obj]
        
        return []
        
    except Exception as e:
        log(f"Error visualizing OD flows: {e}")
        import traceback
        traceback.print_exc()
        return []
