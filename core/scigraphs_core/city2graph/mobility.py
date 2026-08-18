"""Wrappers for city2graph's mobility functions, which turn Origin-Destination
data into spatial graphs. Needs city2graph 0.3.1 or newer.

Computation only: the seam to Blender is a GeoDataFrame, and the drawing side
lives a layer up in ``SciGraphs.ui.operators.city2graph.mobility_ops``.
"""

from scigraphs_core.logger import log
from .get_c2g import get_city2graph


def od_matrix_to_graph(od_data, zones_gdf, zone_id_col=None, matrix_type="edgelist",
                        source_col="source", target_col="target", weight_cols=None,
                        threshold=None, threshold_col=None, include_self_loops=False,
                        compute_edge_geometry=True, directed=True, as_nx=False):
    """Convert OD data, an edge list or an adjacency matrix, into a graph.

    ``matrix_type`` picks which of the two ``od_data`` is; ``source_col`` and
    ``target_col`` only apply to an edge list. ``threshold`` keeps flows at or
    above its value, read from ``threshold_col`` when there are several
    ``weight_cols``. With ``compute_edge_geometry``, edges get LineStrings
    between zone centroids. Returns ``(nodes_gdf, edges_gdf)``, or a NetworkX
    graph when ``as_nx``.
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


def od_flow_edges(od_data, zones_gdf, zone_id_col, weight_col=None,
                  threshold=None, limit=None):
    """Build the OD flow edge table a renderer can turn into curves.

    Resolves the OD data against the zone geometries, keeps the flows at or
    above ``threshold``, and returns edges with LineString geometry between
    zone centroids. ``limit`` caps how many come back, taking them in source
    order. Drawing them is the caller's business, whether that means Blender
    curves, a plot, or a GeoPackage on disk.

    The index is reset so the geometry column is addressable positionally,
    which ``create_curves_from_gdf`` relies on.
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
        return None

    _, edges_gdf = result
    edges_gdf = edges_gdf.reset_index()

    if limit is not None and limit > 0 and len(edges_gdf) > limit:
        log(f"Capping OD flow edges at {limit} of {len(edges_gdf)}")
        edges_gdf = edges_gdf.iloc[:limit]

    log(f"Prepared {len(edges_gdf)} OD flow edges")
    return edges_gdf
