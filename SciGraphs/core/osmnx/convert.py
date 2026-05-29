from ...utils.logger import log
from .get_osmnx import get_osmnx


def ensure_multidigraph(G):
    """Return a MultiDiGraph equivalent of G. If G is already multi+directed,
    returns it unchanged. Otherwise rebuilds a MultiDiGraph so OSMnx internals
    that rely on G.edges(keys=True) / G.has_edge(u, v, k) keep working.

    This is important after the user applies to_digraph / to_undirected — OSMnx's
    own convert / bearing / add_edge_speeds helpers *require* a MultiDiGraph.
    """
    if G is None:
        return None
    try:
        import networkx as nx
    except ImportError:
        return G

    is_multi = getattr(G, "is_multigraph", lambda: False)()
    is_directed = getattr(G, "is_directed", lambda: False)()

    if is_multi and is_directed:
        return G

    H = nx.MultiDiGraph()
    H.graph.update(getattr(G, "graph", {}) or {})
    for node, data in G.nodes(data=True):
        H.add_node(node, **(data or {}))

    if is_multi:
        for u, v, k, data in G.edges(keys=True, data=True):
            H.add_edge(u, v, key=k, **(data or {}))
            if not is_directed:
                H.add_edge(v, u, **(data or {}))
    else:
        for u, v, data in G.edges(data=True):
            H.add_edge(u, v, **(data or {}))
            if not is_directed:
                H.add_edge(v, u, **(data or {}))

    return H


def graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True):
    """
    Convert a NetworkX MultiDiGraph to node and/or edge GeoDataFrames.
    
    Args:
        G: networkx.MultiDiGraph
        nodes: If True, return node GeoDataFrame
        edges: If True, return edge GeoDataFrame
        node_geometry: If True, add geometry column to nodes
        fill_edge_geometry: If True, add geometry to edges missing it
    
    Returns:
        GeoDataFrame or tuple: Single GeoDataFrame or tuple of (nodes_gdf, edges_gdf)
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "convert") and hasattr(ox.convert, "graph_to_gdfs"):
        return ox.convert.graph_to_gdfs(G, nodes=nodes, edges=edges, 
                                       node_geometry=node_geometry, 
                                       fill_edge_geometry=fill_edge_geometry)
    elif hasattr(ox, "graph_to_gdfs"):
        return ox.graph_to_gdfs(G, nodes=nodes, edges=edges, 
                               node_geometry=node_geometry, 
                               fill_edge_geometry=fill_edge_geometry)
    
    log("graph_to_gdfs function not found in OSMnx")
    return None


def graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=None):
    """
    Convert node and edge GeoDataFrames to a NetworkX MultiDiGraph.
    
    Args:
        gdf_nodes: GeoDataFrame of graph nodes
        gdf_edges: GeoDataFrame of graph edges
        graph_attrs: Dict of graph-level attributes to add
    
    Returns:
        networkx.MultiDiGraph: Reconstructed graph
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "convert") and hasattr(ox.convert, "graph_from_gdfs"):
        return ox.convert.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=graph_attrs)
    elif hasattr(ox, "graph_from_gdfs"):
        return ox.graph_from_gdfs(gdf_nodes, gdf_edges, graph_attrs=graph_attrs)
    
    log("graph_from_gdfs function not found in OSMnx")
    return None


def to_digraph(G, weight="length"):
    """
    Convert a MultiDiGraph to a DiGraph.
    
    Selects a single edge between parallel edges based on minimum weight.
    
    Args:
        G: networkx.MultiDiGraph
        weight: Edge attribute to use for selecting among parallel edges
    
    Returns:
        networkx.DiGraph: Simplified directed graph
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    G = ensure_multidigraph(G)
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_digraph"):
        result = ox.convert.to_digraph(G, weight=weight)
        log(f"Converted to DiGraph (weight: {weight})")
        return result
    elif hasattr(ox, "to_digraph"):
        result = ox.to_digraph(G, weight=weight)
        log(f"Converted to DiGraph (weight: {weight})")
        return result
    
    log("to_digraph function not found in OSMnx")
    return None


def to_undirected(G):
    """
    Convert a directed MultiDiGraph to an undirected MultiGraph.
    
    Args:
        G: networkx.MultiDiGraph
    
    Returns:
        networkx.MultiGraph: Undirected graph
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    G = ensure_multidigraph(G)
    if hasattr(ox, "convert") and hasattr(ox.convert, "to_undirected"):
        result = ox.convert.to_undirected(G)
        log("Converted to undirected graph")
        return result
    elif hasattr(ox, "to_undirected"):
        result = ox.to_undirected(G)
        log("Converted to undirected graph")
        return result
    
    log("to_undirected function not found in OSMnx")
    return None
