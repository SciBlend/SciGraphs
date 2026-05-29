from ...utils.logger import log
from .get_osmnx import get_osmnx


def simplify_graph(G, node_attrs_include=None, strict=True, remove_rings=True, track_merged=False):
    """
    Simplify graph topology by removing interstitial nodes.
    
    Nodes that are not intersections (degree 2) are removed and their incident
    edges are merged into a single edge with combined geometry and attributes.
    
    Args:
        G: networkx.MultiDiGraph
        node_attrs_include: Node attributes to preserve during merging
        strict: If True, only remove nodes with exactly 2 neighbors (legacy parameter)
        remove_rings: If True, remove self-contained rings
        track_merged: If True, add list of merged edges to resulting edges
    
    Returns:
        networkx.MultiDiGraph: Simplified graph
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "simplification") and hasattr(ox.simplification, "simplify_graph"):
        import inspect
        sig = inspect.signature(ox.simplification.simplify_graph)
        
        kwargs = {}
        if 'node_attrs_include' in sig.parameters:
            kwargs['node_attrs_include'] = node_attrs_include
        if 'strict' in sig.parameters:
            kwargs['strict'] = strict
        if 'remove_rings' in sig.parameters:
            kwargs['remove_rings'] = remove_rings
        if 'track_merged' in sig.parameters:
            kwargs['track_merged'] = track_merged
        
        result = ox.simplification.simplify_graph(G, **kwargs)
        log(f"Graph simplified: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    elif hasattr(ox, "simplify_graph"):
        import inspect
        sig = inspect.signature(ox.simplify_graph)
        
        kwargs = {}
        if 'node_attrs_include' in sig.parameters:
            kwargs['node_attrs_include'] = node_attrs_include
        if 'strict' in sig.parameters:
            kwargs['strict'] = strict
        if 'remove_rings' in sig.parameters:
            kwargs['remove_rings'] = remove_rings
        if 'track_merged' in sig.parameters:
            kwargs['track_merged'] = track_merged
        
        result = ox.simplify_graph(G, **kwargs)
        log(f"Graph simplified: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    
    log("simplify_graph function not found in OSMnx")
    return None


def consolidate_intersections(G, tolerance=10, rebuild_graph=True, dead_ends=False, reconnect_edges=True):
    """
    Consolidate nearby intersections into a single node.
    
    Merges complex intersections (clusters of nodes within tolerance distance)
    into a single representative node. Useful for cleaning up topology.
    
    Args:
        G: networkx.MultiDiGraph (should be projected)
        tolerance: Distance threshold in meters for consolidation
        rebuild_graph: If True, return rebuilt graph; if False, return GeoSeries of points
        dead_ends: If True, include dead-end nodes in consolidation
        reconnect_edges: If True, reconnect edges to consolidated nodes
    
    Returns:
        networkx.MultiDiGraph or GeoSeries: Consolidated graph or node points
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "simplification") and hasattr(ox.simplification, "consolidate_intersections"):
        result = ox.simplification.consolidate_intersections(
            G,
            tolerance=tolerance,
            rebuild_graph=rebuild_graph,
            dead_ends=dead_ends,
            reconnect_edges=reconnect_edges
        )
        if rebuild_graph:
            log(f"Intersections consolidated: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        else:
            log(f"Consolidated {len(result)} intersection points")
        return result
    elif hasattr(ox, "consolidate_intersections"):
        result = ox.consolidate_intersections(
            G,
            tolerance=tolerance,
            rebuild_graph=rebuild_graph,
            dead_ends=dead_ends,
            reconnect_edges=reconnect_edges
        )
        if rebuild_graph:
            log(f"Intersections consolidated: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        else:
            log(f"Consolidated {len(result)} intersection points")
        return result
    
    log("consolidate_intersections function not found in OSMnx")
    return None

