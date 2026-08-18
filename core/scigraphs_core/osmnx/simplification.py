from scigraphs_core.logger import log
from .get_osmnx import get_osmnx


def simplify_graph(G, node_attrs_include=None, strict=True, remove_rings=True, track_merged=False):
    """Drop degree-2 nodes, merging their two edges into one that keeps the geometry.

    Which keywords OSMnx accepts moves between releases, so the installed
    signature is inspected and only the supported ones are passed through.
    ``strict`` is one of the casualties.
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
    """Merge clusters of nodes within tolerance into one node each, so a big
    junction stops counting as a dozen intersections.

    Project the graph first: tolerance is in meters, and on a lat/lon graph it
    would be read as degrees. Without rebuild_graph the return is a GeoSeries of
    the consolidated points rather than a graph.
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

