from scigraphs_core.logger import log
from .get_osmnx import get_osmnx


def truncate_graph_bbox(G, bbox, truncate_by_edge=False, quadrat_width=0.05, min_num=3):
    """Drop nodes outside a ``(north, south, east, west)`` box.

    truncate_by_edge keeps any edge with one endpoint still inside. quadrat_width
    and min_num are accepted and ignored; OSMnx dropped them. OSMnx also swapped
    from four positional coordinates to a single ``bbox`` argument, so the call
    is chosen by inspecting the installed signature.
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    north, south, east, west = bbox
    
    if hasattr(ox, "truncate") and hasattr(ox.truncate, "truncate_graph_bbox"):
        import inspect
        sig = inspect.signature(ox.truncate.truncate_graph_bbox)
        
        if 'bbox' in sig.parameters:
            result = ox.truncate.truncate_graph_bbox(G, bbox, truncate_by_edge=truncate_by_edge)
        else:
            result = ox.truncate.truncate_graph_bbox(
                G, north, south, east, west, truncate_by_edge=truncate_by_edge
            )
        log(f"Graph truncated by bbox: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    elif hasattr(ox, "truncate_graph_bbox"):
        import inspect
        sig = inspect.signature(ox.truncate_graph_bbox)
        
        if 'bbox' in sig.parameters:
            result = ox.truncate_graph_bbox(G, bbox, truncate_by_edge=truncate_by_edge)
        else:
            result = ox.truncate_graph_bbox(
                G, north, south, east, west, truncate_by_edge=truncate_by_edge
            )
        log(f"Graph truncated by bbox: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    
    log("truncate_graph_bbox function not found in OSMnx")
    return None


def truncate_graph_polygon(G, polygon, truncate_by_edge=False, quadrat_width=0.05, min_num=3):
    """Drop nodes outside a shapely Polygon or MultiPolygon.

    truncate_by_edge keeps any edge with one endpoint still inside. quadrat_width
    and min_num are accepted and ignored; OSMnx dropped them.
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "truncate") and hasattr(ox.truncate, "truncate_graph_polygon"):
        result = ox.truncate.truncate_graph_polygon(
            G,
            polygon=polygon,
            truncate_by_edge=truncate_by_edge
        )
        log(f"Graph truncated by polygon: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    elif hasattr(ox, "truncate_graph_polygon"):
        result = ox.truncate_graph_polygon(
            G,
            polygon=polygon,
            truncate_by_edge=truncate_by_edge
        )
        log(f"Graph truncated by polygon: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    
    log("truncate_graph_polygon function not found in OSMnx")
    return None


def truncate_graph_dist(G, source_node, dist, weight="length"):
    """Drop nodes further than dist from source_node, measured along ``weight``."""
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "truncate") and hasattr(ox.truncate, "truncate_graph_dist"):
        result = ox.truncate.truncate_graph_dist(
            G,
            source_node=source_node,
            dist=dist,
            weight=weight
        )
        log(f"Graph truncated by distance: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    elif hasattr(ox, "truncate_graph_dist"):
        result = ox.truncate_graph_dist(
            G,
            source_node=source_node,
            dist=dist,
            weight=weight
        )
        log(f"Graph truncated by distance: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    
    log("truncate_graph_dist function not found in OSMnx")
    return None


def largest_component(G, strongly=False):
    """Largest connected component, weakly connected unless strongly is set."""
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "truncate") and hasattr(ox.truncate, "largest_component"):
        result = ox.truncate.largest_component(G, strongly=strongly)
        log(f"Extracted largest component: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    elif hasattr(ox, "largest_component"):
        result = ox.largest_component(G, strongly=strongly)
        log(f"Extracted largest component: {G.number_of_nodes()} -> {result.number_of_nodes()} nodes")
        return result
    
    log("largest_component function not found in OSMnx")
    return None

