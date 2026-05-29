from ...utils.logger import log
from .get_osmnx import get_osmnx


def truncate_graph_bbox(G, bbox, truncate_by_edge=False, quadrat_width=0.05, min_num=3):
    """
    Remove nodes outside a bounding box.
    
    Args:
        G: networkx.MultiDiGraph
        bbox: Tuple of (north, south, east, west) coordinates
        truncate_by_edge: If True, retain edges with at least one endpoint in bbox
        quadrat_width: Not used in newer versions
        min_num: Not used in newer versions
    
    Returns:
        networkx.MultiDiGraph: Truncated graph
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
    """
    Remove nodes outside a polygon boundary.
    
    Args:
        G: networkx.MultiDiGraph
        polygon: Shapely Polygon or MultiPolygon
        truncate_by_edge: If True, retain edges with at least one endpoint in polygon
        quadrat_width: Not used in newer versions
        min_num: Not used in newer versions
    
    Returns:
        networkx.MultiDiGraph: Truncated graph
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
    """
    Remove nodes beyond a network distance from source node.
    
    Args:
        G: networkx.MultiDiGraph
        source_node: Node ID to measure distance from
        dist: Maximum network distance to retain
        weight: Edge attribute to use as distance weight
    
    Returns:
        networkx.MultiDiGraph: Truncated graph
    """
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
    """
    Extract the largest connected component from the graph.
    
    Args:
        G: networkx.MultiDiGraph
        strongly: If True, extract largest strongly connected component;
                  if False, extract largest weakly connected component
    
    Returns:
        networkx.MultiDiGraph: Largest component subgraph
    """
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

