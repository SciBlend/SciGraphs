from ...utils.logger import log
from .get_osmnx import get_osmnx


# NOTE: ``add_edge_lengths`` lives in ``core/osmnx/edge_attributes.py`` and is
# re-exported via ``core.osmnx.add_edge_lengths`` / ``osmnx_analysis``.  An older
# duplicate that lived here assumed ``MultiDiGraph`` and broke whenever the user
# converted the graph via ``to_digraph`` / ``to_undirected``.  It has been
# removed to keep a single, multigraph-tolerant implementation.


def euclidean(y1, x1, y2, x2):
    """
    Calculate Euclidean distance between points.
    
    For use with projected coordinate systems. Input arrays can be scalars
    or numpy arrays for vectorized computation.
    
    Args:
        y1: Y coordinate(s) of first point(s)
        x1: X coordinate(s) of first point(s)
        y2: Y coordinate(s) of second point(s)
        x2: X coordinate(s) of second point(s)
    
    Returns:
        float or array: Euclidean distance(s)
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "distance") and hasattr(ox.distance, "euclidean"):
        return ox.distance.euclidean(y1, x1, y2, x2)
    
    log("euclidean function not found in OSMnx")
    return None


def great_circle(lat1, lon1, lat2, lon2, earth_radius=6371009):
    """
    Calculate great circle distance between point(s) using haversine formula.
    
    For use with unprojected (lat/lon) coordinate systems. Input arrays can
    be scalars or numpy arrays for vectorized computation.
    
    Args:
        lat1: Latitude(s) of first point(s) in decimal degrees
        lon1: Longitude(s) of first point(s) in decimal degrees
        lat2: Latitude(s) of second point(s) in decimal degrees
        lon2: Longitude(s) of second point(s) in decimal degrees
        earth_radius: Radius of Earth in meters (default: 6371009)
    
    Returns:
        float or array: Great circle distance(s) in meters
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "distance") and hasattr(ox.distance, "great_circle"):
        return ox.distance.great_circle(lat1, lon1, lat2, lon2, earth_radius=earth_radius)
    
    log("great_circle function not found in OSMnx")
    return None


def nearest_nodes(G, X, Y, return_dist=False):
    """
    Find the nearest node(s) to a point or points.
    
    Args:
        G: networkx.MultiDiGraph
        X: X coordinate(s) (longitude if unprojected, easting if projected)
        Y: Y coordinate(s) (latitude if unprojected, northing if projected)
        return_dist: If True, return (node_ids, distances) tuple
    
    Returns:
        int, list, or tuple: Nearest node ID(s), optionally with distances
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "distance") and hasattr(ox.distance, "nearest_nodes"):
        result = ox.distance.nearest_nodes(G, X=X, Y=Y, return_dist=return_dist)
        log(f"Found nearest node(s)")
        return result
    elif hasattr(ox, "nearest_nodes"):
        result = ox.nearest_nodes(G, X=X, Y=Y, return_dist=return_dist)
        log(f"Found nearest node(s)")
        return result
    
    log("nearest_nodes function not found in OSMnx")
    return None


def nearest_edges(G, X, Y, return_dist=False):
    """
    Find the nearest edge(s) to a point or points.
    
    Args:
        G: networkx.MultiDiGraph
        X: X coordinate(s) (longitude if unprojected, easting if projected)
        Y: Y coordinate(s) (latitude if unprojected, northing if projected)
        return_dist: If True, return (edge_ids, distances) tuple
    
    Returns:
        tuple, list of tuples, or tuple: Nearest edge ID(s) as (u, v, key), 
        optionally with distances
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "distance") and hasattr(ox.distance, "nearest_edges"):
        result = ox.distance.nearest_edges(G, X=X, Y=Y, return_dist=return_dist)
        log(f"Found nearest edge(s)")
        return result
    elif hasattr(ox, "nearest_edges"):
        result = ox.nearest_edges(G, X=X, Y=Y, return_dist=return_dist)
        log(f"Found nearest edge(s)")
        return result
    
    log("nearest_edges function not found in OSMnx")
    return None

