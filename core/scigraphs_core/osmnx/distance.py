from scigraphs_core.logger import log
from .get_osmnx import get_osmnx


# ``add_edge_lengths`` lives in ``edge_attributes.py``, where it also handles the
# DiGraph and undirected graphs that ``to_digraph`` / ``to_undirected`` produce.


def euclidean(y1, x1, y2, x2):
    """Euclidean distance between points in a projected CRS. Scalars or arrays."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "distance") and hasattr(ox.distance, "euclidean"):
        return ox.distance.euclidean(y1, x1, y2, x2)
    
    log("euclidean function not found in OSMnx")
    return None


def great_circle(lat1, lon1, lat2, lon2, earth_radius=6371009):
    """Haversine distance in meters between lat/lon points; scalars or numpy arrays."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "distance") and hasattr(ox.distance, "great_circle"):
        return ox.distance.great_circle(lat1, lon1, lat2, lon2, earth_radius=earth_radius)
    
    log("great_circle function not found in OSMnx")
    return None


def nearest_nodes(G, X, Y, return_dist=False):
    """Nearest node(s) to one point or many. X/Y are lon/lat when the graph is
    unprojected and easting/northing when it is; with return_dist the result
    becomes a ``(node_ids, distances)`` tuple."""
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
    """Nearest edge(s), as ``(u, v, key)``, to one point or many. X/Y follow the
    graph's CRS as in nearest_nodes; with return_dist the result becomes an
    ``(edge_ids, distances)`` tuple."""
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

