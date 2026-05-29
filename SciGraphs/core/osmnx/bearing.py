from ...utils.logger import log
from .get_osmnx import get_osmnx


def _ensure_bearings(G):
    """Ensure edges carry a 'bearing' attribute; compute if missing."""
    if G is None:
        return G
    try:
        for _u, _v, data in G.edges(data=True):
            if "bearing" in data:
                return G
            break
    except Exception:
        pass

    ox = get_osmnx()
    if ox is None:
        return G

    # Need unprojected graph for OSMnx to compute bearings.
    from .projection import is_graph_projected
    from . import convert as _convert

    G_target = _convert.ensure_multidigraph(G)

    if is_graph_projected(G_target):
        import math
        for u, v, data in G_target.edges(data=True):
            u_data = G_target.nodes[u]
            v_data = G_target.nodes[v]
            dx = v_data.get("x", 0) - u_data.get("x", 0)
            dy = v_data.get("y", 0) - u_data.get("y", 0)
            angle = math.degrees(math.atan2(dx, dy))
            data["bearing"] = (angle + 360) % 360
    else:
        try:
            if hasattr(ox, "bearing") and hasattr(ox.bearing, "add_edge_bearings"):
                G_target = ox.bearing.add_edge_bearings(G_target)
            else:
                G_target = ox.add_edge_bearings(G_target)
        except Exception as e:
            log(f"Failed to auto-compute bearings: {e}")
            return G

    return G_target


def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculate the compass bearing between pairs of lat-lon points.
    
    Vectorized function that calculates the initial bearing from (lat1, lon1) 
    to (lat2, lon2). Bearing is measured clockwise from north (0-360 degrees).
    
    Args:
        lat1: Starting latitude(s) in decimal degrees
        lon1: Starting longitude(s) in decimal degrees
        lat2: Ending latitude(s) in decimal degrees
        lon2: Ending longitude(s) in decimal degrees
    
    Returns:
        float or array: Bearing(s) in degrees (0-360)
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "bearing") and hasattr(ox.bearing, "calculate_bearing"):
        return ox.bearing.calculate_bearing(lat1, lon1, lat2, lon2)
    elif hasattr(ox, "calculate_bearing"):
        return ox.calculate_bearing(lat1, lon1, lat2, lon2)
    
    log("calculate_bearing function not found in OSMnx")
    return None


def add_edge_bearings(G):
    """
    Calculate and add bearing attributes to all edges in the graph.
    
    Bearings represent the compass direction (0-360 degrees) of each street
    segment. Graph must be unprojected (in lat-lon coordinates).
    
    Args:
        G: networkx.MultiDiGraph (unprojected)
    
    Returns:
        networkx.MultiDiGraph: Graph with 'bearing' attribute on edges
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "bearing") and hasattr(ox.bearing, "add_edge_bearings"):
        G = ox.bearing.add_edge_bearings(G)
    elif hasattr(ox, "add_edge_bearings"):
        G = ox.add_edge_bearings(G)
    else:
        log("add_edge_bearings function not found in OSMnx")
        return None
    
    log("Edge bearings added to graph")
    return G


def orientation_entropy(G, num_bins=36, min_length=0, weight=None):
    """
    Calculate the Shannon entropy of street network orientation.
    
    Higher entropy indicates more uniform distribution of street orientations
    (less grid-like). Lower entropy indicates dominant orientations (grid-like).
    
    Args:
        G: networkx.MultiDiGraph with 'bearing' attribute on edges
        num_bins: Number of bins to divide 360 degrees (default: 36)
        min_length: Minimum edge length to include (default: 0)
        weight: Edge attribute to use as weight (default: None)
    
    Returns:
        float: Shannon entropy of orientation distribution
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    G = _ensure_bearings(G)
    if hasattr(ox, "bearing") and hasattr(ox.bearing, "orientation_entropy"):
        result = ox.bearing.orientation_entropy(G, num_bins=num_bins, min_length=min_length, weight=weight)
        log(f"Orientation entropy: {result:.4f}")
        return result
    
    log("orientation_entropy function not found in OSMnx")
    return None


def get_bearings_distribution(G, num_bins=36, min_length=0, weight=None):
    """
    Calculate the distribution of edge bearings in uniform bins.
    
    Wrapper for internal OSMnx function to get bearing distribution data.
    
    Args:
        G: networkx.MultiDiGraph with 'bearing' attribute on edges
        num_bins: Number of bins to divide 360 degrees (default: 36)
        min_length: Minimum edge length to include (default: 0)
        weight: Edge attribute to use as weight (default: None)
    
    Returns:
        tuple: (counts array, bin_centers array) or (None, None) on error
    """
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None, None
    
    G = _ensure_bearings(G)
    if hasattr(ox, "bearing") and hasattr(ox.bearing, "_bearings_distribution"):
        return ox.bearing._bearings_distribution(G, num_bins=num_bins, min_length=min_length, weight=weight)
    
    log("_bearings_distribution function not found in OSMnx")
    return None, None


_bearings_distribution = get_bearings_distribution
