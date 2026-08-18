from scigraphs_core.logger import log
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

    # OSMnx can only do this on an unprojected graph.
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
    """Initial compass bearing from one lat/lon point to another, in decimal degrees.

    Measured clockwise from north, 0 to 360. Vectorizes over arrays.
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
    """Add a compass 'bearing' (0-360) to every edge. G must be unprojected."""
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
    """Shannon entropy of the street orientation distribution.

    Low entropy means a few dominant headings, so a grid; high entropy means the
    streets point every which way. num_bins splits the full 360 degrees, weight
    names an edge attribute to weight by, and edges shorter than min_length drop
    out.
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
    """Edge bearings binned uniformly over 360 degrees, as ``(counts, bin_centers)``.

    Wraps the private ``ox.bearing._bearings_distribution``, so it can vanish in
    any OSMnx release; returns ``(None, None)`` when it does.
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
