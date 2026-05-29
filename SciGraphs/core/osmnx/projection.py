from ...utils.logger import log
from .get_osmnx import get_osmnx


def project_graph(G, to_crs=None):
    """
    Project graph to a specified CRS or local UTM.
    
    Args:
        G: OSMnx MultiDiGraph in lat/lon coordinates
        to_crs: Target CRS (e.g., 'EPSG:27700', 27700, or None for auto-UTM)
    
    Returns:
        Tuple of (projected_graph, crs_string) or (None, error_message)
    """
    ox = get_osmnx()
    if ox is None:
        return None, "OSMnx not available"
    
    if G is None:
        return None, "No graph provided"
    
    try:
        if to_crs:
            if isinstance(to_crs, str):
                if to_crs.isdigit():
                    to_crs = int(to_crs)
                elif not to_crs.upper().startswith('EPSG:'):
                    to_crs = f"EPSG:{to_crs}"
            G_proj = ox.project_graph(G, to_crs=to_crs)
        else:
            G_proj = ox.project_graph(G)
        
        crs = G_proj.graph.get("crs", "Unknown CRS")
        log(f"Graph projected to {crs}")
        return G_proj, str(crs)
    except Exception as e:
        log(f"Error projecting graph: {e}")
        return None, str(e)


def is_graph_projected(G):
    """Check if a graph is projected (in meters) or unprojected (in lat/lon)."""
    if G is None:
        return False
    crs = G.graph.get("crs")
    if crs is None:
        return False
    try:
        import pyproj
        crs_obj = pyproj.CRS(crs)
        return crs_obj.is_projected
    except Exception:
        for node, data in G.nodes(data=True):
            x = data.get("x", 0)
            y = data.get("y", 0)
            if abs(x) <= 180 and abs(y) <= 90:
                return False
            return True
        return False

