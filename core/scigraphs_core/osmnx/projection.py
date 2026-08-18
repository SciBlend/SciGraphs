from scigraphs_core.logger import log
from .get_osmnx import get_osmnx


def project_graph(G, to_crs=None):
    """Project a lat/lon graph, to to_crs or to the local UTM zone when it is None.

    to_crs takes 'EPSG:27700', '27700' or 27700 alike. Returns
    ``(projected_graph, crs_string)``, or ``(None, message)`` on failure.
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
    """True if the graph's CRS is projected, False if it is lat/lon.

    Falls back to reading the first node's coordinates when pyproj is missing,
    on the assumption that anything inside +/-180 by +/-90 is degrees.
    """
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

