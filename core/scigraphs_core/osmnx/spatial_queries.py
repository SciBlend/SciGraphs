from scigraphs_core.logger import log
from .get_osmnx import get_osmnx


def find_nearest_node(G, x, y, is_projected=None):
    """Nearest node to (x, y): lon/lat if the graph is unprojected, easting/northing if not."""
    ox = get_osmnx()
    if G is None:
        return None
    
    try:
        if ox is not None:
            if hasattr(ox, "distance") and hasattr(ox.distance, "nearest_nodes"):
                node_id = ox.distance.nearest_nodes(G, X=x, Y=y)
            else:
                node_id = ox.nearest_nodes(G, X=x, Y=y)
            
            log(f"Nearest node to ({y}, {x}): {node_id}")
            return node_id
    except Exception as e:
        log(f"OSMnx nearest_nodes failed ({e}), using numpy fallback")
    
    return _find_nearest_node_numpy(G, x, y)


def _find_nearest_node_numpy(G, x, y):
    """Nearest node by brute-force Euclidean scan, for installs without scikit-learn."""
    import numpy as np
    
    if G is None or G.number_of_nodes() == 0:
        return None
    
    node_ids = []
    coords = []
    
    for node, data in G.nodes(data=True):
        node_x = data.get("x")
        node_y = data.get("y")
        if node_x is not None and node_y is not None:
            node_ids.append(node)
            coords.append([node_x, node_y])
    
    if not coords:
        return None
    
    coords = np.array(coords)
    query_point = np.array([x, y])
    distances = np.sqrt(np.sum((coords - query_point) ** 2, axis=1))
    
    nearest_idx = np.argmin(distances)
    node_id = node_ids[nearest_idx]
    
    log(f"Nearest node (numpy fallback) to ({y}, {x}): {node_id}")
    return node_id


def find_nearest_edge(G, x, y, is_projected=None):
    """Nearest edge to (x, y), as a ``(u, v, key)`` tuple."""
    ox = get_osmnx()
    if G is None:
        return None
    
    try:
        if ox is not None:
            if hasattr(ox, "distance") and hasattr(ox.distance, "nearest_edges"):
                edge = ox.distance.nearest_edges(G, X=x, Y=y)
            else:
                edge = ox.nearest_edges(G, X=x, Y=y)
            
            log(f"Nearest edge to ({y}, {x}): {edge}")
            return edge
    except Exception as e:
        log(f"OSMnx nearest_edges failed ({e}), using numpy fallback")
    
    return _find_nearest_edge_numpy(G, x, y)


def _find_nearest_edge_numpy(G, x, y):
    """Nearest edge without scikit-learn, by midpoint, so a long edge passing close
    by can lose to one whose center is nearer."""
    import numpy as np
    
    if G is None or G.number_of_edges() == 0:
        return None
    
    query_point = np.array([x, y])
    
    min_dist = float("inf")
    nearest_edge = None
    
    is_multi = getattr(G, "is_multigraph", lambda: False)()
    if is_multi:
        edge_iter = G.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, 0, data) for u, v, data in G.edges(data=True))

    for u, v, key, data in edge_iter:
        u_data = G.nodes.get(u, {})
        v_data = G.nodes.get(v, {})
        
        u_x = u_data.get("x")
        u_y = u_data.get("y")
        v_x = v_data.get("x")
        v_y = v_data.get("y")
        
        if None in (u_x, u_y, v_x, v_y):
            continue
        
        mid_x = (u_x + v_x) / 2
        mid_y = (u_y + v_y) / 2
        
        dist = np.sqrt((mid_x - x) ** 2 + (mid_y - y) ** 2)
        
        if dist < min_dist:
            min_dist = dist
            nearest_edge = (u, v, key)
    
    if nearest_edge:
        log(f"Nearest edge (numpy fallback) to ({y}, {x}): {nearest_edge}")
    
    return nearest_edge


def get_node_coordinates(G, node_id):
    """Return a node's ``(lat, lon)``, or None if it has no coordinates."""
    if G is None or node_id not in G.nodes:
        return None
    
    try:
        node_data = G.nodes[node_id]
        lat = node_data.get("y")
        lon = node_data.get("x")
        if lat is not None and lon is not None:
            return (lat, lon)
        return None
    except Exception:
        return None


def get_edge_info(G, u, v, key=0):
    if G is None:
        return None
    
    try:
        if G.has_edge(u, v, key):
            edge_data = G.edges[u, v, key]
            return dict(edge_data)
        return None
    except Exception:
        return None

