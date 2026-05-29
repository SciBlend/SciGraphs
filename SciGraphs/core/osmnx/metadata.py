from ...utils.logger import log
from .projection import is_graph_projected


def get_graph_extent(G):
    """
    Get the geographic extent (bounding box) of the graph.
    
    Args:
        G: OSMnx MultiDiGraph
    
    Returns:
        Dictionary with north, south, east, west bounds, or None on error
    """
    if G is None:
        return None
    
    try:
        import numpy as np
        
        lats = []
        lons = []
        for node, data in G.nodes(data=True):
            if "y" in data and "x" in data:
                lats.append(data["y"])
                lons.append(data["x"])
        
        if not lats or not lons:
            return None
        
        return {
            "north": max(lats),
            "south": min(lats),
            "east": max(lons),
            "west": min(lons),
            "center_lat": np.mean(lats),
            "center_lon": np.mean(lons),
        }
    except Exception as e:
        log(f"Error getting graph extent: {e}")
        return None


def estimate_network_area(G):
    """
    Estimate the area covered by the network in square kilometers.
    Uses the convex hull of all nodes.
    
    Args:
        G: OSMnx MultiDiGraph
    
    Returns:
        Area in km2, or None on error
    """
    if G is None:
        return None
    
    try:
        from shapely.geometry import MultiPoint
        import numpy as np
        
        points = []
        for node, data in G.nodes(data=True):
            if "x" in data and "y" in data:
                points.append((data["x"], data["y"]))
        
        if len(points) < 3:
            return None
        
        mp = MultiPoint(points)
        hull = mp.convex_hull
        
        if is_graph_projected(G):
            area_m2 = hull.area
            area_km2 = area_m2 / 1e6
        else:
            try:
                from shapely.ops import transform
                import pyproj
                
                centroid = hull.centroid
                utm_zone = int((centroid.x + 180) / 6) + 1
                hemisphere = "north" if centroid.y >= 0 else "south"
                epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone
                
                transformer = pyproj.Transformer.from_crs(
                    "EPSG:4326", f"EPSG:{epsg}", always_xy=True
                )
                hull_projected = transform(transformer.transform, hull)
                area_m2 = hull_projected.area
                area_km2 = area_m2 / 1e6
            except Exception as proj_error:
                log(f"Projection error, using approximate calculation: {proj_error}")
                minx, miny, maxx, maxy = hull.bounds
                mid_lat = (miny + maxy) / 2
                km_per_deg_lat = 111.32
                km_per_deg_lon = 111.32 * np.cos(np.radians(mid_lat))
                width_km = (maxx - minx) * km_per_deg_lon
                height_km = (maxy - miny) * km_per_deg_lat
                area_km2 = width_km * height_km * 0.7
        
        log(f"Estimated network area: {area_km2:.2f} km2")
        return area_km2
        
    except Exception as e:
        log(f"Error estimating network area: {e}")
        return None

