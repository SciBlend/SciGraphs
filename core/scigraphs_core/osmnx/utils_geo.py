from scigraphs_core.logger import log
from .get_osmnx import get_osmnx


def bbox_from_point(point, dist=1000, project_utm=True, return_crs=False):
    """A ``(north, south, east, west)`` box dist meters around a ``(lat, lon)`` point.

    project_utm buffers in UTM rather than in degrees, which is the accurate way
    to do it. return_crs gives back ``(bbox, crs)`` instead of the bare box.
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "utils_geo") and hasattr(ox.utils_geo, "bbox_from_point"):
        result = ox.utils_geo.bbox_from_point(point, dist=dist, project_utm=project_utm, return_crs=return_crs)
        log(f"Created bbox from point with {dist}m radius")
        return result
    elif hasattr(ox, "bbox_from_point"):
        result = ox.bbox_from_point(point, dist=dist, project_utm=project_utm, return_crs=return_crs)
        log(f"Created bbox from point with {dist}m radius")
        return result
    
    log("bbox_from_point function not found in OSMnx")
    return None


def bbox_to_poly(bbox):
    """Turn a ``(north, south, east, west)`` box into a rectangular shapely Polygon."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "utils_geo") and hasattr(ox.utils_geo, "bbox_to_poly"):
        return ox.utils_geo.bbox_to_poly(bbox)
    elif hasattr(ox, "bbox_to_poly"):
        return ox.bbox_to_poly(bbox)
    
    log("bbox_to_poly function not found in OSMnx")
    return None


def buffer_geometry(geom, dist):
    """Buffer a shapely geometry by dist meters.

    A lat/lon geometry goes through UTM and back, so the buffer really is in
    meters rather than in degrees.
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "utils_geo") and hasattr(ox.utils_geo, "buffer_geometry"):
        return ox.utils_geo.buffer_geometry(geom, dist)
    
    log("buffer_geometry function not found in OSMnx")
    return None


def interpolate_points(geom, dist):
    """Points spaced dist apart along a LineString, in the geometry's own units."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "utils_geo") and hasattr(ox.utils_geo, "interpolate_points"):
        return ox.utils_geo.interpolate_points(geom, dist)
    
    log("interpolate_points function not found in OSMnx")
    return None


def sample_points(G, n):
    """Sample n random points lying on the graph's edges. Edges need a geometry."""
    ox = get_osmnx()
    if ox is None or G is None:
        log("OSMnx not available or graph is None")
        return None
    
    if hasattr(ox, "utils_geo") and hasattr(ox.utils_geo, "sample_points"):
        points = ox.utils_geo.sample_points(G, n)
        log(f"Sampled {n} random points on graph")
        return points
    
    log("sample_points function not found in OSMnx")
    return None

