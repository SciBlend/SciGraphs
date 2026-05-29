from ...utils.logger import log
from .get_osmnx import get_osmnx


def bbox_from_point(point, dist=1000, project_utm=True, return_crs=False):
    """
    Create a bounding box around a point.
    
    Args:
        point: Tuple of (latitude, longitude)
        dist: Distance in meters from point to bbox edge
        project_utm: If True, project to UTM for accurate buffering
        return_crs: If True, return (bbox, crs) tuple
    
    Returns:
        tuple: (north, south, east, west) bbox coordinates
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
    """
    Convert a bounding box to a Shapely Polygon.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
    
    Returns:
        shapely.Polygon: Rectangular polygon of bbox
    """
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
    """
    Buffer a geometry by distance in meters.
    
    For unprojected (lat-lon) geometries, automatically projects to UTM,
    buffers, then projects back to maintain accuracy.
    
    Args:
        geom: Shapely geometry (Point, LineString, Polygon, etc.)
        dist: Buffer distance in meters
    
    Returns:
        shapely.Polygon: Buffered geometry
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
    """
    Interpolate evenly-spaced points along a LineString geometry.
    
    Args:
        geom: Shapely LineString or MultiLineString
        dist: Spacing distance in geometry's units
    
    Returns:
        generator: Iterator of shapely.Point objects
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "utils_geo") and hasattr(ox.utils_geo, "interpolate_points"):
        return ox.utils_geo.interpolate_points(geom, dist)
    
    log("interpolate_points function not found in OSMnx")
    return None


def sample_points(G, n):
    """
    Randomly sample points constrained to graph edges.
    
    Args:
        G: networkx.MultiDiGraph with geometry attribute on edges
        n: Number of points to sample
    
    Returns:
        GeoSeries: Random points on graph edges
    """
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

