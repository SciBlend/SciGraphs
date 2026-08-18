from scigraphs_core.logger import log
from .get_osmnx import get_osmnx


def features_from_place(query, tags, which_result=None):
    """Download OSM features inside a place boundary, e.g. "Piedmont, California, USA".

    which_result picks among the geocoder's matches; None takes the first.
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "features") and hasattr(ox.features, "features_from_place"):
        gdf = ox.features.features_from_place(query, tags, which_result=which_result)
        log(f"Downloaded {len(gdf)} features from place: {query}")
        return gdf
    elif hasattr(ox, "features_from_place"):
        gdf = ox.features_from_place(query, tags, which_result=which_result)
        log(f"Downloaded {len(gdf)} features from place: {query}")
        return gdf
    elif hasattr(ox, "geometries") and hasattr(ox.geometries, "geometries_from_place"):
        gdf = ox.geometries.geometries_from_place(query, tags, which_result=which_result)
        log(f"Downloaded {len(gdf)} features from place: {query}")
        return gdf
    
    log("features_from_place function not found in OSMnx")
    return None


def features_from_point(center_point, tags, dist=1000):
    """Download OSM features within dist meters of a (latitude, longitude) point."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "features") and hasattr(ox.features, "features_from_point"):
        gdf = ox.features.features_from_point(center_point, tags, dist=dist)
        log(f"Downloaded {len(gdf)} features from point")
        return gdf
    elif hasattr(ox, "features_from_point"):
        gdf = ox.features_from_point(center_point, tags, dist=dist)
        log(f"Downloaded {len(gdf)} features from point")
        return gdf
    elif hasattr(ox, "geometries") and hasattr(ox.geometries, "geometries_from_point"):
        gdf = ox.geometries.geometries_from_point(center_point, tags, dist=dist)
        log(f"Downloaded {len(gdf)} features from point")
        return gdf
    
    log("features_from_point function not found in OSMnx")
    return None


def features_from_address(address, tags, dist=1000):
    """Download OSM features within dist meters of a geocoded address."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "features") and hasattr(ox.features, "features_from_address"):
        gdf = ox.features.features_from_address(address, tags, dist=dist)
        log(f"Downloaded {len(gdf)} features from address: {address}")
        return gdf
    elif hasattr(ox, "features_from_address"):
        gdf = ox.features_from_address(address, tags, dist=dist)
        log(f"Downloaded {len(gdf)} features from address: {address}")
        return gdf
    elif hasattr(ox, "geometries") and hasattr(ox.geometries, "geometries_from_address"):
        gdf = ox.geometries.geometries_from_address(address, tags, dist=dist)
        log(f"Downloaded {len(gdf)} features from address: {address}")
        return gdf
    
    log("features_from_address function not found in OSMnx")
    return None


def _normalize_bbox_for_osmnx(bbox, ox):
    """Reorder an ``(n, s, e, w)`` bbox into the order the installed OSMnx wants.

    OSMnx 1.x took ``bbox=(north, south, east, west)``; 2.0 unified every
    spatial helper on ``bbox=(west, south, east, north)``. A v1-style tuple
    handed to v2 raises nothing: it builds a degenerate polygon spanning the
    hemisphere, which Overpass then splits into thousands of sub-queries before
    timing out.
    """
    n, s, e, w = bbox
    try:
        version = getattr(ox, "__version__", "1.0")
        major = int(str(version).split(".")[0])
    except (ValueError, AttributeError):
        major = 1
    if major >= 2:
        # (left, bottom, right, top) = (west, south, east, north)
        return (w, s, e, n)
    return (n, s, e, w)


def features_from_bbox(bbox, tags):
    """Download OSM features inside a ``(north, south, east, west)`` bounding box."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None

    normalized = _normalize_bbox_for_osmnx(bbox, ox)

    if hasattr(ox, "features") and hasattr(ox.features, "features_from_bbox"):
        gdf = ox.features.features_from_bbox(normalized, tags)
        log(f"Downloaded {len(gdf)} features from bbox")
        return gdf
    elif hasattr(ox, "features_from_bbox"):
        gdf = ox.features_from_bbox(normalized, tags)
        log(f"Downloaded {len(gdf)} features from bbox")
        return gdf
    elif hasattr(ox, "geometries") and hasattr(ox.geometries, "geometries_from_bbox"):
        gdf = ox.geometries.geometries_from_bbox(normalized, tags)
        log(f"Downloaded {len(gdf)} features from bbox")
        return gdf
    
    log("features_from_bbox function not found in OSMnx")
    return None


def features_from_polygon(polygon, tags):
    """Download OSM features inside a shapely Polygon or MultiPolygon."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "features") and hasattr(ox.features, "features_from_polygon"):
        gdf = ox.features.features_from_polygon(polygon, tags)
        log(f"Downloaded {len(gdf)} features from polygon")
        return gdf
    elif hasattr(ox, "features_from_polygon"):
        gdf = ox.features_from_polygon(polygon, tags)
        log(f"Downloaded {len(gdf)} features from polygon")
        return gdf
    elif hasattr(ox, "geometries") and hasattr(ox.geometries, "geometries_from_polygon"):
        gdf = ox.geometries.geometries_from_polygon(polygon, tags)
        log(f"Downloaded {len(gdf)} features from polygon")
        return gdf
    
    log("features_from_polygon function not found in OSMnx")
    return None


def features_from_xml(filepath, polygon=None, tags=None, encoding="utf-8"):
    """Read features from an OSM XML file, optionally filtered by polygon and tags."""
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "features") and hasattr(ox.features, "features_from_xml"):
        gdf = ox.features.features_from_xml(filepath, polygon=polygon, tags=tags, encoding=encoding)
        log(f"Loaded {len(gdf)} features from XML file")
        return gdf
    elif hasattr(ox, "features_from_xml"):
        gdf = ox.features_from_xml(filepath, polygon=polygon, tags=tags, encoding=encoding)
        log(f"Loaded {len(gdf)} features from XML file")
        return gdf
    elif hasattr(ox, "geometries") and hasattr(ox.geometries, "geometries_from_xml"):
        gdf = ox.geometries.geometries_from_xml(filepath, polygon=polygon, tags=tags, encoding=encoding)
        log(f"Loaded {len(gdf)} features from XML file")
        return gdf
    
    log("features_from_xml function not found in OSMnx")
    return None

