from ...utils.logger import log
from .get_osmnx import get_osmnx


def features_from_place(query, tags, which_result=None):
    """
    Download OSM features within the boundaries of a place.
    
    Args:
        query: Place name string (e.g., "Piedmont, California, USA")
        tags: Dict of OSM tags to retrieve (e.g., {"building": True})
        which_result: Which geocoding result to use (default: None uses first)
    
    Returns:
        GeoDataFrame: Features with geometries and OSM tags
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
    """
    Download OSM features within distance of a lat-lon point.
    
    Args:
        center_point: Tuple of (latitude, longitude)
        tags: Dict of OSM tags to retrieve
        dist: Distance in meters from point
    
    Returns:
        GeoDataFrame: Features with geometries and OSM tags
    """
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
    """
    Download OSM features within distance of an address.
    
    Args:
        address: Address string to geocode
        tags: Dict of OSM tags to retrieve
        dist: Distance in meters from address
    
    Returns:
        GeoDataFrame: Features with geometries and OSM tags
    """
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
    """Translate the addon's ``(north, south, east, west)`` bbox to the
    format the installed OSMnx version expects.

    OSMnx 1.x signed the helpers as ``bbox=(north, south, east, west)``;
    OSMnx ≥ 2.0 unified all spatial helpers around
    ``bbox=(left, bottom, right, top) = (west, south, east, north)``.
    Passing a v1-style tuple to v2 silently constructs a degenerate
    polygon spanning the whole hemisphere — that is what causes
    Overpass to subdivide into "thousands of sub-queries" and
    eventually time out.

    We detect the major version once and reorder accordingly so the
    rest of the addon can keep using a consistent (n, s, e, w) tuple.
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
    """
    Download OSM features within a bounding box.
    
    Args:
        bbox: Tuple of (north, south, east, west) lat-lon coordinates
        tags: Dict of OSM tags to retrieve
    
    Returns:
        GeoDataFrame: Features with geometries and OSM tags
    """
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
    """
    Download OSM features within a shapely Polygon or MultiPolygon.
    
    Args:
        polygon: Shapely Polygon or MultiPolygon
        tags: Dict of OSM tags to retrieve
    
    Returns:
        GeoDataFrame: Features with geometries and OSM tags
    """
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
    """
    Create GeoDataFrame of features from an OSM XML file.
    
    Args:
        filepath: Path to OSM XML file
        polygon: Optional polygon to filter features spatially
        tags: Optional dict of tags to filter features
        encoding: File encoding (default: utf-8)
    
    Returns:
        GeoDataFrame: Features with geometries and OSM tags
    """
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

