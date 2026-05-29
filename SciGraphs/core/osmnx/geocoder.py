from ...utils.logger import log
from .get_osmnx import get_osmnx


def geocode(query):
    """
    Geocode a place name or address to (latitude, longitude) coordinates.
    
    Uses Nominatim API. Please respect usage policies and rate limits.
    
    Args:
        query: Place name or address string
    
    Returns:
        tuple: (latitude, longitude) coordinates
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "geocoder") and hasattr(ox.geocoder, "geocode"):
        coords = ox.geocoder.geocode(query)
        log(f"Geocoded '{query}' to {coords}")
        return coords
    elif hasattr(ox, "geocode"):
        coords = ox.geocode(query)
        log(f"Geocoded '{query}' to {coords}")
        return coords
    
    log("geocode function not found in OSMnx")
    return None


def geocode_to_gdf(query, which_result=None, by_osmid=False):
    """
    Geocode a query and return a GeoDataFrame with the result geometry.
    
    Retrieves place boundaries from Nominatim and returns as GeoDataFrame.
    
    Args:
        query: Place name, address, or OSM ID
        which_result: Which result to return if multiple matches (1-indexed)
        by_osmid: If True, treat query as an OSM ID
    
    Returns:
        GeoDataFrame: Place geometry with attributes
    """
    ox = get_osmnx()
    if ox is None:
        log("OSMnx not available")
        return None
    
    if hasattr(ox, "geocoder") and hasattr(ox.geocoder, "geocode_to_gdf"):
        gdf = ox.geocoder.geocode_to_gdf(query, which_result=which_result, by_osmid=by_osmid)
        log(f"Geocoded '{query}' to GeoDataFrame")
        return gdf
    elif hasattr(ox, "geocode_to_gdf"):
        gdf = ox.geocode_to_gdf(query, which_result=which_result, by_osmid=by_osmid)
        log(f"Geocoded '{query}' to GeoDataFrame")
        return gdf
    
    log("geocode_to_gdf function not found in OSMnx")
    return None

