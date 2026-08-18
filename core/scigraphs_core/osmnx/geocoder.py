from scigraphs_core.logger import log
from .get_osmnx import get_osmnx


def geocode(query):
    """Geocode a place name or address to ``(latitude, longitude)`` through Nominatim.

    Nominatim rate-limits, so respect its usage policy when calling in a loop.
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
    """Geocode a query to a GeoDataFrame carrying the place boundary.

    which_result is 1-indexed among Nominatim's matches. With by_osmid the query
    is read as an OSM ID instead of as a name.
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

