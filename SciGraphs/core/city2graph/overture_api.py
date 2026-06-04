"""
Overture Maps REST API client.

Provides direct access to Overture Maps data via their REST API,
bypassing the CLI tool requirement.
"""

from ...utils.logger import log


def get_api_key():
    """
    Retrieve Overture Maps API key from addon preferences.
    
    Returns:
        str: API key or 'DEMO-API-KEY' as fallback
    """
    try:
        from ...preferences import get_preferences
        prefs = get_preferences()
        if prefs and prefs.overture_api_key:
            return prefs.overture_api_key
        return "DEMO-API-KEY"
    except Exception as e:
        log(f"Error getting API key from preferences: {e}")
        return "DEMO-API-KEY"


def _make_api_request(endpoint, params):
    """
    Make a request to Overture Maps API.
    
    Args:
        endpoint: API endpoint path
        params: Query parameters dict
    
    Returns:
        dict: GeoJSON response or None on error
    """
    import requests
    
    base_url = "https://api.overturemapsapi.com"
    api_key = get_api_key()
    
    headers = {
        "x-api-key": api_key,
        "Accept": "application/geo+json"
    }
    
    url = f"{base_url}{endpoint}"
    
    try:
        log(f"Requesting Overture Maps API: {endpoint}")
        log(f"Parameters: {params}")
        response = requests.get(url, headers=headers, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            log(f"Response received: {len(str(data))} bytes")
            if isinstance(data, dict):
                if 'features' in data:
                    log(f"Found {len(data.get('features', []))} features in response")
                else:
                    log(f"Response keys: {list(data.keys())}")
            return data
        elif response.status_code == 401:
            log("Error: Invalid API key")
            log("Update your API key in Edit > Preferences > Add-ons > SciGraphs")
            return None
        elif response.status_code == 429:
            log("Error: Rate limit exceeded")
            log("Consider getting a production API key from Overture Maps")
            return None
        else:
            log(f"API error: HTTP {response.status_code}")
            log(f"Response: {response.text[:500]}")
            return None
            
    except requests.exceptions.Timeout:
        log("Request timeout - try reducing the search area")
        return None
    except requests.exceptions.RequestException as e:
        log(f"Network error: {e}")
        return None
    except Exception as e:
        log(f"Unexpected error: {e}")
        return None


def query_overture_buildings(bbox, limit=10000):
    """
    Query building footprints from Overture Maps.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
        limit: Max features to request from the API (server hard cap
            applies on top; pass 0 / negative to use the API default)
    
    Returns:
        GeoDataFrame or None
    """
    north, south, east, west = bbox
    
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    
    import math
    lat_diff = north - south
    lon_diff = east - west
    radius_km = max(lat_diff, lon_diff) * 111.32 / 2
    radius_m = min(radius_km * 1000, 5000)
    
    log(f"Query center: lat={center_lat:.6f}, lon={center_lon:.6f}, radius={radius_m:.0f}m, limit={limit}")
    
    params = {
        "lat": center_lat,
        "lng": center_lon,
        "radius": int(radius_m),
        "limit": int(limit) if limit and limit > 0 else 10000,
    }
    
    geojson = _make_api_request("/v1/buildings", params)
    
    if geojson is None:
        log("Trying alternate endpoint: /buildings")
        geojson = _make_api_request("/buildings", params)
    
    if geojson is None:
        return None
    
    try:
        import geopandas as gpd
        
        log(f"Response type: {type(geojson)}")
        if isinstance(geojson, dict):
            log(f"Response keys: {list(geojson.keys())}")
            
            if 'features' in geojson:
                log(f"Features type: {type(geojson['features'])}")
                log(f"Number of features: {len(geojson['features'])}")
            
            if 'data' in geojson:
                log(f"Data key found, type: {type(geojson['data'])}")
                if isinstance(geojson['data'], list):
                    log(f"Data list length: {len(geojson['data'])}")
        
        features = None
        if 'features' in geojson and len(geojson['features']) > 0:
            features = geojson['features']
        elif 'data' in geojson and len(geojson['data']) > 0:
            features = geojson['data']
        elif isinstance(geojson, list):
            features = geojson
        
        if features and len(features) > 0:
            log(f"Processing {len(features)} building features")
            
            gdf = gpd.GeoDataFrame.from_features(features) if isinstance(features, list) else gpd.GeoDataFrame.from_dict(features)
            
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            
            gdf_projected = gdf.to_crs("EPSG:3857")
            centroids = gdf_projected.geometry.centroid.to_crs("EPSG:4326")
            
            bounds_filter = (
                (centroids.y >= south) &
                (centroids.y <= north) &
                (centroids.x >= west) &
                (centroids.x <= east)
            )
            gdf = gdf[bounds_filter]
            
            log(f"Retrieved {len(gdf)} buildings")
            return gdf
        else:
            log("No buildings found in response")
            return None
            
    except Exception as e:
        log(f"Error processing buildings response: {e}")
        import traceback
        traceback.print_exc()
        return None


def _place_category_text(value):
    """Return a lowercase searchable string from an Overture category value.

    Overture place features expose categories as a dict (``primary`` plus
    ``alternate``), a list, or a plain string depending on the serialization.
    This normalizes any of those shapes to a single lowercase string.
    """
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = [value.get("primary", "")]
        alternate = value.get("alternate")
        if isinstance(alternate, (list, tuple)):
            parts.extend(str(item) for item in alternate)
        elif alternate:
            parts.append(str(alternate))
        return " ".join(str(part) for part in parts if part).lower()
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value).lower()
    return str(value).lower()


def _filter_places_by_keywords(gdf, keywords):
    """Filter a places GeoDataFrame to rows whose category matches a keyword."""
    if not keywords:
        return gdf

    column = None
    for candidate in ("categories", "category"):
        if candidate in gdf.columns:
            column = candidate
            break
    if column is None:
        log("Places response has no category column; skipping category filter")
        return gdf

    lowered = [kw.lower() for kw in keywords]

    def _matches(value):
        text = _place_category_text(value)
        return any(kw in text for kw in lowered)

    filtered = gdf[gdf[column].apply(_matches)]
    log(f"Filtered places by categories {lowered}: {len(filtered)}/{len(gdf)} kept")
    return filtered


def query_overture_places(bbox, categories=None, limit=10000):
    """
    Query places (POIs) from Overture Maps.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
        categories: Optional list of category keywords; places whose category
            contains any keyword are kept (client-side filter)
        limit: Max features to request from the API
    
    Returns:
        GeoDataFrame or None
    """
    north, south, east, west = bbox
    
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    
    lat_diff = north - south
    lon_diff = east - west
    radius_km = max(lat_diff, lon_diff) * 111.32 / 2
    radius_m = min(radius_km * 1000, 5000)
    
    params = {
        "lat": center_lat,
        "lng": center_lon,
        "radius": int(radius_m),
        "limit": int(limit) if limit and limit > 0 else 10000,
    }
    
    geojson = _make_api_request("/places", params)
    
    if geojson is None:
        return None
    
    try:
        import geopandas as gpd
        
        log(f"Response type: {type(geojson)}")
        
        features = None
        if isinstance(geojson, list) and len(geojson) > 0:
            features = geojson
            log(f"Response is list with {len(features)} items")
        elif isinstance(geojson, dict):
            if 'features' in geojson and len(geojson['features']) > 0:
                features = geojson['features']
            elif 'data' in geojson and len(geojson['data']) > 0:
                features = geojson['data']
        
        if features and len(features) > 0:
            log(f"Processing {len(features)} place features")
            gdf = gpd.GeoDataFrame.from_features(features)
            
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            
            bounds_filter = (
                (gdf.geometry.y >= south) &
                (gdf.geometry.y <= north) &
                (gdf.geometry.x >= west) &
                (gdf.geometry.x <= east)
            )
            gdf = gdf[bounds_filter]
            
            if categories:
                gdf = _filter_places_by_keywords(gdf, categories)
            
            log(f"Retrieved {len(gdf)} places")
            return gdf
        else:
            log("No places found in response")
            return None
            
    except Exception as e:
        log(f"Error processing places response: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_overture_addresses(bbox, limit=10000):
    """
    Query addresses from Overture Maps.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
        limit: Max features to request from the API
    
    Returns:
        GeoDataFrame or None
    """
    north, south, east, west = bbox
    
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    
    lat_diff = north - south
    lon_diff = east - west
    radius_km = max(lat_diff, lon_diff) * 111.32 / 2
    radius_m = min(radius_km * 1000, 5000)
    
    params = {
        "lat": center_lat,
        "lng": center_lon,
        "radius": int(radius_m),
        "limit": int(limit) if limit and limit > 0 else 10000,
    }
    
    geojson = _make_api_request("/addresses", params)
    
    if geojson is None:
        return None
    
    try:
        import geopandas as gpd
        
        if 'features' in geojson and len(geojson['features']) > 0:
            gdf = gpd.GeoDataFrame.from_features(geojson['features'])
            
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            
            bounds_filter = (
                (gdf.geometry.y >= south) &
                (gdf.geometry.y <= north) &
                (gdf.geometry.x >= west) &
                (gdf.geometry.x <= east)
            )
            gdf = gdf[bounds_filter]
            
            log(f"Retrieved {len(gdf)} addresses")
            return gdf
        else:
            log("No addresses found in response")
            return None
            
    except Exception as e:
        log(f"Error processing addresses response: {e}")
        import traceback
        traceback.print_exc()
        return None


def query_overture_transportation(bbox, limit=10000):
    """
    Query transportation infrastructure from Overture Maps.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
        limit: Max features to request from the API
    
    Returns:
        GeoDataFrame or None
    """
    north, south, east, west = bbox
    
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2
    
    lat_diff = north - south
    lon_diff = east - west
    radius_km = max(lat_diff, lon_diff) * 111.32 / 2
    radius_m = min(radius_km * 1000, 5000)
    
    params = {
        "lat": center_lat,
        "lng": center_lon,
        "radius": int(radius_m),
        "limit": int(limit) if limit and limit > 0 else 10000,
    }
    
    geojson = _make_api_request("/transportation", params)
    
    if geojson is None:
        return None
    
    try:
        import geopandas as gpd
        
        if 'features' in geojson and len(geojson['features']) > 0:
            gdf = gpd.GeoDataFrame.from_features(geojson['features'])
            
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            
            log(f"Retrieved {len(gdf)} transportation features")
            return gdf
        else:
            log("No transportation features found in response")
            return None
            
    except Exception as e:
        log(f"Error processing transportation response: {e}")
        import traceback
        traceback.print_exc()
        return None

