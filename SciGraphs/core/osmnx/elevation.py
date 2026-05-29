from ...utils.logger import log
from .get_osmnx import get_osmnx
from .projection import is_graph_projected


def add_node_elevations_raster(G, filepath, band=1):
    """
    Add elevation attribute to nodes from a local DEM raster file.
    
    The graph should be projected to the same CRS as the raster for accurate results.
    
    Args:
        G: OSMnx MultiDiGraph (ideally projected to match raster CRS)
        filepath: Path to DEM raster file (GeoTIFF, etc.) or list of files
        band: Raster band to sample (default 1)
    
    Returns:
        Modified graph with 'elevation' attribute on nodes, or None on error
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None
    
    try:
        if hasattr(ox, "elevation") and hasattr(ox.elevation, "add_node_elevations_raster"):
            G = ox.elevation.add_node_elevations_raster(G, filepath, band=band)
        elif hasattr(ox, "add_node_elevations_raster"):
            G = ox.add_node_elevations_raster(G, filepath, band=band)
        else:
            return _add_elevations_from_raster_manual(G, filepath, band)
        
        nodes_with_elev = sum(1 for n, d in G.nodes(data=True) if "elevation" in d)
        log(f"Added elevation to {nodes_with_elev} nodes from raster")
        return G
        
    except Exception as e:
        log(f"Error adding node elevations from raster: {e}")
        return None


def _add_elevations_from_raster_manual(G, filepath, band=1):
    """Manual implementation of elevation sampling from raster."""
    try:
        import rasterio
        from rasterio.sample import sample_gen
    except ImportError:
        log("rasterio not available for elevation sampling")
        return None
    
    try:
        if isinstance(filepath, (list, tuple)):
            import tempfile
            import os
            
            vrt_path = os.path.join(tempfile.gettempdir(), "scigraphs_dem.vrt")
            
            try:
                from osgeo import gdal
                gdal.BuildVRT(vrt_path, list(filepath))
                filepath = vrt_path
            except ImportError:
                filepath = filepath[0]
                log("Warning: gdal not available, using first raster file only")
        
        with rasterio.open(filepath) as src:
            coords = [(G.nodes[n].get("x", 0), G.nodes[n].get("y", 0)) for n in G.nodes()]
            elevations = list(sample_gen(src, coords, indexes=band))
            
            for node, elev in zip(G.nodes(), elevations):
                elev_val = elev[0] if elev else None
                if elev_val is not None and elev_val != src.nodata:
                    G.nodes[node]["elevation"] = float(elev_val)
                else:
                    G.nodes[node]["elevation"] = 0.0
        
        return G
        
    except Exception as e:
        log(f"Error in manual raster sampling: {e}")
        return None


def add_node_elevations_google(G, api_key=None, batch_size=350, pause=0.1):
    """
    Add elevation attribute to nodes using Google Elevation API or similar service.
    
    Note: Requires an API key for Google Elevation API or a configured alternative.
    The graph should be in unprojected (lat/lon) coordinates.
    
    Args:
        G: OSMnx MultiDiGraph in lat/lon coordinates
        api_key: Google Elevation API key (or None to use osmnx settings)
        batch_size: Number of nodes to query per API request
        pause: Seconds to pause between API requests
    
    Returns:
        Modified graph with 'elevation' attribute on nodes, or None on error
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None
    
    if is_graph_projected(G):
        log("Warning: Graph is projected. API elevation works best with lat/lon coordinates.")
    
    if api_key is None:
        log("No Google API key provided, using Open-Elevation API (free)...")
        return _add_elevations_from_open_elevation(G, batch_size=100, pause=0.5)
    
    try:
        if hasattr(ox, "elevation") and hasattr(ox.elevation, "add_node_elevations_google"):
            G = ox.elevation.add_node_elevations_google(
                G, api_key=api_key, batch_size=batch_size, pause=pause
            )
        elif hasattr(ox, "add_node_elevations_google"):
            G = ox.add_node_elevations_google(
                G, api_key=api_key, batch_size=batch_size, pause=pause
            )
        else:
            return _add_elevations_from_open_elevation(G, batch_size=100, pause=0.5)
        
        nodes_with_elev = sum(1 for n, d in G.nodes(data=True) if "elevation" in d)
        log(f"Added elevation to {nodes_with_elev} nodes from API")
        return G
        
    except Exception as e:
        log(f"Google API error: {e}")
        log("Falling back to Open-Elevation API...")
        return _add_elevations_from_open_elevation(G, batch_size=100, pause=0.5)


def _add_elevations_from_open_elevation(G, batch_size=100, pause=0.5):
    """
    Fallback: Use Open-Elevation API (free, no key required).
    https://open-elevation.com/
    """
    import requests
    import time
    
    try:
        nodes_list = list(G.nodes())
        coords = []
        for n in nodes_list:
            data = G.nodes[n]
            lat = data.get("y", 0)
            lon = data.get("x", 0)
            coords.append({"latitude": lat, "longitude": lon})
        
        log(f"Querying Open-Elevation API for {len(coords)} nodes...")
        
        elevations = []
        for i in range(0, len(coords), batch_size):
            batch = coords[i : i + batch_size]
            
            url = "https://api.open-elevation.com/api/v1/lookup"
            response = requests.post(url, json={"locations": batch}, timeout=30)
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                for result in results:
                    elevations.append(result.get("elevation", 0))
            else:
                log(f"API error: {response.status_code}")
                elevations.extend([0] * len(batch))
            
            if i + batch_size < len(coords):
                time.sleep(pause)
        
        for node, elev in zip(nodes_list, elevations):
            G.nodes[node]["elevation"] = float(elev) if elev is not None else 0.0
        
        log("Added elevation from Open-Elevation API")
        return G
        
    except Exception as e:
        log(f"Error with Open-Elevation API: {e}")
        return None

