# Elevation data download from OpenTopography (SRTM, ASTER and friends).

import os
import tempfile
import urllib.request
import urllib.error
from scigraphs_core.logger import log


DEM_DATASETS = {
    'SRTMGL1': {
        'name': 'SRTM GL1 (30m)',
        'description': 'Shuttle Radar Topography Mission, 1 arc-second (~30m)',
        'coverage': 'Global (60N to 56S)',
        'resolution': 30,
    },
    'SRTMGL3': {
        'name': 'SRTM GL3 (90m)',
        'description': 'Shuttle Radar Topography Mission, 3 arc-second (~90m)',
        'coverage': 'Global (60N to 56S)',
        'resolution': 90,
    },
    'AW3D30': {
        'name': 'ALOS World 3D (30m)',
        'description': 'JAXA ALOS PRISM sensor, 1 arc-second (~30m)',
        'coverage': 'Global',
        'resolution': 30,
    },
    'NASADEM': {
        'name': 'NASADEM (30m)',
        'description': 'NASA improved SRTM, 1 arc-second (~30m)',
        'coverage': 'Global (60N to 56S)',
        'resolution': 30,
    },
    'COP30': {
        'name': 'Copernicus GLO-30',
        'description': 'Copernicus DEM, 1 arc-second (~30m)',
        'coverage': 'Global',
        'resolution': 30,
    },
    'COP90': {
        'name': 'Copernicus GLO-90',
        'description': 'Copernicus DEM, 3 arc-second (~90m)',
        'coverage': 'Global',
        'resolution': 90,
    },
}

OPENTOPOGRAPHY_API_URL = "https://portal.opentopography.org/API/globaldem"

# Key fallback when no caller passes one. An env var is the one channel that
# behaves the same inside Blender, in a notebook and in CI.
API_KEY_ENV_VAR = "OPENTOPOGRAPHY_API_KEY"


def get_api_key(api_key=None):
    """An OpenTopography API key stripped of whitespace: the explicit argument, then
    OPENTOPOGRAPHY_API_KEY, then None, which the caller calls "no key configured"."""
    for candidate in (api_key, os.environ.get(API_KEY_ENV_VAR)):
        if candidate and candidate.strip():
            return candidate.strip()
    return None


def download_from_opentopography(bounds, dataset='SRTMGL1', output_dir=None, 
                                  api_key=None, progress_callback=None):
    """Download DEM data from the OpenTopography API, returning the GeoTIFF path or
    None on error. ``bounds`` is north/south/east/west in WGS84 degrees, and
    ``api_key`` None falls back to OPENTOPOGRAPHY_API_KEY, which core can see."""
    if bounds is None:
        log("Error: No bounds provided")
        return None
    
    north = bounds.get('north')
    south = bounds.get('south')
    east = bounds.get('east')
    west = bounds.get('west')
    
    if None in (north, south, east, west):
        log("Error: Incomplete bounds")
        return None
    
    if not (-90 <= south < north <= 90):
        log(f"Error: Invalid latitude range: {south} to {north}")
        return None
    
    if not (-180 <= west < east <= 180):
        log(f"Error: Invalid longitude range: {west} to {east}")
        return None
    
    api_key = get_api_key(api_key)

    if not api_key:
        log("Error: OpenTopography API key not set")
        log(f"Pass api_key=..., or set the {API_KEY_ENV_VAR} environment variable")
        log("In Blender: Edit > Preferences > Add-ons > SciGraphs")
        return None
    
    if dataset not in DEM_DATASETS:
        log(f"Error: Unknown dataset '{dataset}'")
        log(f"Available: {list(DEM_DATASETS.keys())}")
        return None
    
    params = {
        'demtype': dataset,
        'south': f"{south:.6f}",
        'north': f"{north:.6f}",
        'west': f"{west:.6f}",
        'east': f"{east:.6f}",
        'outputFormat': 'GTiff',
        'API_Key': api_key,
    }
    
    query_string = '&'.join(f"{k}={v}" for k, v in params.items())
    url = f"{OPENTOPOGRAPHY_API_URL}?{query_string}"
    
    safe_url = url.replace(api_key, "***")
    log(f"Requesting DEM from OpenTopography...")
    log(f"Dataset: {DEM_DATASETS[dataset]['name']}")
    log(f"Bounds: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")
    
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    output_path = os.path.join(output_dir, f"dem_{dataset}.tif")
    
    try:
        if progress_callback:
            progress_callback(0)
        
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'SciGraphs-Blender-Addon/1.0')
        
        with urllib.request.urlopen(request, timeout=120) as response:
            content_type = response.headers.get('Content-Type', '')
            
            if 'image/tiff' in content_type or 'application/octet-stream' in content_type:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            percent = int(100 * downloaded / total_size)
                            progress_callback(percent)
                
                if progress_callback:
                    progress_callback(100)
                
                log(f"DEM downloaded: {output_path}")
                log(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
                
                return output_path
            
            elif 'text' in content_type or 'json' in content_type:
                error_msg = response.read().decode('utf-8', errors='ignore')
                log(f"API Error: {error_msg[:500]}")
                return None
            
            else:
                log(f"Unexpected response type: {content_type}")
                return None
    
    except urllib.error.HTTPError as e:
        log(f"HTTP Error {e.code}: {e.reason}")
        if e.code == 401:
            log("Invalid API key. Check your OpenTopography API key.")
        elif e.code == 400:
            log("Bad request. Check coordinate bounds.")
        return None
    
    except urllib.error.URLError as e:
        log(f"Connection error: {e.reason}")
        return None
    
    except Exception as e:
        log(f"Download error: {e}")
        return None


def estimate_download_size(bounds, dataset='SRTMGL1'):
    """Estimate the download size for an area, in MB."""
    if bounds is None:
        return 0
    
    resolution = DEM_DATASETS.get(dataset, {}).get('resolution', 30)
    
    lat_range = bounds['north'] - bounds['south']
    lon_range = bounds['east'] - bounds['west']
    
    # 1 degree of latitude is about 111 km; resolution is in meters.
    lat_pixels = (lat_range * 111000) / resolution
    lon_pixels = (lon_range * 111000) / resolution
    
    total_pixels = lat_pixels * lon_pixels
    
    # int16 GeoTIFF: 2 bytes per pixel plus about 10 percent overhead.
    size_bytes = total_pixels * 2 * 1.1
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def validate_api_key(api_key):
    """Validate an OpenTopography API key by requesting a tiny tile. is_valid is None
    when the request could not settle it: "key is bad" is not "could not check"."""
    if not api_key or not api_key.strip():
        return False, "API key is empty"
    
    api_key = api_key.strip()
    
    # Central Spain: guaranteed SRTM coverage, and small enough to stay cheap.
    params = {
        'demtype': 'SRTMGL3',
        'south': '40.0',
        'north': '40.01',
        'west': '-3.7',
        'east': '-3.69',
        'outputFormat': 'GTiff',
        'API_Key': api_key,
    }
    
    query_string = '&'.join(f"{k}={v}" for k, v in params.items())
    url = f"{OPENTOPOGRAPHY_API_URL}?{query_string}"
    
    try:
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'SciGraphs-Blender-Addon/1.0')
        
        with urllib.request.urlopen(request, timeout=30) as response:
            content_type = response.headers.get('Content-Type', '')
            
            if 'tiff' in content_type or 'octet-stream' in content_type:
                return True, None
            
            body = response.read().decode('utf-8', errors='ignore')[:200]
            return False, f"Unexpected response: {body}"
    
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode('utf-8', errors='ignore')[:300]
        except:
            pass
        
        if e.code == 401:
            return False, "Invalid API key (401 Unauthorized)"
        elif e.code == 400:
            return False, f"Bad request (400): {error_body}"
        elif e.code == 403:
            return False, "Access forbidden (403). Key may be expired or revoked."
        else:
            return None, f"HTTP Error {e.code}: {error_body}"
    
    except urllib.error.URLError as e:
        return None, f"Connection error: {e.reason}"
    
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_dataset_items():
    """DEM_DATASETS as (identifier, name, description) tuples for an EnumProperty."""
    items = []
    for key, info in DEM_DATASETS.items():
        items.append((
            key,
            info['name'],
            f"{info['description']} - {info['coverage']}"
        ))
    return items

