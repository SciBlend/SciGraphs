# DEM Download module
#
# Handles automatic download of elevation data from online services
# Primary source: OpenTopography (SRTM, ASTER, etc.)

import os
import tempfile
import urllib.request
import urllib.error
from ...utils.logger import log


# Available DEM datasets from OpenTopography
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

# OpenTopography API base URL
OPENTOPOGRAPHY_API_URL = "https://portal.opentopography.org/API/globaldem"


def get_api_key():
    """
    Get OpenTopography API key from addon preferences.
    
    Returns:
        API key string (stripped of whitespace) or None if not set
    """
    try:
        from ...preferences import get_preferences
        prefs = get_preferences()
        if prefs and prefs.opentopography_api_key:
            key = prefs.opentopography_api_key.strip()
            return key if key else None
    except Exception as exc:  # noqa: BLE001
        log(f"get_api_key: failed to read addon preferences ({exc})")
    return None


def download_from_opentopography(bounds, dataset='SRTMGL1', output_dir=None, 
                                  api_key=None, progress_callback=None):
    """
    Download DEM data from OpenTopography API.
    
    Args:
        bounds: Dict with 'north', 'south', 'east', 'west' in WGS84 degrees
        dataset: Dataset identifier (see DEM_DATASETS)
        output_dir: Directory to save the file (None = temp dir)
        api_key: OpenTopography API key (None = get from preferences)
        progress_callback: Optional function(percent) for progress updates
    
    Returns:
        Path to downloaded GeoTIFF file, or None on error
    """
    # Validate bounds
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
    
    # Validate coordinate ranges
    if not (-90 <= south < north <= 90):
        log(f"Error: Invalid latitude range: {south} to {north}")
        return None
    
    if not (-180 <= west < east <= 180):
        log(f"Error: Invalid longitude range: {west} to {east}")
        return None
    
    # Get API key
    if api_key is None:
        api_key = get_api_key()
    
    if not api_key:
        log("Error: OpenTopography API key not set")
        log("Set your API key in Edit > Preferences > Add-ons > SciGraphs")
        return None
    
    # Validate dataset
    if dataset not in DEM_DATASETS:
        log(f"Error: Unknown dataset '{dataset}'")
        log(f"Available: {list(DEM_DATASETS.keys())}")
        return None
    
    # Build URL
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
    
    # Log request (hide API key)
    safe_url = url.replace(api_key, "***")
    log(f"Requesting DEM from OpenTopography...")
    log(f"Dataset: {DEM_DATASETS[dataset]['name']}")
    log(f"Bounds: N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")
    
    # Prepare output path
    if output_dir is None:
        output_dir = tempfile.gettempdir()
    
    output_path = os.path.join(output_dir, f"dem_{dataset}.tif")
    
    # Download with progress
    try:
        if progress_callback:
            progress_callback(0)
        
        # Create request with timeout
        request = urllib.request.Request(url)
        request.add_header('User-Agent', 'SciGraphs-Blender-Addon/1.0')
        
        with urllib.request.urlopen(request, timeout=120) as response:
            # Check response
            content_type = response.headers.get('Content-Type', '')
            
            if 'image/tiff' in content_type or 'application/octet-stream' in content_type:
                # Valid GeoTIFF response
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
                # Error response
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
    """
    Estimate the download size for a given area.
    
    Args:
        bounds: Dict with bounds
        dataset: Dataset identifier
    
    Returns:
        Estimated size in MB
    """
    if bounds is None:
        return 0
    
    resolution = DEM_DATASETS.get(dataset, {}).get('resolution', 30)
    
    # Calculate area in degrees
    lat_range = bounds['north'] - bounds['south']
    lon_range = bounds['east'] - bounds['west']
    
    # Approximate pixels
    # 1 degree = ~111 km, resolution in meters
    lat_pixels = (lat_range * 111000) / resolution
    lon_pixels = (lon_range * 111000) / resolution
    
    total_pixels = lat_pixels * lon_pixels
    
    # GeoTIFF with int16 data: ~2 bytes per pixel + overhead
    size_bytes = total_pixels * 2 * 1.1
    size_mb = size_bytes / (1024 * 1024)
    
    return size_mb


def validate_api_key(api_key):
    """
    Validate an OpenTopography API key with a minimal request.
    
    Args:
        api_key: API key to validate
    
    Returns:
        Tuple (is_valid, error_message) - is_valid: True/False/None, error_message: string or None
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty"
    
    # Strip whitespace from key
    api_key = api_key.strip()
    
    # Use coordinates in central Spain (guaranteed to have SRTM data)
    # Small area to minimize download
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
            
            # Unexpected response
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
            # Bad request might mean the key format is wrong
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
    """
    Get list of dataset items for Blender EnumProperty.
    
    Returns:
        List of tuples (identifier, name, description)
    """
    items = []
    for key, info in DEM_DATASETS.items():
        items.append((
            key,
            info['name'],
            f"{info['description']} - {info['coverage']}"
        ))
    return items

