import os
import urllib.request
import urllib.parse
import json
import math
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

TEXTURE_CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    ".naturalearth_cache",
    "textures"
)


def get_preferences():
    """
    Get SciGraphs addon preferences.
    
    Returns:
        AddonPreferences instance or None if not available
    """
    try:
        import bpy
        addon = bpy.context.preferences.addons.get("SciGraphs")
        if addon:
            return addon.preferences
    except:
        pass
    return None


def get_cache_filepath(theme: str, resolution: str, provider: str = "default") -> str:
    """
    Generate a cache filepath for a given texture theme and resolution.
    
    Args:
        theme: Texture theme identifier
        resolution: Resolution identifier
        provider: API provider name
    
    Returns:
        Full path to the cached texture file
    """
    os.makedirs(TEXTURE_CACHE_DIR, exist_ok=True)
    filename = f"{provider}_{theme}_{resolution}.jpg"
    return os.path.join(TEXTURE_CACHE_DIR, filename)


def is_texture_cached(theme: str, resolution: str, provider: str = "default") -> bool:
    """
    Check if a texture is already cached locally.
    
    Args:
        theme: Texture theme identifier
        resolution: Resolution identifier
        provider: API provider name
    
    Returns:
        True if the texture file exists in cache
    """
    filepath = get_cache_filepath(theme, resolution, provider)
    return os.path.exists(filepath) and os.path.getsize(filepath) > 1000


def _make_request(url: str, headers: dict = None, timeout: int = 60) -> bytes:
    """
    Make an HTTP request with proper headers and error handling.
    
    Args:
        url: URL to request
        headers: Optional headers dict
        timeout: Request timeout in seconds
    
    Returns:
        Response content as bytes
    """
    if headers is None:
        headers = {}
    
    headers.setdefault('User-Agent', 'SciGraphs-Blender-Addon/1.0')
    headers.setdefault('Accept', 'image/jpeg,image/png,image/*,*/*')
    
    request = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_nasa_texture(
    theme: str,
    resolution: str,
    api_key: str = "DEMO_KEY"
) -> Optional[str]:
    """
    Download Earth imagery using NASA APIs.
    
    Uses NASA EPIC (Earth Polychromatic Imaging Camera) API for recent imagery
    or falls back to pre-rendered Blue Marble composites.
    
    Args:
        theme: Theme identifier
        resolution: Target resolution
        api_key: NASA API key (DEMO_KEY has rate limits)
    
    Returns:
        Path to downloaded texture or None
    """
    cache_path = get_cache_filepath(theme, resolution, "nasa")
    
    if is_texture_cached(theme, resolution, "nasa"):
        print(f"  Using cached NASA texture: {cache_path}")
        return cache_path
    
    os.makedirs(TEXTURE_CACHE_DIR, exist_ok=True)
    
    print(f"  Downloading NASA {theme} texture...")
    
    if theme == 'NASA_BLUE_MARBLE':
        return _download_nasa_blue_marble(cache_path, resolution, api_key)
    elif theme == 'NASA_VIIRS':
        return _download_nasa_viirs_nightlights(cache_path, resolution, api_key)
    else:
        return _generate_procedural_texture(theme, resolution)


def _download_nasa_blue_marble(cache_path: str, resolution: str, api_key: str) -> Optional[str]:
    """
    Download NASA Blue Marble imagery.
    
    Uses NASA Worldview GIBS service for reliable tile downloads,
    then stitches tiles into an equirectangular projection.
    """
    resolution_map = {
        '2K': (8, 4),    # 8x4 = 32 tiles at 256px = 2048x1024
        '4K': (16, 8),   # 16x8 = 128 tiles
        '8K': (32, 16),  # 32x16 = 512 tiles
    }
    
    tiles_x, tiles_y = resolution_map.get(resolution, (16, 8))
    tile_size = 256
    
    gibs_base = "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best"
    layer = "BlueMarble_NextGeneration"
    
    try:
        img = _stitch_wmts_tiles(
            gibs_base,
            layer,
            tiles_x,
            tiles_y,
            tile_size,
            cache_path
        )
        if img:
            return cache_path
    except Exception as e:
        print(f"  GIBS download failed: {e}")
    
    return _try_alternative_blue_marble(cache_path, resolution)


def _try_alternative_blue_marble(cache_path: str, resolution: str) -> Optional[str]:
    """
    Try alternative sources for Blue Marble imagery.
    
    Falls back to publicly available composite images from reliable CDNs.
    """
    alternative_urls = [
        "https://www.solarsystemscope.com/textures/download/2k_earth_daymap.jpg",
        "https://www.solarsystemscope.com/textures/download/8k_earth_daymap.jpg",
        "https://planetpixelemporium.com/download/download.php?earthmap1k.jpg",
    ]
    
    res_map = {'2K': 0, '4K': 0, '8K': 1}
    url_index = res_map.get(resolution, 0)
    
    for i, url in enumerate(alternative_urls):
        if i < url_index:
            continue
        try:
            print(f"  Trying alternative source: {url}")
            data = _make_request(url, timeout=120)
            if len(data) > 10000:
                with open(cache_path, 'wb') as f:
                    f.write(data)
                print(f"  Downloaded from alternative source")
                return cache_path
        except Exception as e:
            print(f"  Source failed: {e}")
            continue
    
    print("  All download sources failed, generating procedural texture")
    return _generate_procedural_texture('NASA_BLUE_MARBLE', resolution)


def _download_nasa_viirs_nightlights(cache_path: str, resolution: str, api_key: str) -> Optional[str]:
    """
    Download NASA VIIRS Earth at Night imagery.
    """
    alternative_urls = [
        "https://www.solarsystemscope.com/textures/download/2k_earth_nightmap.jpg",
        "https://www.solarsystemscope.com/textures/download/8k_earth_nightmap.jpg",
    ]
    
    res_map = {'2K': 0, '4K': 0, '8K': 1}
    url_index = res_map.get(resolution, 0)
    
    for url in alternative_urls[url_index:]:
        try:
            print(f"  Trying: {url}")
            data = _make_request(url, timeout=120)
            if len(data) > 10000:
                with open(cache_path, 'wb') as f:
                    f.write(data)
                print(f"  Downloaded night lights texture")
                return cache_path
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    return _generate_procedural_texture('NASA_VIIRS', resolution)


def _stitch_wmts_tiles(
    base_url: str,
    layer: str,
    tiles_x: int,
    tiles_y: int,
    tile_size: int,
    output_path: str
) -> bool:
    """
    Download and stitch WMTS tiles into a single equirectangular image.
    
    Args:
        base_url: WMTS service base URL
        layer: Layer name to request
        tiles_x: Number of tiles horizontally
        tiles_y: Number of tiles vertically
        tile_size: Size of each tile in pixels
        output_path: Path to save the stitched image
    
    Returns:
        True if successful
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("  PIL/Pillow required for tile stitching")
        return False
    
    width = tiles_x * tile_size
    height = tiles_y * tile_size
    
    result = Image.new('RGB', (width, height), (20, 40, 80))
    
    zoom = int(math.log2(tiles_x))
    
    def download_tile(col, row):
        url = f"{base_url}/{layer}/default/2021-01-01/250m/{zoom}/{row}/{col}.jpg"
        try:
            data = _make_request(url, timeout=30)
            return col, row, data
        except:
            return col, row, None
    
    print(f"  Downloading {tiles_x * tiles_y} tiles...")
    
    downloaded = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for row in range(tiles_y):
            for col in range(tiles_x):
                futures.append(executor.submit(download_tile, col, row))
        
        for future in as_completed(futures):
            col, row, data = future.result()
            if data:
                try:
                    import io
                    tile_img = Image.open(io.BytesIO(data))
                    result.paste(tile_img, (col * tile_size, row * tile_size))
                    downloaded += 1
                except:
                    failed += 1
            else:
                failed += 1
            
            total = tiles_x * tiles_y
            if (downloaded + failed) % 20 == 0:
                print(f"    Progress: {downloaded + failed}/{total} tiles")
    
    if downloaded > 0:
        result.save(output_path, 'JPEG', quality=95)
        print(f"  Stitched {downloaded} tiles ({failed} failed)")
        return True
    
    return False


def download_mapbox_texture(
    theme: str,
    resolution: str,
    api_key: str
) -> Optional[str]:
    """
    Download satellite imagery using Mapbox Static Tiles API.
    
    Stitches map tiles into an equirectangular projection.
    Requires valid Mapbox access token.
    
    Args:
        theme: Theme identifier
        resolution: Target resolution
        api_key: Mapbox access token
    
    Returns:
        Path to downloaded texture or None
    """
    if not api_key:
        print("  Mapbox API key required")
        return None
    
    cache_path = get_cache_filepath(theme, resolution, "mapbox")
    
    if is_texture_cached(theme, resolution, "mapbox"):
        print(f"  Using cached Mapbox texture: {cache_path}")
        return cache_path
    
    try:
        from PIL import Image
        import io
    except ImportError:
        print("  PIL/Pillow required for Mapbox textures")
        return None
    
    resolution_map = {
        '2K': (8, 4, 2),
        '4K': (16, 8, 3),
        '8K': (32, 16, 4),
    }
    
    tiles_x, tiles_y, zoom = resolution_map.get(resolution, (16, 8, 3))
    tile_size = 256
    
    style = "mapbox.satellite"
    
    width = tiles_x * tile_size
    height = tiles_y * tile_size
    result = Image.new('RGB', (width, height), (20, 40, 80))
    
    print(f"  Downloading Mapbox satellite tiles (zoom {zoom})...")
    
    downloaded = 0
    
    for row in range(tiles_y):
        for col in range(tiles_x):
            lon = (col / tiles_x) * 360 - 180
            lat = 90 - (row / tiles_y) * 180
            
            url = (
                f"https://api.mapbox.com/styles/v1/{style}/static/"
                f"{lon},{lat},{zoom}/{tile_size}x{tile_size}@2x"
                f"?access_token={api_key}&attribution=false&logo=false"
            )
            
            try:
                data = _make_request(url, timeout=30)
                tile_img = Image.open(io.BytesIO(data))
                result.paste(tile_img, (col * tile_size, row * tile_size))
                downloaded += 1
            except Exception as e:
                pass
        
        print(f"    Row {row + 1}/{tiles_y} complete")
    
    if downloaded > 0:
        os.makedirs(TEXTURE_CACHE_DIR, exist_ok=True)
        result.save(cache_path, 'JPEG', quality=95)
        print(f"  Saved Mapbox texture: {cache_path}")
        return cache_path
    
    return None


def download_maptiler_texture(
    theme: str,
    resolution: str,
    api_key: str
) -> Optional[str]:
    """
    Download satellite imagery using MapTiler API.
    
    Args:
        theme: Theme identifier
        resolution: Target resolution
        api_key: MapTiler API key
    
    Returns:
        Path to downloaded texture or None
    """
    if not api_key:
        print("  MapTiler API key required")
        return None
    
    cache_path = get_cache_filepath(theme, resolution, "maptiler")
    
    if is_texture_cached(theme, resolution, "maptiler"):
        print(f"  Using cached MapTiler texture: {cache_path}")
        return cache_path
    
    try:
        from PIL import Image
        import io
    except ImportError:
        print("  PIL/Pillow required for MapTiler textures")
        return None
    
    resolution_map = {
        '2K': (8, 4, 2),
        '4K': (16, 8, 3),
        '8K': (32, 16, 4),
    }
    
    tiles_x, tiles_y, zoom = resolution_map.get(resolution, (16, 8, 3))
    tile_size = 256
    
    width = tiles_x * tile_size
    height = tiles_y * tile_size
    result = Image.new('RGB', (width, height), (20, 40, 80))
    
    print(f"  Downloading MapTiler satellite tiles (zoom {zoom})...")
    
    downloaded = 0
    
    for row in range(tiles_y):
        for col in range(tiles_x):
            x = col
            y = row
            
            url = f"https://api.maptiler.com/tiles/satellite-v2/{zoom}/{x}/{y}.jpg?key={api_key}"
            
            try:
                data = _make_request(url, timeout=30)
                tile_img = Image.open(io.BytesIO(data))
                result.paste(tile_img, (col * tile_size, row * tile_size))
                downloaded += 1
            except Exception as e:
                pass
        
        print(f"    Row {row + 1}/{tiles_y} complete")
    
    if downloaded > 0:
        os.makedirs(TEXTURE_CACHE_DIR, exist_ok=True)
        result.save(cache_path, 'JPEG', quality=95)
        print(f"  Saved MapTiler texture: {cache_path}")
        return cache_path
    
    return None


def download_texture(
    theme: str,
    resolution: str = '4K',
    force_download: bool = False
) -> Optional[str]:
    """
    Download an equirectangular Earth texture using the configured provider.
    
    This is the main entry point for texture downloads. It reads the provider
    and API key from addon preferences and delegates to the appropriate
    download function.
    
    Args:
        theme: Texture theme identifier
        resolution: Texture resolution ('2K', '4K', '8K')
        force_download: If True, re-download even if cached
    
    Returns:
        Path to the downloaded texture file, or None if download failed
    """
    prefs = get_preferences()
    
    if prefs:
        provider = prefs.globe_texture_provider
    else:
        provider = 'NASA'
    
    if theme in ['URBAN_DARK', 'TOPOGRAPHIC_SHADED', 'DATA_OVERLAY']:
        return _generate_procedural_texture(theme, resolution)
    
    if provider == 'PROCEDURAL':
        return _generate_procedural_texture(theme, resolution)
    
    cache_path = get_cache_filepath(theme, resolution, provider.lower())
    if not force_download and os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        print(f"  Using cached texture: {cache_path}")
        return cache_path
    
    if provider == 'NASA':
        api_key = prefs.nasa_api_key if prefs else "DEMO_KEY"
        result = download_nasa_texture(theme, resolution, api_key)
        
    elif provider == 'MAPBOX':
        api_key = prefs.mapbox_api_key if prefs else ""
        if not api_key:
            print("  No Mapbox API key configured. Set it in addon preferences.")
            print("  Falling back to procedural texture.")
            return _generate_procedural_texture(theme, resolution)
        result = download_mapbox_texture(theme, resolution, api_key)
        
    elif provider == 'MAPTILER':
        api_key = prefs.maptiler_api_key if prefs else ""
        if not api_key:
            print("  No MapTiler API key configured. Set it in addon preferences.")
            print("  Falling back to procedural texture.")
            return _generate_procedural_texture(theme, resolution)
        result = download_maptiler_texture(theme, resolution, api_key)
        
    elif provider == 'STADIA':
        api_key = prefs.stadia_api_key if prefs else ""
        if not api_key:
            print("  No Stadia Maps API key configured. Set it in addon preferences.")
            print("  Falling back to procedural texture.")
            return _generate_procedural_texture(theme, resolution)
        result = _generate_procedural_texture(theme, resolution)
        
    else:
        result = _generate_procedural_texture(theme, resolution)
    
    if result is None:
        print("  Download failed, generating procedural fallback texture")
        result = _generate_procedural_texture(theme, resolution)
    
    return result


def _generate_procedural_texture(theme: str, resolution: str) -> Optional[str]:
    """
    Generate a procedural texture for themes that do not use external downloads.
    
    Creates gradient or pattern textures locally using numpy and PIL.
    
    Args:
        theme: Theme identifier
        resolution: Target resolution
    
    Returns:
        Path to generated texture file or None
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("  PIL/numpy required for procedural textures")
        return None
    
    resolution_map = {
        '2K': (2048, 1024),
        '4K': (4096, 2048),
        '8K': (8192, 4096),
    }
    
    width, height = resolution_map.get(resolution, (4096, 2048))
    cache_path = get_cache_filepath(theme, resolution, "procedural")
    
    os.makedirs(TEXTURE_CACHE_DIR, exist_ok=True)
    
    print(f"  Generating procedural {theme} texture ({resolution})...")
    
    if theme in ['NASA_BLUE_MARBLE', 'NATURAL_EARTH']:
        img_array = _create_earth_texture(width, height)
    elif theme == 'NASA_VIIRS':
        img_array = _create_night_lights_texture(width, height)
    elif theme == 'URBAN_DARK':
        img_array = _create_urban_dark_texture(width, height)
    elif theme == 'TOPOGRAPHIC_SHADED':
        img_array = _create_topographic_texture(width, height)
    elif theme == 'DATA_OVERLAY':
        img_array = _create_data_overlay_texture(width, height)
    else:
        img_array = _create_earth_texture(width, height)
    
    img = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    img.save(cache_path, 'JPEG', quality=95)
    
    print(f"  Generated procedural texture: {cache_path}")
    return cache_path


def _create_earth_texture(width: int, height: int):
    """
    Create a procedural Earth-like texture with land and ocean.
    """
    import numpy as np
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    y_coords = np.linspace(-90, 90, height)[:, np.newaxis]
    x_coords = np.linspace(-180, 180, width)[np.newaxis, :]
    
    y_norm = np.tile(y_coords, (1, width))
    x_norm = np.tile(x_coords, (height, 1))
    
    np.random.seed(42)
    
    land_mask = np.zeros((height, width), dtype=np.float32)
    
    for _ in range(15):
        cx = np.random.uniform(-180, 180)
        cy = np.random.uniform(-60, 60)
        rx = np.random.uniform(20, 80)
        ry = np.random.uniform(15, 50)
        
        dist = ((x_norm - cx) / rx) ** 2 + ((y_norm - cy) / ry) ** 2
        land_mask += np.exp(-dist * 2)
    
    land_mask = np.clip(land_mask, 0, 1)
    land_mask = (land_mask > 0.3).astype(np.float32)
    
    noise = np.random.rand(height, width) * 0.1
    
    ocean_r = 20 + noise * 20
    ocean_g = 50 + noise * 30
    ocean_b = 120 + noise * 40
    
    land_base = 60 + y_norm * 0.3
    land_r = land_base + 40 + noise * 30
    land_g = land_base + 80 + noise * 40
    land_b = land_base + 20 + noise * 20
    
    img[:, :, 0] = land_mask * land_r + (1 - land_mask) * ocean_r
    img[:, :, 1] = land_mask * land_g + (1 - land_mask) * ocean_g
    img[:, :, 2] = land_mask * land_b + (1 - land_mask) * ocean_b
    
    polar = np.abs(y_norm) / 90.0
    ice_factor = np.clip((polar - 0.7) * 5, 0, 1)
    img[:, :, 0] = img[:, :, 0] * (1 - ice_factor) + 240 * ice_factor
    img[:, :, 1] = img[:, :, 1] * (1 - ice_factor) + 245 * ice_factor
    img[:, :, 2] = img[:, :, 2] * (1 - ice_factor) + 250 * ice_factor
    
    return np.clip(img, 0, 255)


def _create_night_lights_texture(width: int, height: int):
    """
    Create a procedural Earth at night texture with city lights.
    """
    import numpy as np
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    img[:, :, 0] = 5
    img[:, :, 1] = 8
    img[:, :, 2] = 15
    
    np.random.seed(123)
    
    cities = [
        (40.7, -74.0, 1.0),    # New York
        (51.5, -0.1, 0.9),     # London
        (35.7, 139.7, 1.0),    # Tokyo
        (31.2, 121.5, 0.95),   # Shanghai
        (19.4, -99.1, 0.8),    # Mexico City
        (55.8, 37.6, 0.8),     # Moscow
        (-23.5, -46.6, 0.85),  # Sao Paulo
        (28.6, 77.2, 0.9),     # Delhi
        (39.9, 116.4, 0.95),   # Beijing
        (34.1, -118.2, 0.85),  # Los Angeles
        (48.9, 2.3, 0.8),      # Paris
        (35.2, -106.6, 0.5),   # Albuquerque
        (52.5, 13.4, 0.7),     # Berlin
        (41.9, 12.5, 0.6),     # Rome
        (-33.9, 151.2, 0.7),   # Sydney
    ]
    
    for _ in range(200):
        lat = np.random.uniform(-60, 70)
        lon = np.random.uniform(-180, 180)
        intensity = np.random.uniform(0.1, 0.5)
        cities.append((lat, lon, intensity))
    
    y_coords = np.linspace(90, -90, height)[:, np.newaxis]
    x_coords = np.linspace(-180, 180, width)[np.newaxis, :]
    
    y_grid = np.tile(y_coords, (1, width))
    x_grid = np.tile(x_coords, (height, 1))
    
    for lat, lon, intensity in cities:
        dist = np.sqrt((x_grid - lon) ** 2 + (y_grid - lat) ** 2)
        light = np.exp(-dist ** 2 / (5 * intensity)) * intensity * 255
        
        img[:, :, 0] += light * 1.0
        img[:, :, 1] += light * 0.9
        img[:, :, 2] += light * 0.6
    
    return np.clip(img, 0, 255)


def _create_urban_dark_texture(width: int, height: int):
    """
    Create a dark map texture with subtle continental outlines.
    """
    import numpy as np
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    img[:, :, 0] = 15
    img[:, :, 1] = 20
    img[:, :, 2] = 35
    
    noise = np.random.rand(height, width) * 10
    img[:, :, 0] += noise
    img[:, :, 1] += noise
    img[:, :, 2] += noise * 1.5
    
    return np.clip(img, 0, 255)


def _create_topographic_texture(width: int, height: int):
    """
    Create a topographic-style texture with elevation gradients.
    """
    import numpy as np
    
    img = np.zeros((height, width, 3), dtype=np.float32)
    
    y_coords = np.linspace(-1, 1, height)
    elevation = np.abs(y_coords)
    elevation = elevation[:, np.newaxis]
    elevation = np.tile(elevation, (1, width))
    
    img[:, :, 0] = 180 - elevation * 100
    img[:, :, 1] = 160 + elevation * 80
    img[:, :, 2] = 120 - elevation * 60
    
    return np.clip(img, 0, 255)


def _create_data_overlay_texture(width: int, height: int):
    """
    Create a semi-transparent neutral texture for data overlay mode.
    """
    import numpy as np
    
    img = np.ones((height, width, 3), dtype=np.float32) * 40
    
    y_coords = np.linspace(0, 1, height)
    gradient = y_coords[:, np.newaxis]
    gradient = np.tile(gradient, (1, width))
    
    img[:, :, 0] += gradient * 20
    img[:, :, 1] += gradient * 25
    img[:, :, 2] += gradient * 35
    
    return np.clip(img, 0, 255)


def get_texture_for_globe(
    theme: str,
    resolution: str = '4K'
) -> Tuple[Optional[str], dict]:
    """
    Get the appropriate texture for globe rendering along with material hints.
    
    This is the main entry point for obtaining textures. It handles downloading,
    caching, and provides material configuration hints based on the theme.
    
    Args:
        theme: Globe theme identifier
        resolution: Desired texture resolution
    
    Returns:
        Tuple of (texture_path, material_hints_dict)
    """
    material_hints = {
        'NASA_BLUE_MARBLE': {
            'water_specular': 0.8,
            'water_roughness': 0.15,
            'land_roughness': 0.7,
            'bump_strength': 0.05,
            'emission_strength': 0.0,
        },
        'NASA_VIIRS': {
            'water_specular': 0.3,
            'water_roughness': 0.5,
            'land_roughness': 0.9,
            'bump_strength': 0.0,
            'emission_strength': 0.8,
        },
        'NATURAL_EARTH': {
            'water_specular': 0.6,
            'water_roughness': 0.25,
            'land_roughness': 0.65,
            'bump_strength': 0.1,
            'emission_strength': 0.0,
        },
        'URBAN_DARK': {
            'water_specular': 0.2,
            'water_roughness': 0.8,
            'land_roughness': 0.9,
            'bump_strength': 0.0,
            'emission_strength': 0.5,
        },
        'TOPOGRAPHIC_SHADED': {
            'water_specular': 0.5,
            'water_roughness': 0.3,
            'land_roughness': 0.6,
            'bump_strength': 0.15,
            'emission_strength': 0.0,
        },
        'DATA_OVERLAY': {
            'water_specular': 0.1,
            'water_roughness': 0.9,
            'land_roughness': 0.9,
            'bump_strength': 0.0,
            'emission_strength': 0.0,
            'alpha': 0.3,
        },
    }
    
    hints = material_hints.get(theme, {
        'water_specular': 0.5,
        'water_roughness': 0.3,
        'land_roughness': 0.7,
        'bump_strength': 0.05,
        'emission_strength': 0.0,
    })
    
    if theme == 'NONE':
        return None, hints
    
    texture_path = download_texture(theme, resolution)
    
    return texture_path, hints


def clear_texture_cache() -> int:
    """
    Remove all cached textures to free disk space.
    
    Returns:
        Number of files deleted
    """
    if not os.path.exists(TEXTURE_CACHE_DIR):
        return 0
    
    deleted = 0
    for filename in os.listdir(TEXTURE_CACHE_DIR):
        filepath = os.path.join(TEXTURE_CACHE_DIR, filename)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
                deleted += 1
            except OSError:
                pass
    
    print(f"Cleared {deleted} cached textures")
    return deleted


def get_cache_size() -> Tuple[int, str]:
    """
    Calculate the total size of cached textures.
    
    Returns:
        Tuple of (size_in_bytes, human_readable_string)
    """
    if not os.path.exists(TEXTURE_CACHE_DIR):
        return 0, "0 B"
    
    total_size = 0
    for filename in os.listdir(TEXTURE_CACHE_DIR):
        filepath = os.path.join(TEXTURE_CACHE_DIR, filename)
        if os.path.isfile(filepath):
            total_size += os.path.getsize(filepath)
    
    if total_size < 1024:
        human = f"{total_size} B"
    elif total_size < 1024 * 1024:
        human = f"{total_size / 1024:.1f} KB"
    elif total_size < 1024 * 1024 * 1024:
        human = f"{total_size / 1024 / 1024:.1f} MB"
    else:
        human = f"{total_size / 1024 / 1024 / 1024:.2f} GB"
    
    return total_size, human
