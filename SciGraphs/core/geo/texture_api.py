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
    """Return the addon preferences, or None outside Blender."""
    try:
        import bpy
        addon = bpy.context.preferences.addons.get("SciGraphs")
        if addon:
            return addon.preferences
    except:
        pass
    return None


def get_cache_filepath(theme: str, resolution: str, provider: str = "default") -> str:
    """Return the cache path for one texture, creating the cache dir."""
    os.makedirs(TEXTURE_CACHE_DIR, exist_ok=True)
    filename = f"{provider}_{theme}_{resolution}.jpg"
    return os.path.join(TEXTURE_CACHE_DIR, filename)


def is_texture_cached(theme: str, resolution: str, provider: str = "default") -> bool:
    """Report whether a usable copy of the texture is already on disk.

    Files under 1000 bytes count as absent: a failed download leaves an error
    page behind, and it would otherwise be served as the texture forever.
    """
    filepath = get_cache_filepath(theme, resolution, provider)
    return os.path.exists(filepath) and os.path.getsize(filepath) > 1000


def _make_request(url: str, headers: dict = None, timeout: int = 60) -> bytes:
    """GET a URL and return the body. Raises on any HTTP or network error."""
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
    """Fetch NASA Earth imagery for a theme, returning the cached file path.

    DEMO_KEY works but is rate limited. Themes other than NASA_BLUE_MARBLE and
    NASA_VIIRS are generated procedurally instead.
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
    """Stitch Blue Marble from GIBS WMTS tiles in EPSG:4326.

    The EPSG:4326 endpoint is what makes the result equirectangular; the Web
    Mercator endpoint would need reprojecting before it could wrap a globe.
    """
    resolution_map = {
        '2K': (8, 4),    # 32 tiles of 256 px = 2048x1024
        '4K': (16, 8),
        '8K': (32, 16),
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
    """Pull a Blue Marble composite from public CDNs when GIBS is unreachable."""
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
    """Download the VIIRS Earth at Night texture."""
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
    """Download a WMTS tile grid in parallel and paste it into one image.

    Missing tiles are left at the background fill rather than aborting, so a
    partial download still produces a usable texture.
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


def download_texture(
    theme: str,
    resolution: str = '4K',
    force_download: bool = False
) -> Optional[str]:
    """Get an equirectangular Earth texture from the configured provider.

    Reads provider and API key from addon preferences, falling back to NASA.
    A failed download always ends in a procedural texture, so this returns None
    only if even that fails.
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
        
    else:
        result = _generate_procedural_texture(theme, resolution)
    
    if result is None:
        print("  Download failed, generating procedural fallback texture")
        result = _generate_procedural_texture(theme, resolution)
    
    return result


def _generate_procedural_texture(theme: str, resolution: str) -> Optional[str]:
    """Synthesize a texture locally with numpy and PIL. Needs both installed."""
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
    """Fake continents: blobs of land over ocean, with ice past 63 degrees.

    The seed is fixed so the same invented geography comes back every run.
    These landmasses are decorative and do not correspond to real ones.
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
    """Night lights: 15 real cities at their true coordinates, plus 200 random
    glows scattered between 60 south and 70 north.
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
    
    # Row 0 is the north pole, so latitude runs 90 down to -90 here.
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
    """Dark blue-gray base texture with noise, for overlaying data on."""
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
    """Topographic-style bands, shading by distance from the equator."""
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
    """Neutral dark gradient meant to sit under semitransparent data."""
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
    """Return (texture_path, material_hints) for a globe theme.

    The hints are the shading settings that go with each texture: roughness and
    specular for land and water, bump and emission strength, and alpha for
    DATA_OVERLAY. Theme 'NONE' returns no path, just default hints.
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
    """Delete every cached texture and return how many files went."""
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
    """Return the texture cache size as (bytes, formatted string)."""
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
