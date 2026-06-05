"""Basemap imagery for OSMnx terrain meshes.

Downloads slippy-map (XYZ) tiles or WMS images for an arbitrary geographic
bounding box and stitches them into a single PNG ready to be UV-mapped onto
a 3D terrain object.

Independent from :mod:`bpy` so it can be reused/tested headless. Network code
is :class:`KeyboardInterrupt`-safe so cancelling from Blender does not freeze
the UI waiting for in-flight HTTP requests.
"""

from __future__ import annotations

import io
import math
import os
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from ...utils.logger import log


# ---------------------------------------------------------------------------
# Tile source registry
# ---------------------------------------------------------------------------

#: Each entry describes a slippy-map XYZ tile style. ``url`` is a Python
#: ``str.format`` template using ``{z}`` ``{x}`` ``{y}`` and optionally
#: ``{key}`` for the API key.  ``needs_key`` flags whether the addon must
#: read a key from preferences before requesting tiles, and ``key_pref``
#: names the :class:`AddonPreferences` attribute holding that key so the
#: caller can resolve it generically.  ``provider`` groups styles by the
#: API that serves them, for UI grouping and attribution.
#:
#: Many APIs expose dozens of styles from a single key; this registry lists
#: the most useful raster styles per provider so users can pick the look
#: that fits their scene (satellite, streets, dark, light, topographic, ...).
ESRI_ATTRIBUTION = (
    'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, '
    'and the GIS User Community'
)
CARTO_ATTRIBUTION = (
    '© OpenStreetMap contributors © CARTO'
)
STADIA_ATTRIBUTION = (
    '© Stadia Maps © Stamen Design © OpenMapTiles © OpenStreetMap contributors'
)


def _esri_service(service: str) -> str:
    return (
        'https://server.arcgisonline.com/ArcGIS/rest/services/'
        f'{service}/MapServer/tile/{{z}}/{{y}}/{{x}}'
    )


def _carto_style(style: str) -> str:
    return f'https://basemaps.cartocdn.com/{style}/{{z}}/{{x}}/{{y}}.png'


def _mapbox_style(style: str) -> str:
    return (
        f'https://api.mapbox.com/styles/v1/mapbox/{style}/tiles/256/'
        '{z}/{x}/{y}?access_token={key}'
    )


def _maptiler_map(map_id: str, ext: str = 'jpg') -> str:
    return f'https://api.maptiler.com/maps/{map_id}/256/{{z}}/{{x}}/{{y}}.{ext}?key={{key}}'


def _stadia_style(style: str, ext: str = 'png') -> str:
    return f'https://tiles.stadiamaps.com/tiles/{style}/{{z}}/{{x}}/{{y}}.{ext}?api_key={{key}}'


TILE_SOURCES: dict[str, dict] = {
    # --- Esri (no key required) -------------------------------------------
    'ESRI_IMAGERY': {
        'name': 'Esri · Satellite (World Imagery)',
        'provider': 'Esri',
        'url': _esri_service('World_Imagery'),
        'attribution': ESRI_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 19, 'needs_key': False,
        'key_pref': None, 'extension': 'jpg',
    },
    'ESRI_STREET': {
        'name': 'Esri · Streets',
        'provider': 'Esri',
        'url': _esri_service('World_Street_Map'),
        'attribution': ESRI_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 19, 'needs_key': False,
        'key_pref': None, 'extension': 'jpg',
    },
    'ESRI_TOPO': {
        'name': 'Esri · Topographic',
        'provider': 'Esri',
        'url': _esri_service('World_Topo_Map'),
        'attribution': ESRI_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 19, 'needs_key': False,
        'key_pref': None, 'extension': 'jpg',
    },
    'ESRI_DARK_GRAY': {
        'name': 'Esri · Dark Gray Canvas',
        'provider': 'Esri',
        'url': _esri_service('Canvas/World_Dark_Gray_Base'),
        'attribution': ESRI_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 16, 'needs_key': False,
        'key_pref': None, 'extension': 'jpg',
    },
    'ESRI_LIGHT_GRAY': {
        'name': 'Esri · Light Gray Canvas',
        'provider': 'Esri',
        'url': _esri_service('Canvas/World_Light_Gray_Base'),
        'attribution': ESRI_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 16, 'needs_key': False,
        'key_pref': None, 'extension': 'jpg',
    },
    'ESRI_OCEAN': {
        'name': 'Esri · Ocean Base',
        'provider': 'Esri',
        'url': _esri_service('Ocean/World_Ocean_Base'),
        'attribution': ESRI_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 13, 'needs_key': False,
        'key_pref': None, 'extension': 'jpg',
    },
    'ESRI_HILLSHADE': {
        'name': 'Esri · Hillshade',
        'provider': 'Esri',
        'url': _esri_service('Elevation/World_Hillshade'),
        'attribution': ESRI_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 16, 'needs_key': False,
        'key_pref': None, 'extension': 'jpg',
    },
    'ESRI_TERRAIN': {
        'name': 'Esri · Terrain Base',
        'provider': 'Esri',
        'url': _esri_service('World_Terrain_Base'),
        'attribution': ESRI_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 13, 'needs_key': False,
        'key_pref': None, 'extension': 'jpg',
    },

    # --- OpenStreetMap (no key required) ----------------------------------
    'OSM': {
        'name': 'OpenStreetMap · Standard',
        'provider': 'OpenStreetMap',
        'url': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        'attribution': '© OpenStreetMap contributors',
        'tile_size': 256, 'max_zoom': 19, 'needs_key': False,
        'key_pref': None, 'extension': 'png',
    },
    'OPENTOPOMAP': {
        'name': 'OpenTopoMap · Topographic',
        'provider': 'OpenStreetMap',
        'url': 'https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
        'attribution': (
            '© OpenStreetMap contributors, SRTM — © OpenTopoMap (CC-BY-SA)'
        ),
        'tile_size': 256, 'max_zoom': 17, 'needs_key': False,
        'key_pref': None, 'extension': 'png',
    },

    # --- CARTO (no key required for fair use) -----------------------------
    'CARTO_VOYAGER': {
        'name': 'CARTO · Voyager',
        'provider': 'CARTO',
        'url': _carto_style('rastertiles/voyager'),
        'attribution': CARTO_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 20, 'needs_key': False,
        'key_pref': None, 'extension': 'png',
    },
    'CARTO_POSITRON': {
        'name': 'CARTO · Positron (Light)',
        'provider': 'CARTO',
        'url': _carto_style('light_all'),
        'attribution': CARTO_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 20, 'needs_key': False,
        'key_pref': None, 'extension': 'png',
    },
    'CARTO_DARK_MATTER': {
        'name': 'CARTO · Dark Matter',
        'provider': 'CARTO',
        'url': _carto_style('dark_all'),
        'attribution': CARTO_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 20, 'needs_key': False,
        'key_pref': None, 'extension': 'png',
    },

    # --- Mapbox (requires access token) -----------------------------------
    'MAPBOX_SATELLITE': {
        'name': 'Mapbox · Satellite',
        'provider': 'Mapbox',
        'url': _mapbox_style('satellite-v9'),
        'attribution': '© Mapbox © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'mapbox_api_key', 'extension': 'jpg',
    },
    'MAPBOX_SATELLITE_STREETS': {
        'name': 'Mapbox · Satellite Streets',
        'provider': 'Mapbox',
        'url': _mapbox_style('satellite-streets-v12'),
        'attribution': '© Mapbox © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'mapbox_api_key', 'extension': 'jpg',
    },
    'MAPBOX_STREETS': {
        'name': 'Mapbox · Streets',
        'provider': 'Mapbox',
        'url': _mapbox_style('streets-v12'),
        'attribution': '© Mapbox © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'mapbox_api_key', 'extension': 'png',
    },
    'MAPBOX_OUTDOORS': {
        'name': 'Mapbox · Outdoors',
        'provider': 'Mapbox',
        'url': _mapbox_style('outdoors-v12'),
        'attribution': '© Mapbox © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'mapbox_api_key', 'extension': 'png',
    },
    'MAPBOX_LIGHT': {
        'name': 'Mapbox · Light',
        'provider': 'Mapbox',
        'url': _mapbox_style('light-v11'),
        'attribution': '© Mapbox © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'mapbox_api_key', 'extension': 'png',
    },
    'MAPBOX_DARK': {
        'name': 'Mapbox · Dark',
        'provider': 'Mapbox',
        'url': _mapbox_style('dark-v11'),
        'attribution': '© Mapbox © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'mapbox_api_key', 'extension': 'png',
    },

    # --- MapTiler (requires key, free tier) -------------------------------
    'MAPTILER_SATELLITE': {
        'name': 'MapTiler · Satellite',
        'provider': 'MapTiler',
        'url': _maptiler_map('satellite', 'jpg'),
        'attribution': '© MapTiler © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'maptiler_api_key', 'extension': 'jpg',
    },
    'MAPTILER_HYBRID': {
        'name': 'MapTiler · Satellite Hybrid',
        'provider': 'MapTiler',
        'url': _maptiler_map('hybrid', 'jpg'),
        'attribution': '© MapTiler © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'maptiler_api_key', 'extension': 'jpg',
    },
    'MAPTILER_STREETS': {
        'name': 'MapTiler · Streets',
        'provider': 'MapTiler',
        'url': _maptiler_map('streets-v2', 'png'),
        'attribution': '© MapTiler © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'maptiler_api_key', 'extension': 'png',
    },
    'MAPTILER_TOPO': {
        'name': 'MapTiler · Topographic',
        'provider': 'MapTiler',
        'url': _maptiler_map('topo-v2', 'png'),
        'attribution': '© MapTiler © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'maptiler_api_key', 'extension': 'png',
    },
    'MAPTILER_OUTDOOR': {
        'name': 'MapTiler · Outdoor',
        'provider': 'MapTiler',
        'url': _maptiler_map('outdoor-v2', 'png'),
        'attribution': '© MapTiler © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'maptiler_api_key', 'extension': 'png',
    },
    'MAPTILER_BASIC': {
        'name': 'MapTiler · Basic',
        'provider': 'MapTiler',
        'url': _maptiler_map('basic-v2', 'png'),
        'attribution': '© MapTiler © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'maptiler_api_key', 'extension': 'png',
    },
    'MAPTILER_WINTER': {
        'name': 'MapTiler · Winter',
        'provider': 'MapTiler',
        'url': _maptiler_map('winter-v2', 'png'),
        'attribution': '© MapTiler © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'maptiler_api_key', 'extension': 'png',
    },
    'MAPTILER_OCEAN': {
        'name': 'MapTiler · Ocean',
        'provider': 'MapTiler',
        'url': _maptiler_map('ocean', 'png'),
        'attribution': '© MapTiler © OpenStreetMap',
        'tile_size': 256, 'max_zoom': 22, 'needs_key': True,
        'key_pref': 'maptiler_api_key', 'extension': 'png',
    },

    # --- Stadia Maps / Stamen (requires key, free tier) -------------------
    'STADIA_ALIDADE_SMOOTH': {
        'name': 'Stadia · Alidade Smooth (Light)',
        'provider': 'Stadia',
        'url': _stadia_style('alidade_smooth', 'png'),
        'attribution': STADIA_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 20, 'needs_key': True,
        'key_pref': 'stadia_api_key', 'extension': 'png',
    },
    'STADIA_ALIDADE_SMOOTH_DARK': {
        'name': 'Stadia · Alidade Smooth Dark',
        'provider': 'Stadia',
        'url': _stadia_style('alidade_smooth_dark', 'png'),
        'attribution': STADIA_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 20, 'needs_key': True,
        'key_pref': 'stadia_api_key', 'extension': 'png',
    },
    'STADIA_OUTDOORS': {
        'name': 'Stadia · Outdoors',
        'provider': 'Stadia',
        'url': _stadia_style('outdoors', 'png'),
        'attribution': STADIA_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 20, 'needs_key': True,
        'key_pref': 'stadia_api_key', 'extension': 'png',
    },
    'STADIA_STAMEN_TONER': {
        'name': 'Stadia · Stamen Toner (B/W)',
        'provider': 'Stadia',
        'url': _stadia_style('stamen_toner', 'png'),
        'attribution': STADIA_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 20, 'needs_key': True,
        'key_pref': 'stadia_api_key', 'extension': 'png',
    },
    'STADIA_STAMEN_TERRAIN': {
        'name': 'Stadia · Stamen Terrain',
        'provider': 'Stadia',
        'url': _stadia_style('stamen_terrain', 'png'),
        'attribution': STADIA_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 18, 'needs_key': True,
        'key_pref': 'stadia_api_key', 'extension': 'png',
    },
    'STADIA_STAMEN_WATERCOLOR': {
        'name': 'Stadia · Stamen Watercolor',
        'provider': 'Stadia',
        'url': _stadia_style('stamen_watercolor', 'jpg'),
        'attribution': STADIA_ATTRIBUTION,
        'tile_size': 256, 'max_zoom': 16, 'needs_key': True,
        'key_pref': 'stadia_api_key', 'extension': 'jpg',
    },
}


#: Backward-compatible aliases for the original flat source keys so scenes
#: saved before the catalog expansion keep working.
_LEGACY_SOURCE_ALIASES: dict[str, str] = {
    'ESRI': 'ESRI_IMAGERY',
    'MAPBOX': 'MAPBOX_SATELLITE',
    'MAPTILER': 'MAPTILER_SATELLITE',
}


def resolve_source(source: str) -> str:
    """Map a (possibly legacy) source key onto a current :data:`TILE_SOURCES` key."""
    if source in TILE_SOURCES:
        return source
    return _LEGACY_SOURCE_ALIASES.get(source, source)


_USER_AGENT = (
    "SciGraphs-Blender-Addon/1.0 "
    "(+https://github.com/SciBlend/SciGraphs)"
)


# ---------------------------------------------------------------------------
# Web Mercator math
# ---------------------------------------------------------------------------

def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Convert WGS84 lon/lat to fractional XYZ tile coordinates.

    Returns floats so callers can compute pixel-perfect crops inside a tile.
    """
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _tile_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Inverse of :func:`_lonlat_to_tile`. Returns NW corner of the tile."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    return lon, math.degrees(lat_rad)


def estimate_tiles(bounds: dict, zoom: int) -> int:
    """Return how many XYZ tiles a request would need (for UI hinting)."""
    x0, y0 = _lonlat_to_tile(bounds['west'], bounds['north'], zoom)
    x1, y1 = _lonlat_to_tile(bounds['east'], bounds['south'], zoom)
    return max(1, int(math.floor(x1) - math.floor(x0) + 1)) * max(
        1, int(math.floor(y1) - math.floor(y0) + 1)
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _default_cache_dir() -> str:
    """Cache root used when caller does not provide one."""
    return os.path.join(tempfile.gettempdir(), "scigraphs_basemaps")


def _tile_cache_path(cache_dir: str, source: str, z: int, x: int, y: int, ext: str) -> str:
    return os.path.join(cache_dir, source.lower(), str(z), str(x), f"{y}.{ext}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_basemap(
    bounds: dict,
    source: str = 'ESRI',
    zoom: int = 16,
    out_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    api_key: Optional[str] = None,
    wms_url: Optional[str] = None,
    wms_layer: Optional[str] = None,
    max_tiles: int = 400,
    max_workers: int = 8,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> tuple[str, dict]:
    """Download a basemap image covering ``bounds``.

    Args:
        bounds: WGS84 dict ``{north, south, east, west}``.
        source: One of :data:`TILE_SOURCES` keys, or ``'WMS'``.
        zoom: Slippy-map zoom level (ignored for WMS).
        out_path: Destination PNG. ``None`` writes into the cache dir.
        cache_dir: Tile cache root. ``None`` uses a temp folder.
        api_key: API key for providers that require one.
        wms_url: Base WMS endpoint (only when ``source='WMS'``).
        wms_layer: WMS layer name.
        max_tiles: Hard cap on tile count to protect against typos.
        max_workers: Parallel downloads.
        progress_cb: Optional ``(done, total)`` callback for UI updates.

    Returns:
        Tuple ``(image_path, metadata)``. ``metadata`` contains the
        actual ``bounds`` of the produced image (XYZ rounding may extend
        the requested area), the ``attribution`` string, the source name
        and the resolution.

    Raises:
        ValueError: bad bounds / unknown source / too many tiles.
        KeyboardInterrupt: re-raised if the user cancels mid-fetch.
        RuntimeError: every download failed.
    """
    _validate_bounds(bounds)

    if source == 'WMS':
        return _fetch_wms_basemap(bounds, wms_url, wms_layer, out_path)

    source = resolve_source(source)
    if source not in TILE_SOURCES:
        raise ValueError(f"Unknown basemap source: {source!r}")

    cfg = TILE_SOURCES[source]
    if cfg['needs_key'] and not api_key:
        raise ValueError(
            f"{cfg['name']} requires an API key. "
            "Set it in addon preferences."
        )

    zoom = max(1, min(int(zoom), cfg['max_zoom']))
    cache_dir = cache_dir or _default_cache_dir()
    return _fetch_xyz_basemap(
        bounds=bounds,
        source=source,
        cfg=cfg,
        zoom=zoom,
        out_path=out_path,
        cache_dir=cache_dir,
        api_key=api_key,
        max_tiles=max_tiles,
        max_workers=max_workers,
        progress_cb=progress_cb,
    )


def clear_cache(cache_dir: Optional[str] = None) -> int:
    """Delete every cached tile under ``cache_dir``. Returns deleted count."""
    cache_dir = cache_dir or _default_cache_dir()
    if not os.path.isdir(cache_dir):
        return 0

    deleted = 0
    for root, _dirs, files in os.walk(cache_dir):
        for name in files:
            try:
                os.remove(os.path.join(root, name))
                deleted += 1
            except OSError:
                pass
    return deleted


# ---------------------------------------------------------------------------
# XYZ implementation
# ---------------------------------------------------------------------------

def _fetch_xyz_basemap(
    *,
    bounds: dict,
    source: str,
    cfg: dict,
    zoom: int,
    out_path: Optional[str],
    cache_dir: str,
    api_key: Optional[str],
    max_tiles: int,
    max_workers: int,
    progress_cb: Optional[Callable[[int, int], None]],
) -> tuple[str, dict]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for basemap fetching") from exc

    tile_size = cfg['tile_size']
    ext = cfg['extension']

    x0_f, y0_f = _lonlat_to_tile(bounds['west'], bounds['north'], zoom)
    x1_f, y1_f = _lonlat_to_tile(bounds['east'], bounds['south'], zoom)
    x_min, x_max = int(math.floor(x0_f)), int(math.floor(x1_f))
    y_min, y_max = int(math.floor(y0_f)), int(math.floor(y1_f))

    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    total = cols * rows
    if total > max_tiles:
        raise ValueError(
            f"Requested zoom {zoom} would need {total} tiles "
            f"(cap is {max_tiles}). Lower the zoom level."
        )

    log(
        f"Basemap: {source} z={zoom} → {cols}×{rows}={total} tiles "
        f"({tile_size * cols}×{tile_size * rows} px)"
    )

    canvas = Image.new('RGB', (cols * tile_size, rows * tile_size), (50, 50, 50))
    completed = 0
    failed = 0

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures: dict = {}
    interrupted = False

    def _submit_all() -> None:
        for ty in range(y_min, y_max + 1):
            for tx in range(x_min, x_max + 1):
                fut = executor.submit(
                    _fetch_tile,
                    cfg=cfg,
                    source=source,
                    z=zoom,
                    x=tx,
                    y=ty,
                    cache_dir=cache_dir,
                    api_key=api_key,
                    ext=ext,
                )
                futures[fut] = (tx, ty)

    try:
        _submit_all()
        for fut in as_completed(futures):
            tx, ty = futures[fut]
            data = fut.result()
            if data is None:
                failed += 1
            else:
                try:
                    tile_img = Image.open(io.BytesIO(data)).convert('RGB')
                    canvas.paste(
                        tile_img,
                        ((tx - x_min) * tile_size, (ty - y_min) * tile_size),
                    )
                except Exception:  # noqa: BLE001
                    failed += 1
            completed += 1
            if progress_cb is not None:
                try:
                    progress_cb(completed, total)
                except Exception:  # noqa: BLE001
                    pass
    except KeyboardInterrupt:
        interrupted = True
        log("Basemap fetch interrupted by user — cancelling pending tiles")
        for f in futures:
            f.cancel()
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=True)

    if interrupted:
        raise KeyboardInterrupt("Basemap fetch cancelled")

    if completed - failed == 0:
        raise RuntimeError(
            "Every tile failed to download. Check your network connection "
            "and the API key (if required)."
        )

    # Crop the canvas to the exact bounds requested. Tiles are aligned to
    # integer XYZ indices, so the canvas covers a slightly larger area;
    # cropping in pixel space gives a perfect georeferenced rectangle.
    # We round the crop edges to the nearest integer pixel to avoid Pillow
    # silently shifting them; the same rounded fractions are stored back
    # into the metadata so per-vertex UV mapping is exact.
    pix_left_f = (x0_f - x_min) * tile_size
    pix_top_f = (y0_f - y_min) * tile_size
    pix_right_f = (x1_f - x_min) * tile_size
    pix_bottom_f = (y1_f - y_min) * tile_size
    pix_left = int(round(pix_left_f))
    pix_top = int(round(pix_top_f))
    pix_right = int(round(pix_right_f))
    pix_bottom = int(round(pix_bottom_f))
    cropped = canvas.crop((pix_left, pix_top, pix_right, pix_bottom))

    if out_path is None:
        out_dir = os.path.join(cache_dir, "_composites")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        out_path = os.path.join(out_dir, f"basemap_{source.lower()}_z{zoom}_{ts}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cropped.save(out_path, 'PNG')

    # XYZ tiles live in Web Mercator, so the natural way to map from
    # geographic (lat, lon) to pixel space is via tile-XYZ coordinates,
    # not via the WGS84 bbox (which is non-linear in pixels). We expose
    # the **rounded** pixel rectangle as Mercator-tile coordinates so
    # downstream UV code can do an exact inversion.
    mercator_bounds = {
        'x_min': x_min + pix_left / tile_size,   # left edge in tile units
        'x_max': x_min + pix_right / tile_size,  # right edge in tile units
        'y_min': y_min + pix_top / tile_size,    # top edge (north)
        'y_max': y_min + pix_bottom / tile_size, # bottom edge (south)
    }

    metadata = {
        'source': source,
        'source_name': cfg['name'],
        'attribution': cfg['attribution'],
        'zoom': zoom,
        'tiles_total': total,
        'tiles_failed': failed,
        'requested_bounds': dict(bounds),
        'image_bounds': dict(bounds),
        'image_size': cropped.size,
        'projection': 'WEB_MERCATOR',
        'mercator_tile_bounds': mercator_bounds,
    }
    log(
        f"Basemap saved: {out_path} ({cropped.size[0]}×{cropped.size[1]} px, "
        f"{failed}/{total} failed tiles)"
    )
    return out_path, metadata


def _fetch_tile(
    *,
    cfg: dict,
    source: str,
    z: int,
    x: int,
    y: int,
    cache_dir: str,
    api_key: Optional[str],
    ext: str,
) -> Optional[bytes]:
    """Return raw tile bytes, hitting the cache when possible. ``None`` on failure."""
    cache_path = _tile_cache_path(cache_dir, source, z, x, y, ext)
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        try:
            with open(cache_path, 'rb') as fp:
                return fp.read()
        except OSError:
            pass

    url = cfg['url'].format(z=z, x=x, y=y, key=api_key or "")

    request = urllib.request.Request(url, headers={
        'User-Agent': _USER_AGENT,
        'Accept': 'image/jpeg,image/png,image/*;q=0.9,*/*;q=0.5',
    })

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            data = response.read()
    except Exception as exc:  # noqa: BLE001
        log(f"  Tile {z}/{x}/{y} failed: {exc}")
        return None

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            fp.write(data)
    except OSError:
        pass

    return data


# ---------------------------------------------------------------------------
# WMS implementation (single GetMap request)
# ---------------------------------------------------------------------------

def _fetch_wms_basemap(
    bounds: dict,
    wms_url: Optional[str],
    wms_layer: Optional[str],
    out_path: Optional[str],
) -> tuple[str, dict]:
    if not wms_url or not wms_layer:
        raise ValueError("WMS source requires both wms_url and wms_layer")

    try:
        import PIL  # noqa: F401  (sanity check; payload is saved as raw bytes)
    except ImportError as exc:
        raise RuntimeError("Pillow is required for WMS basemap fetching") from exc

    width, height = 2048, 2048
    bbox = f"{bounds['west']},{bounds['south']},{bounds['east']},{bounds['north']}"

    params = {
        'SERVICE': 'WMS',
        'REQUEST': 'GetMap',
        'VERSION': '1.3.0',
        'LAYERS': wms_layer,
        'STYLES': '',
        'CRS': 'EPSG:4326',
        'BBOX': f"{bounds['south']},{bounds['west']},{bounds['north']},{bounds['east']}",
        'WIDTH': width,
        'HEIGHT': height,
        'FORMAT': 'image/png',
        'TRANSPARENT': 'true',
    }
    sep = '&' if '?' in wms_url else '?'
    url = wms_url + sep + urllib.parse.urlencode(params)

    log(f"WMS request: {wms_layer} from {wms_url}")
    request = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
            content_type = response.headers.get('Content-Type', '')
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"WMS GetMap failed: {exc}") from exc

    if 'image' not in content_type.lower():
        snippet = data[:300].decode('utf-8', errors='ignore')
        raise RuntimeError(f"WMS server returned non-image response: {snippet}")

    if out_path is None:
        out_dir = os.path.join(_default_cache_dir(), "_composites")
        os.makedirs(out_dir, exist_ok=True)
        ts = int(time.time())
        out_path = os.path.join(out_dir, f"basemap_wms_{ts}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as fp:
        fp.write(data)

    metadata = {
        'source': 'WMS',
        'source_name': f'WMS · {wms_layer}',
        'attribution': f'WMS layer "{wms_layer}" — {wms_url}',
        'zoom': None,
        'tiles_total': 1,
        'tiles_failed': 0,
        'requested_bounds': dict(bounds),
        'image_bounds': dict(bounds),
        'image_size': (width, height),
        'bbox_query': bbox,
        # WMS GetMap with CRS=EPSG:4326 returns an image whose pixels
        # are linear in (lat, lon); UV mapping is therefore a plain
        # bbox rescale.
        'projection': 'WGS84_LINEAR',
    }
    log(f"WMS image saved: {out_path}")
    return out_path, metadata


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def latlon_to_image_uv(lat: float, lon: float, metadata: dict) -> tuple[float, float]:
    """Convert a WGS84 ``(lat, lon)`` to UV coordinates on the basemap image.

    Knows two image projections:

    * ``WEB_MERCATOR`` — XYZ / slippy-map tiles. Uses the spherical Web
      Mercator definition shared by all major tile providers, so the
      mapping is exact (modulo Pillow's pixel rounding, which we already
      bake into ``mercator_tile_bounds``).
    * ``WGS84_LINEAR`` — single image returned by a WMS ``GetMap`` with
      ``CRS=EPSG:4326``. UV is a plain bbox rescale.

    Returns ``(u, v)`` in ``[0, 1]`` with ``v=1`` at the top of the image
    (Blender's Image Texture convention).
    """
    projection = metadata.get('projection', 'WGS84_LINEAR')

    if projection == 'WEB_MERCATOR':
        zoom = metadata['zoom']
        mb = metadata['mercator_tile_bounds']
        x, y = _lonlat_to_tile(lon, lat, zoom)
        u = (x - mb['x_min']) / max(mb['x_max'] - mb['x_min'], 1e-12)
        v_from_top = (y - mb['y_min']) / max(mb['y_max'] - mb['y_min'], 1e-12)
    else:
        b = metadata['image_bounds']
        u = (lon - b['west']) / max(b['east'] - b['west'], 1e-12)
        v_from_top = (b['north'] - lat) / max(b['north'] - b['south'], 1e-12)

    # Blender's Image Texture node uses v growing upward, so flip.
    return u, 1.0 - v_from_top


def _validate_bounds(bounds: dict) -> None:
    if not bounds:
        raise ValueError("bounds is empty")
    for key in ('north', 'south', 'east', 'west'):
        if key not in bounds:
            raise ValueError(f"bounds missing '{key}'")
    if not (-90.0 <= bounds['south'] < bounds['north'] <= 90.0):
        raise ValueError(f"Invalid latitude range: {bounds['south']}..{bounds['north']}")
    if not (-180.0 <= bounds['west'] < bounds['east'] <= 180.0):
        raise ValueError(f"Invalid longitude range: {bounds['west']}..{bounds['east']}")
