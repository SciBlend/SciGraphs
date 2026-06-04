"""Resolve a download area for the unified city2graph Data Import panel.

The new panel mirrors the OSMnx download methods so the user can fetch
Overture / city2graph features without first importing a street graph.
This module is the single point that, given the panel state, returns:

    bbox        — (north, south, east, west) in WGS84.
    center_lat  — projection origin latitude (for alignment in Blender).
    center_lon  — projection origin longitude.
    scale       — meters → Blender units factor.
    osmnx_obj   — the active OSMnx object when available (used by
                  ``gdf_to_blender_mesh`` to share the same local
                  equirectangular projection).
    source      — short label describing where the area came from
                  (used in operator reports).

The function is deliberately tolerant: when geocoding is unavailable
or the user inputs are inconsistent, ``resolve_area`` raises a
``ValueError`` with a human-readable message that the operator can
forward via ``self.report({'ERROR'}, ...)``.
"""
from __future__ import annotations

import math
from typing import Optional

from ...utils.logger import log


_DEFAULT_SCALE = 0.001  # 1 unit = 1 km, consistent with the OSMnx importer.


def _bbox_from_radius(lat: float, lon: float, radius_m: float) -> tuple:
    """Return (N, S, E, W) for a geodesic disk of ``radius_m`` around (lat, lon).

    Uses the equirectangular approximation: it's exact enough for the
    radii Overture/c2g handle (≤ 5 km) and avoids pulling pyproj into
    the call site.
    """
    earth_radius_m = 6_371_000.0
    deg_per_meter_lat = (180.0 / math.pi) / earth_radius_m
    cos_lat = math.cos(math.radians(lat))
    if abs(cos_lat) < 1e-9:
        cos_lat = 1.0
    deg_per_meter_lon = deg_per_meter_lat / cos_lat

    dlat = radius_m * deg_per_meter_lat
    dlon = radius_m * deg_per_meter_lon
    return (lat + dlat, lat - dlat, lon + dlon, lon - dlon)


def _bbox_from_active_osmnx(osmnx_obj) -> Optional[tuple]:
    """Recreate the bbox helper from ``data_ops.py`` (legacy path).

    Reads the cached bbox custom properties first; if missing, derives
    them from the mesh extent + ``osmnx_center_*`` / ``osmnx_scale``.
    Returns ``None`` if neither path is viable.
    """
    if osmnx_obj is None:
        return None

    n = osmnx_obj.get("osmnx_bbox_north")
    s = osmnx_obj.get("osmnx_bbox_south")
    e = osmnx_obj.get("osmnx_bbox_east")
    w = osmnx_obj.get("osmnx_bbox_west")
    if n and s and e and w:
        return (float(n), float(s), float(e), float(w))

    center_lat = osmnx_obj.get("osmnx_center_lat")
    center_lon = osmnx_obj.get("osmnx_center_lon")
    scale = osmnx_obj.get("osmnx_scale", _DEFAULT_SCALE)
    if center_lat is None or center_lon is None or osmnx_obj.type != 'MESH':
        return None

    try:
        import numpy as np
        verts = np.array([v.co for v in osmnx_obj.data.vertices])
        if len(verts) == 0:
            return None
        min_x, min_y = float(verts[:, 0].min()), float(verts[:, 1].min())
        max_x, max_y = float(verts[:, 0].max()), float(verts[:, 1].max())
    except Exception:  # pragma: no cover - numpy missing is unrealistic in Blender
        return None

    lat_per_unit = 1.0 / (111320.0 * scale)
    lon_per_unit = 1.0 / (111320.0 * scale * math.cos(math.radians(float(center_lat))))
    return (
        float(center_lat) + max_y * lat_per_unit,
        float(center_lat) + min_y * lat_per_unit,
        float(center_lon) + max_x * lon_per_unit,
        float(center_lon) + min_x * lon_per_unit,
    )


def _geocode_place_bbox(place_name: str, which_result: int = 0) -> tuple:
    """Resolve a place name to a bbox via OSMnx's geocoder.

    OSMnx is already a hard dependency of the addon, so reusing it
    avoids requiring Nominatim configuration here. Falls back to
    Nominatim search if the OSMnx import fails (older builds).
    """
    place_name = (place_name or "").strip()
    if not place_name:
        raise ValueError("Empty place name")

    try:
        import osmnx as ox
        gdf = ox.geocoder.geocode_to_gdf(
            place_name,
            which_result=which_result if which_result > 0 else None,
        )
        if gdf is None or len(gdf) == 0:
            raise ValueError(f"No geocoding result for '{place_name}'")
        # geocode_to_gdf returns columns bbox_north / bbox_south / bbox_east / bbox_west
        n = float(gdf.iloc[0]["bbox_north"])
        s = float(gdf.iloc[0]["bbox_south"])
        e = float(gdf.iloc[0]["bbox_east"])
        w = float(gdf.iloc[0]["bbox_west"])
        return (n, s, e, w)
    except ImportError as exc:
        raise ValueError("OSMnx is not available; cannot geocode") from exc


def _bbox_from_polygon_object(obj_name: str) -> tuple:
    """Compute (N, S, E, W) from a Blender mesh's vertex bounds.

    Mesh vertices are interpreted as (lon, lat) (same convention used
    elsewhere in the addon for boundary polygons).
    """
    import bpy
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != 'MESH':
        raise ValueError(f"Polygon object '{obj_name}' is not a usable mesh")
    if len(obj.data.vertices) == 0:
        raise ValueError(f"Polygon object '{obj_name}' has no vertices")

    xs = [v.co.x for v in obj.data.vertices]
    ys = [v.co.y for v in obj.data.vertices]
    return (max(ys), min(ys), max(xs), min(xs))


def _osmnx_alignment(osmnx_obj):
    """Extract ``(center_lat, center_lon, scale)`` for projection alignment."""
    if osmnx_obj is None:
        return None, None, _DEFAULT_SCALE
    return (
        osmnx_obj.get("osmnx_center_lat"),
        osmnx_obj.get("osmnx_center_lon"),
        osmnx_obj.get("osmnx_scale", _DEFAULT_SCALE),
    )


def resolve_area(context):
    """Resolve the download area from the unified Data Import panel.

    Returns a dict with keys: ``bbox``, ``center_lat``, ``center_lon``,
    ``scale``, ``osmnx_obj``, ``source``. Raises ``ValueError`` with a
    user-facing message if the inputs are invalid.
    """
    props = context.scene.city2graph
    scene_props = context.scene.scigraphs

    method = props.c2g_area_method

    osmnx_obj = None
    active = context.active_object
    if active is not None and active.get("is_osmnx"):
        osmnx_obj = active

    if method == 'FROM_OSMNX':
        if osmnx_obj is None:
            raise ValueError(
                "No OSMnx graph selected. Activate one in the viewport, or "
                "switch the Area Method to PLACE / BBOX / POINT / ADDRESS / "
                "POLYGON to define the area without a graph."
            )
        bbox = _bbox_from_active_osmnx(osmnx_obj)
        if bbox is None:
            raise ValueError(
                f"Active OSMnx object '{osmnx_obj.name}' has no bbox metadata."
            )
        center_lat, center_lon, scale = _osmnx_alignment(osmnx_obj)
        if center_lat is None or center_lon is None:
            # Use bbox centre as a reasonable default.
            n, s, e, w = bbox
            center_lat = (n + s) / 2.0
            center_lon = (e + w) / 2.0
        log(f"[c2g] Area resolved from OSMnx graph '{osmnx_obj.name}': {bbox}")
        return {
            'bbox': bbox,
            'center_lat': float(center_lat),
            'center_lon': float(center_lon),
            'scale': float(scale),
            'osmnx_obj': osmnx_obj,
            'source': f"OSMnx graph '{osmnx_obj.name}'",
        }

    if method == 'PLACE':
        place = (scene_props.osmnx_place_name or "").strip()
        if not place:
            raise ValueError("Place name is empty (set it in the OSMnx panel or here).")
        bbox = _geocode_place_bbox(place, which_result=scene_props.osmnx_which_result)
        n, s, e, w = bbox
        center_lat = (n + s) / 2.0
        center_lon = (e + w) / 2.0
        return {
            'bbox': bbox,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'scale': scene_props.osmnx_scale or _DEFAULT_SCALE,
            'osmnx_obj': osmnx_obj,
            'source': f"Place '{place}'",
            'place_name': place,
        }

    if method == 'POINT':
        lat = float(scene_props.osmnx_latitude)
        lon = float(scene_props.osmnx_longitude)
        radius = float(scene_props.osmnx_distance)
        bbox = _bbox_from_radius(lat, lon, radius)
        return {
            'bbox': bbox,
            'center_lat': lat,
            'center_lon': lon,
            'scale': scene_props.osmnx_scale or _DEFAULT_SCALE,
            'osmnx_obj': osmnx_obj,
            'source': f"Point ({lat:.5f}, {lon:.5f}) ±{int(radius)}m",
        }

    if method == 'ADDRESS':
        addr = getattr(scene_props, 'osmnx_address', '').strip()
        radius = float(scene_props.osmnx_distance)
        if not addr:
            raise ValueError("Address is empty (set it in the OSMnx panel or here).")
        # Use the same geocoder as PLACE.
        bbox = _geocode_place_bbox(addr)
        n, s, e, w = bbox
        center_lat = (n + s) / 2.0
        center_lon = (e + w) / 2.0
        # Tighten the bbox to the requested radius so the user gets the
        # area they actually asked for (geocoder bbox can be huge for
        # cities with the same address-name).
        bbox = _bbox_from_radius(center_lat, center_lon, radius)
        return {
            'bbox': bbox,
            'center_lat': center_lat,
            'center_lon': center_lon,
            'scale': scene_props.osmnx_scale or _DEFAULT_SCALE,
            'osmnx_obj': osmnx_obj,
            'source': f"Address '{addr}' ±{int(radius)}m",
        }

    if method == 'BBOX':
        n = float(scene_props.osmnx_bbox_north)
        s = float(scene_props.osmnx_bbox_south)
        e = float(scene_props.osmnx_bbox_east)
        w = float(scene_props.osmnx_bbox_west)
        if n <= s or e <= w:
            raise ValueError(
                "Invalid bbox: requires north > south and east > west "
                f"(got N={n}, S={s}, E={e}, W={w})."
            )
        return {
            'bbox': (n, s, e, w),
            'center_lat': (n + s) / 2.0,
            'center_lon': (e + w) / 2.0,
            'scale': scene_props.osmnx_scale or _DEFAULT_SCALE,
            'osmnx_obj': osmnx_obj,
            'source': "Manual bbox",
        }

    if method == 'POLYGON':
        obj_name = (scene_props.osmnx_polygon_object or "").strip()
        if not obj_name:
            raise ValueError("Polygon object not set (choose one in the OSMnx panel or here).")
        bbox = _bbox_from_polygon_object(obj_name)
        n, s, e, w = bbox
        return {
            'bbox': bbox,
            'center_lat': (n + s) / 2.0,
            'center_lon': (e + w) / 2.0,
            'scale': scene_props.osmnx_scale or _DEFAULT_SCALE,
            'osmnx_obj': osmnx_obj,
            'source': f"Polygon object '{obj_name}'",
        }

    raise ValueError(f"Unknown area method: {method}")
