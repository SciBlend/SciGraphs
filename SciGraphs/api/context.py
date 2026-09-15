"""Terrain, buildings, and imagery context under a notebook graph.

Display geometry only, not analysis data. Projects through the same anchor as
the graph (`graphs.anchor()`); mismatched anchors are what `check_alignment()` catches.
"""

import math
import time

import bpy

from . import graphs


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: OSM Simple 3D Buildings convention when only `building:levels` is tagged.
METERS_PER_LEVEL = 3.0

#: Fallback when almost nothing is tagged; `buildings()` prefers median ("auto").
FALLBACK_HEIGHT_M = 9.0

#: Vertical gap (real meters) between context top and graph plane; avoids z-fight.
CLEARANCE_M = 1.0

#: Push buildings into the ground so sloped terrain does not show daylight under corners.
EMBED_M = 0.5

#: Default collection for everything this module creates.
COLLECTION = "SGNB_Context"

EARTH_RADIUS_M = 6371000.0


# --------------------------------------------------------------------------
# The georeference
# --------------------------------------------------------------------------

def _link(obj, coll):
    return graphs._link(obj, coll)


def frame(ref):
    """(center_lat, center_lon, scale) from an anchor or geo object, raising if missing."""
    if isinstance(ref, (tuple, list)) and len(ref) == 3:
        return float(ref[0]), float(ref[1]), float(ref[2])
    if ref is None:
        raise ValueError(
            "no anchor: pass the same object you passed to graphs.from_gdf()")

    lat = ref.get("c2g_center_lat")
    lon = ref.get("c2g_center_lon")
    scale = ref.get("c2g_scale")
    if lat is None or lon is None:
        lat = ref.get("osmnx_center_lat")
        lon = ref.get("osmnx_center_lon")
        scale = ref.get("osmnx_scale")
    if lat is None or lon is None:
        raise ValueError(
            f"{getattr(ref, 'name', ref)!r} carries no georeference "
            "(c2g_center_lat/lon/scale or the osmnx_* equivalents)")
    return float(lat), float(lon), float(scale if scale is not None else 0.001)


def project(lat, lon, ref):
    """(lat, lon) -> Blender (x, y) through `ref`."""
    center_lat, center_lon, scale = frame(ref)
    cos_lat = math.cos(math.radians(center_lat))
    y_m = (lat - center_lat) * (math.pi / 180.0) * EARTH_RADIUS_M
    x_m = (lon - center_lon) * (math.pi / 180.0) * EARTH_RADIUS_M * cos_lat
    return x_m * scale, y_m * scale


def bounds_around(center, radius_m):
    """N/S/E/W dict `radius_m` around (lat, lon), using the same projection as `project`."""
    lat, lon = float(center[0]), float(center[1])
    deg_per_m_lat = 180.0 / (math.pi * EARTH_RADIUS_M)
    deg_per_m_lon = deg_per_m_lat / max(math.cos(math.radians(lat)), 1e-6)
    dlat = radius_m * deg_per_m_lat
    dlon = radius_m * deg_per_m_lon
    return {"north": lat + dlat, "south": lat - dlat,
            "east": lon + dlon, "west": lon - dlon}


# --------------------------------------------------------------------------
# Building heights
# --------------------------------------------------------------------------

def _number(value):
    """Leading positive number in an OSM tag value, or None."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            number = float(value)
            return number if number > 0 and math.isfinite(number) else None
    except (TypeError, ValueError):
        return None

    text = str(value).strip().replace(",", ".")
    if not text or text.lower() in {"nan", "none", "yes", "no"}:
        return None
    digits = ""
    for char in text:
        if char.isdigit() or (char == "." and "." not in digits):
            digits += char
        elif digits:
            break
    if not digits:
        return None
    try:
        number = float(digits)
    except ValueError:
        return None
    return number if number > 0 and math.isfinite(number) else None


def _tagged_height(row, meters_per_level):
    """Height in meters from OSM tags, preferring explicit height over levels, or None."""
    for key in ("height", "building:height", "est_height"):
        meters = _number(row.get(key) if hasattr(row, "get") else None)
        if meters:
            return meters, key
    levels = _number(row.get("building:levels") if hasattr(row, "get") else None)
    if levels:
        return levels * meters_per_level, "building:levels"
    return None, None


def height_report(gdf, meters_per_level=METERS_PER_LEVEL):
    """Tagged-height counts, median, and untagged fraction for a footprint GDF."""
    sources = {}
    heights = []
    for _index, row in gdf.iterrows():
        meters, key = _tagged_height(row, meters_per_level)
        if meters:
            heights.append(meters)
            sources[key] = sources.get(key, 0) + 1
    total = len(gdf)
    tagged = len(heights)
    heights.sort()
    median = heights[tagged // 2] if tagged else None
    return {
        "features": total,
        "tagged": tagged,
        "untagged": total - tagged,
        "tagged_fraction": (tagged / total) if total else 0.0,
        "by_tag": sources,
        "median_tagged_height_m": median,
        "meters_per_level": meters_per_level,
    }


def _resolve_default(gdf, default_height, meters_per_level, minimum_tagged=20):
    """Height for untagged footprints, where `"auto"` takes the median of the tagged."""
    if default_height is None:
        default_height = "auto"
    if default_height != "auto":
        return float(default_height), "given"
    report = height_report(gdf, meters_per_level)
    if report["tagged"] >= minimum_tagged and report["median_tagged_height_m"]:
        return float(report["median_tagged_height_m"]), "median of tagged"
    return FALLBACK_HEIGHT_M, "fallback constant"


# --------------------------------------------------------------------------
# Buildings
# --------------------------------------------------------------------------

def _footprint_gdf(source, radius_m=None, tags=None):
    """Polygon GDF from a GDF, place name, or (lat, lon) + radius_m via OSMnx."""
    import geopandas as gpd

    if isinstance(source, gpd.GeoDataFrame):
        gdf = source
    else:
        import osmnx as ox
        tags = tags or {"building": True}
        if isinstance(source, str):
            gdf = ox.features_from_place(source, tags=tags)
        else:
            if radius_m is None:
                raise ValueError("radius_m is required when source is a (lat, lon) point")
            gdf = ox.features_from_point(
                (float(source[0]), float(source[1])), tags=tags, dist=float(radius_m))

    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if len(gdf) == 0:
        return gdf
    if gdf.crs is not None and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def _ring_xy(ring, ref, tolerance):
    """Projected, de-duplicated, counter-clockwise exterior ring."""
    coords = list(ring.coords)
    if len(coords) < 4:
        return None
    points = []
    for lon, lat in coords[:-1]:
        x, y = project(lat, lon, ref)
        if points and abs(x - points[-1][0]) < tolerance and abs(y - points[-1][1]) < tolerance:
            continue
        points.append((x, y))
    while (len(points) > 1
           and abs(points[0][0] - points[-1][0]) < tolerance
           and abs(points[0][1] - points[-1][1]) < tolerance):
        points.pop()
    if len(points) < 3:
        return None

    # Force CCW so cap normals are +Z and sides face out (OSM winding is unreliable).
    area = 0.0
    for i, (x0, y0) in enumerate(points):
        x1, y1 = points[(i + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    if area < 0:
        points.reverse()
    return points


def buildings(source, ref, radius_m=None, default_height="auto",
              meters_per_level=METERS_PER_LEVEL, vertical_scale=1.0,
              terrain=None, base_m=0.0, jitter=0.0, tags=None,
              name="Context_Buildings", coll=COLLECTION, verbose=True):
    """Extrude footprints against `ref` into one mesh, optionally sampling terrain for Z."""
    from shapely.geometry import MultiPolygon

    gdf = _footprint_gdf(source, radius_m=radius_m, tags=tags)
    if len(gdf) == 0:
        if verbose:
            print("  no building footprints")
        return []

    _center_lat, _center_lon, scale = frame(ref)
    default_m, provenance = _resolve_default(gdf, default_height, meters_per_level)
    tolerance = 1e-9

    vertices = []
    faces = []
    face_heights = []
    used_tag = 0
    base_z = base_m * scale

    for index, row in gdf.iterrows():
        geometry = row.geometry
        meters, key = _tagged_height(row, meters_per_level)
        if meters is None:
            meters = default_m
        else:
            used_tag += 1
        if jitter:
            # Deterministic salt; changing it rewrites every jittered height.
            wobble = ((hash((str(index), "sgnb")) % 2000) / 1000.0 - 1.0) * jitter
            meters *= 1.0 + wobble

        parts = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
        for part in parts:
            if part.is_empty:
                continue
            ring = _ring_xy(part.exterior, ref, tolerance)
            if ring is None:
                continue

            if terrain is not None:
                point = part.representative_point()
                ground_z = surface_z(terrain, point.y, point.x)
            else:
                ground_z = base_z
            z0 = ground_z - EMBED_M * scale
            z1 = ground_z + meters * scale * vertical_scale

            start = len(vertices)
            count = len(ring)
            vertices.extend([(x, y, z0) for x, y in ring])
            vertices.extend([(x, y, z1) for x, y in ring])
            for i in range(count):
                j = (i + 1) % count
                faces.append((start + i, start + j,
                              start + count + j, start + count + i))
                face_heights.append(meters)
            faces.append(tuple(start + count + i for i in range(count)))
            face_heights.append(meters)

    if not faces:
        if verbose:
            print("  every footprint was degenerate; nothing built")
        return []

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], [f if isinstance(f, list) else list(f) for f in faces])
    mesh.validate(verbose=False)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    _link(obj, coll)

    # Only write face attr if validate() did not drop faces (stale counts misalign).
    if len(mesh.polygons) == len(face_heights):
        attribute = mesh.attributes.new(name="height", type='FLOAT', domain='FACE')
        attribute.data.foreach_set("value", face_heights)

    center_lat, center_lon, _scale = frame(ref)
    obj["is_scigraphs_context"] = True
    obj["scigraphs_context_kind"] = "buildings"
    obj["c2g_center_lat"] = center_lat
    obj["c2g_center_lon"] = center_lon
    obj["c2g_scale"] = scale
    obj["building_count"] = int(len(gdf))
    obj["height_default_m"] = float(default_m)
    obj["height_default_source"] = provenance
    obj["height_tagged_fraction"] = float(used_tag / len(gdf))
    obj["height_vertical_scale"] = float(vertical_scale)

    if verbose:
        print(f"  {len(gdf)} footprints -> {len(mesh.polygons)} faces, "
              f"{len(mesh.vertices)} vertices")
        print(f"  heights: {used_tag} from OSM tags "
              f"({used_tag / len(gdf) * 100:.1f}%), the rest at "
              f"{default_m:.1f} m ({provenance})")
    return [obj]


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------

def _store_dem(obj, grid, bounds, scale, vertical_scale, min_elevation):
    """Persist elevation grid on the object for later sampling (survives .blend)."""
    obj["_scigraphs_dem"] = [float(v) for v in grid.ravel()]
    obj["_scigraphs_dem_shape"] = [int(grid.shape[0]), int(grid.shape[1])]
    obj["_scigraphs_dem_north"] = float(bounds["north"])
    obj["_scigraphs_dem_south"] = float(bounds["south"])
    obj["_scigraphs_dem_east"] = float(bounds["east"])
    obj["_scigraphs_dem_west"] = float(bounds["west"])
    obj["_scigraphs_dem_min"] = float(min_elevation)
    obj["_scigraphs_dem_scale"] = float(scale)
    obj["_scigraphs_dem_vertical_scale"] = float(vertical_scale)


def sample_elevation(terrain_obj, lat, lon):
    """Elevation (m) at (lat, lon), bilinear; None if no stored grid."""
    import numpy as np

    grid = terrain_obj.get("_scigraphs_dem") if terrain_obj is not None else None
    if grid is None:
        return None
    shape = tuple(int(v) for v in terrain_obj["_scigraphs_dem_shape"])
    grid = np.asarray(list(grid), dtype=float).reshape(shape)
    bounds = {"north": terrain_obj["_scigraphs_dem_north"],
              "south": terrain_obj["_scigraphs_dem_south"],
              "east": terrain_obj["_scigraphs_dem_east"],
              "west": terrain_obj["_scigraphs_dem_west"]}

    from SciGraphs.core.geo.terrain import _sample_elevation_from_grid

    return _sample_elevation_from_grid(lat, lon, grid, bounds)


def surface_z(terrain_obj, lat, lon):
    """Object-local surface Z at (lat, lon); ignores location.z so settle() stays glued."""
    meters = sample_elevation(terrain_obj, lat, lon)
    if meters is None:
        return 0.0
    scale = float(terrain_obj.get("_scigraphs_dem_scale", 0.001))
    vertical = float(terrain_obj.get("_scigraphs_dem_vertical_scale", 1.0))
    minimum = float(terrain_obj.get("_scigraphs_dem_min", 0.0))
    return (meters - minimum) * scale * vertical


def _build(grid, bounds, ref, vertical_scale, name, coll, kind, provenance):
    """Turn an elevation grid into a Blender surface via SciGraphs' terrain builder."""
    from SciGraphs.core.geo import terrain as sg_terrain

    center_lat, center_lon, scale = frame(ref)
    dem_data = {"elevation": grid, "bounds": bounds, "crs": "EPSG:4326",
                "resolution": None, "nodata": None, "transform": None,
                "source": provenance}

    # create_terrain_mesh links into bpy.context.collection, which clear_scene()
    # can leave None, so restore the view layer root first.
    if bpy.context.collection is None:
        view_layer = bpy.context.view_layer
        view_layer.active_layer_collection = view_layer.layer_collection

    obj = sg_terrain.create_terrain_mesh(
        dem_data, scale=scale, vertical_scale=vertical_scale, vertical_offset=0.0,
        subsample=1, name=name, center_lat=center_lat, center_lon=center_lon)
    if obj is None:
        return None

    # Relink from context.collection into the caller's collection.
    for collection in list(obj.users_collection):
        collection.objects.unlink(obj)
    _link(obj, coll)

    import numpy as np

    minimum = float(np.nanmin(grid)) if grid.size else 0.0
    _store_dem(obj, grid, bounds, scale, vertical_scale, minimum)
    obj["is_scigraphs_context"] = True
    obj["scigraphs_context_kind"] = kind
    obj["scigraphs_elevation_source"] = provenance
    obj["scigraphs_real_elevation"] = bool(kind == "terrain")
    obj["c2g_center_lat"] = center_lat
    obj["c2g_center_lon"] = center_lon
    obj["c2g_scale"] = scale
    return obj


def ground(ref, radius_m, name="Context_Ground", coll=COLLECTION, verbose=True):
    """Flat plane via the same path as real terrain (zeros grid); real_elevation=False."""
    import numpy as np

    center_lat, center_lon, _scale = frame(ref)
    bounds = bounds_around((center_lat, center_lon), radius_m)
    grid = np.zeros((2, 2), dtype=float)
    obj = _build(grid, bounds, ref, 1.0, name, coll, "ground", "flat")
    if obj is not None and verbose:
        print(f"  flat ground, {2 * radius_m:.0f} x {2 * radius_m:.0f} m "
              "(no elevation data)")
    return obj


class _capture:
    """Temporarily replace a module's `log` to catch silent DEM batch failures."""

    def __init__(self, module):
        self.module = module
        self.lines = []

    def __enter__(self):
        self._original = getattr(self.module, "log", None)
        self.module.log = lambda *args: self.lines.append(" ".join(str(a) for a in args))
        return self

    def __exit__(self, *exc):
        if self._original is not None:
            self.module.log = self._original
        return False

    def failed_batches(self):
        for line in self.lines:
            if "batch(es) failed" in line:
                for word in line.split():
                    if word.isdigit():
                        return int(word)
        return 0


def probe_elevation_api(api="open-elevation", timeout=10.0):
    """Cheap up/down check returning (ok, seconds, detail), which is not a speed estimate."""
    import requests

    started = time.time()
    try:
        if api == "opentopodata":
            from ..utils.online import online_ok
            if not online_ok():
                raise PermissionError(
                    "Blender is set to work offline; enable "
                    "Preferences > System > Network > Allow Online "
                    "Access to use this")
            response = requests.get("https://api.opentopodata.org/v1/srtm30m",
                                    params={"locations": "0,0"}, timeout=timeout)
        else:
            from ..utils.online import online_ok
            if not online_ok():
                raise PermissionError(
                    "Blender is set to work offline; enable "
                    "Preferences > System > Network > Allow Online "
                    "Access to use this")
            response = requests.post("https://api.open-elevation.com/api/v1/lookup",
                                     json={"locations": [{"latitude": 0, "longitude": 0}]},
                                     timeout=timeout)
        elapsed = time.time() - started
        return response.status_code == 200, elapsed, f"HTTP {response.status_code}"
    except Exception as exc:  # noqa: BLE001 - a probe reports, it does not raise
        return False, time.time() - started, f"{type(exc).__name__}: {exc}"


def terrain(center, radius_m, ref, source="flat", api="open-elevation",
            resolution=24, vertical_scale=1.0, workers=5, probe_timeout=10.0,
            imagery=None, imagery_zoom=17, imagery_brightness=1.0,
            name="Context_Terrain", coll=COLLECTION, verbose=True):
    """Ground surface: `flat` (default), `dem`, or `auto` (dem with flat fallback).

    Rejected DEM batches are interpolated, with the count stored as
    `scigraphs_dem_failed_batches`. Optional `imagery=` drapes last.
    """
    import numpy as np

    if source not in {"flat", "dem", "auto"}:
        raise ValueError("source must be 'flat', 'dem' or 'auto'")
    drape = dict(source=imagery, zoom=imagery_zoom,
                 brightness=imagery_brightness, verbose=verbose)
    if source == "flat":
        return _maybe_drape(
            ground(ref, radius_m, name=name.replace("Terrain", "Ground"),
                   coll=coll, verbose=verbose), **drape)

    bounds = bounds_around(center, radius_m)

    ok, probe_seconds, detail = probe_elevation_api(api, timeout=probe_timeout)
    if verbose:
        print(f"  {api} probe: {'up' if ok else 'unreachable'} "
              f"({probe_seconds:.2f} s, {detail})")
    if not ok:
        if source == "dem":
            print(f"  no terrain: {api} is unreachable ({detail}). "
                  "Pass source='auto' to accept a flat plane instead.")
            return None
        print(f"  FALLING BACK TO A FLAT PLANE: {api} is unreachable ({detail}). "
              "This is NOT elevation data.")
        return _maybe_drape(
            ground(ref, radius_m, name=name.replace("Terrain", "Ground"),
                   coll=coll, verbose=verbose), **drape)

    from SciGraphs.core.geo import terrain as sg_terrain

    started = time.time()
    rejected = 0
    try:
        with _capture(sg_terrain) as captured:
            dem_data = sg_terrain.fetch_dem_from_api(
                bounds, resolution=int(resolution), api=api, max_workers=int(workers))
        rejected = captured.failed_batches()
    except Exception as exc:  # noqa: BLE001
        dem_data, detail = None, f"{type(exc).__name__}: {exc}"
    elapsed = time.time() - started

    if dem_data is None:
        if source == "dem":
            print(f"  no terrain: {api} returned nothing after {elapsed:.1f} s ({detail})")
            return None
        print(f"  FALLING BACK TO A FLAT PLANE: {api} returned nothing after "
              f"{elapsed:.1f} s. This is NOT elevation data.")
        return _maybe_drape(
            ground(ref, radius_m, name=name.replace("Terrain", "Ground"),
                   coll=coll, verbose=verbose), **drape)

    grid = np.asarray(dem_data["elevation"], dtype=float)
    obj = _build(grid, bounds, ref, vertical_scale, name, coll, "terrain", api)
    if obj is None:
        return None
    obj["scigraphs_dem_fetch_seconds"] = float(elapsed)
    obj["scigraphs_dem_resolution"] = int(resolution)
    obj["scigraphs_dem_failed_batches"] = int(rejected)
    if verbose:
        relief = float(np.nanmax(grid) - np.nanmin(grid))
        print(f"  {api}: {resolution}x{resolution} points in {elapsed:.1f} s, "
              f"{np.nanmin(grid):.0f}-{np.nanmax(grid):.0f} m "
              f"({relief:.0f} m of relief -> {relief * frame(ref)[2] * vertical_scale:.4f} "
              "Blender units)")
    if rejected:
        print(f"  WARNING: {rejected} batch(es) were rejected by {api} and have been "
              "interpolated, that part of the surface is invented. Lower `resolution` "
              "or try again.")
    return _maybe_drape(obj, **drape)


def _maybe_drape(surface, source=None, zoom=17, brightness=1.0, verbose=True):
    """Drape when `source` is set; pass through None / no source."""
    if surface is None or not source:
        return surface
    imagery(surface, source=source, zoom=zoom, brightness=brightness,
            verbose=verbose)
    return surface


def is_real_elevation(obj):
    """True only for a surface built from fetched elevation data."""
    return bool(obj is not None and obj.get("scigraphs_real_elevation", False))


# --------------------------------------------------------------------------
# Z: where the context sits relative to the graph
# --------------------------------------------------------------------------

def settle(objects, plane_z=0.0, clearance_m=CLEARANCE_M, scale=0.001,
           reference="terrain", verbose=True):
    """Translate context in Z so the reference top clears the graph plane by clearance_m.

    One shared offset. `reference="terrain"` measures the ground, so buildings
    may cross the plane; `"all"` measures the tallest roof, so outliers pull
    everything down.
    """
    objects = [o for o in objects if o is not None]
    if not objects:
        return 0.0

    measured = objects
    if reference == "terrain":
        surfaces = [o for o in objects
                    if o.get("scigraphs_context_kind") in {"terrain", "ground"}]
        measured = surfaces or objects
    elif reference != "all":
        raise ValueError("reference must be 'terrain' or 'all'")

    tops = []
    for obj in measured:
        box = bbox(obj)
        if box is not None:
            tops.append(box["max"][2])
    if not tops:
        return 0.0

    delta = (plane_z - clearance_m * scale) - max(tops)
    for obj in objects:
        obj.location.z += delta

    # Force depsgraph update so matrix_world / bbox() see the new location.
    bpy.context.view_layer.update()

    if verbose:
        what = "ground" if reference == "terrain" else "tallest roof"
        print(f"  context moved {delta:+.5f} Blender units ({delta / scale:+.1f} m) "
              f"so the {what} clears z={plane_z:g} by {clearance_m:g} m")
    return delta


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------

# Matte Principled palette: ground darkest, buildings lighter; graph stays brightest.
_PALETTE = {
    "terrain":   {"color": (0.075, 0.082, 0.066, 1.0), "roughness": 0.96},
    "ground":    {"color": (0.062, 0.066, 0.062, 1.0), "roughness": 0.96},
    "buildings": {"color": (0.170, 0.163, 0.150, 1.0), "roughness": 0.75},
}


def _set(node, socket, value):
    """Set a Principled input if present in this Blender version."""
    if socket in node.inputs:
        node.inputs[socket].default_value = value
        return True
    return False


def material(kind, name=None):
    """Matte Principled material for one context kind, cached by name."""
    spec = _PALETTE.get(kind, _PALETTE["ground"])
    name = name or f"SGNB_Context_{kind}"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    node = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if node is None:
        node = mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        output = next((n for n in mat.node_tree.nodes if n.type == 'OUTPUT_MATERIAL'), None)
        if output is not None:
            mat.node_tree.links.new(node.outputs[0], output.inputs['Surface'])

    _set(node, "Base Color", spec["color"])
    _set(node, "Roughness", spec["roughness"])
    _set(node, "Metallic", 0.0)
    # Low specular so roofs do not rim-light under a sun.
    _set(node, "Specular IOR Level", 0.12)
    _set(node, "IOR", 1.35)
    _set(node, "Coat Weight", 0.0)
    _set(node, "Sheen Weight", 0.0)

    mat.diffuse_color = spec["color"]
    mat.roughness = spec["roughness"]
    mat.metallic = 0.0
    return mat


# Hypsometric ramp (linear Base Color). Top capped so terrain stays dimmer than the graph.
HYPSOMETRIC = (
    (0.00, (0.055, 0.085, 0.062)),
    (0.25, (0.105, 0.125, 0.070)),
    (0.50, (0.180, 0.155, 0.085)),
    (0.75, (0.255, 0.185, 0.110)),
    (1.00, (0.330, 0.275, 0.205)),
)

#: Marker so style_context() does not wipe a hypsometric tint.
HYPSOMETRIC_KEY = "scigraphs_hypsometric"


def hypsometric(surface, stops=HYPSOMETRIC, roughness=0.94, floor=None,
                ceiling=None, verbose=True):
    """Color surface by object-local Z. floor/ceiling override mesh extremes."""
    if surface is None or surface.type != 'MESH' or not len(surface.data.vertices):
        return None

    zs = [v.co.z for v in surface.data.vertices]
    low = float(floor if floor is not None else min(zs))
    high = float(ceiling if ceiling is not None else max(zs))
    if high - low < 1e-9:
        high = low + 1e-9

    name = f"SGNB_Hypsometric_{surface.name}"
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new('ShaderNodeOutputMaterial')
    output.location = (520, 0)
    bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (260, 0)
    _set(bsdf, "Roughness", float(roughness))
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Specular IOR Level", 0.04)
    _set(bsdf, "Coat Weight", 0.0)
    _set(bsdf, "Sheen Weight", 0.0)

    coords = tree.nodes.new('ShaderNodeTexCoord')
    coords.location = (-620, 0)
    separate = tree.nodes.new('ShaderNodeSeparateXYZ')
    separate.location = (-440, 0)
    tree.links.new(coords.outputs['Object'], separate.inputs['Vector'])

    normalize = tree.nodes.new('ShaderNodeMapRange')
    normalize.location = (-260, 0)
    normalize.clamp = True
    normalize.inputs['From Min'].default_value = low
    normalize.inputs['From Max'].default_value = high
    normalize.inputs['To Min'].default_value = 0.0
    normalize.inputs['To Max'].default_value = 1.0
    tree.links.new(separate.outputs['Z'], normalize.inputs['Value'])

    ramp = tree.nodes.new('ShaderNodeValToRGB')
    ramp.location = (-60, 0)
    elements = ramp.color_ramp.elements
    while len(elements) > 1:
        elements.remove(elements[-1])
    for index, (position, rgb) in enumerate(stops):
        element = elements[0] if index == 0 else elements.new(float(position))
        element.position = float(position)
        element.color = (*rgb, 1.0)
    tree.links.new(normalize.outputs['Result'], ramp.inputs['Fac'])
    tree.links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    surface.data.materials.clear()
    surface.data.materials.append(mat)
    surface[HYPSOMETRIC_KEY] = True
    surface["scigraphs_hypsometric_floor"] = low
    surface["scigraphs_hypsometric_ceiling"] = high

    if verbose:
        scale = float(surface.get("dem_scale") or surface.get("c2g_scale") or 0.001)
        vertical = float(surface.get("dem_vertical_scale")
                         or surface.get("_scigraphs_dem_vertical_scale") or 1.0)
        meters = (high - low) / max(scale * vertical, 1e-12)
        print(f"  elevation tint         {low:.4f} .. {high:.4f} BU local Z "
              f"({meters:.0f} m of ground at {vertical:g}x) over "
              f"{len(stops)} stops")
    return mat


def style_context(objects, kind=None, shade_smooth=False, keep_imagery=True):
    """Assign palette materials, skipping draped surfaces unless keep_imagery=False."""
    if objects is None:
        return []
    if not isinstance(objects, (list, tuple, set)):
        objects = [objects]

    styled = []
    for obj in objects:
        if obj is None or obj.type != 'MESH':
            continue
        if keep_imagery and (obj.get("scigraphs_real_imagery")
                             or obj.get(HYPSOMETRIC_KEY)):
            continue
        this = kind or obj.get("scigraphs_context_kind") or "ground"
        mat = material(this)
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if shade_smooth:
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
        styled.append(obj)
    return styled


# --------------------------------------------------------------------------
# Imagery
# --------------------------------------------------------------------------

#: Keyless TILE_SOURCES in preferred notebook order. WMS needs wms_url + wms_layer.
KEYLESS_SOURCES = (
    "ESRI_IMAGERY",       # satellite / aerial
    "ESRI_STREET",
    "ESRI_TOPO",
    "ESRI_DARK_GRAY",
    "ESRI_LIGHT_GRAY",
    "ESRI_OCEAN",
    "ESRI_HILLSHADE",
    "ESRI_TERRAIN",
    "OSM",
    "OPENTOPOMAP",
    "CARTO_VOYAGER",
    "CARTO_POSITRON",
    "CARTO_DARK_MATTER",
)

#: Refuse fetches above this tile count (politeness limit).
MAX_TILES = 256

#: Rough bytes/tile for pre-flight estimates (order of magnitude only).
_BYTES_PER_TILE = {"jpg": 26000, "png": 30000}

#: Appended to User-Agent during fetch (OSM tile policy requires identifying UA).
_USER_AGENT_SUFFIX = "scigraphs"


def _imagery_module():
    from scigraphs_core.geo import imagery as sg_imagery

    return sg_imagery


def imagery_sources(verbose=True):
    """Keyless sources: {key: name, provider, attribution, max_zoom}."""
    sg_imagery = _imagery_module()
    catalog = {}
    for key in KEYLESS_SOURCES:
        cfg = sg_imagery.TILE_SOURCES.get(key)
        if cfg is None or cfg.get("needs_key"):
            continue
        catalog[key] = {"name": cfg["name"], "provider": cfg["provider"],
                        "attribution": cfg["attribution"],
                        "max_zoom": cfg["max_zoom"]}
    if verbose:
        for key, info in catalog.items():
            print(f"  {key:<18} {info['name']:<34} max zoom {info['max_zoom']}")
    return catalog


def imagery_cache_dir():
    """($TMPDIR/scigraphs_basemaps, file_count, bytes), surviving a restart, not a reboot."""
    import os

    root = _imagery_module()._default_cache_dir()
    files = 0
    total = 0
    for directory, _sub, names in os.walk(root):
        for name in names:
            try:
                total += os.path.getsize(os.path.join(directory, name))
                files += 1
            except OSError:
                pass
    return root, files, total


def clear_imagery_cache():
    """Delete every cached tile and return the number of files removed."""
    return _imagery_module().clear_cache()


def _tile_range(bounds, zoom):
    """XYZ tile rectangle covering bounds as (x0, x1, y0, y1)."""
    sg_imagery = _imagery_module()
    x0, y0 = sg_imagery._lonlat_to_tile(bounds["west"], bounds["north"], zoom)
    x1, y1 = sg_imagery._lonlat_to_tile(bounds["east"], bounds["south"], zoom)
    return (int(math.floor(x0)), int(math.floor(x1)),
            int(math.floor(y0)), int(math.floor(y1)))


def imagery_estimate(bounds, source="ESRI_IMAGERY", zoom=17, verbose=True):
    """Pre-flight tile cost, clamped to the source max_zoom and flagged over MAX_TILES."""
    import os

    sg_imagery = _imagery_module()
    key = sg_imagery.resolve_source(source)
    cfg = sg_imagery.TILE_SOURCES.get(key)
    if cfg is None:
        raise ValueError(
            f"unknown imagery source {source!r}; keyless sources are "
            + ", ".join(KEYLESS_SOURCES))

    requested = int(zoom)
    zoom = max(1, min(requested, cfg["max_zoom"]))
    x0, x1, y0, y1 = _tile_range(bounds, zoom)
    cols, rows = x1 - x0 + 1, y1 - y0 + 1
    tiles = cols * rows

    cache_root = sg_imagery._default_cache_dir()
    cached = 0
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            path = sg_imagery._tile_cache_path(
                cache_root, key, zoom, tx, ty, cfg["extension"])
            if os.path.exists(path) and os.path.getsize(path) > 0:
                cached += 1

    per_tile = _BYTES_PER_TILE.get(cfg["extension"], 20000)
    to_fetch = tiles - cached
    result = {
        "source": key, "source_name": cfg["name"], "provider": cfg["provider"],
        "attribution": cfg["attribution"], "zoom": zoom,
        "zoom_requested": requested, "zoom_clamped": zoom != requested,
        "tiles": tiles, "grid": (cols, rows), "cached": cached,
        "to_fetch": to_fetch, "bytes_estimate": to_fetch * per_tile,
        "image_size": (cols * cfg["tile_size"], rows * cfg["tile_size"]),
        "over_limit": tiles > MAX_TILES,
    }
    if verbose:
        if result["zoom_clamped"]:
            print(f"  zoom {requested} clamped to {zoom}, {cfg['name']} "
                  f"serves no deeper")
        print(f"  {cfg['name']}: {cols}x{rows} = {tiles} tiles at zoom {zoom}, "
              f"{result['image_size'][0]}x{result['image_size'][1]} px")
        print(f"  {cached} already cached, {to_fetch} to fetch "
              f"(~{result['bytes_estimate'] / 1e6:.1f} MB)")
    return result


def _project_uv(terrain_obj, metadata):
    """Per-vertex UVs from the geographic projection, falling back without the UI operator."""
    import contextlib
    import io as _io

    try:
        from SciGraphs.ui.operators.osmnx.elevation_operators import (
            _project_uv_geographic,
        )
    except Exception as exc:  # noqa: BLE001
        return _project_uv_fallback(terrain_obj, metadata, why=str(exc))

    # Suppress UI diagnostic prints from the projection helper.
    noise = _io.StringIO()
    with contextlib.redirect_stdout(noise):
        return _project_uv_geographic(terrain_obj, metadata)


def _project_uv_fallback(terrain_obj, metadata, why=""):
    """XY -> lat/lon -> UV without the add-on UI module."""
    sg_imagery = _imagery_module()

    mesh = terrain_obj.data
    if mesh is None or len(mesh.vertices) == 0:
        return False, "terrain has no geometry"

    center_lat = terrain_obj.get("dem_center_lat")
    center_lon = terrain_obj.get("dem_center_lon")
    scale = terrain_obj.get("dem_scale")
    if center_lat is None or center_lon is None or not scale:
        return False, "terrain carries no dem_center_lat/lon/scale"

    meters_per_deg_lat = (math.pi / 180.0) * EARTH_RADIUS_M
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(float(center_lat)))
    if abs(meters_per_deg_lon) < 1e-9:
        meters_per_deg_lon = meters_per_deg_lat

    if not mesh.uv_layers:
        mesh.uv_layers.new(name="BasemapUV")
    uv_layer = mesh.uv_layers.active or mesh.uv_layers[0]

    matrix_world = terrain_obj.matrix_world
    vertex_uvs = []
    for vertex in mesh.vertices:
        co = matrix_world @ vertex.co
        x_m = co.x / float(scale)
        y_m = co.y / float(scale)
        lat = float(center_lat) + y_m / meters_per_deg_lat
        lon = float(center_lon) + x_m / meters_per_deg_lon
        vertex_uvs.append(sg_imagery.latlon_to_image_uv(lat, lon, metadata))

    uv_data = uv_layer.uv if hasattr(uv_layer, "uv") else uv_layer.data
    use_attr = hasattr(uv_layer, "uv")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            u, v = vertex_uvs[mesh.loops[loop_index].vertex_index]
            if use_attr:
                uv_data[loop_index].vector = (u, v)
            else:
                uv_data[loop_index].uv = (u, v)
    if why:
        print(f"  (used the inline UV projection: {why})")
    return True, None


def imagery_material(image_path, name, brightness=1.0, saturation=1.0,
                     roughness=0.94):
    """Principled Base Color from the photograph; low specular to avoid haze sheen."""
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    output = tree.nodes.new('ShaderNodeOutputMaterial')
    output.location = (520, 0)
    bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (220, 0)
    _set(bsdf, "Roughness", roughness)
    _set(bsdf, "Metallic", 0.0)
    _set(bsdf, "Specular IOR Level", 0.05)
    _set(bsdf, "Coat Weight", 0.0)
    _set(bsdf, "Sheen Weight", 0.0)

    tex = tree.nodes.new('ShaderNodeTexImage')
    tex.location = (-360, 0)
    tex.extension = 'EXTEND'  # REPEAT wraps city edge across UV overflow.
    image = bpy.data.images.load(image_path, check_existing=True)
    # Reload: check_existing matches path; file may have been rewritten.
    try:
        image.reload()
    except RuntimeError:
        pass
    tex.image = image

    grade = tree.nodes.new('ShaderNodeHueSaturation')
    grade.location = (-60, 0)
    _set(grade, "Value", float(brightness))
    _set(grade, "Saturation", float(saturation))

    tree.links.new(tex.outputs['Color'], grade.inputs['Color'])
    tree.links.new(grade.outputs['Color'], bsdf.inputs['Base Color'])
    tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    mat.diffuse_color = (0.5, 0.5, 0.5, 1.0)
    mat.roughness = roughness
    mat.metallic = 0.0
    return mat


def imagery(terrain_obj, source="ESRI_IMAGERY", zoom=17, padding=0.0,
            brightness=1.0, saturation=1.0, roughness=0.94,
            max_tiles=MAX_TILES, workers=8, wms_url=None, wms_layer=None,
            fallback=True, verbose=True):
    """Drape map imagery over terrain using dem_bounds_* and fetch metadata for UVs.

    On failure with fallback=True, restores matte palette. Returns status dict.
    """
    result = {"ok": False, "source": None, "source_name": None,
              "attribution": None, "zoom": None, "tiles": 0, "tiles_failed": 0,
              "cached": 0, "seconds": 0.0, "image": None, "image_size": None,
              "error": None}

    if terrain_obj is None:
        result["error"] = "no terrain object (did terrain(source='dem') return None?)"
        if verbose:
            print(f"  no imagery: {result['error']}")
        return result

    sg_imagery = _imagery_module()

    bounds = _terrain_bounds(terrain_obj)
    if bounds is None:
        result["error"] = (
            f"{terrain_obj.name!r} carries no dem_bounds_*, it was not built "
            "by context.terrain()/ground()")
        if verbose:
            print(f"  no imagery: {result['error']}")
        return result
    if padding:
        bounds = _pad_bounds(bounds, float(padding))

    key = sg_imagery.resolve_source(source)
    is_wms = key == "WMS"
    if not is_wms:
        try:
            estimate = imagery_estimate(bounds, key, zoom, verbose=verbose)
        except ValueError as exc:
            result["error"] = str(exc)
            print(f"  NO IMAGERY: {exc}")
            _imagery_failed(terrain_obj, fallback)
            return result
        if estimate["over_limit"]:
            result["error"] = (
                f"{estimate['tiles']} tiles at zoom {estimate['zoom']} is over "
                f"the {max_tiles}-tile ceiling; lower `zoom` (each step down "
                "quarters the count) or shrink the radius")
            print(f"  REFUSING TO FETCH: {result['error']}")
            _imagery_failed(terrain_obj, fallback)
            return result
        zoom = estimate["zoom"]
        result["cached"] = estimate["cached"]
    elif not (wms_url and wms_layer):
        result["error"] = "source='WMS' needs both wms_url and wms_layer"
        if verbose:
            print(f"  no imagery: {result['error']}")
        return result

    original_ua = getattr(sg_imagery, "_USER_AGENT", "")
    sg_imagery._USER_AGENT = f"{original_ua} {_USER_AGENT_SUFFIX}".strip()
    started = time.time()
    try:
        image_path, metadata = sg_imagery.fetch_basemap(
            bounds=bounds, source=key, zoom=int(zoom), wms_url=wms_url,
            wms_layer=wms_layer, max_tiles=int(max_tiles),
            max_workers=int(workers))
    except KeyboardInterrupt:
        result["error"] = "interrupted"
        print("  imagery fetch canceled")
        return result
    except Exception as exc:  # noqa: BLE001
        result["seconds"] = time.time() - started
        result["error"] = f"{type(exc).__name__}: {exc}"
        print(f"  NO IMAGERY, the fetch failed after {result['seconds']:.1f} s: "
              f"{result['error']}")
        _imagery_failed(terrain_obj, fallback)
        return result
    finally:
        sg_imagery._USER_AGENT = original_ua
    elapsed = time.time() - started

    ok, error = _project_uv(terrain_obj, metadata)
    if not ok:
        result["error"] = error or "UV projection failed"
        print(f"  NO IMAGERY, fetched {image_path} but could not drape it: "
              f"{result['error']}")
        _imagery_failed(terrain_obj, fallback)
        return result

    mat = imagery_material(image_path, f"SGNB_Imagery_{terrain_obj.name}",
                           brightness=brightness, saturation=saturation,
                           roughness=roughness)
    terrain_obj.data.materials.clear()
    terrain_obj.data.materials.append(mat)

    failed = int(metadata.get("tiles_failed", 0))
    total = int(metadata.get("tiles_total", 0))
    terrain_obj["scigraphs_real_imagery"] = True
    terrain_obj["scigraphs_imagery_source"] = metadata["source"]
    terrain_obj["scigraphs_imagery_name"] = metadata["source_name"]
    terrain_obj["scigraphs_imagery_attribution"] = metadata["attribution"]
    terrain_obj["scigraphs_imagery_zoom"] = int(metadata.get("zoom") or 0)
    terrain_obj["scigraphs_imagery_tiles"] = total
    terrain_obj["scigraphs_imagery_tiles_failed"] = failed
    terrain_obj["scigraphs_imagery_image"] = image_path
    terrain_obj["scigraphs_imagery_seconds"] = float(elapsed)

    result.update({
        "ok": True, "source": metadata["source"],
        "source_name": metadata["source_name"],
        "attribution": metadata["attribution"], "zoom": metadata.get("zoom"),
        "tiles": total, "tiles_failed": failed, "seconds": float(elapsed),
        "image": image_path, "image_size": tuple(metadata["image_size"]),
    })

    if verbose:
        size = result["image_size"]
        print(f"  {metadata['source_name']}: {size[0]}x{size[1]} px in "
              f"{elapsed:.1f} s ({result['cached']} tiles from cache)")
        print(f"  {metadata['attribution']}")
    if failed:
        print(f"  WARNING: {failed} of {total} tiles failed and are dark gray "
              "in the composite. Re-run to fill them from the cache.")
    return result


_IMAGERY_KEYS = ("scigraphs_real_imagery", "scigraphs_imagery_source", "scigraphs_imagery_name",
                 "scigraphs_imagery_attribution", "scigraphs_imagery_zoom",
                 "scigraphs_imagery_tiles", "scigraphs_imagery_tiles_failed",
                 "scigraphs_imagery_image", "scigraphs_imagery_seconds")


def _imagery_failed(terrain_obj, fallback=True):
    """Wipe imagery stamps on failure; keep an existing good drape; optional palette fallback."""
    if has_imagery(terrain_obj):
        print(f"  keeping the drape already on {terrain_obj.name}: "
              f"{terrain_obj.get('scigraphs_imagery_name')}. It is left over from an "
              "earlier call, not the result of this one.")
        return
    for key in _IMAGERY_KEYS:
        terrain_obj.pop(key, None)
    if not fallback:
        return
    style_context([terrain_obj], keep_imagery=False)
    print("  the terrain is wearing the plain matte material. It is gray "
          "because there is no photograph, not because the photograph is gray.")


def _terrain_bounds(terrain_obj):
    """WGS84 dem_bounds_* rectangle, or None."""
    keys = ("dem_bounds_north", "dem_bounds_south",
            "dem_bounds_east", "dem_bounds_west")
    if terrain_obj is None or not all(k in terrain_obj for k in keys):
        return None
    return {"north": float(terrain_obj["dem_bounds_north"]),
            "south": float(terrain_obj["dem_bounds_south"]),
            "east": float(terrain_obj["dem_bounds_east"]),
            "west": float(terrain_obj["dem_bounds_west"])}


def _pad_bounds(bounds, fraction):
    """Grow a bbox by a fraction of its span on each side."""
    dlat = (bounds["north"] - bounds["south"]) * fraction
    dlon = (bounds["east"] - bounds["west"]) * fraction
    return {"north": min(bounds["north"] + dlat, 85.0),
            "south": max(bounds["south"] - dlat, -85.0),
            "east": min(bounds["east"] + dlon, 180.0),
            "west": max(bounds["west"] - dlon, -180.0)}


def has_imagery(obj):
    """True only for a surface wearing fetched imagery."""
    return bool(obj is not None and obj.get("scigraphs_real_imagery", False))


def attribution(objects, verbose=True):
    """Distinct imagery credit lines from context objects."""
    if not isinstance(objects, (list, tuple, set)):
        objects = [objects]
    lines = []
    for obj in objects:
        line = obj.get("scigraphs_imagery_attribution") if obj is not None else None
        if line and line not in lines:
            lines.append(line)
    if verbose:
        for line in lines:
            print(f"  {line}")
    return lines


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

def bbox(obj):
    """World-space bbox via matrix_world: min, max, size, center."""
    if obj is None or not hasattr(obj, "bound_box"):
        return None
    corners = [obj.matrix_world @ __import__("mathutils").Vector(c) for c in obj.bound_box]
    if not corners:
        return None
    lo = [min(c[i] for c in corners) for i in range(3)]
    hi = [max(c[i] for c in corners) for i in range(3)]
    return {"min": tuple(lo), "max": tuple(hi),
            "size": tuple(hi[i] - lo[i] for i in range(3)),
            "center": tuple((hi[i] + lo[i]) / 2.0 for i in range(3))}


def _overlap_fraction(a, b, axis):
    """Fraction of `a`'s span on one axis that lies inside `b`."""
    lo = max(a["min"][axis], b["min"][axis])
    hi = min(a["max"][axis], b["max"][axis])
    span = a["max"][axis] - a["min"][axis]
    if span <= 0:
        return 1.0 if lo <= hi else 0.0
    return max(0.0, hi - lo) / span


def align_surface(surface, graph_obj, verbose=True):
    """Translate surface XY onto the graph's projection origin (dem_center vs frame)."""
    if surface is None or graph_obj is None:
        return (0.0, 0.0)

    dem_lat = surface.get("dem_center_lat")
    dem_lon = surface.get("dem_center_lon")
    if dem_lat is None or dem_lon is None:
        if verbose:
            print(f"  {surface.name} carries no dem_center_lat/lon; "
                  "nothing to align against")
        return (0.0, 0.0)

    try:
        graph_frame = frame(graph_obj)
    except ValueError as exc:
        if verbose:
            print(f"  cannot align: {exc}")
        return (0.0, 0.0)

    x, y = project(float(dem_lat), float(dem_lon), graph_frame)
    before = (surface.location.x, surface.location.y)
    surface.location.x = x
    surface.location.y = y

    bpy.context.view_layer.update()

    scale = graph_frame[2] or 0.001
    shift = (x - before[0], y - before[1])
    if verbose:
        if abs(shift[0] / scale) < 1.0 and abs(shift[1] / scale) < 1.0:
            print("  surface aligned        already on the graph's own "
                  "projection origin, nothing to move")
        else:
            print(f"  surface aligned        {shift[0]:+.4f}, {shift[1]:+.4f} BU "
                  f"({shift[0] / scale:+.0f} m, {shift[1] / scale:+.0f} m) onto "
                  "the graph's own projection origin")
    return shift


def check_alignment(graph_obj, context_objs, offset_tolerance=0.05,
                    coverage_minimum=0.95, verbose=True):
    """XY center-offset and coverage checks; terrain can pass at the wrong origin."""
    if not isinstance(context_objs, (list, tuple)):
        context_objs = [context_objs]

    graph_box = bbox(graph_obj)
    if graph_box is None:
        return {"ok": False, "error": "graph has no bounding box"}
    diagonal = math.hypot(graph_box["size"][0], graph_box["size"][1]) or 1.0

    results = {"graph": graph_box, "diagonal_xy": diagonal, "objects": {}, "ok": True}
    for obj in context_objs:
        box = bbox(obj)
        if box is None:
            continue
        offset = math.hypot(box["center"][0] - graph_box["center"][0],
                            box["center"][1] - graph_box["center"][1])
        coverage = min(_overlap_fraction(graph_box, box, 0),
                       _overlap_fraction(graph_box, box, 1))
        size_ratio = ((box["size"][0] / graph_box["size"][0]) if graph_box["size"][0] else 0.0)
        ok = (offset / diagonal <= offset_tolerance) and (coverage >= coverage_minimum)
        results["objects"][obj.name] = {
            "bbox": box, "center_offset": offset,
            "center_offset_fraction": offset / diagonal,
            "graph_coverage": coverage, "size_ratio_x": size_ratio, "ok": ok}
        results["ok"] = results["ok"] and ok
        if verbose:
            print(f"[{'PASS' if ok else 'FAIL'}] "
                  f"{obj.name} aligned with {graph_obj.name}, "
                  f"center off by {offset:.4f} BU ({offset / diagonal * 100:.2f}% of the "
                  f"graph diagonal), covers {coverage * 100:.1f}% of it, "
                  f"{size_ratio:.2f}x its width")
    return results


def report(objects):
    """One line per context object: kind, size, provenance."""
    if not isinstance(objects, (list, tuple)):
        objects = [objects]
    for obj in objects:
        if obj is None:
            continue
        kind = obj.get("scigraphs_context_kind", "?")
        box = bbox(obj)
        extra = ""
        if kind in {"terrain", "ground"}:
            extra = (f"source={obj.get('scigraphs_elevation_source')} "
                     f"real_elevation={bool(obj.get('scigraphs_real_elevation'))}")
            if obj.get("scigraphs_real_imagery"):
                extra += (f" imagery={obj.get('scigraphs_imagery_source')}"
                          f"@z{obj.get('scigraphs_imagery_zoom')} "
                          f"({obj.get('scigraphs_imagery_tiles')} tiles)")
            else:
                extra += " imagery=none"
        elif kind == "buildings":
            extra = (f"{obj.get('building_count')} footprints, "
                     f"{float(obj.get('height_tagged_fraction', 0.0)) * 100:.0f}% tagged, "
                     f"default {obj.get('height_default_m'):.1f} m "
                     f"({obj.get('height_default_source')})")
        size = box["size"] if box else (0, 0, 0)
        print(f"  {obj.name:<22} {kind:<10} "
              f"{size[0]:.3f} x {size[1]:.3f} x {size[2]:.3f} BU   {extra}")


# --------------------------------------------------------------------------
# Notebook entry point
# --------------------------------------------------------------------------

def add_context(graph_obj, center, radius_m, ref, buildings_gdf=None,
                terrain_source="flat", api="open-elevation", resolution=24,
                vertical_scale=1.0, default_height="auto", coll=COLLECTION,
                clearance_m=CLEARANCE_M, reference="terrain",
                imagery=None, imagery_zoom=17, imagery_brightness=1.0,
                verbose=True):
    """Terrain, buildings, style, imagery, then settle, in that order: style clears materials."""
    _center_lat, _center_lon, scale = frame(ref)

    surface = terrain(center, radius_m, ref, source=terrain_source, api=api,
                      resolution=resolution, vertical_scale=vertical_scale,
                      coll=coll, verbose=verbose)

    built = []
    if buildings_gdf is not False:
        source = buildings_gdf if buildings_gdf is not None else center
        built = buildings(source, ref, radius_m=radius_m,
                          default_height=default_height, terrain=surface,
                          vertical_scale=vertical_scale, coll=coll, verbose=verbose)

    objects = ([surface] if surface is not None else []) + built
    style_context(objects)

    drape = None
    if imagery and surface is not None:
        drape = globals()["imagery"](
            surface, source=imagery, zoom=imagery_zoom,
            brightness=imagery_brightness, verbose=verbose)

    settle(objects, plane_z=0.0, clearance_m=clearance_m, scale=scale,
           reference=reference, verbose=verbose)

    alignment = None
    if graph_obj is not None and objects:
        alignment = check_alignment(graph_obj, objects, verbose=verbose)

    return {"terrain": surface, "buildings": built[0] if built else None,
            "objects": objects, "real_elevation": is_real_elevation(surface),
            "real_imagery": has_imagery(surface), "imagery": drape,
            "attribution": (drape or {}).get("attribution"),
            "alignment": alignment}
