# Conversion of GeoDataFrames to Blender mesh / curve objects.
#
# Used by both OSMnx feature operators and City2Graph proximity operators
# to materialise geospatial data inside the 3D viewport.

import bpy
import bmesh
from .geometry import _latlon_to_local_3d
from ...utils.logger import log


def create_feature_mesh_from_gdf(gdf, name="OSM_Features", separate_by_type=False, osmnx_obj=None):
    """
    Create Blender mesh objects from a GeoDataFrame.

    Args:
        gdf: GeoDataFrame with a geometry column.
        name: Base name for the created objects.
        separate_by_type: Group features by the ``building`` column.
        osmnx_obj: Optional OSMnx object to align coordinate systems.

    Returns:
        list of created Blender objects.
    """
    from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString

    if gdf is None or len(gdf) == 0:
        return []

    center_lat = None
    center_lon = None
    scale = 0.001

    if osmnx_obj is not None and osmnx_obj.get("is_osmnx", False):
        center_lat = osmnx_obj.get("osmnx_center_lat")
        center_lon = osmnx_obj.get("osmnx_center_lon")
        scale = osmnx_obj.get("osmnx_scale", 0.001)

    def convert_coord(lon, lat):
        if center_lat is not None and center_lon is not None:
            return _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale)
        return (lon, lat, 0)

    created_objects = []

    if separate_by_type and 'building' in gdf.columns:
        feature_groups = gdf.groupby(gdf.get('building', 'unknown'))
    else:
        feature_groups = [(name, gdf)]

    for group_name, group_gdf in feature_groups:
        bm = bmesh.new()

        for _idx, row in group_gdf.iterrows():
            geom = row.geometry

            if isinstance(geom, (Polygon, MultiPolygon)):
                polygons = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
                for poly in polygons:
                    if poly.is_empty:
                        continue
                    coords = list(poly.exterior.coords)
                    if len(coords) < 3:
                        continue
                    verts = [bm.verts.new(convert_coord(x, y)) for x, y in coords[:-1]]
                    if len(verts) >= 3:
                        bm.faces.new(verts)

            elif isinstance(geom, (LineString, MultiLineString)):
                lines = list(geom.geoms) if isinstance(geom, MultiLineString) else [geom]
                for line in lines:
                    coords = list(line.coords)
                    if len(coords) < 2:
                        continue
                    prev_vert = None
                    for x, y in coords:
                        vert = bm.verts.new(convert_coord(x, y))
                        if prev_vert:
                            bm.edges.new([prev_vert, vert])
                        prev_vert = vert

            elif isinstance(geom, Point):
                bm.verts.new(convert_coord(geom.x, geom.y))

        if len(bm.verts) > 0:
            mesh = bpy.data.meshes.new(f"{group_name}_mesh")
            bm.to_mesh(mesh)
            bm.free()

            obj = bpy.data.objects.new(f"{group_name}", mesh)
            bpy.context.collection.objects.link(obj)

            obj["is_osm_features"] = True
            obj["feature_count"] = len(group_gdf)

            created_objects.append(obj)
        else:
            bm.free()

    return created_objects


def create_curves_from_gdf(edges_gdf, name, feature_obj, thickness=0.0002, limit=1000):
    """
    Create a Blender curve object from an edges GeoDataFrame.

    Args:
        edges_gdf: GeoDataFrame with LineString geometries.
        name: Object name.
        feature_obj: Source feature object used for coordinate transform parameters.
        thickness: Bevel depth of the curve.
        limit: Maximum number of edges to visualise.

    Returns:
        Curve object, or ``None``.
    """
    from shapely.geometry import LineString

    if edges_gdf is None or len(edges_gdf) == 0:
        return None

    center_lat = feature_obj.get("osmnx_center_lat")
    center_lon = feature_obj.get("osmnx_center_lon")
    scale = feature_obj.get("osmnx_scale", 0.001)

    if center_lat is None or center_lon is None:
        return None

    log(f"Creating curves for {name}: center=({center_lat:.6f}, {center_lon:.6f}), scale={scale}")
    log(f"  Source CRS: {edges_gdf.crs}, converting to EPSG:4326")

    if edges_gdf.crs and str(edges_gdf.crs).upper() != "EPSG:4326":
        edges_gdf = edges_gdf.to_crs("EPSG:4326")

    curve_data = bpy.data.curves.new(name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.bevel_depth = thickness

    count = 0
    for _idx, row in edges_gdf.iterrows():
        if count >= limit:
            break

        if hasattr(row, 'geometry') and row.geometry:
            geom = row.geometry

            if isinstance(geom, LineString):
                coords = list(geom.coords)
                if len(coords) >= 2:
                    polyline = curve_data.splines.new('POLY')
                    polyline.points.add(len(coords) - 1)

                    for i, (lon, lat) in enumerate(coords):
                        x, y, z = _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale)
                        polyline.points[i].co = (x, y, z + 0.01, 1.0)

                    count += 1

    if count == 0:
        bpy.data.curves.remove(curve_data)
        return None

    curve_obj = bpy.data.objects.new(name, curve_data)
    return curve_obj


def create_nodes_mesh_from_gdf(nodes_gdf, name, feature_obj):
    """
    Create a point-cloud mesh from a nodes GeoDataFrame.

    Args:
        nodes_gdf: GeoDataFrame with Point geometries.
        name: Object name.
        feature_obj: Source feature object for coordinate transform.

    Returns:
        Mesh object, or ``None``.
    """
    from shapely.geometry import Point

    if nodes_gdf is None or len(nodes_gdf) == 0:
        return None

    center_lat = feature_obj.get("osmnx_center_lat")
    center_lon = feature_obj.get("osmnx_center_lon")
    scale = feature_obj.get("osmnx_scale", 0.001)

    if center_lat is None or center_lon is None:
        return None

    bm = bmesh.new()

    for _idx, row in nodes_gdf.iterrows():
        geom = row.geometry
        if isinstance(geom, Point):
            x, y, z = _latlon_to_local_3d(geom.y, geom.x, center_lat, center_lon, scale)
            bm.verts.new((x, y, z))

    if len(bm.verts) == 0:
        bm.free()
        return None

    mesh = bpy.data.meshes.new(f"{name}_nodes")
    bm.to_mesh(mesh)
    bm.free()

    nodes_obj = bpy.data.objects.new(name, mesh)
    return nodes_obj
