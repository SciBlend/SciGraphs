# Conversion of GeoDataFrames to Blender mesh / curve objects.
#
# Used by both OSMnx feature operators and City2Graph proximity operators
# to materialise geospatial data inside the 3D viewport.

import bpy
import bmesh
from .geometry import _latlon_to_local_3d
from ...utils.logger import log


def _resolve_projection_metadata(obj):
    """Return the (center_lat, center_lon, scale) projection metadata of an object.

    OSMnx feature objects store the projection under ``osmnx_*`` keys, while
    Overture/city2graph objects use the ``c2g_*`` keys. This resolver accepts
    either so visualization works with both sources.
    """
    center_lat = obj.get("osmnx_center_lat")
    center_lon = obj.get("osmnx_center_lon")
    scale = obj.get("osmnx_scale")

    if center_lat is None or center_lon is None:
        center_lat = obj.get("c2g_center_lat")
        center_lon = obj.get("c2g_center_lon")
        scale = obj.get("c2g_scale")

    if scale is None:
        scale = 0.001

    return center_lat, center_lon, scale


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


def _numeric_columns(gdf, skip=()):
    """Return the numeric column names of a GeoDataFrame, excluding geometry."""
    import numpy as np

    columns = []
    for col in gdf.columns:
        if col == "geometry" or col in skip:
            continue
        try:
            if np.issubdtype(gdf[col].dtype, np.number):
                columns.append(col)
        except TypeError:
            continue
    return columns


def _edge_endpoint_indices(edges_gdf, nodes_gdf):
    """Map each edge to a (src_idx, tgt_idx) pair of positional node indices.

    Uses the edge MultiIndex (source_id, target_id) referencing the node index
    when available. Returns a list aligned with ``edges_gdf`` rows; entries that
    cannot be resolved are ``None``.
    """
    import pandas as pd

    id_to_pos = {node_id: i for i, node_id in enumerate(nodes_gdf.index)}

    pairs = []
    index = edges_gdf.index
    if isinstance(index, pd.MultiIndex) and index.nlevels >= 2:
        for key in index:
            src = id_to_pos.get(key[0])
            tgt = id_to_pos.get(key[1])
            pairs.append((src, tgt) if src is not None and tgt is not None else None)
    return pairs


def create_native_graph_from_gdfs(nodes_gdf, edges_gdf, name, ref_obj,
                                   markers=None, node_attr_skip=(), edge_attr_skip=()):
    """Materialise a city2graph result as a native SciGraphs MESH graph.

    Builds a single mesh object whose vertices are the graph nodes and whose
    edges are the graph edges, writing the markers the native coloring/setup
    pipeline expects (num_nodes, num_edges, nodes_data, edges_data,
    node_positions, is_directed) plus numeric GeoDataFrame columns as scalar
    mesh attributes (POINT for nodes, EDGE for edges). Coordinates are projected
    with the reference object's geographic alignment so the result overlays the
    source network.

    Args:
        nodes_gdf: GeoDataFrame of Point nodes (positional index used as id).
        edges_gdf: GeoDataFrame of edges with a (source, target) MultiIndex.
        name: Object name.
        ref_obj: Object providing projection metadata (osmnx_* or c2g_*).
        markers: Optional dict of extra custom properties to set on the object.
        node_attr_skip: Node column names to exclude from attribute import.
        edge_attr_skip: Edge column names to exclude from attribute import.

    Returns:
        The created MESH object, or ``None`` on failure.
    """
    from shapely.geometry import Point

    if nodes_gdf is None or len(nodes_gdf) == 0:
        return None

    center_lat, center_lon, scale = _resolve_projection_metadata(ref_obj)
    if center_lat is None or center_lon is None:
        return None

    nodes_4326 = nodes_gdf
    if nodes_gdf.crs and str(nodes_gdf.crs).upper() != "EPSG:4326":
        nodes_4326 = nodes_gdf.to_crs("EPSG:4326")

    positions = []
    for geom in nodes_4326.geometry:
        if isinstance(geom, Point):
            x, y, z = _latlon_to_local_3d(geom.y, geom.x, center_lat, center_lon, scale)
            positions.append((x, y, z))
        else:
            positions.append((0.0, 0.0, 0.0))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    verts = [bm.verts.new(p) for p in positions]
    bm.verts.ensure_lookup_table()

    edge_pairs = []
    if edges_gdf is not None and len(edges_gdf) > 0:
        edge_pairs = _edge_endpoint_indices(edges_gdf, nodes_gdf)

    created_edge_rows = []
    seen = set()
    for row_idx, pair in enumerate(edge_pairs):
        if pair is None:
            continue
        src_idx, tgt_idx = pair
        if src_idx == tgt_idx:
            continue
        key = (min(src_idx, tgt_idx), max(src_idx, tgt_idx))
        if key in seen:
            continue
        try:
            bm.edges.new([verts[src_idx], verts[tgt_idx]])
            seen.add(key)
            created_edge_rows.append(row_idx)
        except ValueError:
            pass

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)

    obj["num_nodes"] = len(positions)
    obj["num_edges"] = len(created_edge_rows)
    obj["node_positions"] = [c for p in positions for c in p]
    obj["nodes_data"] = ",".join(str(i) for i in range(len(positions)))
    edges_flat = []
    for row_idx in created_edge_rows:
        src_idx, tgt_idx = edge_pairs[row_idx]
        edges_flat.append(str(src_idx))
        edges_flat.append(str(tgt_idx))
    obj["edges_data"] = ",".join(edges_flat)
    obj["is_directed"] = False
    obj["is_city2graph"] = True
    obj["c2g_center_lat"] = center_lat
    obj["c2g_center_lon"] = center_lon
    obj["c2g_scale"] = scale
    if ref_obj.get("is_osmnx"):
        obj["osmnx_center_lat"] = center_lat
        obj["osmnx_center_lon"] = center_lon
        obj["osmnx_scale"] = scale

    if markers:
        for key, value in markers.items():
            obj[key] = value

    _write_node_attributes(obj, nodes_gdf, node_attr_skip)
    if edges_gdf is not None and created_edge_rows:
        _write_edge_attributes(obj, edges_gdf, created_edge_rows, edge_attr_skip)

    return obj


def _write_node_attributes(obj, nodes_gdf, skip=()):
    """Write numeric node columns as POINT mesh attributes (plus node_id)."""
    mesh = obj.data
    n = len(mesh.vertices)
    if n == 0:
        return

    id_attr = mesh.attributes.new(name="node_id", type='INT', domain='POINT')
    id_attr.data.foreach_set("value", list(range(n)))

    for col in _numeric_columns(nodes_gdf, skip=skip):
        values = [float(v) if v is not None else 0.0 for v in nodes_gdf[col].tolist()]
        if len(values) != n:
            continue
        attr = mesh.attributes.new(name=f"node_{col}", type='FLOAT', domain='POINT')
        attr.data.foreach_set("value", values)


def _write_edge_attributes(obj, edges_gdf, created_edge_rows, skip=()):
    """Write numeric edge columns as EDGE mesh attributes."""
    mesh = obj.data
    if len(mesh.edges) != len(created_edge_rows):
        return

    for col in _numeric_columns(edges_gdf, skip=skip):
        series = edges_gdf[col].tolist()
        values = []
        for row_idx in created_edge_rows:
            v = series[row_idx]
            values.append(float(v) if v is not None else 0.0)
        attr = mesh.attributes.new(name=f"edge_{col}", type='FLOAT', domain='EDGE')
        attr.data.foreach_set("value", values)


def create_native_heterograph_from_dicts(nodes_dict, edges_dict, name, ref_obj, markers=None):
    """Materialise a heterogeneous city2graph result as a native MESH graph.

    Combines all node layers into one mesh (vertices) and all edge relations
    into mesh edges, tagging each vertex with a ``layer_id`` (POINT) and each
    edge with an ``edge_type_id`` (EDGE) so the native coloring pipeline can
    colour by layer or relation type. Writes the native graph markers.

    Args:
        nodes_dict: Mapping layer_name -> nodes GeoDataFrame (Point geometry).
        edges_dict: Mapping (src_layer, relation, tgt_layer) -> edges GeoDataFrame.
        name: Object name.
        ref_obj: Object providing projection metadata.
        markers: Optional dict of extra custom properties.

    Returns:
        The created MESH object, or ``None`` on failure.
    """
    from shapely.geometry import Point

    if not nodes_dict:
        return None

    center_lat, center_lon, scale = _resolve_projection_metadata(ref_obj)
    if center_lat is None or center_lon is None:
        return None

    positions = []
    layer_ids = []
    layer_names = list(nodes_dict.keys())
    layer_id_map = {name_: i for i, name_ in enumerate(layer_names)}
    node_index = {}

    for layer_name in layer_names:
        layer_gdf = nodes_dict[layer_name]
        gdf_4326 = layer_gdf
        if layer_gdf.crs and str(layer_gdf.crs).upper() != "EPSG:4326":
            gdf_4326 = layer_gdf.to_crs("EPSG:4326")
        for node_id, geom in zip(layer_gdf.index, gdf_4326.geometry):
            if geom is None or geom.is_empty:
                continue
            point = geom if isinstance(geom, Point) else geom.representative_point()
            x, y, z = _latlon_to_local_3d(point.y, point.x, center_lat, center_lon, scale)
            node_index[(layer_name, node_id)] = len(positions)
            positions.append((x, y, z))
            layer_ids.append(layer_id_map[layer_name])

    if not positions:
        return None

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm = bmesh.new()
    verts = [bm.verts.new(p) for p in positions]
    bm.verts.ensure_lookup_table()

    relation_names = list(edges_dict.keys())
    relation_id_map = {rel: i for i, rel in enumerate(relation_names)}
    edge_type_ids = []
    edges_flat = []
    seen = set()

    for rel_key, rel_gdf in edges_dict.items():
        if rel_gdf is None or len(rel_gdf) == 0:
            continue
        src_layer, _relation, tgt_layer = rel_key
        import pandas as pd
        if not isinstance(rel_gdf.index, pd.MultiIndex) or rel_gdf.index.nlevels < 2:
            continue
        for key in rel_gdf.index:
            src = node_index.get((src_layer, key[0]))
            tgt = node_index.get((tgt_layer, key[1]))
            if src is None or tgt is None or src == tgt:
                continue
            dedup = (min(src, tgt), max(src, tgt), rel_key)
            if dedup in seen:
                continue
            try:
                bm.edges.new([verts[src], verts[tgt]])
                seen.add(dedup)
                edge_type_ids.append(relation_id_map[rel_key])
                edges_flat.append(str(src))
                edges_flat.append(str(tgt))
            except ValueError:
                pass

    bm.to_mesh(mesh)
    bm.free()

    obj = bpy.data.objects.new(name, mesh)
    obj["num_nodes"] = len(positions)
    obj["num_edges"] = len(edge_type_ids)
    obj["node_positions"] = [c for p in positions for c in p]
    obj["nodes_data"] = ",".join(str(i) for i in range(len(positions)))
    obj["edges_data"] = ",".join(edges_flat)
    obj["is_directed"] = False
    obj["is_city2graph"] = True
    obj["c2g_center_lat"] = center_lat
    obj["c2g_center_lon"] = center_lon
    obj["c2g_scale"] = scale
    obj["layer_names"] = ",".join(str(n) for n in layer_names)
    obj["edge_type_names"] = ",".join(f"{k[0]}_{k[1]}_{k[2]}" for k in relation_names)
    if ref_obj.get("is_osmnx"):
        obj["osmnx_center_lat"] = center_lat
        obj["osmnx_center_lon"] = center_lon
        obj["osmnx_scale"] = scale
    if markers:
        for key, value in markers.items():
            obj[key] = value

    if len(mesh.vertices) == len(layer_ids):
        attr = mesh.attributes.new(name="layer_id", type='INT', domain='POINT')
        attr.data.foreach_set("value", layer_ids)
    if len(mesh.edges) == len(edge_type_ids):
        eattr = mesh.attributes.new(name="edge_type_id", type='INT', domain='EDGE')
        eattr.data.foreach_set("value", edge_type_ids)

    return obj


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

    center_lat, center_lon, scale = _resolve_projection_metadata(feature_obj)

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
