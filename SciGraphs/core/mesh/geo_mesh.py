# GeoDataFrames to Blender mesh and curve objects, for the OSMnx and city2graph operators.

import bpy
import bmesh
from .geometry import _latlon_to_local_3d
from scigraphs_core.logger import log


def _resolve_projection_metadata(obj):
    """Return the (center_lat, center_lon, scale) projection metadata, from either the ``osmnx_*`` or the ``c2g_*`` keys."""
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
    """Create Blender mesh objects from a GeoDataFrame, one per feature group. ``separate_by_type`` groups by the ``building`` column; without ``osmnx_obj`` to align against, lon/lat are used unprojected."""
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


def _as_float(values, row_pos):
    """One value from a column list as a float, 0.0 for anything unusable. ``values`` is None when the layer or relation lacks the column, normal in a heterograph."""
    if values is None or row_pos >= len(values):
        return 0.0
    value = values[row_pos]
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if out == out else 0.0  # NaN reads as 0.0, as elsewhere


def _edge_endpoint_indices(edges_gdf, nodes_gdf):
    """Map each edge to a (src_idx, tgt_idx) pair of positional node indices, reading the edge MultiIndex against the node index. Aligned with ``edges_gdf`` rows; unresolvable rows are None."""
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
    """Materialize a city2graph result as a native SciGraphs MESH graph, or None.

    Vertices are nodes, mesh edges are graph edges. Writes the markers the native
    pipeline expects (num_nodes, num_edges, nodes_data, edges_data,
    node_positions, is_directed) plus every numeric column as a mesh attribute,
    POINT for nodes and EDGE for edges. ``ref_obj`` supplies the projection."""
    from shapely.geometry import Point

    if nodes_gdf is None or len(nodes_gdf) == 0:
        return None

    center_lat, center_lon, scale = _resolve_projection_metadata(ref_obj)
    if center_lat is None or center_lon is None:
        return None

    nodes_4326 = nodes_gdf
    if nodes_gdf.crs and str(nodes_gdf.crs).upper() != "EPSG:4326":
        nodes_4326 = nodes_gdf.to_crs("EPSG:4326")

    # One position per row, whatever the geometry: endpoints resolve against the
    # positional index, so dropping a row here silently renumbers every later edge
    # and the object renders as an empty frame. Representative point, not centroid,
    # which can land outside a concave or multi-part zone.
    positions = []
    degenerate = 0
    for geom in nodes_4326.geometry:
        if isinstance(geom, Point):
            point = geom
        elif geom is None or geom.is_empty:
            positions.append((0.0, 0.0, 0.0))
            degenerate += 1
            continue
        else:
            point = geom.representative_point()
        x, y, z = _latlon_to_local_3d(point.y, point.x, center_lat, center_lon, scale)
        positions.append((x, y, z))

    if degenerate:
        log(f"{name}: {degenerate} node(s) had no geometry and sit at the origin")

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
    """Materialize a heterogeneous city2graph result as a native MESH graph, or None.

    Every layer becomes vertices of one mesh and every relation becomes mesh edges,
    tagged ``layer_id`` (POINT) and ``edge_type_id`` (EDGE) so coloring can key on
    either. ``nodes_dict`` maps layer_name to a frame, ``edges_dict`` maps
    (src_layer, relation, tgt_layer) to a frame."""
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

    # Union of numeric columns over all layers, so coloring is not limited to
    # layer identity; a column missing from one layer reads 0.0 there.
    node_columns = []
    for layer_gdf in nodes_dict.values():
        for col in _numeric_columns(layer_gdf):
            if col not in node_columns:
                node_columns.append(col)
    node_values = {col: [] for col in node_columns}

    for layer_name in layer_names:
        layer_gdf = nodes_dict[layer_name]
        gdf_4326 = layer_gdf
        if layer_gdf.crs and str(layer_gdf.crs).upper() != "EPSG:4326":
            gdf_4326 = layer_gdf.to_crs("EPSG:4326")
        columns_here = {col: layer_gdf[col].tolist() for col in node_columns
                        if col in layer_gdf.columns}
        for row_pos, (node_id, geom) in enumerate(zip(layer_gdf.index,
                                                      gdf_4326.geometry)):
            if geom is None or geom.is_empty:
                continue
            point = geom if isinstance(geom, Point) else geom.representative_point()
            x, y, z = _latlon_to_local_3d(point.y, point.x, center_lat, center_lon, scale)
            node_index[(layer_name, node_id)] = len(positions)
            positions.append((x, y, z))
            layer_ids.append(layer_id_map[layer_name])
            for col in node_columns:
                node_values[col].append(_as_float(columns_here.get(col), row_pos))

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

    edge_columns = []
    for rel_gdf in edges_dict.values():
        if rel_gdf is None or len(rel_gdf) == 0:
            continue
        for col in _numeric_columns(rel_gdf):
            if col not in edge_columns:
                edge_columns.append(col)
    edge_values = {col: [] for col in edge_columns}

    for rel_key, rel_gdf in edges_dict.items():
        if rel_gdf is None or len(rel_gdf) == 0:
            continue
        src_layer, _relation, tgt_layer = rel_key
        import pandas as pd
        if not isinstance(rel_gdf.index, pd.MultiIndex) or rel_gdf.index.nlevels < 2:
            continue
        columns_here = {col: rel_gdf[col].tolist() for col in edge_columns
                        if col in rel_gdf.columns}
        for row_pos, key in enumerate(rel_gdf.index):
            src = node_index.get((src_layer, key[0]))
            tgt = node_index.get((tgt_layer, key[1]))
            if src is None or tgt is None or src == tgt:
                continue
            dedup = (min(src, tgt), max(src, tgt), rel_key)
            if dedup in seen:
                continue
            try:
                bm.edges.new([verts[src], verts[tgt]])
            except ValueError:
                # The pair already exists, contributed by another relation.
                continue
            seen.add(dedup)
            edge_type_ids.append(relation_id_map[rel_key])
            edges_flat.append(str(src))
            edges_flat.append(str(tgt))
            for col in edge_columns:
                edge_values[col].append(_as_float(columns_here.get(col), row_pos))

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
        id_attr = mesh.attributes.new(name="node_id", type='INT', domain='POINT')
        id_attr.data.foreach_set("value", list(range(len(layer_ids))))
        for col, values in node_values.items():
            if len(values) != len(layer_ids):
                continue
            col_attr = mesh.attributes.new(name=f"node_{col}", type='FLOAT',
                                           domain='POINT')
            col_attr.data.foreach_set("value", values)
    if len(mesh.edges) == len(edge_type_ids):
        eattr = mesh.attributes.new(name="edge_type_id", type='INT', domain='EDGE')
        eattr.data.foreach_set("value", edge_type_ids)
        for col, values in edge_values.items():
            if len(values) != len(edge_type_ids):
                continue
            col_attr = mesh.attributes.new(name=f"edge_{col}", type='FLOAT',
                                          domain='EDGE')
            col_attr.data.foreach_set("value", values)

    return obj


def create_curves_from_gdf(edges_gdf, name, feature_obj, thickness=0.0002, limit=1000):
    """Create a Blender curve object from LineString edges, or None if none drawn. ``thickness`` is the bevel depth, ``limit`` caps how many edges are drawn, ``feature_obj`` supplies the projection."""
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
    """Create a point-cloud mesh from Point nodes, projected via ``feature_obj``."""
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
