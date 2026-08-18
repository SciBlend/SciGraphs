"""GeoDataFrames -> SciGraphs mesh objects for coloring, layout, and render.

Render via `SciGraphs.api.preview` (GPU) or `SciGraphs.api.render` (EEVEE).
"""

import pathlib

import bpy

from ..core.mesh import geo_mesh

# --------------------------------------------------------------------------
# Blender scene plumbing
# --------------------------------------------------------------------------

def collection(name, parent=None):
    """Get or create a Blender collection under `parent` (or the scene)."""
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    target = parent if parent is not None else bpy.context.scene.collection
    if coll.name not in {c.name for c in target.children}:
        try:
            target.children.link(coll)
        except RuntimeError:
            pass  # already linked elsewhere
    return coll


def anchor(lat, lon, scale=0.001, name="C2G_Anchor"):
    """Create georeference Empty (`c2g_center_*` / `c2g_scale`). scale: meters→Blender units (0.001 → 1 km = 1 unit)."""
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, None)
        obj.empty_display_type = 'PLAIN_AXES'
        obj.empty_display_size = 0.5
        bpy.context.scene.collection.objects.link(obj)
    obj["c2g_center_lat"] = float(lat)
    obj["c2g_center_lon"] = float(lon)
    obj["c2g_scale"] = float(scale)
    obj["is_city2graph"] = True
    return obj


def anchor_from(gdf, scale=0.001, name="C2G_Anchor"):
    """Anchor centered on a GeoDataFrame's bounds (reprojected to WGS84)."""
    src = gdf if (gdf.crs is None or str(gdf.crs).upper() == "EPSG:4326") else gdf.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = src.total_bounds
    return anchor((miny + maxy) / 2.0, (minx + maxx) / 2.0, scale=scale, name=name)


def _link(obj, coll_name):
    """Link `obj` into a named collection, or into the scene root."""
    target = collection(coll_name) if coll_name else bpy.context.scene.collection
    if obj.name not in target.objects:
        target.objects.link(obj)
    return obj


def activate(obj):
    """Make `obj` active and sole selection (needed before most `bpy.ops.scigraphs.*`)."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    return obj


# --------------------------------------------------------------------------
# GeoDataFrames -> Blender
# --------------------------------------------------------------------------

def from_gdf(nodes_gdf, edges_gdf, name, ref=None, coll=None, markers=None):
    """Homogeneous (nodes, edges) graph from city2graph GDFs. edges need (source, target) MultiIndex."""
    ref = ref if ref is not None else anchor_from(nodes_gdf)
    obj = geo_mesh.create_native_graph_from_gdfs(
        nodes_gdf, edges_gdf, name, ref, markers=markers or {})
    if obj is None:
        return None
    return _link(obj, coll)


def from_hetero(nodes_dict, edges_dict, name, ref=None, coll=None, markers=None):
    """Heterogeneous city2graph result: layers → one mesh with layer_id / edge_type_id attrs."""
    if ref is None:
        first = next(iter(nodes_dict.values()))
        ref = anchor_from(first)
    obj = geo_mesh.create_native_heterograph_from_dicts(
        nodes_dict, edges_dict, name, ref, markers=markers or {})
    if obj is None:
        return None
    return _link(obj, coll)


def from_features(gdf, name, ref=None, coll=None):
    """Raw geometry mesh of polygons, lines, or points, with `_c2g_gdf_pickle` cached."""
    from ..core.city2graph import utils as c2g_utils

    ref = ref if ref is not None else anchor_from(gdf)

    # Mesh builder expects WGS84; pickle stores EPSG:4326 for operators.
    wgs84 = gdf if (gdf.crs is None or str(gdf.crs).upper() == "EPSG:4326") else gdf.to_crs("EPSG:4326")

    target = collection(coll) if coll else bpy.context.scene.collection
    return c2g_utils.gdf_to_blender_mesh(
        wgs84, name=name, collection_name=target.name, ref_obj=ref)


def from_networkx(graph, name, ref=None, coll=None, markers=None):
    """Materialize a NetworkX graph produced by city2graph (`as_nx=True`)."""
    import city2graph as c2g

    nodes_gdf, edges_gdf = c2g.nx_to_gdf(graph, nodes=True, edges=True)
    return from_gdf(nodes_gdf, edges_gdf, name, ref=ref, coll=coll, markers=markers)


# --------------------------------------------------------------------------
# Looking at the result
# --------------------------------------------------------------------------

def summary(obj):
    """Graph markers on an object as a plain dict."""
    if obj is None:
        return {"object": None}
    info = {
        "object": obj.name,
        "type": obj.type,
        "num_nodes": obj.get("num_nodes"),
        "num_edges": obj.get("num_edges"),
        "is_directed": obj.get("is_directed"),
    }
    if obj.type == 'MESH':
        info["vertices"] = len(obj.data.vertices)
        info["mesh_edges"] = len(obj.data.edges)
        info["attributes"] = [a.name for a in obj.data.attributes]
    for key in ("layer_names", "edge_type_names", "graph_type", "morpho_relation"):
        if key in obj.keys():
            info[key] = obj[key]
    return info


def attributes(obj):
    """Scalar mesh attributes available for coloring, with their domains."""
    if obj is None or obj.type != 'MESH':
        return []
    return [(a.name, a.domain, a.data_type) for a in obj.data.attributes]


def visualize(obj, node_size=None, edge_thickness=None):
    """Build Geometry Nodes viz (nodes on verts, tubes on edges)."""
    activate(obj)
    viz = getattr(bpy.context.scene, "scigraphs_viz", None)
    if viz is not None:
        if node_size is not None:
            viz.node_scale = float(node_size)
        if edge_thickness is not None:
            viz.edge_thickness = float(edge_thickness)
    return bpy.ops.scigraphs.setup_visualization(target='FULL')


def color_by(obj, attribute, colormap="viridis", reverse=False):
    """Color by mesh attribute (GDF columns arrive as `node_*` / `edge_*`)."""
    activate(obj)
    props = getattr(bpy.context.scene, "scigraphs_coloring", None)
    if props is None:
        return {'CANCELLED'}, "coloring properties are not registered"

    names = [a.name for a in obj.data.attributes]
    if attribute not in names:
        return {'CANCELLED'}, f"'{attribute}' not on this mesh; have {names}"

    props.colormap = colormap
    props.reverse = bool(reverse)
    props.auto_range = True
    result = bpy.ops.scigraphs.color_set_attribute(attribute=attribute)
    return result, f"{attribute} -> {colormap}"


def frame(obj=None):
    """Point the viewport at an object (or at everything)."""
    if obj is not None:
        activate(obj)
    for area in bpy.context.screen.areas if bpy.context.screen else []:
        if area.type != 'VIEW_3D':
            continue
        region = next((r for r in area.regions if r.type == 'WINDOW'), None)
        if region is None:
            continue
        with bpy.context.temp_override(area=area, region=region):
            if obj is not None:
                bpy.ops.view3d.view_selected()
            else:
                bpy.ops.view3d.view_all()
    return obj


def clear_scene(keep_anchor=True):
    """Empty the scene so a notebook can be re-run from the top."""
    for obj in list(bpy.data.objects):
        if keep_anchor and obj.get("c2g_center_lat") is not None and obj.type == 'EMPTY':
            continue
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in list(bpy.data.collections):
        if not coll.objects and not coll.children:
            bpy.data.collections.remove(coll)


def save_gdf(gdf, path, driver="GPKG"):
    """Write GDF to disk; extra geometry cols → WKT, MultiIndex → columns."""
    import geopandas as gpd

    frame = gdf.reset_index() if gdf.index.nlevels > 1 or gdf.index.name else gdf.copy()
    active = frame.geometry.name

    for column in list(frame.columns):
        if column == active:
            continue
        values = frame[column]
        if isinstance(values, gpd.GeoSeries) or getattr(values, "dtype", None) == "geometry":
            frame[column] = gpd.GeoSeries(values, crs=frame.crs).to_wkt()

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(frame, geometry=active, crs=gdf.crs).to_file(path, driver=driver)
    return path


def save_blend(path):
    """Save a copy of the current scene (`copy=True` keeps the session file)."""
    path = pathlib.Path(path)
    if path.suffix.lower() != ".blend":
        path = path.with_suffix(".blend")
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True)
    return path
