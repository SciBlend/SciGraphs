"""Blender mesh to GeoDataFrame conversion, plus thin wrappers over city2graph 0.3.1 or newer, which return None when the library is missing."""
import bpy
import bmesh
from scigraphs_core.logger import log


def _copy_projection_metadata(src, dst):
    """Copy the projection-defining custom properties, so a derived object still produces real lat/lon through ``blender_to_geopandas``."""
    keys = (
        "c2g_center_lat", "c2g_center_lon", "c2g_scale",
        "osmnx_center_lat", "osmnx_center_lon", "osmnx_scale",
        "is_city2graph", "is_osm_features",
    )
    for key in keys:
        if key in src:
            dst[key] = src[key]


def mesh_to_centroids(obj, name=None, collection_name=None):
    """Create a point-only mesh with one vertex per feature centroid, one per connected component. The new object inherits the source's projection metadata, so proximity graphs, morphological graphs and lat/lon export stay aligned."""
    if obj is None or obj.type != 'MESH' or not obj.data:
        log("mesh_to_centroids: invalid mesh object")
        return None

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        # Union-find over the verts to group them into islands.
        parent = list(range(len(bm.verts)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for edge in bm.edges:
            union(edge.verts[0].index, edge.verts[1].index)
        for face in bm.faces:
            verts = list(face.verts)
            for v in verts[1:]:
                union(verts[0].index, v.index)

        groups = {}
        for v in bm.verts:
            groups.setdefault(find(v.index), []).append(v.co.copy())

        if not groups:
            log("mesh_to_centroids: no geometry found")
            return None

        centroids = []
        for verts in groups.values():
            cx = sum(c.x for c in verts) / len(verts)
            cy = sum(c.y for c in verts) / len(verts)
            cz = sum(c.z for c in verts) / len(verts)
            centroids.append((cx, cy, cz))
    finally:
        bm.free()

    new_name = name or f"{obj.name}_Centroids"
    mesh = bpy.data.meshes.new(new_name)
    mesh.from_pydata(centroids, [], [])
    mesh.update()

    new_obj = bpy.data.objects.new(new_name, mesh)
    target_coll = None
    if collection_name:
        target_coll = bpy.data.collections.get(collection_name)
        if target_coll is None:
            target_coll = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(target_coll)
    if target_coll is None:
        if obj.users_collection:
            target_coll = obj.users_collection[0]
        else:
            target_coll = bpy.context.scene.collection
    target_coll.objects.link(new_obj)

    _copy_projection_metadata(obj, new_obj)
    new_obj["feature_count"] = len(centroids)
    new_obj["c2g_geometry_kind"] = "POINT"
    new_obj["c2g_derived_from"] = obj.name
    # Centroids are fresh geometry, so the source GDF cache must not follow.
    new_obj.pop("_c2g_gdf_pickle", None)
    new_obj.pop("_c2g_gdf_crs", None)

    return new_obj


def gdf_to_blender_mesh(gdf, name="C2G_Features", collection_name=None, osmnx_obj=None,
                        ref_obj=None):
    """Convert a GeoDataFrame to Blender mesh object(s), returned as a list. ``ref_obj`` supplies the projection to align against, in either the ``osmnx_*`` or ``c2g_*`` key family; ``osmnx_obj`` is its old name, kept for existing callers."""
    from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
    from ..mesh.geometry import _latlon_to_local_3d
    from ..mesh.geo_mesh import _resolve_projection_metadata

    if gdf is None or len(gdf) == 0:
        log("Empty GeoDataFrame, no objects created")
        return []

    center_lat = None
    center_lon = None
    scale = 0.001

    # Either key family. Requiring `is_osmnx` made a c2g_* reference fall through
    # to the data's own bounds, drifting away from that reference.
    reference = ref_obj if ref_obj is not None else osmnx_obj
    if reference is not None:
        center_lat, center_lon, scale = _resolve_projection_metadata(reference)

    if center_lat is None or center_lon is None:
        bounds = gdf.total_bounds
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        scale = 0.001
        log(f"No projection reference - using data center: ({center_lat:.6f}, {center_lon:.6f})")
    else:
        log(f"Using reference projection: center=({center_lat:.6f}, {center_lon:.6f}), scale={scale}")

    def convert_coord(lon, lat):
        if center_lat is not None and center_lon is not None:
            return _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale)
        else:
            return (lon * 100, lat * 100, 0)
    
    collection = None
    if collection_name:
        if collection_name in bpy.data.collections:
            collection = bpy.data.collections[collection_name]
        else:
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
    
    created_objects = []
    bm = bmesh.new()

    # Do not share vertices across features: where two buildings touch,
    # ``bm.faces.new`` raises "face would overlap existing edge" and loses one.
    for idx, row in gdf.iterrows():
        geom = row.geometry

        if isinstance(geom, (Polygon, MultiPolygon)):
            if isinstance(geom, MultiPolygon):
                polygons = list(geom.geoms)
            else:
                polygons = [geom]

            for poly in polygons:
                if poly.is_empty:
                    continue

                coords = list(poly.exterior.coords)
                if len(coords) < 4:  # need at least 3 unique + closing
                    continue

                verts = [bm.verts.new(convert_coord(x, y)) for x, y in coords[:-1]]

                if len(verts) >= 3:
                    try:
                        bm.faces.new(verts)
                    except ValueError:
                        # Degenerate ring: drop its verts so the mesh stays clean.
                        for v in verts:
                            bm.verts.remove(v)

        elif isinstance(geom, (LineString, MultiLineString)):
            if isinstance(geom, MultiLineString):
                lines = list(geom.geoms)
            else:
                lines = [geom]

            for line in lines:
                coords = list(line.coords)
                if len(coords) < 2:
                    continue

                prev_vert = None
                for x, y in coords:
                    vert = bm.verts.new(convert_coord(x, y))
                    if prev_vert is not None:
                        try:
                            bm.edges.new([prev_vert, vert])
                        except ValueError:
                            pass
                    prev_vert = vert

        elif isinstance(geom, Point):
            x, y = geom.x, geom.y
            bm.verts.new(convert_coord(x, y))

    if len(bm.verts) > 0:
        mesh = bpy.data.meshes.new(f"{name}_mesh")
        bm.to_mesh(mesh)
        bm.free()
        
        obj = bpy.data.objects.new(name, mesh)
        
        if collection:
            collection.objects.link(obj)
        else:
            bpy.context.collection.objects.link(obj)
        
        obj["is_city2graph"] = True
        obj["feature_count"] = len(gdf)
        obj["c2g_center_lat"] = center_lat
        obj["c2g_center_lon"] = center_lon
        obj["c2g_scale"] = scale

        # Cache the source frame as a base64 pickle: morphology and tessellation
        # read it, avoiding a mesh round-trip that loses degenerate polygons.
        try:
            import pickle
            import base64
            payload = base64.b64encode(pickle.dumps(gdf)).decode("ascii")
            # Cap at 4 MB so a big frame does not bloat the .blend.
            if len(payload) < 4 * 1024 * 1024:
                obj["_c2g_gdf_pickle"] = payload
                obj["_c2g_gdf_crs"] = str(gdf.crs) if gdf.crs is not None else "EPSG:4326"
        except Exception as _e:  # noqa: BLE001
            log(f"GDF cache skipped: {_e}")

        created_objects.append(obj)
    else:
        bm.free()
    
    log(f"Created {len(created_objects)} object(s) from GeoDataFrame with {len(gdf)} features")
    return created_objects


def blender_to_geopandas(obj, crs="EPSG:4326"):
    """Convert a Blender mesh object to a GeoDataFrame.

    Geometry follows the mesh contents: one Polygon per face, else one LineString
    per edge chain, else one Point per vertex. ``c2g_center_lat/lon/scale`` or
    ``osmnx_*`` maps back to lon/lat; missing metadata still yields a frame."""
    import geopandas as gpd
    from shapely.geometry import Point, LineString, Polygon
    from ..mesh.geometry import _local_3d_to_latlon

    if obj is None or obj.type != 'MESH':
        log("Invalid object: must be a mesh object")
        return None

    # Fast path: a pickled copy with the exact geometry types and attributes.
    cached = obj.get("_c2g_gdf_pickle")
    if cached:
        try:
            import pickle
            import base64
            cached_gdf = pickle.loads(base64.b64decode(cached))
            target_crs = obj.get("_c2g_gdf_crs", crs)
            if cached_gdf.crs is None and target_crs:
                cached_gdf.set_crs(target_crs, inplace=True)
            if crs and cached_gdf.crs is not None and str(cached_gdf.crs) != str(crs):
                cached_gdf = cached_gdf.to_crs(crs)
            log(
                "Loaded cached GeoDataFrame from object: "
                f"{len(cached_gdf)} features ({cached_gdf.geom_type.iloc[0] if len(cached_gdf) else 'empty'})"
            )
            return cached_gdf
        except Exception as _e:  # noqa: BLE001
            log(f"Cached GDF unusable, falling back to mesh extraction: {_e}")

    center_lat = obj.get("c2g_center_lat") or obj.get("osmnx_center_lat")
    center_lon = obj.get("c2g_center_lon") or obj.get("osmnx_center_lon")
    scale = obj.get("c2g_scale") or obj.get("osmnx_scale", 0.001)

    if center_lat is None or center_lon is None:
        log("Warning: Object has no projection metadata, using approximate conversion")
        center_lat = 0.0
        center_lon = 0.0

    mesh = obj.data

    def vert_to_lonlat(vert):
        x, y, _z = vert.co
        if center_lat != 0 or center_lon != 0:
            lat, lon = _local_3d_to_latlon(x, y, center_lat, center_lon, scale)
        else:
            lon = x / 100
            lat = y / 100
        return (lon, lat)

    coords = [vert_to_lonlat(v) for v in mesh.vertices]

    geometries = []

    if len(mesh.polygons) > 0:
        for poly in mesh.polygons:
            ring = [coords[i] for i in poly.vertices]
            if len(ring) < 3:
                continue
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            try:
                geom = Polygon(ring)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                if geom.is_empty:
                    continue
                geometries.append(geom)
            except Exception:  # noqa: BLE001
                continue
        log(f"Converted {len(geometries)} faces to Polygon GeoDataFrame")

    elif len(mesh.edges) > 0:
        from collections import defaultdict

        adj = defaultdict(list)
        for edge in mesh.edges:
            a, b = edge.vertices
            adj[a].append(b)
            adj[b].append(a)

        visited_edges = set()

        def edge_key(a, b):
            return (a, b) if a < b else (b, a)

        def walk_chain(start, neighbor):
            chain = [start, neighbor]
            visited_edges.add(edge_key(start, neighbor))
            prev, current = start, neighbor
            while True:
                # Only degree-2 nodes are chain interiors.
                neighbors = [n for n in adj[current] if n != prev]
                if len(neighbors) != 1 or len(adj[current]) != 2:
                    break
                nxt = neighbors[0]
                k = edge_key(current, nxt)
                if k in visited_edges:
                    break
                visited_edges.add(k)
                chain.append(nxt)
                prev, current = current, nxt
            return chain

        # Start chains at junctions and endpoints.
        for v in adj:
            if len(adj[v]) == 2:
                continue
            for n in adj[v]:
                k = edge_key(v, n)
                if k in visited_edges:
                    continue
                chain = walk_chain(v, n)
                geometries.append(LineString([coords[i] for i in chain]))

        # Remaining unvisited edges form pure cycles; pick any starting vertex.
        for edge in mesh.edges:
            a, b = edge.vertices
            if edge_key(a, b) in visited_edges:
                continue
            chain = walk_chain(a, b)
            geometries.append(LineString([coords[i] for i in chain]))

        log(f"Converted {len(geometries)} edge chains to LineString GeoDataFrame")

    else:
        geometries = [Point(lon, lat) for (lon, lat) in coords]
        log(f"Converted {len(geometries)} vertices to Point GeoDataFrame")

    gdf = gpd.GeoDataFrame(
        {'feature_id': range(len(geometries))},
        geometry=geometries,
        crs=crs,
    )
    return gdf


def create_collection(name, parent_collection=None):
    """Return the named Blender collection, creating it if it does not exist."""
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    
    collection = bpy.data.collections.new(name)
    
    if parent_collection:
        parent_collection.children.link(collection)
    else:
        bpy.context.scene.collection.children.link(collection)
    
    return collection


def extract_graph_from_blender(obj):
    """Extract a NetworkX graph from a Blender mesh, with local xyz node positions."""
    import networkx as nx
    
    if obj is None or obj.type != 'MESH':
        log("Invalid object: must be a mesh object")
        return None
    
    G = nx.Graph()
    mesh = obj.data
    
    for i, vert in enumerate(mesh.vertices):
        G.add_node(i, pos=(vert.co.x, vert.co.y, vert.co.z))
    
    for edge in mesh.edges:
        v1, v2 = edge.vertices
        G.add_edge(v1, v2)
    
    log(f"Extracted graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def gdf_to_nx(nodes=None, edges=None, keep_geom=True, multigraph=False, directed=False):
    """Convert node and edge GeoDataFrames to a NetworkX graph; a heterogeneous graph takes dicts for ``nodes`` and ``edges``, not frames."""
    from scigraphs_core.city2graph.get_c2g import get_city2graph
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.utils import gdf_to_nx as c2g_gdf_to_nx
        return c2g_gdf_to_nx(nodes=nodes, edges=edges, keep_geom=keep_geom,
                            multigraph=multigraph, directed=directed)
    except Exception as e:
        log(f"Error in gdf_to_nx: {e}")
        return None


def nx_to_gdf(G, nodes=True, edges=True):
    """Convert a NetworkX graph to ``(nodes_gdf, edges_gdf)``, or one frame when only ``nodes`` or only ``edges`` is asked for."""
    from scigraphs_core.city2graph.get_c2g import get_city2graph
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.utils import nx_to_gdf as c2g_nx_to_gdf
        return c2g_nx_to_gdf(G, nodes=nodes, edges=edges)
    except Exception as e:
        log(f"Error in nx_to_gdf: {e}")
        return None


def filter_graph_by_distance(graph, center, threshold, nodes=None, edges=None):
    """Keep the graph elements within ``threshold`` of ``center``, in CRS units: meters for a projected graph, degrees for a geographic one."""
    from scigraphs_core.city2graph.get_c2g import get_city2graph
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.utils import filter_graph_by_distance as c2g_filter

        log(f"Filtering graph by distance: threshold={threshold}")
        return c2g_filter(graph=graph, center_point=center, threshold=threshold)
    except Exception as e:
        log(f"Error in filter_graph_by_distance: {e}")
        return None


def create_isochrone(graph, center, threshold, weight="length"):
    """Return the polygon reachable from ``center`` within travel cost ``threshold``, in the units of the ``weight`` edge attribute."""
    from scigraphs_core.city2graph.get_c2g import get_city2graph
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.utils import create_isochrone as c2g_isochrone

        log(f"Creating isochrone: threshold={threshold}, weight={weight}")
        return c2g_isochrone(graph=graph, center_point=center, threshold=threshold, edge_attr=weight)
    except Exception as e:
        log(f"Error in create_isochrone: {e}")
        return None


def clip_graph(graph, polygon, nodes=None, edges=None):
    """Clip a graph to ``polygon``, given as a Shapely Polygon or a GeoDataFrame."""
    from scigraphs_core.city2graph.get_c2g import get_city2graph
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.utils import clip_graph as c2g_clip

        log("Clipping graph to polygon")
        return c2g_clip(graph=graph, area=polygon, as_nx=True)
    except Exception as e:
        log(f"Error in clip_graph: {e}")
        return None


def remove_isolated_components(graph, nodes=None, edges=None):
    """Keep only the largest connected component of a graph."""
    from scigraphs_core.city2graph.get_c2g import get_city2graph
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.utils import remove_isolated_components as c2g_remove

        log("Removing isolated components")
        return c2g_remove(graph, as_nx=True)
    except Exception as e:
        log(f"Error in remove_isolated_components: {e}")
        return None


def validate_gdf(nodes_gdf=None, edges_gdf=None, allow_empty=True):
    """Validate node and edge GeoDataFrames, returning ``(validated_nodes, validated_edges, is_hetero)``; ``is_hetero`` says whether they were dicts."""
    from scigraphs_core.city2graph.get_c2g import get_city2graph
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.utils import validate_gdf as c2g_validate
        return c2g_validate(nodes_gdf=nodes_gdf, edges_gdf=edges_gdf, 
                           allow_empty=allow_empty)
    except Exception as e:
        log(f"Error in validate_gdf: {e}")
        return None

