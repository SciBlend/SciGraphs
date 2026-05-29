from ...utils.logger import log
from .get_c2g import get_city2graph
from . import utils


def create_tessellation(buildings_obj, barriers_obj=None, shrink=0.4, segment_length=0.5):
    """
    Create Voronoi tessellation from building footprints.
    
    Args:
        buildings_obj: Blender object containing building geometries
        barriers_obj: Optional Blender object containing barriers (roads, etc.)
        shrink: Shrink factor for tessellation (default 0.4)
        segment_length: Discretization segment length (default 0.5)
    
    Returns:
        Blender object with tessellation mesh
    """
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.utils import create_tessellation as c2g_tessellation
        
        buildings_gdf = utils.blender_to_geopandas(buildings_obj)
        if buildings_gdf is None or len(buildings_gdf) == 0:
            log("No building geometries found")
            return None
        
        if buildings_gdf.crs and buildings_gdf.crs.is_geographic:
            log(f"Converting from geographic CRS {buildings_gdf.crs} to projected")
            bounds = buildings_gdf.total_bounds
            center_lon = (bounds[0] + bounds[2]) / 2
            import pyproj
            utm_crs = pyproj.CRS.from_proj4(
                f"+proj=utm +zone={int((center_lon + 180) / 6) + 1} +datum=WGS84 +units=m +no_defs"
            )
            buildings_gdf = buildings_gdf.to_crs(utm_crs)
            log(f"Projected to {utm_crs}")
        
        barriers_gdf = None
        if barriers_obj is not None:
            barriers_gdf = utils.blender_to_geopandas(barriers_obj)
            if barriers_gdf is not None and barriers_gdf.crs and barriers_gdf.crs.is_geographic:
                barriers_gdf = barriers_gdf.to_crs(buildings_gdf.crs)
        
        log("Creating tessellation...")
        log(f"Parameters: shrink={shrink}, segment_length={segment_length}")
        
        tessellation_gdf = c2g_tessellation(
            geometry=buildings_gdf.geometry,
            buildings=buildings_gdf,
            barriers=barriers_gdf,
            shrink=shrink,
            segment=segment_length
        )
        
        if tessellation_gdf is not None and len(tessellation_gdf) > 0:
            tessellation_gdf = tessellation_gdf.to_crs("EPSG:4326")
        
        if tessellation_gdf is None or len(tessellation_gdf) == 0:
            log("Tessellation failed or returned no geometries")
            return None
        
        log(f"Created {len(tessellation_gdf)} tessellation cells")
        
        osmnx_obj = buildings_obj if buildings_obj.get("is_osmnx") else None
        c2g_obj = buildings_obj if buildings_obj.get("is_city2graph") else None
        ref_obj = osmnx_obj or c2g_obj
        
        objects = utils.gdf_to_blender_mesh(
            tessellation_gdf,
            name="Urban_Tessellation",
            collection_name="C2G_Morphology",
            osmnx_obj=ref_obj
        )
        
        if objects and len(objects) > 0:
            tessellation_obj = objects[0]
            tessellation_obj["is_tessellation"] = True
            tessellation_obj["tessellation_shrink"] = shrink
            tessellation_obj["tessellation_segment"] = segment_length
            log("Tessellation object created successfully")
            return tessellation_obj
        
        return None
        
    except Exception as e:
        log(f"Error creating tessellation: {e}")
        import traceback
        traceback.print_exc()
        return None


def _reproject_graph_pos_to_wgs84(G, source_crs):
    """Convert all node positions of *G* (assumed in ``source_crs``) to
    (lon, lat) so that ``create_graph_from_networkx`` can place them
    correctly relative to the OSMnx parent."""
    try:
        from pyproj import Transformer
    except Exception as _e:  # noqa: BLE001
        log(f"pyproj not available: {_e}")
        return
    try:
        transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    except Exception as _e:  # noqa: BLE001
        log(f"Could not build CRS transformer: {_e}")
        return
    for _node, ndata in G.nodes(data=True):
        geom = ndata.get("geometry")
        if geom is not None and not getattr(geom, "is_empty", True):
            lon, lat = transformer.transform(geom.x, geom.y)
            ndata["pos"] = (lon, lat)
            continue
        pos = ndata.get("pos")
        if pos is not None and len(pos) >= 2:
            lon, lat = transformer.transform(pos[0], pos[1])
            ndata["pos"] = (lon, lat)
            continue
        x, y = ndata.get("x"), ndata.get("y")
        if x is not None and y is not None:
            lon, lat = transformer.transform(x, y)
            ndata["pos"] = (lon, lat)


def _resolve_osmnx_parent(*candidates):
    """Pick the first candidate object that carries projection metadata."""
    for cand in candidates:
        if cand is not None and (cand.get("is_osmnx") or cand.get("is_city2graph")):
            return cand
    return None


def _prepare_morpho_inputs(buildings_obj, street_network_obj, center_lat, center_lon):
    """Shared preparation used by both routes.

    Returns a tuple ``(buildings_gdf_utm, street_gdf_utm, center_point,
    target_crs, barrier_col)``. Raises ``ValueError`` on missing data.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    buildings_gdf = utils.blender_to_geopandas(buildings_obj)
    if buildings_gdf is None or len(buildings_gdf) == 0:
        raise ValueError("Failed to extract building geometries")

    street_gdf = utils.blender_to_geopandas(street_network_obj)
    if street_gdf is None or len(street_gdf) == 0:
        raise ValueError("Failed to extract street network geometries")

    if buildings_gdf.crs is None:
        buildings_gdf.set_crs("EPSG:4326", inplace=True)
    if street_gdf.crs is None:
        street_gdf.set_crs("EPSG:4326", inplace=True)

    if buildings_gdf.crs.is_geographic:
        cb = buildings_gdf.total_bounds
        mid_lon = (cb[0] + cb[2]) / 2
        utm_zone = int((mid_lon + 180) / 6) + 1
        utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
        buildings_gdf = buildings_gdf.to_crs(utm_crs)
        street_gdf = street_gdf.to_crs(utm_crs)

    target_crs = buildings_gdf.crs

    center_point = None
    if center_lat is not None and center_lon is not None:
        center_point = (
            gpd.GeoSeries([Point(center_lon, center_lat)], crs="EPSG:4326")
            .to_crs(target_crs)
        )

    barrier_col = (
        "barrier_geometry" if "barrier_geometry" in street_gdf.columns else None
    )

    return buildings_gdf, street_gdf, center_point, target_crs, barrier_col


def _create_subset_morphological_graphs(
    buildings_obj,
    street_network_obj,
    *,
    include_priv_priv=True,
    include_pub_pub=True,
    include_priv_pub=True,
    center_lat=None,
    center_lon=None,
    distance=None,
    clipping_buffer=300.0,
    contiguity="queen",
):
    """Build the requested subset of relations as separate Blender objects.

    Used when the user disables one or more relation types in the UI.
    Each enabled relation produces an independent graph object so it
    can be styled/exported individually.
    """
    if not (include_priv_priv or include_pub_pub or include_priv_pub):
        log("No relations selected; nothing to do.")
        return []

    try:
        from city2graph.morphology import morphological_graph
    except Exception as e:  # noqa: BLE001
        log(f"city2graph imports failed: {e}")
        return []

    try:
        (
            buildings_gdf,
            street_gdf,
            center_point,
            target_crs,
            barrier_col,
        ) = _prepare_morpho_inputs(
            buildings_obj, street_network_obj, center_lat, center_lon
        )
    except ValueError as e:
        log(str(e))
        return []

    # Single call to the high-level builder mirrors the
    # city2graph notebook (morphological.ipynb / examples.txt).
    # We get back fully populated dicts of nodes and edges already
    # tagged with ('private'|'public', RELATION, 'private'|'public')
    # triples, so we just have to pick the ones the user asked for.
    try:
        nodes_dict, edges_dict = morphological_graph(
            buildings_gdf=buildings_gdf,
            segments_gdf=street_gdf,
            center_point=center_point,
            distance=distance,
            clipping_buffer=clipping_buffer if distance else float("inf"),
            primary_barrier_col=barrier_col,
            contiguity=str(contiguity).lower(),
            keep_buildings=True,
            keep_segments=True,
            as_nx=False,
        )
    except Exception as e:  # noqa: BLE001
        log(f"morphological_graph failed: {e}")
        return []

    private_gdf = nodes_dict.get("private") if isinstance(nodes_dict, dict) else None
    public_gdf = nodes_dict.get("public") if isinstance(nodes_dict, dict) else None
    if private_gdf is None and public_gdf is None:
        log("morphological_graph did not return any node GeoDataFrames")
        return []

    # Normalise edge index names exactly like the notebook does so
    # that downstream graph builders pick the right endpoints.
    pp_key = ("private", "touched_to", "private")
    PP_key = ("public", "connected_to", "public")
    pf_key = ("private", "faced_to", "public")
    if isinstance(edges_dict, dict):
        if pp_key in edges_dict and edges_dict[pp_key] is not None:
            try:
                edges_dict[pp_key].index.names = ["from_private_id", "to_private_id"]
            except Exception:  # noqa: BLE001
                pass
        if PP_key in edges_dict and edges_dict[PP_key] is not None:
            try:
                edges_dict[PP_key].index.names = ["from_public_id", "to_public_id"]
            except Exception:  # noqa: BLE001
                pass
        if pf_key in edges_dict and edges_dict[pf_key] is not None:
            try:
                edges_dict[pf_key].index.names = ["from_private_id", "to_public_id"]
            except Exception:  # noqa: BLE001
                pass

    osmnx_obj = _resolve_osmnx_parent(street_network_obj, buildings_obj)
    created = []

    def _build_homogeneous_graph(nodes_gdf, edges_gdf, id_col):
        """Build an nx.Graph from a (nodes, edges) GDF pair.

        Avoids ``as_nx=True`` paths in the c2g sub-functions which in
        practice sometimes drop edges if id columns are missing. Here
        we rely on the explicit MultiIndex set by city2graph.
        """
        import networkx as nx
        G = nx.Graph()
        # Add nodes, capturing centroid as ``geometry`` for reprojection.
        for nid, row in nodes_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid if geom.geom_type != "Point" else geom
            G.add_node(
                nid,
                geometry=centroid,
                pos=(centroid.x, centroid.y),
                kind=id_col.replace("_id", ""),
            )
        if edges_gdf is not None and len(edges_gdf) > 0:
            for (u, v), _row in edges_gdf.iterrows():
                if u in G and v in G:
                    G.add_edge(u, v)
        return G

    def _build_bipartite_graph(priv_gdf, pub_gdf, edges_gdf):
        import networkx as nx
        G = nx.Graph()
        for nid, row in priv_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            c = geom.centroid if geom.geom_type != "Point" else geom
            G.add_node(("priv", nid), geometry=c, pos=(c.x, c.y), kind="private")
        for nid, row in pub_gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            c = geom.centroid if geom.geom_type != "Point" else geom
            G.add_node(("pub", nid), geometry=c, pos=(c.x, c.y), kind="public")
        if edges_gdf is not None and len(edges_gdf) > 0:
            for (u, v), _row in edges_gdf.iterrows():
                a, b = ("priv", u), ("pub", v)
                if a in G and b in G:
                    G.add_edge(a, b)
        return G

    def _spawn(graph, name, relation_tag):
        if graph is None or graph.number_of_nodes() == 0:
            log(f"{name}: empty graph, skipping")
            return
        _reproject_graph_pos_to_wgs84(graph, target_crs)
        log(
            f"{name}: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        obj = create_graph_from_networkx(
            graph, name=name, use_positions=True, osmnx_obj=osmnx_obj,
        )
        if obj:
            obj["is_morphological_graph"] = True
            obj["morpho_relation"] = relation_tag
            created.append(obj)

    if include_priv_priv and private_gdf is not None:
        edges_pp = edges_dict.get(pp_key) if isinstance(edges_dict, dict) else None
        try:
            g = _build_homogeneous_graph(private_gdf, edges_pp, "private_id")
            _spawn(g, "Morpho_Private_Private", "private_private")
        except Exception as e:  # noqa: BLE001
            log(f"private↔private build failed: {e}")

    if include_pub_pub and public_gdf is not None:
        edges_PP = edges_dict.get(PP_key) if isinstance(edges_dict, dict) else None
        try:
            g = _build_homogeneous_graph(public_gdf, edges_PP, "public_id")
            _spawn(g, "Morpho_Public_Public", "public_public")
        except Exception as e:  # noqa: BLE001
            log(f"public↔public build failed: {e}")

    if include_priv_pub and private_gdf is not None and public_gdf is not None:
        edges_pf = edges_dict.get(pf_key) if isinstance(edges_dict, dict) else None
        try:
            g = _build_bipartite_graph(private_gdf, public_gdf, edges_pf)
            _spawn(g, "Morpho_Private_Public", "private_public")
        except Exception as e:  # noqa: BLE001
            log(f"private↔public build failed: {e}")

    log(f"Subset morphological graphs created: {len(created)}")
    return created


def create_morphological_graph(
    buildings_obj,
    street_network_obj,
    *,
    center_lat=None,
    center_lon=None,
    distance=None,
    clipping_buffer=300.0,
    contiguity="queen",
    keep_buildings=True,
    keep_segments=True,
    include_priv_priv=True,
    include_pub_pub=True,
    include_priv_pub=True,
):
    """Create the morphological graph(s) requested by the user.

    When all three relation flags are True, this calls the high-level
    ``city2graph.morphology.morphological_graph`` and returns a single
    heterogeneous Blender object (faster, mirrors the c2g notebook).
    When any flag is False, falls back to per-relation calls and
    returns a **list** of Blender objects (one per active relation).

    Args:
        buildings_obj: Blender object with building polygons (private space).
        street_network_obj: Blender object with street segments (public space).
        center_lat, center_lon: Optional WGS84 centre for distance filtering.
        distance: Analysis radius in metres. If None or 0, uses the
            full extent of the input.
        clipping_buffer: Buffer in metres for clean tessellation borders.
        contiguity: 'queen' or 'rook'.
        keep_buildings, keep_segments: Mirror the c2g flags.
        include_priv_priv, include_pub_pub, include_priv_pub: enable
            each relation type individually.

    Returns:
        Single Blender object (full graph) or list of Blender objects
        (subset mode), or None on failure.
    """
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None

    if not (include_priv_priv and include_pub_pub and include_priv_pub):
        return _create_subset_morphological_graphs(
            buildings_obj,
            street_network_obj,
            include_priv_priv=include_priv_priv,
            include_pub_pub=include_pub_pub,
            include_priv_pub=include_priv_pub,
            center_lat=center_lat,
            center_lon=center_lon,
            distance=distance,
            clipping_buffer=clipping_buffer,
            contiguity=contiguity,
        )

    try:
        from city2graph.morphology import morphological_graph

        (
            buildings_gdf,
            street_gdf,
            center_point,
            target_crs,
            barrier_col,
        ) = _prepare_morpho_inputs(
            buildings_obj, street_network_obj, center_lat, center_lon
        )

        log(
            "Creating morphological graph: "
            f"{len(buildings_gdf)} buildings, {len(street_gdf)} segments, "
            f"distance={distance}, clipping_buffer={clipping_buffer}, "
            f"contiguity={contiguity}"
        )

        morph_graph = morphological_graph(
            buildings_gdf=buildings_gdf,
            segments_gdf=street_gdf,
            center_point=center_point,
            distance=distance,
            clipping_buffer=clipping_buffer if distance else float("inf"),
            primary_barrier_col=barrier_col,
            contiguity=str(contiguity).lower(),
            keep_buildings=keep_buildings,
            keep_segments=keep_segments,
            as_nx=True,
        )

        if morph_graph is None or morph_graph.number_of_nodes() == 0:
            log("Morphological graph creation failed or returned empty graph")
            return None

        log(
            "Created morphological graph: "
            f"{morph_graph.number_of_nodes()} nodes, "
            f"{morph_graph.number_of_edges()} edges"
        )

        _reproject_graph_pos_to_wgs84(morph_graph, target_crs)

        osmnx_obj = _resolve_osmnx_parent(street_network_obj, buildings_obj)

        graph_obj = create_graph_from_networkx(
            morph_graph,
            name="Morphological_Graph",
            use_positions=True,
            osmnx_obj=osmnx_obj,
        )

        if graph_obj:
            graph_obj["is_morphological_graph"] = True
            graph_obj["morpho_relation"] = "all"
            log("Morphological graph object created successfully")

        return graph_obj

    except Exception as e:
        log(f"Error creating morphological graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_graph_from_networkx(G, name="NetworkX_Graph", use_positions=True, osmnx_obj=None):
    """
    Create Blender graph object from NetworkX graph.
    
    Args:
        G: NetworkX graph
        name: Name for the Blender object
        use_positions: Use 'pos' attribute from nodes if available
        osmnx_obj: Optional OSMnx object for coordinate alignment
    
    Returns:
        Blender object
    """
    import bpy
    import bmesh
    
    if G is None or G.number_of_nodes() == 0:
        log("Empty graph provided")
        return None
    
    bm = bmesh.new()
    node_to_vert = {}
    
    center_lat = None
    center_lon = None
    scale = 0.001
    
    if osmnx_obj is not None:
        center_lat = osmnx_obj.get("osmnx_center_lat") or osmnx_obj.get("c2g_center_lat")
        center_lon = osmnx_obj.get("osmnx_center_lon") or osmnx_obj.get("c2g_center_lon")
        scale = osmnx_obj.get("osmnx_scale") or osmnx_obj.get("c2g_scale", 0.001)
    
    from ..mesh.geometry import _latlon_to_local_3d
    
    verts_created = 0
    for node in G.nodes():
        if use_positions and 'pos' in G.nodes[node]:
            pos = G.nodes[node]['pos']
            if len(pos) == 2:
                if center_lat is not None and center_lon is not None:
                    lon, lat = pos
                    pos = _latlon_to_local_3d(lat, lon, center_lat, center_lon, scale)
                else:
                    pos = (pos[0], pos[1], 0)
            elif len(pos) == 3:
                pos = tuple(pos)
            else:
                pos = (0, 0, 0)
        else:
            pos = (0, 0, 0)
        
        vert = bm.verts.new(pos)
        node_to_vert[node] = vert
        verts_created += 1
    
    log(f"Created {verts_created} vertices in Blender mesh from {G.number_of_nodes()} graph nodes")
    
    bm.verts.ensure_lookup_table()
    
    edges_created = 0
    edges_failed = 0
    for u, v in G.edges():
        if u in node_to_vert and v in node_to_vert:
            try:
                bm.edges.new([node_to_vert[u], node_to_vert[v]])
                edges_created += 1
            except ValueError as e:
                edges_failed += 1
    
    if edges_failed > 0:
        log(f"Warning: Failed to create {edges_failed} edges in Blender (likely duplicates or same vertex)")
    log(f"Created {edges_created} edges in Blender mesh")
    
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(mesh)
    bm.free()
    
    obj = bpy.data.objects.new(name, mesh)
    
    collection = utils.create_collection("C2G_Morphology")
    collection.objects.link(obj)
    
    obj["is_city2graph"] = True
    obj["num_nodes"] = G.number_of_nodes()
    obj["num_edges"] = G.number_of_edges()
    
    if center_lat is not None:
        obj["c2g_center_lat"] = center_lat
        obj["c2g_center_lon"] = center_lon
        obj["c2g_scale"] = scale
    
    log(f"Created graph object: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return obj

