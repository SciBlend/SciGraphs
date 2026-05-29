import bpy
from ...utils.logger import log
from .get_c2g import get_city2graph
from . import utils


# Module-level cache for the active GTFS connection.
# Blender id-properties only accept primitive types, so the actual
# DuckDBPyConnection cannot live on the scene. We keep it here and
# expose only serialisable metadata via ``set_active_gtfs``.
_GTFS_CACHE = {"connection": None, "filepath": None}


def get_active_gtfs():
    """Return the currently loaded DuckDB GTFS connection, or None."""
    return _GTFS_CACHE.get("connection")


def set_active_gtfs(connection, filepath=None):
    """Register a freshly loaded GTFS connection in the cache."""
    _GTFS_CACHE["connection"] = connection
    _GTFS_CACHE["filepath"] = filepath


def clear_active_gtfs():
    """Drop the cached connection (called on add-on unregister)."""
    conn = _GTFS_CACHE.get("connection")
    if conn is not None:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    _GTFS_CACHE["connection"] = None
    _GTFS_CACHE["filepath"] = None


def load_gtfs(filepath):
    """
    Load GTFS (General Transit Feed Specification) data from zip file.
    
    In city2graph 0.3.1+, this returns a DuckDB connection with GTFS tables loaded.
    
    Args:
        filepath: Path to GTFS .zip file
    
    Returns:
        duckdb.DuckDBPyConnection: DuckDB connection with GTFS tables
    """
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.transportation import load_gtfs as c2g_load_gtfs
        import os
        import glob
        import tempfile
        import zipfile

        log(f"Loading GTFS data from {filepath}")

        path = filepath
        cleanup_path = None

        # city2graph.load_gtfs only accepts a .zip path. If the user
        # picked an extracted GTFS folder, zip it on the fly into a
        # temp file (lighter than re-implementing the loader). Plain
        # .txt files are detected by GTFS spec table names.
        if os.path.isdir(filepath):
            txt_files = sorted(glob.glob(os.path.join(filepath, "*.txt")))
            if not txt_files:
                log(f"GTFS folder has no .txt files: {filepath}")
                return None
            tmp = tempfile.NamedTemporaryFile(
                prefix="scigraphs_gtfs_", suffix=".zip", delete=False
            )
            tmp.close()
            cleanup_path = tmp.name
            log(
                f"Repackaging {len(txt_files)} GTFS .txt files into "
                f"{cleanup_path}"
            )
            with zipfile.ZipFile(cleanup_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in txt_files:
                    zf.write(f, arcname=os.path.basename(f))
            path = cleanup_path

        gtfs_con = c2g_load_gtfs(path)

        # Best-effort cleanup of the temporary zip — DuckDB has already
        # loaded the data into memory by this point.
        if cleanup_path is not None:
            try:
                os.unlink(cleanup_path)
            except OSError:
                pass

        if gtfs_con is None:
            log("Failed to load GTFS data")
            return None

        tables = {row[0] for row in gtfs_con.execute("SHOW TABLES").fetchall()}
        log(f"GTFS loaded with tables: {', '.join(sorted(tables))}")

        # Sanity check: an empty GTFS connection is almost certainly
        # an upstream silent failure (bad zip, wrong path, etc.). Bail
        # out cleanly so the caller doesn't end up with a useless
        # connection.
        required = {"stops", "trips", "stop_times"}
        if not required.issubset(tables):
            missing = sorted(required - tables)
            log(
                "GTFS feed is incomplete or could not be parsed. "
                f"Missing required tables: {', '.join(missing)}"
            )
            try:
                gtfs_con.close()
            except Exception:  # noqa: BLE001
                pass
            return None

        if 'stops' in tables:
            count = gtfs_con.execute("SELECT COUNT(*) FROM stops").fetchone()[0]
            log(f"Stops: {count}")
        if 'routes' in tables:
            count = gtfs_con.execute("SELECT COUNT(*) FROM routes").fetchone()[0]
            log(f"Routes: {count}")

        return gtfs_con

    except Exception as e:
        log(f"Error loading GTFS data: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_gtfs_graph(
    gtfs_con,
    osmnx_obj=None,
    *,
    name="GTFS_Network",
    collection_name="C2G_Transportation",
):
    """Build a SciGraphs-canonical Blender object from a GTFS feed.

    This produces ONE mesh object with:
      * one vertex per GTFS stop (mesh attribute ``node_id`` for SciGraphs lookup);
      * one edge per directly-connected pair of stops in the
        ``travel_summary_graph`` (i.e. consecutive stops served by the
        same trip);
      * ``edge_*`` mesh attributes for every numeric column the
        travel-summary graph exposes (``trip_count``, ``mean_duration``,
        ``frequency``, etc.) — applied via SciGraphs' standard
        ``import_attributes_from_dataframe`` so all downstream
        analysis tools (Coloring Gizmo, Setup Visualization, GNN
        export…) recognise them automatically;
      * the SciGraphs marker custom properties (``is_scigraphs_graph``,
        ``num_nodes``, ``num_edges``, ``nodes_data``, ``edges_data``,
        ``is_directed``, ``node_positions``);
      * geographic alignment (``c2g_center_lat/lon/scale`` or, when
        ``osmnx_obj`` is provided, the matching ``osmnx_*`` values) so
        the stops land on top of the road network.

    Returns the created object, or ``None`` on failure.
    """
    if gtfs_con is None:
        log("No GTFS connection available")
        return None

    try:
        from city2graph.transportation import travel_summary_graph
        from ..algorithms.graph import GraphData
        from ..mesh.geometry import create_graph_object
        import math
        import numpy as np
        import pandas as pd

        tables = {row[0] for row in gtfs_con.execute("SHOW TABLES").fetchall()}
        if 'stops' not in tables:
            log("No stops table in GTFS data")
            return None

        stops_df = gtfs_con.execute(
            "SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops"
        ).df()

        stops_df = stops_df.dropna(subset=['stop_lat', 'stop_lon'])
        stops_df = stops_df[
            (stops_df['stop_lat'] != 0) | (stops_df['stop_lon'] != 0)
        ]
        if stops_df.empty:
            log("No valid stop coordinates found")
            return None
        stops_df['stop_id'] = stops_df['stop_id'].astype(str)

        # Travel summary graph — gives us the edges between stops.
        log("Computing travel summary graph for GTFS edges (this may take a few seconds)...")
        try:
            nodes_gdf, edges_gdf = travel_summary_graph(gtfs_con, as_nx=False)
        except Exception as _e:  # noqa: BLE001
            log(f"travel_summary_graph failed: {_e}")
            edges_gdf = None
            nodes_gdf = None

        # Some feeds expose stop_lat/stop_lon already in the summary
        # node table — prefer those when present, they match the IDs
        # used by the edges' MultiIndex.
        if nodes_gdf is not None and len(nodes_gdf) > 0:
            try:
                ng = nodes_gdf.reset_index()
                if 'stop_lat' in ng.columns and 'stop_lon' in ng.columns:
                    sid_col = (
                        'stop_id' if 'stop_id' in ng.columns
                        else ng.columns[0]
                    )
                    ng = ng.rename(columns={sid_col: 'stop_id'})
                    ng['stop_id'] = ng['stop_id'].astype(str)
                    extra = ng[['stop_id', 'stop_lat', 'stop_lon']].dropna()
                    if not extra.empty:
                        # Replace stops_df with the canonical list used
                        # by the edges so indices line up.
                        stops_df = extra.merge(
                            stops_df[['stop_id', 'stop_name']],
                            on='stop_id',
                            how='left',
                        )
            except Exception:  # noqa: BLE001
                pass

        # Resolve projection (matches the convention used elsewhere
        # in the addon: equirectangular around a chosen centre).
        center_lat = None
        center_lon = None
        scale = 0.001
        using_osmnx = bool(osmnx_obj is not None and osmnx_obj.get("is_osmnx", False))
        if using_osmnx:
            center_lat = float(osmnx_obj.get("osmnx_center_lat"))
            center_lon = float(osmnx_obj.get("osmnx_center_lon"))
            scale = float(osmnx_obj.get("osmnx_scale", 0.001))
            log(
                f"GTFS aligned to OSMnx graph '{osmnx_obj.name}': "
                f"center=({center_lat:.6f}, {center_lon:.6f}), scale={scale}"
            )
        else:
            lats = stops_df['stop_lat'].to_numpy(dtype=float)
            lons = stops_df['stop_lon'].to_numpy(dtype=float)
            center_lat = float((lats.min() + lats.max()) / 2)
            center_lon = float((lons.min() + lons.max()) / 2)
            log(
                f"GTFS using own bbox centre: "
                f"({center_lat:.6f}, {center_lon:.6f})"
            )

        cos_lat = math.cos(math.radians(center_lat))

        # Build node ordering that matches the SciGraphs convention.
        nodes = stops_df['stop_id'].tolist()
        node_to_idx = {n: i for i, n in enumerate(nodes)}

        # Per-node coordinates: SciGraphs stores them so the geometry
        # nodes setup can read them back without re-projecting.
        lats = stops_df['stop_lat'].to_numpy(dtype=float)
        lons = stops_df['stop_lon'].to_numpy(dtype=float)
        xs = (lons - center_lon) * cos_lat / scale
        ys = (lats - center_lat) / scale
        zs = np.zeros_like(xs)
        node_coordinates = {
            nid: (float(x), float(y), float(z))
            for nid, x, y, z in zip(nodes, xs, ys, zs)
        }

        # Build edges + per-edge dataframe so SciGraphs' standard
        # attribute importer can ingest the GTFS-specific weights.
        edges = []
        if edges_gdf is not None and len(edges_gdf) > 0:
            log(f"Building {len(edges_gdf)} GTFS edges from travel summary")
            ed = edges_gdf.reset_index()
            # The MultiIndex of travel_summary_graph is (from_stop_id, to_stop_id).
            src_col = (
                'from_stop_id' if 'from_stop_id' in ed.columns
                else ed.columns[0]
            )
            tgt_col = (
                'to_stop_id' if 'to_stop_id' in ed.columns
                else ed.columns[1]
            )
            ed[src_col] = ed[src_col].astype(str)
            ed[tgt_col] = ed[tgt_col].astype(str)
            # Keep only edges whose endpoints exist in the stops table
            # (defensive: some GTFS feeds reference dropped stop_ids).
            mask = ed[src_col].isin(node_to_idx) & ed[tgt_col].isin(node_to_idx)
            ed = ed.loc[mask].reset_index(drop=True)

            # Rename src/tgt to the SciGraphs convention so the
            # attribute importer skips them automatically.
            ed = ed.rename(columns={src_col: 'source', tgt_col: 'target'})
            # Drop the geometry column (objects with non-scalar values
            # break attribute import).
            if 'geometry' in ed.columns:
                ed = ed.drop(columns=['geometry'])

            edges = list(zip(ed['source'].tolist(), ed['target'].tolist()))
            edge_df = ed
        else:
            log("No travel-summary edges; building stops-only graph")
            edge_df = pd.DataFrame({'source': [], 'target': []})

        graph_data = GraphData(nodes, edges, edge_df)
        graph_data.node_coordinates = node_coordinates
        graph_data.is_directed = False
        graph_data.source_column_name = 'source'
        graph_data.target_column_name = 'target'

        # Wipe any previous GTFS object with the same base name so
        # repeated runs don't pile up duplicates.
        for existing in list(bpy.data.objects):
            if existing.name == name or existing.get("is_gtfs_graph") and existing.name.startswith(name):
                try:
                    bpy.data.objects.remove(existing, do_unlink=True)
                except Exception:  # noqa: BLE001
                    pass

        obj = create_graph_object(
            graph_data,
            is_directed=False,
            selected_attributes=None,  # import every numeric column
            remove_self_loops=True,
        )
        if obj is None:
            log("create_graph_object returned None")
            return None

        # Move the object to the C2G collection so the user can find
        # it next to the rest of the city2graph outputs.
        try:
            target_coll = utils.create_collection(collection_name)
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            target_coll.objects.link(obj)
        except Exception:  # noqa: BLE001
            pass

        obj.name = name

        # Mark as a GTFS graph + propagate alignment so basemaps and
        # spatial joins downstream snap to the same projection.
        obj["is_scigraphs_graph"] = True
        obj["is_gtfs_graph"] = True
        obj["c2g_center_lat"] = center_lat
        obj["c2g_center_lon"] = center_lon
        obj["c2g_scale"] = scale
        if using_osmnx:
            obj["osmnx_center_lat"] = center_lat
            obj["osmnx_center_lon"] = center_lon
            obj["osmnx_scale"] = scale
            obj["aligned_with_osmnx"] = osmnx_obj.name

        log(
            f"GTFS graph created: {len(nodes)} stops, {len(edges)} edges, "
            f"{len(edge_df.columns) - 2} edge attributes"
        )
        return obj

    except Exception as e:
        log(f"Error building GTFS graph: {e}")
        import traceback
        traceback.print_exc()
        return None


# Backwards-compatible thin wrappers used by older operators / panels.
def visualize_gtfs_stops(gtfs_con, osmnx_obj=None, size=0.05):  # noqa: ARG001
    obj = build_gtfs_graph(gtfs_con, osmnx_obj=osmnx_obj, name="GTFS_Stops")
    return [obj] if obj else []


def visualize_gtfs_routes(gtfs_con, osmnx_obj=None):
    obj = build_gtfs_graph(gtfs_con, osmnx_obj=osmnx_obj, name="GTFS_Routes")
    return [obj] if obj else []


def create_travel_summary_graph(gtfs_con, start_time=None, end_time=None,
                                calendar_start=None, calendar_end=None,
                                osmnx_obj=None):
    """Build the SciGraphs canonical Blender graph for the travel summary.

    Same engine as :func:`build_gtfs_graph` (so the result is a real
    SciGraphs graph with proper attributes), but the underlying
    travel_summary_graph call is parameterised by time / calendar
    filters from the panel.
    """
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None

    try:
        from city2graph.transportation import travel_summary_graph
        from ..algorithms.graph import GraphData
        from ..mesh.geometry import create_graph_object
        import math
        import numpy as np

        log(
            "Creating travel summary graph "
            f"(start={start_time}, end={end_time}, "
            f"cal_start={calendar_start}, cal_end={calendar_end})..."
        )
        nodes_gdf, edges_gdf = travel_summary_graph(
            gtfs_con,
            start_time=start_time,
            end_time=end_time,
            calendar_start=calendar_start,
            calendar_end=calendar_end,
            as_nx=False,
            directed=False,
        )
        if nodes_gdf is None or len(nodes_gdf) == 0:
            log("travel_summary_graph returned no nodes")
            return None

        # Resolve projection.
        # Anchor priority (same rule as build_gtfs_od_graph):
        # OSMnx → existing GTFS object → own bbox.
        center_lat = None
        center_lon = None
        scale = 0.001
        if osmnx_obj is not None and osmnx_obj.get("is_osmnx", False):
            center_lat = float(osmnx_obj.get("osmnx_center_lat"))
            center_lon = float(osmnx_obj.get("osmnx_center_lon"))
            scale = float(osmnx_obj.get("osmnx_scale", 0.001))
        if center_lat is None:
            for cand in bpy.data.objects:
                if not cand.get("is_gtfs_graph"):
                    continue
                clat = cand.get("c2g_center_lat")
                clon = cand.get("c2g_center_lon")
                csc = cand.get("c2g_scale")
                if clat is None or clon is None or csc is None:
                    continue
                center_lat = float(clat)
                center_lon = float(clon)
                scale = float(csc)
                log(
                    f"Travel summary anchored to existing GTFS object "
                    f"'{cand.name}'"
                )
                break
        if center_lat is None:
            ng = nodes_gdf.reset_index()
            if 'stop_lat' in ng.columns and 'stop_lon' in ng.columns:
                lats = ng['stop_lat'].to_numpy(dtype=float)
                lons = ng['stop_lon'].to_numpy(dtype=float)
            else:
                lats = ng.geometry.y.to_numpy()
                lons = ng.geometry.x.to_numpy()
            center_lat = float((lats.min() + lats.max()) / 2)
            center_lon = float((lons.min() + lons.max()) / 2)

        cos_lat = math.cos(math.radians(center_lat))

        ng = nodes_gdf.reset_index()
        sid_col = 'stop_id' if 'stop_id' in ng.columns else ng.columns[0]
        ng[sid_col] = ng[sid_col].astype(str)

        if 'stop_lat' in ng.columns and 'stop_lon' in ng.columns:
            lats = ng['stop_lat'].to_numpy(dtype=float)
            lons = ng['stop_lon'].to_numpy(dtype=float)
        else:
            lats = ng.geometry.y.to_numpy()
            lons = ng.geometry.x.to_numpy()

        nodes = ng[sid_col].tolist()
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        xs = (lons - center_lon) * cos_lat / scale
        ys = (lats - center_lat) / scale
        node_coordinates = {
            nid: (float(x), float(y), 0.0)
            for nid, x, y in zip(nodes, xs, ys)
        }

        if edges_gdf is None or len(edges_gdf) == 0:
            edges = []
            edge_df = None
        else:
            ed = edges_gdf.reset_index()
            src_col = (
                'from_stop_id' if 'from_stop_id' in ed.columns else ed.columns[0]
            )
            tgt_col = (
                'to_stop_id' if 'to_stop_id' in ed.columns else ed.columns[1]
            )
            ed[src_col] = ed[src_col].astype(str)
            ed[tgt_col] = ed[tgt_col].astype(str)
            mask = ed[src_col].isin(node_to_idx) & ed[tgt_col].isin(node_to_idx)
            ed = ed.loc[mask].reset_index(drop=True)
            ed = ed.rename(columns={src_col: 'source', tgt_col: 'target'})
            if 'geometry' in ed.columns:
                ed = ed.drop(columns=['geometry'])
            edges = list(zip(ed['source'].tolist(), ed['target'].tolist()))
            edge_df = ed

        graph_data = GraphData(nodes, edges, edge_df)
        graph_data.node_coordinates = node_coordinates
        graph_data.is_directed = False
        if edge_df is not None:
            graph_data.source_column_name = 'source'
            graph_data.target_column_name = 'target'

        for existing in list(bpy.data.objects):
            if existing.name == "Travel_Summary_Graph":
                bpy.data.objects.remove(existing, do_unlink=True)

        graph_obj = create_graph_object(
            graph_data,
            is_directed=False,
            selected_attributes=None,
            remove_self_loops=True,
        )
        if graph_obj is None:
            return None

        try:
            target_coll = utils.create_collection("C2G_Transportation")
            for c in list(graph_obj.users_collection):
                c.objects.unlink(graph_obj)
            target_coll.objects.link(graph_obj)
        except Exception:  # noqa: BLE001
            pass

        graph_obj.name = "Travel_Summary_Graph"
        graph_obj["is_scigraphs_graph"] = True
        graph_obj["is_gtfs_graph"] = True
        graph_obj["is_travel_graph"] = True
        graph_obj["c2g_center_lat"] = center_lat
        graph_obj["c2g_center_lon"] = center_lon
        graph_obj["c2g_scale"] = scale
        if start_time:
            graph_obj["time_start"] = start_time
        if end_time:
            graph_obj["time_end"] = end_time
        if calendar_start:
            graph_obj["calendar_start"] = calendar_start
        if calendar_end:
            graph_obj["calendar_end"] = calendar_end

        log(
            f"Travel summary graph: {len(nodes)} stops, {len(edges)} edges, "
            f"{0 if edge_df is None else len(edge_df.columns) - 2} edge attrs"
        )
        return graph_obj

    except Exception as e:
        log(f"Error creating travel summary graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def build_gtfs_od_graph(
    gtfs_con,
    *,
    osmnx_obj=None,
    start_date=None,
    end_date=None,
    directed=False,
    top_n=None,
    name="GTFS_OD_Graph",
    collection_name="C2G_Transportation",
):
    """Build a SciGraphs graph from GTFS Origin-Destination pairs.

    The raw output of ``c2g.get_od_pairs`` is a row per **trip leg**, which
    for a metropolitan feed can easily reach millions of rows and OOM-kill
    the Blender process when materialised. We aggregate the pairs into
    one row per (orig, dest) — counting trips and averaging travel
    time — and optionally truncate to the ``top_n`` heaviest pairs.

    The result is a single SciGraphs canonical mesh object:
      * vertices = stops referenced by any OD pair (positioned via
        ``stop_lat``/``stop_lon`` from the GTFS ``stops`` table).
      * edges = aggregated OD pairs, with ``edge_trip_count`` and
        ``edge_travel_time_sec`` mesh attributes.
    """
    if gtfs_con is None:
        log("No GTFS connection available")
        return None

    try:
        from ..algorithms.graph import GraphData
        from ..mesh.geometry import create_graph_object
        import math
        import pandas as pd  # noqa: F401  (used implicitly by DuckDB .df())

        log(
            "Aggregating OD pairs in DuckDB "
            f"(start_date={start_date}, end_date={end_date}, directed={directed})..."
        )

        # We deliberately bypass city2graph.get_od_pairs because it
        # materialises one row per active leg in pandas — for big
        # feeds (Sao Paulo: 22k stops × 1.3k routes × N service days)
        # that is tens of millions of rows and OOM-kills Blender. The
        # query below performs the leg construction AND aggregation
        # entirely inside DuckDB, so only the (orig, dest) summary
        # crosses the Python boundary.

        tables = {row[0] for row in gtfs_con.execute("SHOW TABLES").fetchall()}
        if not {'stop_times', 'trips', 'stops'}.issubset(tables):
            log("GTFS feed missing stop_times/trips/stops")
            return None

        # Build a daily-active-trips CTE if calendar is available, so
        # the trip_count weights actual service days instead of just
        # the schedule pattern. Otherwise fall back to a single count
        # per scheduled trip.
        has_calendar = 'calendar' in tables

        # Note: ``arrival_time`` / ``departure_time`` in GTFS may
        # exceed 24:00:00 (overnight services). Splitting on ':' and
        # using simple integer arithmetic keeps the aggregation
        # SQL-only and avoids type errors with non-canonical times.
        # NULLs are propagated and filtered later.
        select_legs = """
            WITH legs AS (
                SELECT
                    st1.trip_id,
                    st1.stop_id AS orig_stop_id,
                    st2.stop_id AS dest_stop_id,
                    -- Travel time in seconds, parsed manually because
                    -- the strings can wrap past 24h.
                    (
                        TRY_CAST(SPLIT_PART(st2.arrival_time, ':', 1) AS BIGINT) * 3600
                      + TRY_CAST(SPLIT_PART(st2.arrival_time, ':', 2) AS BIGINT) * 60
                      + TRY_CAST(SPLIT_PART(st2.arrival_time, ':', 3) AS BIGINT)
                    )
                    -
                    (
                        TRY_CAST(SPLIT_PART(st1.departure_time, ':', 1) AS BIGINT) * 3600
                      + TRY_CAST(SPLIT_PART(st1.departure_time, ':', 2) AS BIGINT) * 60
                      + TRY_CAST(SPLIT_PART(st1.departure_time, ':', 3) AS BIGINT)
                    )
                    AS travel_time_sec
                FROM stop_times st1
                JOIN stop_times st2
                  ON st1.trip_id = st2.trip_id
                 -- ``stop_sequence`` is sometimes loaded as VARCHAR
                 -- (e.g. when the GTFS file has quoted integers), so
                 -- cast both sides explicitly to keep the join valid.
                 AND CAST(st2.stop_sequence AS BIGINT)
                       = CAST(st1.stop_sequence AS BIGINT) + 1
            )
        """

        if has_calendar:
            # Service multiplier = number of active service days in
            # the (optional) date range across all 7 weekday columns.
            # If the user passed start/end_date we restrict the range;
            # otherwise we use the calendar bounds.
            params = {}
            date_range_sql = ""
            if start_date or end_date:
                date_range_sql = " WHERE 1=1"
                if start_date:
                    date_range_sql += " AND end_date >= ?"
                    params['start_date'] = start_date
                if end_date:
                    date_range_sql += " AND start_date <= ?"
                    params['end_date'] = end_date
            multiplier_sql = f"""
                services AS (
                    SELECT
                        service_id,
                        (CAST(monday AS INT) + CAST(tuesday AS INT)
                       + CAST(wednesday AS INT) + CAST(thursday AS INT)
                       + CAST(friday AS INT) + CAST(saturday AS INT)
                       + CAST(sunday AS INT)) AS service_weight
                    FROM calendar
                    {date_range_sql}
                )
            """
            count_expr = "SUM(COALESCE(s.service_weight, 1))"
            join_services = (
                "LEFT JOIN services s ON t.service_id = s.service_id"
            )
            param_values = [v for v in (
                params.get('start_date'), params.get('end_date')
            ) if v is not None]
        else:
            multiplier_sql = ""
            count_expr = "COUNT(*)"
            join_services = ""
            param_values = []

        if directed:
            pair_select = "l.orig_stop_id AS source, l.dest_stop_id AS target"
        else:
            # Canonicalise so (A,B) and (B,A) merge.
            pair_select = (
                "LEAST(l.orig_stop_id, l.dest_stop_id) AS source, "
                "GREATEST(l.orig_stop_id, l.dest_stop_id) AS target"
            )

        order_clause = ""
        if top_n is not None and top_n > 0:
            order_clause = f" ORDER BY trip_count DESC LIMIT {int(top_n)}"

        if multiplier_sql:
            full_sql = f"""
                {select_legs},
                {multiplier_sql}
                SELECT
                    {pair_select},
                    {count_expr} AS trip_count,
                    AVG(l.travel_time_sec) AS travel_time_sec
                FROM legs l
                JOIN trips t ON l.trip_id = t.trip_id
                {join_services}
                WHERE l.travel_time_sec IS NOT NULL
                  AND l.travel_time_sec >= 0
                GROUP BY source, target
                {order_clause}
            """
        else:
            full_sql = f"""
                {select_legs}
                SELECT
                    {pair_select},
                    {count_expr} AS trip_count,
                    AVG(l.travel_time_sec) AS travel_time_sec
                FROM legs l
                JOIN trips t ON l.trip_id = t.trip_id
                WHERE l.travel_time_sec IS NOT NULL
                  AND l.travel_time_sec >= 0
                GROUP BY source, target
                {order_clause}
            """

        agg = gtfs_con.execute(full_sql, param_values).df()
        log(f"Aggregated OD pairs: {len(agg)} rows")
        if agg.empty:
            log("No OD pairs after aggregation")
            return None

        agg['source'] = agg['source'].astype(str)
        agg['target'] = agg['target'].astype(str)

        # Look up stop coordinates for every node referenced by the
        # surviving OD pairs.
        used_ids = set(agg['source']).union(agg['target'])
        if not used_ids:
            log("No stops referenced after aggregation")
            return None
        stops_df = gtfs_con.execute(
            "SELECT stop_id, stop_name, stop_lat, stop_lon FROM stops"
        ).df()
        stops_df['stop_id'] = stops_df['stop_id'].astype(str)
        stops_df = stops_df[stops_df['stop_id'].isin(used_ids)]
        stops_df = stops_df.dropna(subset=['stop_lat', 'stop_lon'])
        if stops_df.empty:
            log("None of the OD-pair stops have valid coordinates")
            return None

        # Resolve projection.
        # Priority of anchor:
        #   1. Explicit OSMnx graph passed in.
        #   2. Any existing GTFS graph in scene with matching feed
        #      path — keeps the OD graph aligned with GTFS_Network /
        #      Travel_Summary_Graph so they overlay perfectly.
        #   3. Otherwise compute the bbox centre of the OD-pair stops.
        center_lat = None
        center_lon = None
        scale = 0.001
        using_osmnx = False
        anchor_obj = None

        if osmnx_obj is not None and osmnx_obj.get("is_osmnx", False):
            center_lat = float(osmnx_obj.get("osmnx_center_lat"))
            center_lon = float(osmnx_obj.get("osmnx_center_lon"))
            scale = float(osmnx_obj.get("osmnx_scale", 0.001))
            using_osmnx = True
            anchor_obj = osmnx_obj

        if center_lat is None:
            for cand in bpy.data.objects:
                if not cand.get("is_gtfs_graph"):
                    continue
                clat = cand.get("c2g_center_lat")
                clon = cand.get("c2g_center_lon")
                csc = cand.get("c2g_scale")
                if clat is None or clon is None or csc is None:
                    continue
                center_lat = float(clat)
                center_lon = float(clon)
                scale = float(csc)
                anchor_obj = cand
                log(
                    f"OD graph anchored to existing GTFS object "
                    f"'{cand.name}': center=({center_lat:.6f}, {center_lon:.6f}), "
                    f"scale={scale}"
                )
                break

        if center_lat is None:
            lats = stops_df['stop_lat'].to_numpy(dtype=float)
            lons = stops_df['stop_lon'].to_numpy(dtype=float)
            center_lat = float((lats.min() + lats.max()) / 2)
            center_lon = float((lons.min() + lons.max()) / 2)
            log(
                f"OD graph using own bbox centre: "
                f"({center_lat:.6f}, {center_lon:.6f})"
            )

        cos_lat = math.cos(math.radians(center_lat))
        nodes = stops_df['stop_id'].tolist()
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        lats = stops_df['stop_lat'].to_numpy(dtype=float)
        lons = stops_df['stop_lon'].to_numpy(dtype=float)
        xs = (lons - center_lon) * cos_lat / scale
        ys = (lats - center_lat) / scale
        node_coordinates = {
            nid: (float(x), float(y), 0.0)
            for nid, x, y in zip(nodes, xs, ys)
        }

        # Drop OD edges whose endpoints fell out for lack of coords.
        mask = agg['source'].isin(node_to_idx) & agg['target'].isin(node_to_idx)
        agg = agg.loc[mask].reset_index(drop=True)
        if agg.empty:
            log("All OD pairs lost their endpoints — nothing to build")
            return None

        edges = list(zip(agg['source'].tolist(), agg['target'].tolist()))

        graph_data = GraphData(nodes, edges, agg)
        graph_data.node_coordinates = node_coordinates
        graph_data.is_directed = bool(directed)
        graph_data.source_column_name = 'source'
        graph_data.target_column_name = 'target'

        for existing in list(bpy.data.objects):
            if existing.name == name:
                bpy.data.objects.remove(existing, do_unlink=True)

        graph_obj = create_graph_object(
            graph_data,
            is_directed=bool(directed),
            selected_attributes=None,
            remove_self_loops=True,
        )
        if graph_obj is None:
            return None

        try:
            target_coll = utils.create_collection(collection_name)
            for c in list(graph_obj.users_collection):
                c.objects.unlink(graph_obj)
            target_coll.objects.link(graph_obj)
        except Exception:  # noqa: BLE001
            pass

        graph_obj.name = name
        graph_obj["is_scigraphs_graph"] = True
        graph_obj["is_gtfs_graph"] = True
        graph_obj["is_od_graph"] = True
        graph_obj["c2g_center_lat"] = center_lat
        graph_obj["c2g_center_lon"] = center_lon
        graph_obj["c2g_scale"] = scale
        if using_osmnx and osmnx_obj is not None:
            graph_obj["osmnx_center_lat"] = center_lat
            graph_obj["osmnx_center_lon"] = center_lon
            graph_obj["osmnx_scale"] = scale
            graph_obj["aligned_with_osmnx"] = osmnx_obj.name
        elif anchor_obj is not None:
            graph_obj["aligned_with"] = anchor_obj.name
        if start_date:
            graph_obj["calendar_start"] = start_date
        if end_date:
            graph_obj["calendar_end"] = end_date

        log(
            f"OD graph created: {len(nodes)} stops, {len(edges)} pairs, "
            "attrs: edge_trip_count, edge_travel_time_sec"
        )
        return graph_obj

    except Exception as e:
        log(f"Error building OD graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_gtfs_od_pairs(gtfs_con, start_date=None, end_date=None, directed=False):
    """
    Get Origin-Destination pairs from GTFS data.
    
    New in city2graph 0.3.1.
    
    Args:
        gtfs_con: DuckDB connection with GTFS data
        start_date: Optional start date (YYYYMMDD format)
        end_date: Optional end date (YYYYMMDD format)
        directed: Whether to preserve trip direction
    
    Returns:
        GeoDataFrame with OD pairs
    """
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.transportation import get_od_pairs
        
        log("Extracting OD pairs from GTFS...")
        
        od_pairs = get_od_pairs(
            gtfs_con,
            start_date=start_date,
            end_date=end_date,
            include_geometry=True,
            directed=directed
        )
        
        if od_pairs is None or len(od_pairs) == 0:
            log("No OD pairs found")
            return None
        
        log(f"Extracted {len(od_pairs)} OD pairs")
        return od_pairs
        
    except Exception as e:
        log(f"Error extracting OD pairs: {e}")
        import traceback
        traceback.print_exc()
        return None


def visualize_gtfs_network(
    gtfs_con,
    osmnx_obj=None,
    create_stops=True,
    create_routes=True,
    stop_size=0.05,  # noqa: ARG001 — kept for API compatibility
):
    """Create the canonical SciGraphs object for a GTFS feed.

    The previous implementation produced two separate Blender objects
    (one for stops, one for routes), which were not real graphs and
    could not be analysed with the rest of SciGraphs. We now produce a
    SINGLE graph object via :func:`build_gtfs_graph` and return both
    keys pointing at it, so existing operators that read ``['stops']``
    or ``['routes']`` keep working.

    The ``create_stops`` / ``create_routes`` flags are honoured loosely:
    if neither is True we skip the build entirely.
    """
    result = {'stops': [], 'routes': []}
    if not (create_stops or create_routes):
        return result

    obj = build_gtfs_graph(gtfs_con, osmnx_obj=osmnx_obj, name="GTFS_Network")
    if obj is None:
        return result

    if create_stops:
        result['stops'] = [obj]
    if create_routes:
        result['routes'] = [obj]
    
    total_objects = len(result['stops']) + len(result['routes'])
    log(f"GTFS network visualization complete: {total_objects} objects created")
    
    return result
