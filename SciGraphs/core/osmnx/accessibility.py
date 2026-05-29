"""Accessibility and network-constrained spatial analysis.

Port of OSMnx notebook 13 (isochrones) and 18 (network-constrained clustering).
All functions expect an OSMnx MultiDiGraph already loaded in memory.
"""

from ...utils.logger import log


# ---------------------------------------------------------------------------
# Travel-time attribute injection
# ---------------------------------------------------------------------------

def add_travel_time_from_speed(G, travel_speed_kph):
    """Add a ``travel_time`` attribute (seconds) to every edge using a uniform speed.

    Useful for walking / cycling isochrones where edges do not yet have
    travel-time attributes (only length). Does NOT overwrite existing values.
    """
    if G is None:
        return 0
    meters_per_minute = travel_speed_kph * 1000.0 / 60.0
    if meters_per_minute <= 0:
        return 0
    updated = 0
    for u, v, data in G.edges(data=True):
        if "travel_time" not in data or data["travel_time"] is None:
            length = data.get("length", 0.0) or 0.0
            # travel_time in seconds; notebook uses minutes, but OSMnx canonical is seconds.
            data["travel_time"] = (length / meters_per_minute) * 60.0
            updated += 1
    return updated


# ---------------------------------------------------------------------------
# Ego-graph truncation
# ---------------------------------------------------------------------------

def ego_subgraph(G, center_node, radius, distance_attr="length"):
    """Return the subgraph of nodes reachable within ``radius`` from ``center_node``.

    If ``distance_attr`` is None, uses hop count.
    """
    try:
        import networkx as nx
        sub = nx.ego_graph(G, center_node, radius=radius, distance=distance_attr)
        return sub
    except Exception as e:
        log(f"Error building ego subgraph: {e}")
        return None


# ---------------------------------------------------------------------------
# Isochrones
# ---------------------------------------------------------------------------

def make_iso_polygons(
    G,
    center_node,
    trip_times_minutes,
    travel_speed_kph=4.5,
    mode="BUFFER_UNION",
    buffer_m=25.0,
):
    """Build concentric isochrone polygons around ``center_node``.

    Args:
        G: OSMnx graph. If edges miss ``travel_time``, it is computed from
           ``length`` using ``travel_speed_kph`` (walking speed default).
        center_node: Node ID used as origin.
        trip_times_minutes: Iterable of time thresholds (minutes).
        travel_speed_kph: Speed to impute travel_time (only for missing edges).
        mode: 'CONVEX_HULL' (fast) or 'BUFFER_UNION' (accurate).
        buffer_m: Buffer size for BUFFER_UNION mode.

    Returns:
        List of dicts ``[{"time": t, "polygon": shapely_polygon_or_None}, ...]``
        ordered from largest to smallest so that they render nicely stacked.
    """
    import networkx as nx

    if G is None or center_node is None or not trip_times_minutes:
        return []

    times = sorted(set(int(t) for t in trip_times_minutes), reverse=True)

    # Convert travel_time to minutes locally (do not mutate the graph).
    speed_mpm = travel_speed_kph * 1000.0 / 60.0

    try:
        from shapely.geometry import Point, LineString
    except Exception as e:
        log(f"Shapely not available: {e}")
        return []

    def _edge_weight(_u, _v, d):
        # Pick minimum-travel-time among parallel edges.
        if isinstance(d, dict) and d and all(isinstance(value, dict) for value in d.values()):
            edge_dicts = list(d.values())
        elif isinstance(d, dict):
            edge_dicts = [d]
        else:
            edge_dicts = []
        best = None
        for ed in edge_dicts:
            t = ed.get("travel_time")
            if t is None:
                length = ed.get("length", 0.0) or 0.0
                t = length / speed_mpm * 60.0 if speed_mpm > 0 else float("inf")
            if best is None or t < best:
                best = t
        return best if best is not None else float("inf")

    try:
        distances = nx.single_source_dijkstra_path_length(
            G, center_node, weight=_edge_weight,
        )
    except Exception as e:
        log(f"Isochrone dijkstra failed: {e}")
        return []

    results = []
    for t_min in times:
        threshold = t_min * 60.0  # seconds
        reachable = {n for n, d in distances.items() if d <= threshold}
        if not reachable:
            results.append({"time": t_min, "polygon": None})
            continue

        if mode == "CONVEX_HULL":
            points = []
            for n in reachable:
                x = G.nodes[n].get("x")
                y = G.nodes[n].get("y")
                if x is not None and y is not None:
                    points.append(Point(x, y))
            if len(points) < 3:
                results.append({"time": t_min, "polygon": None})
                continue
            try:
                from shapely.geometry import MultiPoint
                poly = MultiPoint(points).convex_hull
            except Exception as e:
                log(f"Convex hull failed: {e}")
                poly = None
            results.append({"time": t_min, "polygon": poly})

        else:  # BUFFER_UNION
            geoms = []
            for n in reachable:
                x = G.nodes[n].get("x")
                y = G.nodes[n].get("y")
                if x is not None and y is not None:
                    geoms.append(Point(x, y).buffer(buffer_m / 111000.0))  # deg approx
            # Add reachable edges as linestrings.
            for u, v in G.edges():
                if u in reachable and v in reachable:
                    xu, yu = G.nodes[u].get("x"), G.nodes[u].get("y")
                    xv, yv = G.nodes[v].get("x"), G.nodes[v].get("y")
                    if None in (xu, yu, xv, yv):
                        continue
                    try:
                        geoms.append(LineString([(xu, yu), (xv, yv)]).buffer(buffer_m / 111000.0))
                    except Exception:
                        continue
            if not geoms:
                results.append({"time": t_min, "polygon": None})
                continue
            try:
                from shapely.ops import unary_union
                poly = unary_union(geoms)
            except Exception as e:
                log(f"Buffer union failed: {e}")
                poly = None
            results.append({"time": t_min, "polygon": poly})

    return results


# ---------------------------------------------------------------------------
# Network-constrained DBSCAN
# ---------------------------------------------------------------------------

def network_dbscan(G, eps_meters=500.0, min_samples=5, weight="length"):
    """DBSCAN on the network-distance matrix between graph nodes.

    Uses scikit-learn with a precomputed distance matrix built via
    ``all_pairs_dijkstra_path_length`` (length in meters).

    Args:
        G: OSMnx MultiDiGraph. For large graphs this is O(n^2) in memory.
        eps_meters: Neighborhood radius in network-length meters.
        min_samples: DBSCAN min samples.
        weight: Edge attribute used as distance.

    Returns:
        Dict ``{node_id: cluster_label}`` where label -1 is noise.
    """
    try:
        import networkx as nx
        import numpy as np
        from sklearn.cluster import DBSCAN
    except Exception as e:
        log(f"Missing dependency for network_dbscan: {e}")
        return None

    if G is None:
        return None

    nodes = list(G.nodes)
    n = len(nodes)
    if n == 0:
        return {}

    idx = {node: i for i, node in enumerate(nodes)}

    # Build distance matrix (float16 to limit memory for large graphs).
    INF = float("inf")
    dist = np.full((n, n), INF, dtype=np.float32)
    try:
        paths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    except Exception as e:
        log(f"Dijkstra failed: {e}")
        return None

    for src, rest in paths.items():
        i = idx[src]
        for tgt, d in rest.items():
            dist[i, idx[tgt]] = float(d)

    # Symmetrize for DBSCAN (OSMnx graphs are directed).
    sym = np.minimum(dist, dist.T)
    # Cap INF to a huge number DBSCAN can handle.
    sym[sym == INF] = 1e12

    db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(sym)

    return {nodes[i]: int(labels[i]) for i in range(n)}
