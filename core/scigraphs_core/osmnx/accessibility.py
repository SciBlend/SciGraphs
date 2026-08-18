"""Accessibility analysis: isochrones, ego subgraphs and clustering by network
distance. Everything here takes an OSMnx MultiDiGraph already in memory."""

from scigraphs_core.logger import log


def add_travel_time_from_speed(G, travel_speed_kph):
    """A ``travel_time`` in seconds per edge at one uniform speed, for walking and
    cycling isochrones. Existing travel times are left alone; returns the count."""
    if G is None:
        return 0
    meters_per_minute = travel_speed_kph * 1000.0 / 60.0
    if meters_per_minute <= 0:
        return 0
    updated = 0
    for u, v, data in G.edges(data=True):
        if "travel_time" not in data or data["travel_time"] is None:
            length = data.get("length", 0.0) or 0.0
            # Seconds, which is what OSMnx means by travel_time, not minutes.
            data["travel_time"] = (length / meters_per_minute) * 60.0
            updated += 1
    return updated


def ego_subgraph(G, center_node, radius, distance_attr="length"):
    """Subgraph within ``radius`` of ``center_node``, by hops if distance_attr None."""
    try:
        import networkx as nx
        sub = nx.ego_graph(G, center_node, radius=radius, distance=distance_attr)
        return sub
    except Exception as e:
        log(f"Error building ego subgraph: {e}")
        return None


def make_iso_polygons(
    G,
    center_node,
    trip_times_minutes,
    travel_speed_kph=4.5,
    mode="BUFFER_UNION",
    buffer_m=25.0,
):
    """Concentric isochrone polygons around ``center_node``, one per threshold in
    ``trip_times_minutes``. Edges with no ``travel_time`` get one imputed from their
    length at ``travel_speed_kph``; 'CONVEX_HULL' is the fast mode, 'BUFFER_UNION'
    the accurate one. Largest first, so drawing them in order stacks correctly."""
    import networkx as nx

    if G is None or center_node is None or not trip_times_minutes:
        return []

    times = sorted(set(int(t) for t in trip_times_minutes), reverse=True)

    # Local only: the graph is not mutated.
    speed_mpm = travel_speed_kph * 1000.0 / 60.0

    try:
        from shapely.geometry import Point, LineString
    except Exception as e:
        log(f"Shapely not available: {e}")
        return []

    def _edge_weight(_u, _v, d):
        # Fastest of the parallel edges wins.
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
                    geoms.append(Point(x, y).buffer(buffer_m / 111000.0))  # m to deg
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


def network_dbscan(G, eps_meters=500.0, min_samples=5, weight="length"):
    """DBSCAN over network distance, not straight-line. Feeds scikit-learn a
    precomputed all-pairs Dijkstra matrix, so memory is O(n^2) and big graphs
    will not fit; label -1 means noise."""
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

    # DBSCAN needs a symmetric matrix; OSMnx graphs are directed.
    sym = np.minimum(dist, dist.T)
    # DBSCAN cannot take infinities, so cap unreachable pairs.
    sym[sym == INF] = 1e12

    db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="precomputed")
    labels = db.fit_predict(sym)

    return {nodes[i]: int(labels[i]) for i in range(n)}
