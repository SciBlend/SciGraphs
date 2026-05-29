"""OSMnx routing helpers.

Covers:
    - single-pair shortest path (length / travel_time / elevation_impedance)
    - k-shortest alternative paths
    - many-to-many batch routing (matrix, optional parallel cpus)
    - route_to_gdf-style summary (length, time, grade_abs, rise)
    - elevation-impedance weight injection (length * (1 + alpha * |grade|))
    - random origin-destination pair sampling
"""

from ...utils.logger import log
from .get_osmnx import get_osmnx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_edge_dicts(G):
    """Yield edge data dicts over G regardless of (Multi)Graph / (Multi)DiGraph."""
    if getattr(G, "is_multigraph", lambda: False)():
        for _u, _v, _k, data in G.edges(keys=True, data=True):
            yield data
    else:
        for _u, _v, data in G.edges(data=True):
            yield data


def _get_first_edge_data(G, u, v):
    """Return one edge data dict for (u, v), tolerating multigraph or simple graph."""
    if getattr(G, "is_multigraph", lambda: False)():
        candidates = G.get_edge_data(u, v)
        if not candidates:
            return None
        return next(iter(candidates.values()))
    if G.has_edge(u, v):
        return G.edges[u, v]
    return None


def _path_has_attribute(G, path, attr):
    """Return True iff every consecutive edge on ``path`` carries ``attr``."""
    for a, b in zip(path[:-1], path[1:]):
        data = _get_first_edge_data(G, a, b)
        if data is None or attr not in data or data.get(attr) in (None, ""):
            return False
    return True


def _ensure_impedance_weight(G, alpha=5.0, attr_name="_elevation_impedance"):
    """Inject ``length * (1 + alpha * |grade|)`` as an edge attribute.

    Skips edges without ``length`` or ``grade_abs`` / ``grade``.  Returns the
    attribute name so callers can pass it to NetworkX routing.
    """
    try:
        for data in _iter_edge_dicts(G):
            length = data.get("length")
            if length is None:
                continue
            grade = data.get("grade_abs")
            if grade is None:
                grade = abs(data.get("grade", 0.0) or 0.0)
            data[attr_name] = float(length) * (1.0 + float(alpha) * float(grade))
        return attr_name
    except Exception as e:
        log(f"Error building elevation impedance weights: {e}")
        return None


def _resolve_weight(G, weight, impedance_alpha=5.0):
    """Translate our UI weight enum into a NetworkX edge-attribute name.

    ``elevation_impedance`` triggers on-the-fly attribute computation.
    """
    if weight == "elevation_impedance":
        return _ensure_impedance_weight(G, alpha=impedance_alpha) or "length"
    return weight


# ---------------------------------------------------------------------------
# Single shortest path
# ---------------------------------------------------------------------------

def calculate_shortest_path(G, orig_node, dest_node, weight="length", impedance_alpha=5.0):
    """Calculate the shortest path between two nodes.

    Args:
        G: OSMnx MultiDiGraph.
        orig_node, dest_node: Node IDs.
        weight: 'length', 'travel_time' or 'elevation_impedance'.
        impedance_alpha: Multiplier when weight == 'elevation_impedance'.

    Returns:
        Dict with path + aggregates, or None on error.
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None

    try:
        import networkx as nx

        actual_weight = _resolve_weight(G, weight, impedance_alpha=impedance_alpha)

        # Pre-flight: routing by travel_time only makes sense if every edge has it.
        # Otherwise ox.shortest_path emits a warning and downstream nx.path_weight
        # raises KeyError('travel_time') on the first edge missing the attribute.
        if actual_weight == "travel_time":
            missing = sum(
                1 for d in _iter_edge_dicts(G)
                if d.get("travel_time") in (None, "")
            )
            if missing:
                return {
                    "error": (
                        f"'travel_time' missing on {missing} edge(s). "
                        "Add edge speeds and travel times before routing."
                    )
                }

        path = ox.shortest_path(G, orig_node, dest_node, weight=actual_weight)

        if path is None:
            return {"error": "No path found between nodes"}

        path_weight = nx.path_weight(G, path, weight=actual_weight)
        path_distance = nx.path_weight(G, path, weight="length")

        result = {
            "path": path,
            "num_nodes": len(path),
            "num_edges": len(path) - 1,
            "distance_m": round(path_distance, 1),
            "distance_km": round(path_distance / 1000.0, 3),
            "weight_used": actual_weight,
        }

        if weight == "travel_time" or _path_has_attribute(G, path, "travel_time"):
            try:
                travel_seconds = nx.path_weight(G, path, weight="travel_time")
                result["travel_time_seconds"] = round(travel_seconds, 1)
                result["travel_time_minutes"] = round(travel_seconds / 60.0, 2)
            except Exception:
                pass

        if weight == "elevation_impedance":
            result["impedance"] = round(path_weight, 2)
            result["alpha"] = impedance_alpha

        log(f"Shortest path: {len(path)} nodes, {path_distance:.1f}m [{actual_weight}]")
        return result

    except nx.NetworkXNoPath:
        return {"error": "No path exists between these nodes"}
    except Exception as e:
        log(f"Error calculating shortest path: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# K-shortest alternative routes
# ---------------------------------------------------------------------------

def k_shortest_paths(G, orig_node, dest_node, k=3, weight="length", impedance_alpha=5.0):
    """Return the k shortest simple paths between two nodes.

    Uses ``ox.routing.k_shortest_paths`` if available, otherwise falls back
    to ``networkx.shortest_simple_paths`` (Yen-like).
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None

    try:
        import networkx as nx

        actual_weight = _resolve_weight(G, weight, impedance_alpha=impedance_alpha)

        # Try the official OSMnx helper first.
        try:
            routing = getattr(ox, "routing", None)
            if routing is not None and hasattr(routing, "k_shortest_paths"):
                paths_iter = routing.k_shortest_paths(
                    G, orig_node, dest_node, k=k, weight=actual_weight,
                )
                paths = list(paths_iter)
            else:
                paths = list(nx.shortest_simple_paths(G, orig_node, dest_node, weight=actual_weight))
                paths = paths[:k]
        except Exception:
            paths = list(nx.shortest_simple_paths(G, orig_node, dest_node, weight=actual_weight))
            paths = paths[:k]

        results = []
        for p in paths:
            try:
                d_m = nx.path_weight(G, p, weight="length")
                entry = {
                    "path": p,
                    "num_nodes": len(p),
                    "distance_m": round(d_m, 1),
                    "distance_km": round(d_m / 1000.0, 3),
                }
                try:
                    t_s = nx.path_weight(G, p, weight="travel_time")
                    entry["travel_time_seconds"] = round(t_s, 1)
                    entry["travel_time_minutes"] = round(t_s / 60.0, 2)
                except Exception:
                    pass
                results.append(entry)
            except Exception:
                continue

        log(f"k-shortest paths: found {len(results)} / k={k}")
        return results

    except Exception as e:
        log(f"Error in k_shortest_paths: {e}")
        return None


# ---------------------------------------------------------------------------
# Many-to-many (batch) routing
# ---------------------------------------------------------------------------

def batch_shortest_paths(G, origs, dests, weight="length", cpus=None, impedance_alpha=5.0):
    """Compute shortest paths for parallel lists of (orig_i, dest_i).

    Mirrors ``ox.routing.shortest_path(G, origs, dests, cpus=cpus)``.
    Returns a list of node-lists (None for unreachable pairs) plus aggregate stats.
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None

    try:
        import networkx as nx

        actual_weight = _resolve_weight(G, weight, impedance_alpha=impedance_alpha)

        routing = getattr(ox, "routing", None)
        if routing is not None and hasattr(routing, "shortest_path"):
            paths = routing.shortest_path(
                G, origs, dests, weight=actual_weight, cpus=cpus or None,
            )
        else:
            paths = [ox.shortest_path(G, o, d, weight=actual_weight) for o, d in zip(origs, dests)]

        results = []
        total_len = 0.0
        total_time = 0.0
        reached = 0
        for p in paths:
            if not p:
                results.append(None)
                continue
            try:
                d_m = nx.path_weight(G, p, weight="length")
                entry = {"path": p, "num_nodes": len(p), "distance_m": round(d_m, 1)}
                total_len += d_m
                try:
                    t_s = nx.path_weight(G, p, weight="travel_time")
                    entry["travel_time_seconds"] = round(t_s, 1)
                    total_time += t_s
                except Exception:
                    pass
                results.append(entry)
                reached += 1
            except Exception:
                results.append(None)

        summary = {
            "total_pairs": len(results),
            "reached": reached,
            "mean_distance_m": round(total_len / max(1, reached), 1),
            "mean_travel_time_s": round(total_time / max(1, reached), 1) if total_time > 0 else None,
        }
        return {"paths": results, "summary": summary}

    except Exception as e:
        log(f"Error in batch_shortest_paths: {e}")
        return None


# ---------------------------------------------------------------------------
# Route summary (route_to_gdf equivalent)
# ---------------------------------------------------------------------------

def summarize_route(G, path):
    """Return aggregate metrics for a route.

    Sums of ``length``, ``travel_time``, ``grade_abs`` * length (weighted
    average absolute grade), and ``rise`` (sum of positive elevation diffs).
    """
    if G is None or not path or len(path) < 2:
        return None

    try:
        total_length = 0.0
        total_time = 0.0
        weighted_grade_sum = 0.0
        total_rise = 0.0

        has_elev = True
        for a, b in zip(path[:-1], path[1:]):
            # Pick the edge with the minimum length between this pair.
            if not G.has_edge(a, b):
                continue
            candidates = G.get_edge_data(a, b)
            if not candidates:
                continue
            edge = min(candidates.values(), key=lambda d: d.get("length", float("inf")))
            length = edge.get("length", 0.0) or 0.0
            total_length += length
            total_time += edge.get("travel_time", 0.0) or 0.0
            g_abs = edge.get("grade_abs")
            if g_abs is None:
                g_abs = abs(edge.get("grade", 0.0) or 0.0)
            weighted_grade_sum += g_abs * length

            elev_a = G.nodes[a].get("elevation")
            elev_b = G.nodes[b].get("elevation")
            if elev_a is None or elev_b is None:
                has_elev = False
            else:
                diff = float(elev_b) - float(elev_a)
                if diff > 0:
                    total_rise += diff

        result = {
            "length_m": round(total_length, 1),
            "length_km": round(total_length / 1000.0, 3),
            "travel_time_s": round(total_time, 1),
            "travel_time_min": round(total_time / 60.0, 2),
            "mean_grade_abs": (
                round(weighted_grade_sum / total_length, 4) if total_length > 0 else 0.0
            ),
        }
        if has_elev:
            result["rise_m"] = round(total_rise, 1)
        return result

    except Exception as e:
        log(f"Error summarizing route: {e}")
        return None


# ---------------------------------------------------------------------------
# Elevation profile (vertices = cumulative distance, Z = elevation)
# ---------------------------------------------------------------------------

def route_elevation_profile(G, path):
    """Return ``[(cum_distance_m, elevation_m), ...]`` along a path.

    Requires node elevations. Returns None if unavailable.
    """
    if G is None or not path:
        return None
    try:
        profile = []
        cum = 0.0
        for i, node in enumerate(path):
            elev = G.nodes[node].get("elevation")
            if elev is None:
                return None
            if i > 0:
                prev = path[i - 1]
                if G.has_edge(prev, node):
                    candidates = G.get_edge_data(prev, node)
                    if candidates:
                        edge = min(candidates.values(), key=lambda d: d.get("length", float("inf")))
                        cum += float(edge.get("length", 0.0) or 0.0)
            profile.append((cum, float(elev)))
        return profile
    except Exception as e:
        log(f"Error building elevation profile: {e}")
        return None


# ---------------------------------------------------------------------------
# Random OD sampling
# ---------------------------------------------------------------------------

def sample_random_od_pairs(G, n=10, seed=None):
    """Return ``n`` random (orig, dest) node-id pairs from G."""
    try:
        import random
        nodes = list(G.nodes)
        if len(nodes) < 2:
            return []
        rng = random.Random(seed)
        origs = [rng.choice(nodes) for _ in range(n)]
        dests = [rng.choice(nodes) for _ in range(n)]
        return list(zip(origs, dests))
    except Exception as e:
        log(f"Error sampling OD pairs: {e}")
        return []
