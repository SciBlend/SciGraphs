from scigraphs_core.logger import log
from .get_osmnx import get_osmnx
from .projection import is_graph_projected


def _iter_edges(G):
    """Iterate edges tolerantly over MultiDiGraph / DiGraph / MultiGraph."""
    if getattr(G, "is_multigraph", lambda: False)():
        for u, v, k, data in G.edges(keys=True, data=True):
            yield u, v, k, data
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, None, data


def _edge_access(G, u, v, k, attr, value):
    """Set an edge attribute tolerantly whether or not G is a multigraph."""
    if k is not None and getattr(G, "is_multigraph", lambda: False)():
        if G.has_edge(u, v, k):
            G.edges[u, v, k][attr] = value
            return True
    else:
        if G.has_edge(u, v):
            G.edges[u, v][attr] = value
            return True
    return False


def add_edge_lengths(G):
    """Add a 'length' to every edge, from its geometry or from straight-line node distance.

    Lengths come out in the graph's own units, so project the graph first if you
    want meters.
    """
    if G is None:
        return None
    
    try:
        for u, v, _k, data in _iter_edges(G):
            if "geometry" in data and data["geometry"] is not None:
                data["length"] = data["geometry"].length
            else:
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                x1, y1 = u_data.get("x", 0), u_data.get("y", 0)
                x2, y2 = v_data.get("x", 0), v_data.get("y", 0)
                data["length"] = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        
        log("Edge lengths added to graph")
        return G
    except Exception as e:
        log(f"Error adding edge lengths: {e}")
        return None


def add_edge_bearings(G, unprojected_G=None):
    """Add a compass 'bearing' in degrees (0-360) to every edge.

    True bearings need lat/lon, so pass the unprojected graph as unprojected_G
    when G is projected. Without it, projected graphs fall back to grid angles
    off the projected coordinates, which is not the same thing.
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None
    
    try:
        if is_graph_projected(G):
            if unprojected_G is not None:
                from . import convert as _convert
                unprojected_G = _convert.ensure_multidigraph(unprojected_G)

                if hasattr(ox, "bearing") and hasattr(ox.bearing, "add_edge_bearings"):
                    unprojected_G = ox.bearing.add_edge_bearings(unprojected_G)
                else:
                    unprojected_G = ox.add_edge_bearings(unprojected_G)

                for u, v, key, data in _iter_edges(unprojected_G):
                    if "bearing" in data:
                        _edge_access(G, u, v, key, "bearing", data["bearing"])

                log("Edge bearings copied from unprojected graph")
                return G
            else:
                import math
                for u, v, _key, data in _iter_edges(G):
                    u_data = G.nodes[u]
                    v_data = G.nodes[v]
                    x1, y1 = u_data.get("x", 0), u_data.get("y", 0)
                    x2, y2 = v_data.get("x", 0), v_data.get("y", 0)

                    dx = x2 - x1
                    dy = y2 - y1
                    angle = math.degrees(math.atan2(dx, dy))
                    bearing = (angle + 360) % 360
                    data["bearing"] = bearing

                log("Edge bearings calculated from projected coordinates")
                return G
        else:
            from . import convert as _convert
            needs_convert = not getattr(G, "is_multigraph", lambda: False)()
            if needs_convert:
                G_multi = _convert.ensure_multidigraph(G)
                if hasattr(ox, "bearing") and hasattr(ox.bearing, "add_edge_bearings"):
                    G_multi = ox.bearing.add_edge_bearings(G_multi)
                else:
                    G_multi = ox.add_edge_bearings(G_multi)
                for u, v, _key, data in _iter_edges(G_multi):
                    if "bearing" in data:
                        _edge_access(G, u, v, _key, "bearing", data["bearing"])
                log("Edge bearings added to graph")
                return G

            if hasattr(ox, "bearing") and hasattr(ox.bearing, "add_edge_bearings"):
                G = ox.bearing.add_edge_bearings(G)
            else:
                G = ox.add_edge_bearings(G)
            log("Edge bearings added to graph")
            return G

    except Exception as e:
        log(f"Error adding edge bearings: {e}")
        return None


def add_edge_speeds(G, fallback_speed=30):
    """Add 'speed_kph' to every edge from its OSM maxspeed and highway type.

    fallback_speed, in km/h, covers edges with neither. A download made with a
    custom_filter can carry no 'highway' tag at all, and then every edge gets the
    fallback.
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None
    
    try:
        from . import convert as _convert
        G = _convert.ensure_multidigraph(G)

        # OSMnx's add_edge_speeds only fills in missing values, so recalculating
        # with a different fallback would skip every edge that already has a
        # speed. Purge first.
        for _u, _v, _k, data in _iter_edges(G):
            data.pop("speed_kph", None)
            data.pop("travel_time", None)

        try:
            G = ox.add_edge_speeds(G, fallback=fallback_speed)
            log(f"Edge speeds added (fallback: {fallback_speed} km/h)")
            return G
        except (KeyError, Exception) as e:
            msg = str(e)
            if "highway" in msg or isinstance(e, KeyError):
                log(f"No 'highway' tag found on edges (custom_filter download?). "
                    f"Assigning uniform fallback speed = {fallback_speed} km/h.")
                for _u, _v, _k, data in _iter_edges(G):
                    data["speed_kph"] = float(fallback_speed)
                return G
            raise
    except Exception as e:
        log(f"Error adding edge speeds: {e}")
        return None


def add_edge_travel_times(G):
    """Add 'travel_time' in seconds to every edge. Run add_edge_speeds first."""
    ox = get_osmnx()
    if ox is None or G is None:
        return None
    
    try:
        from . import convert as _convert
        G = _convert.ensure_multidigraph(G)

        # Wipe stale travel_time so a recomputation with new speeds takes effect.
        for _u, _v, _k, data in _iter_edges(G):
            data.pop("travel_time", None)

        G = ox.add_edge_travel_times(G)
        log("Edge travel times added to graph")
        return G
    except Exception as e:
        log(f"Error adding travel times: {e}")
        return None


def add_edge_grades(G, add_absolute=True):
    """Add 'grade' (rise over length) to every edge, plus 'grade_abs' if add_absolute.

    Needs 'elevation' on the nodes and 'length' on the edges.
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return None
    
    try:
        sample_node = next(iter(G.nodes()))
        if "elevation" not in G.nodes[sample_node]:
            log("Error: Nodes must have 'elevation' attribute. Add elevations first.")
            return None
        
        if hasattr(ox, "elevation") and hasattr(ox.elevation, "add_edge_grades"):
            G = ox.elevation.add_edge_grades(G, add_absolute=add_absolute)
        elif hasattr(ox, "add_edge_grades"):
            G = ox.add_edge_grades(G, add_absolute=add_absolute)
        else:
            G = _add_edge_grades_manual(G, add_absolute)
        
        log("Edge grades calculated")
        return G
        
    except Exception as e:
        log(f"Error calculating edge grades: {e}")
        return None


def _add_edge_grades_manual(G, add_absolute=True):
    """Compute edge grades without OSMnx."""
    for u, v, _key, data in _iter_edges(G):
        elev_u = G.nodes[u].get("elevation", 0)
        elev_v = G.nodes[v].get("elevation", 0)
        length = data.get("length", 0)
        
        if length > 0:
            grade = (elev_v - elev_u) / length
        else:
            grade = 0.0
        
        data["grade"] = grade
        
        if add_absolute:
            data["grade_abs"] = abs(grade)
    
    return G

