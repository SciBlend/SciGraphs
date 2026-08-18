# OSMnx: street networks, projection, routing, elevation, stats.
#
# Flat public API on purpose so callers need not know which submodule owns a
# name. Everything is lazy except get_osmnx (see below): importing one name
# must not pull osmnx/geopandas/networkx/shapely for the whole package.
#
# _LAZY: attribute is the submodule.
# _LAZY_SYMBOLS: attribute is getattr(module, name). Plain strings (not
# (module, attr) pairs) so the purity test sees every edge; every target also
# appears in _LAZY. Names keep the spelling they have in their own module.
#
# get_osmnx is eager: it is both a module and a function in that module, and
# `from .get_osmnx import get_osmnx` must win. Seventeen modules open with that
# import; the first would overwrite a __getattr__ cache with the submodule.
# Cheap (logger only; third-party osmnx loads inside the function).
#
# centrality is in _LAZY for UI operators that bind it on the parent; cache is
# not (callers import the submodule directly). __all__ omits the accessibility
# re-exports: star-import is not the same set as attributes this package answers.

from .get_osmnx import get_osmnx        # noqa: F401  (eager, see above)

_LAZY = {
    "accessibility": ".accessibility",
    "analysis": ".analysis",
    "bearing": ".bearing",
    "centrality": ".centrality",
    "convert": ".convert",
    "distance": ".distance",
    "edge_attributes": ".edge_attributes",
    "elevation": ".elevation",
    "features": ".features",
    "geocoder": ".geocoder",
    "io": ".io",
    "mesh_bridge": ".mesh_bridge",
    "metadata": ".metadata",
    "projection": ".projection",
    "routing": ".routing",
    "simplification": ".simplification",
    "spatial_queries": ".spatial_queries",
    "stats": ".stats",
    "truncate": ".truncate",
    "utils_geo": ".utils_geo",
}

_LAZY_SYMBOLS = {
    "is_graph_projected": ".projection",
    "project_graph": ".projection",

    "_add_edge_grades_manual": ".edge_attributes",
    "add_edge_bearings": ".edge_attributes",
    "add_edge_grades": ".edge_attributes",
    "add_edge_lengths": ".edge_attributes",
    "add_edge_speeds": ".edge_attributes",
    "add_edge_travel_times": ".edge_attributes",

    "circuity_avg": ".stats",
    "get_basic_stats": ".stats",
    "get_bearing_distribution": ".stats",
    "get_elevation_stats": ".stats",
    "get_grade_stats": ".stats",

    "_find_nearest_edge_numpy": ".spatial_queries",
    "_find_nearest_node_numpy": ".spatial_queries",
    "find_nearest_edge": ".spatial_queries",
    "find_nearest_node": ".spatial_queries",
    "get_edge_info": ".spatial_queries",
    "get_node_coordinates": ".spatial_queries",

    "batch_shortest_paths": ".routing",
    "calculate_shortest_path": ".routing",
    "k_shortest_paths": ".routing",
    "route_elevation_profile": ".routing",
    "sample_random_od_pairs": ".routing",
    "summarize_route": ".routing",

    "_add_elevations_from_open_elevation": ".elevation",
    "_add_elevations_from_raster_manual": ".elevation",
    "add_node_elevations_google": ".elevation",
    "add_node_elevations_raster": ".elevation",

    "load_graph_graphml": ".io",
    "save_graph_graphml": ".io",

    "estimate_network_area": ".metadata",
    "get_graph_extent": ".metadata",

    "calculate_bearing": ".bearing",
    "get_bearings_distribution": ".bearing",
    "orientation_entropy": ".bearing",

    "graph_from_gdfs": ".convert",
    "graph_to_gdfs": ".convert",
    "to_digraph": ".convert",
    "to_undirected": ".convert",

    "euclidean": ".distance",
    "great_circle": ".distance",
    "nearest_edges": ".distance",
    "nearest_nodes": ".distance",

    "features_from_address": ".features",
    "features_from_bbox": ".features",
    "features_from_place": ".features",
    "features_from_point": ".features",
    "features_from_polygon": ".features",
    "features_from_xml": ".features",

    "geocode": ".geocoder",
    "geocode_to_gdf": ".geocoder",

    "consolidate_intersections": ".simplification",
    "simplify_graph": ".simplification",

    "largest_component": ".truncate",
    "truncate_graph_bbox": ".truncate",
    "truncate_graph_dist": ".truncate",
    "truncate_graph_polygon": ".truncate",

    "bbox_from_point": ".utils_geo",
    "bbox_to_poly": ".utils_geo",
    "buffer_geometry": ".utils_geo",
    "interpolate_points": ".utils_geo",
    "sample_points": ".utils_geo",

    "add_travel_time_from_speed": ".accessibility",
    "ego_subgraph": ".accessibility",
    "make_iso_polygons": ".accessibility",
    "network_dbscan": ".accessibility",
}


def __getattr__(name):
    """Import the submodule that backs ``name`` on first access (PEP 562)."""
    import importlib

    target = _LAZY.get(name)
    if target is not None:
        value = importlib.import_module(target, __name__)
    else:
        target = _LAZY_SYMBOLS.get(name)
        if target is None:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}")
        module = importlib.import_module(target, __name__)
        # Name in the table but missing from the module = stale table.
        try:
            value = getattr(module, name)
        except AttributeError:
            raise AttributeError(
                f"{__name__}.{name} is listed in _LAZY_SYMBOLS as coming from "
                f"{target!r}, but that module does not define it") from None

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | set(_LAZY_SYMBOLS))


__all__ = [
    "get_osmnx",
    "project_graph",
    "is_graph_projected",
    "add_edge_lengths",
    "add_edge_bearings",
    "add_edge_speeds",
    "add_edge_travel_times",
    "add_edge_grades",
    "_add_edge_grades_manual",
    "circuity_avg",
    "get_basic_stats",
    "get_bearing_distribution",
    "get_elevation_stats",
    "get_grade_stats",
    "find_nearest_node",
    "_find_nearest_node_numpy",
    "find_nearest_edge",
    "_find_nearest_edge_numpy",
    "get_node_coordinates",
    "get_edge_info",
    "calculate_shortest_path",
    "k_shortest_paths",
    "batch_shortest_paths",
    "summarize_route",
    "route_elevation_profile",
    "sample_random_od_pairs",
    "add_node_elevations_raster",
    "_add_elevations_from_raster_manual",
    "add_node_elevations_google",
    "_add_elevations_from_open_elevation",
    "save_graph_graphml",
    "load_graph_graphml",
    "get_graph_extent",
    "estimate_network_area",
    "calculate_bearing",
    "get_bearings_distribution",
    "orientation_entropy",
    "graph_from_gdfs",
    "graph_to_gdfs",
    "to_digraph",
    "to_undirected",
    "euclidean",
    "great_circle",
    "nearest_edges",
    "nearest_nodes",
    "features_from_address",
    "features_from_bbox",
    "features_from_place",
    "features_from_point",
    "features_from_polygon",
    "features_from_xml",
    "geocode",
    "geocode_to_gdf",
    "consolidate_intersections",
    "simplify_graph",
    "largest_component",
    "truncate_graph_bbox",
    "truncate_graph_dist",
    "truncate_graph_polygon",
    "bbox_from_point",
    "bbox_to_poly",
    "buffer_geometry",
    "interpolate_points",
    "sample_points",
]
