# OSMnx graph cache management.
#
# Centralises the in-memory cache of NetworkX graphs so that both operators
# and other core modules can retrieve / store graphs by object metadata.

import uuid


def get_osmnx_graph(obj):
    """Retrieve the NetworkX graph associated with a Blender object.

    Looks up the in-memory cache first (by graph_id, then by object name,
    then by node-count heuristic).  Falls back to loading from the on-disk
    GraphML cache.  Returns ``None`` when the graph cannot be found.
    """
    if obj is None or not obj.get("is_osmnx", False):
        return None

    from .. import importer

    graph_id = obj.get("osmnx_graph_id", "")
    if graph_id and hasattr(importer, '_osmnx_graph_cache'):
        G = importer._osmnx_graph_cache.get(graph_id)
        if G is not None:
            return G

    if hasattr(importer, '_osmnx_graph_cache'):
        G = importer._osmnx_graph_cache.get(obj.name)
        if G is not None:
            return G

    if hasattr(importer, '_osmnx_graph_cache'):
        for key, G in importer._osmnx_graph_cache.items():
            if key.endswith('_unprojected'):
                continue
            if G is not None and G.number_of_nodes() == obj.get("num_nodes", -1):
                obj["osmnx_graph_id"] = key
                return G

    from . import cache

    G = cache.load_graph_from_cache(obj)

    if G is not None:
        if not hasattr(importer, '_osmnx_graph_cache'):
            importer._osmnx_graph_cache = {}

        if not graph_id:
            graph_id = str(uuid.uuid4())
            obj["osmnx_graph_id"] = graph_id

        importer._osmnx_graph_cache[graph_id] = G

        if obj.get("osmnx_projected", False):
            importer._osmnx_graph_cache[graph_id + "_unprojected"] = G.copy()

        return G

    return None


def get_osmnx_graph_diagnostic(obj):
    """Diagnose why a graph cannot be retrieved.

    Returns:
        Tuple ``(diagnostic_message, suggested_action)``.
    """
    import os
    from . import cache

    if obj is None:
        return "No object provided", "Select an OSMnx street network object"

    if not obj.get("is_osmnx", False):
        return "Object is not an OSMnx graph", "Select an object with is_osmnx property"

    query_name = obj.get("osmnx_query_name", "")

    cache_filepath = cache.get_cache_filepath(obj)
    cache_exists = cache_filepath and os.path.exists(cache_filepath)

    if not query_name:
        return (
            "Object is missing 'osmnx_query_name' metadata needed for cache lookup",
            "Re-import the street network to regenerate cache metadata",
        )

    if not cache_exists:
        expected_path = cache_filepath or "unknown"
        return (
            f"Graph cache file not found at: {expected_path}",
            "Re-import the street network or load from a saved .graphml file",
        )

    return (
        f"Cache file exists but could not be loaded: {cache_filepath}",
        "The cache file may be corrupted. Re-import the street network",
    )


def get_unprojected_graph(obj):
    """Return the unprojected copy of the cached graph (e.g. for bearings)."""
    from .. import importer

    graph_id = obj.get("osmnx_graph_id", "")
    if not graph_id or not hasattr(importer, '_osmnx_graph_cache'):
        return None

    return importer._osmnx_graph_cache.get(graph_id + "_unprojected")


def store_unprojected_graph(obj, G):
    """Cache the unprojected version of a graph."""
    from .. import importer

    graph_id = obj.get("osmnx_graph_id", "")
    if not graph_id:
        return

    if not hasattr(importer, '_osmnx_graph_cache'):
        importer._osmnx_graph_cache = {}

    importer._osmnx_graph_cache[graph_id + "_unprojected"] = G


def store_osmnx_graph(obj, G):
    """Store a NetworkX graph in the in-memory cache for *obj*."""
    from .. import importer

    if not hasattr(importer, '_osmnx_graph_cache'):
        importer._osmnx_graph_cache = {}

    graph_id = obj.get("osmnx_graph_id", "")
    if not graph_id:
        graph_id = str(uuid.uuid4())
        obj["osmnx_graph_id"] = graph_id

    importer._osmnx_graph_cache[graph_id] = G


def restore_all_graphs_from_cache():
    """Pre-load every OSMnx graph found in the scene from disk cache.

    Intended to be called from a ``bpy.app.handlers.load_post`` handler so
    that graph data is available immediately after opening a ``.blend`` file.
    """
    import bpy
    from .. import importer
    from ...utils.logger import log
    from . import cache

    if not hasattr(importer, '_osmnx_graph_cache'):
        importer._osmnx_graph_cache = {}

    osmnx_objects = [obj for obj in bpy.data.objects if obj.get("is_osmnx", False)]
    if not osmnx_objects:
        return

    loaded_count = 0
    for obj in osmnx_objects:
        graph_id = obj.get("osmnx_graph_id", "")

        if graph_id and graph_id in importer._osmnx_graph_cache:
            continue

        G = cache.load_graph_from_cache(obj)
        if G is None:
            continue

        if not graph_id:
            graph_id = str(uuid.uuid4())
            obj["osmnx_graph_id"] = graph_id

        importer._osmnx_graph_cache[graph_id] = G

        if obj.get("osmnx_projected", False):
            importer._osmnx_graph_cache[graph_id + "_unprojected"] = G.copy()

        loaded_count += 1

    if loaded_count > 0:
        log(f"Loaded {loaded_count} OSMnx graph(s) from cache")
