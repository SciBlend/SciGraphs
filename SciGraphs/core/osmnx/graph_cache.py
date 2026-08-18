# In-memory cache of NetworkX graphs, keyed by Blender object metadata, shared
# by the operators and the rest of core.

import uuid


def get_osmnx_graph(obj):
    """Find the NetworkX graph behind a Blender object, or None.

    Tries the in-memory cache by graph_id, then by object name, then by
    matching node count, before falling back to the on-disk GraphML cache. The
    node-count match is a guess and will pick the wrong graph if two networks
    happen to have the same size.
    """
    if obj is None or not obj.get("is_osmnx", False):
        return None

    from ...core import importer

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
    """Explain why get_osmnx_graph failed, as (message, suggested action)."""
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
    """Return the unprojected copy of the cached graph, as bearings need."""
    from ...core import importer

    graph_id = obj.get("osmnx_graph_id", "")
    if not graph_id or not hasattr(importer, '_osmnx_graph_cache'):
        return None

    return importer._osmnx_graph_cache.get(graph_id + "_unprojected")


def store_unprojected_graph(obj, G):
    """Cache the unprojected version of a graph."""
    from ...core import importer

    graph_id = obj.get("osmnx_graph_id", "")
    if not graph_id:
        return

    if not hasattr(importer, '_osmnx_graph_cache'):
        importer._osmnx_graph_cache = {}

    importer._osmnx_graph_cache[graph_id + "_unprojected"] = G


def store_osmnx_graph(obj, G):
    """Put a graph in the in-memory cache, assigning obj a graph_id if needed."""
    from ...core import importer

    if not hasattr(importer, '_osmnx_graph_cache'):
        importer._osmnx_graph_cache = {}

    graph_id = obj.get("osmnx_graph_id", "")
    if not graph_id:
        graph_id = str(uuid.uuid4())
        obj["osmnx_graph_id"] = graph_id

    importer._osmnx_graph_cache[graph_id] = G


def restore_all_graphs_from_cache():
    """Load every OSMnx graph in the scene from the disk cache.

    Belongs on a bpy.app.handlers.load_post handler: graph data has to be there
    the moment a .blend opens, since nothing in the file itself holds it.
    """
    import bpy
    from ...core import importer
    from scigraphs_core.logger import log
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
