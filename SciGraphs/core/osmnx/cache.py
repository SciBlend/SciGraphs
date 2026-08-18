"""On-disk GraphML cache for OSMnx graphs, so they survive a Blender restart.

Filenames are derived from the query and network type, which means two objects
built from the same query share one cache entry.
"""

import os
import re
from pathlib import Path
from scigraphs_core.logger import log
from scigraphs_core.osmnx.get_osmnx import get_osmnx
from scigraphs_core.osmnx.io import save_graph_graphml, load_graph_graphml


def get_cache_directory():
    """Cache directory from preferences, or one under Blender's user scripts."""
    import bpy
    from ...preferences import get_preferences

    prefs = get_preferences()
    if prefs and prefs.osmnx_cache_directory:
        cache_dir = bpy.path.abspath(prefs.osmnx_cache_directory)
    else:
        user_scripts = bpy.utils.resource_path('USER')
        cache_dir = os.path.join(user_scripts, "scripts", "addons", "scigraphs_osmnx_cache")
    
    return cache_dir


def ensure_cache_directory():
    """Create the cache directory if needed. False if that fails."""
    try:
        cache_dir = get_cache_directory()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log(f"Error creating cache directory: {e}")
        return False


def sanitize_filename(name):
    """Make a string usable as a filename: reserved characters and whitespace
    become underscores, and the result is cut to 100 characters.
    """
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'[\s,]+', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    if len(name) > 100:
        name = name[:100]
    return name


def generate_cache_filename(obj):
    """Cache filename for an object, from its query name and network type.

    None for anything that is not an OSMnx object. Objects with no stored query
    name fall back to the object name, so renaming one orphans its cache entry.
    """
    if not obj or not obj.get("is_osmnx", False):
        return None

    query_name = obj.get("osmnx_query_name", "")
    network_type = obj.get("osmnx_network_type", "drive")

    if not query_name:
        query_name = obj.name

    safe_name = sanitize_filename(query_name)
    safe_network = sanitize_filename(network_type)

    filename = f"{safe_name}_{safe_network}.graphml"

    return filename


def get_cache_filepath(obj):
    """Full path to an object's cache file, or None if it has no name."""
    filename = generate_cache_filename(obj)
    if not filename:
        return None
    
    cache_dir = get_cache_directory()
    return os.path.join(cache_dir, filename)


def save_graph_to_cache(obj, G):
    """Write a graph to the object's cache slot.

    Returns (success, filepath or None, message); the message is meant for the
    operator to report.
    """
    if not obj or not G:
        return False, None, "Invalid object or graph"
    
    if not ensure_cache_directory():
        return False, None, "Could not create cache directory"
    
    filepath = get_cache_filepath(obj)
    if not filepath:
        return False, None, "Could not determine cache filename"
    
    try:
        success = save_graph_graphml(G, filepath)
        if success:
            log(f"Graph cached to: {filepath}")
            return True, filepath, "Graph saved to cache"
        else:
            return False, None, "Failed to save graph file"
    except Exception as e:
        log(f"Error saving graph to cache: {e}")
        return False, None, str(e)


def load_graph_from_cache(obj):
    """Read an object's cached graph, or None when there is no usable file."""
    if not obj:
        return None
    
    filepath = get_cache_filepath(obj)
    if not filepath:
        return None
    
    if not os.path.exists(filepath):
        log(f"Cache file not found: {filepath}")
        return None
    
    try:
        G = load_graph_graphml(filepath)
        if G:
            log(f"Graph loaded from cache: {filepath}")
        return G
    except Exception as e:
        log(f"Error loading graph from cache: {e}")
        return None


def list_cached_graphs():
    """List cached graphs newest first, as (filename, filepath, size_mb, mtime)."""
    cache_dir = get_cache_directory()
    
    if not os.path.exists(cache_dir):
        return []
    
    cached_graphs = []
    
    try:
        for filename in os.listdir(cache_dir):
            if filename.endswith('.graphml'):
                filepath = os.path.join(cache_dir, filename)
                
                stat = os.stat(filepath)
                size_mb = stat.st_size / (1024 * 1024)
                modified_time = stat.st_mtime
                
                cached_graphs.append((filename, filepath, size_mb, modified_time))
        
        cached_graphs.sort(key=lambda x: x[3], reverse=True)
        
    except Exception as e:
        log(f"Error listing cached graphs: {e}")
    
    return cached_graphs


def delete_cached_graph(filepath):
    """Delete one cached graph file. False if it was missing or locked."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            log(f"Deleted cached graph: {filepath}")
            return True
        else:
            log(f"Cache file not found: {filepath}")
            return False
    except Exception as e:
        log(f"Error deleting cached graph: {e}")
        return False


def clear_all_cache():
    """Delete every cached graph. Returns (deleted, failed)."""
    cached_graphs = list_cached_graphs()
    success_count = 0
    error_count = 0
    
    for filename, filepath, _, _ in cached_graphs:
        if delete_cached_graph(filepath):
            success_count += 1
        else:
            error_count += 1
    
    return success_count, error_count

