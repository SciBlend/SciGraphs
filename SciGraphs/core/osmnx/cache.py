"""
OSMnx Graph Cache Management

Persistent storage for OSMnx graphs to prevent memory loss when Blender restarts.
Graphs are saved to a configurable cache directory with descriptive filenames.
"""

import os
import re
from pathlib import Path
from ...utils.logger import log
from .get_osmnx import get_osmnx
from .io import save_graph_graphml, load_graph_graphml


def get_cache_directory():
    """
    Get the configured cache directory from addon preferences.
    If not set, use default location in Blender's user scripts folder.
    
    Returns:
        Path to cache directory (string)
    """
    import bpy
    from ...preferences import get_preferences
    
    prefs = get_preferences()
    if prefs and prefs.osmnx_cache_directory:
        cache_dir = bpy.path.abspath(prefs.osmnx_cache_directory)
    else:
        # Default to user scripts folder
        user_scripts = bpy.utils.resource_path('USER')
        cache_dir = os.path.join(user_scripts, "scripts", "addons", "scigraphs_osmnx_cache")
    
    return cache_dir


def ensure_cache_directory():
    """
    Create cache directory if it doesn't exist.
    
    Returns:
        True if directory exists or was created, False on error
    """
    try:
        cache_dir = get_cache_directory()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log(f"Error creating cache directory: {e}")
        return False


def sanitize_filename(name):
    """
    Sanitize a string to be safe for use as a filename.
    
    Args:
        name: String to sanitize
        
    Returns:
        Sanitized string safe for filenames
    """
    # Remove or replace invalid filename characters
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Replace spaces and commas with underscores
    name = re.sub(r'[\s,]+', '_', name)
    # Remove multiple consecutive underscores
    name = re.sub(r'_+', '_', name)
    # Remove leading/trailing underscores
    name = name.strip('_')
    # Limit length
    if len(name) > 100:
        name = name[:100]
    return name


def generate_cache_filename(obj):
    """
    Generate descriptive filename from object metadata.
    
    Args:
        obj: Blender object with OSMnx metadata
        
    Returns:
        Filename (without path) for the cached graph, or None if metadata missing
    """
    if not obj or not obj.get("is_osmnx", False):
        return None
    
    query_name = obj.get("osmnx_query_name", "")
    network_type = obj.get("osmnx_network_type", "drive")
    
    if not query_name:
        # Fallback to object name if no query name
        query_name = obj.name
    
    # Sanitize the query name for use in filename
    safe_name = sanitize_filename(query_name)
    safe_network = sanitize_filename(network_type)
    
    # Generate descriptive filename
    filename = f"{safe_name}_{safe_network}.graphml"
    
    return filename


def get_cache_filepath(obj):
    """
    Get full path to cached graph file for an object.
    
    Args:
        obj: Blender object with OSMnx metadata
        
    Returns:
        Full filepath to cached graph, or None if cannot be determined
    """
    filename = generate_cache_filename(obj)
    if not filename:
        return None
    
    cache_dir = get_cache_directory()
    return os.path.join(cache_dir, filename)


def save_graph_to_cache(obj, G):
    """
    Save graph to cache with descriptive filename.
    
    Args:
        obj: Blender object with OSMnx metadata
        G: OSMnx MultiDiGraph to save
        
    Returns:
        Tuple of (success: bool, filepath: str or None, message: str)
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
    """
    Load graph from cache based on object metadata.
    
    Args:
        obj: Blender object with OSMnx metadata
        
    Returns:
        OSMnx MultiDiGraph or None if not found or error
    """
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
    """
    Return list of all cached graph files with metadata.
    
    Returns:
        List of tuples: (filename, filepath, file_size_mb, modified_time)
    """
    cache_dir = get_cache_directory()
    
    if not os.path.exists(cache_dir):
        return []
    
    cached_graphs = []
    
    try:
        for filename in os.listdir(cache_dir):
            if filename.endswith('.graphml'):
                filepath = os.path.join(cache_dir, filename)
                
                # Get file stats
                stat = os.stat(filepath)
                size_mb = stat.st_size / (1024 * 1024)  # Convert to MB
                modified_time = stat.st_mtime
                
                cached_graphs.append((filename, filepath, size_mb, modified_time))
        
        # Sort by modification time (newest first)
        cached_graphs.sort(key=lambda x: x[3], reverse=True)
        
    except Exception as e:
        log(f"Error listing cached graphs: {e}")
    
    return cached_graphs


def delete_cached_graph(filepath):
    """
    Delete a specific cached graph file.
    
    Args:
        filepath: Full path to the cached graph file
        
    Returns:
        True on success, False on error
    """
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
    """
    Delete all cached graph files.
    
    Returns:
        Tuple of (success_count, error_count)
    """
    cached_graphs = list_cached_graphs()
    success_count = 0
    error_count = 0
    
    for filename, filepath, _, _ in cached_graphs:
        if delete_cached_graph(filepath):
            success_count += 1
        else:
            error_count += 1
    
    return success_count, error_count

