from ...utils.logger import log
from .get_osmnx import get_osmnx


def save_graph_graphml(G, filepath):
    """
    Save graph to GraphML format.
    
    Args:
        G: OSMnx MultiDiGraph
        filepath: Output file path
    
    Returns:
        True on success, False on error
    """
    ox = get_osmnx()
    if ox is None or G is None:
        return False
    
    try:
        ox.save_graphml(G, filepath)
        log(f"Graph saved to {filepath}")
        return True
    except Exception as e:
        log(f"Error saving graph: {e}")
        return False


def load_graph_graphml(filepath):
    """
    Load graph from GraphML format.
    
    Args:
        filepath: Input file path
    
    Returns:
        OSMnx MultiDiGraph, or None on error
    """
    ox = get_osmnx()
    if ox is None:
        return None
    
    try:
        G = ox.load_graphml(filepath)
        log(f"Graph loaded from {filepath}")
        return G
    except Exception as e:
        log(f"Error loading graph: {e}")
        return None

