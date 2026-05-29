from ...utils.logger import log


def get_osmnx():
    """Import and return osmnx module, or None if not available."""
    try:
        import osmnx as ox
        return ox
    except ImportError:
        log("OSMnx is not available")
        return None

