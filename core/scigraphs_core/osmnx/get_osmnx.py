from scigraphs_core.logger import log


def get_osmnx():
    try:
        import osmnx as ox
        return ox
    except ImportError:
        log("OSMnx is not available")
        return None

