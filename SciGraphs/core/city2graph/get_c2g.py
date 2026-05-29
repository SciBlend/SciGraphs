from ...utils.logger import log


def get_city2graph():
    """
    Import and return city2graph module, or None if not available.
    
    Returns:
        city2graph module or None if import fails
    """
    try:
        import city2graph as c2g
        return c2g
    except ImportError as e:
        log(f"city2graph is not available: {e}")
        return None

