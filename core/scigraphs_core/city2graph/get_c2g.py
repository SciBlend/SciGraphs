from scigraphs_core.logger import log


def get_city2graph():
    try:
        import city2graph as c2g
        return c2g
    except ImportError as e:
        log(f"city2graph is not available: {e}")
        return None

