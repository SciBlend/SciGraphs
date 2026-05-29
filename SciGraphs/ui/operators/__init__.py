# Operators module for SciGraphs addon

from . import scigraphs
from . import osmnx
from . import city2graph


def register():
    """Register all operator modules."""
    scigraphs.register()
    osmnx.register()
    city2graph.register()


def unregister():
    """Unregister all operator modules in reverse order."""
    city2graph.unregister()
    osmnx.unregister()
    scigraphs.unregister()
