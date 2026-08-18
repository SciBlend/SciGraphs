
from . import scigraphs
from . import osmnx
from . import city2graph


def register():
    scigraphs.register()
    osmnx.register()
    city2graph.register()


def unregister():
    city2graph.unregister()
    osmnx.unregister()
    scigraphs.unregister()
