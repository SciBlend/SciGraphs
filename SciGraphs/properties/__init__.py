# Properties module for SciGraphs addon

from . import scene_properties
from . import city2graph_properties

def register():
    scene_properties.register()
    city2graph_properties.register()

def unregister():
    city2graph_properties.unregister()
    scene_properties.unregister()

