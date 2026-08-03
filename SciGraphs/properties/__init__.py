# Properties module for SciGraphs addon

from . import scene_properties
from . import city2graph_properties
from . import viz_properties

def register():
    scene_properties.register()
    city2graph_properties.register()
    viz_properties.register()

def unregister():
    viz_properties.unregister()
    city2graph_properties.unregister()
    scene_properties.unregister()

