from . import graph_operators
from . import spatial_operators
from . import io_operators
from . import elevation_operators
from . import features_operators
from . import geocoding_operators
from . import cache_operators
from . import routing_operators
from . import accessibility_operators
from . import centrality_operators
from . import export_operators


def register():
    graph_operators.register()
    spatial_operators.register()
    io_operators.register()
    elevation_operators.register()
    features_operators.register()
    geocoding_operators.register()
    cache_operators.register()
    routing_operators.register()
    accessibility_operators.register()
    centrality_operators.register()
    export_operators.register()


def unregister():
    export_operators.unregister()
    centrality_operators.unregister()
    accessibility_operators.unregister()
    routing_operators.unregister()
    cache_operators.unregister()
    geocoding_operators.unregister()
    features_operators.unregister()
    elevation_operators.unregister()
    io_operators.unregister()
    spatial_operators.unregister()
    graph_operators.unregister()
