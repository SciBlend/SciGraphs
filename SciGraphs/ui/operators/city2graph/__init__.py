from . import data_ops
from . import morphology_ops
from . import transport_ops
from . import proximity_graph_ops
from . import metapath_operators
from . import mobility_ops
from . import graph_tools_ops


def register():
    data_ops.register()
    morphology_ops.register()
    transport_ops.register()
    proximity_graph_ops.register()
    metapath_operators.register()
    mobility_ops.register()
    graph_tools_ops.register()


def unregister():
    graph_tools_ops.unregister()
    mobility_ops.unregister()
    metapath_operators.unregister()
    proximity_graph_ops.unregister()
    transport_ops.unregister()
    morphology_ops.unregister()
    data_ops.unregister()
