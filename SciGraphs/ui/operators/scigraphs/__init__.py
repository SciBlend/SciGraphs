from . import algorithms_operators
from . import analysis_operators
from . import data_operators
from . import edge_style_operators
from . import export_operators
from . import layout_operators
from . import repro_operators
from . import sql_operators
from . import suitesparse_operators
from . import temporal_operators
from . import text_overlay_operators
from . import topology_operators
from . import visualization_operators


def register():
    data_operators.register()
    layout_operators.register()
    analysis_operators.register()
    algorithms_operators.register()
    visualization_operators.register()
    edge_style_operators.register()
    export_operators.register()
    topology_operators.register()
    text_overlay_operators.register()
    sql_operators.register()
    suitesparse_operators.register()
    temporal_operators.register()
    repro_operators.register()


def unregister():
    repro_operators.unregister()
    temporal_operators.unregister()
    suitesparse_operators.unregister()
    sql_operators.unregister()
    text_overlay_operators.unregister()
    topology_operators.unregister()
    export_operators.unregister()
    edge_style_operators.unregister()
    visualization_operators.unregister()
    algorithms_operators.unregister()
    analysis_operators.unregister()
    layout_operators.unregister()
    data_operators.unregister()
