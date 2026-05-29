from . import algorithms_panels
from . import analysis_panels
from . import data_panels
from . import export_panels
from . import layout_panels
from . import topology_panels
from . import visualization_panels


def register():
    data_panels.register()
    layout_panels.register()
    algorithms_panels.register()
    analysis_panels.register()
    visualization_panels.register()
    export_panels.register()
    topology_panels.register()


def unregister():
    topology_panels.unregister()
    export_panels.unregister()
    visualization_panels.unregister()
    analysis_panels.unregister()
    algorithms_panels.unregister()
    layout_panels.unregister()
    data_panels.unregister()
