# UI module for SciGraphs addon

from . import operators
from . import panels
from . import menus
from . import pie_menus
from . import overlays
from . import tools
from . import gizmos
from . import coloring
from . import modal_visual


def register():
    """Register all UI components."""
    operators.register()
    panels.register()
    menus.register()
    pie_menus.register()
    overlays.register()
    tools.register()
    gizmos.register()
    coloring.register()
    modal_visual.register()


def unregister():
    """Unregister all UI components."""
    modal_visual.unregister()
    coloring.unregister()
    gizmos.unregister()
    tools.unregister()
    overlays.unregister()
    pie_menus.unregister()
    menus.unregister()
    panels.unregister()
    operators.unregister()
