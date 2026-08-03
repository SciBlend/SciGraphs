import bpy
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

# ImportError, not ModuleNotFoundError: `from . import x` on an absent submodule
# falls back to an attribute lookup on the half-built package and raises the
# base class.
try:
    from . import gpu_render
except ImportError:
    gpu_render = None


class SCIGRAPHS_PT_engine_missing(bpy.types.Panel):
    """Shown in place of the GPU preview when the render engine is absent.

    Deliberately not an error dialog. The add-on works; one feature is not
    installed, and the panel that would have driven it says so, where the user
    was already looking for it.
    """

    bl_label = "GPU Render Engine"
    bl_idname = "SCIGRAPHS_PT_engine_missing"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SciGraphs"

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Not installed", icon='INFO')
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="The fast GPU preview and the")
        col.label(text="SciGraphs render engine need the")
        col.label(text="scigraphs-engine package.")
        box.label(text="pip install scigraphs-engine")
        col = layout.column(align=True)
        col.scale_y = 0.8
        col.label(text="Everything else works:", icon='CHECKMARK')
        for line in ("import and layouts",
                     "analysis and colouring",
                     "the Geometry Nodes path"):
            col.label(text="    " + line)


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
    if gpu_render is not None:
        gpu_render.register()
    else:
        bpy.utils.register_class(SCIGRAPHS_PT_engine_missing)


def unregister():
    """Unregister all UI components."""
    if gpu_render is not None:
        gpu_render.unregister()
    else:
        bpy.utils.unregister_class(SCIGRAPHS_PT_engine_missing)
    modal_visual.unregister()
    coloring.unregister()
    gizmos.unregister()
    tools.unregister()
    overlays.unregister()
    pie_menus.unregister()
    menus.unregister()
    panels.unregister()
    operators.unregister()
