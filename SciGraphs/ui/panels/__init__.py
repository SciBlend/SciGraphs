# Panel modules for SciGraphs addon

import bpy

from . import scigraphs
from . import osmnx
from . import city2graph
from . import repro_panel


class SCIGRAPHS_PT_main(bpy.types.Panel):
    """Main SciGraphs panel - root of all subpanels."""
    bl_label = "SciGraphs"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'SciGraphs'

    def draw(self, context):
        from .. import gizmos
        gizmos.set_active_toolbar(context, 'SCIGRAPHS')

        layout = self.layout
        layout.label(text="Graph Visualization Tools")


def register():
    """Register main panel and all subpanel modules."""
    bpy.utils.register_class(SCIGRAPHS_PT_main)
    scigraphs.register()
    osmnx.register()
    city2graph.register()
    repro_panel.register()


def unregister():
    """Unregister all panel modules in reverse order."""
    repro_panel.unregister()
    city2graph.unregister()
    osmnx.unregister()
    scigraphs.unregister()
    bpy.utils.unregister_class(SCIGRAPHS_PT_main)
