bl_info = {
    "name": "SciGraphs",
    "author": "José Marín",
    "version": (1, 1, 0),
    "blender": (5, 1, 0),
    "location": "3D View > Sidebar > SciGraphs",
    "description": "Create, visualize and analyze graphs from data.",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}


# scigraphs_core is a separate distribution: a wheel in the normal case, and
# <repo>/core in a checkout. Put it within reach before any submodule imports.
from . import _locate_core as _locate_core_module

_locate_core_module.locate()

try:
    import bpy
    from bpy.app.handlers import persistent
except ModuleNotFoundError:
    bpy = None

    def persistent(fn):
        return fn

if bpy is not None:
    from . import preferences
    from . import properties
    from . import ui
    # The subpackage, not the wheel: twenty-one Blender-side modules still live
    # here, and binding the wheel under this name would shadow them.
    from . import core
    from . import utils


@persistent
def _on_file_loaded(dummy):
    """Restore OSMnx graphs from disk cache after a .blend file is opened."""
    from .core.osmnx.graph_cache import restore_all_graphs_from_cache
    restore_all_graphs_from_cache()

    from .api import anim as _anim
    _anim.ensure_handler()
    for obj in bpy.context.scene.objects:
        _anim.apply_frame(obj, bpy.context.scene.frame_current)


def register():
    """Register all addon classes."""
    preferences.register()
    properties.register()
    ui.register()

    if _on_file_loaded not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_file_loaded)

    from .api import anim as _anim
    _anim.ensure_handler()

def unregister():
    """Unregister all addon classes."""
    ui.unregister()
    properties.unregister()
    preferences.unregister()

    # Drop the active GTFS DuckDB connection or reloading leaks file handles.
    try:
        from .core.city2graph import transportation as _transportation
        _transportation.clear_active_gtfs()
    except Exception:  # noqa: BLE001
        pass

    if _on_file_loaded in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_file_loaded)

    from .api import anim as _anim
    _anim.remove_handler()

if __name__ == "__main__":
    register()

