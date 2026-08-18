# SciGraphs

bl_info = {
    "name": "SciGraphs",
    "author": "José Marín",
    "version": (1, 0, 1),
    "blender": (5, 1, 0),
    "location": "3D View > Sidebar > SciGraphs",
    "description": "Create, visualize and analyze graphs from data.",
    "warning": "",
    "doc_url": "",
    "category": "3D View",
}


# Put scigraphs_core within reach before any submodule imports from it. It is
# a separate distribution: installed as a wheel in the normal case, resolved
# from <repo>/core in a checkout. See _locate_core for the two-case search and
# why there is no vendored third case.
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
    # Still the subpackage, not the wheel. The analysis half moved out to
    # `scigraphs_core`, but twenty-one modules stayed here because their job is
    # Blender: building meshes, projecting terrain, driving the scene, replaying
    # a pipeline. Binding the wheel under this name would shadow them, and
    # `from .core.osmnx.graph_cache import ...` below would then resolve through
    # the file system to a package the attribute no longer describes.
    from . import core
    from . import utils


@persistent
def _on_file_loaded(dummy):
    """Restore OSMnx graphs from disk cache after a .blend file is opened."""
    from .core.osmnx.graph_cache import restore_all_graphs_from_cache
    restore_all_graphs_from_cache()


def register():
    """Register all addon classes."""
    preferences.register()
    properties.register()
    ui.register()

    # Register load_post handler for auto-loading cached graphs
    if _on_file_loaded not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_on_file_loaded)

def unregister():
    """Unregister all addon classes."""
    ui.unregister()
    properties.unregister()
    preferences.unregister()

    # Drop any active GTFS DuckDB connection so we don't leak file
    # handles when the add-on is reloaded or disabled.
    try:
        from .core.city2graph import transportation as _transportation
        _transportation.clear_active_gtfs()
    except Exception:  # noqa: BLE001
        pass

    # Unregister load_post handler
    if _on_file_loaded in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_file_loaded)

if __name__ == "__main__":
    register()

