# Shared runtime state. Imports nothing from the package; anyone may import it.

import bpy

HANDLE = None
ENABLED = False

# Guard against re-entrancy while display_type is tweaked from a handler.
SYNCING = False

# Object pointer -> batch bundle; see batches.build_bundle.
CACHE = {}

# {object_name: original_display_type}, so 'BOUNDS' stand-ins can be restored.
HIDDEN = {}

SIZE_BUCKETS = 12


def preview_prop(scene, name, default):
    return getattr(scene, f"scigraphs_preview_{name}", default)


def is_graph_object(obj):
    return (
        obj is not None
        and obj.type == 'MESH'
        and "num_nodes" in obj
        and len(obj.data.vertices) > 0
    )


def is_enabled():
    scene = getattr(bpy.context, "scene", None)
    return ENABLED and scene is not None and bool(preview_prop(scene, "enabled", False))


def tag_redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
