# GPU-driven viewport preview of graphs; the drawing half lives in core/render.

import bpy

from . import draw, dynamic, engine, host, operators, panel, picking, playback, properties, visibility
from .attributes import compute_visible_mask, write_visibility_attribute
from .toon import panel as toon_panel
from .draw import (
    apply_display_engine,
    disable_preview,
    enable_preview,
    invalidate,
    invalidate_geometry,
    refresh_positions,
)
from .state import is_enabled, preview_prop

__all__ = [
    "register",
    "unregister",
    "is_enabled",
    "invalidate",
    "invalidate_geometry",
    "refresh_positions",
    "enable_preview",
    "disable_preview",
    "apply_display_engine",
    "compute_visible_mask",
    "write_visibility_attribute",
]

_CLASSES = [
    operators.SCIGRAPHS_OT_toggle_gpu_preview,
    operators.SCIGRAPHS_OT_refresh_gpu_preview,
    operators.SCIGRAPHS_OT_animate_layout,
    operators.SCIGRAPHS_OT_reset_animation,
    operators.SCIGRAPHS_OT_set_edge_attr,
    operators.SCIGRAPHS_OT_bake_edge_channel,
    picking.SCIGRAPHS_OT_pick_node,
    operators.SCIGRAPHS_OT_set_preview_attr,
    operators.SCIGRAPHS_OT_filter_add,
    operators.SCIGRAPHS_OT_filter_remove,
    operators.SCIGRAPHS_OT_filter_move,
    operators.SCIGRAPHS_OT_filter_seed,
    panel.SCIGRAPHS_UL_filter_stack,
    visibility.SCIGRAPHS_OT_measure_visibility,
    visibility.SCIGRAPHS_OT_refine_cut,
    visibility.SCIGRAPHS_OT_benchmark_visibility,
    panel.SCIGRAPHS_PT_gpu_preview,
    panel.SCIGRAPHS_RENDER_PT_engine,
    panel.SCIGRAPHS_RENDER_PT_lighting,
    # After their parent, since a child cannot register before its bl_parent_id.
    *toon_panel.CLASSES,
    engine.SciGraphsRenderEngine,
]

_PANEL_EXCLUDE = {
    "RENDER_PT_eevee_next_sampling",
    "RENDER_PT_freestyle",
}

# Wanted despite not advertising BLENDER_RENDER: DoF lives on the camera panel.
_EXTRA_PANEL_NAMES = (
    "DATA_PT_camera_dof",
    "DATA_PT_camera_dof_aperture",
)


def _compatible_panels():
    panels = []
    for cls in bpy.types.Panel.__subclasses__():
        engines = getattr(cls, "COMPAT_ENGINES", None)
        if engines and 'BLENDER_RENDER' in engines and cls.__name__ not in _PANEL_EXCLUDE:
            panels.append(cls)
    for name in _EXTRA_PANEL_NAMES:
        cls = getattr(bpy.types, name, None)
        if cls is not None and getattr(cls, "COMPAT_ENGINES", None) is not None:
            panels.append(cls)
    return panels


def register():
    # host caches the shape of the property set; drop it when that changes.
    host.invalidate_properties()
    properties.register_properties()
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    for cls in _compatible_panels():
        cls.COMPAT_ENGINES.add(engine.SciGraphsRenderEngine.bl_idname)

    if draw._on_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(draw._on_load_post)

    # GPU is the default engine, so imported graphs draw on it straight away.
    scene = getattr(bpy.context, "scene", None)
    if scene is not None and bool(preview_prop(scene, "enabled", True)):
        enable_preview()


def unregister():
    if draw._on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(draw._on_load_post)
    # Trajectories are dropped, not saved: unregistering must not edit the scene.
    playback.unregister_handler()
    disable_preview()

    engine_id = engine.SciGraphsRenderEngine.bl_idname
    for cls in _compatible_panels():
        cls.COMPAT_ENGINES.discard(engine_id)

    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
    properties.unregister_properties()
    host.invalidate_properties()
