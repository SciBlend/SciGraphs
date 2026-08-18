"""Floating top toolbar to color graphs from any scalar mesh attribute."""

from . import properties  # pylint: disable=import-self
from . import functions  # pylint: disable=import-self  # noqa: F401
from . import operators  # pylint: disable=import-self
from . import gizmo  # pylint: disable=import-self

from ...core.mesh.geometry import (
    register_viz_rebuild_hook,
    unregister_viz_rebuild_hook,
)


def register():
    properties.register()
    operators.register()
    gizmo.register()
    # Keeps the material and node marker alive across SciGraphs_Viz rebuilds.
    register_viz_rebuild_hook(functions.reapply_coloring_after_viz_rebuild)


def unregister():
    unregister_viz_rebuild_hook(functions.reapply_coloring_after_viz_rebuild)
    gizmo.unregister()
    operators.unregister()
    properties.unregister()
