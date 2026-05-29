"""Floating top toolbar to color graphs from any scalar mesh attribute.

Layout:

* ``properties.py``  – ``Scene.scigraphs_coloring`` PropertyGroup driving the
  active attribute, colormap, value range, and material auto-setup.
* ``functions.py``   – Helper functions used by operators (no Blender RNA in
  the function bodies beyond what is strictly needed to read/write the
  mesh).
* ``operators.py``   – Action operators (``apply``, ``refresh range``,
  ``set colormap``, ``cycle attribute``, ``settings popup``,
  ``toggle toolbar``, ``drag toolbar``).
* ``gizmo.py``       – Horizontal floating ``GizmoGroup`` anchored at the
  top of the 3D viewport.
"""

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
    # Keep the toolbar's material + node marker alive across SciGraphs_Viz
    # rebuilds (centrality, edge styles, ...).
    register_viz_rebuild_hook(functions.reapply_coloring_after_viz_rebuild)


def unregister():
    unregister_viz_rebuild_hook(functions.reapply_coloring_after_viz_rebuild)
    gizmo.unregister()
    operators.unregister()
    properties.unregister()
