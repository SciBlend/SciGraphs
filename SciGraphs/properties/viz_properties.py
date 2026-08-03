"""``Scene.scigraphs_viz`` - settings for the interactive Geometry Nodes tree.

``core.mesh.geometry.update_geometry_nodes_parameters`` reads these attributes
and ``core.repro.executor`` sets them by name, so renaming a property here
silently breaks those callers.
"""

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
)
from bpy.types import PropertyGroup

from .callbacks import get_attribute_items

# Defaults follow the simple tree's constants, not the interactive tree's
# untuned socket defaults.
try:  # pragma: no cover - import guard for partial installs
    from ..core.mesh.geometry import (
        DEFAULT_EDGE_ATTR_MULT,
        DEFAULT_EDGE_RESOLUTION,
        DEFAULT_EDGE_THICKNESS,
        DEFAULT_NODE_ATTR_MULT,
        DEFAULT_NODE_RESOLUTION,
        DEFAULT_NODE_SIZE,
    )
except Exception:  # noqa: BLE001 - keep registration working regardless
    DEFAULT_NODE_SIZE = 0.02
    DEFAULT_NODE_RESOLUTION = 10
    DEFAULT_NODE_ATTR_MULT = 1.0
    DEFAULT_EDGE_THICKNESS = 0.005
    DEFAULT_EDGE_RESOLUTION = 8
    DEFAULT_EDGE_ATTR_MULT = 1.0


# Blender keeps no reference to the list a dynamic enum callback returns, so
# it must survive on the Python side or the strings are freed.
_ATTRIBUTE_ITEMS_CACHE = []


def _attribute_items(self, context):
    """Attribute enum with an explicit 'NONE' first entry.

    Without a leading sentinel the enum default is whatever attribute sorts
    first (Blender's own ``.select_vert``), binding sizing or filtering to a
    bogus attribute.
    """
    global _ATTRIBUTE_ITEMS_CACHE
    items = [('NONE', "(None)", "No attribute - uniform value")]
    for entry in get_attribute_items(self, context):
        if entry and entry[0] != 'NONE':
            items.append(entry)
    _ATTRIBUTE_ITEMS_CACHE = items
    return _ATTRIBUTE_ITEMS_CACHE


def _viz_update(self, context):
    """Push the changed value onto the active object's modifier, live.

    Best-effort: a property edit must never raise out of the UI, and the simple
    tree has no interface sockets, so the push is then a silent no-op.
    """
    obj = getattr(context, "active_object", None) if context else None
    if obj is None or obj.type != 'MESH':
        return
    if obj.modifiers.get("SciGraphs_Viz") is None:
        return

    try:
        from ..core.mesh.geometry import update_geometry_nodes_parameters
        update_geometry_nodes_parameters(obj)
    except Exception:  # noqa: BLE001 - never break the properties UI
        pass


class SCIGRAPHS_PG_viz(PropertyGroup):
    """Interactive visualization settings (``scene.scigraphs_viz``)."""

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    node_scale: FloatProperty(
        name="Node Scale",
        description="Base radius of node glyphs (upper bound when driven by an attribute)",
        default=DEFAULT_NODE_SIZE,
        min=0.001,
        max=5.0,
        update=_viz_update,
    )

    node_resolution: IntProperty(
        name="Node Resolution",
        description="Resolution of the generated node primitives",
        default=DEFAULT_NODE_RESOLUTION,
        min=3,
        max=128,
        update=_viz_update,
    )

    node_shape: EnumProperty(
        name="Node Shape",
        description="Mesh primitive instanced on every node",
        items=[
            ('SPHERE', "Sphere", "UV sphere nodes"),
            ('ICOSPHERE', "Icosphere", "Icosphere nodes"),
            ('CUBE', "Cube", "Cube nodes"),
            ('CONE', "Cone", "Cone nodes"),
            ('CYLINDER', "Cylinder", "Cylinder nodes"),
        ],
        default='SPHERE',
        update=_viz_update,
    )

    node_scale_attribute: EnumProperty(
        name="Node Size Attribute",
        description="Mesh attribute driving node size ('NONE' = uniform size)",
        items=_attribute_items,
        update=_viz_update,
    )

    node_scale_multiplier: FloatProperty(
        name="Node Size Multiplier",
        description="Multiplier applied to attribute-driven node scaling",
        default=DEFAULT_NODE_ATTR_MULT,
        min=0.0,
        max=100.0,
        update=_viz_update,
    )

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    edge_thickness: FloatProperty(
        name="Edge Thickness",
        description="Radius of edge tubes (upper bound when driven by an attribute)",
        default=DEFAULT_EDGE_THICKNESS,
        min=0.0,
        max=1.0,
        update=_viz_update,
    )

    edge_resolution: IntProperty(
        name="Edge Resolution",
        description="Number of segments around each edge tube",
        default=DEFAULT_EDGE_RESOLUTION,
        min=3,
        max=64,
        update=_viz_update,
    )

    edge_scale_attribute: EnumProperty(
        name="Edge Width Attribute",
        description="Mesh attribute driving edge thickness ('NONE' = uniform)",
        items=_attribute_items,
        update=_viz_update,
    )

    edge_thickness_multiplier: FloatProperty(
        name="Edge Width Multiplier",
        description="Multiplier applied to attribute-driven edge thickness",
        default=DEFAULT_EDGE_ATTR_MULT,
        min=0.0,
        max=100.0,
        update=_viz_update,
    )

    # ------------------------------------------------------------------
    # Direction arrows
    # ------------------------------------------------------------------

    show_arrows: BoolProperty(
        name="Show Direction Arrows",
        description="Place a cone along each edge to show its direction",
        default=False,
        update=_viz_update,
    )

    arrow_size: FloatProperty(
        name="Arrow Size",
        description="Size of the direction arrows",
        default=0.15,
        min=0.001,
        max=5.0,
        update=_viz_update,
    )

    arrow_position: FloatProperty(
        name="Arrow Position",
        description="Position of the arrow along each edge (0 = source, 1 = target)",
        default=0.7,
        min=0.0,
        max=1.0,
        update=_viz_update,
    )

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    enable_filtering: BoolProperty(
        name="Enable Filtering",
        description="Hide nodes whose filter attribute falls outside the range",
        default=False,
        update=_viz_update,
    )

    filter_attribute: EnumProperty(
        name="Filter Attribute",
        description="Mesh attribute the filter range applies to",
        items=_attribute_items,
        update=_viz_update,
    )

    filter_min: FloatProperty(
        name="Filter Min",
        description="Lower bound of the visible attribute range",
        default=0.0,
        update=_viz_update,
    )

    filter_max: FloatProperty(
        name="Filter Max",
        description="Upper bound of the visible attribute range",
        default=1.0,
        update=_viz_update,
    )


_classes = (SCIGRAPHS_PG_viz,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.scigraphs_viz = bpy.props.PointerProperty(
        type=SCIGRAPHS_PG_viz,
        name="SciGraphs Visualization",
    )


def unregister():
    if hasattr(bpy.types.Scene, "scigraphs_viz"):
        del bpy.types.Scene.scigraphs_viz
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


__all__ = ["SCIGRAPHS_PG_viz", "register", "unregister"]
