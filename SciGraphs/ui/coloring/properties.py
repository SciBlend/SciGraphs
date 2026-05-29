"""PropertyGroup backing the floating coloring toolbar.

The PropertyGroup lives on ``Scene`` so the configuration travels with the
.blend file. Toolbar visibility and screen position live on
``WindowManager`` instead, because they are transient session state.
"""

import bpy
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from ...core.coloring.colormaps import (
    QUICK_COLORMAPS,
    colormap_items_for_enum,
)
from ...core.coloring.attributes import (
    list_scalar_attributes,
)


# ---------------------------------------------------------------------------
# Dynamic enums
# ---------------------------------------------------------------------------

# We keep references to the dynamically generated enum items because Blender
# does not retain them: returning a plain list each call leaks the strings.
_ATTRIBUTE_ITEMS_CACHE = []


def _attribute_enum_items(_self, context):
    """Populate the attribute enum with the active object's scalar attributes."""
    global _ATTRIBUTE_ITEMS_CACHE
    items = [("__NONE__", "<no attribute>", "No scalar attribute available")]

    obj = getattr(context, "active_object", None)
    if obj and obj.type == 'MESH':
        for name, data_type, domain in list_scalar_attributes(obj.data):
            description = f"{data_type} attribute on {domain.title()} domain"
            items.append((name, name, description))

    _ATTRIBUTE_ITEMS_CACHE = items
    return _ATTRIBUTE_ITEMS_CACHE


def _on_attribute_enum_change(self, _context):
    """Mirror the enum selection into the persistent ``attribute_name`` string.

    Blender forbids writing to ID-block properties from a panel/operator
    ``draw`` method, so we keep the bridge in this update callback instead
    of touching ``attribute_name`` while the popup is being drawn.
    """
    value = getattr(self, "attribute_enum", "") or ""
    if value and value != "__NONE__":
        self.attribute_name = value


_COLORMAP_ITEMS_CACHE = colormap_items_for_enum()


def _colormap_enum_items(_self, _context):
    return _COLORMAP_ITEMS_CACHE


# ---------------------------------------------------------------------------
# Property group
# ---------------------------------------------------------------------------

class SCIGRAPHS_PG_coloring(PropertyGroup):
    """Persistent settings that drive the floating coloring toolbar."""

    attribute_name: StringProperty(
        name="Attribute",
        description="Scalar mesh attribute used as input for the colormap",
        default="",
    )

    attribute_enum: EnumProperty(
        name="Attribute",
        description="Pick a scalar attribute available on the active mesh",
        items=_attribute_enum_items,
        update=_on_attribute_enum_change,
    )

    colormap: EnumProperty(
        name="Colormap",
        description="Colormap applied when mapping attribute values to RGBA",
        items=_colormap_enum_items,
        default=None,
    )

    reverse: BoolProperty(
        name="Reverse",
        description="Reverse the colormap (low values get the high-end color)",
        default=False,
    )

    auto_range: BoolProperty(
        name="Auto Range",
        description=(
            "Automatically read vmin/vmax from the selected attribute. "
            "Disable to lock the range to the manual values below"
        ),
        default=True,
    )

    vmin: FloatProperty(
        name="vmin",
        description="Lower bound of the value range mapped to the colormap",
        default=0.0,
    )

    vmax: FloatProperty(
        name="vmax",
        description="Upper bound of the value range mapped to the colormap",
        default=1.0,
    )

    opacity: FloatProperty(
        name="Opacity",
        description="Alpha multiplier applied to every output color",
        min=0.0,
        max=1.0,
        default=1.0,
    )

    color_domain: EnumProperty(
        name="Color Domain",
        description=(
            "Domain used to store the resulting color attribute. "
            "Auto picks POINT for vertex attributes, CORNER otherwise"
        ),
        items=[
            ('AUTO', "Auto", "Pick the closest domain to the source attribute"),
            ('POINT', "Vertex", "Store colors per vertex"),
            ('CORNER', "Corner", "Store colors per face corner"),
        ],
        default='AUTO',
    )

    color_attribute_name: StringProperty(
        name="Color Layer Name",
        description=(
            "Name of the color attribute written by the toolbar. "
            "Leave empty to use '<attribute>_color'"
        ),
        default="",
    )

    auto_setup_material: BoolProperty(
        name="Auto Material",
        description=(
            "Create or reuse a simple shader on the active object so the "
            "color attribute is visible in Material Preview / Rendered modes"
        ),
        default=True,
    )

    nodes_only: BoolProperty(
        name="Nodes Only (skip is_intersection=0)",
        description=(
            "When the mesh has an 'is_intersection' attribute (typical for "
            "OSMnx graphs), only color real nodes (is_intersection=1). "
            "Intermediate curve points along streets stay in the neutral "
            "edge color so the colormap is not diluted by tube vertices"
        ),
        default=True,
    )

    edge_color: FloatVectorProperty(
        name="Edge Color",
        description=(
            "Neutral color used for geometry skipped by the 'Nodes Only' "
            "filter (street tubes, intermediate curve points, ...)"
        ),
        subtype='COLOR',
        size=4,
        min=0.0,
        max=1.0,
        default=(0.18, 0.18, 0.20, 1.0),
    )

    auto_apply_on_chip: BoolProperty(
        name="Apply When Picking Colormap",
        description=(
            "Re-color the mesh immediately after clicking a colormap chip. "
            "Disable to only update the property and apply manually"
        ),
        default=True,
    )

    last_vmin: FloatProperty(
        name="Last vmin",
        description="Last computed lower bound (read-only feedback)",
        default=0.0,
    )

    last_vmax: FloatProperty(
        name="Last vmax",
        description="Last computed upper bound (read-only feedback)",
        default=0.0,
    )


# ---------------------------------------------------------------------------
# WindowManager (transient state)
# ---------------------------------------------------------------------------

def _register_window_manager_properties():
    bpy.types.WindowManager.scigraphs_show_color_toolbar = BoolProperty(
        name="Show SciGraphs Color Toolbar",
        description="Show the floating horizontal coloring toolbar at the top of the 3D View",
        default=True,
    )
    bpy.types.WindowManager.scigraphs_color_toolbar_x = IntProperty(
        name="SciGraphs Color Toolbar X",
        description="Horizontal position for the floating coloring toolbar",
        default=-1,
    )
    bpy.types.WindowManager.scigraphs_color_toolbar_y = IntProperty(
        name="SciGraphs Color Toolbar Y",
        description="Vertical position for the floating coloring toolbar",
        default=-1,
    )


def _unregister_window_manager_properties():
    for attr in (
        "scigraphs_color_toolbar_y",
        "scigraphs_color_toolbar_x",
        "scigraphs_show_color_toolbar",
    ):
        if hasattr(bpy.types.WindowManager, attr):
            delattr(bpy.types.WindowManager, attr)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (SCIGRAPHS_PG_coloring,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.scigraphs_coloring = bpy.props.PointerProperty(
        type=SCIGRAPHS_PG_coloring,
        name="SciGraphs Coloring",
    )
    _register_window_manager_properties()


def unregister():
    _unregister_window_manager_properties()
    if hasattr(bpy.types.Scene, "scigraphs_coloring"):
        del bpy.types.Scene.scigraphs_coloring
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)


__all__ = [
    "SCIGRAPHS_PG_coloring",
    "QUICK_COLORMAPS",
]
