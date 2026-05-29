"""Floating horizontal toolbar gizmo for the coloring system.

The gizmo group renders a row of 2D buttons anchored at the top of the
3D viewport. The row is drag-able (left handle), exposes 8 colormap chips
in the middle, and packs the supporting actions (settings popup, refresh
range, reverse, apply, cycle attribute, remove, close) on the right.
"""

from __future__ import annotations

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

from ...core.coloring.colormaps import (
    COLORMAP_ICONS,
    QUICK_COLORMAPS,
    sample_colormap,
)
from . import functions as fn


# ---------------------------------------------------------------------------
# Tooltip overlay
# ---------------------------------------------------------------------------

_TOOLTIP_HANDLE = None
_TOOLTIP_STATE = None
_PREVIEW_HANDLE = None


def _set_tooltip(context, title: str, description: str, x: int, y: int):
    global _TOOLTIP_STATE
    region = getattr(context, "region", None)
    if region is None:
        _TOOLTIP_STATE = None
        return

    description_lines = _wrap_text(description)
    line_height = 16
    width = 320
    height = 38 + len(description_lines) * line_height
    px = max(12, min(int(x - width * 0.5), region.width - width - 12))
    py = max(12, int(y - height - 18))

    _TOOLTIP_STATE = {
        "title": title,
        "lines": description_lines,
        "x": px,
        "y": py,
        "width": width,
        "height": height,
    }


def _clear_tooltip():
    global _TOOLTIP_STATE
    _TOOLTIP_STATE = None


def _wrap_text(text: str, max_chars: int = 50):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines or [text]


def _draw_rect(x, y, width, height, color):
    vertices = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ]
    indices = [(0, 1, 2), (0, 2, 3)]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
    gpu.state.blend_set('ALPHA')
    shader.bind()
    shader.uniform_float("color", color)
    batch.draw(shader)
    gpu.state.blend_set('NONE')


def _draw_text(x, y, text, size, color):
    font_id = 0
    blf.size(font_id, size)
    blf.color(font_id, *color)
    blf.position(font_id, x, y, 0)
    blf.draw(font_id, text)


def _draw_tooltip_callback():
    context = bpy.context
    if context.area is None or context.area.type != 'VIEW_3D':
        return
    if not getattr(context.window_manager, "scigraphs_show_color_toolbar", True):
        return
    if not _TOOLTIP_STATE:
        return

    state = _TOOLTIP_STATE
    x = state["x"]
    y = state["y"]
    width = state["width"]
    height = state["height"]

    _draw_rect(x, y, width, height, (0.04, 0.06, 0.08, 0.92))
    _draw_rect(x, y + height - 22, width, 22, (0.13, 0.30, 0.42, 0.95))
    _draw_text(x + 10, y + height - 16, state["title"], 12, (0.9, 0.97, 1.0, 1.0))

    text_y = y + height - 38
    for line in state["lines"]:
        _draw_text(x + 10, text_y, line, 11, (0.92, 0.92, 0.92, 0.95))
        text_y -= 16


def _draw_preview_callback():
    """Render a thin colormap preview strip + range labels under the toolbar."""
    context = bpy.context
    if context.area is None or context.area.type != 'VIEW_3D':
        return
    if not getattr(context.window_manager, "scigraphs_show_color_toolbar", True):
        return

    obj = fn.active_mesh_object(context)
    if obj is None or not fn.has_scalar_attributes(obj):
        return

    props = fn.coloring_props(context)
    if props is None:
        return

    region = context.region
    if region is None:
        return

    wm = context.window_manager
    # Mirror _toolbar_origin in the gizmo group so the strip sits below it.
    button_count = len(QUICK_COLORMAPS) + 8
    spacing = 30
    width = button_count * spacing
    if wm.scigraphs_color_toolbar_x < 0:
        center_x = region.width * 0.5
    else:
        center_x = wm.scigraphs_color_toolbar_x
    if wm.scigraphs_color_toolbar_y < 0:
        center_y = region.height - 64
    else:
        center_y = wm.scigraphs_color_toolbar_y

    strip_width = max(220, width - 60)
    strip_height = 10
    x = int(center_x - strip_width * 0.5)
    y = int(center_y - 32)

    samples = sample_colormap(
        props.colormap, samples=64, reverse=props.reverse,
    )
    if samples.size == 0:
        return

    seg_w = strip_width / samples.shape[0]
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    for i, c in enumerate(samples):
        seg_x = x + i * seg_w
        verts = [
            (seg_x, y),
            (seg_x + seg_w + 1.0, y),
            (seg_x + seg_w + 1.0, y + strip_height),
            (seg_x, y + strip_height),
        ]
        batch = batch_for_shader(
            shader, 'TRIS', {"pos": verts}, indices=[(0, 1, 2), (0, 2, 3)]
        )
        shader.bind()
        shader.uniform_float(
            "color",
            (float(c[0]), float(c[1]), float(c[2]), float(c[3])),
        )
        batch.draw(shader)
    gpu.state.blend_set('NONE')

    attribute_name = fn.resolve_attribute_name(props, obj) or "<no attribute>"
    label = f"{attribute_name}  ·  {props.colormap}"
    if props.reverse:
        label += "  (reversed)"
    _draw_text(x, y + strip_height + 4, label, 11, (0.85, 0.92, 1.0, 0.95))

    range_text = f"[{props.last_vmin:.4g} … {props.last_vmax:.4g}]"
    blf.size(0, 11)
    text_w, _ = blf.dimensions(0, range_text)
    _draw_text(
        x + strip_width - int(text_w),
        y + strip_height + 4,
        range_text,
        11,
        (0.78, 0.86, 0.96, 0.85),
    )


def enable_color_overlay():
    global _TOOLTIP_HANDLE, _PREVIEW_HANDLE
    if _TOOLTIP_HANDLE is None:
        _TOOLTIP_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_tooltip_callback, (), 'WINDOW', 'POST_PIXEL'
        )
    if _PREVIEW_HANDLE is None:
        _PREVIEW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            _draw_preview_callback, (), 'WINDOW', 'POST_PIXEL'
        )


def disable_color_overlay():
    global _TOOLTIP_HANDLE, _PREVIEW_HANDLE
    if _TOOLTIP_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_TOOLTIP_HANDLE, 'WINDOW')
        _TOOLTIP_HANDLE = None
    if _PREVIEW_HANDLE is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_PREVIEW_HANDLE, 'WINDOW')
        _PREVIEW_HANDLE = None
    _clear_tooltip()


# ---------------------------------------------------------------------------
# Toolbar layout helpers
# ---------------------------------------------------------------------------

def _toolbar_origin(context, button_count: int):
    """Return ``(x, y, spacing)`` for the leftmost button of the toolbar."""
    wm = context.window_manager
    region = context.region
    spacing = 30
    width = button_count * spacing

    x = wm.scigraphs_color_toolbar_x
    y = wm.scigraphs_color_toolbar_y

    if x < 0:
        x = max(40, (region.width - width) * 0.5 + spacing * 0.5)
    else:
        # Anchor the stored center to the leftmost button.
        x = x - width * 0.5 + spacing * 0.5

    if y < 0:
        y = region.height - 64

    return x, y, spacing


# ---------------------------------------------------------------------------
# Action descriptors
# ---------------------------------------------------------------------------

_CHIP_TOOLTIPS = {
    "viridis": "Viridis – perceptually uniform sequential. Good default for densities.",
    "plasma": "Plasma – sequential, vivid contrast for centrality / accessibility.",
    "inferno": "Inferno – sequential dark-to-light, ideal on dark viewports.",
    "magma": "Magma – sequential, smoother than Inferno for subtle gradients.",
    "turbo": "Turbo – high-contrast rainbow, useful for spotting outliers.",
    "cividis": "Cividis – colorblind-friendly sequential.",
    "coolwarm": "Cool-Warm – diverging blue-red, perfect for signed metrics.",
    "RdYlBu": "Red-Yellow-Blue – diverging, popular for elevation / temperature.",
}


_AUX_BUTTONS = (
    {
        "id": "settings",
        "icon": 'PREFERENCES',
        "operator": "scigraphs.color_settings_dialog",
        "color": (0.10, 0.16, 0.26),
        "highlight": (0.20, 0.45, 0.70),
        "tooltip": (
            "Open the full settings popup: attribute, colormap, range, "
            "opacity, color domain, and material auto-setup."
        ),
        "label": "Settings",
    },
    {
        "id": "refresh_range",
        "icon": 'FILE_REFRESH',
        "operator": "scigraphs.color_refresh_range",
        "color": (0.06, 0.22, 0.18),
        "highlight": (0.18, 0.55, 0.42),
        "tooltip": (
            "Recalculate vmin/vmax from the active attribute and "
            "re-color the mesh with the current colormap."
        ),
        "label": "Refresh Range",
    },
    {
        "id": "reverse",
        "icon": 'ARROW_LEFTRIGHT',
        "operator": "scigraphs.color_toggle_reverse",
        "color": (0.18, 0.10, 0.28),
        "highlight": (0.45, 0.20, 0.60),
        "tooltip": (
            "Flip the orientation of the colormap and re-apply. "
            "Handy when low values should look 'hot'."
        ),
        "label": "Reverse Colormap",
    },
    {
        "id": "pick_attribute",
        "icon": 'PRESET',
        "operator": "scigraphs.color_pick_attribute",
        "color": (0.20, 0.18, 0.06),
        "highlight": (0.55, 0.45, 0.10),
        "tooltip": (
            "Open a popup listing every scalar attribute on the active mesh; "
            "pick one to switch the coloring to it."
        ),
        "label": "Pick Attribute",
    },
    {
        "id": "apply",
        "icon": 'BRUSH_DATA',
        "operator": "scigraphs.color_apply",
        "color": (0.10, 0.26, 0.16),
        "highlight": (0.25, 0.62, 0.36),
        "tooltip": "Re-apply the current settings to the active mesh.",
        "label": "Apply",
    },
    {
        "id": "remove",
        "icon": 'TRASH',
        "operator": "scigraphs.color_remove",
        "color": (0.22, 0.10, 0.10),
        "highlight": (0.60, 0.18, 0.18),
        "tooltip": "Remove the most recent color attribute created by the toolbar.",
        "label": "Remove Coloring",
    },
)


# ---------------------------------------------------------------------------
# Gizmo group
# ---------------------------------------------------------------------------

class SCIGRAPHS_GGT_color_toolbar(bpy.types.GizmoGroup):
    """Floating horizontal coloring toolbar."""
    bl_idname = "SCIGRAPHS_GGT_color_toolbar"
    bl_label = "SciGraphs Color Toolbar"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'PERSISTENT', 'SCALE'}

    @classmethod
    def poll(cls, context):
        if not getattr(context.window_manager, "scigraphs_show_color_toolbar", True):
            return False
        # Only show the toolbar when an active mesh has scalar attributes the
        # user could colorize. Otherwise it would just clutter the viewport.
        return fn.has_scalar_attributes(fn.active_mesh_object(context))

    # Attributes initialised in setup() because Blender constructs the
    # GizmoGroup itself; declaring them here keeps Pylint happy.
    _move_button = None
    _chip_buttons = ()
    _aux_buttons = ()
    _close_button = None

    def setup(self, _context):
        self._move_button = self._make_button(
            'VIEW_PAN',
            "scigraphs.color_drag_toolbar",
            color=(0.06, 0.16, 0.24),
            highlight=(0.18, 0.42, 0.62),
        )

        self._chip_buttons = []
        for cmap in QUICK_COLORMAPS:
            icon = COLORMAP_ICONS.get(cmap, 'COLOR')
            button = self._make_button(
                icon,
                "scigraphs.color_set_colormap",
                color=(0.10, 0.10, 0.16),
                highlight=(0.32, 0.55, 0.80),
                op_attrs={"colormap": cmap, "apply_now": True},
            )
            self._chip_buttons.append((cmap, button))

        self._aux_buttons = []
        for descriptor in _AUX_BUTTONS:
            button = self._make_button(
                descriptor["icon"],
                descriptor["operator"],
                color=descriptor["color"],
                highlight=descriptor["highlight"],
            )
            self._aux_buttons.append((descriptor, button))

        self._close_button = self._make_button(
            'PANEL_CLOSE',
            "scigraphs.color_toggle_toolbar",
            color=(0.32, 0.08, 0.08),
            highlight=(0.72, 0.18, 0.18),
        )

    def _make_button(self, icon, operator, color, highlight, op_attrs=None):
        button = self.gizmos.new("GIZMO_GT_button_2d")
        button.icon = icon
        button.color = color
        button.color_highlight = highlight
        button.alpha = 0.62
        button.alpha_highlight = 0.95
        button.scale_basis = 13
        op_props = button.target_set_operator(operator)
        if op_attrs:
            for key, value in op_attrs.items():
                setattr(op_props, key, value)
        return button

    def draw_prepare(self, context):
        all_buttons = (
            [self._move_button]
            + [btn for _, btn in self._chip_buttons]
            + [btn for _, btn in self._aux_buttons]
            + [self._close_button]
        )

        x, y, spacing = _toolbar_origin(context, len(all_buttons))

        for index, button in enumerate(all_buttons):
            button.matrix_basis = Matrix.Translation((x + index * spacing, y, 0))

        tooltip = self._tooltip_for_hover(x, y, spacing)
        if tooltip is None:
            _clear_tooltip()
        else:
            title, description, button_x, button_y = tooltip
            _set_tooltip(context, title, description, button_x, button_y)

    def _tooltip_for_hover(self, origin_x, origin_y, spacing):
        index = 0
        if getattr(self._move_button, "is_highlight", False):
            return (
                "Move Toolbar",
                "Drag to reposition the floating coloring toolbar.",
                origin_x + index * spacing,
                origin_y,
            )
        index += 1

        for cmap, button in self._chip_buttons:
            if getattr(button, "is_highlight", False):
                description = _CHIP_TOOLTIPS.get(cmap, f"Apply {cmap} colormap.")
                return (
                    f"Colormap · {cmap}",
                    description,
                    origin_x + index * spacing,
                    origin_y,
                )
            index += 1

        for descriptor, button in self._aux_buttons:
            if getattr(button, "is_highlight", False):
                return (
                    descriptor["label"],
                    descriptor["tooltip"],
                    origin_x + index * spacing,
                    origin_y,
                )
            index += 1

        if getattr(self._close_button, "is_highlight", False):
            return (
                "Hide Toolbar",
                "Hide the floating coloring toolbar. Re-enable it from the SciGraphs viewport menu.",
                origin_x + index * spacing,
                origin_y,
            )

        return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_classes = (SCIGRAPHS_GGT_color_toolbar,)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    enable_color_overlay()


def unregister():
    disable_color_overlay()
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
