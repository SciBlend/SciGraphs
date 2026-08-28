# Mesh attributes as the numpy arrays the batch builder needs.

import numpy as np

from scigraphs_core.coloring.attributes import find_attribute, read_attribute_values
from scigraphs_core.coloring.colormaps import colormap_exists, values_to_rgba
from ...core.render.channels import scale_to_unit  # noqa: F401


def active_point_color_attribute(mesh):
    color_attrs = getattr(mesh, "color_attributes", None)
    if not color_attrs:
        return None
    active = color_attrs.active_color
    if active is None or active.domain != 'POINT':
        return None
    if len(active.data) != len(mesh.vertices):
        return None
    return active


def read_value_channel(mesh, attr_name, num_verts):
    """A POINT scalar attribute as (values, vmin, vmax), or (None, 0, 1)."""
    if not attr_name:
        return None, 0.0, 1.0
    attr = find_attribute(mesh, attr_name)
    if attr is None or getattr(attr, "domain", "") != 'POINT':
        return None, 0.0, 1.0
    values = read_attribute_values(mesh, attr_name)
    if values.size != num_verts:
        return None, 0.0, 1.0
    finite = np.isfinite(values)
    if not finite.any():
        return None, 0.0, 1.0
    vmin = float(values[finite].min())
    vmax = float(values[finite].max())
    return values.astype(np.float32), vmin, vmax


def read_edge_scalar(mesh, attr_name):
    if not attr_name:
        return None
    attr = mesh.attributes.get(attr_name)
    if attr is None or attr.domain != 'EDGE' \
            or attr.data_type not in ('FLOAT', 'INT'):
        return None
    num_edges = len(mesh.edges)
    if len(attr.data) != num_edges or num_edges == 0:
        return None
    values = np.empty(num_edges,
                      dtype=np.float32 if attr.data_type == 'FLOAT' else np.int32)
    try:
        attr.data.foreach_get("value", values)
    except (RuntimeError, TypeError):
        return None
    return values.astype(np.float32)


def read_edge_value_channel(mesh, attr_name, scale='LINEAR'):
    """Unit-scaled. LINEAR flattens weights; see :func:`scale_to_unit`."""
    values = read_edge_scalar(mesh, attr_name)
    if values is None:
        return None
    if not np.isfinite(values).any():
        return None
    return scale_to_unit(values, scale)


def edge_scalar_attr_items(obj):
    from .state import is_graph_object
    if not is_graph_object(obj):
        return []
    return [a.name for a in obj.data.attributes
            if a.domain == 'EDGE' and a.data_type in ('FLOAT', 'INT')
            and not a.name.startswith(".")]


def normalized_values(values, vmin, vmax, num_verts):
    """``(norm, finite)`` over ``[vmin, vmax]``, ``(None, None)`` with no channel.

    Non-finite samples come back as 0.0, never NaN. ``_radii_for`` multiplies
    this array in and ``volume.build`` weighs its density field by it, so one
    unmeasured node in a partially populated attribute was enough to give that
    node a quad of undefined size. ``finite`` comes back too, so a caller that
    wants to drop those nodes rather than draw them at the base radius can,
    without reconstructing the mask from the numbers.
    """
    if values is None:
        return None, None
    finite = np.isfinite(values)
    if vmax > vmin:
        norm = (values - vmin) / (vmax - vmin)
    else:
        norm = np.zeros(num_verts, dtype=np.float32)
    return np.where(finite, norm, 0.0).astype(np.float32), finite


_SRGB_KNEE = 0.04045


def srgb_to_linear(rgb):
    """Inverse sRGB transfer, piecewise rather than a 2.2 gamma. The two agree
    to a hundredth above the knee and are a factor apart below it, and below it
    is where a colormap's dark end sits."""
    a = np.asarray(rgb, dtype=np.float32)
    curve = np.power((np.maximum(a, 0.0) + 0.055) / 1.055, 2.4)
    return np.where(a <= _SRGB_KNEE, a / 12.92, curve).astype(np.float32)


def linearize_rgba(rgba):
    """Colormap RGBA into the scene-linear space the passes are in. Alpha is
    coverage, never a color, so the transfer must not touch it."""
    out = np.array(rgba, dtype=np.float32).reshape(-1, 4)
    out[:, :3] = srgb_to_linear(out[:, :3])
    return out


def compute_colors(mesh, num_verts, scene, values, vmin, vmax, st=None):
    """(N, 4) float32 colors, per the scene's color mode."""
    from .host import settings_from_scene
    st = st if st is not None else settings_from_scene(scene)
    mode = st.color_mode

    if mode == 'ATTRIBUTE' and values is not None:
        cmap = st.colormap
        if colormap_exists(cmap):
            clip_low = float(st.clip_low_pct)
            clip_high = float(st.clip_high_pct)
            clipping = clip_low > 0.0 or clip_high < 100.0
            rgba = values_to_rgba(
                values, cmap_name=cmap,
                vmin=None if clipping else vmin,
                vmax=None if clipping else vmax,
                reverse=bool(st.reverse_colormap),
                norm_mode=st.norm_mode,
                gamma=float(st.norm_gamma),
                clip_low_pct=clip_low,
                clip_high_pct=clip_high,
            )
            return linearize_rgba(rgba)

    if mode in ('VERTEX', 'ATTRIBUTE'):
        color_attr = active_point_color_attribute(mesh)
        if color_attr is not None:
            colors = np.empty(num_verts * 4, dtype=np.float32)
            color_attr.data.foreach_get("color", colors)
            return colors.reshape(num_verts, 4)

    flat = np.array(st.node_color, dtype=np.float32)
    return np.tile(flat, (num_verts, 1))


def compute_visible_mask(obj, scene=None):
    """``(mask, active)``; ``active`` only when a filter is narrowing the set."""
    import bpy
    if scene is None:
        scene = bpy.context.scene
    # Deferred: filters.py reads its channels through this module.
    from . import filters
    node_mask, _, active = filters.masks(obj, scene)
    return node_mask, active


def write_visibility_attribute(obj, mask, name="scigraphs_visible"):
    mesh = obj.data
    if name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[name])
        except RuntimeError:
            pass
    attr = mesh.attributes.new(name=name, type='BOOLEAN', domain='POINT')
    attr.data.foreach_set("value", np.ascontiguousarray(mask, dtype=bool))
    return name


def scalar_attr_items(obj):
    from scigraphs_core.coloring.attributes import list_scalar_attributes
    from .state import is_graph_object
    if not is_graph_object(obj):
        return []
    return list_scalar_attributes(obj.data)
