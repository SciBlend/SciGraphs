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
    if values is None:
        return None
    if vmax > vmin:
        return (values - vmin) / (vmax - vmin)
    return np.zeros(num_verts, dtype=np.float32)


def compute_colors(mesh, num_verts, scene, values, vmin, vmax, st=None):
    """(N, 4) float32 colors, per the scene's color mode."""
    from .host import settings_from_scene
    st = st if st is not None else settings_from_scene(scene)
    mode = st.color_mode

    if mode == 'ATTRIBUTE' and values is not None:
        cmap = st.colormap
        if colormap_exists(cmap):
            rgba = values_to_rgba(
                values, cmap_name=cmap, vmin=vmin, vmax=vmax,
                reverse=bool(st.reverse_colormap),
            )
            return np.asarray(rgba, dtype=np.float32).reshape(-1, 4)

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
