# Edge backbone and community coarsening into supernodes. Size comes from an
# explicit choice, not the community id. Deterministic: F12 matches viewport.

import math

import numpy as np


def edge_weight_candidates(attr_name):
    """GeoDataFrame imports prefix edge columns, so ``travel_time`` may have
    landed as ``edge_travel_time``. No match silently means uniform weights."""
    names = []
    for candidate in (attr_name, f"edge_{attr_name}" if attr_name else None,
                      "weight", "edge_weight"):
        if candidate and candidate not in names:
            names.append(candidate)
    return names


def resolve_edge_weight_attr(mesh, attr_name):
    """Separate from ``edge_weights_raw``, so a caller can name what it ranks."""
    num_edges = len(mesh.edges)
    if num_edges == 0:
        return None
    for name in edge_weight_candidates(attr_name):
        attr = mesh.attributes.get(name)
        if attr is None or attr.domain != 'EDGE' \
                or attr.data_type not in ('FLOAT', 'INT'):
            continue
        if len(attr.data) != num_edges:
            continue
        return name
    return None


def edge_weights_raw(mesh, attr_name):
    """Raw EDGE scalars, or None, tried in ``edge_weight_candidates`` order."""
    num_edges = len(mesh.edges)
    if num_edges == 0:
        return None
    name = resolve_edge_weight_attr(mesh, attr_name)
    if name is None:
        return None
    attr = mesh.attributes.get(name)
    values = np.empty(
        num_edges,
        dtype=np.float32 if attr.data_type == 'FLOAT' else np.int32)
    try:
        attr.data.foreach_get("value", values)
    except (RuntimeError, TypeError):
        return None
    values = values.astype(np.float64)
    values[~np.isfinite(values)] = 0.0
    return values


def point_int_labels(mesh, attr_name):
    values = point_scalars_raw(mesh, attr_name)
    if values is None:
        return None
    return np.rint(values).astype(np.int64)


def point_scalars_raw(mesh, attr_name):
    """Raw, so a community of 200 nodes reads twice the area of one of 100."""
    if not attr_name:
        return None
    attr = mesh.attributes.get(attr_name)
    if attr is None or attr.domain != 'POINT' \
            or attr.data_type not in ('INT', 'INT8', 'FLOAT'):
        return None
    n = len(mesh.vertices)
    if len(attr.data) != n or n == 0:
        return None
    values = np.empty(
        n, dtype=np.float32 if attr.data_type == 'FLOAT' else np.int32)
    try:
        attr.data.foreach_get("value", values)
    except (RuntimeError, TypeError):
        return None
    values = values.astype(np.float64)
    values[~np.isfinite(values)] = 0.0
    return values


# The structural half is core.render.simplify; the mesh readers stay here.
from ...core.render.simplify import (  # noqa: E402,F401
    aggregate_per_community, backbone_mask, build_coarse_level,
    build_hierarchy, community_radii, stand_in_radii, weighted_edge_list,
)
