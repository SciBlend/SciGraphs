# Mesh extraction, plus the spatial blocks the draw loop culls and sorts by.

import numpy as np


def extract_node_coords(mesh):
    """(N, 3) float32 node positions, in object space."""
    num_verts = len(mesh.vertices)
    mixed = shape_key_mix(mesh)
    if mixed is not None:
        return mixed
    coords = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    return coords.reshape(num_verts, 3)


def shape_key_mix(mesh):
    """Vertex positions after relative shape keys, or None when there are none."""
    keys = getattr(mesh, "shape_keys", None)
    if keys is None or not getattr(keys, "key_blocks", None):
        return None
    blocks = list(keys.key_blocks)
    if len(blocks) < 2:
        return None

    num_verts = len(mesh.vertices)
    reference = keys.reference_key or blocks[0]
    base = np.empty(num_verts * 3, dtype=np.float32)
    try:
        reference.data.foreach_get("co", base)
    except (RuntimeError, TypeError, ValueError):
        return None
    base = base.reshape(num_verts, 3)

    if not getattr(keys, "use_relative", True):
        return base

    result = base.copy()
    scratch = np.empty(num_verts * 3, dtype=np.float32)
    for block in blocks:
        if block == reference:
            continue
        value = float(block.value)
        if value == 0.0:
            continue
        try:
            block.data.foreach_get("co", scratch)
        except (RuntimeError, TypeError, ValueError):
            continue
        result += value * (scratch.reshape(num_verts, 3) - base)
    return result


def shape_key_signature(mesh):
    """A cheap fingerprint of the current deformation, for the batch cache."""
    keys = getattr(mesh, "shape_keys", None)
    if keys is None or not getattr(keys, "key_blocks", None):
        return None
    return tuple(round(float(block.value), 5)
                 for block in keys.key_blocks)


def extract_node_mask(mesh):
    """None when the attribute is absent; styled meshes interleave curves."""
    attr = mesh.attributes.get("is_intersection")
    if attr is None or attr.domain != 'POINT':
        return None
    num_verts = len(mesh.vertices)
    raw = np.empty(num_verts, dtype=np.int32)
    try:
        attr.data.foreach_get("value", raw)
    except (RuntimeError, TypeError):
        return None
    return raw == 1


def extract_edges(mesh):
    """(E, 2) int32 edge endpoint indices, or None."""
    num_edges = len(mesh.edges)
    if not num_edges:
        return None
    edges = np.empty(num_edges * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", edges)
    return edges.reshape(num_edges, 2)

from ...core.render.blocks import build_spatial_blocks  # noqa: E402,F401
