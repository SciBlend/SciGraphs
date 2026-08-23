# Mesh extraction, plus the spatial blocks the draw loop culls and sorts by.

import numpy as np


def extract_node_coords(mesh):
    """(N, 3) float32 node positions, in object space."""
    num_verts = len(mesh.vertices)
    coords = np.empty(num_verts * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", coords)
    return coords.reshape(num_verts, 3)


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
