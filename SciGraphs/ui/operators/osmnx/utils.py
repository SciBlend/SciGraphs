# Viewport drawing helpers for OSMnx spatial operators.
#
# All graph-cache and mesh-bridge logic has moved to:
#   core.osmnx.graph_cache
#   core.osmnx.mesh_bridge
#
# This module re-exports those symbols under their original names so that
# existing operator imports keep working.

import math
import gpu
from gpu_extras.batch import batch_for_shader

# --- Viewport GPU drawing helpers (stay here, they are UI-only) ------------

_draw_handlers = {}


def _draw_highlight_circle(position_3d, radius, color, segments=32):
    """Draw a circle at a 3D position for highlighting nodes."""
    vertices = []
    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        x = position_3d[0] + radius * math.cos(angle)
        y = position_3d[1] + radius * math.sin(angle)
        z = position_3d[2]
        vertices.append((x, y, z))
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": vertices})
    
    shader.bind()
    shader.uniform_float("color", color)
    
    gpu.state.line_width_set(3.0)
    gpu.state.blend_set('ALPHA')
    batch.draw(shader)
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


def _draw_highlight_line(start_3d, end_3d, color, width=4.0):
    """Draw a line between two 3D positions for highlighting edges."""
    vertices = [start_3d, end_3d]
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'LINES', {"pos": vertices})
    
    shader.bind()
    shader.uniform_float("color", color)
    
    gpu.state.line_width_set(width)
    gpu.state.blend_set('ALPHA')
    batch.draw(shader)
    gpu.state.blend_set('NONE')
    gpu.state.line_width_set(1.0)


def _draw_highlight_point(position_3d, color, size=10.0):
    """Draw a point at a 3D position."""
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'POINTS', {"pos": [position_3d]})
    
    shader.bind()
    shader.uniform_float("color", color)
    
    gpu.state.point_size_set(size)
    gpu.state.blend_set('ALPHA')
    batch.draw(shader)
    gpu.state.blend_set('NONE')
    gpu.state.point_size_set(1.0)


# --- Re-exports from core (backward-compatible aliases) --------------------

from ....core.osmnx.graph_cache import (          # noqa: E402, F401
    get_osmnx_graph as _get_osmnx_graph,
    get_osmnx_graph_diagnostic as _get_osmnx_graph_diagnostic,
    get_unprojected_graph as _get_unprojected_graph,
    store_unprojected_graph as _store_unprojected_graph,
    store_osmnx_graph as _store_osmnx_graph,
)

from ....core.osmnx.mesh_bridge import (           # noqa: E402, F401
    build_edge_mapping as _build_edge_mapping,
    find_mesh_edge_path as _find_mesh_edge_path,
    transfer_edge_attribute_to_mesh as _transfer_edge_attribute_to_mesh,
    transfer_node_attribute_to_mesh as _transfer_node_attribute_to_mesh,
    mark_shortest_path_attributes as _mark_shortest_path_attributes,
)
