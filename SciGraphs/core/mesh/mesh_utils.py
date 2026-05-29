# Utilities for bridging graph data with Blender mesh objects.
#
# Provides functions to parse graph topology from Blender object properties,
# expand per-node values to per-vertex arrays (handling OSMnx curve vertices),
# extract vertex positions, and collect/create mesh attributes.

import numpy as np
from ..algorithms.graph import GraphData


def parse_graph_data(obj):
    """
    Parse basic graph topology from a Blender object's custom properties.

    Returns:
        GraphData with nodes and edges lists.
    """
    nodes_str = obj.get("nodes_data", "")
    nodes = nodes_str.split(",") if nodes_str else []

    edges_str = obj.get("edges_data", "")
    if edges_str:
        edges_flat = edges_str.split(",")
        edges = [(edges_flat[i], edges_flat[i + 1]) for i in range(0, len(edges_flat), 2)]
    else:
        edges = []

    return GraphData(nodes, edges, None)


def parse_graph_data_filtered(obj):
    """
    Parse graph data filtering by ``is_intersection`` when present.

    For OSMnx graphs only nodes marked ``is_intersection == 1`` are real
    network nodes; extra vertices are curve interpolation points.

    Returns:
        GraphData containing only intersection nodes.
    """
    mesh = obj.data

    nodes_str = obj.get("nodes_data", "")
    nodes_list = nodes_str.split(",") if nodes_str else []

    edges_str = obj.get("edges_data", "")
    if edges_str:
        edges_flat = edges_str.split(",")
        edges_list = [(edges_flat[i], edges_flat[i + 1]) for i in range(0, len(edges_flat), 2)]
    else:
        edges_list = []

    if "is_intersection" in mesh.attributes:
        is_intersection_attr = mesh.attributes["is_intersection"]

        intersection_count = sum(
            1 for i in range(len(mesh.vertices))
            if is_intersection_attr.data[i].value == 1
        )

        if len(nodes_list) == intersection_count:
            return GraphData(nodes=nodes_list, edges=edges_list)

        if len(nodes_list) > intersection_count:
            nodes_list = nodes_list[:intersection_count]

    return GraphData(nodes=nodes_list, edges=edges_list)


def expand_node_values_to_mesh(obj, node_values, default_value=0.0):
    """
    Expand per-node values to a per-vertex list matching the mesh vertex count.

    OSMnx meshes contain additional curve-point vertices beyond the real graph
    nodes.  This function maps each node value to its intersection vertex and
    fills non-intersection vertices with *default_value*.

    Handles numpy arrays and scalar types transparently.
    """
    mesh = obj.data
    num_mesh_verts = len(mesh.vertices)

    def _to_native(val):
        if hasattr(val, 'item'):
            return val.item()
        if hasattr(val, 'tolist'):
            return val.tolist()
        return val

    if hasattr(node_values, 'tolist'):
        node_values = node_values.tolist()
    else:
        node_values = [_to_native(v) for v in node_values]

    num_nodes = len(node_values)

    if num_mesh_verts == num_nodes:
        return node_values

    if "is_intersection" in mesh.attributes:
        is_intersection_attr = mesh.attributes["is_intersection"]
        expanded = []
        node_idx = 0

        for i in range(num_mesh_verts):
            if is_intersection_attr.data[i].value == 1 and node_idx < num_nodes:
                expanded.append(node_values[node_idx])
                node_idx += 1
            else:
                expanded.append(_to_native(default_value))

        return expanded

    expanded = list(node_values) + [_to_native(default_value)] * (num_mesh_verts - num_nodes)
    return expanded


def get_vertex_positions(obj):
    """Extract vertex positions from a mesh object as a numpy array of shape (N, 3)."""
    mesh = obj.data
    num_verts = len(mesh.vertices)

    positions = np.zeros((num_verts, 3), dtype=np.float64)
    for i, vert in enumerate(mesh.vertices):
        positions[i] = vert.co

    return positions


def collect_mesh_attributes(obj):
    """
    Collect all numeric point-domain attributes from a mesh.

    Returns:
        dict mapping attribute name -> list of values, or ``None`` if empty.
    """
    mesh = obj.data
    attributes = {}

    for attr_name in mesh.attributes.keys():
        attr = mesh.attributes[attr_name]
        if attr.domain == 'POINT':
            if attr.data_type == 'FLOAT_COLOR':
                values = []
                for i in range(len(mesh.vertices)):
                    color = attr.data[i].color
                    avg = (color[0] + color[1] + color[2]) / 3.0
                    values.append(avg)
                attributes[attr_name] = values
            elif attr.data_type in ('FLOAT', 'INT', 'BOOLEAN'):
                values = [attr.data[i].value for i in range(len(mesh.vertices))]
                attributes[attr_name] = values

    return attributes if attributes else None


def create_or_update_attribute(mesh, name, attr_type, values, obj=None):
    """
    Create (or replace) a mesh attribute, padding/truncating *values* to match
    the vertex count.

    Args:
        mesh: ``bpy.types.Mesh``
        name: Attribute name.
        attr_type: ``'FLOAT'`` or ``'INT'``.
        values: Iterable of values (one per graph node).
        obj: Optional Blender object – when given, values are expanded via
             :func:`expand_node_values_to_mesh` first.
    """
    if obj is not None:
        values = expand_node_values_to_mesh(obj, values)

    if hasattr(values, 'tolist'):
        values = values.tolist()
    else:
        converted = []
        for v in values:
            if hasattr(v, 'item'):
                converted.append(v.item())
            elif hasattr(v, 'tolist'):
                converted.append(v.tolist())
            else:
                converted.append(v)
        values = converted

    num_verts = len(mesh.vertices)
    default_val = 0 if attr_type == 'INT' else 0.0

    if len(values) < num_verts:
        values = list(values) + [default_val] * (num_verts - len(values))
    elif len(values) > num_verts:
        values = values[:num_verts]

    if attr_type == 'INT':
        values = [int(v) if v is not None else 0 for v in values]
    else:
        values = [float(v) if v is not None else 0.0 for v in values]

    if name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[name])
    attr = mesh.attributes.new(name=name, type=attr_type, domain='POINT')
    attr.data.foreach_set("value", values)
