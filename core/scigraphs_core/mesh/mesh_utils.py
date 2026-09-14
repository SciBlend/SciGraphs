# Bridge between graph data and Blender mesh objects: reading topology, spreading
# per-node values over the curve vertices OSMnx meshes carry, point attributes.

import json

import numpy as np
from ..algorithms.graph import GraphData


def mesh_edge_pairs(obj, num_nodes):
    """Read ``[v0, v1]`` index pairs from the mesh, where mesh-native objects
    keep topology rather than in ``nodes_data``/``edges_data``. Node vertices
    occupy ``0..num_nodes-1``; a higher index is curve geometry and is dropped."""
    mesh = getattr(obj, "data", None)
    if mesh is None or not hasattr(mesh, "edges") or len(mesh.edges) == 0:
        return []

    raw = np.empty(len(mesh.edges) * 2, dtype=np.int32)
    mesh.edges.foreach_get("vertices", raw)
    raw = raw.reshape(-1, 2)
    keep = (raw[:, 0] < num_nodes) & (raw[:, 1] < num_nodes)
    return raw[keep].tolist()


def layout_edge_pairs(obj):
    """``mesh_edge_pairs`` for a caller about to run a layout. Nothing under
    ``core/mesh/layouts/`` may call ``mesh_edge_pairs`` directly: it duck-types a
    Blender datablock, which keeps its callers out of the Blender-free wheel.
    Returns ``[]`` for a non-graph object, read as "looked, found none"."""
    try:
        num_nodes = int(obj.get("num_nodes", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return []
    if num_nodes <= 0:
        return []
    return mesh_edge_pairs(obj, num_nodes)


def resolve_node_labels(obj, num_nodes):
    """Node identifiers in vertex order for an object with no ``nodes_data``,
    read from a ``node_names`` JSON list. Falls back to stringified vertex
    indices when that is missing, unparseable, the wrong length or duplicated,
    since repeated labels would silently merge nodes in :class:`GraphData`."""
    raw = obj.get("node_names")
    if raw:
        try:
            names = [str(n) for n in json.loads(raw)]
        except (ValueError, TypeError):
            names = None
        if names is not None and len(names) == num_nodes and len(set(names)) == num_nodes:
            return names

    return [str(i) for i in range(num_nodes)]


def _mesh_native_topology(obj, nodes_list):
    """``(nodes, edges)`` for an object whose edges live in the mesh. Edges use
    the same identifiers as *nodes*, so :attr:`GraphData.node_to_index` works."""
    num_nodes = obj.get("num_nodes")
    if num_nodes is None:
        mesh = getattr(obj, "data", None)
        num_nodes = len(nodes_list) or (len(mesh.vertices) if mesh is not None else 0)
    num_nodes = int(num_nodes)

    nodes = nodes_list
    if len(nodes) != num_nodes or len(set(nodes)) != num_nodes:
        nodes = resolve_node_labels(obj, num_nodes)

    edges = [(nodes[v0], nodes[v1]) for v0, v1 in mesh_edge_pairs(obj, num_nodes)]
    return nodes, edges


WEIGHT_ATTRIBUTES = ("weight", "length", "edge_weight", "edge_length", "cost")


def edge_weights_from_mesh(obj, edge_count):
    """The mesh's per-edge weights, or None when it carries none."""
    mesh = getattr(obj, "data", None)
    attributes = getattr(mesh, "attributes", None)
    if attributes is None or edge_count <= 0:
        return None

    for name in WEIGHT_ATTRIBUTES:
        attr = attributes.get(name)
        if attr is None or getattr(attr, "domain", "") != 'EDGE':
            continue
        try:
            values = [item.value for item in attr.data]
        except (AttributeError, RuntimeError, TypeError):
            continue
        if len(values) == edge_count:
            return [float(v) for v in values]
    return None


def parse_graph_data(obj):
    """Parse graph topology from an object, falling back to the mesh edges."""
    nodes_str = obj.get("nodes_data", "")
    nodes = nodes_str.split(",") if nodes_str else []

    edges_str = obj.get("edges_data", "")
    if not edges_str:
        nodes, edges = _mesh_native_topology(obj, nodes)
        data = GraphData(nodes, edges, None)
        _attach_weights(obj, data)
        return data

    edges_flat = edges_str.split(",")
    edges = [(edges_flat[i], edges_flat[i + 1]) for i in range(0, len(edges_flat), 2)]

    data = GraphData(nodes, edges, None)
    _attach_weights(obj, data)
    return data


def _attach_weights(obj, data):
    """Set `edge_weights` when the mesh has them, leave it unset otherwise."""
    weights = edge_weights_from_mesh(obj, len(data.edges))
    if weights is not None:
        data.edge_weights = weights
    return data


def parse_graph_data_filtered(obj):
    """Parse graph topology keeping only the real nodes: in an OSMnx mesh only
    vertices with ``is_intersection == 1`` are network nodes."""
    mesh = obj.data

    nodes_str = obj.get("nodes_data", "")
    nodes_list = nodes_str.split(",") if nodes_str else []

    edges_str = obj.get("edges_data", "")
    if edges_str:
        edges_flat = edges_str.split(",")
        edges_list = [(edges_flat[i], edges_flat[i + 1]) for i in range(0, len(edges_flat), 2)]
    elif "is_intersection" not in mesh.attributes:
        # Mesh-native object: nothing to filter, so read topology from the mesh.
        nodes_list, edges_list = _mesh_native_topology(obj, nodes_list)
        return GraphData(nodes=nodes_list, edges=edges_list)
    else:
        # OSMnx: real nodes are not the first num_nodes vertices, so no map.
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
    """Spread per-node values over every mesh vertex. OSMnx meshes carry
    curve-point vertices on top of the real nodes, so each node value lands on
    its intersection vertex and the rest get *default_value*."""
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
    """Numeric point-domain attributes as ``{name: values}``, or ``None``. Color
    attributes flatten to the mean of their RGB channels."""
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
    """Create or replace a point attribute, fitting *values* to the vertex count.
    ``attr_type`` is 'FLOAT' or 'INT'; short lists pad with zero, long ones
    truncate. Pass *obj* to expand per-node values to per-vertex first."""
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
