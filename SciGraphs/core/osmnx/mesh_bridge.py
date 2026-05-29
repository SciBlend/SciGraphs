# Bridge between NetworkX graph attributes and Blender mesh attributes.
#
# Handles the mapping between graph edges/nodes and mesh edges/vertices,
# including the non-trivial case of OSMnx meshes where a single graph edge
# corresponds to multiple mesh edges (curved geometry).

from collections import defaultdict


def _iter_edges_with_data(G):
    """Yield ``(u, v, data)`` over G regardless of whether it is a (Multi)Graph
    or a (Multi)DiGraph.

    OSMnx graphs start as ``MultiDiGraph`` but the user may convert them to
    ``DiGraph`` / ``Graph`` via ``to_digraph`` / ``to_undirected``, in which
    case ``G.edges(keys=True)`` raises ``TypeError``.  Using this helper keeps
    callers agnostic.
    """
    if getattr(G, "is_multigraph", lambda: False)():
        for u, v, _k, data in G.edges(keys=True, data=True):
            yield u, v, data
    else:
        for u, v, data in G.edges(data=True):
            yield u, v, data


def build_edge_mapping(obj):
    """
    Map graph edges ``(u, v)`` to lists of mesh edge indices.

    For OSMnx meshes a single graph edge may span several mesh edges because
    intermediate curve-point vertices are inserted between intersection nodes.

    Returns:
        ``dict[(str, str), list[int]]`` – bidirectional mapping.
    """
    mesh = obj.data
    nodes_str = obj.get("nodes_data", "")
    edges_str = obj.get("edges_data", "")

    if not nodes_str or not edges_str:
        return {}

    node_ids = nodes_str.split(",")
    num_intersections = len(node_ids)

    edges_flat = edges_str.split(",")
    graph_edges = [(edges_flat[i], edges_flat[i + 1]) for i in range(0, len(edges_flat), 2)]

    mesh_adj = defaultdict(dict)
    for edge_idx, edge in enumerate(mesh.edges):
        v0, v1 = edge.vertices
        mesh_adj[v0][v1] = edge_idx
        mesh_adj[v1][v0] = edge_idx

    node_to_vert = {node_id: i for i, node_id in enumerate(node_ids)}

    edge_mapping = {}
    processed = set()

    for src_id, tgt_id in graph_edges:
        edge_key = (min(src_id, tgt_id), max(src_id, tgt_id))
        if edge_key in processed:
            continue
        processed.add(edge_key)

        src_vert = node_to_vert.get(src_id)
        tgt_vert = node_to_vert.get(tgt_id)

        if src_vert is None or tgt_vert is None:
            continue

        mesh_edges = find_mesh_edge_path(mesh_adj, src_vert, tgt_vert, num_intersections)

        if mesh_edges:
            edge_mapping[(src_id, tgt_id)] = mesh_edges
            edge_mapping[(tgt_id, src_id)] = mesh_edges

    return edge_mapping


def find_mesh_edge_path(mesh_adj, src_vert, tgt_vert, num_intersections):
    """
    BFS through non-intersection vertices to find the mesh-edge path between
    two intersection vertices.
    """
    if src_vert == tgt_vert:
        return []

    if tgt_vert in mesh_adj[src_vert]:
        return [mesh_adj[src_vert][tgt_vert]]

    visited = {src_vert}
    queue = [(src_vert, [])]

    while queue:
        curr, path = queue.pop(0)

        for neighbor, edge_idx in mesh_adj[curr].items():
            if neighbor in visited:
                continue

            new_path = path + [edge_idx]

            if neighbor == tgt_vert:
                return new_path

            if neighbor >= num_intersections:
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return []


def transfer_edge_attribute_to_mesh(obj, G, attr_name, mesh_attr_name=None):
    """
    Copy a NetworkX edge attribute onto the Blender mesh as an ``EDGE``-domain
    ``FLOAT`` attribute.

    Returns:
        Number of mesh edges that received a value.
    """
    mesh = obj.data
    if mesh_attr_name is None:
        mesh_attr_name = attr_name

    edge_mapping = build_edge_mapping(obj)
    if not edge_mapping:
        return 0

    if mesh_attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[mesh_attr_name])

    attr = mesh.attributes.new(name=mesh_attr_name, type='FLOAT', domain='EDGE')

    values = [0.0] * len(mesh.edges)

    edges_transferred = 0

    for u, v, data in _iter_edges_with_data(G):
        if attr_name not in data:
            continue

        value = float(data[attr_name])

        mesh_edge_indices = edge_mapping.get((str(u), str(v)), [])

        for edge_idx in mesh_edge_indices:
            if edge_idx < len(values):
                values[edge_idx] = value
                edges_transferred += 1

    attr.data.foreach_set("value", values)
    return edges_transferred


def transfer_node_attribute_to_mesh(obj, G, attr_name, mesh_attr_name=None):
    """
    Copy a NetworkX node attribute onto the Blender mesh as a ``POINT``-domain
    ``FLOAT`` attribute.

    Returns:
        Number of vertices that received a value.
    """
    mesh = obj.data
    if mesh_attr_name is None:
        mesh_attr_name = attr_name

    nodes_str = obj.get("nodes_data", "")
    if not nodes_str:
        return 0

    node_ids = nodes_str.split(",")

    if mesh_attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[mesh_attr_name])

    attr = mesh.attributes.new(name=mesh_attr_name, type='FLOAT', domain='POINT')

    values = [0.0] * len(mesh.vertices)

    nodes_transferred = 0

    for i, node_id in enumerate(node_ids):
        if i >= len(values):
            break

        node_key = int(node_id) if node_id.isdigit() else node_id

        if node_key in G.nodes and attr_name in G.nodes[node_key]:
            values[i] = float(G.nodes[node_key][attr_name])
            nodes_transferred += 1

    attr.data.foreach_set("value", values)
    return nodes_transferred


def mark_shortest_path_attributes(obj, path_nodes):
    """
    Create ``on_path`` (POINT, INT) and ``on_path_edge`` (EDGE, INT) attributes
    marking the nodes/edges that belong to the given shortest path.
    """
    mesh = obj.data
    nodes_str = obj.get("nodes_data", "")

    if not nodes_str:
        return

    node_ids = nodes_str.split(",")
    num_intersections = len(node_ids)

    node_to_vert = {node_id: i for i, node_id in enumerate(node_ids)}

    mesh_adj = defaultdict(dict)
    for edge_idx, edge in enumerate(mesh.edges):
        v0, v1 = edge.vertices
        mesh_adj[v0][v1] = edge_idx
        mesh_adj[v1][v0] = edge_idx

    path_node_strs = set(str(n) for n in path_nodes)

    path_edges = set()
    for i in range(len(path_nodes) - 1):
        u, v = str(path_nodes[i]), str(path_nodes[i + 1])
        path_edges.add((u, v))
        path_edges.add((v, u))

    # -- point attribute --
    attr_name_point = "on_path"
    if attr_name_point in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name_point])

    point_attr = mesh.attributes.new(name=attr_name_point, type='INT', domain='POINT')
    point_values = [0] * len(mesh.vertices)

    for node_str in path_node_strs:
        vert_idx = node_to_vert.get(node_str)
        if vert_idx is not None and vert_idx < len(point_values):
            point_values[vert_idx] = 1

    edge_mapping = build_edge_mapping(obj)

    path_mesh_edges = set()
    for (u, v) in path_edges:
        mesh_edge_indices = edge_mapping.get((u, v), [])
        for idx in mesh_edge_indices:
            path_mesh_edges.add(idx)

    for vert_idx in range(num_intersections, len(mesh.vertices)):
        for neighbor, edge_idx in mesh_adj[vert_idx].items():
            if edge_idx in path_mesh_edges:
                point_values[vert_idx] = 1
                break

    point_attr.data.foreach_set("value", point_values)

    # -- edge attribute --
    attr_name_edge = "on_path_edge"
    if attr_name_edge in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name_edge])

    edge_attr = mesh.attributes.new(name=attr_name_edge, type='INT', domain='EDGE')
    edge_values = [0] * len(mesh.edges)

    for edge_idx in path_mesh_edges:
        if edge_idx < len(edge_values):
            edge_values[edge_idx] = 1

    edge_attr.data.foreach_set("value", edge_values)
