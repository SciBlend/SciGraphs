"""Hierarchical and directed layout algorithms."""

from .common import *
from .basic import _random_layout

def _hierarchical_layout_3d(G, scale):
    """
    Hierarchical layout for tree-like structures.
    Places nodes in layers based on graph hierarchy.
    """
    import time
    start = time.time()
    print(f"Computing Hierarchical 3D layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return np.zeros((0, 3))

    # Find root nodes (nodes with no incoming edges in directed graph)
    # For undirected graphs, use node with highest degree
    if G.is_directed():
        roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    else:
        # Use node with highest degree as root
        degrees = dict(G.degree())
        if degrees:
            root = max(degrees, key=degrees.get)
            roots = [root]
        else:
            roots = [0] if num_nodes > 0 else []

    if not roots:
        roots = [0]  # Default to first node

    # BFS to assign levels
    levels = {}
    queue = [(root, 0) for root in roots]
    visited = set(roots)

    while queue:
        node, level = queue.pop(0)
        levels[node] = level

        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, level + 1))

    # Assign positions based on levels
    max_level = max(levels.values()) if levels else 0
    positions = np.zeros((num_nodes, 3))

    level_counts = {}
    level_indices = {}

    for node, level in levels.items():
        if level not in level_counts:
            level_counts[level] = 0
            level_indices[level] = 0
        level_counts[level] += 1

    for node in range(num_nodes):
        if node in levels:
            level = levels[node]

            # Y position based on level
            y = (level / max(1, max_level)) * scale * 2 - scale

            # X and Z position: arrange in circle at this level
            angle = (level_indices[level] / max(1, level_counts[level])) * 2 * np.pi
            radius = scale * 0.5

            x = radius * np.cos(angle)
            z = radius * np.sin(angle)

            positions[node] = [x, y, z]
            level_indices[level] += 1
        else:
            # Unconnected nodes: place randomly
            positions[node] = (np.random.rand(3) * 2 - 1) * scale

    print(f"  Hierarchical 3D completed in {time.time() - start:.2f}s")
    return positions

def _bipartite_layout_3d(G, scale):
    """
    Layout for bipartite graphs with two sets in parallel planes.
    Attempts to detect bipartite structure if not specified.
    """
    import time
    start = time.time()
    print(f"Computing Bipartite 3D layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return np.zeros((0, 3))

    # Try to detect bipartite sets
    if nx.is_bipartite(G):
        sets = nx.bipartite.sets(G)
        set0 = sorted(list(sets[0]))
        set1 = sorted(list(sets[1]))
    else:
        # Not bipartite, split by degree or randomly
        degrees = dict(G.degree())
        median_degree = np.median(list(degrees.values())) if degrees else 0
        set0 = [n for n in G.nodes() if degrees.get(n, 0) <= median_degree]
        set1 = [n for n in G.nodes() if n not in set0]

    positions = np.zeros((num_nodes, 3))

    # Place set0 on one plane
    for i, node in enumerate(set0):
        angle = (i / max(1, len(set0))) * 2 * np.pi
        radius = scale * 0.6
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = -scale * 0.5
        positions[node] = [x, y, z]

    # Place set1 on parallel plane
    for i, node in enumerate(set1):
        angle = (i / max(1, len(set1))) * 2 * np.pi
        radius = scale * 0.6
        x = radius * np.cos(angle)
        z = radius * np.sin(angle)
        y = scale * 0.5
        positions[node] = [x, y, z]

    print(f"  Bipartite 3D completed in {time.time() - start:.2f}s")
    return positions

def _sugiyama_layout(G, scale):
    """
    Sugiyama/Layered hierarchical layout for directed acyclic graphs (DAGs).
    Arranges nodes in horizontal layers to minimize edge crossings.
    Perfect for workflows and process diagrams.
    """
    import time
    start = time.time()
    print(f"Computing Sugiyama layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return np.zeros((0, 3))

    # Convert to directed graph if not already
    if not G.is_directed():
        G = G.to_directed()

    # Check if it's a DAG, if not try to remove cycles
    if not nx.is_directed_acyclic_graph(G):
        # Find and remove feedback arc set to make it a DAG
        try:
            feedback_edges = list(nx.edge_dfs(G, orientation='reverse'))
            G_dag = G.copy()
            edges_to_remove = []
            for edge in feedback_edges:
                if not nx.is_directed_acyclic_graph(G_dag):
                    edges_to_remove.append((edge[0], edge[1]))
                    G_dag.remove_edge(edge[0], edge[1])
            G = G_dag
        except:
            # Fallback to hierarchical layout
            return _hierarchical_layout_3d(G, scale)

    # Topological sort to get layers
    try:
        layers_dict = {}
        for node in nx.topological_sort(G):
            # Calculate layer based on longest path from sources
            predecessors = list(G.predecessors(node))
            if not predecessors:
                layers_dict[node] = 0
            else:
                layers_dict[node] = max(layers_dict.get(pred, 0) for pred in predecessors) + 1
    except:
        # Fallback: assign layers by BFS from nodes with no incoming edges
        layers_dict = {}
        sources = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not sources:
            sources = [list(G.nodes())[0]]

        queue = [(source, 0) for source in sources]
        visited = set(sources)

        while queue:
            node, layer = queue.pop(0)
            layers_dict[node] = layer

            for successor in G.successors(node):
                if successor not in visited:
                    visited.add(successor)
                    queue.append((successor, layer + 1))

    # Organize nodes by layer
    max_layer = max(layers_dict.values()) if layers_dict else 0
    layers = [[] for _ in range(max_layer + 1)]
    for node, layer in layers_dict.items():
        layers[layer].append(node)

    # Position nodes
    positions = np.zeros((num_nodes, 3))

    for layer_idx, layer_nodes in enumerate(layers):
        # Y position based on layer
        y = (layer_idx / max(1, max_layer)) * scale * 2 - scale

        # X positions: spread evenly across layer
        num_in_layer = len(layer_nodes)
        for i, node in enumerate(layer_nodes):
            x = ((i / max(1, num_in_layer - 1)) if num_in_layer > 1 else 0.5) * scale * 2 - scale
            z = 0  # Keep in 2D plane
            positions[node] = [x, y, z]

    # Handle nodes not in any layer (disconnected)
    for node in G.nodes():
        if node not in layers_dict:
            positions[node] = (np.random.rand(3) * 2 - 1) * scale * 0.5

    print(f"  Sugiyama layout completed in {time.time() - start:.2f}s")
    return positions

def _circular_hierarchy_layout(G, scale):
    """
    Circular hierarchical layout for directed graphs.
    Root nodes at center, descendants in concentric circles.
    Perfect for hierarchies and tree-like directed structures.
    """
    import time
    start = time.time()
    print(f"Computing Circular Hierarchy layout for {len(G.nodes())} nodes...")

    num_nodes = len(G.nodes())

    if num_nodes == 0:
        return np.zeros((0, 3))

    # Convert to directed if needed
    if not G.is_directed():
        G = G.to_directed()

    # Find root nodes (no incoming edges)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]

    if not roots:
        # Use nodes with highest out-degree as roots
        out_degrees = dict(G.out_degree())
        if out_degrees:
            max_out = max(out_degrees.values())
            roots = [n for n, d in out_degrees.items() if d == max_out][:3]
        else:
            roots = [0] if num_nodes > 0 else []

    # BFS to assign levels from roots
    levels = {}
    queue = [(root, 0) for root in roots]
    visited = set(roots)

    while queue:
        node, level = queue.pop(0)
        levels[node] = level

        for successor in G.successors(node):
            if successor not in visited:
                visited.add(successor)
                queue.append((successor, level + 1))

    # Assign positions
    max_level = max(levels.values()) if levels else 0
    positions = np.zeros((num_nodes, 3))

    # Group nodes by level
    level_nodes = {}
    for node, level in levels.items():
        if level not in level_nodes:
            level_nodes[level] = []
        level_nodes[level].append(node)

    # Position nodes in concentric circles
    for level in range(max_level + 1):
        nodes_at_level = level_nodes.get(level, [])

        if level == 0:
            # Center position for roots
            radius = 0
        else:
            # Increasing radius for each level
            radius = level * scale / max(2, max_level)

        num_at_level = len(nodes_at_level)
        for i, node in enumerate(nodes_at_level):
            if num_at_level == 1 and level == 0:
                # Single root at center
                x, y, z = 0, 0, 0
            else:
                angle = (i / num_at_level) * 2 * np.pi
                x = radius * np.cos(angle)
                z = radius * np.sin(angle)
                y = 0  # Keep flat

            positions[node] = [x, y, z]

    # Handle unvisited nodes (disconnected)
    for node in G.nodes():
        if node not in levels:
            # Place randomly on outer ring
            angle = np.random.rand() * 2 * np.pi
            radius = scale * 1.2
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0
            positions[node] = [x, y, z]

    print(f"  Circular Hierarchy layout completed in {time.time() - start:.2f}s")
    return positions

__all__ = [name for name in globals() if not name.startswith('__')]
