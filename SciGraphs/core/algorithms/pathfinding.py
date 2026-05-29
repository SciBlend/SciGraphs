# Pathfinding algorithms for graphs

import networkx as nx
import numpy as np


def dijkstra_shortest_path(graph_data, source, target=None):
    """
    Compute shortest path using Dijkstra's algorithm.
    
    Args:
        graph_data: GraphData object
        source: Source node index
        target: Target node index (optional, if None returns paths to all nodes)
    
    Returns:
        dict with 'path', 'distance', or 'distances' and 'paths'
    """
    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges with weights if available
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0  # Default weight
            # If graph_data has edge weights, use them
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    try:
        if target is not None:
            # Single source-target path
            path = nx.shortest_path(G, source, target, weight='weight')
            distance = nx.shortest_path_length(G, source, target, weight='weight')
            return {
                'path': path,
                'distance': distance,
                'exists': True
            }
        else:
            # All paths from source
            lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
            paths = nx.single_source_dijkstra_path(G, source, weight='weight')
            return {
                'distances': lengths,
                'paths': paths
            }
    except nx.NetworkXNoPath:
        return {
            'path': None,
            'distance': float('inf'),
            'exists': False
        }


def a_star_path(graph_data, source, target, positions):
    """
    Compute shortest path using A* algorithm with euclidean distance heuristic.
    
    Args:
        graph_data: GraphData object
        source: Source node index
        target: Target node index
        positions: Node positions for heuristic (nx3 array)
    
    Returns:
        dict with 'path' and 'distance'
    """
    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    # Define heuristic function
    def heuristic(u, v):
        pos_u = positions[u]
        pos_v = positions[v]
        return np.linalg.norm(pos_u - pos_v)
    
    try:
        path = nx.astar_path(G, source, target, heuristic=heuristic, weight='weight')
        distance = nx.astar_path_length(G, source, target, heuristic=heuristic, weight='weight')
        return {
            'path': path,
            'distance': distance,
            'exists': True
        }
    except nx.NetworkXNoPath:
        return {
            'path': None,
            'distance': float('inf'),
            'exists': False
        }


def bellman_ford_path(graph_data, source):
    """
    Compute shortest paths using Bellman-Ford algorithm (handles negative weights).
    
    Args:
        graph_data: GraphData object
        source: Source node index
    
    Returns:
        dict with 'distances', 'paths', 'has_negative_cycle'
    """
    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    try:
        lengths, paths = nx.single_source_bellman_ford(G, source, weight='weight')
        return {
            'distances': lengths,
            'paths': paths,
            'has_negative_cycle': False
        }
    except nx.NetworkXUnbounded:
        return {
            'distances': {},
            'paths': {},
            'has_negative_cycle': True
        }


def all_pairs_shortest_path(graph_data):
    """
    Compute shortest paths between all pairs of nodes.
    
    Args:
        graph_data: GraphData object
    
    Returns:
        dict with 'distance_matrix' (numpy array)
    """
    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    # Compute all pairs
    n = len(graph_data.nodes)
    distance_matrix = np.full((n, n), float('inf'))
    
    for source in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
        for target, dist in lengths.items():
            distance_matrix[source, target] = dist
    
    return {
        'distance_matrix': distance_matrix
    }


def k_shortest_paths(graph_data, source, target, k=3):
    """
    Find k shortest simple paths between source and target.
    
    Args:
        graph_data: GraphData object
        source: Source node index
        target: Target node index
        k: Number of paths to find
    
    Returns:
        list of dicts with 'path' and 'length'
    """
    G = nx.DiGraph() if hasattr(graph_data, 'is_directed') and graph_data.is_directed else nx.Graph()
    
    # Add nodes
    for i, node in enumerate(graph_data.nodes):
        G.add_node(i)
    
    # Add edges
    for edge in graph_data.edges:
        src_idx = graph_data.node_to_index.get(edge[0], None)
        tgt_idx = graph_data.node_to_index.get(edge[1], None)
        if src_idx is not None and tgt_idx is not None:
            weight = 1.0
            if hasattr(graph_data, 'edge_weights') and graph_data.edge_weights:
                weight = graph_data.edge_weights.get((edge[0], edge[1]), 1.0)
            G.add_edge(src_idx, tgt_idx, weight=weight)
    
    # Handle invalid k
    if k <= 0:
        return []
    
    paths_found = []
    try:
        for path in nx.shortest_simple_paths(G, source, target, weight='weight'):
            # Calculate path length
            length = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            paths_found.append({
                'path': path,
                'length': length
            })
            if len(paths_found) >= k:
                break
    except nx.NetworkXNoPath:
        pass
    
    return paths_found

